"""训练 latent 统计：唯一 origin、独立加权 oracle、落盘损坏与溯源隔离。"""
import copy
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch

from exp.gslaformer_support import (LatentStatistics, collect_latent_statistics,
                                   frozen_representation_fingerprint)
from data_provider.spatial_temporal.batching import collate_samples
from test_gslaformer_model import real_batch


class LatentStatisticsTests(unittest.TestCase):
    def test_collector_reads_only_last_history_origin_and_not_supervision(self):
        torch.manual_seed(2023)
        model, samples, config = real_batch()
        batch = collate_samples(samples, config)
        model.set_training_stage("B")
        model.train()
        schema = batch.inputs.metadata[0]["schema_fingerprint"]
        expected = model.encode_history(batch).z[:, -1].cpu().numpy()
        batch.supervision = None  # 统计不能依赖未来 teacher 或 target。
        stats = collect_latent_statistics(model, [batch, batch],
                                          schema_fingerprint=schema, data_fingerprint="dataset")
        arrays = stats.finalize()
        self.assertEqual(stats.origin_count, 2)
        self.assertEqual(arrays["assets__count"].item(), 2)
        np.testing.assert_allclose(arrays["assets__mean"], expected[:, 0].mean(axis=0), atol=1e-7)
        fingerprint = frozen_representation_fingerprint(model)
        with torch.no_grad():
            model.load_head.weight.add_(1)
        self.assertEqual(frozen_representation_fingerprint(model), fingerprint)
        bad = copy.deepcopy(batch)
        bad.inputs.metadata[0]["split"] = "valid"
        with self.assertRaisesRegex(ValueError, "train"):
            collect_latent_statistics(model, [bad], schema_fingerprint=schema, data_fingerprint="dataset")

    def collector(self):
        return LatentStatistics(["a", "b"], 2, schema_fingerprint="schema",
                                data_fingerprint="data", representation_fingerprint="frozen")

    def populate(self):
        stats = self.collector()
        zs = np.array([[[0., 2.], [3., 7.]], [[4., 6.], [3., 7.]], [[8., 10.], [3., 7.]]])
        qs = np.array([[1., 1.], [.5, 1.], [0., 0.]])
        for i, (z, q) in enumerate(zip(zs, qs)):
            stats.add("episode", i, z, q, request_fingerprint=f"v{i}", split="train")
        return stats, zs, qs

    def test_unique_origins_weighted_population_and_partial_degeneracy(self):
        stats, zs, qs = self.populate()
        stats.add("episode", 0, zs[0], qs[0], request_fingerprint="v0", split="train")
        with warnings.catch_warnings(record=True) as notices:
            warnings.simplefilter("always")
            arrays = stats.finalize()
        self.assertTrue(any("b" in str(w.message) for w in notices))
        for index, name in enumerate(("a", "b")):
            mean = np.average(zs[:, index], axis=0, weights=qs[:, index])
            variance = np.average((zs[:, index] - mean) ** 2, axis=0, weights=qs[:, index])
            np.testing.assert_allclose(arrays[f"{name}__mean"], mean)
            np.testing.assert_allclose(arrays[f"{name}__variance"], variance)
            self.assertEqual(arrays[f"{name}__count"].item(), 2)
        self.assertEqual(stats.origin_count, 3)

    def test_reject_nontrain_conflicting_duplicate_and_wrong_shape(self):
        stats, zs, qs = self.populate()
        for split in ("valid", "test", "forecast"):
            with self.assertRaisesRegex(ValueError, "train"):
                stats.add("episode", 9, zs[0], qs[0], request_fingerprint="v9", split=split)
        for changed_z, changed_q, fingerprint in ((zs[0] + 1, qs[0], "v0"),
                                                  (zs[0], qs[0] / 2, "v0"),
                                                  (zs[0], qs[0], "changed")):
            with self.assertRaisesRegex(ValueError, "重复"):
                stats.add("episode", 0, changed_z, changed_q, request_fingerprint=fingerprint, split="train")
        with self.assertRaisesRegex(ValueError, "shape"):
            stats.add("episode", 9, zs, qs, request_fingerprint="v9", split="train")

    def test_all_degenerate_and_insufficient_origins_fail(self):
        stats = self.collector()
        z = np.ones((2, 2))
        stats.add("e", 0, z, np.ones(2), request_fingerprint="r0", split="train")
        with self.assertRaisesRegex(ValueError, "两个"):
            stats.finalize()
        stats.add("e", 1, z, np.ones(2), request_fingerprint="r1", split="train")
        with self.assertRaisesRegex(ValueError, "representation_degenerate"):
            stats.finalize()

    def test_npz_roundtrip_and_context_or_bytes_tampering_rejected(self):
        stats, _, _ = self.populate()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                stats.save(root)
            context = dict(schema_fingerprint="schema", data_fingerprint="data",
                           representation_fingerprint="frozen")
            restored = LatentStatistics.load(root, **context)
            np.testing.assert_allclose(restored["a__variance"], stats.finalize()["a__variance"])
            for key in context:
                wrong = dict(context, **{key: "wrong"})
                with self.assertRaisesRegex(ValueError, "指纹"):
                    LatentStatistics.load(root, **wrong)
            with self.assertRaises(FileExistsError):
                stats.save(root)
            path = root / "latent_stats.npz"
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "校验"):
                LatentStatistics.load(root, **context)


if __name__ == "__main__":
    unittest.main()
