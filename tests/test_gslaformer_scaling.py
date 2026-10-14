"""训练统计与 batch 的公开行为：独立统计 oracle、留出隔离和变长实体轴。"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from data_provider.spatial_temporal.dataset import PackageDataset, EpisodePackage
from data_provider.spatial_temporal.scaling import TrainingTransform
from data_provider.spatial_temporal.batching import collate_samples
from test_gslaformer_package import small_config, write_package, rehash, edit_arrays


class ScalingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "episode"
        self.config = small_config()
        self.config["data"]["known_future"] = []
        write_package(self.path, self.config)
        self.config["data"]["package"] = str(self.path)

    def test_population_statistics_use_each_training_cell_once(self):
        train = PackageDataset(self.config, "train")
        transform = TrainingTransform.fit(train)
        state = transform.to_dict()
        self.assertEqual(state["target"]["count"], [6])
        self.assertEqual(state["target"]["mean"], [25])
        np.testing.assert_allclose(state["target"]["std"], [np.sqrt(875 / 3)])
        self.assertEqual(state["groups"]["assets"]["dynamic"]["count"], [12])
        self.assertEqual(state["groups"]["assets"]["dynamic"]["mean"], [2.5])
        self.assertEqual(state["groups"]["assets"]["static"]["count"], [2])
        self.assertEqual(state["groups"]["assets"]["static"]["mean"], [2.5])
        sample = transform.apply(train[0])
        restored = transform.inverse_target(sample["supervision"]["target"])
        np.testing.assert_allclose(restored.ravel(), [30, 40], atol=1e-5)
        rebuilt = TrainingTransform.from_dict(state, train.episodes[0].schema)
        np.testing.assert_array_equal(rebuilt.apply(train[0])["inputs"]["target_history"], sample["inputs"]["target_history"])
        # 训练单元唯一采样，而不是重叠窗口频数加权。
        self.assertEqual(state, transform.to_dict())
        with self.assertRaisesRegex(ValueError, "train"):
            TrainingTransform.fit(PackageDataset(self.config, "valid"))

    def test_validation_extremes_late_training_data_and_new_categories(self):
        def modify(arrays):
            arrays["target"][6:] = 1000000
            arrays["assets__values"][6:] = 1000000
            arrays["assets__available_at"][5] = 1000000
        edit_arrays(self.path, "observations.npz", modify)
        transform = TrainingTransform.fit(PackageDataset(self.config, "train"))
        self.assertEqual(transform.to_dict()["target"]["mean"], [25])
        self.assertEqual(transform.to_dict()["groups"]["assets"]["dynamic"]["mean"], [2])
        episode = EpisodePackage(self.path, self.config)
        sample = episode.sample(8)
        sample["inputs"]["groups"]["assets"]["static_category_history"][:, 0, 0] = "new_kind"
        sample["inputs"]["groups"]["assets"]["observed"][:, 0] = False
        sample["inputs"]["groups"]["assets"]["values"][:, 0] = np.nan
        sample["inputs"]["groups"]["assets"]["entity_ids"][0] = "brand_new"
        before = transform.fingerprint
        transformed = transform.apply(sample)
        group = transformed["inputs"]["groups"]["assets"]
        self.assertTrue((group["static_category_history"][:, 0, 0] == 1).all())
        self.assertTrue((group["values"][:, 0] == 0).all())
        self.assertEqual(transformed["metadata"]["unknown_category_count"], 1)
        self.assertEqual(transformed["metadata"]["no_history_entity_count"], 1)
        self.assertEqual(transformed["metadata"]["new_entity_count"], 1)
        self.assertEqual(before, transform.fingerprint)

    def test_empty_feature_and_damaged_artifact_fail_without_refitting(self):
        train = PackageDataset(self.config, "train")
        transform = TrainingTransform.fit(train)
        damaged = transform.to_dict()
        damaged["target"]["std"] = [0]
        with self.assertRaises(ValueError):
            TrainingTransform.from_dict(damaged, train.episodes[0].schema)
        edit_arrays(self.path, "observations.npz", lambda a: a["assets__observed"].__setitem__(slice(0, 6), False))
        with self.assertRaisesRegex(ValueError, "训练观测"):
            TrainingTransform.fit(PackageDataset(self.config, "train"))

    def test_variable_batch_padding_relations_and_supervision_are_separate(self):
        train = PackageDataset(self.config, "train")
        transform = TrainingTransform.fit(train)
        first, second = transform.apply(train[0]), transform.apply(train[1])
        second = copy.deepcopy(second)
        group = second["inputs"]["groups"]["assets"]
        for key in ("values", "observed", "present", "static_numeric_history", "static_numeric_mask", "static_category_history"):
            group[key] = group[key][:, :1]
        for key in ("padding", "applicable"):
            group[key] = group[key][:1]
        group["entity_ids"] = group["entity_ids"][:1]
        second["inputs"]["relations_history"] = [[edge for edge in edges if edge[1] == 0] for edges in second["inputs"]["relations_history"]]
        second["inputs"]["relations_origin"] = second["inputs"]["relations_history"][-1]
        for key in ("values", "observed"):
            second["supervision"]["future_groups"]["assets"][key] = second["supervision"]["future_groups"]["assets"][key][:, :1]
        batch = collate_samples([first, second], self.config)
        self.assertEqual(tuple(batch.inputs.groups["assets"]["values"].shape), (2, 3, 2, 1))
        self.assertTrue(batch.inputs.groups["assets"]["padding"][1, 1].item())
        self.assertFalse(batch.inputs.groups["assets"]["present"][1, :, 1].any().item())
        self.assertEqual(batch.inputs.groups["assets"]["static_category_history"][1, :, 1].sum().item(), 0)
        self.assertFalse(hasattr(batch.inputs, "supervision"))
        self.assertEqual(tuple(batch.supervision.target.shape), (2, 2, 1))
        self.assertEqual(batch.inputs.relations_history[1][-1], [(0, 2, 0)])
        # 修改独立标签不会写进输入视图。
        original = batch.inputs.target_history.clone()
        batch.supervision.target.fill_(999)
        self.assertTrue((batch.inputs.target_history == original).all().item())
        moved = batch.to("cpu")
        self.assertEqual(moved.inputs.site_ids, ["synthetic_site", "synthetic_site"])
        self.assertTrue((moved.inputs.target_history == original).all().item())


if __name__ == "__main__":
    unittest.main()
