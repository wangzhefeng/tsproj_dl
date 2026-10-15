"""A/B 状态机与 CPU epoch-boundary exact resume 的真实训练验收。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

import torch

from data_provider.spatial_temporal.schema import read_json
from exp.exp_gslaformer_forecasting import Exp_GSLAformer_Forecast, SiteOriginSampler
from test_gslaformer_package import small_config, write_package, edit_arrays


class TrainingStateMachineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = small_config()
        self.config["data"]["known_future"] = []
        self.config["data"]["package"] = str(self.root / "data")
        self.config["training"].update(stage_a_max_epochs=1, b_short_epochs=1,
                                         b_medium_epochs=1, b_full_max_epochs=2)
        write_package(self.root / "data", self.config)

    def test_full_training_and_epoch_boundary_resume_are_identical(self):
        uninterrupted = Exp_GSLAformer_Forecast(self.config, self.root / "continuous")
        rows = uninterrupted.train()
        self.assertEqual([r["phase"] for r in rows], ["A", "B-short", "B-medium", "B-full", "B-full"])
        self.assertEqual([r["curriculum_steps"] for r in rows], [2, 1, 2, 2, 2])
        interrupted = Exp_GSLAformer_Forecast(self.config, self.root / "resumed")
        interrupted.train(max_epochs_this_call=2)
        self.assertFalse((self.root / "resumed/checkpoints/best.pt").exists())
        resumed = Exp_GSLAformer_Forecast(self.config, self.root / "resumed",
                                         resume=self.root / "resumed/checkpoints/last.pt")
        resumed_rows = resumed.train()
        self.assertEqual(rows, resumed_rows)
        for key, value in uninterrupted.model.state_dict().items():
            torch.testing.assert_close(value, resumed.model.state_dict()[key], atol=0, rtol=0)
        best = torch.load(self.root / "resumed/checkpoints/best.pt", weights_only=False)
        self.assertEqual(best["phase"], "B-full")
        self.assertIsNotNone(best["latent_stats"])
        manifest = read_json(self.root / "resumed/latent_stats_manifest.json")
        self.assertEqual(best["latent_stats_manifest"], manifest)
        self.assertTrue(best["frozen_parameters"])
        resources = read_json(self.root / "resumed/resource_metrics.json")
        self.assertEqual(len(resources["epochs"]), len(rows))
        self.assertGreater(resources["parameter_count"], 0)
        self.assertTrue(all(row["train_seconds"] > 0 for row in resources["epochs"]))
        self.assertEqual(read_json(self.root / "resumed/status.json")["status"], "completed")
        with self.assertRaises(ValueError):
            Exp_GSLAformer_Forecast(self.config, self.root / "resumed", resume=self.root / "resumed/checkpoints/last.pt")

    def test_resume_at_a_freeze_boundary_and_reject_changed_config(self):
        experiment = Exp_GSLAformer_Forecast(self.config, self.root / "boundary")
        experiment.train(max_epochs_this_call=1)
        altered = deepcopy(self.config)
        altered["training"]["learning_rate"] *= 2
        with self.assertRaisesRegex(ValueError, "不一致"):
            Exp_GSLAformer_Forecast(altered, self.root / "boundary", resume=self.root / "boundary/checkpoints/last.pt")
        resumed = Exp_GSLAformer_Forecast(self.config, self.root / "boundary", resume=self.root / "boundary/checkpoints/last.pt")
        resumed.train()
        self.assertTrue((self.root / "boundary/latent_stats.npz").exists())
        with self.assertRaises(FileExistsError):
            Exp_GSLAformer_Forecast(self.config, self.root / "boundary")

    def test_training_exports_loss_gradient_and_representation_diagnostics(self):
        experiment = Exp_GSLAformer_Forecast(self.config, self.root / "diagnostics")
        rows = experiment.train()
        for row in rows:
            self.assertAlmostEqual(row["mean_batch_loss"], row["mean_load_loss"]
                                   + 0.1 * row["mean_latent_loss"], places=5)
            self.assertGreater(row["mean_gradient_norm"], 0)
            self.assertGreaterEqual(row["history_target_sensitivity"], 0)
        self.assertEqual(rows[0]["latent_groups"], {})
        for row in rows[1:]:
            group = row["latent_groups"]["assets"]
            self.assertGreater(group["valid_count"], 0)
            self.assertGreater(group["prediction_std"], 0)
            self.assertGreaterEqual(group["mean_batch_loss"], 0)
        self.assertEqual(rows, read_json(self.root / "diagnostics/training_metrics.json"))

    def test_initialization_failure_is_recorded_and_corrupt_artifact_blocks_completion(self):
        broken = deepcopy(self.config)
        broken["data"]["package"] = str(self.root / "missing_package")
        with self.assertRaises((ValueError, FileNotFoundError)):
            Exp_GSLAformer_Forecast(broken, self.root / "bad_init")
        self.assertEqual(read_json(self.root / "bad_init/status.json")["status"], "failed")
        experiment = Exp_GSLAformer_Forecast(self.config, self.root / "damaged")
        experiment.train(max_epochs_this_call=5)
        (self.root / "damaged/latent_stats.npz").write_bytes(b"invalid npz")
        with self.assertRaises((ValueError, OSError)):
            experiment.train()
        self.assertEqual(read_json(self.root / "damaged/status.json")["status"], "failed")

    def test_total_only_h1_and_direct_have_correct_stage_artifacts(self):
        for variant in ("total_only", "direct"):
            config = deepcopy(self.config)
            config["experiment"]["variant"] = variant
            config["data"]["pred_len"] = 1
            experiment = Exp_GSLAformer_Forecast(config, self.root / variant)
            rows = experiment.train()
            if variant == "total_only":
                self.assertEqual([r["phase"] for r in rows[:4]], ["A", "B-short", "B-medium", "B-full"])
            else:
                self.assertEqual({r["phase"] for r in rows}, {"direct"})
                self.assertFalse((self.root / variant / "checkpoints/best_A.pt").exists())
            self.assertFalse((self.root / variant / "latent_stats.npz").exists())
            self.assertTrue((self.root / variant / "checkpoints/best.pt").exists())

    def test_site_sampling_balances_sites_not_window_count(self):
        class Dataset:
            entries = [{"site_id": "a"}, {"site_id": "b"}]
            windows = [(0, i) for i in range(10)] + [(1, i) for i in range(90)]
            def __len__(self):
                return len(self.windows)
        sampler = SiteOriginSampler(Dataset())
        generator = torch.Generator().manual_seed(2023)
        state = generator.get_state()
        first = sampler.draw(generator)
        generator.set_state(state)
        self.assertEqual(first, sampler.draw(generator))
        draws = [index for _ in range(100) for index in sampler.draw(generator)]
        share_a = sum(index < 10 for index in draws) / len(draws)
        self.assertGreater(share_a, 0.47)
        self.assertLess(share_a, 0.53)

    def test_unlabeled_batch_skips_update_and_unlabeled_epoch_fails(self):
        self.config["data"]["batch_size"] = 1
        edit_arrays(self.root / "data", "observations.npz",
                    lambda a: a["target_observed"].__setitem__(slice(3, 5), False))
        experiment = Exp_GSLAformer_Forecast(self.config, self.root / "skip")
        rows = experiment.train(max_epochs_this_call=1)
        self.assertEqual(rows[0]["skipped_batches"], 1)
        self.assertEqual(rows[0]["updates"], 1)
        edit_arrays(self.root / "data", "observations.npz",
                    lambda a: a["target_observed"].__setitem__(slice(3, 6), False))
        empty = Exp_GSLAformer_Forecast(self.config, self.root / "empty")
        with self.assertRaisesRegex(ValueError, "整 epoch"):
            empty.train()
        self.assertEqual(read_json(self.root / "empty/status.json")["status"], "failed")

    def test_short_courses_ignore_early_stopping(self):
        self.config["training"].update(stage_a_max_epochs=3, stage_a_patience=1,
                                         b_short_epochs=2, b_medium_epochs=2,
                                         b_full_max_epochs=3, b_full_patience=1, min_delta=1e6)
        experiment = Exp_GSLAformer_Forecast(self.config, self.root / "patience")
        rows = experiment.train()
        for phase in ("A", "B-short", "B-medium", "B-full"):
            self.assertEqual(sum(row["phase"] == phase for row in rows), 2)

    def test_no_latent_loss_inherits_exact_full_best_a_and_new_optimizer(self):
        full = Exp_GSLAformer_Forecast(self.config, self.root / "full_source")
        full.train(max_epochs_this_call=1)
        source = self.root / "full_source/checkpoints/best_A.pt"
        saved = torch.load(source, weights_only=False)  # 此测试刚创建的可信产物。
        config = deepcopy(self.config)
        config["experiment"]["variant"] = "no_latent_loss"
        config["training"]["latent_weight"] = 0
        ablated = Exp_GSLAformer_Forecast(config, self.root / "no_aux", stage_a_checkpoint=source)
        for key, tensor in ablated.model.state_dict().items():
            torch.testing.assert_close(tensor.cpu(), saved["model"][key], atol=0, rtol=0)
        self.assertEqual(len(ablated.optimizer.state), 0)
        before = {k: v.clone() for k, v in ablated.model.latent_heads.state_dict().items()}
        rows = ablated.train(max_epochs_this_call=1)
        self.assertEqual(rows[0]["phase"], "B-short")
        for key, tensor in ablated.model.latent_heads.state_dict().items():
            torch.testing.assert_close(tensor, before[key], atol=0, rtol=0)
        self.assertTrue(all(p.grad is None for p in ablated.model.latent_heads.parameters()))
        ablated.train()
        self.assertEqual(ablated.phase, "B-full")
        inherited = torch.load(self.root / "no_aux/checkpoints/best_A.pt", weights_only=False)
        self.assertEqual(inherited["phase"], "A")
        for key, value in saved["model"].items():
            torch.testing.assert_close(inherited["model"][key], value, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
