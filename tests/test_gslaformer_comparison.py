"""比较任务真实执行、朴素预测的起点可见性与 fallback 计数。"""
import copy
import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from exp.gslaformer_comparison import baseline_forecast, run_comparison
from data_provider.spatial_temporal.schema import read_json
from test_gslaformer_package import small_config, write_package


class ComparisonTests(unittest.TestCase):
    def test_naive_only_reads_observed_history_and_counts_fallback(self):
        inputs = {"target_history": np.array([[1.], [2.], [3.], [999.]]),
                  "target_history_observed": np.array([[True], [True], [True], [False]]),
                  "history_time": np.array([0, 3600, 7200, 10800]),
                  "origin": 10800, "future_time": np.array([14400, 18000])}
        persistence, count = baseline_forecast(inputs, "persistence", 2, 3600)
        np.testing.assert_array_equal(persistence[:, 0], [3., 3.])
        self.assertEqual(count, 0)
        seasonal, count = baseline_forecast(inputs, "seasonal_naive", 2, 3600)
        np.testing.assert_array_equal(seasonal[:, 0], [3., 2.])
        self.assertEqual(count, 0)
        fallback, count = baseline_forecast(inputs, "seasonal_naive", 24, 3600)
        np.testing.assert_array_equal(fallback, persistence)
        self.assertEqual(count, 2)
        inputs["target_history_observed"][:] = False
        with self.assertRaises(ValueError):
            baseline_forecast(inputs, "persistence", 24, 3600)

    def test_base_and_shared_best_a_comparisons_execute_real_training(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_config()
            config["data"].update(known_future=[], package=str(root / "data"))
            config["training"].update(stage_a_max_epochs=1, b_short_epochs=1,
                                         b_medium_epochs=1, b_full_max_epochs=1)
            write_package(root / "data", config)
            base = run_comparison(config, root / "base_comparison")
            self.assertEqual(base["status"], "passed")
            self.assertEqual(set(base["runs"]), {"persistence", "seasonal_naive", "full"})
            self.assertFalse((root / "base_comparison/persistence_test/checkpoints").exists())
            for name in ("resolved_config.json", "schema.json", "data_manifest.json", "environment.json", "status.json"):
                self.assertTrue(read_json(root / "base_comparison" / name))
            self.assertEqual(read_json(root / "base_comparison/status.json")["status"], "completed")
            self.assertTrue((root / "base_comparison/run.log").read_text())
            with (root / "base_comparison/persistence_test/predictions.csv").open() as stream:
                row = next(csv.DictReader(stream))
            for name in ("new_site", "new_entity_count", "no_history_entity_count", "unknown_category_count"):
                self.assertIn(name, row)
            mechanism = copy.deepcopy(config)
            mechanism["experiment"]["profile"] = "M3"
            result = run_comparison(mechanism, root / "m3_dispatch")
            self.assertEqual(result["status"], "passed")
            self.assertEqual(set(result["runs"]), {"full", "direct", "no_latent_loss", "total_only"})
            self.assertTrue((root / "m3_dispatch/no_latent_loss_train/initialization.json").exists())
            self.assertTrue((root / "m3_dispatch/comparison.csv").exists())


if __name__ == '__main__':
    unittest.main()
