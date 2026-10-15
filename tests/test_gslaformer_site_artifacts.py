"""小型 site_time 集合的训练、独立推理及分站场景审计产物。"""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from data_provider.spatial_temporal.config import schema_from_config
from exp.exp_gslaformer_forecasting import Exp_GSLAformer_Forecast
from exp.gslaformer_inference import evaluate_checkpoint
from exp.gslaformer_comparison import run_baseline
from test_gslaformer_package import small_config, write_package, rehash


class SiteArtifactsTests(unittest.TestCase):
    def test_site_time_outputs_and_standalone_unseen_site_forecast(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = small_config()
            config["data"].update(package=str(root / "collection"), known_future=[], split_mode="site_time",
                                 site_split={"train": ["s0"], "heldout_valid": ["s1"], "heldout_test": ["s2"]},
                                 sampler="uniform_site_then_origin")
            config["evaluation"]["selection_metric"] = "valid_site_macro_mse"
            config["training"].update(stage_a_max_epochs=1, b_short_epochs=1, b_medium_epochs=1, b_full_max_epochs=1)
            collection = root / "collection"
            collection.mkdir()
            entries = []
            for i in range(3):
                path = collection / f"e{i}"
                write_package(path, config)
                site = json.loads((path / "site.json").read_text())
                site["site_id"] = f"s{i}"
                (path / "site.json").write_text(json.dumps(site))
                rehash(path)
                entries.append({"id": f"e{i}", "family_id": f"f{i}", "site_id": f"s{i}", "split": None,
                                "path": f"e{i}", "topology_family_id": f"topo{i}"})
            manifest = {"schema_version": 3, "data_kind": "synthetic", "layout": "episode_collection",
                        "schema": schema_from_config(config), "episodes": entries, "split_mode": "site_time",
                        "split_index": config["data"]["split_index"], "site_split": config["data"]["site_split"]}
            (collection / "manifest.json").write_text(json.dumps(manifest))
            audit = {"kind": "test_fixture", "sites": ["s0", "s1", "s2"]}
            (collection / "generation_audit.json").write_text(json.dumps(audit))
            experiment = Exp_GSLAformer_Forecast(config, root / "train")
            experiment.train()
            checkpoint = root / "train/checkpoints/best.pt"
            evaluate_checkpoint(config, checkpoint, root / "test", mode="test")
            entries = json.loads((root / "test/scenario_metrics.json").read_text())["entries"]
            self.assertEqual({row["site_id"] for row in entries}, {"s0", "s2"})
            self.assertEqual({row["site_role"] for row in entries}, {"train", "heldout_test"})
            baseline = deepcopy(config)
            baseline["experiment"]["variant"] = "persistence"
            run_baseline(baseline, root / "baseline")
            report = json.loads((root / "baseline/scenario_metrics.json").read_text())
            # 固定夹具 target 每步 +10，H=2 persistence 的误差固定 10/20。
            self.assertEqual(report["site_macro_mae"], 15.0)
            self.assertEqual(set(report["per_site"]), {"s0", "s2"})
            request = root / "request"
            write_package(request, config, forecast=True)
            site = json.loads((request / "site.json").read_text())
            site["site_id"] = "s2"
            (request / "site.json").write_text(json.dumps(site))
            relations = json.loads((request / "relations.json").read_text())
            relations["edges"] = relations["edges"][:1]  # a1/z1 成为合法孤立节点，仍保留一条有效边。
            (request / "relations.json").write_text(json.dumps(relations))
            rehash(request)
            independent = deepcopy(config)
            independent["data"]["package"] = str(root / "unavailable_training_source")
            evaluate_checkpoint(independent, checkpoint, root / "forecast", mode="forecast", request=request)
            self.assertEqual(json.loads((root / "forecast/generation_audit.json").read_text()), audit)
            self.assertIsNone(json.loads((root / "forecast/scenario_metrics.json").read_text())["metrics"])
            metadata = json.loads((root / "forecast/metadata.json").read_text())["requests"][0]
            self.assertEqual(metadata["orphan_count"], 2)
            self.assertTrue(metadata["degraded_input"])
            for name in ("site_split.json", "generation_audit.json", "metadata.json", "scenario_metrics.json"):
                self.assertTrue((root / "train" / name).exists())
            with self.assertRaises(FileNotFoundError):
                evaluate_checkpoint(config, root / "absent.pt", root / "failed_load", mode="test")
            self.assertEqual(json.loads((root / "failed_load/status.json").read_text())["status"], "failed")


if __name__ == '__main__':
    unittest.main()
