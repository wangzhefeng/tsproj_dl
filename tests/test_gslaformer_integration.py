"""CLI 隔离路由、真实 train/test/forecast 和单文件 checkpoint 推理。"""
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import torch

from test_gslaformer_package import small_config, write_package

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        config = small_config()
        config["data"].update(package=str(self.root / "data"), known_future=[])
        config["experiment"]["results_root"] = str(self.root / "runs")
        config["training"].update(stage_a_max_epochs=1, b_short_epochs=1, b_medium_epochs=1, b_full_max_epochs=1)
        write_package(self.root / "data", config)
        write_package(self.root / "request", config, forecast=True)
        self.config = config
        self.path = self.root / "config.json"
        self.path.write_text(json.dumps(config))

    def command(self, mode, run_id, *extra, success=True, config_path=None):
        argv = [sys.executable, str(ROOT / "run.py"), "--model", "GSLAformer", "--gsla_config",
                str(config_path or self.path), "--gsla_mode", mode, "--gsla_run_id", run_id, *map(str, extra)]
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        result = subprocess.run(argv, cwd=ROOT, env=environment, text=True, capture_output=True, timeout=120)
        self.assertEqual(result.returncode == 0, success, result.stdout + result.stderr)
        return self.root / "runs" / run_id

    def test_cli_train_test_and_standalone_checkpoint_forecast(self):
        trained = self.command("train", "training")
        best = trained / "checkpoints/best.pt"
        tested = self.command("test", "testing", "--gsla_checkpoint", best)
        metrics = json.loads((tested / "metrics.json").read_text())
        metadata = json.loads((tested / "metadata.json").read_text())
        self.assertGreater(metadata["parameter_count"], 0)
        self.assertGreater(metadata["prediction_seconds"], 0)
        for row in metadata["requests"]:
            self.assertEqual(len(row["relation_fingerprint"]), 64)
            self.assertIn("site_role", row)
        self.assertTrue((tested / "latent_metrics.json").exists())
        diagnostics = json.loads((tested / "diagnostics.json").read_text())
        self.assertFalse(diagnostics["ranking_eligible"])
        self.assertTrue(diagnostics["diagnostics"]["teacher_z"]["uses_future_observations"])
        self.assertEqual(len(metrics["per_horizon"]), 2)
        self.assertGreater(metrics["per_horizon"][0]["count"], 0)
        standalone = self.root / "standalone.pt"
        shutil.copyfile(best, standalone)
        # 配置故意指向不存在的训练包：forecast 只能读独立 request 与自包含 checkpoint。
        self.config["data"]["package"] = str(self.root / "nonexistent_training_package")
        independent = self.root / "independent.json"
        independent.write_text(json.dumps(self.config))
        rows = []
        for run in ("forecast1", "forecast2"):
            predicted = self.command("forecast", run, "--gsla_checkpoint", standalone,
                                     "--gsla_request", self.root / "request", config_path=independent)
            self.assertFalse((predicted / "metrics.json").exists())
            self.assertTrue((predicted / "latents.npz").exists())
            with (predicted / "predictions.csv").open() as stream:
                current = list(csv.DictReader(stream))
            self.assertEqual(len(current), 2)
            self.assertTrue(all(row["y_true"] == "" for row in current))
            self.assertEqual(int(current[0]["target_time"]) - int(current[0]["origin"]), 3600)
            rows.append([row["y_pred"] for row in current])
        self.assertEqual(rows[0], rows[1])
        # 本测试刚生成的可信 checkpoint；溯源缺失不能被自包含推理静默接受。
        damaged = torch.load(standalone, weights_only=False)
        damaged["latent_stats_manifest"] = None
        torch.save(damaged, self.root / "damaged.pt")
        self.command("forecast", "missing_provenance", "--gsla_checkpoint", self.root / "damaged.pt",
                     "--gsla_request", self.root / "request", config_path=independent, success=False)
        for name, edit, reason in (
            ("old_version", lambda x: x.update(version=1), "计算版本不兼容"),
            ("bad_freeze", lambda x: x.update(frozen_parameters=[]), "冻结参数清单不一致"),
            ("bad_provenance", lambda x: x["latent_stats_manifest"].update(origins=[]), "溯源指纹错误"),
            ("missing_code", lambda x: x.pop("training_source"), "源码溯源缺失"),
        ):
            with self.subTest(invalid_checkpoint=name):
                damaged = torch.load(standalone, weights_only=False)  # 本测试生成的可信本地 artifact。
                edit(damaged)
                torch.save(damaged, self.root / "damaged.pt")
                failed = self.command("forecast", name, "--gsla_checkpoint", self.root / "damaged.pt",
                                      "--gsla_request", self.root / "request", config_path=independent, success=False)
                self.assertIn(reason, json.loads((failed / "status.json").read_text())["error"])
        # 旧计算版本仍可用，但不能虚构历史源码；负预测必须完整保留。
        legacy = torch.load(standalone, weights_only=False)
        for key in ("provenance_version", "training_source", "execution_history"):
            legacy.pop(key)
        legacy["model"]["load_head.weight"].zero_()
        legacy["model"]["load_head.bias"].fill_(-1000)
        legacy_path = self.root / "legacy.pt"
        torch.save(legacy, legacy_path)
        output = self.command("forecast", "legacy_negative", "--gsla_checkpoint", legacy_path,
                              "--gsla_request", self.root / "request", config_path=independent)
        env = json.loads((output / "environment.json").read_text())
        self.assertEqual(env["training_source"], {"status": "legacy_missing"})
        with (output / "predictions.csv").open() as stream:
            negative = list(csv.DictReader(stream))
        expected = -1000 * legacy["transform"]["target"]["std"][0] + legacy["transform"]["target"]["mean"][0]
        for row in negative:
            self.assertAlmostEqual(float(row["y_pred"]), expected, delta=.01)
            self.assertEqual(row["negative_prediction"], "True")
        scalers = json.loads((trained / "scalers.json").read_text())
        scalers["target"]["mean"] += 1
        (trained / "scalers.json").write_text(json.dumps(scalers))
        rejected = self.command("test", "tampered_sidecar", "--gsla_checkpoint", best, success=False)
        self.assertEqual(json.loads((rejected / "status.json").read_text())["status"], "failed")
        self.command("train", "training", success=False)

    def test_cli_rejects_legacy_flags_and_unsafe_ids(self):
        self.command("train", "legacy", "--is_training", "0", success=False)
        self.command("train", "../escape", success=False)
        self.command("forecast", "missing", success=False)


if __name__ == "__main__":
    unittest.main()
