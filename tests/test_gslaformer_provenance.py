"""源码身份与配置合同：防止产物不可追溯、隐式种子漂移。"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import json

from data_provider.spatial_temporal.config import validate_config
from exp.gslaformer_provenance import capture_source, validate_source, execution_environment
from test_gslaformer_package import small_config


class ProvenanceTests(unittest.TestCase):
    def test_source_identity_excludes_results_and_detects_code_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.py").write_text("# isolated source\n")
            original = capture_source(root)
            self.assertIsNone(original["git_sha"])
            self.assertEqual(original["git_status"], "unavailable")
            validate_source(original)
            (root / "results").mkdir()
            (root / "results/result.json").write_text("{}")
            self.assertEqual(original, capture_source(root))
            (root / "run.py").write_text("# changed source\n")
            changed = capture_source(root)
            self.assertNotEqual(original["source_fingerprint"], changed["source_fingerprint"])
            changed["files"]["run.py"] = "0" * 64
            with self.assertRaisesRegex(ValueError, "源码.*指纹"):
                validate_source(changed)

    def test_live_git_and_missing_provenance(self):
        source = capture_source()
        self.assertEqual(len(source["git_sha"]), 40)
        self.assertIsInstance(source["git_dirty"], bool)
        self.assertIn("uv.lock", source["files"])
        self.assertIn("exp/exp_basic.py", source["files"])
        validate_source(source)
        with self.assertRaisesRegex(ValueError, "源码"):
            validate_source(None)
        env = execution_environment("cpu")
        self.assertEqual(env["device"], "cpu")
        self.assertEqual(env["execution_source"], source)

    def test_conflicting_seed_rejected(self):
        config = small_config()
        config["evaluation"]["compare_seed"] += 1
        with self.assertRaisesRegex(ValueError, "compare_seed.*seed"):
            validate_config(config)

    def test_unavailable_mps_is_explicit_failure(self):
        from exp.exp_gslaformer_forecasting import Exp_GSLAformer_Forecast
        config = small_config()
        config["experiment"]["device"] = "mps"
        with tempfile.TemporaryDirectory() as directory, patch("torch.backends.mps.is_available", return_value=False):
            output = Path(directory) / "run"
            with self.assertRaisesRegex(ValueError, "MPS.*不可用"):
                Exp_GSLAformer_Forecast(config, output)
            status = json.loads((output / "status.json").read_text())
            self.assertEqual(status["status"], "failed")
            self.assertIn("MPS", status["error"])

    def test_legacy_profile_is_resolved_only_for_standard_shapes(self):
        from data_provider.spatial_temporal.config import profile_name
        config = small_config()
        config["experiment"].pop("profile")
        with self.assertRaisesRegex(ValueError, "显式"):
            profile_name(config)
        config["data"].update(seq_len=24, pred_len=6)
        self.assertEqual(profile_name(config), "base")
