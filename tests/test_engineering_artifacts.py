import unittest
import importlib.util
import tempfile
from pathlib import Path


class EngineeringArtifactsTestCase(unittest.TestCase):

    def setUp(self):
        self.repo_root = Path(__file__).resolve().parents[1]

    def test_smoke_script_exists(self):
        smoke_script = self.repo_root / "scripts" / "smoke" / "smoke_engineering_checks.sh"
        self.assertTrue(smoke_script.exists())

    def test_utils_audit_tool_exists(self):
        audit_tool = self.repo_root / "scripts" / "dev" / "audit_runtime_utils_imports.py"
        self.assertTrue(audit_tool.exists())

    def test_utils_audit_scans_current_entry_without_executing_modules(self):
        spec = importlib.util.spec_from_file_location("runtime_audit", self.repo_root / "scripts/dev/audit_runtime_utils_imports.py")
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("exp", "data_provider", "models", "layers"):
                (root / name).mkdir()
            (root / "run.py").write_text("from utils.log_util import logger\nraise RuntimeError('must not execute')\n")
            (root / "models/sample.py").write_text("import utils.metrics_dl as metrics\nfrom .utils import local\n")
            records = audit.scan_utils_imports(root)
            self.assertEqual([(row["file"], row["module"]) for row in records],
                             [("run.py", "utils.log_util"), ("models/sample.py", "utils.metrics_dl")])

    def test_results_directory_is_ignored(self):
        gitignore = (self.repo_root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("results/", gitignore)
        self.assertNotIn("utils/", gitignore)

    def test_utils_is_no_longer_nested_git_repo(self):
        nested_git = self.repo_root / "utils" / ".git"
        self.assertFalse(nested_git.exists())


if __name__ == "__main__":
    unittest.main()
