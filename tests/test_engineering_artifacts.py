import unittest
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

    def test_results_directory_is_ignored(self):
        gitignore = (self.repo_root / ".gitignore").read_text(encoding="utf-8")
        self.assertIn("results/", gitignore)
        self.assertNotIn("utils/", gitignore)

    def test_utils_is_no_longer_nested_git_repo(self):
        nested_git = self.repo_root / "utils" / ".git"
        self.assertFalse(nested_git.exists())


if __name__ == "__main__":
    unittest.main()
