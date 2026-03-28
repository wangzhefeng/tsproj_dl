import importlib
import os
import unittest


class RunDlImportTestCase(unittest.TestCase):

    def test_run_dl_imports(self):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsproj_dl_matplotlib")
        module = importlib.import_module("run_dl")
        self.assertTrue(hasattr(module, "args_parse"))


if __name__ == "__main__":
    unittest.main()
