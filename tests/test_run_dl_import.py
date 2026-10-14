import importlib
import os
import unittest
from types import SimpleNamespace
from unittest import mock


class RunImportTestCase(unittest.TestCase):

    def test_run_imports(self):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsproj_dl_matplotlib")
        module = importlib.import_module("run")
        self.assertTrue(hasattr(module, "args_parse"))

    def test_short_term_forecast_dispatches_to_short_term_experiment(self):
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/tsproj_dl_matplotlib")
        module = importlib.import_module("run")
        args = SimpleNamespace(
            task_name="short_term_forecast",
            model="iTransformer",
            data="m4",
            seasonal_patterns="Monthly",
            freq="m",
            features="M",
            target="OT",
            seq_len=1,
            label_len=1,
            pred_len=1,
            d_model=8,
            n_heads=1,
            e_layers=1,
            d_layers=1,
            d_ff=16,
            train_epochs=1,
            is_training=0,
            is_testing=0,
            is_forecasting=0,
            itr=1,
            train_ratio=0.7,
            test_ratio=0.2,
        )

        with mock.patch.object(module, "Exp_Short_Term_Forecast") as short_exp, \
             mock.patch.object(module, "Exp_Long_Term_Forecast") as long_exp, \
             mock.patch.object(module, "torch_gc"):
            module.run(args)

        short_exp.assert_called_once_with(args)
        long_exp.assert_not_called()


if __name__ == "__main__":
    unittest.main()
