import unittest
from types import SimpleNamespace
from unittest import mock

import torch.nn as nn

from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast


class DummyModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args


def build_args(**overrides):
    args = dict(
        data="m4",
        seasonal_patterns="Monthly",
        model="Dummy",
        use_multi_gpu=0,
        use_gpu=0,
        devices="0",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class ShortTermForecastTestCase(unittest.TestCase):

    def test_m4_build_model_overrides_window_lengths_from_seasonal_pattern(self):
        exp = object.__new__(Exp_Short_Term_Forecast)
        exp.args = build_args()

        with mock.patch.object(exp, "get_model_class", return_value=DummyModel):
            model = exp._build_model()

        self.assertIsInstance(model, DummyModel)
        self.assertEqual(exp.args.pred_len, 18)
        self.assertEqual(exp.args.seq_len, 36)
        self.assertEqual(exp.args.label_len, 18)
        self.assertEqual(exp.args.frequency_map, 12)


if __name__ == "__main__":
    unittest.main()
