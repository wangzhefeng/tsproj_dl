import unittest
from types import SimpleNamespace

import torch

from models.mlp.N_BEATS import Model as NBeatsModel
from models.mlp.N_HiTs import Model as NHiTsModel
from models.mlp.TSMixer import Model as TSMixerModel


def build_args():
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=24,
        label_len=12,
        pred_len=24,
        enc_in=7,
        dec_in=7,
        c_out=1,
        d_model=64,
        d_ff=128,
        e_layers=2,
        d_layers=2,
        dropout=0.05,
        down_sampling_window=2,
    )


class ModelForwardSmokeTestCase(unittest.TestCase):

    def setUp(self):
        self.x = torch.randn(2, 24, 7)

    def test_tsmixer_forward_shape(self):
        model = TSMixerModel(build_args())
        output = model(self.x)
        self.assertEqual(output.shape, (2, 24, 1))

    def test_nhits_forward_shape(self):
        model = NHiTsModel(build_args())
        output = model(self.x)
        self.assertEqual(output.shape, (2, 24, 1))

    def test_nbeats_forward_shape(self):
        model = NBeatsModel(build_args())
        output = model(self.x)
        self.assertEqual(output.shape, (2, 24, 1))


if __name__ == "__main__":
    unittest.main()
