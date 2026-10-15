import unittest
from types import SimpleNamespace

import torch

from models.mlp.N_BEATS_todo import Model as NBeatsModel
from models.mlp.N_HiTs_todo import Model as NHiTsModel
from models.mlp.TSMixer import Model as TSMixerModel
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


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
        # 当前模型按主线四参数接口输出全通道，MS 的目标选择属于实验层。
        args = build_args()
        args.model, args.features = "TSMixer", "MS"
        args.use_gpu, args.gpu_type, args.devices, args.use_multi_gpu = False, "cpu", "0", False
        args.use_amp, args.output_attention, args.down_sampling_layers = False, False, 0
        experiment = Exp_Long_Term_Forecast(args)
        self.assertIsInstance(experiment.model, TSMixerModel)
        experiment.model.eval()
        target = torch.randn(2, 36, 7)
        history_marks, future_marks = torch.zeros(2, 24, 4), torch.zeros(2, 36, 4)
        output = experiment.model(self.x, history_marks, target, future_marks)
        self.assertEqual(output.shape, (2, 24, 7))
        prediction, truth = experiment._model_forward(
            SimpleNamespace(scale=False, target_idx=2), self.x, target, history_marks,
            future_marks, flag="train")
        self.assertEqual(prediction.shape, (2, 24, 1))
        torch.testing.assert_close(prediction, output[:, :, 2:3])
        torch.testing.assert_close(truth, target[:, -24:, 2:3])
        (prediction - truth).square().mean().backward()
        self.assertGreater(experiment.model.projection.weight.grad.abs().sum().item(), 0)

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
