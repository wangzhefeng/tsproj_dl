import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.transformer.PatchTST import Model as PatchTSTModel
from models.transformer.LSTMTransformer_todo import Model as LSTMTransformerModel
from models.transformer.Transformer import Model as TransformerModel
from exp.exp_basic import Exp_Basic, LazyModelDict
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


def unwrap_model_outputs_for_test(outputs, output_attention: bool):
    if output_attention and isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


def build_args(embed_type=0, rev=True, output_attention=False):
    return SimpleNamespace(
        task_name="long_term_forecast",
        seq_len=24,
        label_len=12,
        pred_len=24,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=32,
        d_ff=64,
        e_layers=2,
        d_layers=1,
        factor=1,
        n_heads=1,
        dropout=0.05,
        activation="gelu",
        output_attention=output_attention,
        embed="timeF",
        freq="h",
        embed_type=embed_type,
        rev=rev,
        hidden_size=32,
        num_layers=1,
    )


class DummyForwardData:
    scale = False


class DummyScaledForwardData:
    scale = True
    target_idx = 2

    def inverse_transform_full(self, data):
        return np.asarray(data) + 10.0

    def inverse_transform_target(self, data):
        data = np.asarray(data)
        if data.shape[-1] != 1:
            raise AssertionError(f"target inverse expected one channel, got {data.shape}")
        return data + 100.0


class DummyForwardModel(nn.Module):
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        return batch_x[:, -1:, :].repeat(1, 2, 1)


class DummyDecoderEchoModel(nn.Module):
    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        return dec_inp


class DummyForecastData:
    def __init__(self, features="MS"):
        self.scale = True
        self.feature_names = ["load", "temp", "OT"]
        self.pred_columns = ["OT"] if features == "MS" else self.feature_names
        self.raw_history_values = np.array(
            [
                [10.0, 20.0, 30.0],
                [11.0, 21.0, 31.0],
                [12.0, 22.0, 32.0],
            ],
            dtype=np.float32,
        )
        self.scaled_history_values = np.zeros((3, 3), dtype=np.float32)
        self.history_dates = pd.date_range("2026-01-01", periods=3, freq="h").to_numpy()
        self.future_dates = pd.date_range("2026-01-01 03:00:00", periods=1, freq="h").to_numpy()

    def inverse_transform_target(self, values):
        return np.asarray(values) + 100.0

    def inverse_transform_full(self, values):
        return np.asarray(values) + 1000.0

    def inverse_transform_history(self, values):
        return np.asarray(values) + 1000.0


class DummyForecastModel(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.output_channels = output_channels
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        batch = batch_x.shape[0]
        values = torch.arange(
            1,
            self.output_channels + 1,
            dtype=batch_x.dtype,
            device=batch_x.device,
        ).reshape(1, 1, self.output_channels)
        return values.repeat(batch, 1, 1) * self.weight


class DummyForecastExperiment(Exp_Long_Term_Forecast):
    def _build_model(self):
        output_channels = 1 if self.args.features == "MS" else 3
        return DummyForecastModel(output_channels)

    def _get_data(self, flag):
        assert flag == "pred"
        batch_x = torch.zeros((1, 3, 3), dtype=torch.float32)
        batch_y = torch.zeros((1, 2, 3), dtype=torch.float32)
        batch_x_mark = torch.zeros((1, 3, 4), dtype=torch.float32)
        batch_y_mark = torch.zeros((1, 3, 4), dtype=torch.float32)
        return DummyForecastData(self.args.features), [(batch_x, batch_y, batch_x_mark, batch_y_mark)]


class NumpyHostileLoss:
    def item(self):
        return 0.25

    def __array__(self, dtype=None):
        raise TypeError("device tensor cannot be converted to numpy directly")


class DummyValidExperiment(Exp_Long_Term_Forecast):
    def _model_forward(self, data, batch_x, batch_y, batch_x_mark, batch_y_mark, flag, reverse=False):
        return torch.ones((1, 2, 2), dtype=torch.float32), torch.zeros((1, 2, 2), dtype=torch.float32)


def build_forecast_args(output_dir, features="MS"):
    return SimpleNamespace(
        model="Dummy",
        use_gpu=0,
        gpu_type="cpu",
        use_multi_gpu=0,
        devices="0",
        checkpoints=str(output_dir / "checkpoints"),
        forecast_results=str(output_dir / "forecast_results"),
        test_results=str(output_dir / "test_results"),
        seq_len=3,
        label_len=2,
        pred_len=1,
        features=features,
        target="OT",
        freq="h",
        inverse=1,
        output_attention=False,
        use_amp=False,
        down_sampling_layers=0,
    )


class TransformerFamilyTestCase(unittest.TestCase):

    def setUp(self):
        self.x_enc = torch.randn(2, 24, 7)
        self.x_dec = torch.randn(2, 36, 7)
        self.x_mark_enc = torch.randint(0, 4, (2, 24, 4)).float()
        self.x_mark_dec = torch.randint(0, 4, (2, 36, 4)).float()

    def test_transformer_default_shape(self):
        model = TransformerModel(build_args(embed_type=0, rev=True))
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        self.assertEqual(output.shape, (2, 24, 7))

    def test_transformer_without_positional_embedding_shape(self):
        model = TransformerModel(build_args(embed_type=1, rev=False))
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        self.assertEqual(output.shape, (2, 24, 7))

    def test_transformer_with_output_attention(self):
        model = TransformerModel(build_args(embed_type=0, rev=True, output_attention=True))
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        self.assertEqual(output.shape, (2, 24, 7))

    def test_unwrap_model_outputs_keeps_tensor_when_attention_enabled(self):
        outputs = torch.randn(2, 24, 7)
        unwrapped = unwrap_model_outputs_for_test(outputs, output_attention=True)
        self.assertTrue(torch.equal(unwrapped, outputs))

    def test_unwrap_model_outputs_extracts_attention_tuple(self):
        outputs = (torch.randn(2, 24, 7), [torch.randn(2, 1, 24, 24)])
        unwrapped = unwrap_model_outputs_for_test(outputs, output_attention=True)
        self.assertEqual(unwrapped.shape, (2, 24, 7))

    def test_lstm_transformer_shape(self):
        model = LSTMTransformerModel(build_args())
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        self.assertEqual(output.shape, (2, 24, 7))

    def test_patchtst_shape(self):
        model = PatchTSTModel(build_args())
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        self.assertEqual(output.shape, (2, 24, 7))

    def test_exp_basic_lazy_loads_patchtst_model_class(self):
        exp = object.__new__(Exp_Basic)
        model_map = exp._scan_models_directory("transformer")
        exp.model_dict = LazyModelDict(model_map)

        model_class = exp.get_model_class("PatchTST")

        self.assertIs(model_class, PatchTSTModel)

    def test_acquire_device_respects_explicit_cpu_request(self):
        exp = object.__new__(Exp_Basic)
        exp.args = SimpleNamespace(use_gpu=1, gpu_type="cpu", devices="0", use_multi_gpu=0)

        with mock.patch("torch.cuda.is_available", return_value=True), \
             mock.patch("torch.backends.mps.is_available", return_value=True):
            device = exp._acquire_device()

        self.assertEqual(device.type, "cpu")
        self.assertFalse(exp.args.use_gpu)

    def test_acquire_device_does_not_use_cuda_when_only_mps_is_available(self):
        exp = object.__new__(Exp_Basic)
        exp.args = SimpleNamespace(use_gpu=1, gpu_type="cuda", devices="0", use_multi_gpu=0)

        with mock.patch("torch.cuda.is_available", return_value=False), \
             mock.patch("torch.backends.mps.is_available", return_value=True):
            device = exp._acquire_device()

        self.assertEqual(device.type, "cpu")
        self.assertFalse(exp.args.use_gpu)

    def test_model_forward_valid_returns_tensors_for_loss(self):
        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            label_len=2,
            pred_len=2,
            features="M",
            output_attention=False,
            use_amp=False,
            down_sampling_layers=0,
        )
        exp.device = torch.device("cpu")
        exp.model = DummyForwardModel()
        batch_x = torch.ones((1, 3, 2), dtype=torch.float32)
        batch_y = torch.ones((1, 4, 2), dtype=torch.float32)
        batch_x_mark = torch.zeros((1, 3, 4), dtype=torch.float32)
        batch_y_mark = torch.zeros((1, 4, 4), dtype=torch.float32)

        outputs, targets = exp._model_forward(
            DummyForwardData(), batch_x, batch_y, batch_x_mark, batch_y_mark, flag="valid", reverse=False
        )

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual(tuple(outputs.shape), (1, 2, 2))
        self.assertEqual(tuple(targets.shape), (1, 2, 2))

    def test_valid_averages_host_loss_items_not_device_tensors(self):
        exp = object.__new__(DummyValidExperiment)
        exp.model = DummyForwardModel()
        vali_loader = [
            (
                torch.zeros((1, 3, 2), dtype=torch.float32),
                torch.zeros((1, 4, 2), dtype=torch.float32),
                torch.zeros((1, 3, 4), dtype=torch.float32),
                torch.zeros((1, 4, 4), dtype=torch.float32),
            )
        ]

        loss = exp.valid(DummyForwardData(), vali_loader, lambda outputs, targets: NumpyHostileLoss())

        self.assertEqual(loss, 0.25)

    def test_model_forward_train_returns_tensors_for_loss(self):
        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            label_len=2,
            pred_len=2,
            features="M",
            output_attention=False,
            use_amp=False,
            down_sampling_layers=0,
        )
        exp.device = torch.device("cpu")
        exp.model = DummyForwardModel()
        batch_x = torch.ones((1, 3, 2), dtype=torch.float32)
        batch_y = torch.ones((1, 4, 2), dtype=torch.float32)
        batch_x_mark = torch.zeros((1, 3, 4), dtype=torch.float32)
        batch_y_mark = torch.zeros((1, 4, 4), dtype=torch.float32)

        outputs, targets = exp._model_forward(
            DummyForwardData(), batch_x, batch_y, batch_x_mark, batch_y_mark, flag="train", reverse=False
        )

        self.assertIsInstance(outputs, torch.Tensor)
        self.assertIsInstance(targets, torch.Tensor)
        self.assertEqual(tuple(outputs.shape), (1, 2, 2))
        self.assertEqual(tuple(targets.shape), (1, 2, 2))

    def test_model_forward_test_returns_numpy_for_results(self):
        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            label_len=2,
            pred_len=2,
            features="M",
            output_attention=False,
            use_amp=False,
            down_sampling_layers=0,
        )
        exp.device = torch.device("cpu")
        exp.model = DummyForwardModel()
        batch_x = torch.ones((1, 3, 2), dtype=torch.float32)
        batch_y = torch.ones((1, 4, 2), dtype=torch.float32)
        batch_x_mark = torch.zeros((1, 3, 4), dtype=torch.float32)
        batch_y_mark = torch.zeros((1, 4, 4), dtype=torch.float32)

        outputs, targets = exp._model_forward(
            DummyForwardData(), batch_x, batch_y, batch_x_mark, batch_y_mark, flag="test", reverse=False
        )

        self.assertIsInstance(outputs, np.ndarray)
        self.assertIsInstance(targets, np.ndarray)
        self.assertEqual(outputs.shape, (1, 2, 2))
        self.assertEqual(targets.shape, (1, 2, 2))

    def test_model_forward_test_full_inverse_keeps_all_m_features(self):
        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            label_len=2,
            pred_len=3,
            features="M",
            output_attention=False,
            use_amp=False,
            down_sampling_layers=0,
        )
        exp.device = torch.device("cpu")
        exp.model = DummyDecoderEchoModel()
        batch_x = torch.zeros((2, 4, 3), dtype=torch.float32)
        batch_y = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        batch_x_mark = torch.zeros((2, 4, 4), dtype=torch.float32)
        batch_y_mark = torch.zeros((2, 5, 4), dtype=torch.float32)

        outputs, targets = exp._model_forward(
            DummyScaledForwardData(), batch_x, batch_y, batch_x_mark, batch_y_mark, flag="test", reverse=True
        )

        self.assertIsInstance(outputs, np.ndarray)
        self.assertIsInstance(targets, np.ndarray)
        self.assertEqual(outputs.shape, (2, 3, 3))
        self.assertEqual(targets.shape, (2, 3, 3))
        np.testing.assert_allclose(outputs, np.full((2, 3, 3), 10.0, dtype=np.float32))

    def test_model_forward_test_ms_inverse_slices_target_before_inverse(self):
        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = SimpleNamespace(
            label_len=2,
            pred_len=3,
            features="MS",
            output_attention=False,
            use_amp=False,
            down_sampling_layers=0,
        )
        exp.device = torch.device("cpu")
        exp.model = DummyDecoderEchoModel()
        batch_x = torch.zeros((2, 4, 3), dtype=torch.float32)
        batch_y = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(2, 5, 3)
        batch_x_mark = torch.zeros((2, 4, 4), dtype=torch.float32)
        batch_y_mark = torch.zeros((2, 5, 4), dtype=torch.float32)

        outputs, targets = exp._model_forward(
            DummyScaledForwardData(), batch_x, batch_y, batch_x_mark, batch_y_mark, flag="test", reverse=True
        )

        self.assertIsInstance(outputs, np.ndarray)
        self.assertIsInstance(targets, np.ndarray)
        self.assertEqual(outputs.shape, (2, 3, 1))
        self.assertEqual(targets.shape, (2, 3, 1))
        np.testing.assert_allclose(outputs, np.full((2, 3, 1), 100.0, dtype=np.float32))
        np.testing.assert_allclose(targets, batch_y[:, -3:, 2:3].numpy() + 100.0)

    def test_forecast_ms_saves_only_target_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = DummyForecastExperiment(build_forecast_args(output_dir, features="MS"))

            exp.forecast("unit_forecast_ms", load=False)

            result_path = output_dir / "forecast_results" / "unit_forecast_ms"
            forecast = pd.read_csv(result_path / "forecast.csv")
            prediction = np.load(result_path / "prediction.npy")

            self.assertEqual(prediction.shape, (1, 1))
            self.assertEqual(list(forecast.columns), ["date", "OT"])
            self.assertAlmostEqual(float(forecast["OT"].iloc[0]), 101.0)

    def test_forecast_m_saves_all_feature_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = DummyForecastExperiment(build_forecast_args(output_dir, features="M"))

            exp.forecast("unit_forecast_m", load=False)

            result_path = output_dir / "forecast_results" / "unit_forecast_m"
            forecast = pd.read_csv(result_path / "forecast.csv")
            prediction = np.load(result_path / "prediction.npy")

            self.assertEqual(prediction.shape, (1, 3))
            self.assertEqual(list(forecast.columns), ["date", "load", "temp", "OT"])
            self.assertAlmostEqual(float(forecast["load"].iloc[0]), 1001.0)
            self.assertAlmostEqual(float(forecast["temp"].iloc[0]), 1002.0)
            self.assertAlmostEqual(float(forecast["OT"].iloc[0]), 1003.0)

    def test_exp_basic_does_not_register_transformer_aliases(self):
        exp = object.__new__(Exp_Basic)
        exp.args = build_args()
        exp.model_dict = {
            'Autoformer': "models.transformer.Autoformer",
            'Transformer': "models.transformer.Transformer",
            'LSTMTransformer': "models.transformer.LSTMTransformer",
        }
        self.assertNotIn("Transformer_v2", exp.model_dict)
        self.assertNotIn("Transformer_v3", exp.model_dict)


if __name__ == "__main__":
    unittest.main()
