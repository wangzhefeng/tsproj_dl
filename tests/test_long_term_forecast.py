import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics_dl import MAPE, MSPE


class DummyPredData:
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
        device = batch_x.device
        values = torch.arange(
            1,
            self.output_channels + 1,
            dtype=batch_x.dtype,
            device=device,
        ).reshape(1, 1, self.output_channels)
        return values.repeat(batch, 1, 1) * self.weight


class DummyTrainData:
    scale = False

    def save_scalers(self, path):
        return None


class DummyTrainModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, batch_x, batch_x_mark, dec_inp, batch_y_mark):
        return torch.ones_like(dec_inp[:, -1:, :]) * self.weight


class ForecastExperiment(Exp_Long_Term_Forecast):
    def _build_model(self):
        output_channels = 1 if self.args.features == "MS" else 3
        return DummyForecastModel(output_channels)

    def _get_data(self, flag):
        assert flag == "pred"
        batch_x = torch.zeros((1, 3, 3), dtype=torch.float32)
        batch_y = torch.zeros((1, 2, 3), dtype=torch.float32)
        batch_x_mark = torch.zeros((1, 3, 4), dtype=torch.float32)
        batch_y_mark = torch.zeros((1, 3, 4), dtype=torch.float32)
        return DummyPredData(self.args.features), [(batch_x, batch_y, batch_x_mark, batch_y_mark)]


class TrainExperiment(Exp_Long_Term_Forecast):
    def _build_model(self):
        return DummyTrainModel()

    def _get_data(self, flag):
        batch = (
            torch.zeros((1, 2, 1), dtype=torch.float32),
            torch.zeros((1, 2, 1), dtype=torch.float32),
            torch.zeros((1, 2, 1), dtype=torch.float32),
            torch.zeros((1, 2, 1), dtype=torch.float32),
        )
        return DummyTrainData(), [batch]


def build_args(output_dir, features="MS"):
    return SimpleNamespace(
        task_name="long_term_forecast",
        model="Dummy",
        model_id="unit_model",
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
        scale=1,
        forecast_require_artifacts=1,
    )


def build_train_args(output_dir, lradj):
    return SimpleNamespace(
        model="Dummy",
        use_gpu=0,
        gpu_type="cpu",
        use_multi_gpu=0,
        devices="0",
        checkpoints=str(output_dir / "checkpoints"),
        forecast_results=str(output_dir / "forecast_results"),
        test_results=str(output_dir / "test_results"),
        seq_len=2,
        label_len=1,
        pred_len=1,
        features="S",
        target="OT",
        freq="h",
        inverse=0,
        output_attention=False,
        use_amp=False,
        down_sampling_layers=0,
        loss="MSE",
        optimizer="adam",
        learning_rate=1e-3,
        lradj=lradj,
        pct_start=0.2,
        train_epochs=1,
        patience=1,
    )


class LongTermForecastTestCase(unittest.TestCase):

    def test_test_results_save_writes_window_and_stitched_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = object.__new__(Exp_Long_Term_Forecast)
            exp.args = SimpleNamespace(use_dtw=0)

            preds = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
            trues = np.array([[1.0], [2.0], [4.0]], dtype=np.float32)
            stitched_preds = np.array([[1.5], [2.5]], dtype=np.float32)
            stitched_trues = np.array([[1.0], [3.0]], dtype=np.float32)
            stitched_dates = pd.date_range("2026-01-01", periods=2, freq="h").to_numpy()

            exp._test_results_save(
                preds,
                trues,
                "unit_test",
                output_dir,
                stitched_preds=stitched_preds,
                stitched_trues=stitched_trues,
                overlap_counts=np.array([1, 2]),
                stitched_dates=stitched_dates,
            )

            result_text = (output_dir / "result_forecast.txt").read_text(encoding="utf-8")
            window_results = pd.read_csv(output_dir / "test_results_windows.csv")
            stitched_results = pd.read_csv(output_dir / "test_results.csv")

            self.assertIn("Window metrics:", result_text)
            self.assertIn("Timeline metrics:", result_text)
            self.assertEqual(list(window_results.columns), ["preds", "trues"])
            self.assertEqual(len(window_results), 3)
            self.assertEqual(list(stitched_results.columns), ["step", "date", "overlap_count", "preds", "trues"])
            self.assertEqual(len(stitched_results), 2)
            self.assertEqual(stitched_results["overlap_count"].tolist(), [1, 2])

    def test_stitch_window_predictions_with_stride_one_averages_overlap(self):
        preds = np.array([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
        trues = np.array([[[2.0], [4.0]], [[6.0], [8.0]]], dtype=np.float32)

        stitched_preds, stitched_trues, counts = Exp_Long_Term_Forecast._stitch_window_predictions(preds, trues, step=1)

        np.testing.assert_allclose(stitched_preds[:, 0], [1.0, 4.0, 7.0])
        np.testing.assert_allclose(stitched_trues[:, 0], [2.0, 5.0, 8.0])
        np.testing.assert_array_equal(counts, [1, 2, 1])

    def test_stitch_window_predictions_with_pred_len_stride_concatenates(self):
        preds = np.array([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)
        trues = np.array([[[5.0], [6.0]], [[7.0], [8.0]]], dtype=np.float32)

        stitched_preds, stitched_trues, counts = Exp_Long_Term_Forecast._stitch_window_predictions(preds, trues, step=2)

        np.testing.assert_allclose(stitched_preds[:, 0], [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(stitched_trues[:, 0], [5.0, 6.0, 7.0, 8.0])
        np.testing.assert_array_equal(counts, [1, 1, 1, 1])

    def test_stitch_window_predictions_with_partial_overlap_counts(self):
        preds = np.array([[[1.0], [3.0], [5.0]], [[7.0], [9.0], [11.0]]], dtype=np.float32)
        trues = np.array([[[2.0], [4.0], [6.0]], [[8.0], [10.0], [12.0]]], dtype=np.float32)

        stitched_preds, stitched_trues, counts = Exp_Long_Term_Forecast._stitch_window_predictions(preds, trues, step=2)

        np.testing.assert_allclose(stitched_preds[:, 0], [1.0, 3.0, 6.0, 9.0, 11.0])
        np.testing.assert_allclose(stitched_trues[:, 0], [2.0, 4.0, 7.0, 10.0, 12.0])
        np.testing.assert_array_equal(counts, [1, 1, 2, 1, 1])

    def test_safe_percentage_metrics_do_not_return_inf_for_zero_true(self):
        pred = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        true = np.array([0.0, 2.0, 0.0], dtype=np.float32)

        self.assertTrue(np.isfinite(MAPE(pred, true)))
        self.assertTrue(np.isfinite(MSPE(pred, true)))

    def test_forecast_saves_history_forecast_prediction_and_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = ForecastExperiment(build_args(output_dir))

            exp.forecast("unit_forecast", load=False)

            result_path = output_dir / "forecast_results" / "unit_forecast"
            history = pd.read_csv(result_path / "history.csv")
            forecast = pd.read_csv(result_path / "forecast.csv")
            prediction = np.load(result_path / "prediction.npy")
            summary = (result_path / "summary.txt").read_text(encoding="utf-8")
            metadata = pd.read_json(result_path / "metadata.json", typ="series")

            self.assertTrue((result_path / "history.csv").exists())
            self.assertTrue((result_path / "forecast.csv").exists())
            self.assertTrue((result_path / "metadata.json").exists())
            self.assertEqual(len(history), 3)
            self.assertEqual(len(forecast), 1)
            self.assertEqual(prediction.shape, (1, 1))
            self.assertEqual(list(forecast.columns), ["date", "OT"])
            self.assertAlmostEqual(float(forecast["OT"].iloc[0]), 101.0)
            self.assertIn("history_points:3, forecast_points:1", summary)
            self.assertIn("forecast_target:OT", summary)
            self.assertIn("history_range:2026-01-01 00:00:00 -> 2026-01-01 02:00:00", summary)
            self.assertEqual(metadata["setting"], "unit_forecast")
            self.assertEqual(metadata["forecast_points"], 1)
            self.assertEqual(metadata["prediction_columns"], ["OT"])

    def test_forecast_saves_all_m_feature_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = ForecastExperiment(build_args(output_dir, features="M"))

            exp.forecast("unit_forecast_m", load=False)

            result_path = output_dir / "forecast_results" / "unit_forecast_m"
            forecast = pd.read_csv(result_path / "forecast.csv")
            prediction = np.load(result_path / "prediction.npy")
            summary = (result_path / "summary.txt").read_text(encoding="utf-8")

            self.assertEqual(prediction.shape, (1, 3))
            self.assertEqual(list(forecast.columns), ["date", "load", "temp", "OT"])
            self.assertAlmostEqual(float(forecast["load"].iloc[0]), 1001.0)
            self.assertAlmostEqual(float(forecast["temp"].iloc[0]), 1002.0)
            self.assertAlmostEqual(float(forecast["OT"].iloc[0]), 1003.0)
            self.assertIn("forecast_target:OT", summary)

    def test_forecast_requires_checkpoint_when_loading_for_production(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = ForecastExperiment(build_args(output_dir))

            with self.assertRaises(FileNotFoundError) as ctx:
                exp.forecast("missing_forecast", load=True)

            self.assertIn("required forecast checkpoint not found", str(ctx.exception))

    def test_train_completes_with_type1_epoch_scheduler_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = TrainExperiment(build_train_args(output_dir, lradj="type1"))

            model = exp.train("unit_train_type1")

            self.assertIs(model, exp.model)
            self.assertTrue((output_dir / "checkpoints" / "unit_train_type1" / "checkpoint.pth").exists())

    def test_train_completes_with_tst_onecycle_scheduler_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            exp = TrainExperiment(build_train_args(output_dir, lradj="TST"))

            model = exp.train("unit_train_tst")

            self.assertIs(model, exp.model)
            self.assertTrue((output_dir / "checkpoints" / "unit_train_tst" / "checkpoint.pth").exists())


if __name__ == "__main__":
    unittest.main()
