import unittest
from pathlib import Path
from types import SimpleNamespace
import tempfile

import numpy as np
import torch

from exp.exp_forecasting_rnns import Exp_Long_Term_Forecast
from utils.metrics_dl import metric


def build_args(**overrides):
    args = dict(
        task_name="long_term_forecast",
        is_training=1,
        is_testing=1,
        is_forecasting=0,
        model_id="unit",
        model="LSTM_todo",
        data="ETTh1",
        root_path="./dataset/ETT-small",
        data_path="ETTh1.csv",
        features="MS",
        target="OT",
        time="date",
        freq="h",
        checkpoints="results/pretrained_models/",
        test_results="results/test_results/",
        forecast_results="results/forecast_results/",
        seq_len=24,
        label_len=0,
        pred_len=6,
        seasonal_patterns="Monthly",
        train_ratio=0.7,
        test_ratio=0.2,
        inverse=0,
        scale=1,
        embed="timeF",
        top_k=5,
        num_kernels=6,
        enc_in=7,
        dec_in=7,
        c_out=7,
        d_model=512,
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=2048,
        moving_avg=25,
        factor=1,
        distil=1,
        dropout=0.1,
        activation="gelu",
        output_attention=0,
        num_workers=0,
        itr=1,
        train_epochs=1,
        batch_size=2,
        patience=3,
        learning_rate=0.005,
        loss="MSE",
        lradj="type1",
        use_amp=0,
        use_dtw=0,
        use_gpu=0,
        gpu_type="cpu",
        use_multi_gpu=0,
        devices="0",
        p_hidden_dims="128,128",
        p_hidden_layers=2,
        feature_size=7,
        hidden_size=8,
        num_layers=1,
        output_size=1,
        pred_method="direct_multi_output",
        step_size=1,
        train_step=256,
        valid_step=256,
        testing_step=256,
        forecast_require_artifacts=1,
        rolling_data_path="ETTh1.csv",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class RNNExperimentTestCase(unittest.TestCase):

    def test_align_prediction_target_keeps_matching_horizon_shapes(self):
        outputs = torch.randn(4, 6, 1)
        targets = torch.randn(4, 6, 1)

        aligned_outputs, aligned_targets = Exp_Long_Term_Forecast._align_prediction_target(outputs, targets)

        self.assertEqual(tuple(aligned_outputs.shape), (4, 6, 1))
        self.assertEqual(tuple(aligned_targets.shape), (4, 6, 1))

    def test_align_prediction_target_supports_single_step_outputs(self):
        outputs = torch.randn(4, 1)
        targets = torch.randn(4, 1, 1)

        aligned_outputs, aligned_targets = Exp_Long_Term_Forecast._align_prediction_target(outputs, targets)

        self.assertEqual(tuple(aligned_outputs.shape), (4, 1))
        self.assertEqual(tuple(aligned_targets.shape), (4, 1))

    def test_stitch_window_predictions_respects_step(self):
        preds = np.array([[[1.0], [2.0]], [[10.0], [20.0]]])
        trues = np.array([[[3.0], [4.0]], [[30.0], [40.0]]])

        stitched_preds, stitched_trues, counts = Exp_Long_Term_Forecast._stitch_window_predictions(
            preds,
            trues,
            step=2,
        )

        np.testing.assert_allclose(stitched_preds.reshape(-1), np.array([1.0, 2.0, 10.0, 20.0]))
        np.testing.assert_allclose(stitched_trues.reshape(-1), np.array([3.0, 4.0, 30.0, 40.0]))
        np.testing.assert_array_equal(counts, np.array([1, 1, 1, 1]))

    def test_stitch_window_predictions_stride_one_averages_overlap(self):
        preds = np.array([[[1.0], [3.0]], [[5.0], [7.0]]])
        trues = np.array([[[2.0], [4.0]], [[6.0], [8.0]]])

        stitched_preds, stitched_trues, counts = Exp_Long_Term_Forecast._stitch_window_predictions(
            preds,
            trues,
            step=1,
        )

        np.testing.assert_allclose(stitched_preds[:, 0], np.array([1.0, 4.0, 7.0]))
        np.testing.assert_allclose(stitched_trues[:, 0], np.array([2.0, 5.0, 8.0]))
        np.testing.assert_array_equal(counts, np.array([1, 2, 1]))

    def test_stitch_window_predictions_marks_gaps_as_nan(self):
        preds = np.array([[[1.0], [2.0]], [[10.0], [20.0]]])
        trues = np.array([[[3.0], [4.0]], [[30.0], [40.0]]])

        stitched_preds, stitched_trues, counts = Exp_Long_Term_Forecast._stitch_window_predictions(
            preds,
            trues,
            step=4,
        )

        np.testing.assert_array_equal(counts, np.array([1, 1, 0, 0, 1, 1]))
        self.assertTrue(np.isnan(stitched_preds[2:4]).all())
        self.assertTrue(np.isnan(stitched_trues[2:4]).all())

    def test_test_results_save_excludes_gap_rows_from_timeline_csv(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args()
            exp = Exp_Long_Term_Forecast(args)
            preds = np.array([[[1.0], [2.0]], [[10.0], [20.0]]])
            trues = np.array([[[1.0], [2.0]], [[10.0], [20.0]]])
            stitched_preds = np.array([[1.0], [2.0], [np.nan], [np.nan], [10.0], [20.0]])
            stitched_trues = np.array([[1.0], [2.0], [np.nan], [np.nan], [10.0], [20.0]])
            overlap_counts = np.array([1, 1, 0, 0, 1, 1])

            exp._test_results_save(
                preds,
                trues,
                "gap_rows",
                root,
                stitched_preds=stitched_preds,
                stitched_trues=stitched_trues,
                overlap_counts=overlap_counts,
            )

            results = np.genfromtxt(root / "test_results.csv", delimiter=",", names=True)
            self.assertEqual(len(results), 4)
            self.assertTrue((results["overlap_count"] > 0).all())

    def test_test_results_save_writes_structured_window_results(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args()
            exp = Exp_Long_Term_Forecast(args)
            preds = np.array([[[1.0], [2.0]], [[10.0], [20.0]]])
            trues = np.array([[[3.0], [4.0]], [[30.0], [40.0]]])
            window_dates = np.array([
                ["2026-01-01T00:00:00", "2026-01-01T01:00:00"],
                ["2026-01-01T01:00:00", "2026-01-01T02:00:00"],
            ])

            exp._test_results_save(
                preds,
                trues,
                "window_rows",
                root,
                window_dates=window_dates,
            )

            window_results = np.genfromtxt(root / "test_results_windows.csv", delimiter=",", names=True, dtype=None, encoding="utf-8")
            self.assertEqual(len(window_results), 4)
            self.assertEqual(list(window_results.dtype.names), ["window_idx", "horizon", "date", "preds", "trues"])
            np.testing.assert_array_equal(window_results["window_idx"], np.array([0, 0, 1, 1]))
            np.testing.assert_array_equal(window_results["horizon"], np.array([1, 2, 1, 2]))

    def test_test_results_save_reports_baseline_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args()
            exp = Exp_Long_Term_Forecast(args)
            preds = np.arange(30, dtype=np.float32).reshape(15, 2, 1)
            trues = preds + 1.0
            stitched_preds = np.arange(30, dtype=np.float32).reshape(30, 1)
            stitched_trues = stitched_preds + 1.0
            overlap_counts = np.ones(30, dtype=int)

            exp._test_results_save(
                preds,
                trues,
                "baseline_metrics",
                root,
                stitched_preds=stitched_preds,
                stitched_trues=stitched_trues,
                overlap_counts=overlap_counts,
            )

            summary = (root / "result_forecast.txt").read_text(encoding="utf-8")
            self.assertIn("Timeline rolling metrics", summary)
            self.assertIn("Persistence baseline metrics", summary)
            self.assertIn("Seasonal naive baseline metrics", summary)

    def test_rolling_origin_plot_series_keeps_history_as_context_only(self):
        history = np.array([1.0, 2.0, 3.0])
        future_true = np.array([4.0, 5.0])
        future_pred = np.array([4.5, 4.8])

        series = Exp_Long_Term_Forecast._build_rolling_origin_plot_series(history, future_true, future_pred)

        np.testing.assert_array_equal(series["context_x"], np.array([-3, -2, -1]))
        np.testing.assert_array_equal(series["future_x"], np.array([0, 1]))
        np.testing.assert_array_equal(series["context"], history)
        np.testing.assert_array_equal(series["future_true"], future_true)
        np.testing.assert_array_equal(series["future_pred"], future_pred)

    def test_metric_ignores_near_zero_values_for_percentage_errors(self):
        preds = np.array([[100.0], [4.0]])
        trues = np.array([[0.0], [2.0]])

        _, _, _, _, mape, _, mspe, _ = metric(preds, trues, use_dtw=False)

        self.assertEqual(mape, 1.0)
        self.assertEqual(mspe, 1.0)

    def test_model_forward_lstm_outputs_horizon_target(self):
        args = build_args(model="LSTM_todo")
        exp = Exp_Long_Term_Forecast(args)
        data, loader = exp._get_data("train")
        batch_x, batch_y = next(iter(loader))

        outputs, targets = exp._model_forward(data, batch_x, batch_y, flag="train", reverse=False)

        self.assertEqual(tuple(outputs.shape), (2, 6, 1))
        self.assertEqual(tuple(targets.shape), (2, 6, 1))

    def test_model_forward_recursive_train_uses_single_step_target(self):
        args = build_args(model="LSTM_todo", pred_method="recursive_multi_step")
        exp = Exp_Long_Term_Forecast(args)
        data, loader = exp._get_data("train")
        batch_x, batch_y = next(iter(loader))

        outputs, targets = exp._model_forward(data, batch_x, batch_y, flag="train", reverse=False)

        self.assertEqual(tuple(batch_y.shape), (2, 1, 1))
        self.assertEqual(tuple(outputs.shape), (2, 1, 1))
        self.assertEqual(tuple(targets.shape), (2, 1, 1))

    def test_model_forward_recursive_test_generates_full_horizon(self):
        args = build_args(model="LSTM_todo", pred_method="recursive_multi_step", testing_step=512)
        exp = Exp_Long_Term_Forecast(args)
        data, loader = exp._get_data("test")
        batch_x, batch_y = next(iter(loader))

        outputs, targets = exp._model_forward(data, batch_x, batch_y, flag="test", reverse=False)

        self.assertEqual(tuple(batch_y.shape), (1, 6, 1))
        self.assertEqual(outputs.shape, (1, 6, 1))
        self.assertEqual(targets.shape, (1, 6, 1))

    def test_recursive_ms_rollout_only_replaces_target_column(self):
        args = build_args(model="LSTM_todo", pred_method="recursive_multi_step", features="MS")
        exp = object.__new__(Exp_Long_Term_Forecast)
        exp.args = args
        exp.device = torch.device("cpu")

        class RecordingOneStepModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inputs = []

            def forward(self, x):
                self.inputs.append(x.detach().clone())
                return x[:, -1:, -1:] + 1.0

        exp.model = RecordingOneStepModel()
        batch_x = torch.arange(1 * 4 * 3, dtype=torch.float32).reshape(1, 4, 3)
        exp.args.pred_len = 2

        preds = exp._forecast_recursive_tensor(batch_x)

        self.assertEqual(tuple(preds.shape), (1, 2, 1))
        second_input = exp.model.inputs[1]
        torch.testing.assert_close(second_input[:, -1:, :-1], batch_x[:, -1:, :-1])
        torch.testing.assert_close(second_input[:, -1:, -1:], preds[:, :1, :])

    def test_model_forward_gru_outputs_horizon_target(self):
        args = build_args(model="GRU_todo")
        exp = Exp_Long_Term_Forecast(args)
        data, loader = exp._get_data("train")
        batch_x, batch_y = next(iter(loader))

        outputs, targets = exp._model_forward(data, batch_x, batch_y, flag="train", reverse=False)

        self.assertEqual(tuple(outputs.shape), (2, 6, 1))
        self.assertEqual(tuple(targets.shape), (2, 6, 1))

    def test_forecast_direct_multi_output_saves_prediction_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args(
                checkpoints=str(root / "checkpoints"),
                test_results=str(root / "test_results"),
                forecast_results=str(root / "forecast_results"),
            )
            setting = "unit_setting"
            exp = Exp_Long_Term_Forecast(args)
            checkpoint_path = exp._get_model_path(setting)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": exp.model.state_dict()}, checkpoint_path)
            _, train_loader = exp._get_data("train")
            train_data, _ = exp._get_data("train")
            train_data.save_scalers(checkpoint_path.parent)

            exp.forecast(setting, load=True)

            forecast_path = root / "forecast_results" / setting
            self.assertTrue((forecast_path / "history.csv").exists())
            self.assertTrue((forecast_path / "forecast.csv").exists())
            self.assertTrue((forecast_path / "prediction.npy").exists())
            self.assertTrue((forecast_path / "metadata.json").exists())
            self.assertTrue((forecast_path / "summary.txt").exists())

    def test_test_load_requires_scaler_artifact_when_scaled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args(
                checkpoints=str(root / "checkpoints"),
                test_results=str(root / "test_results"),
                scale=1,
            )
            setting = "missing_scaler"
            exp = Exp_Long_Term_Forecast(args)
            checkpoint_path = exp._get_model_path(setting)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": exp.model.state_dict()}, checkpoint_path)

            with self.assertRaises(FileNotFoundError) as ctx:
                exp.test(setting, load=True)

            self.assertIn("test scaler artifact", str(ctx.exception))

    def test_test_saves_window_predictions_with_horizon_shape(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args(
                checkpoints=str(root / "checkpoints"),
                test_results=str(root / "test_results"),
                seq_len=24,
                pred_len=6,
                testing_step=512,
                batch_size=2,
                scale=1,
                inverse=0,
            )
            setting = "window_shape"
            exp = Exp_Long_Term_Forecast(args)
            checkpoint_path = exp._get_model_path(setting)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": exp.model.state_dict()}, checkpoint_path)
            train_data, _ = exp._get_data("train")
            train_data.save_scalers(checkpoint_path.parent)

            exp.test(setting, load=True)

            result_path = root / "test_results" / setting
            preds = np.load(result_path / "preds.npy")
            trues = np.load(result_path / "trues.npy")
            self.assertEqual(preds.ndim, 3)
            self.assertEqual(trues.ndim, 3)
            self.assertEqual(preds.shape[1:], (6, 1))
            self.assertEqual(trues.shape[1:], (6, 1))

    def test_test_results_save_reports_target_metrics_for_multichannel_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args(features="M", c_out=7, output_size=7)
            exp = Exp_Long_Term_Forecast(args)
            preds = np.ones((2, 3, 7), dtype=np.float32)
            trues = np.ones((2, 3, 7), dtype=np.float32)

            exp._test_results_save(preds, trues, "target_metrics", root, target_dim=-1)

            summary = (root / "result_forecast.txt").read_text(encoding="utf-8")
            self.assertIn("Target metrics", summary)

    def test_forecast_missing_checkpoint_has_clear_error(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args(
                checkpoints=str(root / "checkpoints"),
                forecast_results=str(root / "forecast_results"),
            )
            exp = Exp_Long_Term_Forecast(args)

            with self.assertRaises(FileNotFoundError) as ctx:
                exp.forecast("missing_setting", load=True)

            self.assertIn("forecast checkpoint", str(ctx.exception))

    def test_forecast_seq2seq_without_protocol_raises_not_implemented(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            args = build_args(
                checkpoints=str(root / "checkpoints"),
                forecast_results=str(root / "forecast_results"),
                pred_method="seq2seq_multi_step",
                forecast_require_artifacts=0,
            )
            exp = Exp_Long_Term_Forecast(args)

            with self.assertRaises(NotImplementedError):
                exp.forecast("seq2seq_setting", load=False)


if __name__ == "__main__":
    unittest.main()
