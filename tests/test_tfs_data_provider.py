import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from data_provider.TFs_type.data_factory import data_provider


def build_args(**overrides):
    args = dict(
        task_name="long_term_forecast",
        root_path="./dataset/ETT-small",
        data_path="ETTh1.csv",
        data="ETTh1",
        target="OT",
        time="date",
        freq="h",
        features="MS",
        seq_len=24,
        label_len=12,
        pred_len=6,
        scale=1,
        inverse=1,
        train_ratio=0.7,
        test_ratio=0.2,
        batch_size=4,
        num_workers=0,
        embed="timeF",
        train_step=1,
        valid_step=1,
        testing_step=3,
        seasonal_patterns="Monthly",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class TFsDataProviderTestCase(unittest.TestCase):

    def test_valid_flag_is_supported(self):
        args = build_args()
        data_set, data_loader = data_provider(args, "valid")

        self.assertEqual(data_set.flag, "valid")
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(data_loader))
        self.assertEqual(tuple(batch_x.shape), (4, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (4, 18, 7))
        self.assertEqual(tuple(batch_x_mark.shape[:2]), (4, 24))
        self.assertEqual(tuple(batch_y_mark.shape[:2]), (4, 18))

    def test_test_length_respects_testing_step(self):
        args = build_args(testing_step=6, batch_size=1)
        data_set, _ = data_provider(args, "test")

        expected_total = len(data_set.data_x) - data_set.seq_len - data_set.pred_len + 1
        expected_len = (expected_total - 1) // args.testing_step + 1
        self.assertEqual(len(data_set), expected_len)

    def test_train_valid_and_test_lengths_respect_split_steps(self):
        args = build_args(train_step=2, valid_step=4, testing_step=6, batch_size=1)

        for flag, step_attr in (("train", "train_step"), ("valid", "valid_step"), ("test", "testing_step")):
            with self.subTest(flag=flag):
                data_set, _ = data_provider(args, flag)
                total = len(data_set.data_x) - data_set.seq_len - data_set.pred_len + 1
                expected_len = (total - 1) // getattr(args, step_attr) + 1

                self.assertEqual(len(data_set), expected_len)

    def test_only_train_loader_shuffles(self):
        args = build_args(batch_size=1)

        _, train_loader = data_provider(args, "train")
        _, valid_loader = data_provider(args, "valid")
        _, test_loader = data_provider(args, "test")
        _, pred_loader = data_provider(args, "pred")

        self.assertEqual(train_loader.sampler.__class__.__name__, "RandomSampler")
        self.assertEqual(valid_loader.sampler.__class__.__name__, "SequentialSampler")
        self.assertEqual(test_loader.sampler.__class__.__name__, "SequentialSampler")
        self.assertEqual(pred_loader.sampler.__class__.__name__, "SequentialSampler")

    def test_ms_test_dataset_has_target_inverse_and_segment_dates(self):
        args = build_args(features="MS", batch_size=1)
        data_set, data_loader = data_provider(args, "test")
        _, batch_y, _, _ = next(iter(data_loader))

        self.assertEqual(len(data_set.segment_dates), len(data_set.data_x))
        restored = data_set.inverse_transform_target(batch_y[:, -args.pred_len:, -1:].numpy())
        self.assertEqual(restored.shape, (1, args.pred_len, 1))

    def test_m_feature_full_inverse_round_trip_for_train_test_and_pred(self):
        args = build_args(features="M", batch_size=1)

        for flag in ("train", "test", "pred"):
            with self.subTest(flag=flag):
                data_set, _ = data_provider(args, flag)
                if flag == "pred":
                    scaled = data_set.scaled_history_values[:48]
                    raw = data_set.raw_history_values[:48]
                else:
                    scaled = data_set.scaled_segment_values[:48]
                    raw = data_set.raw_segment_values[:48]

                restored = data_set.inverse_transform_full(scaled)

                np.testing.assert_allclose(restored, raw, rtol=1e-5, atol=1e-5)

    def test_ms_target_inverse_round_trip_for_train_test_and_pred(self):
        args = build_args(features="MS", batch_size=1)

        for flag in ("train", "test", "pred"):
            with self.subTest(flag=flag):
                data_set, _ = data_provider(args, flag)
                target_slice = slice(data_set.target_idx, data_set.target_idx + 1)
                if flag == "pred":
                    scaled = data_set.scaled_history_values[:48, target_slice]
                    raw = data_set.raw_history_values[:48, target_slice]
                    self.assertEqual(data_set.pred_columns, ["OT"])
                else:
                    scaled = data_set.scaled_segment_values[:48, target_slice]
                    raw = data_set.raw_segment_values[:48, target_slice]

                restored = data_set.inverse_transform_target(scaled)

                np.testing.assert_allclose(restored, raw, rtol=1e-5, atol=1e-5)

    def test_pred_dataset_exposes_history_and_future_dates(self):
        args = build_args(features="MS", batch_size=1)
        data_set, data_loader = data_provider(args, "pred")
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(data_loader))

        self.assertEqual(tuple(batch_x.shape), (1, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (1, 12, 7))
        self.assertEqual(len(data_set.history_dates), args.seq_len)
        self.assertEqual(len(data_set.future_dates), args.pred_len)
        self.assertEqual(batch_y_mark.shape[1], args.label_len + args.pred_len)
        restored = data_set.inverse_transform_target(np.zeros((1, args.pred_len, 1), dtype=np.float32))
        self.assertEqual(restored.shape, (1, args.pred_len, 1))

    def test_pred_dataset_loads_training_scalers_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scaler_dir = Path(temp_dir)
            train_args = build_args(features="M", batch_size=1)
            train_data, _ = data_provider(train_args, "train")
            train_data.save_scalers(scaler_dir)

            pred_args = build_args(features="M", batch_size=1, scaler_path=str(scaler_dir))
            pred_data, _ = data_provider(pred_args, "pred")
            expected = train_data.full_scaler.transform(pred_data.raw_history_values)

            self.assertTrue((scaler_dir / "scalers.pkl").exists())
            self.assertTrue((scaler_dir / "scaler_metadata.json").exists())
            np.testing.assert_allclose(pred_data.scaled_history_values, expected, rtol=1e-5, atol=1e-5)

    def test_production_pred_requires_training_scaler_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            args = build_args(
                features="S",
                batch_size=1,
                scaler_path=str(Path(temp_dir) / "missing_scalers.pkl"),
                require_scaler_artifact_for_pred=True,
            )

            with self.assertRaises(FileNotFoundError) as ctx:
                data_provider(args, "pred")

            self.assertIn("required scaler artifacts not found", str(ctx.exception))

    def test_pred_dataset_rejects_discontinuous_forecast_history(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dates = pd.date_range("2026-01-01", periods=30, freq="h").tolist()
            dates[-2] = dates[-2] + pd.Timedelta(hours=1)
            frame = pd.DataFrame({
                "date": dates,
                "OT": np.arange(30, dtype=float),
            })
            frame.to_csv(root / "broken.csv", index=False)
            args = build_args(
                root_path=str(root),
                data_path="broken.csv",
                features="S",
                seq_len=24,
                label_len=12,
                pred_len=3,
                batch_size=1,
            )

            with self.assertRaises(ValueError) as ctx:
                data_provider(args, "pred")

            self.assertIn("forecast history timestamps are not continuous", str(ctx.exception))

    def test_test_dataset_loads_training_scalers_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            scaler_dir = Path(temp_dir)
            train_args = build_args(features="M", batch_size=1)
            train_data, _ = data_provider(train_args, "train")
            train_data.save_scalers(scaler_dir)

            test_args = build_args(features="M", batch_size=1, scaler_path=str(scaler_dir))
            test_data, _ = data_provider(test_args, "test")
            expected = train_data.full_scaler.transform(test_data.raw_segment_values)

            np.testing.assert_allclose(test_data.scaled_segment_values, expected, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
