import unittest
from types import SimpleNamespace

import pandas as pd

from data_provider.RNNs_type.data_factory import data_provider
from data_provider.RNNs_type import data_factory


def build_args(**overrides):
    args = dict(
        root_path="./dataset/ETT-small",
        data_path="ETTh1.csv",
        target="OT",
        time="date",
        freq="h",
        features="MS",
        seq_len=24,
        pred_len=6,
        pred_method="recursive_multi_step",
        step_size=1,
        scale=1,
        train_ratio=0.7,
        test_ratio=0.2,
        batch_size=4,
        num_workers=0,
        embed="timeF",
        train_step=1,
        valid_step=1,
        testing_step=1,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class RNNDataProviderTestCase(unittest.TestCase):

    def test_ms_recursive_shape(self):
        args = build_args(features="MS", pred_method="recursive_multi_step")
        _, data_loader = data_provider(args, "train")
        batch_x, batch_y = next(iter(data_loader))

        self.assertEqual(tuple(batch_x.shape), (4, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (4, 1, 1))

    def test_recursive_test_keeps_full_horizon_for_backtest_metrics(self):
        args = build_args(features="MS", pred_method="recursive_multi_step", batch_size=1)
        _, data_loader = data_provider(args, "test")
        batch_x, batch_y = next(iter(data_loader))

        self.assertEqual(tuple(batch_x.shape), (1, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (1, 6, 1))

    def test_m_direct_multi_output_shape(self):
        args = build_args(features="M", pred_method="direct_multi_output")
        _, data_loader = data_provider(args, "train")
        batch_x, batch_y = next(iter(data_loader))

        self.assertEqual(tuple(batch_x.shape), (4, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (4, 6, 7))

    def test_direct_recursive_mix_keeps_horizon_supervision(self):
        args = build_args(features="MS", pred_method="direct_recursive_multi_step_mix")
        _, data_loader = data_provider(args, "train")
        batch_x, batch_y = next(iter(data_loader))

        self.assertEqual(tuple(batch_x.shape), (4, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (4, 6, 1))

    def test_pred_dataset_shape(self):
        args = build_args(
            features="MS",
            pred_method="recursive_multi_step",
            batch_size=1,
            rolling_data_path="ETTh1.csv",
        )
        _, data_loader = data_provider(args, "pred")
        batch_x, batch_y = next(iter(data_loader))

        self.assertEqual(tuple(batch_x.shape), (1, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (1, 6, 1))

    def test_inverse_transform_target_shape(self):
        args = build_args(features="MS", pred_method="direct_multi_output")
        data_set, data_loader = data_provider(args, "train")
        _, batch_y = next(iter(data_loader))
        restored = data_set.inverse_transform(batch_y)

        self.assertEqual(tuple(restored.shape), (4, 6, 1))

    def test_valid_flag_is_supported(self):
        args = build_args(pred_method="direct_multi_output")
        data_set, data_loader = data_provider(args, "valid")
        batch_x, batch_y = next(iter(data_loader))

        self.assertEqual(data_set.flag, "valid")
        self.assertEqual(tuple(batch_x.shape), (4, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (4, 6, 1))

    def test_val_flag_is_rejected(self):
        args = build_args()

        with self.assertRaises(ValueError) as ctx:
            data_provider(args, "val")

        self.assertIn("Unsupported data flag", str(ctx.exception))

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

    def test_train_valid_and_test_lengths_respect_split_steps(self):
        args = build_args(pred_method="direct_multi_output", train_step=2, valid_step=3, testing_step=5, batch_size=1)

        for flag, step_attr in (("train", "train_step"), ("valid", "valid_step"), ("test", "testing_step")):
            with self.subTest(flag=flag):
                data_set, _ = data_provider(args, flag)
                expected_total = len(data_set.data) - data_set.seq_len - data_set.pred_len + 1
                expected_len = (expected_total - 1) // getattr(args, step_attr) + 1

                self.assertEqual(len(data_set), expected_len)

    def test_invalid_flag_is_rejected(self):
        args = build_args()

        with self.assertRaises(ValueError) as ctx:
            data_provider(args, "validation")

        self.assertIn("Unsupported data flag", str(ctx.exception))

    def test_data_factory_main_smoke_runs_all_local_flags(self):
        data_factory.main()

    def test_missing_time_column_has_clear_error(self):
        with self.subTest("missing time column"):
            import tempfile
            from pathlib import Path

            with tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                pd.DataFrame({"OT": [1.0, 2.0, 3.0]}).to_csv(root / "bad.csv", index=False)
                args = build_args(root_path=str(root), data_path="bad.csv", seq_len=1, pred_len=1)

                with self.assertRaises(ValueError) as ctx:
                    data_provider(args, "train")

                self.assertIn("Time column `date` not found", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
