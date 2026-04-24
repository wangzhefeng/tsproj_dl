import unittest
from types import SimpleNamespace

import numpy as np

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
        testing_step=3,
        seasonal_patterns="Monthly",
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class TFsDataProviderTestCase(unittest.TestCase):

    def test_valid_alias_is_supported(self):
        args = build_args()
        data_set, data_loader = data_provider(args, "val")

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

    def test_ms_test_dataset_has_target_inverse_and_segment_dates(self):
        args = build_args(features="MS", batch_size=1)
        data_set, data_loader = data_provider(args, "test")
        _, batch_y, _, _ = next(iter(data_loader))

        self.assertEqual(len(data_set.segment_dates), len(data_set.data_x))
        restored = data_set.inverse_transform_target(batch_y[:, -args.pred_len:, -1:].numpy())
        self.assertEqual(restored.shape, (1, args.pred_len, 1))

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


if __name__ == "__main__":
    unittest.main()
