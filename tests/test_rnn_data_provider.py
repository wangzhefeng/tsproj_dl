import unittest
from types import SimpleNamespace

from data_provider.RNNs_type.data_factory import data_provider


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
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class RNNDataProviderTestCase(unittest.TestCase):

    def test_ms_recursive_shape(self):
        args = build_args(features="MS", pred_method="recursive_multi_step")
        _, data_loader = data_provider(args, "train")
        batch_x, batch_y = next(iter(data_loader))

        self.assertEqual(tuple(batch_x.shape), (4, 24, 7))
        self.assertEqual(tuple(batch_y.shape), (4, 6, 1))

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


if __name__ == "__main__":
    unittest.main()
