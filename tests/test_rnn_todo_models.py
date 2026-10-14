import unittest
from types import SimpleNamespace

import torch

from models.rnn.GRU_todo import Model as GRUModel
from models.rnn.LSTM_todo import Model as LSTMModel


def build_args(**overrides):
    args = dict(
        features="MS",
        feature_size=7,
        output_size=1,
        hidden_size=16,
        num_layers=2,
        pred_len=6,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


class RNNTodoModelsTestCase(unittest.TestCase):

    def test_lstm_todo_ms_forward_shape(self):
        model = LSTMModel(build_args())
        outputs = model(torch.randn(2, 24, 7))

        self.assertEqual(tuple(outputs.shape), (2, 6, 1))

    def test_gru_todo_ms_forward_shape(self):
        model = GRUModel(build_args())
        outputs = model(torch.randn(2, 24, 7))

        self.assertEqual(tuple(outputs.shape), (2, 6, 1))

    def test_gru_todo_m_forward_shape(self):
        model = GRUModel(build_args(features="M", output_size=7))
        outputs = model(torch.randn(2, 24, 7))

        self.assertEqual(tuple(outputs.shape), (2, 6, 7))

    def test_gru_todo_ms_ignores_incorrect_output_size(self):
        model = GRUModel(build_args(features="MS", output_size=99))
        outputs = model(torch.randn(2, 24, 7))

        self.assertEqual(tuple(outputs.shape), (2, 6, 1))

    def test_gru_todo_output_allows_negative_scaled_values(self):
        model = GRUModel(build_args())
        for param in model.parameters():
            param.data.zero_()

        output_layer = getattr(model, "horizon_projection", getattr(model, "fc", None))
        self.assertIsNotNone(output_layer)
        output_layer.bias.data.fill_(-1.0)

        outputs = model(torch.zeros(2, 24, 7))

        self.assertTrue(torch.all(outputs < 0))

    def test_lstm_todo_horizon_head_not_limited_by_input_length(self):
        model = LSTMModel(build_args(pred_len=30))
        outputs = model(torch.randn(2, 24, 7))

        self.assertEqual(tuple(outputs.shape), (2, 30, 1))


if __name__ == "__main__":
    unittest.main()
