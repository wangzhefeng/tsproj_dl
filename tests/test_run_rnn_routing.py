import unittest
from types import SimpleNamespace

from exp.exp_forecasting_rnns import Exp_Long_Term_Forecast as RNNExp
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast as TFExp
from run import _select_experiment_class


class RunRoutingTestCase(unittest.TestCase):

    def test_rnn_todo_model_uses_rnn_experiment(self):
        args = SimpleNamespace(task_name="long_term_forecast", model="LSTM_todo", data="ETTh1")

        self.assertIs(_select_experiment_class(args), RNNExp)

    def test_gru_todo_model_uses_rnn_experiment(self):
        args = SimpleNamespace(task_name="long_term_forecast", model="GRU_todo", data="ETTh1")

        self.assertIs(_select_experiment_class(args), RNNExp)

    def test_transformer_model_stays_on_transformer_experiment(self):
        args = SimpleNamespace(task_name="long_term_forecast", model="Transformer", data="ETTh1")

        self.assertIs(_select_experiment_class(args), TFExp)


if __name__ == "__main__":
    unittest.main()
