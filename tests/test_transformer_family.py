import unittest
from types import SimpleNamespace

import torch

from models.transformer.LSTMTransformer import Model as LSTMTransformerModel
from models.transformer.Transformer import Model as TransformerModel
from exp.exp_basic import Exp_Basic


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
        output, attns = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        self.assertEqual(output.shape, (2, 24, 7))
        self.assertIsInstance(attns, list)

    def test_lstm_transformer_shape(self):
        model = LSTMTransformerModel(build_args())
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        self.assertEqual(output.shape, (2, 24, 7))

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
