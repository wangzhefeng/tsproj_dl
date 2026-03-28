import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import DataEmbedding, DataEmbedding_wo_pos
from layers.Invertible import RevIN
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.Transformer_EncDec import (
    Decoder, DecoderLayer,
    Encoder, EncoderLayer,
)


class Model(nn.Module):
    """
    Vanilla Transformer
    with O(L^2) complexity
    Paper link: https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        
        # params
        self.task_name = configs.task_name
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_revin = bool(getattr(configs, "rev", False))

        if getattr(configs, "embed_type", 0) == 0:
            embedding_cls = DataEmbedding
        elif getattr(configs, "embed_type", 0) == 1:
            embedding_cls = DataEmbedding_wo_pos
        else:
            raise ValueError(f"Unsupported embed_type: {configs.embed_type}")
        
        # Encoder Embedding
        self.enc_embedding = embedding_cls(
            c_in=configs.enc_in, 
            d_model=configs.d_model, 
            embed_type=configs.embed, 
            freq=configs.freq,
            dropout=configs.dropout,
        )
        # Encoder
        self.encoder = Encoder(
            attn_layers = [
                EncoderLayer(
                    attention = AttentionLayer(
                        attention=FullAttention(
                            mask_flag=False, 
                            factor=configs.factor, 
                            attention_dropout=configs.dropout, 
                            output_attention=configs.output_attention
                        ), 
                        d_model=configs.d_model, 
                        n_heads=configs.n_heads,
                    ),
                    d_model = configs.d_model,
                    d_ff = configs.d_ff,
                    dropout = configs.dropout,
                    activation = configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer = torch.nn.LayerNorm(configs.d_model)
        )
        # Decoder
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.dec_embedding = embedding_cls(
                c_in = configs.dec_in, 
                d_model = configs.d_model, 
                embed_type = configs.embed, 
                freq = configs.freq,
                dropout = configs.dropout
            )
            self.decoder = Decoder(
                layers = [
                    DecoderLayer(
                        self_attention = AttentionLayer(
                            attention=FullAttention(
                                mask_flag=True, 
                                factor=configs.factor, 
                                attention_dropout=configs.dropout, 
                                output_attention=False
                            ),
                            d_model=configs.d_model, 
                            n_heads=configs.n_heads,
                        ),
                        cross_attention = AttentionLayer(
                            attention=FullAttention(
                                mask_flag=False, 
                                factor=configs.factor, 
                                attention_dropout=configs.dropout, 
                                output_attention=False
                            ),
                            d_model=configs.d_model, 
                            n_heads=configs.n_heads
                        ),
                        d_model=configs.d_model,
                        d_ff = configs.d_ff,
                        dropout = configs.dropout,
                        activation = configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer = torch.nn.LayerNorm(configs.d_model),
                projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
            )
        if self.task_name == 'imputation':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'anomaly_detection':
            self.projection = nn.Linear(configs.d_model, configs.c_out, bias=True)
        if self.task_name == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(configs.d_model * configs.seq_len, configs.num_class)

        self.revin = RevIN(configs.enc_in) if self.use_revin else None

    def _maybe_norm(self, x):
        return self.revin(x, 'norm') if self.revin is not None else x

    def _maybe_denorm(self, x):
        if self.revin is None:
            return x

        if x.shape[-1] == self.revin.num_features:
            return self.revin(x, 'denorm')

        return x

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        x_enc = self._maybe_norm(x_enc)
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        dec_out = self.dec_embedding(x_dec, x_mark_dec)
        # Encoder
        enc_out, attns = self.encoder(enc_out, attn_mask = None)
        # Decoder
        dec_out = self.decoder(dec_out, enc_out, x_mask = None, cross_mask = None)
        dec_out = self._maybe_denorm(dec_out)

        if self.output_attention:
            return dec_out, attns

        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        x_enc = self._maybe_norm(x_enc)
        # Embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        dec_out = self._maybe_denorm(dec_out)
        return dec_out

    def anomaly_detection(self, x_enc):
        x_enc = self._maybe_norm(x_enc)
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        dec_out = self.projection(enc_out)
        dec_out = self._maybe_denorm(dec_out)
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)  # the output transformer encoder/decoder embeddings don't include non-linearity
        output = self.dropout(output)
        output = output * x_mark_enc.unsqueeze(-1)  # zero-out padding embeddings
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.projection(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask = None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            if self.output_attention:
                forecast, attns = dec_out
                return forecast[:, -self.pred_len:, :], attns
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        
        return None




# 测试代码 main 函数
def main():
    pass

if __name__ == "__main__":
    main()
