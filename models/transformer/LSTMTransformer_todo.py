# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].detach()
        return self.dropout(x)


class Model(nn.Module):
    """
    Hybrid baseline: LSTM encoder followed by TransformerEncoder.
    """

    def __init__(self, configs) -> None:
        super().__init__()

        self.task_name = configs.task_name
        if self.task_name not in {"long_term_forecast", "short_term_forecast"}:
            raise ValueError("LSTMTransformer currently supports forecast tasks only.")

        self.pred_len = configs.pred_len
        self.c_out = configs.c_out
        self.hidden_size = getattr(configs, "hidden_size", configs.d_model)
        self.lstm_layers = max(1, getattr(configs, "num_layers", 1))
        self.transformer_layers = max(1, getattr(configs, "e_layers", 1))
        self.n_heads = configs.n_heads

        self.lstm = nn.LSTM(
            input_size=configs.enc_in,
            hidden_size=self.hidden_size,
            num_layers=self.lstm_layers,
            dropout=configs.dropout if self.lstm_layers > 1 else 0.0,
            batch_first=True,
        )
        self.positional_encoding = PositionalEncoding(
            d_model=self.hidden_size,
            dropout=configs.dropout,
            max_len=max(configs.seq_len, configs.pred_len) + 1,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.n_heads,
            dim_feedforward=configs.d_ff,
            dropout=configs.dropout,
            activation=configs.activation,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.transformer_layers,
        )
        self.head = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.pred_len * self.c_out),
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        encoded, _ = self.lstm(x_enc)
        encoded = self.positional_encoding(encoded)
        encoded = self.transformer(encoded)
        summary = encoded.mean(dim=1)
        output = self.head(summary)

        return output.view(x_enc.shape[0], self.pred_len, self.c_out)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        return self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)


__all__ = ["Model"]
