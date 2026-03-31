# -*- coding: utf-8 -*-

# ***************************************************
# * File        : TSMixer.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-01-14
# * Version     : 1.0.011423
# * Description : TSMixer forecast model
# * Link        : https://arxiv.org/abs/2303.06053
# ***************************************************

import torch
import torch.nn as nn


class MixerLayer(nn.Module):

    def __init__(self, seq_len: int, channels: int, hidden_channels: int, dropout: float):
        super().__init__()
        self.seq_norm = nn.LayerNorm(channels)
        self.seq_mixer = nn.Sequential(
            nn.Linear(seq_len, seq_len),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(seq_len, seq_len),
        )
        self.channel_norm = nn.LayerNorm(channels)
        self.channel_mixer = nn.Sequential(
            nn.Linear(channels, hidden_channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_residual = x
        x = self.seq_norm(x)
        x = x.transpose(1, 2)
        x = self.seq_mixer(x)
        x = x.transpose(1, 2)
        x = x + seq_residual

        channel_residual = x
        x = self.channel_norm(x)
        x = self.channel_mixer(x)
        x = x + channel_residual

        return x


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.dropout = getattr(configs, "dropout", 0.1)
        self.num_blocks = max(1, getattr(configs, "e_layers", 2))
        hidden_channels = max(self.enc_in, getattr(configs, "d_ff", self.enc_in * 2))

        self.blocks = nn.ModuleList(
            [
                MixerLayer(
                    seq_len=self.seq_len,
                    channels=self.enc_in,
                    hidden_channels=hidden_channels,
                    dropout=self.dropout,
                )
                for _ in range(self.num_blocks)
            ]
        )
        self.temporal_projection = nn.Linear(self.seq_len, self.pred_len)
        self.channel_projection = (
            nn.Identity() if self.enc_in == self.c_out else nn.Linear(self.enc_in, self.c_out)
        )

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        x = x_enc
        for block in self.blocks:
            x = block(x)
        x = x.transpose(1, 2)
        x = self.temporal_projection(x)
        x = x.transpose(1, 2)
        x = self.channel_projection(x)
        return x

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)
        if self.task_name in {"imputation", "anomaly_detection"}:
            return self.forecast(x_enc)
        if self.task_name == "classification":
            x = self.forecast(x_enc)
            return x.reshape(x.shape[0], -1)
        return None
