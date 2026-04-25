# -*- coding: utf-8 -*-

# ***************************************************
# * File        : N_HiTs.py
# * Author      : Zhefeng Wang
# * Email       : zfwang7@gmail.com
# * Date        : 2026-01-14
# * Version     : 1.0.011423
# * Description : N-HiTS forecast model
# * Link        : https://arxiv.org/abs/2201.12886
# ***************************************************

import sys
from pathlib import Path

ROOT = str(Path.cwd())
if ROOT not in sys.path:
    sys.path.append(ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F


class NHiTSBlock(nn.Module):

    def __init__(
        self,
        input_size: int,
        forecast_size: int,
        hidden_size: int,
        pool_kernel_size: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.pool = nn.AvgPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size, ceil_mode=True)
        pooled_size = (input_size + pool_kernel_size - 1) // pool_kernel_size

        layers = []
        in_features = pooled_size
        for _ in range(num_layers):
            layers.extend([nn.Linear(in_features, hidden_size), nn.GELU(), nn.Dropout(dropout)])
            in_features = hidden_size
        self.mlp = nn.Sequential(*layers)
        knot_size = max(4, min(forecast_size, pooled_size))
        self.backcast_head = nn.Linear(hidden_size, knot_size)
        self.forecast_head = nn.Linear(hidden_size, knot_size)

    def forward(self, x: torch.Tensor):
        pooled = self.pool(x.unsqueeze(1)).squeeze(1)
        features = self.mlp(pooled)
        backcast_knots = self.backcast_head(features).unsqueeze(1)
        forecast_knots = self.forecast_head(features).unsqueeze(1)
        backcast = F.interpolate(backcast_knots, size=self.input_size, mode="linear", align_corners=False).squeeze(1)
        forecast = F.interpolate(forecast_knots, size=self.forecast_size, mode="linear", align_corners=False).squeeze(1)
        return backcast, forecast


class Model(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        hidden_size = max(64, getattr(configs, "d_model", 128))
        pool_kernel = max(1, getattr(configs, "down_sampling_window", 2))
        num_blocks = max(1, getattr(configs, "e_layers", 3))
        num_layers = max(1, min(4, getattr(configs, "d_layers", 2)))
        dropout = getattr(configs, "dropout", 0.1)

        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(
                    input_size=self.seq_len,
                    forecast_size=self.pred_len,
                    hidden_size=hidden_size,
                    pool_kernel_size=pool_kernel,
                    num_layers=num_layers,
                    dropout=dropout,
                )
                for _ in range(num_blocks)
            ]
        )
        self.channel_projection = (
            nn.Identity() if self.enc_in == self.c_out else nn.Linear(self.enc_in, self.c_out)
        )

    def forecast(self, x_enc: torch.Tensor) -> torch.Tensor:
        batch_size, _, channels = x_enc.shape
        residuals = x_enc.transpose(1, 2).reshape(batch_size * channels, self.seq_len)
        forecast = x_enc[:, -1:, :].repeat(1, self.pred_len, 1)
        forecast = forecast.transpose(1, 2).reshape(batch_size * channels, self.pred_len)

        for block in self.blocks:
            backcast, block_forecast = block(residuals)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        forecast = forecast.reshape(batch_size, channels, self.pred_len).transpose(1, 2)
        forecast = self.channel_projection(forecast)
        return forecast

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        if self.task_name in {"long_term_forecast", "short_term_forecast"}:
            return self.forecast(x_enc)
        if self.task_name in {"imputation", "anomaly_detection"}:
            return self.forecast(x_enc)
        if self.task_name == "classification":
            x = self.forecast(x_enc)
            return x.reshape(x.shape[0], -1)
        return None
