"""系统时间链、完整未来路径与条件融合（implementation.md §1.4–1.5）。

- 历史 token：动态状态按固定组顺序展平，拼标准化总负荷、observed／generated 标记、
  逐动态组 coverage／generated 和本地日历（日内／周内 sin/cos 四个数）；
  联合 Linear 映射至 d_t，再加窗口内 0..L-1 固定正弦位置。
- 时间 Transformer：Post-LN、GELU、FFN=4*d_t，显式上三角屏蔽未来；最后 token 输出 h。
- FuturePathEncoder：读取 origin 已知的整段计划／日历与 mask（K=0 时仅日历），
  加相对原起点的正弦位置后经一层双向注意力生成 p [B,H,d_t]。
- ConditionFusion：拼接历史或回灌 hidden、静态 c、p[:,k] 与相对原 origin 的 lead(k)。
"""
from datetime import datetime, timezone
import math

import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
# 确定性编码
# ------------------------------------------------------------------------------

def sinusoidal_positions(length, dim, device=None, dtype=torch.float32, offset=0):
    """固定正弦位置编码 [length, dim]；窗口内 0..L-1，滑窗时按窗口重算。"""
    position = torch.arange(offset, offset + length, dtype=torch.float32, device=device)
    half = dim // 2
    frequency = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                                            device=device) / max(half - 1, 1))
    angle = position.unsqueeze(1) * frequency.unsqueeze(0)
    encoding = torch.zeros(length, dim, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angle[:, :encoding[:, 0::2].shape[1]])
    encoding[:, 1::2] = torch.cos(angle[:, :encoding[:, 1::2].shape[1]])
    return encoding.to(dtype)


def lead_encoding(leads, dim):
    """相对原预测起点的 lead(k) 正弦编码 [B,H,dim]。"""
    half = dim // 2
    frequency = torch.exp(-math.log(10000.0) * torch.arange(half, dtype=torch.float32,
                                                            device=leads.device) / max(half - 1, 1))
    angle = leads.unsqueeze(-1).float() * frequency
    encoding = torch.zeros(*leads.shape, dim, device=leads.device, dtype=torch.float32)
    encoding[..., 0::2] = torch.sin(angle[..., :encoding[..., 0::2].shape[-1]])
    encoding[..., 1::2] = torch.cos(angle[..., :encoding[..., 1::2].shape[-1]])
    return encoding


def calendar_features(times, tz_name):
    """本地日历相位 [B,T,4]：日内与周内的 sin/cos；UTC 秒输入，按 manifest 时区取值。"""
    from zoneinfo import ZoneInfo
    zone = ZoneInfo(tz_name)
    rows = []
    for sample in times.tolist():
        marks = []
        for stamp in sample:
            local = datetime.fromtimestamp(stamp, timezone.utc).astimezone(zone)
            day = (local.hour * 3600 + local.minute * 60 + local.second) / 86400
            week = (local.weekday() + day) / 7
            marks.append((math.sin(2 * math.pi * day), math.cos(2 * math.pi * day),
                          math.sin(2 * math.pi * week), math.cos(2 * math.pi * week)))
        rows.append(marks)
    return torch.tensor(rows, dtype=torch.float32, device=times.device)


# ------------------------------------------------------------------------------
# 模块
# ------------------------------------------------------------------------------

class TemporalEncoder(nn.Module):
    """历史窗口 token → 系统时间表示 h（因果）。"""

    def __init__(self, latent_dim, dynamic_groups, d_t, heads, layers, dropout=0.0):
        super().__init__()
        self.d_t = d_t
        width = dynamic_groups * latent_dim + 1 + 2 + 2 * dynamic_groups + 4
        self.project = nn.Linear(width, d_t)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model=d_t, nhead=heads, dim_feedforward=4 * d_t,
                                       activation="gelu", dropout=dropout, batch_first=True)
            for _ in range(layers))

    def forward(self, tokens):
        """tokens [B,L,width] → h 全序列 [B,L,d_t]（因果上三角屏蔽未来）。"""
        hidden = self.project(tokens)
        length = tokens.shape[1]
        positions = sinusoidal_positions(length, self.d_t, device=tokens.device)
        hidden = hidden + positions.unsqueeze(0)
        causal = torch.triu(torch.ones(length, length, dtype=torch.bool,
                                       device=tokens.device), diagonal=1)
        mask = torch.zeros(length, length, device=tokens.device)
        mask[causal] = -math.inf
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask)
        return hidden


class FuturePathEncoder(nn.Module):
    """origin 已知的整段未来路径一次编码；双向注意力不违反预测因果性。"""

    def __init__(self, known_dim, d_t, heads, layers, dropout=0.0):
        super().__init__()
        self.d_t = d_t
        # 数值与 mask 拥有独立权重列，避免 observed=-1 与缺失输入抵消。
        self.project = nn.Linear(4 + 2 * known_dim, d_t)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(d_model=d_t, nhead=heads, dim_feedforward=4 * d_t,
                                       activation="gelu", dropout=dropout, batch_first=True)
            for _ in range(layers))

    def forward(self, future_known, future_mask, calendar):
        """future_known [B,H,K]、mask [B,H,K]、calendar [B,H,4] → p [B,H,d_t]。"""
        values = torch.where(future_mask, future_known, torch.zeros_like(future_known))
        source = self.project(torch.cat([calendar, values, future_mask.to(values.dtype)], dim=-1))
        length = calendar.shape[1]
        positions = sinusoidal_positions(length, self.d_t, device=calendar.device)
        hidden = source + positions.unsqueeze(0)
        for layer in self.layers:
            hidden = layer(hidden)
        return hidden


class ConditionFusion(nn.Module):
    """历史或回灌 hidden、静态 c、p[:,k] 与 lead(k) 的共享条件融合。"""

    def __init__(self, d_t, lead_dim, dropout=0.0):
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(3 * d_t + lead_dim, d_t), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_t, d_t))

    def forward(self, hidden, condition, path_step, lead):
        return self.project(torch.cat([hidden, condition, path_step, lead], dim=-1))
