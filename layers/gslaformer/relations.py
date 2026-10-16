"""类型关系注意力：方向 bias 与统一加性 mask 的构建和消费（implementation.md §1.2）。

- query 行是接收方、key 列是发送方；self、注册的 src→dst 及 reverse_message 允许的
  反向通信有效，其余位置为 -inf。
- 每种业务关系的两个方向各有按 head 的可学习偏置；同端点多类型偏置相加；未连接位置
  不能因零偏置被误开放（偏置只叠加在已开放位置上）。
- padding／未生效 query 仅开放其 self 作为数值安全通路，且其输出随后置零；有效 query
  不得读取这些 key。禁止用 key_padding_mask 再次屏蔽安全 self，统一用加性 mask。
- 两层交互使用同时更新的上一层状态，不按节点遍历顺序原位更新。
"""
import math

import torch
import torch.nn as nn


class RelationBlock(nn.Module):
    """显式 MHA + Post-LN，始终按浮点加性 bias 计算，不经过 encoder 推理融合路径。"""

    def __init__(self, width, heads, dropout):
        super().__init__()
        self.attention = nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * width, width))

    def forward(self, hidden, src_mask):
        message, _ = self.attention(hidden, hidden, hidden, attn_mask=src_mask, need_weights=False)
        hidden = self.norm1(hidden + self.dropout1(message))
        return self.norm2(hidden + self.dropout2(self.ffn(hidden)))


class RelationAttention(nn.Module):
    """两层关系交互；消费逐时刻边表并生成 [B*L, H, N, N] 的加性 mask 与 bias。"""

    def __init__(self, relation_types, d_s, heads, layers=2, dropout=0.0):
        super().__init__()
        self.relation_types = relation_types  # [{name, reverse_message, ...}]
        self.heads = heads
        self.d_s = d_s
        self.layers = nn.ModuleList(RelationBlock(d_s, heads, dropout) for _ in range(layers))
        # 每类型每方向（0=正向 src→dst，1=反向 dst→src）按 head 的偏置
        self.direction_bias = nn.Parameter(torch.zeros(len(relation_types), 2, heads))
        nn.init.normal_(self.direction_bias, std=0.02)
        self._reverse_flags = [bool(spec["reverse_message"]) for spec in relation_types]

    def build_masks(self, total_nodes, edges_by_frame, effective):
        """从边表构建加性 mask 与偏置张量。

        edges_by_frame：长度 B*L 的列表，元素为 (src, dst, type_idx) 元组列表（batch 统一
        拼接轴索引）；effective：[B*L, N] bool，False 表示 padding 或未生效。
        返回 additive [F,1,N,N]（0/-inf）与 bias [F,H,N,N]。
        """
        frames = len(edges_by_frame)
        device = effective.device
        allow = torch.eye(total_nodes, dtype=torch.bool, device=device).expand(frames, -1, -1).clone()
        # 在 Python 中只收集整数拓扑；一次 index_add 累加可学习 bias。
        # 逐边修改稠密 bias 会产生大量 CopySlices 反向节点，每条边反复复制整个分数平面。
        positions, kinds, directions = [], [], []
        structures = {}
        for frame, edges in enumerate(edges_by_frame):
            key = tuple(tuple(edge) for edge in edges)
            if key not in structures:
                local_positions, local_kinds, local_directions = [], [], []
                for src, dst, kind in edges:
                    if src == dst:
                        continue
                    local_positions.append(dst * total_nodes + src)
                    local_kinds.append(kind)
                    local_directions.append(0)
                    if self._reverse_flags[kind]:
                        local_positions.append(src * total_nodes + dst)
                        local_kinds.append(kind)
                        local_directions.append(1)
                structures[key] = local_positions, local_kinds, local_directions
            local_positions, local_kinds, local_directions = structures[key]
            positions.extend(frame * total_nodes * total_nodes + p for p in local_positions)
            kinds.extend(local_kinds)
            directions.extend(local_directions)
        flat_bias = self.direction_bias.new_zeros(frames * total_nodes * total_nodes, self.heads)
        if positions:
            positions = torch.tensor(positions, dtype=torch.long, device=device)
            kinds = torch.tensor(kinds, dtype=torch.long, device=device)
            directions = torch.tensor(directions, dtype=torch.long, device=device)
            allow.reshape(-1)[positions] = True
            flat_bias = flat_bias.index_add(0, positions, self.direction_bias[kinds, directions])
        bias = flat_bias.reshape(frames, total_nodes, total_nodes, self.heads).permute(0, 3, 1, 2)
        invalid = ~effective
        if invalid.any():
            # 无效节点：行仅 self，列对其他行屏蔽（加性 mask 统一表达，不用 key_padding）
            row_block = invalid.unsqueeze(-1).expand(-1, -1, total_nodes)
            allow = allow & ~row_block & ~row_block.transpose(1, 2)
            diagonal = torch.eye(total_nodes, dtype=torch.bool, device=device).expand(
                frames, -1, -1).bool()
            allow = allow | (diagonal & invalid.unsqueeze(-1))
            bias = bias * allow.unsqueeze(1)
        additive = torch.zeros_like(allow, dtype=torch.float32)
        additive[~allow] = -math.inf
        return additive.unsqueeze(1), bias

    def forward(self, tokens, edges_by_frame, effective):
        """tokens [F, N, d_s]（F=B*L）；返回关系增强实体 [F, N, d_s]，无效行置零。"""
        additive, bias = self.build_masks(tokens.shape[1], edges_by_frame, effective)
        frames, nodes = tokens.shape[0], tokens.shape[1]
        merged = (additive + bias).reshape(frames * self.heads, nodes, nodes)
        hidden = tokens
        for layer in self.layers:
            hidden = layer(hidden, src_mask=merged)
        if effective is not None:
            hidden = hidden * effective.unsqueeze(-1).to(hidden.dtype)
        return hidden
