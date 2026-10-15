"""属性驱动的实体 token 与池化前实体历史编码（implementation.md §1.1）。

- 每组独立适配器：标准化动态值／observed／applicable／静态数值及 mask／类别 embedding
  拼接后 Linear→GELU→Linear 映射到 d_s；类别 embedding 宽度为 category_dim。
- 动态组用单层单向 GRUCell(d_s, d_s)，同组实体与时刻共享参数；不使用实体 ID 查表、
  不加实体序号位置编码。
- present=false 或 padding=true 的状态置零，重新出现从零开始；缺测但 present=true 时
  用填零数值与 mask 输入 GRU，状态不清零。
- 纯静态组无 GRU，直接用当时属性 token；父级／属性变化不重置 hidden。
"""
import torch
import torch.nn as nn


# ------------------------------------------------------------------------------
# 组适配器与实体历史编码
# ------------------------------------------------------------------------------

class GroupAdapter(nn.Module):
    """单组实体 token 适配：数值与类别证据拼接后投影到 d_s。"""

    def __init__(self, dynamic_features, static_numeric_features, static_category_features,
                 vocab_sizes, d_s, category_dim):
        super().__init__()
        self.dynamic_features = dynamic_features
        self.static_numeric_features = static_numeric_features
        self.category_embeddings = nn.ModuleList(
            [nn.Embedding(size, category_dim) for size in
             (vocab_sizes.get(name, 2) for name in static_category_features)])
        width = (dynamic_features * 3                  # value, observed, applicable
                 + static_numeric_features * 2          # value, mask
                 + len(static_category_features) * category_dim)
        self.project = nn.Sequential(nn.Linear(width, d_s), nn.GELU(), nn.Linear(d_s, d_s))

    def forward(self, values, observed, applicable, static_numeric, static_mask, static_category):
        """输入为单组张量：values/observed [B,L,N,F]、applicable [B,N,F]、
        static_numeric/mask [B,L,N,S]、static_category [B,L,N,C]；输出 [B,L,N,d_s]。"""
        pieces = [values, observed.to(values.dtype),
                  applicable[:, None].expand(-1, values.shape[1], -1, -1).to(values.dtype),
                  static_numeric, static_mask.to(values.dtype)]
        for index, embedding in enumerate(self.category_embeddings):
            pieces.append(embedding(static_category[..., index]))
        return self.project(torch.cat(pieces, dim=-1))


class EntityHistoryEncoder(nn.Module):
    """全部组的适配器与动态组共享 GRU；输出逐组 [B,L,N,d_s] 与末端实体 memory。"""

    def __init__(self, groups_spec, vocab_sizes, d_s, category_dim, *, use_gru=True):
        super().__init__()
        self.groups_spec = groups_spec
        self.d_s = d_s
        self.use_gru = use_gru
        self.adapters = nn.ModuleDict()
        self.cells = nn.ModuleDict()
        for spec in groups_spec:
            name = spec["name"]
            self.adapters[name] = GroupAdapter(
                len(spec["dynamic"]), len(spec["static_numeric"]), spec["static_category"],
                vocab_sizes.get(name, {}), d_s, category_dim)
            if spec["dynamic"] and use_gru:
                self.cells[name] = nn.GRUCell(d_s, d_s)

    def forward(self, batch, initial_memory=None):
        """batch 为 collate_samples 输出；返回 (states, memory) 两个 {group: tensor}。"""
        states, memory = {}, {}
        for spec in self.groups_spec:
            name = spec["name"]
            tensors = batch.inputs.groups[name]
            token = self.adapters[name](
                tensors["values"].float(), tensors["observed"], tensors["applicable"],
                tensors["static_numeric_history"].float(), tensors["static_numeric_mask"],
                tensors["static_category_history"])
            if spec["dynamic"] and self.use_gru:
                initial = None if initial_memory is None else initial_memory[name]
                token, last = self._run_gru(name, token, tensors, initial)
                states[name], memory[name] = token, last
            elif spec["dynamic"]:
                # no_entity_history 消融：逐时刻共享适配输出，不保留跨时记忆
                states[name] = token
                memory[name] = token[:, -1]
            else:
                states[name] = token
                memory[name] = token[:, -1]
        return states, memory

    def _run_gru(self, name, token, tensors, initial=None):
        """逐时刻共享 GRUCell；present=false／padding 的状态置零后继续。"""
        batch, length, nodes, width = token.shape
        cell = self.cells[name]
        if initial is None:
            hidden = torch.zeros(batch * nodes, width, device=token.device, dtype=token.dtype)
        else:
            if initial.shape != (batch, nodes, width):
                raise ValueError(f"{name}: entity memory shape 不匹配")
            hidden = initial.clone().reshape(batch * nodes, width)
        outputs = []
        present = tensors["present"] & ~tensors["padding"][:, None, :]
        for step in range(length):
            raw = hidden.reshape(batch, nodes, width)
            fresh = cell(token[:, step].reshape(batch * nodes, width), hidden).reshape(batch, nodes, width)
            alive = present[:, step].unsqueeze(-1)
            raw = torch.where(alive, fresh, torch.zeros_like(fresh))
            outputs.append(raw)
            hidden = raw.reshape(batch * nodes, width)
        return torch.stack(outputs, dim=1), outputs[-1]
