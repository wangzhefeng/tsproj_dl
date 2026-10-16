"""组内摘要注意力、分组读出与独立静态条件链（implementation.md §1.2–1.3）。"""
import torch
import torch.nn as nn


class GroupSpatialEncoder(nn.Module):
    """组内 TransformerEncoderLayer 加摘要 token；摘要始终有效，无效实体不贡献信息。"""

    def __init__(self, d_s, heads, dropout=0.0):
        super().__init__()
        self.summary = nn.Parameter(torch.zeros(1, 1, d_s))
        nn.init.normal_(self.summary, std=0.02)
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_s, nhead=heads, dim_feedforward=4 * d_s, activation="gelu",
            dropout=dropout, batch_first=True)

    def forward(self, tokens, effective):
        """tokens [F, N, d_s]；effective [F, N] bool。返回 [F, N, d_s]（真实实体输出）。"""
        frames, nodes, width = tokens.shape
        summary = self.summary.expand(frames, 1, width)
        source = torch.cat([summary, tokens], dim=1)
        key_padding = torch.cat(
            [torch.zeros(frames, 1, dtype=torch.bool, device=tokens.device), ~effective], dim=1)
        output = self.layer(source, src_key_padding_mask=key_padding)
        return output[:, 1:] * effective.unsqueeze(-1).to(output.dtype)


class GroupReadout(nn.Module):
    """关系增强实体的查询池化；补可学习空组 token 作为始终有效的 key。"""

    def __init__(self, d_s, d_z, dropout=0.0, heads=4):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, d_s))
        nn.init.normal_(self.query, std=0.02)
        self.empty_token = nn.Parameter(torch.zeros(1, 1, d_s))
        nn.init.normal_(self.empty_token, std=0.02)
        self.attention = nn.MultiheadAttention(d_s, heads, dropout=dropout, batch_first=True)
        self.scale = nn.Linear(d_s + 2, d_z)
        self.norm = nn.LayerNorm(d_z)

    def forward(self, tokens, effective, coverage):
        """tokens [F, N, d_s]；effective [F,N]；coverage [F]（组级当前动态观测覆盖率）。"""
        frames = tokens.shape[0]
        keys = torch.cat([self.empty_token.expand(frames, 1, -1), tokens], dim=1)
        padding = torch.cat([torch.zeros(frames, 1, dtype=torch.bool, device=tokens.device),
                             ~effective], dim=1)
        query = self.query.expand(frames, 1, -1)
        pooled, _ = self.attention(query, keys, keys, key_padding_mask=padding)
        counts = effective.sum(dim=1, keepdim=True).to(tokens.dtype)
        evidence = torch.cat([pooled.squeeze(1), torch.log1p(counts),
                              coverage.unsqueeze(1).to(tokens.dtype)], dim=-1)
        return self.norm(self.scale(evidence))


class GroupMeanReadout(nn.Module):
    """no_spatial 对照：masked mean，不建立实体间注意力。"""

    def __init__(self, d_s, d_z, dropout=0.0):
        super().__init__()
        self.scale = nn.Linear(d_s + 2, d_z)
        self.norm = nn.LayerNorm(d_z)

    def forward(self, tokens, effective, coverage):
        counts = effective.sum(dim=1, keepdim=True).to(tokens.dtype)
        masked = torch.where(effective.unsqueeze(-1), tokens, torch.zeros_like(tokens))
        pooled = masked.sum(dim=1) / counts.clamp(min=1)
        evidence = torch.cat([pooled, torch.log1p(counts), coverage.unsqueeze(1)], dim=-1)
        return self.norm(self.scale(evidence))


class StaticConditionEncoder(nn.Module):
    """独立静态链：只读静态属性、mask、逐类型与方向的度数与起点有效实体集合（§1.3）。"""

    def __init__(self, groups_spec, relation_types, vocab_sizes, d_s, d_z, d_t, category_dim):
        super().__init__()
        self.groups_spec = groups_spec
        self.relation_types = relation_types
        self.per_group = nn.ModuleDict()
        for spec in groups_spec:
            numeric = len(spec["static_numeric"])
            # masked mean、mask 可用率、masked sum（各 numeric 维）+ 有效数量 + log1p(节点数)
            width = (numeric * 3
                     + len(spec["static_category"]) * category_dim
                     + 2 * len(relation_types)            # 逐类型逐方向平均度数
                     + 2)
            self.per_group[spec["name"]] = nn.Sequential(
                nn.Linear(width, d_z), nn.GELU())
        self.embeddings = nn.ModuleDict()
        for spec in groups_spec:
            self.embeddings[spec["name"]] = nn.ModuleList(
                nn.Embedding(vocab_sizes.get(spec["name"], {}).get(name, 2), category_dim)
                for name in spec["static_category"])
        self.project = nn.Linear(len(groups_spec) * d_z, d_t)

    def _group_features(self, name, spec, tensors, degrees, active, endpoint, category_dim):
        """单组静态特征：masked mean／sum／数量、类别 embedding mean、平均度数。"""
        batch = active.shape[0]
        numeric = tensors["static_numeric_history"][:, -1].float()
        mask = tensors["static_numeric_mask"][:, -1].float() * active.unsqueeze(-1).float()
        node_counts = active.sum(dim=1, keepdim=True).float()
        counts = node_counts.clamp(min=1)
        masked_numeric = torch.where(mask.bool(), numeric, torch.zeros_like(numeric))
        total = masked_numeric.sum(dim=1)
        mean = total / mask.sum(dim=1).clamp(min=1)
        pieces = [mean, mask.sum(dim=1) / counts]
        for index, embedding in enumerate(self.embeddings[name]):
            token = embedding(tensors["static_category_history"][:, -1, :, index])
            pooled = (token * active.unsqueeze(-1).float()).sum(dim=1) / counts
            pieces.append(pooled)
        pieces.append(degrees)
        pieces.append(total)
        pieces.append(active.sum(dim=1, keepdim=True).float())
        pieces.append(torch.log1p(node_counts))
        return torch.cat(pieces, dim=-1)

    def forward(self, batch, relation_stats):
        """relation_stats：{group: [B, 2*T] 平均度数}（方向 0=出、1=入）。"""
        outputs = []
        for spec in self.groups_spec:
            name = spec["name"]
            tensors = batch.inputs.groups[name]
            active = tensors["present"][:, -1] & ~tensors["padding"]
            features = self._group_features(name, spec, tensors, relation_stats[name], active,
                                            None, None)
            outputs.append(self.per_group[name](features))
        return self.project(torch.cat(outputs, dim=-1))


def relation_degrees(batch, relation_types):
    """从 origin 边表统计每组的逐类型逐方向平均度数（供静态链与消融使用）。

    返回 {group: [B, 2*T] float}；方向 0=出度（src 在该组）、1=入度（dst 在该组）。
    """
    device = batch.inputs.origins.device
    offsets, total = {}, 0
    for group, tensors in batch.inputs.groups.items():
        offsets[group] = total
        total += tensors["padding"].shape[1]
    type_count = len(relation_types)
    node_groups = [group for group, tensors in batch.inputs.groups.items() for _ in range(tensors["padding"].shape[1])]
    stats = {group: torch.zeros(len(batch.inputs.site_ids), 2 * type_count, device=device)
             for group in batch.inputs.groups}
    for sample, edges in enumerate(batch.inputs.relations_origin):
        members = {group: int((~tensors["padding"][sample] & tensors["present"][sample, -1]).sum())
                   for group, tensors in batch.inputs.groups.items()}
        counters = {group: [[0] * type_count, [0] * type_count] for group in stats}
        for src, dst, kind in edges:
            counters[node_groups[src]][0][kind] += 1
            counters[node_groups[dst]][1][kind] += 1
        for group, counters in counters.items():
            divisor = max(members[group], 1)
            stats[group][sample] = torch.tensor(
                [value / divisor for pair in counters for value in pair],
                dtype=torch.float32, device=device)
    return stats
