"""GSLAformer 在建模型：空间编码、时间链和双头自由递推。

实现范围（implementation.md §7）：`Model.encode_history` 组合实体历史编码、组内空间
交互、类型关系交互、分组读出与静态条件链，产出 `EncodedHistory` 边界对象。
total_only 不消费内部实体；no_spatial 使用 masked mean。完整训练验收仍在建。

不使用实体 ID 查表；同步重排整段历史、属性、mask 与关系后输出不变；新增实体不改变
参数形状（d_s/d_z 与实体数无关）。
"""
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from layers.gslaformer.entity_history import EntityHistoryEncoder
from layers.gslaformer.relations import RelationAttention
from layers.gslaformer.spatial import GroupReadout, GroupMeanReadout, GroupSpatialEncoder, StaticConditionEncoder, relation_degrees
from layers.gslaformer.temporal import (ConditionFusion, FuturePathEncoder, TemporalEncoder,
                                        calendar_features, lead_encoding)


@dataclass
class RolloutState:
    """当前状态窗口视图（implementation.md §7）：不含标签与 teacher memory。"""

    z: torch.Tensor                    # [B, L, Gd, d_z]
    y: torch.Tensor                    # [B, L, 1] 标准化总负荷
    y_observed: torch.Tensor           # [B, L]
    y_generated: torch.Tensor          # [B, L]
    coverage: torch.Tensor             # [B, L, Gd]
    group_generated: torch.Tensor      # [B, L, Gd]
    calendar: torch.Tensor             # [B, L, 4]
    condition: torch.Tensor            # [B, d_t] 静态条件 c
    entity_ids: dict = field(default_factory=dict)


@dataclass
class StepPrediction:
    """单步预测（implementation.md §7）。"""

    y: torch.Tensor                    # [B, 1] target 标准化坐标
    z: torch.Tensor                    # [B, Gd, d_z] 规范坐标


@dataclass
class ForecastResult:
    """rollout 输出（implementation.md §2）。"""

    y: torch.Tensor                    # [B, H, 1]
    z: torch.Tensor                    # [B, H, Gd, d_z]
    generated: torch.Tensor            # [B, H] 全部为 true
    metadata: dict = field(default_factory=dict)


@dataclass
class EncodedHistory:
    """编码历史边界对象（字段与 implementation.md §7 一致，首版固定）。"""

    z: torch.Tensor                    # [B, L, Gd, d_z] 规范 latent（仅动态组）
    observed_coverage: torch.Tensor    # [B, L, Gd] 逐动态组逐时刻起点可见覆盖率
    generated: torch.Tensor            # [B, L] 真实历史全 false
    c: torch.Tensor                    # [B, d_t] 独立静态条件
    entity_memory: dict                # {group: [B, N, d_s]} 按组末端实体 memory
    entity_ids: dict = field(default_factory=dict)  # 元数据，不进入数值计算


class Model(nn.Module):
    """组合空间链；variant 开关与 acceptance.md §5 消融定义一致。"""

    COMPUTATION_VERSION = 2

    def __init__(self, *, groups_spec, relation_types, model_params, vocab_sizes,
                 variant="full", known_future_dim=0, timezone="UTC"):
        super().__init__()
        # total_only 独立基线：构造时排除实体、关系及静态分支，不用零输入冒充消融。
        total_only = variant == "total_only"
        if total_only:
            groups_spec, relation_types = [], []
        self.groups_spec = groups_spec
        self.relation_types = relation_types
        self.variant = variant
        self._representation_frozen = False
        self.timezone = timezone
        d_s = model_params["spatial_dim"]
        d_z = model_params["latent_dim"]
        d_t = model_params["temporal_dim"]
        self.d_s, self.d_z, self.d_t = d_s, d_z, d_t
        self.lead_dim = model_params["lead_dim"]
        self.dynamic_groups = [spec["name"] for spec in groups_spec if spec["dynamic"]]
        self.use_gru = variant not in {"no_entity_history", "total_only"}
        self.use_spatial = variant not in {"no_spatial", "total_only"}
        self.use_relation = variant in {"full", "direct", "no_latent_loss",
                                        "no_entity_history"}
        self.use_relation_degrees = variant in {"full", "direct",
                                                "no_latent_loss", "no_entity_history",
                                                "no_relation_message"}
        if not total_only:
            self.history = EntityHistoryEncoder(groups_spec, vocab_sizes, d_s,
                                                model_params["category_embedding_dim"],
                                                use_gru=self.use_gru)
        if self.use_spatial:
            self.spatial = nn.ModuleDict({
                spec["name"]: GroupSpatialEncoder(d_s, model_params["spatial_heads"],
                                                  model_params["dropout"])
                for spec in groups_spec})
        if self.use_relation:
            self.relation = RelationAttention(relation_types, d_s, model_params["relation_heads"],
                                              model_params["relation_layers"],
                                              model_params["dropout"])
        self.readout = nn.ModuleDict({
            name: (GroupReadout(d_s, d_z, model_params["dropout"], heads=model_params["spatial_heads"])
                   if self.use_spatial else GroupMeanReadout(d_s, d_z, model_params["dropout"]))
            for name in self.dynamic_groups})
        if not total_only:
            self.static = StaticConditionEncoder(groups_spec, relation_types, vocab_sizes,
                                                 d_s, d_z, d_t, model_params["category_embedding_dim"])
        # ------------------------------------------------------------------ 时间链 ----
        self.temporal = TemporalEncoder(d_z, len(self.dynamic_groups), d_t,
                                        model_params["temporal_heads"],
                                        model_params["temporal_layers"], model_params["dropout"])
        self.future_path = FuturePathEncoder(known_future_dim, d_t,
                                             model_params["temporal_heads"],
                                             model_params["future_path_layers"],
                                             model_params["dropout"])
        self.fusion = ConditionFusion(d_t, self.lead_dim, model_params["dropout"])
        self.load_head = nn.Linear(d_t, 1)
        # latent 头在模型建立时初始化（§4.1）；A／direct 不用于 loss，B 阶段继承该初值
        self.has_latent = not total_only
        if self.has_latent:
            self.latent_heads = nn.ModuleDict(
                {name: nn.Linear(d_t, d_z) for name in self.dynamic_groups})

    # ------------------------------------------------------------------
    # 辅助：帧平铺与拼接轴
    # ------------------------------------------------------------------

    def _flatten_edges(self, batch):
        """relations_history [B][L] → [B*L]（b*L+t 顺序），与帧平铺一致。"""
        frames = []
        for sample in range(len(batch.inputs.site_ids)):
            frames.extend(batch.inputs.relations_history[sample])
        return frames

    def _group_coverage(self, name, tensors):
        """单动态组覆盖率 [B,L]：起点可见 observed／(present & ~padding & applicable)。"""
        eligible = ((tensors["present"] & ~tensors["padding"][:, None, :]).unsqueeze(-1)
                    & tensors["applicable"][:, None])
        seen = eligible & tensors["observed"]
        divisor = eligible.sum(dim=(2, 3)).clamp(min=1)
        return seen.sum(dim=(2, 3)).float() / divisor.float()

    # ------------------------------------------------------------------
    # 历史编码（I2 主入口）
    # ------------------------------------------------------------------

    def representation_modules(self):
        """冻结清单只含表示链；未来路径与系统时间链不在其中。"""
        return tuple(getattr(self, name) for name in
                     ("history", "spatial", "relation", "readout", "static")
                     if hasattr(self, name))

    def set_training_stage(self, stage):
        """配置 A/B 参数集合；不承担 epoch 课程或 checkpoint 状态转换。"""
        if stage not in {"A", "B"}:
            raise ValueError("训练阶段仅支持 A/B")
        self.requires_grad_(True)
        self._representation_frozen = stage == "B"
        if self._representation_frozen:
            for module in self.representation_modules():
                module.requires_grad_(False)
        elif self.has_latent:
            self.latent_heads.requires_grad_(False)
        # 切阶段不保留上一阶段的陈旧梯度。
        self.zero_grad(set_to_none=True)
        self.train(self.training)
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def train(self, mode=True):
        super().train(mode)
        if self._representation_frozen:
            for module in self.representation_modules():
                module.eval()
        return self

    def encode_history(self, batch, initial_memory=None):
        """生产调用从零编码；隔离 teacher 可显式传入 memory 副本延续。"""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self._representation_frozen):
            return self._encode_history(batch, initial_memory)

    def _encode_history(self, batch, initial_memory=None):
        """真实历史 → 规范 z／覆盖率／静态 c／末端实体 memory；不读监督对象。"""
        if self.variant == "total_only":
            target = batch.inputs.target_history
            b, length = target.shape[:2]
            return EncodedHistory(
                z=target.new_zeros(b, length, 0, self.d_z),
                observed_coverage=target.new_zeros(b, length, 0),
                generated=torch.zeros(b, length, dtype=torch.bool, device=target.device),
                c=target.new_zeros(b, self.d_t), entity_memory={}, entity_ids={})
        if initial_memory is None:
            states, memory = self.history(batch)
        else:
            states, memory = self.history(batch, initial_memory=initial_memory)
        batch_size = batch.inputs.target_history.shape[0]
        length = batch.inputs.target_history.shape[1]
        effective = {}
        for spec in self.groups_spec:
            name = spec["name"]
            tensors = batch.inputs.groups[name]
            effective[name] = tensors["present"] & ~tensors["padding"][:, None, :]
        # 组内注意力（逐组合并帧）
        if self.use_spatial:
            encoded = {}
            for spec in self.groups_spec:
                name = spec["name"]
                tokens = states[name]
                frames = tokens.shape[0] * tokens.shape[1]
                flat = tokens.reshape(frames, tokens.shape[2], tokens.shape[3])
                encoded[name] = self.spatial[name](flat, effective[name].reshape(
                    frames, -1)).reshape(tokens.shape)
        else:
            encoded = states
        # 拼接轴与关系交互
        if self.use_relation:
            offsets, pieces, masks = {}, [], []
            total = 0
            for spec in self.groups_spec:
                name = spec["name"]
                offsets[name] = total
                pieces.append(encoded[name])
                masks.append(effective[name])
                total += encoded[name].shape[2]
            tokens = torch.cat(pieces, dim=2)
            alive = torch.cat(masks, dim=2)
            frames = batch_size * length
            relation_out = self.relation(
                tokens.reshape(frames, total, self.d_s),
                self._flatten_edges(batch),
                alive.reshape(frames, total))
            relation_out = relation_out.reshape(batch_size, length, total, self.d_s)
            enhanced = {}
            for spec in self.groups_spec:
                name = spec["name"]
                start = offsets[name]
                enhanced[name] = relation_out[:, :, start:start + encoded[name].shape[2]]
        else:
            enhanced = encoded
        # 分组读出（仅动态组）
        latents, coverage = [], []
        for name in self.dynamic_groups:
            tensors = batch.inputs.groups[name]
            tokens = enhanced[name]
            frames = batch_size * length
            pooled = self.readout[name](
                tokens.reshape(frames, tokens.shape[2], tokens.shape[3]),
                effective[name].reshape(frames, -1),
                self._group_coverage(name, tensors).reshape(frames))
            latents.append(pooled.reshape(batch_size, length, self.d_z))
            coverage.append(self._group_coverage(name, tensors))
        z = torch.stack(latents, dim=2)
        coverage = torch.stack(coverage, dim=-1)
        # 静态条件链
        degrees = relation_degrees(batch, self.relation_types)
        if not self.use_relation_degrees:
            degrees = {name: torch.zeros_like(stats) for name, stats in degrees.items()}
        c = self.static(batch, degrees)
        return EncodedHistory(
            z=z,
            observed_coverage=coverage,
            generated=torch.zeros(batch_size, length, dtype=torch.bool, device=z.device),
            c=c,
            entity_memory=memory,
            entity_ids={name: tensors["entity_ids"]
                        for name, tensors in batch.inputs.groups.items()})

    # ------------------------------------------------------------------
    # 时间链与预测（I3 主入口）
    # ------------------------------------------------------------------

    def _window_tokens(self, z, y, y_observed, y_generated, coverage, group_generated, calendar):
        """拼接窗口 token：z 展平 + 总负荷 + 标记 + 逐组 coverage/generated + 日历。"""
        batch, length = z.shape[0], z.shape[1]
        if y_observed.dim() == 3:
            y_observed = y_observed.squeeze(-1)
        if y_generated.dim() == 3:
            y_generated = y_generated.squeeze(-1)
        if y.dim() == 2:
            y = y.unsqueeze(-1)
        return torch.cat([
            z.flatten(start_dim=2), y,
            y_observed.unsqueeze(-1).to(z.dtype),
            y_generated.unsqueeze(-1).to(z.dtype),
            coverage.to(z.dtype), group_generated.to(z.dtype),
            calendar.to(z.dtype)], dim=-1)

    def encode_future_path(self, batch):
        """origin 已知的整段未来路径一次编码（训练时缓存当前计算图 tensor，不 detach）。"""
        return self.future_path(batch.inputs.future_known.float(),
                                batch.inputs.future_known_mask,
                                self._calendar(batch.inputs, "future"))

    def _calendar(self, inputs, part):
        prepared = getattr(inputs, part + "_calendar", None)
        return (prepared if prepared is not None and inputs.calendar_timezone == self.timezone
                else calendar_features(getattr(inputs, part + "_time"), self.timezone))

    def _leads(self, horizon, batch_size, device):
        steps = torch.arange(1, horizon + 1, device=device).unsqueeze(0).expand(batch_size, -1)
        return lead_encoding(steps, self.lead_dim)

    def _heads(self, fused):
        """共享融合结果 → 总负荷头与各动态组 latent 头。"""
        y = self.load_head(fused)
        if self.has_latent:
            z = torch.stack([self.latent_heads[name](fused) for name in self.dynamic_groups],
                            dim=1)
        else:
            z = torch.zeros(fused.shape[0], 0, self.d_z, device=fused.device)
        return StepPrediction(y=y, z=z)

    def predict_direct(self, batch):
        """A／直接对照：完整关系空间编码 + 一次直接多步 H 输出，不回灌。"""
        history = self.encode_history(batch)
        path = self.encode_future_path(batch)
        calendar = self._calendar(batch.inputs, "history")
        tokens = self._window_tokens(
            history.z, batch.inputs.target_history.float(),
            batch.inputs.target_history_observed.squeeze(-1), history.generated,
            history.observed_coverage,
            history.generated.unsqueeze(-1).expand_as(history.observed_coverage), calendar)
        hidden = self.temporal(tokens)[:, -1]
        batch_size = history.z.shape[0]
        leads = self._leads(path.shape[1], batch_size, history.z.device)
        outputs = [self.load_head(self.fusion(hidden, history.c, path[:, step], leads[:, step]))
                   for step in range(path.shape[1])]
        return torch.stack(outputs, dim=1)

    def predict_step(self, state, path_context_k, origin_lead_k):
        """下一 y 与各组 z；state 不含标签或 teacher memory。"""
        tokens = self._window_tokens(state.z, state.y, state.y_observed, state.y_generated,
                                     state.coverage, state.group_generated, state.calendar)
        hidden = self.temporal(tokens)[:, -1]
        return self._heads(self.fusion(hidden, state.condition, path_context_k, origin_lead_k))

    def rollout(self, batch, horizon):
        """训练、valid、test、forecast 共用的唯一自由递推闭环（implementation.md §2）。"""
        history = self.encode_history(batch)
        return self._rollout_from_history(batch, horizon, history)

    def _rollout_from_history(self, batch, horizon, history):
        """同 batch 冻结历史可供 teacher 复用；未来路径仍完整 H 且保留梯度。"""
        path = self.encode_future_path(batch)
        batch_size, length = history.z.shape[0], history.z.shape[1]
        total = length + horizon
        device, dtype = history.z.device, history.z.dtype
        d_gd = len(self.dynamic_groups)
        # 历史 z 只计算一次；窗口移位不重跑实体编码器
        z_series = torch.zeros(batch_size, total, d_gd, self.d_z, device=device, dtype=dtype)
        z_series[:, :length] = history.z
        y_series = torch.zeros(batch_size, total, 1, device=device, dtype=dtype)
        y_series[:, :length] = batch.inputs.target_history.float()
        observed = torch.zeros(batch_size, total, dtype=torch.bool, device=device)
        observed[:, :length] = batch.inputs.target_history_observed.squeeze(-1)
        generated = torch.zeros(batch_size, total, dtype=torch.bool, device=device)
        coverage = torch.zeros(batch_size, total, d_gd, device=device, dtype=dtype)
        coverage[:, :length] = history.observed_coverage
        group_generated = torch.zeros(batch_size, total, d_gd, device=device, dtype=dtype)
        calendar_all = torch.cat([self._calendar(batch.inputs, "history"), self._calendar(batch.inputs, "future")], dim=1)
        leads = self._leads(horizon, batch_size, device)
        predictions = []
        for step in range(horizon):
            window = slice(step, step + length)
            tokens = self._window_tokens(
                z_series[:, window], y_series[:, window], observed[:, window],
                generated[:, window], coverage[:, window], group_generated[:, window],
                calendar_all[:, window])
            hidden = self.temporal(tokens)[:, -1]
            fused = self.fusion(hidden, history.c, path[:, step], leads[:, step])
            step_prediction = self._heads(fused)
            index = length + step
            # 追加预测 z 与同一预测 y；生成值不是缺失值，不因 observed=false 被清零
            z_series[:, index] = step_prediction.z
            y_series[:, index] = step_prediction.y
            observed[:, index] = False
            generated[:, index] = True
            coverage[:, index] = 0.0
            group_generated[:, index] = 1.0
            predictions.append(step_prediction)
        return ForecastResult(
            y=torch.stack([item.y for item in predictions], dim=1),
            z=torch.stack([item.z for item in predictions], dim=1),
            generated=torch.ones(batch_size, horizon, dtype=torch.bool, device=device),
            metadata={"origins": batch.inputs.origins.detach().cpu(),
                      "horizon": horizon, "site_ids": list(batch.inputs.site_ids)})

    def forward(self, batch, horizon=None, mode="rollout"):
        """模式分派：rollout（含 A/B/AR）与 direct 对照；不改变计算定义。"""
        if mode == "direct":
            return self.predict_direct(batch)
        if horizon is None:
            horizon = batch.inputs.future_time.shape[1]
        return self.rollout(batch, horizon)
