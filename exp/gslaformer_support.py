"""GSLAformer 训练支持：隔离 teacher、非学习质量与精确损失 reduction。

契约见 implementation.md §3；不接受未来标签作为生产 rollout 输入。
"""
from copy import copy, deepcopy
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import uuid
import warnings

import numpy as np
import torch

from data_provider.spatial_temporal.batching import Batch
from data_provider.spatial_temporal.schema import canonical_hash, read_json


@dataclass
class LatentLabels:
    z: torch.Tensor
    quality: torch.Tensor


@torch.no_grad()
def group_quality(inputs, groups_spec, relation_types, *, use_relations=True):
    """最近窗口的逐组 r 与 origin 通信图两跳 q；返回 [B,Gd]。"""
    dynamic = [g["name"] for g in groups_spec if g["dynamic"]]
    counts, ratios, active = {}, {}, {}
    offset = 0
    node_group, active_nodes = {}, []
    for spec in groups_spec:
        name = spec["name"]
        group = inputs.groups[name]
        alive = group["present"] & ~group["padding"][:, None]
        active[name] = alive[:, -1]
        nodes = alive.shape[-1]
        node_group.update({offset + i: name for i in range(nodes)})
        active_nodes.append(active[name])
        offset += nodes
        if name in dynamic:
            eligible = alive.unsqueeze(-1) & group["applicable"][:, None]
            counts[name] = eligible.sum(dim=(1, 2, 3))
            ratios[name] = (group["observed"] & eligible).sum(dim=(1, 2, 3)).float()
            ratios[name] /= counts[name].clamp(min=1)
    b = inputs.target_history.shape[0]
    q = inputs.target_history.new_zeros(b, len(dynamic))
    if not dynamic:
        return q
    all_active = torch.cat(active_nodes, dim=1)
    for sample in range(b):
        adjacency = {}
        if use_relations:
            for src, dst, kind in inputs.relations_origin[sample]:
                if src == dst or not (all_active[sample, src] and all_active[sample, dst]):
                    continue
                adjacency.setdefault(src, set()).add(dst)
                if relation_types[kind]["reverse_message"]:
                    adjacency.setdefault(dst, set()).add(src)
        for index, name in enumerate(dynamic):
            if not active[name][sample].any():
                continue
            frontier = {node for node, group in node_group.items()
                        if group == name and all_active[sample, node]}
            reached = set()
            for _ in range(2):
                frontier = {dst for src in frontier for dst in adjacency.get(src, ())}
                reached.update(frontier)
            neighbors = sorted({node_group[n] for n in reached} & set(dynamic) - {name})
            neighbor_r = torch.stack([ratios[n][sample] for n in neighbors]).mean() if neighbors else 1.0
            q[sample, index] = ratios[name][sample] * (0.5 + 0.5 * neighbor_r)
    return q


def _teacher_step_inputs(inputs, supervision, step):
    """单步 teacher 只用 origin 上下文；所有可写 tensor 均与生产输入隔离。"""
    result = copy(inputs)
    result.groups = {}
    for name, group in inputs.groups.items():
        current = {}
        for key, value in group.items():
            if torch.is_tensor(value):
                temporal = key in {"values", "observed", "present", "static_numeric_history",
                                   "static_numeric_mask", "static_category_history"}
                current[key] = (value[:, -1:] if temporal else value).clone()
            else:
                current[key] = deepcopy(value)
        future = supervision.future_groups[name]
        mask = (future["observed"][:, step:step + 1]
                & current["present"].unsqueeze(-1)
                & ~current["padding"][:, None, :, None]
                & current["applicable"][:, None])
        current["observed"] = mask
        current["values"] = torch.where(mask, future["values"][:, step:step + 1], 0).clone()
        result.groups[name] = current
    result.relations_origin = deepcopy(inputs.relations_origin)
    result.relations_history = [[edges] for edges in result.relations_origin]
    result.history_time = inputs.future_time[:, step:step + 1].clone()
    # teacher 空间编码不需要负荷标签；明确放零，禁止从 supervision.target 读取。
    result.target_history = torch.zeros_like(inputs.target_history[:, -1:])
    result.target_history_observed = torch.zeros_like(inputs.target_history_observed[:, -1:])
    return result


@torch.no_grad()
def build_latent_labels(model, inputs, supervision, encoded_history=None, *, steps=None):
    """从同起点历史末端 memory 副本延续 teacher；不写回原输入、memory 或模型参数。"""
    if not model._representation_frozen:
        raise ValueError("teacher 必须使用冻结的 B 阶段表示链")
    if not model.has_latent:
        raise ValueError("total_only 不构造 latent 标签")
    if supervision is None:
        raise ValueError("teacher 缺少独立监督")
    steps = inputs.future_time.shape[1] if steps is None else steps
    if type(steps) is not int or not 1 <= steps <= inputs.future_time.shape[1]:
        raise ValueError("teacher steps 必须在 1..H 内")
    history = encoded_history if encoded_history is not None else model.encode_history(Batch(inputs, None))
    memory = {name: value.detach().clone() for name, value in history.entity_memory.items()}
    length = inputs.target_history.shape[1]
    rolling = copy(inputs)
    rolling.groups = {name: dict(group) for name, group in inputs.groups.items()}
    labels, qualities = [], []
    for step in range(steps):
        teacher = _teacher_step_inputs(inputs, supervision, step)
        encoded = model.encode_history(Batch(teacher, None), initial_memory=memory)
        memory = encoded.entity_memory
        labels.append(encoded.z[:, 0])
        for name, group in teacher.groups.items():
            for key in ("observed", "present"):
                rolling.groups[name][key] = torch.cat(
                    [rolling.groups[name][key], group[key]], dim=1)[:, -length:]
        qualities.append(group_quality(rolling, model.groups_spec, model.relation_types,
                                       use_relations=model.use_relation_degrees))
    return LatentLabels(torch.stack(labels, dim=1), torch.stack(qualities, dim=1))


def masked_load_loss(prediction, target, mask):
    """每 horizon 在有效样本上平均，再对有效 horizon 等权；无主目标返回 None。"""
    if prediction.shape != target.shape or target.shape != mask.shape or prediction.shape[-1] != 1:
        raise ValueError("主目标 shape 不一致")
    if mask.dtype != torch.bool:
        raise ValueError("主目标 mask 必须为 bool")
    counts = mask.sum(dim=(0, 2))
    valid = counts > 0
    if not valid.any():
        return None, counts
    if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
        raise ValueError("有效主目标或预测存在 NaN/Inf")
    difference = torch.where(mask, prediction, 0) - torch.where(mask, target, 0)
    losses = difference.square().sum(dim=(0, 2)) / counts.clamp(min=1)
    return losses[valid].mean(), counts


def latent_dimension_weights(variance):
    if not torch.isfinite(variance).all() or (variance < 0).any():
        raise ValueError("latent 方差必须有限且非负")
    return (variance + 0.01).reciprocal().clamp(max=100)


def weighted_latent_loss(prediction, labels, weights):
    """sum(q*error)/count(q>0)，然后有效 horizon／组等权，不除 sum(q)。"""
    q = labels.quality
    if (prediction.shape != labels.z.shape or q.shape != prediction.shape[:-1]
            or weights.shape != prediction.shape[-2:]):
        raise ValueError("latent 标签／权重 shape 不一致")
    if not torch.isfinite(q).all() or (q < 0).any() or (q > 1).any():
        raise ValueError("q 必须有限且位于 [0,1]")
    if not torch.isfinite(weights).all() or (weights <= 0).any() or (weights > 100).any():
        raise ValueError("latent 维度权重非法")
    valid = q > 0
    if not torch.isfinite(prediction[valid]).all() or not torch.isfinite(labels.z[valid]).all():
        raise ValueError("有效 latent 存在 NaN/Inf")
    difference = (torch.where(valid.unsqueeze(-1), prediction, 0)
                  - torch.where(valid.unsqueeze(-1), labels.z, 0))
    error = (difference.square() * weights).mean(dim=-1)
    counts = valid.sum(dim=0)  # [H,Gd]
    per_horizon = (q * error).sum(dim=0) / counts.clamp(min=1)
    valid_horizon = counts > 0
    group_counts = valid_horizon.sum(dim=0)
    per_group = per_horizon.sum(dim=0) / group_counts.clamp(min=1)
    loss = per_group[group_counts > 0].mean() if (group_counts > 0).any() else difference.sum() * 0
    return loss, counts


class ValidationMSE:
    """全 split 累积 SSE/count；先 horizon 等权，再按配置决定是否站点等权。"""

    def __init__(self, horizon, *, expected_sites=None):
        if type(horizon) is not int or horizon < 1:
            raise ValueError("horizon 必须为正整数")
        self.horizon = horizon
        self.expected_sites = None if expected_sites is None else set(expected_sites)
        if self.expected_sites == set():
            raise ValueError("expected_sites 不能为空")
        self.by_site = {}

    @torch.no_grad()
    def update(self, prediction, target, mask, site_ids):
        if (prediction.shape != target.shape or prediction.shape != mask.shape
                or prediction.shape[1:] != (self.horizon, 1)
                or len(site_ids) != prediction.shape[0] or mask.dtype != torch.bool):
            raise ValueError("验证指标输入不符合 [B,H,1] 合同")
        if not torch.isfinite(prediction[mask]).all() or not torch.isfinite(target[mask]).all():
            raise ValueError("验证有效单元含 NaN/Inf")
        residual = (torch.where(mask, prediction, 0).detach().cpu().double()
                    - torch.where(mask, target, 0).detach().cpu().double()).numpy()[..., 0]
        valid = mask.detach().cpu().numpy()[..., 0]
        for index, site in enumerate(site_ids):
            if self.expected_sites is not None and site not in self.expected_sites:
                raise ValueError(f"未声明验证站点: {site}")
            sse, counts = self.by_site.setdefault(
                site, (np.zeros(self.horizon, np.float64), np.zeros(self.horizon, np.int64)))
            sse += residual[index] ** 2
            counts += valid[index]

    def compute(self):
        if not self.by_site:
            raise ValueError("验证 split 没有样本")
        if self.expected_sites is not None:
            missing = self.expected_sites - self.by_site.keys()
            if missing:
                raise ValueError(f"缺少验证站点: {sorted(missing)}")
            scores = []
            for site in sorted(self.expected_sites):
                sse, counts = self.by_site[site]
                if (counts == 0).any():
                    raise ValueError(f"{site}: horizon 缺少有效验证标签")
                scores.append(float((sse / counts).mean()))
            return float(np.mean(scores))
        sse = sum(values[0] for values in self.by_site.values())
        counts = sum(values[1] for values in self.by_site.values())
        if (counts == 0).any():
            raise ValueError("horizon 缺少有效验证标签")
        return float((sse / counts).mean())


class LatentStatistics:
    """训练唯一 origin 的 q 加权 Welford 总体统计，不收集窗口内部或未来 teacher。"""

    def __init__(self, groups, latent_dim, *, schema_fingerprint, data_fingerprint,
                 representation_fingerprint):
        if not groups or len(set(groups)) != len(groups) or type(latent_dim) is not int or latent_dim < 1:
            raise ValueError("latent groups/dim 非法")
        self.groups = list(groups)
        self.dim = latent_dim
        self.context = dict(schema_fingerprint=schema_fingerprint,
                            data_fingerprint=data_fingerprint,
                            representation_fingerprint=representation_fingerprint)
        if any(not isinstance(v, str) or not v for v in self.context.values()):
            raise ValueError("统计必须绑定非空指纹")
        self._seen = {}
        shape = (len(groups), latent_dim)
        self._mean = np.zeros(shape, np.float64)
        self._m2 = np.zeros(shape, np.float64)
        self._weight = np.zeros(len(groups), np.float64)
        self._count = np.zeros(len(groups), np.int64)

    @property
    def origin_count(self):
        return len(self._seen)

    def add(self, episode, origin, last_z, quality, *, request_fingerprint, split):
        if split != "train":
            raise ValueError("latent 统计仅接受 train")
        if (not isinstance(episode, str) or not episode or type(origin) is not int
                or not isinstance(request_fingerprint, str) or not request_fingerprint):
            raise ValueError("origin 溯源字段非法")
        z, q = np.asarray(last_z, np.float64), np.asarray(quality, np.float64)
        if z.shape != self._mean.shape or q.shape != self._weight.shape:
            raise ValueError("只接受最后一个历史 z；shape 应为 [Gd,d_z]/[Gd]")
        if not np.isfinite(z).all() or not np.isfinite(q).all() or ((q < 0) | (q > 1)).any():
            raise ValueError("latent/q 非有限或越界")
        record = {"episode": episode, "origin": origin,
                  "request_fingerprint": request_fingerprint,
                  "content_fingerprint": canonical_hash({"z": z.tolist(), "q": q.tolist()})}
        key = (episode, origin)
        if key in self._seen:
            if self._seen[key] != record:
                raise ValueError("重复 episode/origin 内容或数据版本冲突")
            return
        self._seen[key] = record
        valid = q > 0
        total = self._weight + q
        delta = z - self._mean
        ratio = np.divide(q, total, out=np.zeros_like(q), where=total > 0)
        mean = self._mean + ratio[:, None] * delta
        self._m2 += q[:, None] * delta * (z - mean)
        self._mean = mean
        self._weight = total
        self._count += valid

    def finalize(self):
        if (self._count < 2).any():
            raise ValueError("每组至少需要两个 q>0 的训练 origin")
        variance = np.maximum(self._m2 / self._weight[:, None], 0)
        degenerate = (variance < 1e-8).all(axis=1)
        if degenerate.all():
            raise ValueError("representation_degenerate: 全部动态组退化")
        result = {}
        for i, name in enumerate(self.groups):
            if degenerate[i]:
                warnings.warn(f"latent 组 {name} 近常量，保留有界加权目标", RuntimeWarning)
            result[f"{name}__mean"] = self._mean[i].copy()
            result[f"{name}__variance"] = variance[i].copy()
            result[f"{name}__count"] = np.array(self._count[i], np.int64)
        return result

    def save(self, directory):
        import json
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        path, manifest_path = root / "latent_stats.npz", root / "latent_stats_manifest.json"
        if path.exists() or manifest_path.exists():
            raise FileExistsError("不覆盖已有 latent 统计")
        arrays = self.finalize()
        temporary = root / f"latent_stats.{uuid.uuid4().hex}.tmp"
        with temporary.open("xb") as stream:
            np.savez(stream, **arrays)
        checksum = hashlib.sha256(temporary.read_bytes()).hexdigest()
        manifest = {"version": 1, "groups": self.groups, "latent_dim": self.dim,
                    **self.context, "origins": [self._seen[key] for key in sorted(self._seen)],
                    "npz_sha256": checksum,
                    "statistics_fingerprint": canonical_hash({k: v.tolist() for k, v in arrays.items()})}
        manifest["fingerprint"] = canonical_hash(manifest)
        os.replace(temporary, path)
        temporary = root / f"latent_stats_manifest.{uuid.uuid4().hex}.tmp"
        temporary.write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, manifest_path)

    @staticmethod
    def load(directory, *, schema_fingerprint, data_fingerprint, representation_fingerprint):
        root = Path(directory)
        manifest = read_json(root / "latent_stats_manifest.json")
        expected = manifest.pop("fingerprint", None)
        if manifest.get("version") != 1 or expected != canonical_hash(manifest):
            raise ValueError("latent manifest 指纹校验失败")
        for name, value in dict(schema_fingerprint=schema_fingerprint, data_fingerprint=data_fingerprint,
                                representation_fingerprint=representation_fingerprint).items():
            if manifest.get(name) != value:
                raise ValueError(f"latent {name} 指纹不匹配")
        path = root / "latent_stats.npz"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["npz_sha256"]:
            raise ValueError("latent NPZ 字节校验失败")
        groups, dim = manifest["groups"], manifest["latent_dim"]
        expected_keys = {f"{g}__{field}" for g in groups for field in ("mean", "variance", "count")}
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != expected_keys:
                raise ValueError("latent NPZ 字段不完整")
            result = {key: archive[key].copy() for key in expected_keys}
        for group in groups:
            for field in ("mean", "variance"):
                value = result[f"{group}__{field}"]
                if value.shape != (dim,) or value.dtype != np.float64 or not np.isfinite(value).all():
                    raise ValueError("latent 数值 shape/dtype 非法")
            count = result[f"{group}__count"]
            if count.shape != () or count.dtype != np.int64 or count < 2:
                raise ValueError("latent origin count 非法")
            if (result[f"{group}__variance"] < 0).any():
                raise ValueError("latent variance 不能为负")
        return result


def frozen_representation_fingerprint(model):
    """仅散列冻结表示链的权重／buffer，预测链更新不改变 latent 坐标指纹。"""
    if not model._representation_frozen:
        raise ValueError("统计前必须冻结表示链")
    digest = hashlib.sha256()
    for index, module in enumerate(model.representation_modules()):
        for name, value in sorted(module.state_dict().items()):
            tensor = value.detach().cpu().contiguous()
            digest.update(canonical_hash({"module": index, "name": name,
                                          "shape": list(tensor.shape), "dtype": str(tensor.dtype)}).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@torch.no_grad()
def collect_latent_statistics(model, batches, *, schema_fingerprint, data_fingerprint):
    """只从已冻结模型的训练输入收集最后 z；调用方提供完整非抽样训练窗口集。"""
    fingerprint = frozen_representation_fingerprint(model)
    collector = LatentStatistics(model.dynamic_groups, model.d_z,
                                 schema_fingerprint=schema_fingerprint,
                                 data_fingerprint=data_fingerprint,
                                 representation_fingerprint=fingerprint)
    for batch in batches:
        metadata = batch.inputs.metadata
        if any(row.get("split") != "train" for row in metadata):
            raise ValueError("latent 统计仅接受 train batch")
        if any(row.get("schema_fingerprint") != schema_fingerprint for row in metadata):
            raise ValueError("训练 batch schema 指纹不一致")
        encoded = model.encode_history(batch)
        quality = group_quality(batch.inputs, model.groups_spec, model.relation_types,
                                use_relations=model.use_relation_degrees)
        for i, row in enumerate(metadata):
            collector.add(row["episode_id"], int(batch.inputs.origins[i].item()),
                          encoded.z[i, -1].cpu().numpy(), quality[i].cpu().numpy(),
                          request_fingerprint=row["data_fingerprint"], split=row["split"])
    if frozen_representation_fingerprint(model) != fingerprint:
        raise ValueError("收集过程中冻结表示链发生变化")
    return collector


@torch.no_grad()
def diagnostic_rollout(model, batch, kind):
    """同权重回灌诊断，独立于生产 rollout；teacher_z 使用未来信息，不用于排名或选模。"""
    from models.transformer.GSLAformer import RolloutState
    from layers.gslaformer.temporal import calendar_features
    if kind not in {"fixed_z", "fixed_y", "teacher_z"} or not model.has_latent:
        raise ValueError("不支持的 latent 诊断")
    if model.training:
        raise ValueError("诊断必须在 eval 模式执行")
    history = model.encode_history(batch)
    path = model.encode_future_path(batch)
    horizon = batch.inputs.future_time.shape[1]
    leads = model._leads(horizon, history.z.shape[0], history.z.device)
    teacher = build_latent_labels(model, batch.inputs, batch.supervision, history) if kind == "teacher_z" else None
    observed = batch.inputs.target_history_observed.squeeze(-1)
    last_y = None
    if kind == "fixed_y":
        if not observed.any(dim=1).all():
            raise ValueError("fixed_y 诊断缺少有效历史总负荷")
        indices = torch.arange(observed.shape[1], device=observed.device).expand_as(observed)
        last = torch.where(observed, indices, -1).max(dim=1).values
        last_y = batch.inputs.target_history[torch.arange(observed.shape[0], device=observed.device), last]
    future_calendar = calendar_features(batch.inputs.future_time, model.timezone)
    state = RolloutState(history.z, batch.inputs.target_history, observed, history.generated,
                         history.observed_coverage, torch.zeros_like(history.observed_coverage),
                         calendar_features(batch.inputs.history_time, model.timezone), history.c)
    outputs = []
    for step in range(horizon):
        prediction = model.predict_step(state, path[:, step], leads[:, step])
        outputs.append(prediction.y)
        z = history.z[:, -1] if kind == "fixed_z" else teacher.z[:, step] if teacher is not None else prediction.z
        y = last_y if kind == "fixed_y" else prediction.y
        state = RolloutState(
            torch.cat([state.z[:, 1:], z[:, None]], 1),
            torch.cat([state.y[:, 1:], y[:, None]], 1),
            torch.cat([state.y_observed[:, 1:], torch.zeros_like(observed[:, :1])], 1),
            torch.cat([state.y_generated[:, 1:], torch.ones_like(observed[:, :1])], 1),
            torch.cat([state.coverage[:, 1:], torch.zeros_like(history.observed_coverage[:, :1])], 1),
            torch.cat([state.group_generated[:, 1:], torch.ones_like(history.observed_coverage[:, :1])], 1),
            torch.cat([state.calendar[:, 1:], future_calendar[:, step:step + 1]], 1), history.c)
    return torch.stack(outputs, 1)
