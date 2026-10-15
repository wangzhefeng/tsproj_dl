"""组级训练统计、类别词表与标准化变换；fit 强制训练分区与 as-of 可见性。

统计语义（input_output.md §6.4/§6.7）：
- 动态观测按 (group, feature) 在训练窗口覆盖的唯一 (t, n) 单元上统计，重叠窗口只计一次；
- 可见性以「单元所在窗口的监督末端时刻」为界：离线训练时 history 与标签数据都已到达，
  永不到达（available_at 超过一切窗口末端）的单元不参与统计；
- 静态数值与类别按训练可见版本记录去重，不按时间格广播重复拟合；
- target 使用独立统计；协变量只使用训练 origin 可见的记录。
"""
from copy import deepcopy
import math

import numpy as np

from .schema import canonical_hash

# ------------------------------------------------------------------------------
# 常量与基础工具
# ------------------------------------------------------------------------------

VERSION = 1
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
_UNREACHED = np.int64(-1)


def _feature_stats(cells):
    """单一 (group, feature) 的总体统计；无观测的已声明特征直接报错。"""
    values = np.concatenate(cells).astype(np.float64) if cells else np.empty(0, np.float64)
    if not len(values):
        raise ValueError("已声明数值特征完全没有训练观测")
    mean = float(values.mean())
    variance = float(((values - mean) ** 2).mean())
    std = math.sqrt(variance)
    if std < 1e-6:
        std = 1.0
    return {"mean": mean, "std": std, "count": int(len(values))}


def _role_stats(cells_by_feature):
    """按 schema 特征顺序合并为并行 list；零特征角色返回空 list。"""
    merged = {"mean": [], "std": [], "count": []}
    for cells in cells_by_feature:
        stats = _feature_stats(cells)
        for key in merged:
            merged[key].append(stats[key])
    return merged


def _standardize(values, mask, stats):
    """逐特征标准化并按 mask 置零；未观测位置不残留 NaN。"""
    out = np.zeros(values.shape, np.float32)
    for index in range(values.shape[-1]):
        column = values[..., index]
        out[..., index] = np.where(mask[..., index],
                                   (column - stats["mean"][index]) / stats["std"][index], 0.0)
    return out


def _tokenize(column, index, unknown):
    """单类别列文本转 token；缺失映射 UNK，未知类别去重记录并保持 UNK。"""
    token = np.ones(column.shape, np.int64)
    for position, value in np.ndenumerate(column):
        if value is None:
            continue
        code = index.get(value)
        if code is None:
            unknown.add(value)
            continue
        token[position] = code
    return token


# ------------------------------------------------------------------------------
# 训练变换
# ------------------------------------------------------------------------------

class TrainingTransform:
    """fit 只接受 train split；统计单元在重叠窗口间只计一次；apply 幂等不改变状态。"""

    def __init__(self, state):
        self.state = state
        self.fingerprint = canonical_hash(state)
        self._vocab_index = {}
        for group, features in state["vocab"].items():
            self._vocab_index[group] = {}
            for feature, tokens in features.items():
                self._vocab_index[group][feature] = {token: i for i, token in enumerate(tokens)}

    # ---------------------------------------------------------------- fit ----

    @classmethod
    def fit(cls, dataset):
        if dataset.split != "train":
            raise ValueError("TrainingTransform.fit 只接受 train split 的 PackageDataset")
        config = dataset.config
        data = config["data"]
        length, horizon = data["seq_len"], data["pred_len"]
        schema = dataset.episodes[0].schema
        # 每 episode 的训练 origin 集合
        per_episode = {}
        for episode_index, origin in dataset.windows:
            per_episode.setdefault(episode_index, set()).add(origin)
        target_cells, known_cells, static_cells = [], {}, {}
        dynamic_cells, vocab, scope = {}, {}, {}
        for group in schema["groups"]:
            name = group["name"]
            dynamic_cells[name] = {feature: [] for feature in group["dynamic"]}
            static_cells[name] = {feature: [] for feature in group["static_numeric"]}
            vocab[name] = {feature: set() for feature in group["static_category"]}
            scope[name] = set()
        known_cells = {feature: [] for feature in schema["known_future"]}
        freq = schema["freq_seconds"]
        for episode_index, origins in per_episode.items():
            episode = dataset.episodes[episode_index]
            arrays, times = episode.observations, episode.times
            total = len(times)
            # 每个位置的「最晚可见末端时刻」：覆盖该位置的窗口中末端最大者
            end_origin = np.full(total, -1, np.int64)
            for origin in origins:
                window = slice(max(origin - length + 1, 0), origin + horizon + 1)
                np.maximum(end_origin[window], origin, out=end_origin[window])
            visible_time = np.full(total, _UNREACHED, np.int64)
            covered = end_origin >= 0
            visible_time[covered] = times[end_origin[covered] + horizon]
            # target：独立统计
            target_mask = (arrays["target_observed"][:, 0]
                           & (arrays["target_available_at"][:, 0] <= visible_time))
            target_cells.append(arrays["target"][target_mask, 0])
            # 动态组：唯一 (t, n) 单元
            for group in schema["groups"]:
                name = group["name"]
                present = arrays[f"{name}__present"]
                applicable = arrays[f"{name}__applicable"]
                for index, feature in enumerate(group["dynamic"]):
                    eligible = (present & applicable[:, index][None])
                    observed = (arrays[f"{name}__observed"][:, :, index] & eligible
                                & (arrays[f"{name}__available_at"][:, :, index] <= visible_time[:, None]))
                    dynamic_cells[name][feature].append(arrays[f"{name}__values"][:, :, index][observed])
                # 静态属性与类别：可见版本去重
                for entity, records in episode.entity_versions[name].items():
                    for record in records:
                        if not cls._version_visible(record, times, visible_time):
                            continue
                        scope[name].add(entity)
                        for feature in group["static_numeric"]:
                            value = record["numeric"][feature]
                            if value is not None:
                                static_cells[name][feature].append(np.array([value], np.float64))
                        for feature in group["static_category"]:
                            value = record["category"][feature]
                            if value is not None:
                                vocab[name][feature].add(value)
            # 已知未来协变量：只统计训练 origin 可见的记录
            if schema["known_future"]:
                for origin in origins:
                    stamp = int(times[origin])
                    future_time = stamp + np.arange(1, horizon + 1, dtype=np.int64) * freq
                    values, mask = episode.select_known(stamp, future_time)
                    for index, feature in enumerate(schema["known_future"]):
                        if mask[:, index].any():
                            known_cells[feature].append(values[mask[:, index], index])
        state = {
            "version": VERSION,
            "schema_fingerprint": dataset.episodes[0].schema_fingerprint,
            "groups": {},
            "target": {},
            "known_future": {},
            "vocab": {},
            "entity_scope": {},
        }
        specs = {group["name"]: group for group in schema["groups"]}
        for name, spec in specs.items():
            state["groups"][name] = {
                "dynamic": _role_stats(dynamic_cells[name][feature] for feature in spec["dynamic"]),
                "static": _role_stats(static_cells[name][feature] for feature in spec["static_numeric"]),
            }
        state["target"] = _role_stats([target_cells])
        state["known_future"] = _role_stats(known_cells[feature] for feature in schema["known_future"])
        state["vocab"] = {name: {feature: [PAD_TOKEN, UNK_TOKEN] + sorted(values)
                                 for feature, values in features.items()}
                          for name, features in vocab.items()}
        state["entity_scope"] = {name: sorted(entities) for name, entities in scope.items()}
        return cls(state)

    @staticmethod
    def _version_visible(record, times, visible_time):
        """版本区间与训练窗口相交，且相交位置上至少一个末端不早于 available_at。"""
        low = np.searchsorted(times, record["valid_from"], "left")
        high = (np.searchsorted(times, record["valid_to"], "left")
                if record["valid_to"] is not None else len(times))
        if low >= high:
            return False
        return bool((visible_time[low:high] >= record["available_at"]).any())

    # ---------------------------------------------------------- 序列化 ----

    def to_dict(self):
        return deepcopy(self.state)

    @classmethod
    def from_dict(cls, state, schema):
        """重建变换并校验 artifact；损坏统计拒绝加载，不现场重拟合。"""
        if not isinstance(state, dict) or state.get("version") != VERSION:
            raise ValueError("统计 artifact 版本不兼容")
        from .config import validate_schema
        validate_schema(schema)
        if state.get("schema_fingerprint") != canonical_hash(schema):
            raise ValueError("统计 artifact 与 schema 指纹不一致")
        groups = {group["name"]: group for group in schema["groups"]}
        if set(state.get("groups", {})) != set(groups):
            raise ValueError("统计 artifact 组集合与 schema 不一致")
        for name, group in groups.items():
            section = state["groups"][name]
            if not isinstance(section, dict) or set(section) != {"dynamic", "static"}:
                raise ValueError("统计 artifact 组字段不完整")
            _check_numeric(section["dynamic"], f"{name}.dynamic", len(group["dynamic"]))
            _check_numeric(section["static"], f"{name}.static", len(group["static_numeric"]))
        _check_numeric(state["target"], "target", 1)
        _check_numeric(state.get("known_future", {}), "known_future", len(schema["known_future"]))
        vocab = state.get("vocab", {})
        if set(vocab) != set(groups):
            raise ValueError("词表组集合不一致")
        for name, group in groups.items():
            if set(vocab[name]) != set(group["static_category"]):
                raise ValueError(f"词表 {name} 类别特征集合不一致")
            for feature, tokens in vocab[name].items():
                if (not isinstance(tokens, list) or len(tokens) < 2
                        or tokens[0] != PAD_TOKEN or tokens[1] != UNK_TOKEN
                        or len(set(tokens)) != len(tokens)):
                    raise ValueError(f"词表 {name}.{feature} 非法（前两位须为 PAD/UNK 且不重复）")
        if set(state.get("entity_scope", {})) != set(groups):
            raise ValueError("实体溯源集合与 schema 不一致")
        return cls(deepcopy(state))

    # -------------------------------------------------------------- apply ----

    def apply(self, sample):
        """标准化数值、类别 token 化并补充审计计数；原地更新并返回同一 sample。"""
        inputs = sample["inputs"]
        unknown = set()
        no_history = 0
        new_entity = 0
        for group in self.state["groups"]:
            section = inputs["groups"][group]
            stats = self.state["groups"][group]
            section["values"] = _standardize(section["values"], section["observed"], stats["dynamic"])
            section["static_numeric_history"] = _standardize(
                section["static_numeric_history"], section["static_numeric_mask"], stats["static"])
            raw = section["static_category_history"]
            token = np.ones(raw.shape, np.int64)  # 缺失类别映射 UNK
            for index, feature in enumerate(self.state["vocab"][group]):
                token[..., index] = _tokenize(raw[..., index], self._vocab_index[group][feature], unknown)
            section["static_category_history"] = token
            observed = section["observed"]
            if observed.shape[-1]:  # 纯静态组没有历史观测语义
                for node in range(observed.shape[1]):
                    if not observed[:, node, :].any():
                        no_history += 1
            scope = set(self.state["entity_scope"][group])
            new_entity += sum(entity not in scope for entity in section["entity_ids"])
        inputs["target_history"] = _standardize(
            inputs["target_history"], inputs["target_history_observed"], self.state["target"])
        if inputs["future_known"].shape[1]:
            inputs["future_known"] = _standardize(
                inputs["future_known"], inputs["future_known_mask"], self.state["known_future"])
        supervision = sample["supervision"]
        if supervision is not None:
            supervision["target"] = _standardize(
                supervision["target"], supervision["target_mask"], self.state["target"])
            for group in self.state["groups"]:
                future = supervision["future_groups"].get(group)
                if future is None or not future["values"].shape[-1]:
                    continue
                future["values"] = _standardize(future["values"], future["observed"],
                                                self.state["groups"][group]["dynamic"])
        sample["metadata"].update(unknown_category_count=len(unknown),
                                  no_history_entity_count=no_history,
                                  new_entity_count=new_entity)
        return sample

    def inverse_target(self, values):
        """标准化坐标回到原始单位；仅供输出逆变换，不改变内部统计。"""
        state = self.state["target"]
        array = np.asarray(values, np.float64)
        return array * state["std"][0] + state["mean"][0]


def _check_numeric(stats, label, expected):
    """校验合并后的并行 list 统计；长度必须等于声明特征数。"""
    if (not isinstance(stats, dict) or set(stats) != {"mean", "std", "count"}
            or any(not isinstance(stats[key], list) or len(stats[key]) != expected for key in stats)):
        raise ValueError(f"统计 artifact {label}: 结构或特征数非法")
    for index in range(expected):
        for key in ("mean", "std"):
            value = stats[key][index]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"统计 artifact {label}.{key}[{index}]: 非有限数值")
        if stats["std"][index] <= 0 or not isinstance(stats["count"][index], int) or stats["count"][index] <= 0:
            raise ValueError(f"统计 artifact {label}[{index}]: std 须为正、count 须为正整数")
