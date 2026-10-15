#!/usr/bin/env python
"""GSLAformer synthetic 标准包生成器：base 与机制任务 M1–M4（acceptance.md §7–§13）。

- 所有包 data_kind=synthetic；生成后立即以 EpisodePackage/PackageDataset 复检。
- 机制 oracle（M1 8/9、M2 7/5、M3 递推）由独立公式函数核算，不手填数字。
- 本生成器只提供输入，不构成模型或验收通过的证据。
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_provider.spatial_temporal.config import load_config, schema_from_config, validate_config  # noqa: E402
from data_provider.spatial_temporal.dataset import EpisodePackage, PackageDataset  # noqa: E402
from data_provider.spatial_temporal.scaling import TrainingTransform  # noqa: E402

GENERATOR_VERSION = "mechanisms-v1"
EPOCH = int(1577836800)  # 2020-01-01T00:00:00Z
HOUR = 3600
FILES = ("entities.json", "relations.json", "site.json", "measurements.json",
         "observations.npz", "known_future.npz")


# ==============================================================================
# 通用写包工具
# ==============================================================================

def _write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_episode(path, schema, order, arrays, entities, edges, site, measurements, known,
                  *, purpose=None, family_id=None, scenario=None):
    """写一个 single_episode 子包；observations 与 known_future 由调用方备好。"""
    path = Path(path)
    path.mkdir(parents=True)
    _write_json(path / "entities.json", entities)
    _write_json(path / "relations.json", {"edges": edges})
    _write_json(path / "site.json", site)
    _write_json(path / "measurements.json", {"mappings": measurements})
    np.savez(path / "observations.npz", **arrays)
    np.savez(path / "known_future.npz", **known)
    manifest = {"schema_version": 3, "data_kind": "synthetic", "layout": "single_episode",
                "schema": schema, "entity_order": order, "generator": GENERATOR_VERSION,
                "files": {name: _hash(path / name) for name in FILES}}
    if purpose is not None:
        manifest["purpose"] = purpose
    if family_id is not None:
        manifest["family_id"] = family_id
    if scenario is not None:
        manifest["scenario"] = scenario
    _write_json(path / "manifest.json", manifest)
    return manifest


def write_collection(path, schema, entries, split_mode, split_index, *, site_split=None):
    """写 episode_collection 根 manifest；子包由调用方先写好。"""
    manifest = {"schema_version": 3, "data_kind": "synthetic", "layout": "episode_collection",
                "schema": schema, "episodes": entries, "split_mode": split_mode,
                "split_index": split_index, "generator": GENERATOR_VERSION}
    if site_split is not None:
        manifest["site_split"] = site_split
    _write_json(Path(path) / "manifest.json", manifest)


def episode_arrays(schema, order, length):
    """按 schema 形状准备 observations 数组骨架（时间轴从 EPOCH 起每小时一步）。"""
    times = EPOCH + np.arange(length, dtype=np.int64) * HOUR
    arrays = {"timestamps": times, "target": np.zeros((length, 1), np.float32),
              "target_observed": np.ones((length, 1), bool),
              "target_available_at": times[:, None].copy()}
    for group in schema["groups"]:
        name, features = group["name"], len(group["dynamic"])
        nodes = len(order[name])
        arrays[f"{name}__values"] = np.zeros((length, nodes, features), np.float32)
        arrays[f"{name}__observed"] = np.ones((length, nodes, features), bool)
        arrays[f"{name}__applicable"] = np.ones((nodes, features), bool)
        arrays[f"{name}__present"] = np.ones((length, nodes), bool)
        arrays[f"{name}__available_at"] = np.broadcast_to(
            times[:, None, None], (length, nodes, features)).copy()
    return arrays


def apply_observed(arrays, name, mask):
    """施加观测缺失：未观测位置数值置零，保持真值只存在于 mask 处。"""
    arrays[f"{name}__observed"] = mask
    arrays[f"{name}__values"] = np.where(mask, arrays[f"{name}__values"], 0).astype(np.float32)


def empty_known():
    future = {key: np.empty(0, np.int64) for key in ("origin", "target_time", "feature_index",
                                                     "published_at", "revision")}
    future.update(value=np.empty(0, np.float32), observed=np.empty(0, bool))
    return future


def entity_records(order, numeric_fn, category_fn):
    """按组生成 entities.json 记录；numeric_fn/category_fn 返回该实体的属性 dict。"""
    document = {"groups": {}}
    for group, ids in order.items():
        rows = []
        for name in ids:
            rows.append({"id": name, "valid_from": 0, "valid_to": None, "available_at": 0,
                         "revision": 0, "numeric": numeric_fn(group, name) or {},
                         "category": category_fn(group, name) or {}})
        document["groups"][group] = rows
    return document


def primary_mappings(order, schema, prefix, unit):
    """为每个实体的每个动态特征登记一个 primary 测点；静态属性无测点。

    unit 可为全局单位字符串或 {feature: unit} 映射（AIDC 按特征单位登记）。
    """
    mappings = []
    feature_units = unit if isinstance(unit, dict) else None

    def unit_of(feature):
        return feature_units.get(feature, "arbitrary_unit") if feature_units else unit

    for group in schema["groups"]:
        name = group["name"]
        for entity in order[name]:
            for feature in group["dynamic"]:
                mappings.append({"measurement_id": f"{prefix}_{name}_{entity}_{feature}",
                                 "source_name": f"{prefix}:{name}/{entity}/{feature}",
                                 "group": name, "entity_id": entity, "feature": feature,
                                 "source_unit": unit_of(feature), "scale": 1.0, "offset": 0.0,
                                 "role": "primary"})
    return mappings


def target_contract():
    return {"target_id": "main", "quantity": "synthetic_total", "boundary_id": "whole",
            "unit": "arbitrary_unit", "sign_convention": "positive_consumption",
            "interval_label": "right", "aggregation": "time_weighted_mean",
            "meter_ids": ["main"], "coverage_description": "synthetic"}


def full_config(groups, relations, known_future, *, package, split_mode, split_index,
                seq_len, pred_len, variant="full", target="synthetic_total", unit="arbitrary_unit",
                extra_schema=None, batch_size=8, valid_stride=1, test_stride=1,
                limits=(96, 24)):
    """构造并通过 validate_config 的完整配置（机制任务共用取值）。"""
    data_block = {"package": package, "timezone": "UTC", "freq_seconds": 3600,
                  "seq_len": seq_len, "pred_len": pred_len, "train_stride": 1,
                  "valid_stride": valid_stride, "test_stride": test_stride,
                  "split_mode": split_mode, "split_index": split_index,
                  "batch_size": batch_size, "num_workers": 0,
                  "target_name": target, "target_unit": unit,
                  "groups": deepcopy(groups), "relation_types": deepcopy(relations),
                  "known_future": list(known_future)}
    if extra_schema:
        data_block.update(extra_schema)
    config = {
        "schema_version": 3,
        "experiment": {"model": "GSLAformer", "variant": variant, "data_kind": "synthetic",
                       "seed": 2023, "device": "cpu", "results_root": "./results/gslaformer"},
        "data": data_block,
        "model": {"spatial_dim": 32, "latent_dim": 32, "temporal_dim": 64,
                  "spatial_heads": 4, "relation_heads": 4, "temporal_heads": 4,
                  "spatial_layers": 1, "entity_history_layers": 1, "relation_layers": 2,
                  "temporal_layers": 2, "future_path_layers": 1, "dropout": 0.1,
                  "category_embedding_dim": 8, "lead_dim": 16},
        "limits": {"max_total_nodes": 128, "max_history_steps": limits[0],
                   "max_horizon_steps": limits[1], "max_explicit_attention_bytes": 268435456},
        "training": {"optimizer": "AdamW", "learning_rate": 0.001, "weight_decay": 0.0001,
                     "clip_grad_norm": 1.0, "latent_weight": 0.1, "stage_a_max_epochs": 20,
                     "stage_a_patience": 5, "b_short_epochs": 2, "b_medium_epochs": 2,
                     "b_full_max_epochs": 20, "b_full_patience": 5, "min_delta": 1e-6,
                     "amp": False, "truncate_rollout_grad": False},
        "evaluation": {"selection_metric": "valid_site_macro_mse" if split_mode == "site_time"
                       else "valid_macro_mse", "seasonal_period": 24,
                       "compare_seed": 2023},
    }
    return validate_config(config)


# ==============================================================================
# 周期 base（acceptance.md §7）
# ==============================================================================

BASE_GROUPS = [
    {"name": "assets", "dynamic": ["load", "utilization"], "static_numeric": ["capacity"],
     "static_category": ["kind"]},
    {"name": "zones", "dynamic": ["temperature"], "static_numeric": ["capacity"],
     "static_category": []},
    {"name": "sites", "dynamic": [], "static_numeric": ["capacity"], "static_category": []},
]
BASE_RELATIONS = [
    {"name": "asset_in_zone", "src_groups": ["assets"], "dst_groups": ["zones"],
     "acyclic": True, "reverse_message": True},
    {"name": "zone_in_site", "src_groups": ["zones"], "dst_groups": ["sites"],
     "acyclic": True, "reverse_message": True},
]
BASE_LENGTH = 320


def base_series(length, offset=0):
    """周期公式（§7）：返回 scheduled、load、utilization、temperature、capacity、target。

    offset 允许把相位整体平移，用于新增实体的「同公式新相位」历史。
    """
    t = np.arange(length, dtype=np.float64) + offset
    scheduled = (1 + np.sin(2 * np.pi * t / 24)) / 2
    capacity = np.array([1 + i / 10 for i in range(6)])
    utilization = np.stack([(1 + np.sin(2 * np.pi * t / 24 + i / 3)) / 2 for i in range(6)])
    load = capacity[:, None] * (1 + utilization) + 0.1 * scheduled[None, :]
    temperature = np.stack([20 + 2 * np.sin(2 * np.pi * t / 24 + j) for j in range(2)])
    parent_temp = temperature[[0, 0, 0, 1, 1, 1]]
    target = (load * (1 + 0.1 * parent_temp)).sum(axis=0) + 0.2 * scheduled
    return scheduled, capacity, utilization, load, temperature, target


def build_base(output):
    root = output / "base"
    length = BASE_LENGTH
    scheduled, capacity, utilization, load, temperature, target = base_series(length)
    rng = np.random.default_rng(2023)
    order = {"assets": [f"a{i}" for i in range(6)], "zones": ["z0", "z1"], "sites": ["s0"]}
    config = full_config(BASE_GROUPS, BASE_RELATIONS, ["scheduled_activity"], package=".",
                         split_mode="target_index", split_index=[0, 192, 256, 320],
                         seq_len=24, pred_len=6)
    schema = schema_from_config(config)
    arrays = episode_arrays(schema, order, length)
    arrays["target"][:, 0] = target.astype(np.float32)
    arrays["assets__values"][:, :, 0] = load.T
    arrays["assets__values"][:, :, 1] = utilization.T
    arrays["zones__values"][:, 0, 0] = temperature[0]
    arrays["zones__values"][:, 1, 0] = temperature[1]
    # 动态观测施加 5% 缺失；target 与 available_at 保持有效（§7）
    for name in ("assets", "zones"):
        apply_observed(arrays, name, rng.random(arrays[f"{name}__observed"].shape) >= 0.05)

    def numeric(group, name):
        if group == "assets":
            return {"capacity": 1 + int(name[1:]) / 10}
        if group == "zones":
            return {"capacity": float(int(name[1:]) + 1)}
        return {"capacity": 3.0}

    def category(group, name):
        return {"kind": "even" if int(name[1:]) % 2 == 0 else "odd"} if group == "assets" else {}

    entities = entity_records(order, numeric, category)
    edges = [{"src_group": "assets", "src_id": f"a{i}", "dst_group": "zones",
              "dst_id": f"z{i // 3}", "type": "asset_in_zone",
              "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
             for i in range(6)]
    edges += [{"src_group": "zones", "src_id": f"z{j}", "dst_group": "sites", "dst_id": "s0",
               "type": "zone_in_site", "valid_from": 0, "valid_to": None,
               "available_at": 0, "revision": 0} for j in range(2)]
    # scheduled_activity 计划：提前 24 小时发布，覆盖全部合法 origin（o+H 不越时间轴）
    origins = np.arange(23, length - 6)
    known = {"origin": np.repeat(EPOCH + origins * HOUR, 6).astype(np.int64),
             "target_time": np.concatenate(
                 [EPOCH + (o + np.arange(1, 7)) * HOUR for o in origins]).astype(np.int64),
             "feature_index": np.zeros(len(origins) * 6, np.int64),
             "value": np.concatenate([scheduled[o + 1:o + 7] for o in origins]).astype(np.float32),
             "observed": np.ones(len(origins) * 6, bool),
             "published_at": np.concatenate(
                 [EPOCH + (o + np.arange(1, 7) - 24) * HOUR for o in origins]).astype(np.int64),
             "revision": np.zeros(len(origins) * 6, np.int64)}
    write_episode(root, schema, order, arrays, entities, edges,
                  {"site_id": "synthetic_site", "target_contract": target_contract()},
                  primary_mappings(order, schema, "base", "arbitrary_unit"), known)
    _write_json(root / "config.json", config)
    _write_json(root / "oracle.json", {
        "target_all_finite": bool(np.isfinite(arrays["target"]).all()),
        "target_observed_ratio": float(arrays["target_observed"].mean()),
        "dynamic_missing_ratio": float(1 - np.concatenate(
            [arrays["assets__observed"].ravel(), arrays["zones__observed"].ravel()]).mean()),
        "formula_tail": float(target[-1]),
    })
    return root


# ==============================================================================
# 机制任务骨架（acceptance.md §8）
# ==============================================================================

MECH_LENGTH, MECH_HORIZON = 24, 6
FAMILY_BOUNDS = (48, 64, 80)  # 0..47 train / 48..63 valid / 64..79 test


def family_split(family):
    return "train" if family < FAMILY_BOUNDS[0] else "valid" if family < FAMILY_BOUNDS[1] else "test"


def mech_time(index):
    return EPOCH + index * HOUR


def write_task(root, task_id, groups, relations, builder, oracle, *, known_future=()):
    """写机制任务 collection（episodes + config + oracle）。"""
    task = Path(root) / task_id
    entries = builder(task)
    config = full_config(groups, relations, known_future, package=".", split_mode="episode_family",
                         split_index=None, seq_len=MECH_LENGTH, pred_len=MECH_HORIZON)
    write_collection(task, schema_from_config(config), entries, "episode_family", None)
    _write_json(task / "config.json", config)
    _write_json(task / "oracle.json", oracle)
    return task


def mech_episode(path, groups, relations, order, arrays, target, edges, numeric_fn,
                 category_fn, *, family_id, prefix, known=None, observed_mask=None,
                 present=None, known_future=()):
    """写一个机制 episode 子包；K=0 默认无协变量，掩码由调用方传入。"""
    schema = {"timezone": "UTC", "freq_seconds": 3600, "groups": deepcopy(groups),
              "relation_types": deepcopy(relations), "known_future": list(known_future),
              "target_name": "synthetic_total", "target_unit": "arbitrary_unit"}
    arrays["target"][:, 0] = target.astype(np.float32)
    for name, mask in (observed_mask or {}).items():
        apply_observed(arrays, name, mask)
    for name, flag in (present or {}).items():
        arrays[f"{name}__present"] = flag
    write_episode(path, schema, order, arrays,
                  entity_records(order, numeric_fn, category_fn), edges,
                  {"site_id": "synthetic_site", "target_contract": target_contract()},
                  primary_mappings(order, schema, prefix, "arbitrary_unit"),
                  empty_known() if known is None else known, family_id=family_id)


def plan_records(origin_index, values, *, feature_index=0, publish_lead=1):
    """构造单 origin 的未来计划记录；published_at 早于 origin。"""
    count = len(values)
    return {"origin": np.full(count, mech_time(origin_index), np.int64),
            "target_time": np.array([mech_time(origin_index + k + 1) for k in range(count)], np.int64),
            "feature_index": np.full(count, feature_index, np.int64),
            "value": np.asarray(values, np.float32),
            "observed": np.ones(count, bool),
            "published_at": np.full(count, mech_time(origin_index - publish_lead), np.int64),
            "revision": np.zeros(count, np.int64)}


def split_entry(episode_id, family_id, split, path_name):
    return {"id": episode_id, "family_id": family_id, "site_id": "synthetic_site",
            "split": split, "path": path_name}


# ==============================================================================
# M1：保留同一实体的时间轨迹（§9）
# ==============================================================================

M1_GROUPS = [{"name": "assets", "dynamic": ["load"], "static_numeric": [], "static_category": []},
             {"name": "zones", "dynamic": [], "static_numeric": [], "static_category": []}]
M1_RELATIONS = [{"name": "belongs", "src_groups": ["assets"], "dst_groups": ["zones"],
                 "acyclic": True, "reverse_message": True}]


def m1_base_history():
    """前 L-1 步恒 [1,3]；origin 时甲 [2,4]、乙 [4,2]。"""
    steady = np.tile(np.array([[1.0, 3.0]]), (MECH_LENGTH - 1, 1))
    return np.vstack([steady, [[2.0, 4.0]]]), np.vstack([steady, [[4.0, 2.0]]])


def m1_rollout(history):
    """x_{s+1} = x_s + max(x_s - x_{s-1}, 0)；返回 H 步 [H,2]。"""
    future, current, previous = [], history[-1], history[-2]
    for _ in range(MECH_HORIZON):
        nxt = current + np.maximum(current - previous, 0)
        previous, current = current, nxt
        future.append(nxt)
    return np.stack(future)


def build_m1(root):
    # oracle：基础对（a=1,b=0）下一总量 8/9，由公式核算
    first, second = m1_base_history()
    oracle = {"first_next_total": float(m1_rollout(first)[0].sum()),
              "second_next_total": float(m1_rollout(second)[0].sum())}
    assert abs(oracle["first_next_total"] - 8) < 1e-9
    assert abs(oracle["second_next_total"] - 9) < 1e-9
    pair = {"distinct": True}

    def builder(task):
        entries = []
        for family in range(FAMILY_BOUNDS[2]):
            rng = np.random.default_rng(np.random.SeedSequence([2023, 1, family]))
            a, b = rng.uniform(0.5, 2), rng.uniform(0, 2)
            for index, base in enumerate(m1_base_history()):
                history = b + a * base  # 正尺度与偏移只作用于数值，不作为额外属性输入
                values = np.vstack([history, m1_rollout(history)])
                target = values.sum(axis=1)
                episode_id = f"m1_f{family:03d}_{'ab'[index]}"
                order = {"assets": ["a0", "a1"], "zones": ["z0"]}
                arrays = episode_arrays({"groups": M1_GROUPS}, order, MECH_LENGTH + MECH_HORIZON)
                arrays["assets__values"][:, :, 0] = values
                edges = [{"src_group": "assets", "src_id": f"a{i}", "dst_group": "zones",
                          "dst_id": "z0", "type": "belongs", "valid_from": 0, "valid_to": None,
                          "available_at": 0, "revision": 0} for i in range(2)]
                mech_episode(task / episode_id, M1_GROUPS, M1_RELATIONS, order, arrays, target,
                             edges, lambda g, n: {}, lambda g, n: {},
                             family_id=f"f{family:03d}", prefix=f"m1f{family:03d}")
                entries.append(split_entry(episode_id, f"f{family:03d}",
                                           family_split(family), episode_id))
            # 成对区分核算：甲乙 origin 后首步总量必须不同
            y1 = m1_rollout(b + a * first)[0].sum()
            y2 = m1_rollout(b + a * second)[0].sum()
            pair["distinct"] = pair["distinct"] and abs(y1 - y2) > 1e-9
        return entries

    task = write_task(root, "M1", M1_GROUPS, M1_RELATIONS, builder,
                      {"oracle": oracle, "families": FAMILY_BOUNDS[2]})
    assert pair["distinct"]
    oracle_result = json.loads((Path(task) / "oracle.json").read_text())
    oracle_result["pair_distinct_all_families"] = True
    _write_json(Path(task) / "oracle.json", oracle_result)
    return task


# ==============================================================================
# M2：归属关系承载可识别信息（§10）
# ==============================================================================

M2_GROUPS = [{"name": "assets", "dynamic": ["load"], "static_numeric": [], "static_category": []},
             {"name": "zones", "dynamic": [], "static_numeric": ["gain"], "static_category": []}]
M2_RELATIONS = M1_RELATIONS
M2_GAINS = np.array([1.0, 2.0])


def m2_edges(swap):
    """origin 前归属相同（0→z0,1→z1）；乙在 origin 交换父级（旧边关闭、新边生效）。"""
    origin_time = mech_time(MECH_LENGTH - 1)
    edges = []
    for asset, zone in ((0, 0), (1, 1)):
        edges.append({"src_group": "assets", "src_id": f"a{asset}", "dst_group": "zones",
                      "dst_id": f"z{zone}", "type": "belongs", "valid_from": EPOCH,
                      "valid_to": origin_time, "available_at": EPOCH, "revision": 0})
    for asset, zone in ((0, 1), (1, 0)) if swap else ((0, 0), (1, 1)):
        edges.append({"src_group": "assets", "src_id": f"a{asset}", "dst_group": "zones",
                      "dst_id": f"z{zone}", "type": "belongs", "valid_from": origin_time,
                      "valid_to": None, "available_at": origin_time, "revision": 1})
    return edges


def m2_target(load, swap):
    """y_s = Σ x_{i,s-1} gain_{parent(i,s-1)}；x[-1]=x[0]、parent[-1]=初始归属。

    未来 load 恒为 origin 值、未来关系恒为 origin 归属，故未来 y 恒定。
    """
    origin_parent = np.array([1, 0]) if swap else np.array([0, 1])
    initial_parent = np.array([0, 1])
    # 前驱序列：s 的 x_{s-1}；x[-1]=x[0]，未来 load 恒为 origin 值
    values = np.vstack([load[:1], load, np.tile(load[-1], (MECH_HORIZON - 1, 1))])
    y = []
    for s in range(MECH_LENGTH + MECH_HORIZON):
        parent = initial_parent if s < MECH_LENGTH else origin_parent
        y.append((values[s] * M2_GAINS[parent]).sum())
    return np.array(y)


def build_m2(root):
    base = np.vstack([np.tile([[1.0, 1.0]], (MECH_LENGTH - 1, 1)), [[1.0, 3.0]]])
    oracle = {"first_future": float(m2_target(base, False)[MECH_LENGTH]),
              "second_future": float(m2_target(base, True)[MECH_LENGTH]),
              "history_target_identical": bool(np.allclose(m2_target(base, False)[:MECH_LENGTH],
                                                            m2_target(base, True)[:MECH_LENGTH]))}
    assert abs(oracle["first_future"] - 7) < 1e-9
    assert abs(oracle["second_future"] - 5) < 1e-9
    assert oracle["history_target_identical"]

    def builder(task):
        entries = []
        for family in range(FAMILY_BOUNDS[2]):
            rng = np.random.default_rng(np.random.SeedSequence([2023, 2, family]))
            scale = rng.uniform(0.5, 2)
            load = scale * base
            for index, swap in enumerate((False, True)):
                target = m2_target(load, swap)
                values = np.vstack([load, np.tile(load[-1], (MECH_HORIZON, 1))])
                episode_id = f"m2_f{family:03d}_{'ab'[index]}"
                order = {"assets": ["a0", "a1"], "zones": ["z0", "z1"]}
                arrays = episode_arrays({"groups": M2_GROUPS}, order, MECH_LENGTH + MECH_HORIZON)
                arrays["assets__values"][:, :, 0] = values
                mech_episode(task / episode_id, M2_GROUPS, M2_RELATIONS, order, arrays, target,
                             m2_edges(swap),
                             lambda g, n: {"gain": M2_GAINS[int(n[1:])]} if g == "zones" else {},
                             lambda g, n: {}, family_id=f"f{family:03d}", prefix=f"m2f{family:03d}")
                entries.append(split_entry(episode_id, f"f{family:03d}",
                                           family_split(family), episode_id))
        return entries

    return write_task(root, "M2", M2_GROUPS, M2_RELATIONS, builder, {"oracle": oracle,
                                                                     "families": FAMILY_BOUNDS[2]})


# ==============================================================================
# M3：未来计划路径与状态积累（§11）
# ==============================================================================

M3_GROUPS = [{"name": "assets", "dynamic": ["load"], "static_numeric": [], "static_category": []},
             {"name": "zones", "dynamic": [], "static_numeric": [], "static_category": []}]
M3_RELATIONS = [{"name": "belongs", "src_groups": ["assets"], "dst_groups": ["zones"],
                 "acyclic": True, "reverse_message": True}]


def m3_series(x0, u):
    """x_{s+1} = 0.8 x_s + u_{s+1}；u 长度 L+H-1，返回完整 x 序列 [L+H]。"""
    xs = [np.float64(x0)]
    for step in range(len(u)):
        xs.append(0.8 * xs[-1] + u[step])
    return np.array(xs)


def build_m3(root):
    # oracle：x_origin=10 示例由公式独立核算（§11.1）
    first = m3_series(10.0, np.concatenate([np.zeros(6)]))
    second = m3_series(10.0, np.concatenate([[5.0], np.zeros(5)]))
    oracle = {"x_origin": 10.0, "first_step2": float(first[2]), "second_step2": float(second[2])}
    assert abs(oracle["first_step2"] - 6.4) < 1e-9 and abs(oracle["second_step2"] - 10.4) < 1e-9

    def builder(task):
        entries = []
        for family in range(FAMILY_BOUNDS[2]):
            rng = np.random.default_rng(np.random.SeedSequence([2023, 3, family]))
            x0 = rng.uniform(0, 10)
            u_history = rng.uniform(0, 2, MECH_LENGTH - 1)
            split = family_split(family)
            for index in range(2):
                # train 成对：甲计划全 0，乙第一步 +5；valid/test：甲随机计划，乙同甲首步 +5
                if split == "train":
                    base_plan = np.zeros(MECH_HORIZON)
                else:
                    base_plan = rng.uniform(0, 2, MECH_HORIZON)
                plan = base_plan.copy()
                if index == 1:
                    plan[0] += 5.0
                u_all = np.concatenate([u_history, plan])
                x = m3_series(x0, u_all)
                target = x.copy()  # y = x（状态即已观测 load）
                episode_id = f"m3_f{family:03d}_{'ab'[index]}"
                order = {"assets": ["a0"], "zones": ["z0"]}
                arrays = episode_arrays({"groups": M3_GROUPS}, order, MECH_LENGTH + MECH_HORIZON)
                arrays["assets__values"][:, 0, 0] = x
                edges = [{"src_group": "assets", "src_id": "a0", "dst_group": "zones",
                          "dst_id": "z0", "type": "belongs", "valid_from": 0, "valid_to": None,
                          "available_at": 0, "revision": 0}]
                mech_episode(task / episode_id, M3_GROUPS, M3_RELATIONS, order, arrays, target,
                             edges, lambda g, n: {}, lambda g, n: {},
                             family_id=f"f{family:03d}", prefix=f"m3f{family:03d}",
                             known=plan_records(MECH_LENGTH - 1, plan),
                             known_future=["scheduled_activity"])
                entries.append(split_entry(episode_id, f"f{family:03d}", split, episode_id))
        return entries

    return write_task(root, "M3", M3_GROUPS, M3_RELATIONS, builder, {"oracle": oracle,
                                                                     "families": FAMILY_BOUNDS[2]},
                      known_future=["scheduled_activity"])


# ==============================================================================
# M4：规模变化与新实体（§12）
# ==============================================================================

M4_GROUPS = [{"name": "assets", "dynamic": ["load"], "static_numeric": ["capacity"],
              "static_category": []},
             {"name": "zones", "dynamic": [], "static_numeric": ["gain"], "static_category": []}]
M4_RELATIONS = M1_RELATIONS
M4_GAINS = np.array([1.0, 2.0])


def m4_family_params(family, rng):
    """按 family 编号决定实体数量与拓扑（§12.1）。"""
    split = family_split(family)
    assets = (2, 4, 6)[family % 3] if split == "train" else (3, 5, 7)[family % 3]
    zones = 1 if family % 2 == 0 else 2
    capacity = rng.uniform(0.5, 2, assets)
    parent = rng.integers(0, zones, assets)
    return split, assets, zones, capacity, parent


def m4_series(capacity, parent, u):
    """x_{i,0}=capacity；x_{s+1}=0.8x+0.2capacity(1+u_{s+1})；y=Σ gain_parent x。"""
    total = MECH_LENGTH + MECH_HORIZON
    x = np.zeros((total, len(capacity)))
    x[0] = capacity
    for step in range(1, total):
        x[step] = 0.8 * x[step - 1] + 0.2 * capacity * (1 + u[step])
    y = (x * M4_GAINS[parent][None, :]).sum(axis=1)
    return x, y


def m4_edges(parent, order_assets):
    return [{"src_group": "assets", "src_id": order_assets[i], "dst_group": "zones",
             "dst_id": f"z{int(parent[i])}", "type": "belongs", "valid_from": 0,
             "valid_to": None, "available_at": 0, "revision": 0} for i in range(len(parent))]


def build_m4(root):
    # oracle：family 0 独立重算并核对生成一致性（在 builder 内逐 family 断言）
    def builder(task):
        entries = []
        scale_report = {"train": set(), "valid": set(), "test": set()}
        for family in range(FAMILY_BOUNDS[2]):
            rng = np.random.default_rng(np.random.SeedSequence([2023, 4, family]))
            split, assets, zones, capacity, parent = m4_family_params(family, rng)
            u = (rng.random(MECH_LENGTH + MECH_HORIZON) < 0.5).astype(np.float64)
            masked_asset = int(rng.integers(0, assets))
            scale_report[split].add(assets)
            x, y = m4_series(capacity, parent, u)
            order = {"assets": [f"a{i}" for i in range(assets)],
                     "zones": [f"z{j}" for j in range(zones)]}
            for index in range(2):
                episode_id = f"m4_f{family:03d}_{'ab'[index]}"
                arrays = episode_arrays({"groups": M4_GROUPS}, order, MECH_LENGTH + MECH_HORIZON)
                arrays["assets__values"][:, :, 0] = x
                mask = None
                if index == 1:  # 屏蔽一个随机 asset 的全部历史观测；真值与属性保留
                    mask = {"assets": np.ones_like(arrays["assets__observed"])}
                    mask["assets"][:MECH_LENGTH, masked_asset, :] = False
                capacity_map = {f"a{i}": float(capacity[i]) for i in range(assets)}
                gain_map = {f"z{j}": float(M4_GAINS[j]) for j in range(zones)}
                mech_episode(task / episode_id, M4_GROUPS, M4_RELATIONS, order, arrays, y,
                             m4_edges(parent, order["assets"]),
                             lambda g, n: ({"capacity": capacity_map[n]} if g == "assets"
                                           else {"gain": gain_map[n]}),
                             lambda g, n: {}, family_id=f"f{family:03d}",
                             prefix=f"m4f{family:03d}",
                             known=plan_records(MECH_LENGTH - 1, u[MECH_LENGTH:]),
                             observed_mask=mask, known_future=["scheduled_activity"])
                entries.append(split_entry(episode_id, f"f{family:03d}", split, episode_id))
        return entries

    task = write_task(root, "M4", M4_GROUPS, M4_RELATIONS, builder,
                      {"oracle": {"train_scales": [2, 4, 6], "valid_scales": [3, 5, 7],
                                  "test_scales": [3, 5, 7], "masked_variant": True},
                       "families": FAMILY_BOUNDS[2]},
                      known_future=["scheduled_activity"])
    # origin 生效请求：新实体恰在请求 origin（包末索引 29）投运，此前未投运、无任何观测
    rng = np.random.default_rng(np.random.SeedSequence([2023, 4, 0]))
    split, assets, zones, capacity, parent = m4_family_params(0, rng)
    u = (rng.random(MECH_LENGTH + MECH_HORIZON) < 0.5).astype(np.float64)
    x, y = m4_series(capacity, parent, u)
    total = MECH_LENGTH + MECH_HORIZON
    origin_index = total - 1
    new_capacity = float(rng.uniform(0.5, 2))
    x_new = np.zeros(total)
    x_new[origin_index] = new_capacity  # origin 时刻按生成规则初值投运
    target = y + M4_GAINS[0] * x_new  # target 自 origin 起真实反映激活后总量
    order = {"assets": [f"a{i}" for i in range(assets)] + ["a_new"],
             "zones": [f"z{j}" for j in range(zones)]}
    arrays = episode_arrays({"groups": M4_GROUPS}, order, total)
    arrays["assets__values"][:, :-1, 0] = x
    arrays["assets__values"][:, -1, 0] = x_new
    present = np.ones((total, assets + 1), bool)
    present[:origin_index, -1] = False  # 新实体此前未投运
    observed = np.ones_like(arrays["assets__observed"])
    observed[:, -1, :] = False  # 新实体无任何观测（含 origin）
    capacity_map = {f"a{i}": float(capacity[i]) for i in range(assets)}
    capacity_map["a_new"] = new_capacity
    gain_map = {f"z{j}": float(M4_GAINS[j]) for j in range(zones)}
    edges = m4_edges(parent, order["assets"][:-1])
    edges.append({"src_group": "assets", "src_id": "a_new", "dst_group": "zones", "dst_id": "z0",
                  "type": "belongs", "valid_from": mech_time(origin_index), "valid_to": None,
                  "available_at": mech_time(origin_index), "revision": 0})
    request = Path(root) / "M4" / "request_origin_activation"
    mech_episode(request, M4_GROUPS, M4_RELATIONS, order, arrays, target, edges,
                 lambda g, n: ({"capacity": capacity_map[n]} if g == "assets"
                               else {"gain": gain_map[n]}),
                 lambda g, n: {}, family_id="request", prefix="m4req",
                 known=plan_records(origin_index, u[MECH_LENGTH:]),
                 observed_mask={"assets": observed},
                 present={"assets": present}, known_future=["scheduled_activity"])
    # forecast 请求只保留历史 target：purpose 标记后 dataset 拒绝其参与训练
    manifest = json.loads((request / "manifest.json").read_text())
    manifest["purpose"] = "forecast"
    _write_json(request / "manifest.json", manifest)
    return task


# ==============================================================================
# M5：契约与数值边界夹具（§13，不训练模型）
# ==============================================================================

def build_m5(root):
    """从合法 mini 包派生坏输入夹具；预期错误关键词写入 M5/suite.json。"""
    task = Path(root) / "M5"
    bad = task / "bad_inputs"
    bad.mkdir(parents=True, exist_ok=True)
    # 合法基线需要至少一个动态组：assets(load) + zones(space_level) 同组 contains 链
    groups = [{"name": "assets", "dynamic": ["load"], "static_numeric": [], "static_category": []},
              {"name": "zones", "dynamic": [], "static_numeric": [], "static_category": ["space_level"]}]
    relations = [{"name": "contains", "src_groups": ["zones"], "dst_groups": ["zones"],
                  "acyclic": True, "reverse_message": True}]

    def clone(path):
        import shutil
        target = bad / path
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(bad / "_valid", target)
        return target

    # 合法模板：2 个 assets + 三个 zone 的 contains 链
    order = {"assets": ["a0", "a1"], "zones": ["z0", "z1", "z2"]}
    arrays = episode_arrays({"groups": groups}, order, MECH_LENGTH)
    arrays["assets__values"][:, :, 0] = np.arange(MECH_LENGTH, dtype=np.float32)[:, None]
    edges = [{"src_group": "zones", "src_id": "z0", "dst_group": "zones", "dst_id": "z1",
              "type": "contains", "valid_from": 0, "valid_to": None, "available_at": 0,
              "revision": 0},
             {"src_group": "zones", "src_id": "z1", "dst_group": "zones", "dst_id": "z2",
              "type": "contains", "valid_from": 0, "valid_to": None, "available_at": 0,
              "revision": 0}]
    write_episode(bad / "_valid", {"timezone": "UTC", "freq_seconds": 3600, "groups": groups,
                                   "relation_types": relations, "known_future": [],
                                   "target_name": "synthetic_total",
                                   "target_unit": "arbitrary_unit"}, order, arrays,
                  entity_records(order, lambda g, n: {},
                                 lambda g, n: ({"space_level": "building" if n == "z0" else "room"}
                                               if g == "zones" else {})), edges,
                  {"site_id": "synthetic_site", "target_contract": target_contract()},
                  primary_mappings(order, {"groups": groups}, "m5", "arbitrary_unit"),
                  empty_known())

    fixtures = {}
    # 1) 同组 contains 成环
    path = clone("type_cycle")
    document = json.loads((path / "relations.json").read_text())
    document["edges"].append({"src_group": "zones", "src_id": "z2", "dst_group": "zones",
                              "dst_id": "z0", "type": "contains", "valid_from": 0,
                              "valid_to": None, "available_at": 0, "revision": 0})
    _write_json(path / "relations.json", document)
    _rehash(path)
    fixtures["type_cycle"] = {"error": "无环|循环", "trigger": "sample"}
    # 2) 悬空端点
    path = clone("dangling_edge")
    document = json.loads((path / "relations.json").read_text())
    document["edges"][0]["dst_id"] = "z_missing"
    _write_json(path / "relations.json", document)
    _rehash(path)
    fixtures["dangling_edge"] = {"error": "悬空"}
    # 3) 实体 ID 重复
    path = clone("duplicate_id")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["entity_order"]["zones"] = ["z0", "z0", "z2"]
    _write_json(path / "manifest.json", manifest)
    _rehash(path)
    fixtures["duplicate_id"] = {"error": "不重复|不一致"}
    # 4) observed 位置非有限值
    path = clone("inf_value")
    with np.load(path / "observations.npz") as archive:
        arrays_bad = {key: archive[key] for key in archive.files}
    arrays_bad["target"][0, 0] = np.inf
    np.savez(path / "observations.npz", **arrays_bad)
    _rehash(path)
    fixtures["inf_value"] = {"error": "有限"}
    # 5) 未知 schema 版本
    path = clone("unknown_schema")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["schema_version"] = 2
    _write_json(path / "manifest.json", manifest)
    fixtures["unknown_schema"] = {"error": "schema_version"}
    # 6) 路径越界引用
    path = clone("path_escape")
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["files"]["../escape"] = "0" * 64
    _write_json(path / "manifest.json", manifest)
    fixtures["path_escape"] = {"error": "越界"}
    # 7) 对象 dtype NPZ
    path = clone("object_npz")
    with np.load(path / "observations.npz") as archive:
        arrays_bad = {key: archive[key] for key in archive.files}
    arrays_bad["target"] = arrays_bad["target"].astype(object)
    np.savez(path / "observations.npz", **arrays_bad)
    _rehash(path)
    fixtures["object_npz"] = {"error": "对象 dtype|NPZ"}
    _write_json(task / "suite.json", {"kind": "data_contract",
                                      "schema": {"groups": groups, "relation_types": relations},
                                      "fixtures": fixtures,
                                      "note": "q helper 数值夹具由 tests 内联构造（§13），不落盘"})
    return task


def _rehash(path):
    manifest = json.loads((Path(path) / "manifest.json").read_text())
    for name in FILES:
        if name in manifest["files"] and (Path(path) / name).exists():
            manifest["files"][name] = _hash(Path(path) / name)
    _write_json(Path(path) / "manifest.json", manifest)


# ==============================================================================
# AIDC 多站点场景（acceptance.md §16–§19，generator_version=aidc-v1）
# ==============================================================================

AIDC_GROUPS = [
    {"name": "compute", "dynamic": ["allocated_gpu", "active_jobs", "queue_depth",
                                    "throughput", "arrival_rate"],
     "static_numeric": ["gpu_capacity"], "static_category": ["pool_type"]},
    {"name": "it_power", "dynamic": ["active_power_kw", "cpu_utilization", "gpu_utilization",
                                     "online_fraction"],
     "static_numeric": ["rated_power_kw", "represented_count"],
     "static_category": ["entity_type"]},
    {"name": "distribution", "dynamic": ["active_power_kw", "load_ratio"],
     "static_numeric": ["rated_capacity_kw"], "static_category": ["entity_type", "port"]},
    {"name": "hvac", "dynamic": ["active_power_kw", "running_fraction", "frequency_hz",
                                 "flow_rate", "supply_temp", "return_temp"],
     "static_numeric": ["rated_power_kw", "rated_flow_rate", "design_cooling_kw"],
     "static_category": ["device_type"]},
    {"name": "space", "dynamic": ["active_power_kw", "room_temp", "room_humidity"],
     "static_numeric": ["area_m2", "it_capacity_kw"], "static_category": ["space_level"]},
    {"name": "environment", "dynamic": ["dry_bulb_temp", "humidity"],
     "static_numeric": [], "static_category": ["station_type"]},
]
AIDC_RELATIONS = [
    {"name": "contains", "src_groups": ["space"], "dst_groups": ["space", "it_power"],
     "acyclic": True, "reverse_message": True},
    {"name": "supplies", "src_groups": ["distribution"], "dst_groups": ["distribution", "it_power"],
     "acyclic": True, "reverse_message": False},
    {"name": "meter_covers", "src_groups": ["space"], "dst_groups": ["space", "distribution", "it_power"],
     "acyclic": True, "reverse_message": False},
    {"name": "cools", "src_groups": ["hvac"], "dst_groups": ["hvac", "space", "it_power"],
     "acyclic": False, "reverse_message": True},
    {"name": "runs_on", "src_groups": ["compute"], "dst_groups": ["it_power"],
     "acyclic": False, "reverse_message": False},
    {"name": "associated_env", "src_groups": ["environment"], "dst_groups": ["space", "hvac"],
     "acyclic": False, "reverse_message": True},
]
AIDC_SITES, AIDC_PREWARM, AIDC_STEPS = 12, 168, 1440
AIDC_SPLIT = [0, 864, 1152, 1440]
AIDC_SITE_SPLIT = {"train": [f"aidc_{i:02d}" for i in range(8)],
                   "heldout_valid": ["aidc_08", "aidc_09"],
                   "heldout_test": ["aidc_10", "aidc_11"]}
AIDC_FEATURE_UNITS = {
    "allocated_gpu": "count", "active_jobs": "count", "queue_depth": "count",
    "throughput": "count_per_hour", "arrival_rate": "count_per_hour",
    "gpu_capacity": "count", "active_power_kw": "kW", "cpu_utilization": "fraction",
    "gpu_utilization": "fraction", "online_fraction": "fraction",
    "rated_power_kw": "kW", "represented_count": "count", "rated_capacity_kw": "kW",
    "load_ratio": "fraction", "running_fraction": "fraction", "frequency_hz": "Hz",
    "flow_rate": "m3_per_hour", "supply_temp": "degree_celsius", "return_temp": "degree_celsius",
    "rated_flow_rate": "m3_per_hour", "design_cooling_kw": "kW",
    "room_temp": "degree_celsius", "room_humidity": "fraction", "area_m2": "m2",
    "it_capacity_kw": "kW", "dry_bulb_temp": "degree_celsius", "humidity": "fraction",
    "planned_extra_utilization": "fraction", "forecast_temp": "degree_celsius",
}
AIDC_KNOWN = ["planned_extra_utilization", "forecast_temp"]
AIDC_MAX_KNOWN_STEPS = 72


def aidc_topology(site):
    """§19.1 参数网格与 §18.6 站点行：实体注册、参数、typed relations。

    stream 0=拓扑（含一次抽取的服务器/配电/暖通参数）。
    """
    rng = np.random.default_rng(np.random.SeedSequence([2023, 6, site, 0]))
    n_server = (8, 12, 16)[site % 3]
    n_rack = (2, 3, 4)[site % 3]
    n_room = 1 + site % 2
    n_building = 1 + (site // 2) % 2
    n_hvac = 1 + site % 2
    dual = site % 2 == 1
    deep = site == 11
    rated = rng.uniform(0.5, 2, n_server)
    idle_ratio = rng.uniform(0.15, 0.45, n_server)
    alpha = rng.uniform(0.8, 1.5, n_server)
    capacity = float(rated.sum())
    eta_ups = float(rng.uniform(0.92, 0.98))
    eta_tr = float(rng.uniform(0.96, 0.99))
    order = {"compute": ["pool_inference", "pool_training"],
             "it_power": [f"s{i}" for i in range(n_server)] + [f"r{j}" for j in range(n_rack)],
             "distribution": (["bus0"] if deep else []) + ["tr0"]
             + [f"ups{tag}_{side}" for tag in ("A", "B")[: 2 if dual else 1] for side in ("in", "out")],
             "hvac": [f"ch{j}" for j in range(n_hvac)] + [f"ct{j}" for j in range(n_hvac)]
             + [f"pp{j}" for j in range(n_hvac)] + [f"sp{j}" for j in range(n_hvac)],
             "space": ["site"] + [f"bd{j}" for j in range(n_building)] + [f"rm{k}" for k in range(n_room)],
             "environment": ["env0"]}
    attributes = {"compute": {}, "it_power": {}, "distribution": {}, "hvac": {},
                  "space": {}, "environment": {}}
    for pool, odd in (("pool_inference", 0), ("pool_training", 1)):
        members = [i for i in range(n_server) if i % 2 == odd]
        attributes["compute"][pool] = {"numeric": {"gpu_capacity": float(8 * len(members))},
                                       "category": {"pool_type": "training" if odd else "inference"}}
    for i in range(n_server):
        attributes["it_power"][f"s{i}"] = {"numeric": {"rated_power_kw": float(rated[i]),
                                                       "represented_count": 1.0},
                                           "category": {"entity_type": "server"}}
    rack_members = {j: [i for i in range(n_server) if i % n_rack == j] for j in range(n_rack)}
    for j, members in rack_members.items():
        attributes["it_power"][f"r{j}"] = {"numeric": {"rated_power_kw": float(rated[members].sum()),
                                                       "represented_count": float(len(members))},
                                           "category": {"entity_type": "rack"}}
    tr_rated = capacity * 2.0
    attributes["distribution"]["tr0"] = {"numeric": {"rated_capacity_kw": tr_rated},
                                         "category": {"entity_type": "transformer", "port": "low_side"}}
    if deep:
        attributes["distribution"]["bus0"] = {"numeric": {"rated_capacity_kw": tr_rated},
                                              "category": {"entity_type": "busway", "port": "branch"}}
    ups_rated = capacity * (0.8 if dual else 1.3)
    for tag in ("A", "B")[: 2 if dual else 1]:
        for side in ("in", "out"):
            attributes["distribution"][f"ups{tag}_{side}"] = {
                "numeric": {"rated_capacity_kw": ups_rated},
                "category": {"entity_type": "ups", "port": side}}
    pump_budget = 0.05 * capacity  # 0.01C + 0.04C 的峰值预算（§18.6）
    chiller_rated = 1.2 * capacity / 2 / max(n_hvac, 1)  # Qmax 份额 / COP 下限
    n_cool_devices = 4 * n_hvac
    for kind, prefix in (("chiller", "ch"), ("cooling_tower", "ct"),
                         ("chilled_pump_primary", "pp"), ("chilled_pump_secondary", "sp")):
        for j in range(n_hvac):
            rated_kw = chiller_rated if kind == "chiller" else pump_budget / (3 * n_hvac)
            attributes["hvac"][f"{prefix}{j}"] = {
                "numeric": {"rated_power_kw": float(rated_kw),
                            "rated_flow_rate": float(rated_kw * 10),
                            "design_cooling_kw": float(1.2 * capacity / n_hvac)},
                "category": {"device_type": kind}}
    room_area = 200.0
    for k in range(n_room):
        it_kw = float(sum(rated[rack_members[j]].sum() for j in range(n_rack) if j % n_room == k))
        attributes["space"][f"rm{k}"] = {"numeric": {"area_m2": room_area, "it_capacity_kw": it_kw},
                                         "category": {"space_level": "room"}}
    for b in range(n_building):
        it_kw = float(sum(attributes["space"][f"rm{k}"]["numeric"]["it_capacity_kw"]
                          for k in range(n_room) if k % n_building == b))
        attributes["space"][f"bd{b}"] = {"numeric": {"area_m2": 500.0, "it_capacity_kw": it_kw},
                                         "category": {"space_level": "building"}}
    attributes["space"]["site"] = {"numeric": {"area_m2": 500.0 * n_building,
                                               "it_capacity_kw": float(capacity)},
                                   "category": {"space_level": "aidc_boundary"}}
    attributes["environment"]["env0"] = {"numeric": {}, "category": {"station_type": "outdoor_station"}}
    total_steps = AIDC_PREWARM + AIDC_STEPS
    edges = [{"src_group": "space", "src_id": "site", "dst_group": "space", "dst_id": f"bd{b}",
              "type": "contains", "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
             for b in range(n_building)]
    edges += [{"src_group": "space", "src_id": f"bd{b}", "dst_group": "space", "dst_id": f"rm{k}",
               "type": "contains", "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
              for k in range(n_room) for b in [k % n_building]]
    edges += [{"src_group": "space", "src_id": f"rm{k}", "dst_group": "it_power", "dst_id": f"r{j}",
               "type": "contains", "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
              for j in range(n_rack) for k in [j % n_room]]
    # 供电链（supplies DAG）：变压器低压侧(→母线)→UPS 输入→UPS 输出→机柜
    for tag in ("A", "B")[: 2 if dual else 1]:
        edges.append({"src_group": "distribution", "src_id": "bus0" if deep else "tr0",
                      "dst_group": "distribution", "dst_id": f"ups{tag}_in", "type": "supplies",
                      "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0})
        edges.append({"src_group": "distribution", "src_id": f"ups{tag}_in",
                      "dst_group": "distribution", "dst_id": f"ups{tag}_out", "type": "supplies",
                      "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0})
    if deep:
        edges.append({"src_group": "distribution", "src_id": "tr0", "dst_group": "distribution",
                      "dst_id": "bus0", "type": "supplies", "valid_from": 0, "valid_to": None,
                      "available_at": 0, "revision": 0})
    ups_tags = ("A", "B")[: 2 if dual else 1]
    for j in range(n_rack):
        for tag in ups_tags:
            edges.append({"src_group": "distribution", "src_id": f"ups{tag}_out",
                          "dst_group": "it_power", "dst_id": f"r{j}", "type": "supplies",
                          "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0})
    # 计量覆盖（meter_covers 层级树）
    edges += [{"src_group": "space", "src_id": "site", "dst_group": "space", "dst_id": f"bd{b}",
               "type": "meter_covers", "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
              for b in range(n_building)]
    edges += [{"src_group": "space", "src_id": f"bd{b}", "dst_group": "space", "dst_id": f"rm{k}",
               "type": "meter_covers", "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
              for k in range(n_room) for b in [k % n_building]]
    edges += [{"src_group": "space", "src_id": f"rm{k}", "dst_group": "it_power", "dst_id": f"r{j}",
               "type": "meter_covers", "valid_from": 0, "valid_to": None, "available_at": 0, "revision": 0}
              for j in range(n_rack) for k in [j % n_room]]
    # 冷却配对（cools 多对多；站点 11 交换配对）
    for j in range(n_hvac):
        room = (j + 1) % n_room if site == 11 else j % n_room
        edges.append({"src_group": "hvac", "src_id": f"ch{j}", "dst_group": "space",
                      "dst_id": f"rm{room}", "type": "cools", "valid_from": 0, "valid_to": None,
                      "available_at": 0, "revision": 0})
        edges.append({"src_group": "hvac", "src_id": f"ct{j}", "dst_group": "hvac",
                      "dst_id": f"ch{j}", "type": "cools", "valid_from": 0, "valid_to": None,
                      "available_at": 0, "revision": 0})
    edges += [{"src_group": "compute", "src_id": "pool_inference", "dst_group": "it_power",
               "dst_id": f"s{i}", "type": "runs_on", "valid_from": 0, "valid_to": None,
               "available_at": 0, "revision": 0} for i in range(n_server) if i % 2 == 0]
    edges += [{"src_group": "compute", "src_id": "pool_training", "dst_group": "it_power",
               "dst_id": f"s{i}", "type": "runs_on", "valid_from": 0, "valid_to": None,
               "available_at": 0, "revision": 0} for i in range(n_server) if i % 2 == 1]
    edges.append({"src_group": "environment", "src_id": "env0", "dst_group": "space",
                  "dst_id": "site", "type": "associated_env", "valid_from": 0, "valid_to": None,
                  "available_at": 0, "revision": 0})
    params = {"n_server": n_server, "n_rack": n_rack, "n_room": n_room, "n_building": n_building,
              "n_hvac": n_hvac, "dual": dual, "deep": deep, "rated": rated, "idle_ratio": idle_ratio,
              "alpha": alpha, "capacity": capacity, "eta_ups": eta_ups, "eta_tr": eta_tr,
              "rack_members": rack_members, "ups_tags": ups_tags,
              "chiller_rated": chiller_rated, "total_steps": total_steps}
    return order, attributes, edges, params


def aidc_task_blocks(rng, total_steps):
    """§18.6 训练计划／执行偏差／突发：返回两池的块表与站点级序列。"""
    pools = {}
    for pool in ("inference", "training"):
        blocks = []
        for block_start in range(0, total_steps, 12):
            start = block_start + int(rng.integers(0, 3))
            duration = int(rng.integers(3, 7))
            extra = float(rng.uniform(0.10, 0.35))
            delay = int(rng.integers(0, 3))
            exec_duration = max(1, duration - int(rng.binomial(1, 0.2)))
            revised = pool == "training" and bool(rng.binomial(1, 0.2))
            extra1 = extra * 1.2
            blocks.append({"start": start, "duration": duration, "extra": extra,
                           "pub0": start - 24, "exec_start": start + delay,
                           "exec_duration": exec_duration,
                           "revised": revised, "extra1": extra1, "pub1": start - 12})
        pools[pool] = blocks
    return pools


def aidc_physics(params, task_rng, env_rng):
    """§18.2–§18.5 顺序仿真：服务器→配电→热与暖通→总边界；返回真值序列与任务观测。"""
    total = params["total_steps"]
    t = np.arange(total, dtype=np.float64)
    # 背景、任务计划与执行（两池独立块；突发独立流）
    phase = float(task_rng.uniform(0, 2 * np.pi))
    base = 0.30 + 0.12 * np.sin(2 * np.pi * t / 24 + phase)
    # 块表多生成 72 步：无标签请求的 origin 未来计划仍可发布（§17 顺序 6）
    pools = aidc_task_blocks(task_rng, total + AIDC_MAX_KNOWN_STEPS)
    plan, actual, burst = {}, {}, {}
    for pool, blocks in pools.items():
        plan_series = np.zeros(total)
        actual_series = np.zeros(total)
        for block in blocks:
            plan_series[block["start"]:block["start"] + block["duration"]] += block["extra"]
            lo = block["exec_start"]
            actual_series[lo:lo + block["exec_duration"]] += block["extra"]
        hits = task_rng.binomial(1, 0.05, total).astype(bool)
        amplitude = task_rng.uniform(0.1, 0.3, total)
        burst[pool] = np.where(hits, amplitude, 0.0)
        plan[pool], actual[pool] = plan_series, actual_series
    epsilon = task_rng.uniform(-0.03, 0.03, (total, params["n_server"]))
    pool_index = np.array([i % 2 for i in range(params["n_server"])])
    plan_server = np.stack([plan["inference"], plan["training"]], axis=1)[:, pool_index]
    actual_server = np.stack([actual["inference"], actual["training"]], axis=1)[:, pool_index]
    burst_server = np.stack([burst["inference"], burst["training"]], axis=1)[:, pool_index]
    utilization = np.clip(base[:, None] + actual_server + burst_server + epsilon, 0, 1)
    idle = params["rated"] * params["idle_ratio"]
    p_server = idle[None, :] + (params["rated"] - idle)[None, :] * utilization ** params["alpha"][None, :]
    p_it = p_server.sum(axis=1)
    # 环境天气与扰动
    phase_env = float(env_rng.uniform(0, 2 * np.pi))
    shock = np.zeros(total)
    for step in range(1, total):
        shock[step] = 0.85 * shock[step - 1] + env_rng.uniform(-0.5, 0.5)
    t_out = 24 + 6 * np.sin(2 * np.pi * t / 24 + phase_env) + shock
    humidity = np.clip(0.55 + 0.1 * np.sin(2 * np.pi * t / 24), 0, 1)
    # 热惯性（§18.4）：站点级单温度，参数由 C 派生
    capacity = params["capacity"]
    ua, kp, cth = 0.1 * capacity, 0.5 * capacity, 8 * capacity
    q0, qmax, tset = 0.6 * capacity, 1.2 * capacity, 24.0
    stability = (ua + kp) / cth
    assert 0 < stability < 1
    temperature = np.zeros(total)
    temperature[0] = tset
    q_cool = np.zeros(total)
    for step in range(total):
        q_cool[step] = np.clip(q0 + kp * (temperature[step] - tset), 0, qmax)
        cooling = q_cool[step]
        if step + 1 < total:
            temperature[step + 1] = temperature[step] + (
                p_it[step] + ua * (t_out[step] - temperature[step]) - cooling) / cth
    cop = np.clip(4.5 - 0.06 * (t_out - 20) - 0.5 * (1 - q_cool / qmax) ** 2, 2, 6)
    p_chiller_total = q_cool / cop
    speed = np.clip(q_cool / qmax, 0, 1)
    p_pump_tower_total = 0.01 * capacity + 0.04 * capacity * speed ** 3
    p_hvac = p_chiller_total + p_pump_tower_total
    p_aux = 0.03 * capacity * (1 + 0.1 * np.sin(2 * np.pi * t / 24))
    it_upstream = p_it * 1.01 / params["eta_ups"]
    tr_low = it_upstream + p_hvac + p_aux
    target = tr_low / params["eta_tr"]
    loss = target - (p_it + p_hvac + p_aux)
    assert (loss > -1e-9).all()
    # 任务观测（§18.6 任务观测行；先吞吐再更新 queue）
    compute_series = {}
    for index, pool in enumerate(("inference", "training")):
        members = [i for i in range(params["n_server"]) if i % 2 == index]
        cap = 8.0 * len(members)
        pool_util = utilization[:, members].mean(axis=1)
        allocated = cap * pool_util
        active = np.ceil(allocated / 2)
        arrivals = np.ceil(cap * (base + plan[pool]) / 2)
        service = np.floor(cap * pool_util / 2)
        queue = np.zeros(total)
        throughput = np.zeros(total)
        for step in range(total):
            throughput[step] = min(queue[step] + arrivals[step], service[step])
            if step + 1 < total:
                queue[step + 1] = max(0.0, queue[step] + arrivals[step] - service[step])
        compute_series[pool] = {"allocated": allocated, "active": active, "queue": queue,
                                "throughput": throughput, "arrivals": arrivals,
                                "cap": cap, "util": pool_util}
    return {"base": base, "plan": plan, "actual": actual, "utilization": utilization,
            "p_server": p_server, "p_it": p_it, "t_out": t_out, "humidity": humidity,
            "shock": shock, "phase_env": phase_env, "temperature": temperature,
            "q_cool": q_cool, "cop": cop, "speed": speed,
            "p_chiller_total": p_chiller_total, "p_pump_tower_total": p_pump_tower_total,
            "p_hvac": p_hvac, "p_aux": p_aux, "it_upstream": it_upstream, "tr_low": tr_low,
            "target": target, "loss": loss, "compute": compute_series,
            "stability": stability, "blocks": pools}


def aidc_measurements(params, physics, site):
    """从真值派生各组计量观测（§18.6 计量行）；空间归属用 cools 配对。"""
    rack_members, n_rack = params["rack_members"], params["n_rack"]
    n_room, n_hvac, n_building = params["n_room"], params["n_hvac"], params["n_building"]
    p_rack = np.stack([physics["p_server"][:, members].sum(axis=1)
                       for members in rack_members.values()], axis=1)
    rack_util = np.stack([physics["utilization"][:, members].mean(axis=1)
                          for members in rack_members.values()], axis=1)
    tags = params["ups_tags"]
    share = 1.0 / len(tags)
    ups_out = {tag: physics["p_it"] * 1.01 * share for tag in tags}
    ups_in = {tag: ups_out[tag] / params["eta_ups"] for tag in tags}
    # 归属矩阵：行=下级索引，列=上级索引，按索引轮转分配
    rack_to_room = (np.arange(n_rack)[:, None] % n_room == np.arange(n_room)[None, :]).astype(float)
    room_it = p_rack @ rack_to_room
    chiller_power = np.repeat(physics["p_chiller_total"][:, None] / n_hvac, n_hvac, axis=1)
    pump_power = np.repeat(physics["p_pump_tower_total"][:, None] / (3 * n_hvac), 3 * n_hvac, axis=1)
    hvac_power = np.concatenate([chiller_power, pump_power], axis=1)  # 对应 order: ch,ct,pp,sp
    # 暖通按 cools 配对归属机房（站点 11 交换配对）
    hvac_room_power = np.zeros((len(p_rack), n_room))
    for j in range(n_hvac):
        room = (j + 1) % n_room if site == 11 else j % n_room
        hvac_room_power[:, room] += (chiller_power[:, j] + pump_power[:, j]
                                     + pump_power[:, n_hvac + j] + pump_power[:, 2 * n_hvac + j])
    room_to_building = (np.arange(n_room)[:, None] % n_building
                        == np.arange(n_building)[None, :]).astype(float)
    building_power = (room_it + hvac_room_power) @ room_to_building
    return {"p_rack": p_rack, "rack_util": rack_util, "ups_out": ups_out, "ups_in": ups_in,
            "room_it": room_it, "hvac_power": hvac_power, "chiller_power": chiller_power,
            "pump_power": pump_power, "building_power": building_power}


def aidc_attributes_records(attributes, order):
    """attributes {group: {id: {numeric, category}}} → entities.json 记录。"""
    document = {"groups": {}}
    for group, ids in order.items():
        rows = []
        for name in ids:
            rows.append({"id": name, "valid_from": 0, "valid_to": None, "available_at": 0,
                         "revision": 0, "numeric": dict(attributes[group][name]["numeric"]),
                         "category": dict(attributes[group][name]["category"])})
        document["groups"][group] = rows
    return document


def aidc_episode_arrays(order, attributes, params, physics, meters, site, scenario,
                        noise_rng, missing_rng):
    """组装单站点单场景的 observations（预热后 1440 步）并施加观测腐化（§19.2）。"""
    prewarm = AIDC_PREWARM
    steps = AIDC_STEPS
    arrays = episode_arrays({"groups": AIDC_GROUPS}, order, steps)
    values = {}
    # compute：[allocated, active, queue, throughput, arrivals]
    compute = np.zeros((steps, 2, 5))
    for index, pool in enumerate(("inference", "training")):
        series = physics["compute"][pool]
        compute[:, index, :] = np.stack([series["allocated"], series["active"], series["queue"],
                                         series["throughput"], series["arrivals"]], axis=1)[prewarm:]
    values["compute"] = compute
    # it_power：servers + racks → [power, cpu_util, gpu_util, online_fraction]
    n_server, n_rack = params["n_server"], params["n_rack"]
    it = np.zeros((steps, n_server + n_rack, 4))
    it[:, :n_server, 0] = physics["p_server"][prewarm:]
    it[:, :n_server, 1] = physics["utilization"][prewarm:]
    it[:, :n_server, 2] = physics["utilization"][prewarm:]
    it[:, :n_server, 3] = 1.0
    it[:, n_server:, 0] = meters["p_rack"][prewarm:]
    it[:, n_server:, 1] = meters["rack_util"][prewarm:]
    it[:, n_server:, 2] = meters["rack_util"][prewarm:]
    it[:, n_server:, 3] = 1.0
    values["it_power"] = it
    # distribution：按 order 顺序 [bus0?] tr0 ups{tag}_in/out
    rows = []
    if params["deep"]:
        rows.append((physics["tr_low"][prewarm:], attributes["distribution"]["bus0"]["numeric"]["rated_capacity_kw"]))
    rows.append((physics["tr_low"][prewarm:], attributes["distribution"]["tr0"]["numeric"]["rated_capacity_kw"]))
    for tag in params["ups_tags"]:
        rows.append((meters["ups_in"][tag][prewarm:], attributes["distribution"][f"ups{tag}_in"]["numeric"]["rated_capacity_kw"]))
        rows.append((meters["ups_out"][tag][prewarm:], attributes["distribution"][f"ups{tag}_out"]["numeric"]["rated_capacity_kw"]))
    dist = np.zeros((steps, len(rows), 2))
    for index, (power, rated) in enumerate(rows):
        dist[:, index, 0] = power
        dist[:, index, 1] = power / rated
    values["distribution"] = dist
    # hvac：[power, running, frequency, flow, supply, return]，order = ch,ct,pp,sp
    speed = physics["speed"][prewarm:]
    supply = 7 + 0.02 * (physics["t_out"][prewarm:] - 20)
    hvac_ids = order["hvac"]
    hvac = np.zeros((steps, len(hvac_ids), 6))
    hvac[:, :, 0] = meters["hvac_power"][prewarm:]
    hvac[:, :, 1] = 1.0
    hvac[:, :, 2] = 50 * speed[:, None]
    for index, name in enumerate(hvac_ids):
        rated_flow = attributes["hvac"][name]["numeric"]["rated_flow_rate"]
        hvac[:, index, 3] = rated_flow * speed
    hvac[:, :, 4] = supply[:, None]
    hvac[:, :, 5] = (supply + 5 * speed)[:, None]
    values["hvac"] = hvac
    # space：order = site, buildings, rooms → [power, room_temp, room_humidity]
    space_ids = order["space"]
    space = np.zeros((steps, len(space_ids), 3))
    space[:, 0, 0] = physics["target"][prewarm:]
    space[:, 1:params["n_building"] + 1, 0] = meters["building_power"][prewarm:]
    space[:, params["n_building"] + 1:, 0] = meters["room_it"][prewarm:]
    space[:, :, 1] = physics["temperature"][prewarm:, None]
    space[:, :, 2] = 0.5
    values["space"] = space
    values["environment"] = np.stack([physics["t_out"][prewarm:], physics["humidity"][prewarm:]], axis=1)[:, None, :]
    arrays["target"][:, 0] = physics["target"][prewarm:].astype(np.float32)
    # ----------------------------------------------------------- 观测腐化 ----
    missing_rate = (0.0, 0.05, 0.2)[site % 3]
    offline_hours = (6, 24)[site % 2]
    delay = (0, 1, 3)[site % 3]
    power_columns = {"it_power": 0, "distribution": 0, "hvac": 0, "space": 0}
    for group, tensor in values.items():
        shape = (steps,) + tensor.shape[1:]
        mask = rng_random(missing_rng, shape) >= missing_rate
        if group == "it_power":
            # 首个服务器与首个机柜测点每 168 小时连续离线（§19.2）
            cycle = np.arange(steps) % 168 < offline_hours
            mask[cycle, 0, :] = False
            mask[cycle, n_server, :] = False
        if scenario == "sparse" and group == "it_power":
            mask[:, :n_server, :] = False  # 稀疏视图：屏蔽服务器动态观测，保留已知资产
        arrays[f"{group}__values"] = tensor.astype(np.float32)
        if group in power_columns:
            column = power_columns[group]
            noise = 1 + rng_random(noise_rng, tensor[..., column].shape, low=-0.01, high=0.01)
            tensor[..., column] = tensor[..., column] * noise  # 只腐化观测，不碰 target
            arrays[f"{group}__values"] = tensor.astype(np.float32)
        apply_observed(arrays, group, mask)
        if delay:
            arrays[f"{group}__available_at"] += delay * HOUR
    return arrays


def rng_random(rng, shape, low=0.0, high=1.0):
    return rng.uniform(low, high, shape)


def aidc_visible_plan(blocks, origins_axis):
    """按块逐层核算站点级可见计划（§19.2 计划修订）：known 展开与 oracle 共用。

    返回 (value, published, revision, has)，形状 [n_origins, 72]；origin 为主轴索引。
    修订块在 pub1 后贡献 extra1；未发布块不贡献；两池按 2 平均（缺块池计 0）。
    """
    count = len(origins_axis)
    horizon = AIDC_MAX_KNOWN_STEPS
    origin_phys = origins_axis + AIDC_PREWARM
    target_axis = origins_axis[:, None] + 1 + np.arange(horizon)[None, :]
    total_sum = np.zeros((count, horizon))
    total_pub = np.full((count, horizon), -np.inf)
    any_has = np.zeros((count, horizon), bool)
    any_v1 = np.zeros((count, horizon), bool)
    for pool in ("inference", "training"):
        for block in blocks[pool]:
            low = block["start"] - AIDC_PREWARM
            high = block["start"] + block["duration"] - AIDC_PREWARM
            covered = (target_axis >= low) & (target_axis < high)
            visible = covered & (origin_phys[:, None] >= block["pub0"])
            contribution = np.where(visible, block["extra"], 0.0)
            published = np.where(visible, block["pub0"], -np.inf)
            if block["revised"]:
                revised_visible = visible & (origin_phys[:, None] >= block["pub1"])
                contribution = np.where(revised_visible, block["extra1"], contribution)
                published = np.where(revised_visible, block["pub1"], published)
                any_v1 |= revised_visible
            total_sum += contribution
            total_pub = np.maximum(total_pub, published)
            any_has |= visible
    return total_sum / 2, total_pub, any_v1, any_has


def aidc_known_records(params, physics):
    """站点级未来条件（§18.6/§19.2）：计划版本化 + 起点状态预报。"""
    origins = np.arange(47, AIDC_STEPS)  # 主轴 origin（含无标签请求的末尾 origin）
    steps_axis = np.arange(1, AIDC_MAX_KNOWN_STEPS + 1)
    target_index = origins[:, None] + steps_axis[None, :]  # 主轴目标步
    plan_value_matrix, plan_pub_matrix, plan_v1_matrix, plan_has = aidc_visible_plan(
        physics["blocks"], origins)
    selected = plan_has & (plan_pub_matrix > -np.inf)
    rows, cols = np.nonzero(selected)
    plan_origin = origins[rows]
    plan_target = target_index[rows, cols]
    # 块表发布时间是物理轴索引；落盘绝对时间须转主轴（pub0 可为负，早于时间轴起点合法）
    plan_pub = (plan_pub_matrix[rows, cols] - AIDC_PREWARM).astype(np.int64)
    plan_value = plan_value_matrix[rows, cols].astype(np.float32)
    plan_revision = plan_v1_matrix[rows, cols].astype(np.int64)
    # 预报：只依赖起点状态（§18.6 天气行）
    shock = physics["shock"][AIDC_PREWARM + origins]
    k_axis = steps_axis[None, :]
    forecast_value = (24 + 6 * np.sin(2 * np.pi * (origins[:, None] + k_axis) / 24
                                      + physics["phase_env"])
                      + np.power(0.85, k_axis) * shock[:, None]).astype(np.float32)
    forecast_count = forecast_value.size
    known = {
        "origin": np.concatenate([EPOCH + plan_origin * HOUR,
                                  np.repeat(EPOCH + origins * HOUR, AIDC_MAX_KNOWN_STEPS)]).astype(np.int64),
        "target_time": np.concatenate([
            EPOCH + plan_target * HOUR,
            (EPOCH + (origins[:, None] + k_axis) * HOUR).ravel()]).astype(np.int64),
        "feature_index": np.concatenate([np.zeros(len(rows), np.int64),
                                         np.ones(forecast_count, np.int64)]),
        "value": np.concatenate([plan_value, forecast_value.ravel()]).astype(np.float32),
        "observed": np.ones(len(rows) + forecast_count, bool),
        "published_at": np.concatenate([EPOCH + plan_pub * HOUR,
                                        np.repeat(EPOCH + origins * HOUR, AIDC_MAX_KNOWN_STEPS)]).astype(np.int64),
        "revision": np.concatenate([plan_revision,
                                    np.zeros(forecast_count, np.int64)]),
    }
    oracle_arrays = {"blocks": physics["blocks"], "origins": origins}
    return known, oracle_arrays


def aidc_schema_block():
    return {"feature_units": dict(AIDC_FEATURE_UNITS), "target_quantity": "facility_demand",
            "interval_label": "right", "target_aggregation": "time_weighted_mean"}


def aidc_site_contract(site):
    return {"target_id": f"aidc_{site:02d}_main", "quantity": "facility_demand",
            "boundary_id": f"aidc_{site:02d}", "unit": "kW",
            "sign_convention": "positive_consumption", "interval_label": "right",
            "aggregation": "time_weighted_mean", "meter_ids": [f"aidc_{site:02d}_main"],
            "coverage_description": "synthetic full-facility boundary"}


AIDC_PROFILES = {"aidc_hourly_short": (48, 6), "aidc_hourly_day": (168, 24),
                 "aidc_hourly_multiday": (168, 72)}


def build_aidc(output):
    """生成 12 站 dense/sparse collection、三 profile 配置、审计与无标签请求。"""
    root = output / "aidc"
    root.mkdir(parents=True, exist_ok=True)
    entries, audit = [], {"generator_version": "aidc-v1", "seed": 2023, "sites": {}, "oracle": {}}
    known_oracles = {}
    for site in range(AIDC_SITES):
        order, attributes, edges, params = aidc_topology(site)
        task_rng = np.random.default_rng(np.random.SeedSequence([2023, 6, site, 1]))
        env_rng = np.random.default_rng(np.random.SeedSequence([2023, 6, site, 2]))
        noise_rng = np.random.default_rng(np.random.SeedSequence([2023, 6, site, 3]))
        missing_rng = np.random.default_rng(np.random.SeedSequence([2023, 6, site, 4]))
        physics = aidc_physics(params, task_rng, env_rng)
        meters = aidc_measurements(params, physics, site)
        known, oracle_arrays = aidc_known_records(params, physics)
        known_oracles[site] = oracle_arrays
        schema = {"timezone": "UTC", "freq_seconds": 3600, "groups": deepcopy(AIDC_GROUPS),
                  "relation_types": deepcopy(AIDC_RELATIONS), "known_future": list(AIDC_KNOWN),
                  "target_name": "facility_demand_kw", "target_unit": "kW", **aidc_schema_block()}
        last_arrays = None
        for scenario in ("dense", "sparse"):
            arrays = aidc_episode_arrays(order, attributes, params, physics, meters, site,
                                         scenario, noise_rng, missing_rng)
            episode_id = f"aidc_{site:02d}_{scenario}"
            write_episode(root / episode_id, schema, order, arrays,
                          aidc_attributes_records(attributes, order), edges,
                          {"site_id": f"aidc_{site:02d}", "target_contract": aidc_site_contract(site)},
                          primary_mappings(order, schema, f"aidc{site:02d}", AIDC_FEATURE_UNITS), known,
                          family_id=f"aidc_{site:02d}", scenario=scenario)
            entries.append({"id": episode_id, "family_id": f"aidc_{site:02d}",
                            "site_id": f"aidc_{site:02d}", "split": None, "path": episode_id,
                            "topology_family_id": f"topo_{site % 3}_{site % 2}"})
            if scenario == "dense":
                last_arrays = arrays
        if site in (0, 10, 11):
            # 无标签请求：dense 视图最后 L_max=168 步，origin 为最后观测时刻（§19.3）
            request = root / "requests" / f"aidc_{site:02d}_forecast"
            tail = {key: value[-168:].copy() for key, value in last_arrays.items()}
            stamp = EPOCH + 1439 * HOUR
            request_known = {key: value[known["origin"] == stamp].copy() for key, value in known.items()}
            write_episode(request, schema, order, tail,
                          aidc_attributes_records(attributes, order), edges,
                          {"site_id": f"aidc_{site:02d}", "target_contract": aidc_site_contract(site)},
                          primary_mappings(order, schema, f"aidc{site:02d}", AIDC_FEATURE_UNITS),
                          request_known, purpose="forecast", family_id=f"aidc_{site:02d}",
                          scenario="forecast_request")
        audit["sites"][f"aidc_{site:02d}"] = {
            "entities": sum(len(ids) for ids in order.values()),
            "servers": params["n_server"], "racks": params["n_rack"], "rooms": params["n_room"],
            "buildings": params["n_building"], "hvac_per_type": params["n_hvac"],
            "dual_path": params["dual"], "deep_chain": params["deep"],
            "capacity_kw": params["capacity"], "eta_ups": params["eta_ups"],
            "eta_transformer": params["eta_tr"], "stability": physics["stability"],
            "temperature_range": [float(physics["temperature"][AIDC_PREWARM:].min()),
                                  float(physics["temperature"][AIDC_PREWARM:].max())],
            "loss_min": float(physics["loss"].min()),
            "target_mean_kw": float(physics["target"][AIDC_PREWARM:].mean()),
            "missing_rate": (0.0, 0.05, 0.2)[site % 3], "offline_hours": (6, 24)[site % 2],
            "delay_steps": (0, 1, 3)[site % 3],
        }
    schema = {"timezone": "UTC", "freq_seconds": 3600, "groups": deepcopy(AIDC_GROUPS),
              "relation_types": deepcopy(AIDC_RELATIONS), "known_future": list(AIDC_KNOWN),
              "target_name": "facility_demand_kw", "target_unit": "kW", **aidc_schema_block()}
    write_collection(root, schema, entries, "site_time", AIDC_SPLIT, site_split=AIDC_SITE_SPLIT)
    for name, (seq, horizon) in AIDC_PROFILES.items():
        config = full_config(AIDC_GROUPS, AIDC_RELATIONS, AIDC_KNOWN, package=".",
                             split_mode="site_time", split_index=AIDC_SPLIT,
                             seq_len=seq, pred_len=horizon, target="facility_demand_kw",
                             unit="kW", batch_size=2, valid_stride=horizon,
                             test_stride=horizon, limits=(168, 72),
                             extra_schema={"sampler": "uniform_site_then_origin",
                                           "site_split": deepcopy(AIDC_SITE_SPLIT),
                                           **aidc_schema_block()})
        _write_json(root / f"{name}.json", config)
    _write_json(root / "generation_audit.json", audit)
    _write_json(root / "generation_audit_oracle.json",
                {site: {"revised_blocks": sum(1 for b in arrays["blocks"]["training"] if b["revised"])}
                 for site, arrays in known_oracles.items()})
    _verify_aidc_oracles(root, known_oracles)
    return root


def _verify_aidc_oracles(root, known_oracles):
    """生成后 oracle：计划修订 as-of 可见性与 dense/sparse target 一致（§17/§19.2）。"""
    from data_provider.spatial_temporal.dataset import EpisodePackage
    results = {}
    for site in range(AIDC_SITES):
        arrays = known_oracles[site]
        config = full_config(AIDC_GROUPS, AIDC_RELATIONS, AIDC_KNOWN,
                             package=str(root / f"aidc_{site:02d}_dense"),
                             split_mode="site_time", split_index=AIDC_SPLIT,
                             seq_len=48, pred_len=6, target="facility_demand_kw", unit="kW",
                             batch_size=2, valid_stride=6, test_stride=6, limits=(168, 72),
                             extra_schema={"sampler": "uniform_site_then_origin",
                                           "site_split": deepcopy(AIDC_SITE_SPLIT),
                                           **aidc_schema_block()})
        pkg = EpisodePackage(root / f"aidc_{site:02d}_dense", config)
        # 计划修订 as-of：origin 在 pub1 前只见 v0，到达 pub1 见 v1（与 known 同口径核算）
        blocks = arrays["blocks"]
        asof = None
        for block in blocks["training"]:
            if not block["revised"]:
                continue
            axis = block["start"] - AIDC_PREWARM  # 修订块覆盖的主轴起点
            o2 = block["pub1"] - AIDC_PREWARM
            o1 = o2 - 1
            if o1 < 47 or axis <= o1 + 1:
                continue
            values, _, revisions, _ = aidc_visible_plan(blocks, np.array([o1, o2]))
            expectations = [values[0, axis - o1 - 1], values[1, axis - o2 - 1]]
            for origin_axis, expect, revision in zip((o1, o2), expectations,
                                                      (revisions[0, axis - o1 - 1],
                                                       revisions[1, axis - o2 - 1])):
                origin_stamp = EPOCH + origin_axis * HOUR
                future_time = EPOCH + np.arange(origin_axis + 1, origin_axis + 73) * HOUR
                picked, _ = pkg.select_known(origin_stamp, future_time)
                if abs(float(picked[axis - origin_axis - 1, 0]) - float(expect)) > 1e-5:
                    raise ValueError(f"aidc_{site:02d} as-of oracle 值不符: origin={origin_axis}")
            if not revisions[1, axis - o2 - 1] or revisions[0, axis - o1 - 1]:
                raise ValueError(f"aidc_{site:02d} as-of 修订标记不符")
            asof = {"origin_v0": o1, "origin_v1": o2, "revised_target_axis": axis}
            break
        # dense/sparse target 一致（同一物理过程的两个观测视图）
        dense = np.load(root / f"aidc_{site:02d}_dense" / "observations.npz")["target"]
        sparse = np.load(root / f"aidc_{site:02d}_sparse" / "observations.npz")["target"]
        if not np.array_equal(dense, sparse):
            raise ValueError(f"aidc_{site:02d} dense/sparse target 不一致")
        results[f"aidc_{site:02d}"] = {"asof_revision_check": asof, "target_identical": True}
    _write_json(root / "oracle_results.json", results)


# ==============================================================================
# base 的无标签请求包（§7：新增实体有／无历史、新增父子节点）
# ==============================================================================

def build_requests(output):
    """base 同分布 forecast 请求：normal / 新实体冷启动 / 新实体有历史 / 新父子节点。"""
    root = output / "requests"
    length, history = 30, 24  # 请求只含历史与原点已知条件；origin = 索引 29
    offset = BASE_LENGTH - length  # 与 base 尾部时间轴对齐
    rng = np.random.default_rng(2023)
    order = {"assets": [f"a{i}" for i in range(6)], "zones": ["z0", "z1"], "sites": ["s0"]}
    config = full_config(BASE_GROUPS, BASE_RELATIONS, ["scheduled_activity"], package=".",
                         split_mode="target_index", split_index=[0, 24, 27, 30],
                         seq_len=24, pred_len=6)
    schema = schema_from_config(config)

    def new_entity_series(prefix_index):
        """同公式新相位（§7：新相位不与训练实体重合）。"""
        t = np.arange(length, dtype=np.float64) + offset
        scheduled = (1 + np.sin(2 * np.pi * t / 24)) / 2
        utilization = (1 + np.sin(2 * np.pi * t / 24 + prefix_index)) / 2
        load = 1.6 * (1 + utilization) + 0.1 * scheduled
        return load

    def make_request(name, *, extra_asset=None, extra_zone=None, observed_all=True):
        path = root / name
        # 请求的未来 6 步超出 base 主轴，公式序列延长 H 以提供计划值
        scheduled, capacity, utilization, load, temperature, target = base_series(BASE_LENGTH + 6)
        local = slice(offset, offset + length)
        this_order = {group: list(ids) for group, ids in order.items()}
        if extra_zone is not None:
            this_order["zones"] = this_order["zones"] + [extra_zone]
        if extra_asset is not None:
            this_order["assets"] = this_order["assets"] + [extra_asset["id"]]
        arrays = episode_arrays(schema, this_order, length)
        arrays["timestamps"] = (EPOCH + (offset + np.arange(length)) * HOUR)
        arrays["target_available_at"] = arrays["timestamps"][:, None].copy()
        for group in schema["groups"]:
            key = f"{group['name']}__available_at"
            arrays[key] = np.broadcast_to(arrays["timestamps"][:, None, None],
                                          arrays[key].shape).copy()
        arrays["target"][:, 0] = target[local]
        arrays["assets__values"][:, :6, 0] = load.T[local]
        arrays["assets__values"][:, :6, 1] = utilization.T[local]
        arrays["zones__values"][:, 0, 0] = temperature[0][local]
        arrays["zones__values"][:, 1, 0] = temperature[1][local]
        edges = [{"src_group": "assets", "src_id": f"a{i}", "dst_group": "zones",
                  "dst_id": f"z{i // 3}", "type": "asset_in_zone", "valid_from": 0,
                  "valid_to": None, "available_at": 0, "revision": 0} for i in range(6)]
        edges += [{"src_group": "zones", "src_id": f"z{j}", "dst_group": "sites",
                   "dst_id": "s0", "type": "zone_in_site", "valid_from": 0, "valid_to": None,
                   "available_at": 0, "revision": 0} for j in range(2)]
        numeric_special = {}
        category_special = {}
        if extra_asset is not None:
            index = -1
            arrays["assets__values"][:, index, 0] = extra_asset["load"]
            arrays["assets__values"][:, index, 1] = extra_asset["utilization"]
            arrays["target"][:, 0] += extra_asset["load"] * (1 + 0.1 * temperature[1][local])
            if not observed_all:
                observed = arrays["assets__observed"]
                observed[:, index, :] = False
                arrays["assets__observed"] = observed
                arrays["assets__values"][:, index, :] = 0
            numeric_special[("assets", extra_asset["id"])] = {"capacity": 1.6}
            category_special[("assets", extra_asset["id"])] = {"kind": "even"}
            edges.append({"src_group": "assets", "src_id": extra_asset["id"],
                          "dst_group": "zones", "dst_id": extra_asset["parent"],
                          "type": "asset_in_zone", "valid_from": 0, "valid_to": None,
                          "available_at": 0, "revision": 0})
        if extra_zone is not None:
            # 新空间对象无温度计：temperature 列不适用而非缺失（§7/§2.3 语义）
            applicable = arrays["zones__applicable"]
            applicable[-1, 0] = False
            arrays["zones__applicable"] = applicable
            arrays["zones__values"][:, -1, 0] = 0
            observed = arrays["zones__observed"]
            observed[:, -1, 0] = False
            arrays["zones__observed"] = observed
            numeric_special[("zones", extra_zone)] = {"capacity": 2.0}
            edges.append({"src_group": "zones", "src_id": extra_zone, "dst_group": "sites",
                          "dst_id": "s0", "type": "zone_in_site", "valid_from": 0,
                          "valid_to": None, "available_at": 0, "revision": 0})

        def numeric(group, name):
            if (group, name) in numeric_special:
                return numeric_special[(group, name)]
            if group == "assets":
                return {"capacity": 1 + int(name[1:]) / 10}
            if group == "zones":
                return {"capacity": float(int(name[1:]) + 1)}
            return {"capacity": 3.0}

        def category(group, name):
            if (group, name) in category_special:
                return category_special[(group, name)]
            return {"kind": "even" if int(name[1:]) % 2 == 0 else "odd"} if group == "assets" else {}
        # 计划：origin（索引 29）可见的未来 6 步，提前 24 小时发布
        origin_index = length - 1
        known = plan_records(origin_index, scheduled[offset + origin_index + 1:
                                                     offset + origin_index + 7],
                             publish_lead=24)
        known["origin"] = np.full(6, arrays["timestamps"][origin_index], np.int64)
        known["target_time"] = np.array(
            [arrays["timestamps"][origin_index] + (k + 1) * HOUR for k in range(6)], np.int64)
        known["published_at"] = np.full(6, arrays["timestamps"][origin_index] - 24 * HOUR, np.int64)
        write_episode(path, schema, this_order, arrays,
                      entity_records(this_order, numeric, category), edges,
                      {"site_id": "synthetic_site", "target_contract": target_contract()},
                      primary_mappings(this_order, schema, name, "arbitrary_unit"), known,
                      purpose="forecast", scenario=name)
        return path

    new_load = new_entity_series(5 / 3)
    new_utilization = (new_load - 0.1 * (1 + np.sin(2 * np.pi * (np.arange(length) + offset) / 24)) / 2) / 1.6 - 1
    requests = {
        "normal_forecast": make_request("normal_forecast"),
        "new_entity_cold": make_request("new_entity_cold",
                                        extra_asset={"id": "asset_new", "parent": "z1",
                                                     "load": new_load,
                                                     "utilization": np.maximum(new_utilization, 0)},
                                        observed_all=False),
        "new_entity_history": make_request("new_entity_history",
                                           extra_asset={"id": "asset_new", "parent": "z1",
                                                        "load": new_load,
                                                        "utilization": np.maximum(new_utilization, 0)}),
        "new_parent": make_request("new_parent",
                                   extra_asset={"id": "asset_new_parent", "parent": "zone_new",
                                                "load": new_load,
                                                "utilization": np.maximum(new_utilization, 0)},
                                   extra_zone="zone_new"),
    }
    return requests


# ==============================================================================
# 机制 suite 汇总
# ==============================================================================

def write_suite(output):
    """汇总机制任务清单与请求包（acceptance.md §8.1 suite.json）。"""
    tasks = []
    for task_id in ("M1", "M2", "M3", "M4"):
        task = output / "mechanisms" / task_id
        config = json.loads((task / "config.json").read_text())
        oracle = json.loads((task / "oracle.json").read_text())
        tasks.append({"id": task_id, "package": f"mechanisms/{task_id}",
                      "config": f"mechanisms/{task_id}/config.json",
                      "oracle": f"mechanisms/{task_id}/oracle.json",
                      "split": {"train": FAMILY_BOUNDS[0], "valid": FAMILY_BOUNDS[1] - FAMILY_BOUNDS[0],
                                "test": FAMILY_BOUNDS[2] - FAMILY_BOUNDS[1]},
                      "known_future": config["data"]["known_future"],
                      "assertions": oracle["oracle"]})
    tasks.append({"id": "M5", "kind": "data_contract",
                  "package": "mechanisms/M5", "suite": "mechanisms/M5/suite.json"})
    if (output / "aidc" / "manifest.json").exists():
        tasks.append({"id": "aidc", "kind": "multi_site",
                      "package": "aidc", "profiles": {name: f"aidc/{name}.json"
                                                      for name in AIDC_PROFILES},
                      "site_split": AIDC_SITE_SPLIT, "requests": {
                          f"aidc_{site:02d}": f"aidc/requests/aidc_{site:02d}_forecast"
                          for site in (0, 10, 11)},
                      "audit": "aidc/generation_audit.json",
                      "oracle": "aidc/oracle_results.json"})
    _write_json(output / "mechanisms" / "suite.json",
                {"generator_version": GENERATOR_VERSION, "seed": 2023, "tasks": tasks,
                 "requests": {name: f"requests/{name}"
                              for name in ("normal_forecast", "new_entity_cold",
                                           "new_entity_history", "new_parent")}})


# ==============================================================================
# 生成后复检与 CLI
# ==============================================================================

def verify_task(config_path):
    """加载 config 后对三个 split 真实构造 Dataset、取样并拟合训练统计。"""
    config = load_config(str(config_path))
    counts = {}
    for split in ("train", "valid", "test"):
        dataset = PackageDataset(config, split)
        counts[split] = len(dataset)
        dataset[0]
    TrainingTransform.fit(PackageDataset(config, "train"))
    return counts


def verify_request(path, config_path):
    """无标签请求包：只能以最后 origin 读取且不含监督。"""
    config = load_config(str(config_path))
    dataset = PackageDataset({**config, "data": {**config["data"], "package": str(path)}}, "forecast")
    sample = dataset[0]
    assert sample["supervision"] is None
    return {"origin_index": sample["metadata"]["origin_index"]}


def main():
    parser = argparse.ArgumentParser(description="生成 GSLAformer synthetic 标准包")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--suite", choices=("base", "mechanisms", "aidc", "all"), default="all")
    args = parser.parse_args()
    if args.seed != 2023:
        raise SystemExit("首轮固定 seed=2023（acceptance.md §8.1）")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    report = {"generator_version": GENERATOR_VERSION, "seed": args.seed, "tasks": {}}
    if args.suite in ("aidc", "all"):
        build_aidc(output)
        for profile in AIDC_PROFILES:
            counts = verify_task(output / "aidc" / f"{profile}.json")
            report["tasks"][f"aidc/{profile}"] = {"windows": counts}
            print(f"aidc/{profile} 完成: {counts}")
    if args.suite in ("base", "all"):
        build_base(output)
        report["tasks"]["base"] = {"windows": verify_task(output / "base" / "config.json")}
        print(f"base 完成: {report['tasks']['base']}")
    if args.suite in ("mechanisms", "all"):
        builders = {"M1": build_m1, "M2": build_m2, "M3": build_m3, "M4": build_m4}
        for task_id, builder in builders.items():
            builder(output / "mechanisms")
            report["tasks"][task_id] = {"windows": verify_task(output / "mechanisms" / task_id / "config.json")}
            print(f"{task_id} 完成: {report['tasks'][task_id]}")
        build_m5(output / "mechanisms")
        report["tasks"]["M5"] = {"kind": "data_contract",
                                 "fixtures": len(json.loads((output / "mechanisms" / "M5" / "suite.json").read_text())["fixtures"])}
        print(f"M5 完成: {report['tasks']['M5']}")
    if args.suite in ("base", "all"):
        requests = build_requests(output)
        report["requests"] = {name: verify_request(path, output / "base" / "config.json")
                              for name, path in requests.items()}
        print(f"请求包完成: {list(report['requests'])}")
    write_suite(output)
    _write_json(output / "generation_report.json", report)
    print(f"生成报告: {output / 'generation_report.json'}")


if __name__ == "__main__":
    main()
