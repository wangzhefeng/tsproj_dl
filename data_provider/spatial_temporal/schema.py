"""标准 JSON 指纹与输入质量；不读取标签、不维护可学习状态。"""
import hashlib
import json
from pathlib import Path

import numpy as np


def canonical_hash(value):
    """规范 JSON 的内容指纹；拒绝非有限 JSON 数值。"""
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"重复 JSON 键: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"非有限 JSON 数值: {value}")

    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream, object_pairs_hook=pairs, parse_constant=constant)


def orphan_entity_count(groups, relations_origin):
    """单请求 origin 的孤立有效实体数；padding／未生效对象不计入。"""
    active = {(name, index) for name, group in groups.items()
              for index in range(len(group["padding"]))
              if group["present"][-1, index] and not group["padding"][index]}
    connected = set()
    for sg, si, dg, di, _ in relations_origin:
        src, dst = (sg, si), (dg, di)
        if src in active and dst in active and src != dst:
            connected.update((src, dst))
    return len(active - connected)


def quality_flags(groups, target_observed, future_mask, *, no_history_entity_count=0,
                  unknown_category_count=0, orphan_count=0):
    """单请求质量；按适用单元计数，不将零列计划视为缺失计划。"""
    count = seen = 0
    for group in groups.values():
        observed = np.asarray(group["observed"], dtype=bool)
        applicable = np.asarray(group["applicable"], dtype=bool)
        present = np.asarray(group["present"], dtype=bool)
        padding = np.asarray(group["padding"], dtype=bool)
        if (observed.ndim != 3 or applicable.shape != observed.shape[1:]
                or present.shape != observed.shape[:2] or padding.shape != observed.shape[1:2]):
            raise ValueError("质量输入必须是单请求 [L,N,F] 及对齐的 masks")
        eligible = present[..., None] & ~padding[None, :, None] & applicable[None, ...]
        count += int(eligible.sum())
        seen += int((observed & eligible).sum())
    target = np.asarray(target_observed, dtype=bool)
    future = np.asarray(future_mask, dtype=bool)
    if target.ndim != 2 or target.shape[1] != 1 or len(target) == 0:
        raise ValueError("target_observed 必须为非空 [L,1]")
    if future.ndim != 2 or future.shape[0] == 0:
        raise ValueError("future_mask 必须为非空 horizon 的 [H,K]")
    target_coverage = float(target.mean())
    plan_coverage = float(future.mean()) if future.shape[1] else None
    return {
        "input_coverage": seen / count if count else 0.0,
        "target_history_coverage": target_coverage,
        "plan_available": bool(future.any()),
        "plan_coverage": plan_coverage,
        "degraded_input": bool(not count or seen < count or target_coverage < 1
                               or no_history_entity_count or unknown_category_count or orphan_count
                               or (plan_coverage is not None and plan_coverage < 1)),
    }
