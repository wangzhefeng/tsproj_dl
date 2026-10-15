"""GSLAformer 模式产物完成门禁与统一测试指标；不参与训练选模。"""
import csv
from pathlib import Path

import numpy as np
import torch

from data_provider.spatial_temporal.schema import read_json, canonical_hash


def synchronize(device):
    if str(device).startswith("mps"):
        torch.mps.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


def request_audit(inputs, config):
    roles = config["data"].get("site_split")
    role = (next((name for name, sites in roles.items() if inputs["site_id"] in sites), "unseen")
            if roles else "single_site")
    return {"site_role": role, "input_cutoff": int(inputs["origin"]),
            "relation_fingerprint": canonical_hash({"history": inputs["relations_history"],
                "origin": inputs["relations_origin"],
                "entity_ids": {name: group["entity_ids"] for name, group in inputs["groups"].items()}})}


def validate_checkpoint_contract(saved, model=None):
    from models.transformer.GSLAformer import Model
    if saved.get("version") != 2 or saved.get("computation_version") != Model.COMPUTATION_VERSION:
        raise ValueError("checkpoint 计算版本不兼容；旧权重保留历史用途，须重新训练，不自动迁移")
    names = saved.get("frozen_parameters")
    if not isinstance(names, list):
        raise ValueError("checkpoint 缺冻结参数清单")
    if model is not None:
        expected = sorted(name for name, p in model.named_parameters() if not p.requires_grad) if model._representation_frozen else []
        if names != expected:
            raise ValueError("checkpoint 冻结参数清单不一致")
    needs_stats = saved["phase"].startswith("B-") and saved["config"]["experiment"]["variant"] != "total_only"
    if needs_stats:
        manifest = saved.get("latent_stats_manifest")
        if not isinstance(manifest, dict):
            raise ValueError("checkpoint 缺 latent 统计溯源")
        unsigned = {k: v for k, v in manifest.items() if k != "fingerprint"}
        if manifest.get("fingerprint") != canonical_hash(unsigned):
            raise ValueError("checkpoint latent 溯源指纹错误")
        for key, expected in {"schema_fingerprint": canonical_hash(saved["schema"]),
                              "data_fingerprint": saved["data_fingerprint"],
                              "representation_fingerprint": saved["representation_fingerprint"],
                              "statistics_fingerprint": canonical_hash({k: v.tolist() for k, v in saved["latent_stats"].items()})}.items():
            if manifest.get(key) != expected:
                raise ValueError(f"checkpoint latent 溯源 {key} 不一致")
        if not manifest.get("origins"):
            raise ValueError("checkpoint latent 溯源缺 origins")


def scaling_artifacts(transform, schema):
    """训练写入与重载核验共用唯一 §6.7 序列化口径。"""
    state = transform.to_dict()
    def features(names, values):
        return {name: {key: values[key][i] for key in ("mean", "std")} for i, name in enumerate(names)}
    scalers = {"version": 1, "schema_fingerprint": canonical_hash(schema), "groups": {},
               "target": {key: state["target"][key][0] for key in ("mean", "std")},
               "known_future": features(schema["known_future"], state["known_future"])}
    for group in schema["groups"]:
        name = group["name"]
        scalers["groups"][name] = {"dynamic": features(group["dynamic"], state["groups"][name]["dynamic"]),
                                   "static_numeric": features(group["static_numeric"], state["groups"][name]["static"])}
    return {"scalers.json": scalers, "category_vocab.json": {"version": 1, "groups": state["vocab"]}}


def validate_artifacts(root, mode, variant, *, aidc=False):
    """所有必需文件可读后才允许 completed；检查数值而非仅检查存在。"""
    root = Path(root)
    names = {"resolved_config.json", "schema.json", "data_manifest.json", "environment.json", "status.json"}
    latent = variant not in {"total_only", "direct", "persistence", "seasonal_naive"}
    if mode == "train":
        names.update({"scalers.json", "training_metrics.json", "resource_metrics.json"})
        if variant != "total_only":
            names.add("category_vocab.json")
        if latent:
            names.add("latent_stats_manifest.json")
    elif mode in {"test", "forecast", "baseline"}:
        names.add("metadata.json")
        if mode != "forecast":
            names.add("metrics.json")
        if mode == "baseline":
            names.add("baseline_parameters.json")
    elif mode == "compare":
        names.update({"comparison.json", "suite_result.json"})
    else:
        raise ValueError(f"未知产物模式: {mode}")
    if aidc:
        names.update({"site_split.json", "generation_audit.json", "metadata.json", "scenario_metrics.json"})
    for name in sorted(names):
        read_json(root / name)
    if mode == "train":
        # 仅复检本次用户授权产生的本地 artifact，包含 RNG/optimizer，不能用 weights_only。
        checkpoints = ["last.pt", "best.pt"] + ([] if variant == "direct" else ["best_A.pt"])
        for name in checkpoints:
            payload = torch.load(root / "checkpoints" / name, map_location="cpu", weights_only=False)
            validate_checkpoint_contract(payload)
            if payload.get("latent_stats_manifest") is not None:
                if read_json(root / "latent_stats_manifest.json") != payload["latent_stats_manifest"]:
                    raise ValueError("checkpoint 与 latent 溯源旁车不一致")
            expected = "A" if name == "best_A.pt" else "direct" if variant == "direct" else "B-full"
            if payload["phase"] != expected:
                raise ValueError(f"{name} 阶段不一致")
    if not (root / "run.log").read_text(encoding="utf-8").strip():
        raise ValueError("run.log 为空")
    if mode in {"test", "forecast", "baseline", "compare"}:
        name = "comparison.csv" if mode == "compare" else "predictions.csv"
        with (root / name).open(encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"{name} 为空")
        if mode != "compare":
            for row in rows:
                if not np.isfinite(float(row["y_pred"])):
                    raise ValueError("预测产物含非有限值")
    if latent and mode in {"train", "test", "forecast"}:
        filename = "latent_stats.npz" if mode == "train" else "latents.npz"
        with np.load(root / filename, allow_pickle=False) as values:
            if not values.files:
                raise ValueError(f"{filename} 为空")
            for key in values.files:
                value = values[key]
                if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                    raise ValueError(f"{filename}/{key} 含非有限值")


def site_metrics(rows, config):
    """按站点/角色/场景/horizon 报告，并提供站点等权原始量纲指标。"""
    grouped, sites = {}, {}
    roles = config["data"]["site_split"]
    for row in rows:
        if not row["target_valid"]:
            continue
        role = next((role for role, ids in roles.items() if row["site_id"] in ids), "unknown")
        residual = row["y_pred"] - row["y_true"]
        key = (row["site_id"], role, row["scenario"], row["horizon"])
        grouped.setdefault(key, []).append(residual)
        sites.setdefault(row["site_id"], []).append(residual)
    by_site = {site: {"count": len(errors), "mae": float(np.abs(errors).mean()),
                      "rmse": float(np.sqrt(np.square(errors).mean()))} for site, errors in sites.items()}
    return {"split": "test", "pred_len": config["data"]["pred_len"], "per_site": by_site,
            "site_macro_mae": float(np.mean([v["mae"] for v in by_site.values()])),
            "site_macro_rmse": float(np.mean([v["rmse"] for v in by_site.values()])),
            "entries": [{"site_id": key[0], "site_role": key[1], "scenario": key[2], "horizon": key[3],
                         "count": len(errors), "mae": float(np.abs(errors).mean()),
                         "rmse": float(np.sqrt(np.square(errors).mean()))} for key, errors in grouped.items()]}
