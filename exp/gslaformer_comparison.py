"""GSLAformer 对照调度：标准包测试、朴素基线、共享 best_A 与逐子 run 状态。

只按 acceptance 的任务矩阵调度；比较成功不代表 full 获胜或整体模拟验收完成。
"""
from copy import deepcopy
import csv
from pathlib import Path
import platform
import time

import numpy as np

from data_provider.spatial_temporal.config import validate_config
from data_provider.spatial_temporal.dataset import PackageDataset
from data_provider.spatial_temporal.schema import quality_flags, read_json, orphan_entity_count
from exp.exp_gslaformer_forecasting import Exp_GSLAformer_Forecast, _json
from exp.gslaformer_inference import evaluate_checkpoint
from exp.gslaformer_artifacts import validate_artifacts, site_metrics, request_audit


MATRICES = {
    "base": ["persistence", "seasonal_naive", "full"],
    "M1": ["full", "no_entity_history", "direct"],
    "M2": ["full", "no_relation_message", "no_relation_all"],
    "M3": ["full", "direct", "no_latent_loss", "total_only"],
    "M4": ["full", "direct", "no_spatial", "total_only"],
}


def required_variants(config):
    if config["data"]["split_mode"] == "site_time":
        variants = ["full", "direct", "total_only", "persistence", "seasonal_naive"]
        if config["data"]["pred_len"] == 24:
            variants += ["no_relation_all", "no_spatial"]
        return variants
    task = config["experiment"].get("profile", Path(config["data"]["package"]).name)
    if task in MATRICES:
        return MATRICES[task]
    if config["data"]["split_mode"] == "target_index":
        return MATRICES["base"]
    raise ValueError(f"未知机制任务 {task}，需在 experiment.profile 明确任务名")


def baseline_forecast(inputs, variant, seasonal_period, freq_seconds):
    history = inputs["target_history"][:, 0]
    observed = inputs["target_history_observed"][:, 0]
    eligible = np.flatnonzero(observed)
    if not len(eligible):
        raise ValueError("朴素基线没有有效历史总负荷")
    last = history[eligible[-1]]
    values, fallback = [], 0
    for stamp in inputs["future_time"]:
        if variant == "persistence":
            values.append(last)
        elif variant == "seasonal_naive":
            phase = ((stamp - inputs["history_time"]) % (seasonal_period * freq_seconds)) == 0
            candidates = np.flatnonzero(observed & phase & (inputs["history_time"] <= inputs["origin"]))
            if len(candidates):
                values.append(history[candidates[-1]])
            else:
                values.append(last)
                fallback += 1
        else:
            raise ValueError("未知朴素基线")
    return np.asarray(values, np.float64)[:, None], fallback


def run_baseline(config, output_dir):
    config = validate_config(deepcopy(config))
    variant = config["experiment"]["variant"]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "status.json", {"status": "running"})
    try:
        dataset = PackageDataset(config, "test")
        metadata = []
        rows, predictions, truths, masks = [], [], [], []
        fallback_count = 0
        prediction_seconds = 0.0
        for index in range(len(dataset)):
            sample = dataset[index]
            inputs, supervision = sample["inputs"], sample["supervision"]
            no_history = sum(int((g["present"][-1] & ~g["padding"] &
                ~(g["observed"] & g["applicable"][None] & g["present"][..., None]).any(axis=(0, 2))).sum())
                for g in inputs["groups"].values() if g["values"].shape[-1])
            started = time.perf_counter()
            prediction, fallback = baseline_forecast(inputs, variant, config["evaluation"]["seasonal_period"], config["data"]["freq_seconds"])
            elapsed = time.perf_counter() - started
            prediction_seconds += elapsed
            fallback_count += fallback
            predictions.append(prediction)
            truths.append(supervision["target"])
            masks.append(supervision["target_mask"])
            flags = quality_flags(inputs["groups"], inputs["target_history_observed"], inputs["future_known_mask"],
                                  orphan_count=orphan_entity_count(inputs["groups"], inputs["relations_origin"]),
                                  no_history_entity_count=no_history)
            contract = inputs["target_contract"]
            metadata.append({**sample["metadata"], **flags, **request_audit(inputs, config),
                             "prediction_seconds": elapsed, "site_id": inputs["site_id"], "target_contract": contract})
            for k, value in enumerate(prediction[:, 0]):
                valid = bool(supervision["target_mask"][k, 0])
                rows.append({"run_id": root.name, "model_version": variant, "data_kind": "synthetic",
                             "site_id": inputs["site_id"], "target_id": contract["target_id"], "boundary_id": contract["boundary_id"],
                             "scenario": sample["metadata"]["scenario"],
                             "new_site": inputs["site_id"] not in (config["data"].get("site_split") or {}).get("train", [inputs["site_id"]]),
                             "new_entity_count": "", "unknown_category_count": "", "no_history_entity_count": no_history,
                             "origin": int(inputs["origin"]), "target_time": int(inputs["future_time"][k]), "horizon": k + 1,
                             "freq_seconds": config["data"]["freq_seconds"], "unit": contract["unit"],
                             "y_pred": float(value), "y_true": float(supervision["target"][k, 0]) if valid else "",
                             "target_valid": valid, "negative_prediction": bool(value < 0),
                             "input_coverage": flags["input_coverage"], "degraded_input": flags["degraded_input"],
                             "plan_available": flags["plan_available"]})
        p, y, valid = np.stack(predictions), np.stack(truths), np.stack(masks)
        residual = p - y
        if not valid.any() or not np.isfinite(p).all():
            raise ValueError("朴素基线预测无效")
        metrics = {"micro_mae": float(np.abs(residual[valid]).mean()),
                   "micro_rmse": float(np.sqrt(np.square(residual[valid]).mean())), "per_horizon": []}
        for k in range(p.shape[1]):
            r = residual[:, k][valid[:, k]]
            metrics["per_horizon"].append({"horizon": k + 1, "count": len(r),
                                           "mae": float(np.abs(r).mean()) if len(r) else None,
                                           "rmse": float(np.sqrt(np.square(r).mean())) if len(r) else None})
        complete = valid[:, :, 0].all(axis=1)
        pp, yy = p[complete, :, 0], y[complete, :, 0]
        metrics["complete_window_count"] = int(complete.sum())
        metrics["peak_amplitude_mae"] = float(np.abs(pp.max(axis=1) - yy.max(axis=1)).mean()) if complete.any() else None
        metrics["peak_time_mae_seconds"] = float(np.abs(pp.argmax(axis=1) - yy.argmax(axis=1)).mean() * config["data"]["freq_seconds"]) if complete.any() else None
        ramp = valid[:, 1:, 0] & valid[:, :-1, 0]
        metrics["ramp_count"] = int(ramp.sum())
        metrics["ramp_mae"] = float(np.abs(np.diff(p[:, :, 0], axis=1) - np.diff(y[:, :, 0], axis=1))[ramp].mean()) if ramp.any() else None
        with (root / "predictions.csv.tmp").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (root / "predictions.csv.tmp").replace(root / "predictions.csv")
        _json(root / "metrics.json", metrics)
        _json(root / "resolved_config.json", config)
        _json(root / "schema.json", dataset.episodes[0].schema)
        _json(root / "data_manifest.json", dataset.manifest)
        _json(root / "environment.json", {"python": platform.python_version(), "device": "cpu", "model": variant})
        _json(root / "baseline_parameters.json", {"variant": variant, "seasonal_period": config["evaluation"]["seasonal_period"],
                                                  "fallback_count": fallback_count, "input": "historical_target_only"})
        (root / "run.log").write_text(f"{variant}: {len(dataset)} windows\n")
        _json(root / "metadata.json", {"requests": metadata, "schema_fingerprint": dataset.episodes[0].schema_fingerprint,
             "parameter_count": 0, "prediction_seconds": prediction_seconds,
             "data_fingerprint": dataset.fingerprint, "identity_counts": "new_entity/unknown_category not applicable: no learned vocabulary"})
        aidc = config["data"]["split_mode"] == "site_time"
        if aidc:
            _json(root / "site_split.json", config["data"]["site_split"])
            _json(root / "generation_audit.json", read_json(Path(config["data"]["package"]) / "generation_audit.json"))
            _json(root / "scenario_metrics.json", site_metrics(rows, config))
        validate_artifacts(root, "baseline", variant, aidc=aidc)
        _json(root / "status.json", {"status": "completed", "rows": len(rows)})
        return metrics
    except BaseException as exc:
        _json(root / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": str(exc)})
        raise


def run_comparison(config, output_dir):
    config = validate_config(deepcopy(config))
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "status.json", {"status": "running"})
    try:
        result = _run_comparison(config, root)
        validate_artifacts(root, "compare", config["experiment"]["variant"],
                           aidc=config["data"]["split_mode"] == "site_time")
        _json(root / "status.json", {"status": "completed" if result["status"] == "passed" else "failed"})
        return result
    except BaseException as exc:
        _json(root / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error": str(exc)})
        raise


def _run_comparison(config, output_dir):
    config = validate_config(deepcopy(config))
    variants = required_variants(config)
    root = Path(output_dir)
    dataset = PackageDataset(config, "test")
    _json(root / "schema.json", dataset.episodes[0].schema)
    _json(root / "data_manifest.json", dataset.manifest)
    _json(root / "environment.json", {"python": platform.python_version(), "device": config["experiment"]["device"]})
    (root / "run.log").write_text("GSLAformer configured comparison\n")
    if config["data"]["split_mode"] == "site_time":
        _json(root / "site_split.json", config["data"]["site_split"])
        _json(root / "generation_audit.json", read_json(Path(config["data"]["package"]) / "generation_audit.json"))
        _json(root / "metadata.json", {"data_fingerprint": dataset.fingerprint, "scope": "comparison_child_runs"})
    _json(root / "resolved_config.json", config)
    result = {"scope": "configured_task_comparison_only", "required_variants": variants,
              "status": "running", "runs": {v: {"status": "not_run"} for v in variants}}
    _json(root / "comparison.json", result)
    for variant in variants:
        current = deepcopy(config)
        current["experiment"]["variant"] = variant
        current["training"]["latent_weight"] = 0 if variant == "no_latent_loss" else 0.1
        started = time.perf_counter()
        _json(root / f"{variant}_config.json", current)
        result["runs"][variant] = {"status": "running", "config": str(root / f"{variant}_config.json")}
        _json(root / "comparison.json", result)
        try:
            if variant in {"persistence", "seasonal_naive"}:
                metrics = run_baseline(current, root / f"{variant}_test")
                updates = 0
                parameter_count = 0
            else:
                source = root / "full_train/checkpoints/best_A.pt" if variant == "no_latent_loss" else None
                experiment = Exp_GSLAformer_Forecast(current, root / f"{variant}_train", stage_a_checkpoint=source)
                history = experiment.train()
                updates = sum(row["updates"] for row in history)
                parameter_count = sum(p.numel() for p in experiment.model.parameters())
                evaluate_checkpoint(current, root / f"{variant}_train/checkpoints/best.pt", root / f"{variant}_test", mode="test")
                metrics = read_json(root / f"{variant}_test/metrics.json")
                write_pair_metrics(current, root / f"{variant}_test")
            result["runs"][variant].update(status="passed", metrics=metrics, updates=updates,
                                           parameter_count=parameter_count,
                                           test_path=str(root / f"{variant}_test"))
        except Exception as exc:
            result["runs"][variant].update(status="failed", error=str(exc))
        result["runs"][variant]["wall_seconds"] = time.perf_counter() - started
        _json(root / "comparison.json", result)
    result["status"] = "passed" if all(r["status"] == "passed" for r in result["runs"].values()) else "failed"
    _json(root / "comparison.json", result)
    _json(root / "suite_result.json", result)
    if config["data"]["split_mode"] == "site_time":
        _json(root / "scenario_metrics.json", {variant: read_json(root / f"{variant}_test/scenario_metrics.json")
              for variant, row in result["runs"].items() if row["status"] == "passed"})
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "status", "micro_mae", "micro_rmse", "updates", "wall_seconds"])
        writer.writeheader()
        for variant, row in result["runs"].items():
            writer.writerow({"variant": variant, "status": row["status"], "updates": row.get("updates"),
                             "wall_seconds": row["wall_seconds"],
                             **{key: row.get("metrics", {}).get(key) for key in ("micro_mae", "micro_rmse")}})
    return result


def write_pair_metrics(config, test_dir):
    """M1/M2 按 family 报告泛化成对区分；不以 pooled MSE 代替机制指标。"""
    task = config["experiment"].get("profile", Path(config["data"]["package"]).name)
    if task not in {"M1", "M2"}:
        return None
    root = Path(test_dir)
    manifest = read_json(Path(config["data"]["package"]) / "manifest.json")
    families = {entry["id"]: entry["family_id"] for entry in manifest["episodes"]}
    requests = read_json(root / "metadata.json")["requests"]
    with (root / "predictions.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    h = config["data"]["pred_len"]
    if len(rows) != len(requests) * h:
        raise ValueError("pair metrics 请求与行数不一致")
    paired = {}
    for i, request in enumerate(requests):
        block = rows[i * h:(i + 1) * h]
        if any(row["y_true"] == "" for row in block):
            raise ValueError("机制成对指标要求完整真值")
        paired.setdefault(families[request["episode_id"]], []).append(
            (np.array([float(row["y_pred"]) for row in block]), np.array([float(row["y_true"]) for row in block])))
    outcomes = {}
    for family, pair in paired.items():
        if len(pair) != 2:
            raise ValueError("机制成对指标必须每 family 恰有两个 episode")
        p, y = np.stack([v[0] for v in pair]), np.stack([v[1] for v in pair])
        delta = y[1] - y[0]
        signs = (p[1] - p[0]) * delta > 0
        separated = (np.abs(p - y) < np.abs(delta)[None] / 2).all(axis=0)
        outcomes[family] = {"sign_fraction": float(signs.mean()), "half_gap_fraction": float(separated.mean()),
                             "all_horizons_distinguished": bool((signs & separated).all())}
    result = {"split": "test", "task": task, "families": outcomes,
              "fully_distinguished_family_fraction": float(np.mean([v["all_horizons_distinguished"] for v in outcomes.values()])),
              "note": "single-seed heldout report; not the small-fit acceptance gate"}
    _json(root / "mechanism_metrics.json", result)
    return result
