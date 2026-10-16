"""自包含 checkpoint 的离线 test/forecast；不读取训练包、不重拟合 scaler。"""
import csv
from copy import deepcopy
import hashlib
from pathlib import Path
import platform
import time

import numpy as np
import torch

from data_provider.spatial_temporal.batching import collate_samples
from data_provider.spatial_temporal.config import schema_from_config, validate_config, profile_name
from exp.gslaformer_provenance import checkpoint_sources, execution_environment
from exp.gslaformer_metrics import forecast_metrics, latent_metrics
from data_provider.spatial_temporal.dataset import PackageDataset
from data_provider.spatial_temporal.scaling import TrainingTransform
from data_provider.spatial_temporal.schema import canonical_hash, quality_flags, read_json, orphan_entity_count
from exp.exp_gslaformer_forecasting import _json
from exp.gslaformer_artifacts import (validate_artifacts, site_metrics, scaling_artifacts,
                                     validate_checkpoint_contract, request_audit, synchronize)
from exp.gslaformer_support import frozen_representation_fingerprint, diagnostic_rollout, build_latent_labels
from models.transformer.GSLAformer import Model


@torch.no_grad()
def evaluate_checkpoint(config, checkpoint_path, output_dir, *, mode, request=None):
    config = validate_config(deepcopy(config))
    root = Path(output_dir).resolve()
    root.mkdir(parents=True, exist_ok=False)
    _json(root / "status.json", {"status": "running", "mode": mode})
    try:
        return _evaluate_checkpoint(config, checkpoint_path, root, mode=mode, request=request)
    except BaseException as exc:
        _json(root / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                     "mode": mode, "error": str(exc)})
        raise


@torch.no_grad()
def _evaluate_checkpoint(config, checkpoint_path, output_dir, *, mode, request=None):
    if mode not in {"test", "forecast"}:
        raise ValueError("仅接受 test/forecast")
    if (mode == "forecast") != (request is not None):
        raise ValueError("只有 forecast 必须提供 request")
    config = validate_config(deepcopy(config))
    device = config["experiment"]["device"]
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS 不可用")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用")
    checkpoint_path = Path(checkpoint_path).resolve()
    # 仅支持可信本地训练产物；不自动下载和反序列化外部模型。
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    validate_checkpoint_contract(saved)
    trained = saved["config"]
    if (saved["schema"] != schema_from_config(config) or trained["model"] != config["model"]
            or trained["experiment"]["variant"] != config["experiment"]["variant"]
            or any(trained["data"][k] != config["data"][k] for k in ("seq_len", "pred_len"))):
        raise ValueError("checkpoint 与请求 schema/model/L/H/variant 不兼容")
    direct = config["experiment"]["variant"] == "direct"
    if saved["phase"] != ("direct" if direct else "B-full"):
        raise ValueError("只能使用最终选优阶段 checkpoint")
    transform = TrainingTransform.from_dict(saved["transform"], saved["schema"])
    if transform.fingerprint != saved["transform_fingerprint"]:
        raise ValueError("scaler/词表指纹不一致")
    # 自包含 checkpoint 可脱离原目录；原目录存在旁车时不能忽略损坏或冲突。
    source = checkpoint_path.parent.parent if checkpoint_path.parent.name == "checkpoints" else checkpoint_path.parent
    for name, expected in {"schema.json": saved["schema"], **scaling_artifacts(transform, saved["schema"])}.items():
        path = source / name
        if path.exists() and read_json(path) != expected:
            raise ValueError(f"checkpoint 来源旁车 {name} 不一致")
    sizes = {g: {f: len(tokens) for f, tokens in features.items()} for g, features in transform.state["vocab"].items()}
    model = Model(groups_spec=saved["schema"]["groups"], relation_types=saved["schema"]["relation_types"],
                  model_params=config["model"], vocab_sizes=sizes, variant=config["experiment"]["variant"],
                  known_future_dim=len(saved["schema"]["known_future"]), timezone=saved["schema"]["timezone"])
    model.load_state_dict(saved["model"])
    model.set_training_stage("A" if direct else "B")
    validate_checkpoint_contract(saved, model)
    manifest_path = source / "latent_stats_manifest.json"
    if saved.get("latent_stats_manifest") is not None and manifest_path.exists():
        if read_json(manifest_path) != saved["latent_stats_manifest"]:
            raise ValueError("checkpoint 来源 latent 溯源旁车不一致")
    stats_path = source / "latent_stats.npz"
    if saved.get("latent_stats_manifest") is not None and stats_path.exists():
        if hashlib.sha256(stats_path.read_bytes()).hexdigest() != saved["latent_stats_manifest"]["npz_sha256"]:
            raise ValueError("checkpoint 来源 latent 统计旁车不一致")
    if not direct and model.has_latent:
        if frozen_representation_fingerprint(model) != saved["representation_fingerprint"]:
            raise ValueError("冻结表示链指纹不一致")
        if canonical_hash({k: v.tolist() for k, v in saved["latent_stats"].items()}) != saved["latent_stats_fingerprint"]:
            raise ValueError("latent 统计指纹不一致")
    model.to(device).eval()
    data_config = deepcopy(config)
    if request is not None:
        data_config["data"]["package"] = str(Path(request).resolve())
    dataset = PackageDataset(data_config, mode)
    root = Path(output_dir).resolve()
    checkpoint_hash = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    _json(root / "status.json", {"status": "running", "mode": mode})
    _json(root / "resolved_config.json", data_config)
    _json(root / "schema.json", saved["schema"])
    _json(root / "data_manifest.json", dataset.manifest)
    _json(root / "environment.json", execution_environment(device, profile=profile_name(config), **checkpoint_sources(saved)))
    aidc = config["data"]["split_mode"] == "site_time"
    if aidc:
        if saved.get("generation_audit") is None:
            raise ValueError("AIDC checkpoint 缺少生成审计")
        _json(root / "site_split.json", config["data"]["site_split"])
        _json(root / "generation_audit.json", saved["generation_audit"])
    (root / "run.log").write_text(f"{mode} checkpoint={checkpoint_path}\n", encoding="utf-8")
    try:
        rows, latents, predictions, truths, masks, metadata = [], [], [], [], [], []
        diagnostics = {kind: [] for kind in ("fixed_z", "fixed_y", "teacher_z")}
        unavailable_diagnostics = {}
        latent_errors, latent_quality = [], []
        prediction_seconds = 0.0
        h = config["data"]["pred_len"]
        size = config["data"]["batch_size"]
        for start in range(0, len(dataset), size):
            samples = [transform.apply(dataset[i]) for i in range(start, min(start + size, len(dataset)))]
            batch = collate_samples(samples, data_config).to(device)
            synchronize(device)
            started = time.perf_counter()
            if direct:
                y = model.predict_direct(batch)
            else:
                result = model.rollout(batch, h)
                y = result.y
                if model.has_latent:
                    latents.append(result.z.cpu().numpy())
            if not torch.isfinite(y).all():
                raise ValueError("预测含 NaN/Inf")
            original_y = transform.inverse_target(y.cpu().numpy())
            synchronize(device)
            batch_seconds = time.perf_counter() - started
            prediction_seconds += batch_seconds
            predictions.append(original_y)
            true = None if mode == "forecast" else transform.inverse_target(batch.supervision.target.cpu().numpy())
            mask = None if mode == "forecast" else batch.supervision.target_mask.cpu().numpy()
            if true is not None:
                truths.append(true)
                masks.append(mask)
                if model.has_latent and not direct:
                    teacher = build_latent_labels(model, batch.inputs, batch.supervision)
                    latent_errors.append((result.z - teacher.z).square().cpu().numpy())
                    latent_quality.append(teacher.quality.cpu().numpy())
                    for kind in diagnostics:
                        if kind == "fixed_y" and not batch.inputs.target_history_observed.squeeze(-1).any(dim=1).all():
                            unavailable_diagnostics[kind] = "at_least_one_request_has_no_observed_target_history"
                            continue
                        diagnostics[kind].append(transform.inverse_target(diagnostic_rollout(model, batch, kind).cpu().numpy()))
            for i, sample in enumerate(samples):
                inputs = sample["inputs"]
                sample["metadata"]["orphan_count"] = orphan_entity_count(inputs["groups"], inputs["relations_origin"])
                flags = quality_flags(inputs["groups"], inputs["target_history_observed"], inputs["future_known_mask"],
                                      no_history_entity_count=sample["metadata"]["no_history_entity_count"],
                                      unknown_category_count=sample["metadata"]["unknown_category_count"],
                                      orphan_count=sample["metadata"]["orphan_count"])
                contract = inputs["target_contract"]
                metadata.append({**sample["metadata"], **flags, **request_audit(inputs, config),
                                 "prediction_batch_seconds": batch_seconds, "prediction_batch_size": len(samples),
                                 "site_id": inputs["site_id"],
                                 "origin": int(inputs["origin"]), "target_contract": contract})
                for k in range(h):
                    rows.append({"run_id": root.name, "model_version": checkpoint_hash, "data_kind": "synthetic",
                                 "site_id": inputs["site_id"], "scenario": sample["metadata"]["scenario"], "target_id": contract["target_id"],
                                 "boundary_id": contract["boundary_id"], "origin": int(inputs["origin"]),
                                 "target_time": int(inputs["future_time"][k]), "horizon": k + 1,
                                 "freq_seconds": config["data"]["freq_seconds"], "unit": contract["unit"],
                                 "y_pred": float(original_y[i, k, 0]),
                                 "y_true": "" if true is None or not mask[i, k, 0] else float(true[i, k, 0]),
                                 "target_valid": "" if mask is None else bool(mask[i, k, 0]),
                                 "new_site": inputs["site_id"] not in saved["training_sites"],
                                 **{key: sample["metadata"][key] for key in ("new_entity_count", "no_history_entity_count", "unknown_category_count")},
                                 **{key: flags[key] for key in ("input_coverage", "degraded_input", "plan_available")},
                                 "negative_prediction": bool(original_y[i, k, 0] < 0)})
        temporary = root / "predictions.csv.tmp"
        with temporary.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(root / "predictions.csv")
        if latents:
            with (root / "latents.npz.tmp").open("wb") as stream:
                np.savez(stream, z=np.concatenate(latents), groups=np.array(model.dynamic_groups))
            (root / "latents.npz.tmp").replace(root / "latents.npz")
        _json(root / "metadata.json", {"checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint_hash,
                                       "profile": profile_name(config),
                                       "computation_version": model.COMPUTATION_VERSION,
                                       "parameter_count": sum(p.numel() for p in model.parameters()),
                                       "prediction_seconds": prediction_seconds,
                                       "timing_scope": "batch forward plus inverse scale; excludes loader and teacher diagnostics",
                                       "schema_fingerprint": canonical_hash(saved["schema"]),
                                       "data_fingerprint": dataset.fingerprint, "requests": metadata})
        if mode == "test":
            prediction, target, valid = np.concatenate(predictions), np.concatenate(truths), np.concatenate(masks)
            summary = forecast_metrics(prediction, target, valid, config["data"]["freq_seconds"])
            _json(root / "metrics.json", summary)
            if latent_errors:
                errors, quality = np.concatenate(latent_errors), np.concatenate(latent_quality)
                _json(root / "latent_metrics.json", latent_metrics(errors, quality, model.dynamic_groups, saved["latent_stats"]))
            if aidc:
                _json(root / "scenario_metrics.json", site_metrics(rows, config))
            if model.has_latent and not direct:
                diagnostic_metrics = {}
                for kind, batches in diagnostics.items():
                    if kind in unavailable_diagnostics:
                        diagnostic_metrics[kind] = {"status": "not_run", "reason": unavailable_diagnostics[kind]}
                        continue
                    error = np.concatenate(batches) - target
                    diagnostic_metrics[kind] = {"status": "passed", "micro_mae": float(np.abs(error[valid]).mean()),
                                                "micro_rmse": float(np.sqrt(np.square(error[valid]).mean())),
                                                "uses_future_observations": kind == "teacher_z"}
                _json(root / "diagnostics.json", {"ranking_eligible": False, "model_selection_eligible": False,
                                                   "not_a_mathematical_upper_bound": True, "diagnostics": diagnostic_metrics})
        # 完成前重新解析落盘文件；无标签模式不强制 metrics。
        read_json(root / "metadata.json")
        if aidc and mode == "forecast":
            _json(root / "scenario_metrics.json", {"split": "forecast", "metrics": None, "reason": "unlabeled_request"})
        with (root / "predictions.csv").open(encoding="utf-8") as stream:
            if sum(1 for _ in csv.DictReader(stream)) != len(dataset) * h:
                raise ValueError("预测输出行数不一致")
        validate_artifacts(root, mode, config["experiment"]["variant"], aidc=aidc)
        _json(root / "status.json", {"status": "completed", "mode": mode, "rows": len(rows)})
        return root
    except BaseException as exc:
        _json(root / "status.json", {"status": "failed", "mode": mode, "error": str(exc)})
        raise
