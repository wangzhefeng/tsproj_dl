#!/usr/bin/env python
"""有界测量真实 profile 的 A/B/valid 成本；不是训练验收，不生成可部署权重。"""
import argparse
import math
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_provider.spatial_temporal.config import load_config
from exp.exp_gslaformer_forecasting import Exp_GSLAformer_Forecast, _json
from exp.gslaformer_support import (build_latent_labels, collect_latent_statistics,
                                   latent_dimension_weights, masked_load_loss, weighted_latent_loss)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batches", type=int, default=3)
    args = parser.parse_args()
    if args.batches < 2:
        parser.error("至少两个 batch，用于非退化统计探针")
    torch.set_num_threads(1)
    config = load_config(args.config)
    started = time.perf_counter()
    print("initializing dataset/scaler/model", flush=True)
    experiment = Exp_GSLAformer_Forecast(config, args.output)
    report = {"purpose": "bounded_resource_probe_not_training_acceptance", "status": "running",
              "config": config, "setup_seconds": time.perf_counter() - started,
              "threads": torch.get_num_threads(), "device": str(experiment.device),
              "parameter_count": sum(p.numel() for p in experiment.model.parameters()),
              "train_windows": len(experiment.train_data), "valid_windows": len(experiment.valid_data),
              "train_updates_per_epoch": math.ceil(len(experiment.train_data) / config["data"]["batch_size"]),
              "stages": {}, "limitations": ["Not a complete epoch or convergence run",
                  "B uses representation after bounded A updates, not selected best_A",
                  "Probe latent statistics use sampled training origins, not full Freeze statistics",
                  "Epoch cost is extrapolated; full Freeze/checkpoint/test costs excluded",
                  "CPU RSS is process lifetime high-water mark, not per-stage incremental peak"]}
    _json(args.output / "resource_report.json", report)
    try:
        order = (experiment.site_sampler.draw(experiment.generator) if experiment.site_sampler
                 else torch.randperm(len(experiment.train_data), generator=experiment.generator).tolist())
        subset = order[:args.batches * config["data"]["batch_size"]]
        horizon = config["data"]["pred_len"]
        weights = None
        for phase, steps in (("A", horizon), ("B-short", 1), ("B-medium", min(3, horizon)), ("B-full", horizon)):
            experiment.phase = phase
            if phase == "B-short":
                experiment.optimizer = experiment._new_optimizer("B")
                before = time.perf_counter()
                collector = collect_latent_statistics(experiment.model, experiment._batches(experiment.train_data, subset),
                    schema_fingerprint=experiment.schema_fingerprint, data_fingerprint=experiment.data_fingerprint)
                stats = collector.finalize()
                variance = np.stack([stats[f"{name}__variance"] for name in experiment.model.dynamic_groups])
                weights = latent_dimension_weights(torch.tensor(variance, dtype=torch.float32, device=experiment.device))
                report["sampled_freeze_seconds"] = time.perf_counter() - before
            experiment.model.train()
            batches = iter(experiment._batches(experiment.train_data, subset))
            records = []
            for index in range(args.batches):
                synchronize(experiment.device)
                begin = time.perf_counter()
                try:
                    batch = next(batches)
                except StopIteration:
                    break
                data_seconds = time.perf_counter() - begin
                experiment.optimizer.zero_grad(set_to_none=True)
                before = time.perf_counter()
                if phase == "A":
                    prediction = experiment.model.predict_direct(batch)
                    loss, _ = masked_load_loss(prediction, batch.supervision.target, batch.supervision.target_mask)
                else:
                    history = experiment.model.encode_history(batch)
                    prediction = experiment.model._rollout_from_history(batch, steps, history)
                    loss, _ = masked_load_loss(prediction.y, batch.supervision.target[:, :steps], batch.supervision.target_mask[:, :steps])
                    labels = build_latent_labels(experiment.model, batch.inputs, batch.supervision, history, steps=steps)
                    latent, _ = weighted_latent_loss(prediction.z, labels, weights)
                    loss = loss + config["training"]["latent_weight"] * latent
                synchronize(experiment.device)
                forward_seconds = time.perf_counter() - before
                before = time.perf_counter()
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in experiment.model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
                experiment.optimizer.step()
                synchronize(experiment.device)
                backward_seconds = time.perf_counter() - before
                row = {"data_seconds": data_seconds, "forward_teacher_seconds": forward_seconds,
                       "backward_optimizer_seconds": backward_seconds, "total_seconds": time.perf_counter() - begin,
                       "loss": loss.item(), "sites": batch.inputs.site_ids,
                       "nodes": {name: g["values"].shape[2] for name, g in batch.inputs.groups.items()},
                       "cpu_process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)}
                if experiment.device.type == "mps":
                    row["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
                records.append(row)
                print(f"{phase} {index + 1}/{args.batches}: {row['total_seconds']:.3f}s", flush=True)
            average = float(np.mean([row["total_seconds"] for row in records]))
            report["stages"][phase] = {"steps": steps, "batches": records, "mean_batch_seconds": average,
                                       "estimated_epoch_seconds_excluding_valid": average * report["train_updates_per_epoch"]}
            _json(args.output / "resource_report.json", report)
        before = time.perf_counter()
        score = experiment.valid()
        synchronize(experiment.device)
        report["full_valid_seconds"] = time.perf_counter() - before
        report["probe_valid_mse_not_accuracy_result"] = score
        report["status"] = "completed"
        _json(args.output / "resource_report.json", report)
        experiment._status("interrupted", "resource_probe_only; no complete training epoch or deployable checkpoint")
        print("resource probe completed", flush=True)
    except BaseException as exc:
        report.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
        _json(args.output / "resource_report.json", report)
        experiment._status(report["status"], str(exc))
        raise


if __name__ == "__main__":
    main()
