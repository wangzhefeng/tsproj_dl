#!/usr/bin/env python
"""M1–M4 前八个训练 family 的独立小集拟合；不是正式 test 选模。

固定 seed=2023、dropout=0；M1/A lr=0.001，其余 lr=0.003，各阶段最多 1000 update。
M1/M2 另要求每对每 horizon 符号正确且误差严格小于目标半间距。
"""
import argparse
from copy import deepcopy
import json
import hashlib
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from data_provider.spatial_temporal.batching import collate_samples
from data_provider.spatial_temporal.config import load_config
from data_provider.spatial_temporal.dataset import PackageDataset
from data_provider.spatial_temporal.scaling import TrainingTransform
from exp.gslaformer_support import (build_latent_labels, collect_latent_statistics,
                                   latent_dimension_weights, masked_load_loss, weighted_latent_loss)
from models.transformer.GSLAformer import Model
from exp.gslaformer_provenance import execution_environment


def measure(prediction, target, task, initial):
    mse = (prediction - target).square().mean().item()
    threshold = 1e-4 if initial <= 1e-4 else max(1e-4, initial * 0.2)
    report = {"mse": mse, "threshold": threshold, "loss_passed": mse <= threshold}
    if task in {"M1", "M2"}:
        true_pair = target.reshape(8, 2, target.shape[1], 1)
        pred_pair = prediction.reshape_as(true_pair)
        delta = true_pair[:, 1] - true_pair[:, 0]
        direction = (pred_pair[:, 1] - pred_pair[:, 0]) * delta > 0
        error = (pred_pair - true_pair).abs() < delta.abs().unsqueeze(1) / 2
        report["pair_passed"] = bool(direction.all() and error.all())
        report["pair_cell_pass_fraction"] = float(error.float().mean())
    else:
        report["pair_passed"] = True
    report["passed"] = report["loss_passed"] and report["pair_passed"]
    return report


def run(task, variant, fixtures, output, *, stage_a_checkpoint=None):
    run_dir = output / f"{task}_{variant}"
    run_dir.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(2023)
    np.random.seed(2023)
    config = load_config(fixtures / "mechanisms" / task / "config.json")
    config["model"]["dropout"] = 0.0
    config["training"]["learning_rate"] = 0.003
    config["experiment"]["variant"] = variant
    config["data"]["batch_size"] = 16
    dataset = PackageDataset(config, "train")
    families = sorted({dataset.entries[e]["family_id"] for e, _ in dataset.windows})[:8]
    dataset.windows = [(e, o) for e, o in dataset.windows if dataset.entries[e]["family_id"] in families]
    if len(dataset) != 16:
        raise ValueError("小集必须是前八 family 的十六个成对样本")
    transform = TrainingTransform.fit(dataset)
    batch = collate_samples([transform.apply(dataset[i]) for i in range(len(dataset))], config)
    sizes = {g: {f: len(tokens) for f, tokens in features.items()} for g, features in transform.state["vocab"].items()}
    model = Model(groups_spec=config["data"]["groups"], relation_types=config["data"]["relation_types"],
                  model_params=config["model"], vocab_sizes=sizes, variant=variant,
                  known_future_dim=len(config["data"]["known_future"]))
    target = batch.supervision.target
    report = {"task": task, "variant": variant, "data_kind": "synthetic", "purpose": "small_fit_only",
              "config": config, "family_ids": families, "stages": {},
              "stage_learning_rates": {"A": 0.001 if task == "M1" else 0.003, "B": 0.003}}
    stages = ["A"] if variant == "direct" else ["A", "B"]
    if stage_a_checkpoint is not None:
        if (task, variant) != ('M1', 'full'):
            raise ValueError('本轮仅允许 M1/full 复用已验收的 A 权重')
        source = Path(stage_a_checkpoint).resolve()
        # 用户授权的本机小集权重，含 transform 元数据；不接受外部不可信 pickle。
        saved = torch.load(source, map_location='cpu', weights_only=False)
        evidence = json.loads((source.parent / 'report.json').read_text())
        source_config = deepcopy(saved['config'])
        source_config['training']['learning_rate'] = 0.003
        if (saved.get('stage') != 'A' or saved.get('purpose') != 'small_fit_only'
                or saved.get('computation_version') != Model.COMPUTATION_VERSION
                or source_config != config or evidence['family_ids'] != families
                or evidence['data_fingerprint'] != dataset.fingerprint
                or evidence['transform_fingerprint'] != transform.fingerprint
                or TrainingTransform(saved['transform']).fingerprint != transform.fingerprint
                or evidence['learning_rate'] != 0.001):
            raise ValueError('小集 A 来源、数据、配置或变换不匹配')
        model.load_state_dict(saved['model'])
        model.eval()
        with torch.no_grad():
            checked = measure(model.predict_direct(batch), target, task, evidence['initial_mse'])
        if not checked['passed'] or checked != evidence['outcome']:
            raise ValueError('小集 A 重载未通过原始门槛或与来源回执不符')
        report['stages']['A'] = {**checked, 'initial_mse': evidence['initial_mse'],
                                 'updates': 0, 'source_updates': evidence['updates'], 'reused': True}
        report['initialization'] = {'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
        stages = ['B']
    (run_dir / "resolved_config.json").write_text(json.dumps(config, indent=2))
    (run_dir / "environment.json").write_text(json.dumps(execution_environment('cpu'), indent=2))
    for stage in stages:
        optimizer = torch.optim.AdamW(model.set_training_stage(stage), lr=report['stage_learning_rates'][stage], weight_decay=0.0001)
        labels = weights = None
        if stage == "B":
            collector = collect_latent_statistics(model, [batch], schema_fingerprint=batch.inputs.metadata[0]["schema_fingerprint"],
                                                   data_fingerprint=dataset.fingerprint)
            collector.save(run_dir)
            stats = collector.finalize()
            weights = latent_dimension_weights(torch.tensor(np.stack([stats[f"{g}__variance"] for g in model.dynamic_groups]), dtype=torch.float32))
            labels = build_latent_labels(model, batch.inputs, batch.supervision)
        def predict():
            return model.predict_direct(batch) if stage == "A" else model.rollout(batch, target.shape[1]).y
        model.eval()
        with torch.no_grad():
            initial = (predict() - target).square().mean().item()
        best_score, best_state = float("inf"), None
        start = time.perf_counter()
        outcome = measure(predict().detach(), target, task, initial)
        for update in range(1, 1001):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if stage == "A":
                prediction = model.predict_direct(batch)
                loss, _ = masked_load_loss(prediction, target, batch.supervision.target_mask)
            else:
                prediction = model.rollout(batch, target.shape[1])
                loss, _ = masked_load_loss(prediction.y, target, batch.supervision.target_mask)
                auxiliary, _ = weighted_latent_loss(prediction.z, labels, weights)
                loss = loss + 0.1 * auxiliary
            if not torch.isfinite(loss):
                raise ValueError("拟合 loss 非有限")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
            optimizer.step()
            model.eval()
            with torch.no_grad():
                outcome = measure(predict(), target, task, initial)
            if outcome["mse"] < best_score:
                best_score, best_state = outcome["mse"], deepcopy(model.state_dict())
            if update == 1 or update % 100 == 0 or outcome["passed"]:
                print(task, variant, stage, update, outcome, flush=True)
            if outcome["passed"]:
                break
        # 成对门槛通过时保留实际通过权重；失败则保留最小训练 MSE 权重作诊断。
        if not outcome["passed"]:
            model.load_state_dict(best_state)
            with torch.no_grad():
                outcome = measure(predict(), target, task, initial)
        report["stages"][stage] = {"initial_mse": initial, "updates": update,
                                    "seconds": time.perf_counter() - start, **outcome}
        torch.save({"model": model.state_dict(), "config": config, "transform": transform.to_dict(),
                    "computation_version": Model.COMPUTATION_VERSION,
                    "stage": stage, "purpose": "small_fit_only"}, run_dir / f"fit_{stage}.pt")
        (run_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
        if not outcome["passed"]:
            break  # A 未拟合不冻结进入 B，不靠后续阶段掩盖失败。
    report["passed"] = all(stage["passed"] for stage in report["stages"].values())
    (run_dir / "fit_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument('--stage_a_checkpoint', type=Path)
    parser.add_argument("--tasks", nargs='+', choices=['M1_full', 'M2_full', 'M3_full', 'M3_direct', 'M4_full', 'M4_direct'],
                        default=['M1_full', 'M2_full', 'M3_full', 'M3_direct', 'M4_full', 'M4_direct'])
    args = parser.parse_args()
    if args.stage_a_checkpoint is not None and args.tasks != ['M1_full']:
        parser.error('--stage_a_checkpoint 仅接受 --tasks M1_full')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    reports = []
    for name in args.tasks:
        task, variant = name.split('_')
        reports.append(run(task, variant, args.fixtures.resolve(), args.output, stage_a_checkpoint=args.stage_a_checkpoint))
        (args.output / "suite_result.json").write_text(json.dumps(reports, indent=2))
        if not reports[-1]['passed']:
            break
    raise SystemExit(0 if all(report["passed"] for report in reports) else 1)


if __name__ == "__main__":
    main()
