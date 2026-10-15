"""架构审计的有界结构探针；不训练模型，不保存权重，不修改源配置。"""
from collections import Counter
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parents[1] / "evidence"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import torch
from data_provider.spatial_temporal.batching import collate_samples
from data_provider.spatial_temporal.config import load_config
from data_provider.spatial_temporal.dataset import PackageDataset
from data_provider.spatial_temporal.scaling import TrainingTransform
from models.transformer.GSLAformer import Model

SOURCES = [
    "models/transformer/GSLAformer.py", "layers/gslaformer/entity_history.py",
    "layers/gslaformer/spatial.py", "layers/gslaformer/relations.py",
    "layers/gslaformer/temporal.py", "exp/exp_gslaformer_forecasting.py",
    "exp/gslaformer_support.py", "data_provider/spatial_temporal/config.py",
    "data_provider/spatial_temporal/batching.py", "data_provider/spatial_temporal/scaling.py",
    "data_provider/spatial_temporal/dataset.py", "docs/gslaformer/arch/model-v4.0.png",
    "scripts/smoke/gslaformer_aidc_hourly_day.json",
    "tests/test_gslaformer_model.py", "tests/test_gslaformer_contract_repairs.py",
    "tests/test_gslaformer_training.py",
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCES}


def shapes(value):
    if torch.is_tensor(value):
        return list(value.shape)
    if isinstance(value, dict):
        return {k: shapes(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [shapes(v) for v in value]
    return type(value).__name__


def main():
    OUT.mkdir(exist_ok=True)
    started = time.perf_counter()
    before = hashes()
    torch.set_num_threads(1)
    torch.manual_seed(2023)
    config = load_config(ROOT / "scripts/smoke/gslaformer_aidc_hourly_day.json")
    print("Loading existing AIDC synthetic package; no generation/training", flush=True)
    dataset = PackageDataset(config, "train")
    available_windows = len(dataset.windows)
    # 审计专用临时统计：仅使用真实训练分区前两个窗口；不改配置或落盘 scaler。
    dataset.windows = dataset.windows[:2]
    transform = TrainingTransform.fit(dataset)
    sample = transform.apply(dataset[0])
    sample["supervision"] = None
    batch = collate_samples([sample], config)
    vocab = {name: {feature: len(tokens) for feature, tokens in features.items()}
             for name, features in transform.state["vocab"].items()}
    model = Model(groups_spec=config["data"]["groups"],
                  relation_types=config["data"]["relation_types"],
                  model_params=config["model"], vocab_sizes=vocab,
                  variant="full", known_future_dim=len(config["data"]["known_future"]),
                  timezone=config["data"]["timezone"]).eval()
    events, calls, temporal_inputs = {}, Counter(), []

    def hook(name):
        def record(module, args, result):
            calls[name] += 1
            events.setdefault(name, {"input": shapes(args), "output": shapes(result)})
        return record

    names = ["history", "relation", "static", "temporal", "future_path", "fusion", "load_head"]
    names += [f"spatial.{g}" for g in model.dynamic_groups]
    names += [f"readout.{g}" for g in model.dynamic_groups]
    names += [f"latent_heads.{g}" for g in model.dynamic_groups]
    handles = [model.get_submodule(name).register_forward_hook(hook(name)) for name in names]
    handles.append(model.temporal.register_forward_pre_hook(
        lambda module, args: temporal_inputs.append(args[0].detach().clone())))
    print("Forward full day profile: L=168, H=24, batch=1", flush=True)
    with torch.no_grad():
        result = model(batch)
    rollout_calls = dict(calls)
    h = config["data"]["pred_len"]
    gd, dz = len(model.dynamic_groups), model.d_z
    assert result.y.shape == (1, h, 1)
    assert result.z.shape == (1, h, gd, dz)
    assert torch.isfinite(result.y).all() and torch.isfinite(result.z).all()
    assert result.generated.all()
    assert calls["history"] == calls["future_path"] == 1
    assert calls["temporal"] == calls["fusion"] == calls["load_head"] == h
    # 独立核对：第二步窗口末 token 必须消费第一步实际生成的 y 与规范 z。
    second = temporal_inputs[1][:, -1]
    flat_z = gd * dz
    torch.testing.assert_close(second[:, :flat_z], result.z[:, 0].flatten(1), atol=0, rtol=0)
    torch.testing.assert_close(second[:, flat_z:flat_z + 1], result.y[:, 0], atol=0, rtol=0)
    assert torch.equal(second[:, flat_z + 1], torch.zeros(1))  # y_observed
    assert torch.equal(second[:, flat_z + 2], torch.ones(1))  # y_generated
    assert torch.count_nonzero(second[:, flat_z + 3:flat_z + 3 + gd]) == 0
    assert torch.all(second[:, flat_z + 3 + gd:flat_z + 3 + 2 * gd] == 1)
    for handle in handles:
        handle.remove()
    with torch.no_grad():
        direct = model.predict_direct(batch)
    assert direct.shape == result.y.shape
    torch.testing.assert_close(direct[:, 0], result.y[:, 0], atol=1e-6, rtol=1e-5)
    core_trainable = model.set_training_stage("B")
    model.train()
    assert all(not p.requires_grad for m in model.representation_modules() for p in m.parameters())
    assert all(not m.training for m in model.representation_modules())
    with torch.no_grad():
        encoded = model.encode_history(batch)
    suite = unittest.TestSuite()
    for pattern in ("test_gslaformer_model.py", "test_gslaformer_contract_repairs.py",
                    "test_gslaformer_training.py"):
        suite.addTests(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern=pattern))
    print("Running existing model/contract/training-support tests", flush=True)
    with (OUT / "targeted-tests.log").open("w") as log:
        tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "base_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "current uncommitted worktree; architecture audit, not trained model evaluation",
        "python": sys.executable, "torch": torch.__version__, "device": "cpu", "seed": 2023,
        "profile": "scripts/smoke/gslaformer_aidc_hourly_day.json",
        "temporary_scaling_scope": "first two actual train windows; no persisted scaler or checkpoint",
        "available_train_windows": available_windows,
        "sample": {"site_ids": batch.inputs.site_ids, "origins": batch.inputs.origins.tolist()},
        "input_shapes": {name: shapes(group["values"]) for name, group in batch.inputs.groups.items()},
        "config_model": config["model"], "dynamic_groups": model.dynamic_groups,
        "known_future": config["data"]["known_future"],
        "relation_types": config["data"]["relation_types"],
        "computation_version": model.COMPUTATION_VERSION,
        "encoded_history": {"z": shapes(encoded.z), "c": shapes(encoded.c),
                            "coverage": shapes(encoded.observed_coverage)},
        "rollout": {"y": shapes(result.y), "z": shapes(result.z), "generated": shapes(result.generated)},
        "direct": shapes(direct), "modules": events, "rollout_calls": rollout_calls,
        "temporal_input_width": model.temporal.project.in_features,
        "static_final_projection": repr(model.static.project),
        "checks": {"finite_outputs": True, "generated_flags": True,
                   "first_step_direct_equals_rollout": True, "actual_y_and_z_feedback": True,
                   "generated_quality_flags": True, "encode_history_and_future_once": True,
                   "stage_b_frozen_representation": True},
        "tests": {"run": tests.testsRun, "failures": len(tests.failures),
                  "errors": len(tests.errors), "skipped": len(tests.skipped),
                  "success": tests.wasSuccessful(), "log": "targeted-tests.log"},
        "source_sha256": before, "sources_unchanged_during_probe": before == hashes(),
        "elapsed_seconds": time.perf_counter() - started,
    }
    assert report["sources_unchanged_during_probe"], "source changed during probe"
    (OUT / "runtime.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("input_shapes", "encoded_history", "rollout", "temporal_input_width", "checks", "tests", "elapsed_seconds")}, ensure_ascii=False, indent=2))
    if not tests.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
