"""GSLAformer A→Freeze→B 训练状态机；epoch 边界恢复，不走旧四张量实验接口。

程序化训练入口先落地；run.py 路由、test/forecast 文件输出由 I5 接入。
只加载本地可信 checkpoint；状态机不支持 batch 中途精确恢复。
"""
from copy import copy, deepcopy
import json
import os
from pathlib import Path
import platform
import random
import time
from types import SimpleNamespace
import uuid

import numpy as np
import torch

from data_provider.spatial_temporal.batching import Batch, collate_samples
from data_provider.spatial_temporal.config import validate_config, schema_from_config
from data_provider.spatial_temporal.dataset import PackageDataset
from data_provider.spatial_temporal.scaling import TrainingTransform
from data_provider.spatial_temporal.schema import canonical_hash, read_json, quality_flags
from exp.exp_basic import Exp_Basic
from exp.gslaformer_artifacts import validate_artifacts, scaling_artifacts, validate_checkpoint_contract, synchronize
from exp.gslaformer_support import (LatentLabels, ValidationMSE, build_latent_labels, collect_latent_statistics,
                                   frozen_representation_fingerprint, latent_dimension_weights,
                                   masked_load_loss, weighted_latent_loss)


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def _cpu_state(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


class SiteOriginSampler:
    """站点均匀→站内窗口均匀；有放回，epoch 长度等于全部训练窗口数。"""

    def __init__(self, dataset):
        self.size = len(dataset)
        self.indices = {}
        for index, (episode, _) in enumerate(dataset.windows):
            site = dataset.entries[episode]["site_id"]
            self.indices.setdefault(site, []).append(index)
        self.sites = sorted(self.indices)

    def draw(self, generator):
        chosen = torch.randint(len(self.sites), (self.size,), generator=generator).tolist()
        result = []
        for index in chosen:
            candidates = self.indices[self.sites[index]]
            position = torch.randint(len(candidates), (), generator=generator).item()
            result.append(candidates[position])
        return result


class Exp_GSLAformer_Forecast(Exp_Basic):
    """显式阶段选优；只 B-full 生成 full 最终 best。"""

    def __init__(self, config, run_dir, *, resume=None, stage_a_checkpoint=None):
        started = time.perf_counter()
        self.config = validate_config(deepcopy(config))
        self.run_dir = Path(run_dir).resolve()
        if resume is None:
            self.run_dir.mkdir(parents=True, exist_ok=False)
        elif read_json(self.run_dir / "status.json")["status"] == "completed":
            raise ValueError("completed run 不接受续训")
        self.phase, self.global_epoch = "initializing", 0
        self._status("running")
        try:
            self._initialize(config, run_dir, resume=resume, stage_a_checkpoint=stage_a_checkpoint)
            resource_path = self.run_dir / "resource_metrics.json"
            self.resources = read_json(resource_path) if resource_path.exists() else {
                "device": str(self.device), "parameter_count": sum(p.numel() for p in self.model.parameters()),
                "epochs": [], "initializations_seconds": []}
            self.resources["epochs"] = [row for row in self.resources["epochs"] if row["global_epoch"] <= self.global_epoch]
            self.resources["initializations_seconds"].append(time.perf_counter() - started)
            _json(resource_path, self.resources)
        except BaseException as exc:
            self._status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
            raise

    def _initialize(self, config, run_dir, *, resume=None, stage_a_checkpoint=None):
        self.config = validate_config(deepcopy(config))
        if self.config["data"]["num_workers"] != 0:
            raise ValueError("当前训练状态机仅实现 workers=0；不静默忽略 workers")
        variant = self.config["experiment"]["variant"]
        if variant in {"persistence", "seasonal_naive"}:
            raise ValueError("该变体需独立基线路由或 full 的 best_A 来源，当前训练入口不接受")
        if variant == "no_latent_loss" and resume is None and stage_a_checkpoint is None:
            raise ValueError("no_latent_loss 必须指定 full 的 best_A")
        if stage_a_checkpoint is not None and (variant != "no_latent_loss" or resume is not None):
            raise ValueError("共享 best_A 仅用于新建 no_latent_loss，不与 resume 混用")
        device = self.config["experiment"]["device"]
        if device == "mps" and not torch.backends.mps.is_available():
            raise ValueError("明确指定 MPS，但设备不可用")
        if device == "cuda" and not torch.cuda.is_available():
            raise ValueError("明确指定 CUDA，但设备不可用")
        self.run_dir = Path(run_dir).resolve()

        seed = self.config["experiment"]["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.generator = torch.Generator().manual_seed(seed)
        self.train_data = PackageDataset(self.config, "train")
        self.valid_data = PackageDataset(self.config, "valid")
        self.schema = self.train_data.episodes[0].schema
        self.schema_fingerprint = canonical_hash(self.schema)
        self.data_fingerprint = self.train_data.fingerprint
        self.generation_audit = None
        if self.config["data"]["split_mode"] == "site_time":
            self.generation_audit = read_json(Path(self.config["data"]["package"]) / "generation_audit.json")
        # 仅接受用户认可的本地训练产物；包含 Python/NumPy RNG 状态，不加载远程或不可信 pickle。
        checkpoint = None if resume is None else torch.load(resume, map_location="cpu", weights_only=False)
        shared = None if stage_a_checkpoint is None else torch.load(stage_a_checkpoint, map_location="cpu", weights_only=False)
        for saved in (checkpoint, shared):
            if saved is not None:
                validate_checkpoint_contract(saved)
        if shared is not None:
            if (shared["phase"] != "A" or shared["config"]["experiment"]["variant"] != "full"
                    or shared["schema"] != self.schema or shared["data_fingerprint"] != self.data_fingerprint
                    or shared["config"]["model"] != self.config["model"]
                    or any(shared["config"]["data"][key] != self.config["data"][key] for key in ("seq_len", "pred_len"))):
                raise ValueError("共享 best_A 的阶段/schema/数据/model/L/H 不一致")
        if checkpoint is not None:
            if (checkpoint["config"] != self.config or checkpoint["data_fingerprint"] != self.data_fingerprint
                    or checkpoint["schema"] != self.schema):
                raise ValueError("续训 config/schema/数据不一致")
            self.transform = TrainingTransform.from_dict(checkpoint["transform"], self.schema)
            if self.transform.fingerprint != checkpoint["transform_fingerprint"]:
                raise ValueError("续训 scaler/词表指纹不一致")
        elif shared is not None:
            self.transform = TrainingTransform.from_dict(shared["transform"], self.schema)
            if self.transform.fingerprint != shared["transform_fingerprint"]:
                raise ValueError("共享 best_A 的 scaler 指纹不一致")
        else:
            self.transform = TrainingTransform.fit(self.train_data)
        super().__init__(SimpleNamespace(use_gpu=device != "cpu", gpu_type=device,
                                         devices="0", use_multi_gpu=False))
        self.phases = ["direct"] if variant == "direct" else ["A", "B-short", "B-medium", "B-full"]
        if variant == "no_latent_loss":
            self.phases = ["B-short", "B-medium", "B-full"]
        self.phase = self.phases[0]
        self.phase_epoch = self.global_epoch = self.bad_epochs = 0
        self.best_score = float("inf")
        self.best_a = None
        self.latent_stats = None
        self.representation_fingerprint = None
        self.metrics = []
        self.optimizer = self._new_optimizer("A")
        self.site_sampler = SiteOriginSampler(self.train_data) if self.config["data"]["split_mode"] == "site_time" else None
        if checkpoint is not None:
            self._restore(checkpoint)
        else:
            (self.run_dir / "checkpoints").mkdir()
            _json(self.run_dir / "resolved_config.json", self.config)
            _json(self.run_dir / "schema.json", self.schema)
            _json(self.run_dir / "data_manifest.json", self.train_data.manifest)
            _json(self.run_dir / "environment.json", {"python": platform.python_version(), "torch": torch.__version__,
                                                       "numpy": np.__version__, "device": str(self.device),
                                                       "resume_contract": "epoch_boundary_workers0"})
            self._save_scalers()
            if self.site_sampler:
                _json(self.run_dir / "site_split.json", self.config["data"]["site_split"])
                _json(self.run_dir / "generation_audit.json", self.generation_audit)
                _json(self.run_dir / "metadata.json", {"schema_fingerprint": self.schema_fingerprint,
                                                       "data_fingerprint": self.data_fingerprint,
                                                       "target_contracts": {e.site["site_id"]: e.site["target_contract"] for e in self.train_data.episodes}})
            (self.run_dir / "run.log").write_text("GSLAformer programmatic training\n", encoding="utf-8")
            if shared is not None:
                self.best_a = shared["model"]
                # 保留真正的 full/A 来源，不把当前 B 状态伪装成 best_A。
                temporary = self.run_dir / "checkpoints/best_A.pt.tmp"
                torch.save(shared, temporary)
                temporary.replace(self.run_dir / "checkpoints/best_A.pt")
                self._prepare_b()
                _json(self.run_dir / "initialization.json", {"source": str(Path(stage_a_checkpoint).resolve()),
                        "source_global_epoch": shared["global_epoch"], "optimizer": "fresh_B_AdamW"})
            self._status("running")

    def _build_model(self):
        sizes = {name: {feature: len(tokens) for feature, tokens in features.items()}
                 for name, features in self.transform.state["vocab"].items()}
        return self.get_model_class("GSLAformer")(
            groups_spec=self.schema["groups"], relation_types=self.schema["relation_types"],
            model_params=self.config["model"], vocab_sizes=sizes,
            variant=self.config["experiment"]["variant"],
            known_future_dim=len(self.schema["known_future"]), timezone=self.schema["timezone"])

    def _save_scalers(self):
        for name, payload in scaling_artifacts(self.transform, self.schema).items():
            _json(self.run_dir / name, payload)

    def _new_optimizer(self, stage):
        parameters = self.model.set_training_stage(stage)
        training = self.config["training"]
        return torch.optim.AdamW(parameters, lr=training["learning_rate"], weight_decay=training["weight_decay"])

    def _batches(self, dataset, order=None):
        size = self.config["data"]["batch_size"]
        order = list(range(len(dataset))) if order is None else order
        for start in range(0, len(order), size):
            samples = [self.transform.apply(dataset[index]) for index in order[start:start + size]]
            yield collate_samples(samples, self.config).to(self.device)

    def _limit(self):
        t = self.config["training"]
        return {"A": t["stage_a_max_epochs"], "B-short": t["b_short_epochs"],
                "B-medium": t["b_medium_epochs"], "B-full": t["b_full_max_epochs"],
                "direct": t["stage_a_max_epochs"] + t["b_short_epochs"] + t["b_medium_epochs"] + t["b_full_max_epochs"]}[self.phase]

    def _phase_finished(self):
        if self.phase_epoch >= self._limit():
            return True
        t = self.config["training"]
        patience = t["stage_a_patience"] if self.phase in {"A", "direct"} else t["b_full_patience"]
        return self.phase in {"A", "direct", "B-full"} and self.bad_epochs >= patience

    def _advance(self):
        index = self.phases.index(self.phase)
        if index == len(self.phases) - 1:
            return False
        if self.phase == "A":
            self._prepare_b()
        self.phase = self.phases[index + 1]
        self.phase_epoch = self.bad_epochs = 0
        self.best_score = float("inf")
        return True

    def _prepare_b(self):
        if self.best_a is None:
            raise ValueError("缺少 best_A，不能 Freeze")
        self.model.load_state_dict(self.best_a)
        self.optimizer = self._new_optimizer("B")
        if self.model.has_latent:
            collector = collect_latent_statistics(self.model, self._batches(self.train_data),
                schema_fingerprint=self.schema_fingerprint, data_fingerprint=self.data_fingerprint)
            self.latent_stats = collector.finalize()
            self.representation_fingerprint = frozen_representation_fingerprint(self.model)
            manifest = self.run_dir / "latent_stats_manifest.json"
            if manifest.exists():
                from exp.gslaformer_support import LatentStatistics
                saved = LatentStatistics.load(self.run_dir, schema_fingerprint=self.schema_fingerprint,
                    data_fingerprint=self.data_fingerprint, representation_fingerprint=self.representation_fingerprint)
                if any(not np.array_equal(saved[k], v) for k, v in self.latent_stats.items()):
                    raise ValueError("恢复 Freeze 的 latent 统计不一致")
            else:
                collector.save(self.run_dir)

    def _epoch(self):
        self.model.train()
        horizon = self.config["data"]["pred_len"]
        steps = 1 if self.phase == "B-short" else min(3, horizon) if self.phase == "B-medium" else horizon
        order = self.site_sampler.draw(self.generator) if self.site_sampler else torch.randperm(len(self.train_data), generator=self.generator).tolist()
        updates = skipped = 0
        loss_sum = 0.0
        load_sum = latent_sum = gradient_sum = 0.0
        group_diagnostics = {}
        started = time.perf_counter()
        for batch in self._batches(self.train_data, order):
            supervision = batch.supervision
            if not supervision.target_mask[:, :steps].any():
                skipped += 1
                continue
            self.optimizer.zero_grad(set_to_none=True)
            latent = None
            if self.phase in {"A", "direct"}:
                prediction = self.model.predict_direct(batch)
                load, _ = masked_load_loss(prediction, supervision.target, supervision.target_mask)
                loss = load
            else:
                prediction = self.model.rollout(batch, steps)
                load, _ = masked_load_loss(prediction.y, supervision.target[:, :steps], supervision.target_mask[:, :steps])
                loss = load
                if self.model.has_latent and self.config["training"]["latent_weight"] > 0:
                    labels = build_latent_labels(self.model, batch.inputs, supervision)
                    labels.z, labels.quality = labels.z[:, :steps], labels.quality[:, :steps]
                    variance = np.stack([self.latent_stats[f"{name}__variance"] for name in self.model.dynamic_groups])
                    weights = latent_dimension_weights(torch.tensor(variance, dtype=torch.float32, device=self.device))
                    latent, _ = weighted_latent_loss(prediction.z, labels, weights)
                    loss = loss + self.config["training"]["latent_weight"] * latent
                    # 诊断只读 detached tensor，不延长训练图或改变 loss reduction。
                    for index, name in enumerate(self.model.dynamic_groups):
                        q = labels.quality[:, :, index:index + 1]
                        values = prediction.z.detach()[:, :, index:index + 1]
                        group_loss, counts = weighted_latent_loss(values,
                            LatentLabels(labels.z[:, :, index:index + 1], q), weights[index:index + 1])
                        selected = values[q > 0].cpu().double()
                        stats = group_diagnostics.setdefault(name, dict(loss=0., batches=0, valid_count=0,
                                                                       sum=0., square_sum=0., cells=0))
                        stats["loss"] += group_loss.item()
                        stats["batches"] += int(counts.sum() > 0)
                        stats["valid_count"] += counts.sum().item()
                        stats["sum"] += selected.sum().item()
                        stats["square_sum"] += selected.square().sum().item()
                        stats["cells"] += selected.numel()
            if not torch.isfinite(loss):
                raise ValueError("训练 loss NaN/Inf")
            loss.backward()
            gradient = torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], 1, error_if_nonfinite=True)
            self.optimizer.step()
            updates += 1
            loss_sum += loss.item()
            load_sum += load.item()
            latent_sum += 0.0 if latent is None else latent.item()
            gradient_sum += gradient.item()
            if updates == 1 or updates % 100 == 0:
                print(f"{self.phase} epoch={self.phase_epoch + 1} updates={updates} elapsed={time.perf_counter() - started:.1f}s", flush=True)
        if not updates:
            raise ValueError("整 epoch 无有效主目标")
        groups = {}
        for name, stats in group_diagnostics.items():
            cells = stats["cells"]
            groups[name] = {"valid_count": stats["valid_count"],
                            "mean_batch_loss": stats["loss"] / stats["batches"] if stats["batches"] else None,
                            "prediction_std": max(0., stats["square_sum"] / cells - (stats["sum"] / cells) ** 2) ** 0.5 if cells else None}
        return {"updates": updates, "skipped_batches": skipped, "mean_batch_loss": loss_sum / updates,
                "mean_load_loss": load_sum / updates, "mean_latent_loss": latent_sum / updates,
                "mean_gradient_norm": gradient_sum / updates, "latent_groups": groups,
                "curriculum_steps": steps}

    @torch.no_grad()
    def valid(self):
        self.model.eval()
        roles = self.config["data"].get("site_split")
        sites = set(roles["train"] + roles["heldout_valid"]) if roles else None
        metric = ValidationMSE(self.config["data"]["pred_len"], expected_sites=sites)
        scenario_metrics = {}
        self.history_target_sensitivity = None
        for batch in self._batches(self.valid_data):
            y = self.model.predict_direct(batch) if self.phase in {"A", "direct"} else self.model.rollout(batch, self.config["data"]["pred_len"]).y
            if self.history_target_sensitivity is None:
                changed = copy(batch.inputs)
                changed.target_history = torch.zeros_like(batch.inputs.target_history)
                altered = Batch(changed, None)
                alternate = self.model.predict_direct(altered) if self.phase in {"A", "direct"} else self.model.rollout(altered, self.config["data"]["pred_len"]).y
                self.history_target_sensitivity = (alternate - y).abs().mean().item()
            metric.update(y, batch.supervision.target, batch.supervision.target_mask, batch.inputs.site_ids)
            if self.site_sampler:
                for i, metadata in enumerate(batch.inputs.metadata):
                    key = batch.inputs.site_ids[i] + "/" + metadata["scenario"]
                    collector = scenario_metrics.setdefault(key, ValidationMSE(self.config["data"]["pred_len"]))
                    collector.update(y[i:i+1], batch.supervision.target[i:i+1], batch.supervision.target_mask[i:i+1], batch.inputs.site_ids[i:i+1])
        if scenario_metrics:
            _json(self.run_dir / "scenario_metrics.json", {"split": "valid", "phase": self.phase,
                   "site_scenario_macro_mse": {key: value.compute() for key, value in scenario_metrics.items()}})
        return metric.compute()

    def _status(self, status, error=None):
        _json(self.run_dir / "status.json", {"status": status, "phase": self.phase,
                                            "global_epoch": self.global_epoch, "error": error})

    def _checkpoint(self, filename):
        stats_hash = None if self.latent_stats is None else canonical_hash({k: v.tolist() for k, v in self.latent_stats.items()})
        rng = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
               "sampler": self.generator.get_state(), "device": None}
        if self.device.type == "mps":
            rng["device"] = torch.mps.get_rng_state()
        elif self.device.type == "cuda":
            rng["device"] = torch.cuda.get_rng_state_all()
        payload = {"version": 2, "computation_version": self.model.COMPUTATION_VERSION,
                   "frozen_parameters": sorted(name for name, p in self.model.named_parameters() if not p.requires_grad) if self.model._representation_frozen else [],
                   "latent_stats_manifest": read_json(self.run_dir / "latent_stats_manifest.json") if self.latent_stats is not None else None,
                   "config": self.config, "schema": self.schema,
                   "generation_audit": self.generation_audit,
                   "training_sites": sorted({self.train_data.entries[e]["site_id"] for e, _ in self.train_data.windows}),
                   "data_fingerprint": self.data_fingerprint, "model": _cpu_state(self.model),
                   "transform": self.transform.to_dict(), "transform_fingerprint": self.transform.fingerprint,
                   "phase": self.phase, "phase_epoch": self.phase_epoch, "global_epoch": self.global_epoch,
                   "bad_epochs": self.bad_epochs, "best_score": self.best_score, "best_a": self.best_a,
                   "optimizer": self.optimizer.state_dict(), "rng": rng, "metrics": self.metrics,
                   "latent_stats": self.latent_stats, "latent_stats_fingerprint": stats_hash,
                   "representation_fingerprint": self.representation_fingerprint}
        path = self.run_dir / "checkpoints" / filename
        temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def _restore(self, saved):
        if saved["version"] != 2 or saved["phase"] not in self.phases:
            raise ValueError("checkpoint 阶段/版本非法")
        for key in ("phase", "phase_epoch", "global_epoch", "bad_epochs", "best_score", "best_a", "metrics", "latent_stats", "representation_fingerprint"):
            setattr(self, key, saved[key])
        self.model.load_state_dict(saved["model"])
        self.optimizer = self._new_optimizer("B" if self.phase.startswith("B-") else "A")
        validate_checkpoint_contract(saved, self.model)
        self.optimizer.load_state_dict(saved["optimizer"])
        if self.phase.startswith("B-") and self.model.has_latent:
            if read_json(self.run_dir / "latent_stats_manifest.json") != saved["latent_stats_manifest"]:
                raise ValueError("续训 latent 溯源旁车与 checkpoint 不一致")
            if frozen_representation_fingerprint(self.model) != self.representation_fingerprint:
                raise ValueError("冻结链指纹不一致")
            actual = canonical_hash({k: v.tolist() for k, v in self.latent_stats.items()})
            if actual != saved["latent_stats_fingerprint"]:
                raise ValueError("latent 统计指纹不一致")
        random.setstate(saved["rng"]["python"])
        np.random.set_state(saved["rng"]["numpy"])
        torch.set_rng_state(saved["rng"]["torch"])
        self.generator.set_state(saved["rng"]["sampler"])
        if self.device.type == "mps":
            torch.mps.set_rng_state(saved["rng"]["device"])
        elif self.device.type == "cuda":
            torch.cuda.set_rng_state_all(saved["rng"]["device"])

    def train(self, *, max_epochs_this_call=None):
        """可在明确 epoch 边界暂停；续训必须传 last checkpoint，不能只重载权重。"""
        if max_epochs_this_call is not None and (type(max_epochs_this_call) is not int or max_epochs_this_call < 1):
            raise ValueError("max_epochs_this_call 必须为正整数")
        self._status("running")
        completed_here = 0
        try:
            while True:
                if self._phase_finished():
                    previous_phase = self.phase
                    started = time.perf_counter()
                    advanced = self._advance()
                    synchronize(self.device)
                    self.resources.setdefault("phase_transitions", []).append({
                        "from": previous_phase, "to": self.phase,
                        "seconds": time.perf_counter() - started, "global_epoch": self.global_epoch})
                    _json(self.run_dir / "resource_metrics.json", self.resources)
                    if not advanced:
                        for name in ("resolved_config.json", "schema.json", "data_manifest.json", "environment.json",
                                     "training_metrics.json", "scalers.json", "category_vocab.json"):
                            read_json(self.run_dir / name)
                        # 自己刚生成的可信本地 checkpoint：完成前确证最终 best 来源。
                        best = torch.load(self.run_dir / "checkpoints/best.pt", map_location="cpu", weights_only=False)
                        if best["phase"] != self.phase:
                            raise ValueError("最终 best 不属于最终选优阶段")
                        validate_artifacts(self.run_dir, "train", self.config["experiment"]["variant"],
                                           aidc=self.site_sampler is not None)
                        for name, expected in {"schema.json": self.schema,
                                               **scaling_artifacts(self.transform, self.schema)}.items():
                            if read_json(self.run_dir / name) != expected:
                                raise ValueError(f"{name} 与训练状态不一致")
                        if self.model.has_latent and self.phase != "direct":
                            from exp.gslaformer_support import LatentStatistics
                            LatentStatistics.load(self.run_dir, schema_fingerprint=self.schema_fingerprint,
                                data_fingerprint=self.data_fingerprint, representation_fingerprint=self.representation_fingerprint)
                        self._status("completed")
                        return self.metrics
                synchronize(self.device)
                started = time.perf_counter()
                row = self._epoch()
                synchronize(self.device)
                train_seconds = time.perf_counter() - started
                started = time.perf_counter()
                score = self.valid()  # 短课程也监控完整 H，但不选最终模型。
                synchronize(self.device)
                valid_seconds = time.perf_counter() - started
                improved = score < self.best_score - self.config["training"]["min_delta"]
                self.bad_epochs = 0 if improved else self.bad_epochs + 1
                if improved:
                    self.best_score = score
                self.phase_epoch += 1
                self.global_epoch += 1
                completed_here += 1
                row.update(phase=self.phase, phase_epoch=self.phase_epoch, global_epoch=self.global_epoch,
                           valid_mse=score, history_target_sensitivity=self.history_target_sensitivity)
                self.metrics.append(row)
                started = time.perf_counter()
                if improved and self.phase == "A":
                    self.best_a = _cpu_state(self.model)
                    self._checkpoint("best_A.pt")
                if improved and self.phase in {"B-full", "direct"}:
                    self._checkpoint("best.pt")
                self._checkpoint("last.pt")
                _json(self.run_dir / "training_metrics.json", self.metrics)
                self.resources["epochs"].append({"global_epoch": self.global_epoch, "phase": self.phase,
                    "updates": row["updates"], "train_seconds": train_seconds, "valid_seconds": valid_seconds,
                    "save_seconds": time.perf_counter() - started})
                _json(self.run_dir / "resource_metrics.json", self.resources)
                with (self.run_dir / "run.log").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                if max_epochs_this_call is not None and completed_here >= max_epochs_this_call:
                    self._status("interrupted")
                    return self.metrics
        except BaseException as exc:
            self._status("interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", str(exc))
            raise
