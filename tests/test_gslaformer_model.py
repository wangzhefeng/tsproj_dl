"""I2 空间链模型断言：轨迹、shape、mask、空组、同步重排、新实体、关系方向与跨样本独立。"""
import copy
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_provider.spatial_temporal.batching import collate_samples
from data_provider.spatial_temporal.config import load_config
from data_provider.spatial_temporal.dataset import PackageDataset
from data_provider.spatial_temporal.scaling import TrainingTransform
from layers.gslaformer.relations import RelationAttention
from layers.gslaformer.temporal import calendar_features
from models.transformer.GSLAformer import Model, RolloutState
from test_gslaformer_package import small_config, write_package

ROOT = Path(__file__).resolve().parents[1]


def real_batch(limit=2, variant="full"):
    """base 夹具真数据 → 已 apply 的样本列表与模型构造要素。"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "episode"
        config = small_config()
        config["data"]["known_future"] = []  # 夹具不含计划记录，声明须同步清空
        write_package(path, config)
        config["data"]["package"] = str(path)
        dataset = PackageDataset(config, "train")
        transform = TrainingTransform.fit(dataset)
        samples = [transform.apply(dataset[index]) for index in range(limit)]
    groups_spec = config["data"]["groups"]
    vocab = {group: {feature: len(tokens) for feature, tokens in features.items()}
             for group, features in transform.state["vocab"].items()}
    model = Model(groups_spec=groups_spec, relation_types=config["data"]["relation_types"],
                  model_params=config["model"], vocab_sizes=vocab, variant=variant)
    model.eval()
    return model, samples, config


def mini_sample(values, present=None, edges=(), *, relation_names=("belongs",),
                static_values=None, categories=None, horizon=2, known=None, known_mask=None):
    """手工构造已 apply 形态的单组动态样本（含一个可选空静态组）。"""
    length, nodes = values.shape
    observed = np.ones((length, nodes, 1), bool)
    present_flag = np.ones((length, nodes), bool) if present is None else present
    static = np.zeros((length, nodes, 1), np.float32) if static_values is None else static_values
    static_mask = np.ones_like(static, bool)
    category = np.full((length, nodes, 1), 2, np.int64) if categories is None else categories
    known_dim = 0 if known is None else known.shape[1]
    future_time = (length - 1 + np.arange(1, horizon + 1, dtype=np.int64)) * 3600
    groups = {
        "assets": {"values": values[..., None].astype(np.float32), "observed": observed,
                   "present": present_flag, "applicable": np.ones((nodes, 1), bool),
                   "padding": np.zeros(nodes, bool), "static_numeric_history": static,
                   "static_numeric_mask": static_mask, "static_category_history": category,
                   "entity_ids": [f"a{i}" for i in range(nodes)]},
        "zones": {"values": np.zeros((length, 0, 0), np.float32),
                  "observed": np.zeros((length, 0, 0), bool),
                  "present": np.ones((length, 0), bool), "applicable": np.zeros((0, 0), bool),
                  "padding": np.zeros(0, bool),
                  "static_numeric_history": np.zeros((length, 0, 0), np.float32),
                  "static_numeric_mask": np.zeros((length, 0, 0), bool),
                  "static_category_history": np.zeros((length, 0, 0), np.int64),
                  "entity_ids": []},
    }
    relations = [[(sg, si, dg, di, kind) for sg, si, dg, di, kind in edges] for _ in range(length)]
    inputs = {"groups": groups, "relations_history": relations, "relations_origin": relations[-1],
              "target_history": np.zeros((length, 1), np.float32),
              "target_history_observed": np.ones((length, 1), bool),
              "history_time": np.arange(length, dtype=np.int64) * 3600,
              "future_time": future_time,
              "future_known": (np.zeros((horizon, known_dim), np.float32) if known is None
                               else known.astype(np.float32)),
              "future_known_mask": (np.zeros((horizon, known_dim), bool) if known_mask is None
                                    else known_mask),
              "origin": 3600 * (length - 1), "site_id": "synthetic_site", "target_contract": {}}
    return {"inputs": inputs, "supervision": None,
            "metadata": {"schema_fingerprint": "", "data_fingerprint": ""}}


MINI_GROUPS = [{"name": "assets", "dynamic": ["load"], "static_numeric": ["capacity"],
                "static_category": ["kind"]},
               {"name": "zones", "dynamic": [], "static_numeric": [], "static_category": []}]
MINI_RELATIONS = [{"name": "belongs", "src_groups": ["assets"], "dst_groups": ["zones"],
                   "acyclic": True, "reverse_message": True}]
MINI_PARAMS = {"spatial_dim": 16, "latent_dim": 12, "temporal_dim": 24,
               "spatial_heads": 2, "relation_heads": 2, "temporal_heads": 2,
               "spatial_layers": 1, "entity_history_layers": 1, "relation_layers": 2,
               "temporal_layers": 2, "future_path_layers": 1, "dropout": 0.0,
               "category_embedding_dim": 8, "lead_dim": 16}
MINI_VOCAB = {"assets": {"kind": 4}, "zones": {}}


def mini_model(variant="full", known_future_dim=0):
    model = Model(groups_spec=MINI_GROUPS, relation_types=MINI_RELATIONS,
                  model_params=MINI_PARAMS, vocab_sizes=MINI_VOCAB, variant=variant,
                  known_future_dim=known_future_dim)
    model.eval()
    return model


def mini_config_dict(seq_len=3, pred_len=2):
    return {"data": {"groups": MINI_GROUPS, "relation_types": MINI_RELATIONS,
                     "batch_size": 1, "known_future": [], "seq_len": seq_len,
                     "pred_len": pred_len},
            "model": dict(MINI_PARAMS),
            "limits": {"max_total_nodes": 16, "max_explicit_attention_bytes": 268435456}}


class SpatialModelTests(unittest.TestCase):
    def test_reparenting_reaches_final_prediction_without_degree_shortcut(self):
        with torch.random.fork_rng():
            torch.manual_seed(2023)
            model, samples, config = real_batch(limit=1)
        first = samples[0]
        first["supervision"] = None
        first["inputs"]["groups"]["assets"]["values"][:, 1] += 3
        changed = copy.deepcopy(first)
        changed["inputs"]["relations_history"][-1] = [
            (sg, si, dg, 1 - di, kind) for sg, si, dg, di, kind
            in changed["inputs"]["relations_history"][-1]]
        changed["inputs"]["relations_origin"] = changed["inputs"]["relations_history"][-1]
        left, right = [collate_samples([sample], config) for sample in (first, changed)]
        with torch.no_grad():
            # 固定、可区分属性和权重；静态度数不变，只有端点配对变化。
            a, b = model.encode_history(left), model.encode_history(right)
            torch.testing.assert_close(a.c, b.c, atol=1e-6, rtol=1e-5)
            self.assertGreater((a.z - b.z).abs().max().item(), 1e-5)
            self.assertFalse(torch.allclose(model.rollout(left, 2).y, model.rollout(right, 2).y,
                                            atol=1e-6, rtol=1e-5))
            with torch.random.fork_rng():
                torch.manual_seed(2023)
                negative, _, _ = real_batch(limit=1, variant="no_relation_all")
            torch.testing.assert_close(negative.rollout(left, 2).y, negative.rollout(right, 2).y,
                                       atol=1e-6, rtol=1e-5)

    def test_shapes_and_finiteness(self):
        model, samples, config = real_batch()
        batch = collate_samples(samples, config)
        with torch.no_grad():
            history = model.encode_history(batch)
        self.assertEqual(tuple(history.z.shape), (2, config["data"]["seq_len"], 1, 32))
        self.assertEqual(tuple(history.observed_coverage.shape), (2, config["data"]["seq_len"], 1))
        self.assertEqual(tuple(history.c.shape), (2, 64))
        self.assertFalse(history.generated.any())
        for tensor in (history.z, history.c):
            self.assertTrue(torch.isfinite(tensor).all().item())
        self.assertEqual(tuple(history.entity_memory["assets"].shape), (2, 2, 32))

    def test_synchronized_permutation_invariance(self):
        model, samples, config = real_batch(limit=2)
        permuted = copy.deepcopy(samples[0])

        def swap(group, tensor, axis_node):
            if tensor.shape[axis_node] == 2:
                index = [1, 0] if axis_node == 1 else None
                if index is not None:
                    tensor[index] = tensor[[1, 0]]

        assets = permuted["inputs"]["groups"]["assets"]
        for key in ("values", "observed", "present", "static_numeric_history",
                    "static_numeric_mask", "static_category_history"):
            assets[key] = assets[key][:, [1, 0]]
        assets["applicable"] = assets["applicable"][[1, 0]]
        assets["entity_ids"] = [assets["entity_ids"][1], assets["entity_ids"][0]]
        remap = lambda edges: [(sg, 1 - si if sg == "assets" else si,
                                dg, 1 - di if dg == "assets" else di, kind)
                               for sg, si, dg, di, kind in edges]
        permuted["inputs"]["relations_history"] = [remap(step) for step in
                                                   permuted["inputs"]["relations_history"]]
        permuted["inputs"]["relations_origin"] = remap(permuted["inputs"]["relations_origin"])
        with torch.no_grad():
            first = model.encode_history(collate_samples([samples[0]], config))
            second = model.encode_history(collate_samples([permuted], config))
        torch.testing.assert_close(first.z, second.z, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(first.c, second.c, atol=1e-5, rtol=1e-5)

    def test_id_rename_invariance(self):
        model, samples, config = real_batch(limit=1)
        renamed = copy.deepcopy(samples[0])
        renamed["inputs"]["groups"]["assets"]["entity_ids"] = ["x0", "x1"]
        with torch.no_grad():
            first = model.encode_history(collate_samples(samples, config))
            second = model.encode_history(collate_samples([renamed], config))
        torch.testing.assert_close(first.z, second.z, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first.c, second.c, atol=1e-6, rtol=1e-6)

    def test_new_entity_changes_shape_not_parameters(self):
        model, samples, config = real_batch(limit=1)
        extended = copy.deepcopy(samples[0])
        extended["supervision"] = None  # 此用例只验证新增实体无标签输入。
        assets = extended["inputs"]["groups"]["assets"]
        for key in ("values", "observed", "present", "static_numeric_history",
                    "static_numeric_mask", "static_category_history"):
            assets[key] = np.concatenate([assets[key], assets[key][:, -1:]], axis=1)
        assets["applicable"] = np.concatenate([assets["applicable"], assets["applicable"][-1:]])
        assets["entity_ids"] = assets["entity_ids"] + ["asset_new"]
        before = {name: tuple(p.shape) for name, p in model.named_parameters()}
        with torch.no_grad():
            history = model.encode_history(collate_samples([extended], config))
        after = {name: tuple(p.shape) for name, p in model.named_parameters()}
        self.assertEqual(before, after)
        self.assertEqual(history.z.shape[2], 1)  # Gd 由 schema 决定，不随实体数变化
        self.assertEqual(history.entity_memory["assets"].shape[1], 3)
        self.assertTrue(torch.isfinite(history.z).all().item())

        # 新设备的真实观测和静态属性必须能到达最终 y，不只检验 shape。
        batch = collate_samples([extended], config)
        for key in ("values", "static_numeric_history"):
            value = batch.inputs.groups["assets"][key]
            value.requires_grad_(True)
            y = model.rollout(batch, config["data"]["pred_len"]).y
            gradient, = torch.autograd.grad(y.sum(), value)
            self.assertGreater(gradient[:, :, -1].abs().sum().item(), 1e-8)
            value.requires_grad_(False)

    def test_cross_sample_independence(self):
        model, samples, config = real_batch(limit=2)
        # 第二样本裁掉一个实体，制造不同 N 与 padding
        trimmed = copy.deepcopy(samples[1])
        trimmed["supervision"] = None
        samples[0]["supervision"] = None
        assets = trimmed["inputs"]["groups"]["assets"]
        for key in ("values", "observed", "present", "static_numeric_history",
                    "static_numeric_mask", "static_category_history"):
            assets[key] = assets[key][:, :1]
        assets["applicable"] = assets["applicable"][:1]
        assets["entity_ids"] = assets["entity_ids"][:1]
        trimmed["inputs"]["relations_history"] = [
            [e for e in step if e[1] == 0] for step in trimmed["inputs"]["relations_history"]]
        trimmed["inputs"]["relations_origin"] = trimmed["inputs"]["relations_history"][-1]
        with torch.no_grad():
            joint = model.encode_history(collate_samples([samples[0], trimmed], config))
            solo = model.encode_history(collate_samples([samples[0]], config))
        torch.testing.assert_close(joint.z[0], solo.z[0], atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(joint.c[0], solo.c[0], atol=1e-5, rtol=1e-5)

    def test_relation_direction_and_bias_accumulation(self):
        attention = RelationAttention(
            [{"name": "belongs", "src_groups": ["assets"], "dst_groups": ["zones"],
              "acyclic": True, "reverse_message": True},
             {"name": "cools", "src_groups": ["assets"], "dst_groups": ["assets"],
              "acyclic": False, "reverse_message": False}], 16, 2)
        edges = [[(0, 1, 0), (0, 1, 1)], []]
        effective = torch.ones(2, 3, dtype=torch.bool)
        additive, bias = attention.build_masks(3, edges, effective)
        allow = torch.isfinite(additive[:, 0])
        self.assertTrue(allow[0, 0, 1].item() and allow[0, 1, 0].item())  # 正向与 reverse 反向
        self.assertFalse(allow[0, 0, 2].item() and allow[0, 2, 0].item() is True and
                         allow[0, 1, 2].item())
        # 同端点多类型偏置相加：belongs 正向 + cools 正向
        expected = (attention.direction_bias[0, 0] + attention.direction_bias[1, 0])
        torch.testing.assert_close(bias[0, :, 1, 0], expected)
        torch.testing.assert_close(bias[0, :, 0, 1], attention.direction_bias[0, 1])

    def test_invalid_rows_keep_only_self_and_zero_output(self):
        model = mini_model()
        sample = mini_sample(np.array([[1.0, 2.0, 3.0]]),
                             present=np.array([[True, False, True]]))
        batch = collate_samples([sample], mini_config_dict(seq_len=1))
        attention = model.relation
        frames = 1
        edges = model._flatten_edges(batch)
        effective = (batch.inputs.groups["assets"]["present"][0]
                     & ~batch.inputs.groups["assets"]["padding"][0]).reshape(frames, -1)
        additive, _ = attention.build_masks(3, edges, effective)
        allow = torch.isfinite(additive[0, 0])
        self.assertTrue(allow[1, 1].item())          # 无效行保留 self 数值安全通路
        self.assertFalse(allow[0, 1].item())         # 有效 query 不读无效 key
        self.assertFalse(allow[1, 0].item())
        tokens = torch.randn(frames, 3, model.d_s)
        with torch.no_grad():
            output = attention(tokens, edges, effective)
        self.assertEqual(float(output[:, 1].abs().max()), 0.0)  # 无效输出置零

    def test_empty_group_and_no_edge_produce_finite_output(self):
        model = mini_model(variant="direct")
        sample = mini_sample(np.array([[1.0], [2.0]]))  # zones 空组、无边（非 full 允许）
        batch = collate_samples([sample], mini_config_dict(seq_len=2))
        with torch.no_grad():
            history = model.encode_history(batch)
        self.assertTrue(torch.isfinite(history.z).all().item())
        self.assertTrue(torch.isfinite(history.c).all().item())

    def test_gru_present_reset_semantics(self):
        model = mini_model()
        values = np.array([[1.0, 5.0], [2.0, 5.0], [3.0, 5.0]])
        alive = mini_sample(values)
        interrupted = mini_sample(values, present=np.array([[True, True], [True, False],
                                                            [True, True]]))
        with torch.no_grad():
            first = model.encode_history(collate_samples([alive], mini_config_dict()))
            second = model.encode_history(collate_samples([interrupted], mini_config_dict()))
        # 中断实体重启后末端 memory 与连续实体不同（状态被清零）
        self.assertGreater(float((first.entity_memory["assets"][0, 1]
                                  - second.entity_memory["assets"][0, 1]).abs().max()), 1e-6)

    def test_whole_trajectory_swap_preserves_latent(self):
        """整段同步交换是合法置换；真正 M1 单时刻交换见 contract_repairs。"""
        model = mini_model()
        left = mini_sample(np.array([[1.0, 3.0], [3.0, 1.0], [2.0, 2.0]]))
        right = mini_sample(np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]]))
        with torch.no_grad():
            first = model.encode_history(collate_samples([left], mini_config_dict()))
            second = model.encode_history(collate_samples([right], mini_config_dict()))
        # GRU memory 按实体轴保留轨迹；同步重排（整段交换）才是不变量
        self.assertGreater(float((first.entity_memory["assets"]
                                  - second.entity_memory["assets"]).abs().max()), 1e-3)
        # 此用例仅保护整段置换不变性，不作为 M1 轨迹区分证据。
        torch.testing.assert_close(first.z, second.z, atol=1e-5, rtol=1e-5)

    def test_no_entity_history_variant_runs(self):
        model = mini_model(variant="no_entity_history")
        sample = mini_sample(np.array([[1.0, 2.0], [2.0, 1.0]]))
        batch = collate_samples([sample], mini_config_dict(seq_len=2))
        with torch.no_grad():
            history = model.encode_history(batch)
        self.assertTrue(torch.isfinite(history.z).all().item())
        self.assertEqual(model.use_gru, False)

    def test_no_spatial_and_no_relation_variants_run(self):
        for variant in ("no_spatial", "no_relation_message", "no_relation_all"):
            model = mini_model(variant=variant)
            sample = mini_sample(np.array([[1.0, 2.0], [2.0, 1.0]]))
            batch = collate_samples([sample], mini_config_dict(seq_len=2))
            with torch.no_grad():
                history = model.encode_history(batch)
            self.assertTrue(torch.isfinite(history.z).all().item(), variant)


class TemporalRolloutTests(unittest.TestCase):
    """I3：因果、条件公平性、泄漏保护、回灌梯度、双头索引与计算复用。"""

    def _batch(self, horizon=2, known=None, known_mask=None, seq_len=3):
        sample = mini_sample(np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]]),
                             horizon=horizon, known=known, known_mask=known_mask)
        return collate_samples([sample], mini_config_dict(seq_len=seq_len, pred_len=horizon))

    def test_rollout_shapes_and_generated_flags(self):
        model = mini_model()
        batch = self._batch(horizon=2)
        with torch.no_grad():
            result = model.rollout(batch, 2)
        self.assertEqual(tuple(result.y.shape), (1, 2, 1))
        self.assertEqual(tuple(result.z.shape), (1, 2, 1, 12))  # Gd=1
        self.assertTrue(result.generated.all())
        self.assertTrue(torch.isfinite(result.y).all() and torch.isfinite(result.z).all())
        self.assertEqual(result.metadata["horizon"], 2)

    def test_history_window_is_causal(self):
        model = mini_model()
        batch = self._batch()
        calendar = calendar_features(batch.inputs.history_time, "UTC")
        tokens = model._window_tokens(
            torch.randn(1, 3, 1, 12), batch.inputs.target_history,
            batch.inputs.target_history_observed, torch.zeros(1, 3, dtype=torch.bool),
            torch.zeros(1, 3, 1), torch.zeros(1, 3, 1), calendar)
        with torch.no_grad():
            first = model.temporal(tokens)
            mutated = tokens.clone()
            mutated[:, -1] += 5.0
            second = model.temporal(mutated)
        torch.testing.assert_close(first[:, :-1], second[:, :-1], atol=1e-6, rtol=1e-5)
        self.assertGreater(float((first[:, -1] - second[:, -1]).abs().max()), 1e-6)

    def test_direct_and_rollout_read_same_full_condition(self):
        known = np.array([[0.5], [0.0]])
        mask = np.ones_like(known, bool)
        model = mini_model(known_future_dim=1)
        batch = self._batch(known=known, known_mask=mask)
        with torch.no_grad():
            direct = model.predict_direct(batch)
            result = model.rollout(batch, 2)
        self.assertEqual(tuple(direct.shape), (1, 2, 1))
        # 只改目标步 2 之前的计划：direct 与 rollout 的第二步都应有合法梯度路径
        for name in ("direct", "rollout"):
            probe = copy.deepcopy(batch)
            probe.inputs.future_known = torch.tensor([[[1.5], [0.0]]], requires_grad=True)
            output = model.predict_direct(probe) if name == "direct" else model.rollout(probe, 2)
            value = output[0, 1, 0] if name == "direct" else output.y[0, 1, 0]
            grad = torch.autograd.grad(value, probe.inputs.future_known)[0]
            self.assertGreater(float(grad[0, 0].abs().max()), 1e-8, name)

    def test_mask_false_plan_values_are_ignored(self):
        model = mini_model(known_future_dim=1)
        mask = np.array([[True], [False]])
        first = self._batch(known=np.array([[0.5], [0.0]]), known_mask=mask)
        second = self._batch(known=np.array([[0.5], [9.9]]), known_mask=mask)
        with torch.no_grad():
            left = model.rollout(first, 2)
            right = model.rollout(second, 2)
        torch.testing.assert_close(left.y, right.y, atol=1e-6, rtol=1e-6)

    def test_label_change_does_not_affect_rollout(self):
        model = mini_model()
        batch = self._batch()
        batch.supervision = type("S", (), {"target": torch.full((1, 2, 1), 999.0)})()
        with torch.no_grad():
            first = model.rollout(batch, 2)
        batch.supervision.target.fill_(-999.0)
        with torch.no_grad():
            second = model.rollout(batch, 2)
        torch.testing.assert_close(first.y, second.y, atol=1e-7, rtol=1e-7)

    def test_history_encoder_runs_once_per_rollout(self):
        model = mini_model()
        calls = {"count": 0}
        original = model.history.forward

        def counted(*args, **kwargs):
            calls["count"] += 1
            return original(*args, **kwargs)

        model.history.forward = counted
        future_calls = []
        handle = model.future_path.register_forward_hook(lambda *args: future_calls.append(True))
        self.addCleanup(handle.remove)
        batch = self._batch(horizon=4)
        with torch.no_grad():
            model.rollout(batch, 4)
        self.assertEqual(calls["count"], 1)  # 历史编码一次，窗口移位不重跑
        self.assertEqual(len(future_calls), 1)
        with torch.no_grad():
            model.rollout(batch, 4)
        self.assertEqual(len(future_calls), 2)  # 第二个请求不复用旧图。

    def test_rollout_keeps_autograd_path_across_steps(self):
        """回灌梯度：第二步预测能沿第一步生成 z 与 y 回传，不发生意外 detach。"""
        model = mini_model()
        batch = self._batch(horizon=3)
        captured = []
        original = model._heads

        def spy(fused):
            prediction = original(fused)
            captured.append(prediction)
            return prediction

        model._heads = spy
        result = model.rollout(batch, 3)
        self.assertEqual(len(captured), 3)
        for prediction in captured:
            self.assertIsNotNone(prediction.y.grad_fn)
            self.assertIsNotNone(prediction.z.grad_fn)
        # 直接对第一步输出求导，避免只检查共享 head 参数梯度而漏掉 detach。
        gy, gz = torch.autograd.grad(result.y[:, 1].sum(),
                                    (captured[0].y, captured[0].z), retain_graph=True)
        self.assertGreater(gy.abs().max().item(), 1e-8)
        self.assertGreater(gz.abs().max().item(), 1e-8)
        result.y[:, -1].sum().backward()
        self.assertIsNotNone(model.load_head.weight.grad)
        self.assertGreater(float(model.load_head.weight.grad.abs().max()), 0.0)

    def test_direct_variant_shares_structure_with_latent_head(self):
        """direct 与 full 结构一致（§4.1：latent 头建立但不用于 A／direct 的 loss）。"""
        model = mini_model(variant="direct")
        batch = self._batch()
        with torch.no_grad():
            direct = model.predict_direct(batch)
            result = model.rollout(batch, 2)
        self.assertEqual(tuple(direct.shape), (1, 2, 1))
        self.assertEqual(tuple(result.z.shape), (1, 2, 1, 12))
        self.assertTrue(hasattr(model, "latent_heads"))
        self.assertEqual(model.use_spatial, True)  # 「完整关系空间编码」

    def test_predict_step_matches_rollout_step(self):
        """predict_step 与 rollout 用同一窗口/条件口径：逐步调用结果一致。"""
        model = mini_model()
        batch = self._batch(horizon=2)
        with torch.no_grad():
            result = model.rollout(batch, 2)
            history = model.encode_history(batch)
            path = model.encode_future_path(batch)
            calendar = calendar_features(batch.inputs.history_time, "UTC")
            state = RolloutState(
                z=history.z, y=batch.inputs.target_history,
                y_observed=batch.inputs.target_history_observed, y_generated=history.generated,
                coverage=history.observed_coverage,
                group_generated=torch.zeros_like(history.observed_coverage),
                calendar=calendar, condition=history.c)
            leads = model._leads(2, 1, history.z.device)
            first = model.predict_step(state, path[:, 0], leads[:, 0])
        torch.testing.assert_close(first.y, result.y[:, 0], atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
