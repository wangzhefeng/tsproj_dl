"""I2/I3 独立契约回归：方向、单元覆盖、时间与消融信息边界。"""
import copy
import math
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from data_provider.spatial_temporal.batching import collate_samples
from data_provider.spatial_temporal.schema import orphan_entity_count, quality_flags
from layers.gslaformer.relations import RelationAttention
from layers.gslaformer.temporal import calendar_features
from layers.gslaformer.temporal import FuturePathEncoder
from layers.gslaformer.spatial import StaticConditionEncoder
from test_gslaformer_model import mini_model, mini_sample, mini_config_dict, MINI_PARAMS, MINI_GROUPS
from data_provider.spatial_temporal.dataset import check_attention_budget
from models.transformer.GSLAformer import Model


class ContractRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)

    def test_one_way_message_reaches_receiver_not_sender(self):
        layer = RelationAttention([{"reverse_message": False}], 8, 2, layers=1).eval()
        x = torch.randn(1, 3, 8, requires_grad=True)
        out = layer(x, [[(0, 1, 0)]], torch.ones(1, 3, dtype=torch.bool))
        receiver = torch.autograd.grad(out[0, 1, 0], x, retain_graph=True)[0]
        sender = torch.autograd.grad(out[0, 0, 0], x)[0]
        self.assertGreater(receiver[0, 0].abs().max().item(), 1e-7)
        self.assertEqual(sender[0, 1].abs().max().item(), 0)
        self.assertEqual(receiver[0, 2].abs().max().item(), 0)

    def test_relation_float_bias_is_identical_in_training_and_inference(self):
        layer = RelationAttention([{"reverse_message": True}], 16, 2, dropout=0)
        tokens = torch.randn(1, 3, 16)
        effective = torch.ones(1, 3, dtype=torch.bool)
        layer.train()
        training = layer(tokens, [[(0, 1, 0)]], effective)
        layer.eval()
        with torch.no_grad():
            inference = layer(tokens, [[(0, 1, 0)]], effective)
        torch.testing.assert_close(training, inference, atol=1e-6, rtol=1e-5)

    def test_coverage_counts_feature_cells(self):
        model = mini_model()
        tensors = {"present": torch.ones(1, 1, 1, dtype=torch.bool),
                   "padding": torch.zeros(1, 1, dtype=torch.bool),
                   "applicable": torch.ones(1, 1, 2, dtype=torch.bool),
                   "observed": torch.tensor([[[[True, False]]]])}
        torch.testing.assert_close(model._group_coverage("assets", tensors), torch.tensor([[0.5]]))
        tensors["present"].fill_(False)
        self.assertEqual(model._group_coverage("assets", tensors).item(), 0)

    def test_calendar_minutes_seconds_timezone_and_device(self):
        for stamp, zone, seconds in ((1800, "UTC", 1800), (1831, "UTC", 1831),
                                     (1800, "Asia/Shanghai", 30600)):
            times = torch.tensor([[stamp]])
            actual = calendar_features(times, zone)
            expected = torch.tensor([math.sin(2 * math.pi * seconds / 86400),
                                     math.cos(2 * math.pi * seconds / 86400)])
            torch.testing.assert_close(actual[0, 0, :2], expected)
            self.assertEqual(actual.device, times.device)

    def test_total_only_ignores_internal_inputs_and_has_no_latents(self):
        model = mini_model("total_only")
        sample = mini_sample(np.array([[1., 3.], [1., 3.], [2., 4.]]))
        batch = collate_samples([sample], mini_config_dict())
        changed = copy.deepcopy(batch)
        changed.inputs.groups = {}
        changed.inputs.relations_history = []
        changed.inputs.relations_origin = []
        with torch.no_grad():
            a = model.rollout(batch, 2)
            b = model.rollout(changed, 2)
            torch.testing.assert_close(a.y, b.y)
            torch.testing.assert_close(model.predict_direct(batch), model.predict_direct(changed))
        self.assertEqual(a.z.shape, (1, 2, 0, model.d_z))
        self.assertFalse(hasattr(model, "latent_heads"))

    def test_no_spatial_readout_is_masked_mean(self):
        model = mini_model("no_spatial")
        pool = model.readout["assets"]
        x = torch.randn(1, 3, model.d_s, requires_grad=True)
        y = pool(x, torch.tensor([[True, True, False]]), torch.ones(1))
        gradient = torch.autograd.grad(y[0, 0], x)[0]
        torch.testing.assert_close(gradient[0, 0], gradient[0, 1], atol=1e-7, rtol=1e-6)
        self.assertEqual(gradient[0, 2].abs().max().item(), 0)
        self.assertGreater(gradient[0, 0].abs().max().item(), 1e-7)
        empty = pool(x[:, :0], torch.zeros(1, 0, dtype=torch.bool), torch.zeros(1))
        self.assertTrue(torch.isfinite(empty).all())

    def test_m1_origin_swap_is_not_whole_trajectory_permutation(self):
        a = mini_sample(np.array([[1., 3.], [1., 3.], [2., 4.]]))
        b = mini_sample(np.array([[1., 3.], [1., 3.], [4., 2.]]))
        left, right = (collate_samples([s], mini_config_dict()) for s in (a, b))
        model = mini_model()
        with torch.no_grad():
            za, zb = model.encode_history(left).z, model.encode_history(right).z
        self.assertGreater((za[:, -1] - zb[:, -1]).abs().max().item(), 1e-6)
        control = mini_model("no_entity_history")
        with torch.no_grad():
            torch.testing.assert_close(control.encode_history(left).z,
                                       control.encode_history(right).z, atol=1e-6, rtol=1e-5)

    def test_repaired_variants_long_rollout_backward(self):
        for variant in ("full", "no_spatial", "total_only"):
            for horizon in (6, 24, 72):
                with self.subTest(variant=variant, horizon=horizon):
                    model = mini_model(variant).train()
                    sample = mini_sample(np.array([[1., 3.], [1., 3.], [2., 4.]]),
                                         horizon=horizon)
                    batch = collate_samples([sample], mini_config_dict(pred_len=horizon))
                    result = model.rollout(batch, horizon)
                    self.assertEqual(result.y.shape, (1, horizon, 1))
                    self.assertEqual(result.z.shape[2], 0 if variant == "total_only" else 1)
                    result.y.square().mean().backward()
                    self.assertGreater(model.load_head.weight.grad.abs().max().item(), 0)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_future_node_axis_is_not_truncated_by_horizon(self):
        sample = mini_sample(np.ones((3, 3)), horizon=1)
        sample["supervision"] = {"target": np.zeros((1, 1), np.float32),
                                 "target_mask": np.ones((1, 1), bool), "future_groups": {}}
        for name, group in sample["inputs"]["groups"].items():
            shape = (1,) + group["values"].shape[1:]
            sample["supervision"]["future_groups"][name] = {
                "values": np.ones(shape, np.float32), "observed": np.ones(shape, bool)}
        batch = collate_samples([sample], mini_config_dict(pred_len=1))
        self.assertEqual(batch.supervision.future_groups["assets"]["observed"].sum().item(), 3)

    def test_masked_future_nan_does_not_change_path(self):
        encoder = FuturePathEncoder(1, 8, 2, 1, dropout=0).eval()
        values = torch.tensor([[[1.], [0.]]])
        mask = torch.tensor([[[True], [False]]])
        calendar = torch.zeros(1, 2, 4)
        with torch.no_grad():
            expected = encoder(values, mask, calendar)
            values[:, 1] = float('nan')
            actual = encoder(values, mask, calendar)
        torch.testing.assert_close(actual, expected)

    def test_observed_negative_one_is_distinguishable_from_missing(self):
        encoder = FuturePathEncoder(1, 8, 2, 1, dropout=0).eval()
        calendar = torch.zeros(1, 2, 4)
        with torch.no_grad():
            measured = encoder(-torch.ones(1, 2, 1), torch.ones(1, 2, 1, dtype=torch.bool), calendar)
            missing = encoder(torch.zeros(1, 2, 1), torch.zeros(1, 2, 1, dtype=torch.bool), calendar)
        self.assertFalse(torch.allclose(measured, missing, atol=1e-6, rtol=1e-5))

    def test_actual_batch_attention_budget_is_enforced(self):
        config = mini_config_dict()
        config["model"] = copy.deepcopy(MINI_PARAMS)
        config["limits"]["max_explicit_attention_bytes"] = 300
        check_attention_budget(config, [2, 0])  # 单样本最大分数 216 字节，合法。
        sample = mini_sample(np.ones((3, 2)))
        with self.assertRaisesRegex(ValueError, "attention"):
            collate_samples([sample, copy.deepcopy(sample)], config)  # 合批 432 字节，必须拒绝。

    def test_readout_supports_declared_spatial_heads(self):
        params = {**MINI_PARAMS, "spatial_dim": 6, "spatial_heads": 2, "relation_heads": 2}
        config = mini_config_dict()
        model = Model(groups_spec=MINI_GROUPS, relation_types=config["data"]["relation_types"],
                      model_params=params, vocab_sizes={}).eval()
        sample = mini_sample(np.ones((3, 2)), categories=np.ones((3, 2, 1), np.int64))
        result = model.rollout(collate_samples([sample], config), 2)
        self.assertEqual(result.y.shape, (1, 2, 1))

    def test_orphans_exclude_padding_and_inactive_entities_and_degrade_quality(self):
        sample = mini_sample(np.ones((3, 3)))
        group = sample["inputs"]["groups"]["assets"]
        group["present"][:, 1] = False
        group["padding"][2] = True
        orphans = orphan_entity_count(sample["inputs"]["groups"], [])
        self.assertEqual(orphans, 1)
        flags = quality_flags(sample["inputs"]["groups"], sample["inputs"]["target_history_observed"],
                              sample["inputs"]["future_known_mask"], orphan_count=orphans)
        self.assertEqual(flags["input_coverage"], 1)
        self.assertTrue(flags["degraded_input"])

    def test_static_mean_uses_observed_attributes_and_empty_count_is_zero(self):
        spec = [{"name": "g", "dynamic": [], "static_numeric": ["capacity"], "static_category": []}]
        encoder = StaticConditionEncoder(spec, [], {"g": {}}, 4, 1, 1, 8)
        group = {"present": torch.ones(1, 1, 2, dtype=torch.bool),
                 "padding": torch.zeros(1, 2, dtype=torch.bool),
                 "static_numeric_history": torch.tensor([[[[2.], [0.]]]]),
                 "static_numeric_mask": torch.tensor([[[[True], [False]]]])}
        batch = SimpleNamespace(inputs=SimpleNamespace(groups={"g": group}))
        with torch.no_grad():
            encoder.per_group["g"][0].weight.zero_()
            encoder.per_group["g"][0].bias.zero_()
            encoder.per_group["g"][0].weight[0, 0] = 1
            encoder.project.weight.fill_(1)
            encoder.project.bias.zero_()
        expected = torch.nn.functional.gelu(torch.tensor([[2.]]))
        torch.testing.assert_close(encoder(batch, {"g": torch.zeros(1, 0)}), expected)
        group["present"].fill_(False)
        with torch.no_grad():
            encoder.per_group["g"][0].weight.zero_()
            encoder.per_group["g"][0].weight[0, -1] = 1
        self.assertEqual(encoder(batch, {"g": torch.zeros(1, 0)}).item(), 0)


if __name__ == "__main__":
    unittest.main()
