"""训练支持契约：冻结、teacher 延续隔离、独立手算 reduction 与质量图。"""
import copy
import unittest
from types import SimpleNamespace

import torch

from data_provider.spatial_temporal.batching import collate_samples
from exp.gslaformer_support import (LatentLabels, ValidationMSE, build_latent_labels, group_quality, diagnostic_rollout,
                                   latent_dimension_weights, masked_load_loss,
                                   weighted_latent_loss)
from test_gslaformer_model import real_batch


class TrainingSupportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2023)

    def fixture(self):
        model, samples, config = real_batch()
        return model, collate_samples(samples, config)

    def test_stage_b_frozen_chain_survives_train_and_optimizer_step(self):
        model, batch = self.fixture()
        a_params = model.set_training_stage("A")
        a_ids = {id(p) for p in a_params}
        self.assertTrue(all(id(p) not in a_ids for p in model.latent_heads.parameters()))
        model.set_training_stage("B")
        model.train()
        frozen = [p for module in model.representation_modules() for p in module.parameters()]
        before = [p.detach().clone() for p in frozen]
        self.assertTrue(all(not m.training for module in model.representation_modules() for m in module.modules()))
        batch.inputs.groups["assets"]["values"].requires_grad_(True)
        encoded = model.encode_history(batch)
        self.assertIsNone(encoded.z.grad_fn)
        self.assertIsNone(encoded.c.grad_fn)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertFalse({id(p) for p in frozen} & {id(p) for p in trainable})
        optimizer = torch.optim.AdamW(trainable, lr=0.001)
        labels = build_latent_labels(model, batch.inputs, batch.supervision, encoded)
        output = model.rollout(batch, 2)
        load, _ = masked_load_loss(output.y, batch.supervision.target, batch.supervision.target_mask)
        latent, _ = weighted_latent_loss(output.z, labels, torch.ones(1, model.d_z))
        (load + 0.1 * latent).backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1, error_if_nonfinite=True)
        optimizer.step()
        for old, new in zip(before, frozen):
            torch.testing.assert_close(old, new, atol=0, rtol=0)
            self.assertIsNone(new.grad)
        self.assertIsNone(batch.inputs.groups["assets"]["values"].grad)
        for module in (model.temporal, model.future_path, model.latent_heads):
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            self.assertTrue(gradients)
            self.assertTrue(any(g.abs().max() > 0 for g in gradients if g.numel()))

    def test_teacher_matches_single_continuous_encoding_without_mutation(self):
        model, batch = self.fixture()
        model.set_training_stage("B")
        model.train()
        history = model.encode_history(batch)
        memory = {k: v.clone() for k, v in history.entity_memory.items()}
        before_inputs = copy.deepcopy(batch.inputs)
        labels = build_latent_labels(model, batch.inputs, batch.supervision, history)
        self.assertIsNone(labels.z.grad_fn)
        # 独立 oracle：拼接整个历史与未来，在同一起点从零连续编码一次。
        extended = copy.deepcopy(batch)
        horizon = batch.inputs.future_time.shape[1]
        length = batch.inputs.history_time.shape[1]
        for name, group in extended.inputs.groups.items():
            future = batch.supervision.future_groups[name]
            for key in ("values", "observed"):
                group[key] = torch.cat([group[key], future[key]], dim=1)
            for key in ("present", "static_numeric_history", "static_numeric_mask", "static_category_history"):
                repeats = [1, horizon] + [1] * (group[key].ndim - 2)
                group[key] = torch.cat([group[key], group[key][:, -1:].repeat(*repeats)], dim=1)
        extended.inputs.target_history = torch.cat([batch.inputs.target_history, torch.zeros_like(batch.supervision.target)], 1)
        extended.inputs.relations_history = [old + [origin] * horizon for old, origin in
                                             zip(batch.inputs.relations_history, batch.inputs.relations_origin)]
        expected = model.encode_history(extended).z[:, length:]
        torch.testing.assert_close(labels.z, expected, atol=1e-6, rtol=1e-5)
        for name in memory:
            torch.testing.assert_close(history.entity_memory[name], memory[name], atol=0, rtol=0)
            torch.testing.assert_close(batch.inputs.groups[name]["values"], before_inputs.groups[name]["values"])
        # 每一步错误地重置 memory 不能通过上述 oracle。
        reset = copy.deepcopy(extended)
        for group in reset.inputs.groups.values():
            for key in ("values", "observed", "present", "static_numeric_history", "static_numeric_mask", "static_category_history"):
                group[key] = group[key][:, -horizon:]
        reset.inputs.target_history = reset.inputs.target_history[:, -horizon:]
        reset.inputs.relations_history = [edges[-horizon:] for edges in reset.inputs.relations_history]
        self.assertGreater((model.encode_history(reset).z - labels.z).abs().max().item(), 1e-6)

    def test_future_target_not_used_by_teacher_and_future_groups_not_used_by_rollout(self):
        model, batch = self.fixture()
        with self.assertRaisesRegex(ValueError, "冻结"):
            build_latent_labels(model, batch.inputs, batch.supervision)
        model.set_training_stage("B")
        model.eval()
        labels = build_latent_labels(model, batch.inputs, batch.supervision)
        with torch.no_grad():
            prediction = model.rollout(batch, 2).y
        batch.supervision.target.fill_(float('nan'))
        torch.testing.assert_close(labels.z, build_latent_labels(model, batch.inputs, batch.supervision).z)
        batch.supervision.future_groups["assets"]["values"].add_(100)
        changed = build_latent_labels(model, batch.inputs, batch.supervision)
        self.assertGreater((changed.z - labels.z).abs().max().item(), 1e-6)
        with torch.no_grad():
            torch.testing.assert_close(prediction, model.rollout(batch, 2).y)

    def test_load_loss_horizon_macro_and_masked_nan(self):
        prediction = torch.tensor([[[1.], [3.]], [[3.], [float('nan')]]], requires_grad=True)
        mask = torch.tensor([[[True], [True]], [[True], [False]]])
        loss, counts = masked_load_loss(prediction, torch.zeros_like(prediction), mask)
        # horizon0=(1+9)/2=5, horizon1=9；macro=7，不是 micro=19/3。
        self.assertEqual(loss.item(), 7)
        self.assertEqual(counts.tolist(), [2, 1])
        loss.backward()
        self.assertEqual(prediction.grad[1, 1].item(), 0)
        empty, counts = masked_load_loss(prediction, torch.zeros_like(prediction), torch.zeros_like(mask))
        self.assertIsNone(empty)
        self.assertEqual(counts.sum().item(), 0)

    def test_latent_loss_quality_not_renormalized(self):
        prediction = torch.ones(2, 2, 2, 1, requires_grad=True)
        q = torch.tensor([[[0.5, 0.], [0.25, 0.]], [[0., 0.], [0.25, 0.]]])
        labels = LatentLabels(torch.zeros_like(prediction), q)
        loss, counts = weighted_latent_loss(prediction, labels, torch.ones(2, 1))
        # 仅 group0 有效；h0=0.5,h1=(0.25+0.25)/2=0.25；组均值=0.375。
        self.assertEqual(loss.item(), 0.375)
        self.assertEqual(counts.tolist(), [[1, 0], [2, 0]])
        loss.backward()
        self.assertEqual(prediction.grad[:, :, 1].abs().max().item(), 0)
        labels.quality.zero_()
        zero, _ = weighted_latent_loss(prediction, labels, torch.ones(2, 1))
        self.assertEqual(zero.item(), 0)
        torch.testing.assert_close(latent_dimension_weights(torch.tensor([0., 0.01, 0.99])),
                                   torch.tensor([100., 50., 1.]))
        with self.assertRaises(ValueError):
            latent_dimension_weights(torch.tensor([-1.]))
        grid = torch.tensor([0., 1e-12, 1e-10, 1e-8, 1e-6, 0.01, 1., 100.])
        bounded = latent_dimension_weights(grid)
        self.assertTrue(torch.isfinite(bounded).all())
        self.assertTrue(((bounded > 0) & (bounded <= 100)).all())
        self.assertTrue((bounded[1:] <= bounded[:-1]).all())
        self.assertLess(abs(bounded[1].item() - bounded[0].item()), 1e-4)

    def test_quality_two_hops_direction_and_relation_ablation(self):
        specs = [{"name": n, "dynamic": ["x"]} for n in ("a", "b", "c")]
        groups = {}
        for name, observed in zip(("a", "b", "c"), ([True, True], [True, False], [False, False])):
            groups[name] = {"observed": torch.tensor(observed).reshape(1, 2, 1, 1),
                            "present": torch.ones(1, 2, 1, dtype=torch.bool),
                            "padding": torch.zeros(1, 1, dtype=torch.bool),
                            "applicable": torch.ones(1, 1, 1, dtype=torch.bool)}
        inputs = SimpleNamespace(groups=groups, target_history=torch.zeros(1, 2, 1),
                                 relations_origin=[[(0, 1, 0), (1, 2, 0)]])
        q = group_quality(inputs, specs, [{"reverse_message": False}])
        # a reaches b,c: r_neighbor=(0.5+0)/2；b reaches c；c 自身 r=0。
        torch.testing.assert_close(q, torch.tensor([[0.625, 0.25, 0.]]))
        torch.testing.assert_close(group_quality(inputs, specs, [{"reverse_message": False}], use_relations=False),
                                   torch.tensor([[1., 0.5, 0.]]))
        # 保持覆盖比例扩大实体轴：q 不能随 all-observed 概率衰减。
        expanded = copy.deepcopy(inputs)
        for group in expanded.groups.values():
            group["observed"] = group["observed"].repeat(1, 1, 100, 1)
            group["present"] = group["present"].repeat(1, 1, 100)
            group["padding"] = group["padding"].repeat(1, 100)
            group["applicable"] = group["applicable"].repeat(1, 100, 1)
        expanded.relations_origin = [[(i, 100 + i, 0) for i in range(100)]
                                     + [(100 + i, 200 + i, 0) for i in range(100)]]
        torch.testing.assert_close(group_quality(expanded, specs, [{"reverse_message": False}]), q)

    def test_validation_site_macro_is_batch_partition_invariant(self):
        prediction = torch.tensor([[[1.], [1.]], [[1.], [1.]], [[3.], [3.]]])
        target, mask = torch.zeros_like(prediction), torch.ones_like(prediction, dtype=torch.bool)
        sites = ["a", "a", "b"]
        together = ValidationMSE(2, expected_sites=sites)
        split = ValidationMSE(2, expected_sites=sites)
        together.update(prediction, target, mask, sites)
        for i in range(3):
            split.update(prediction[i:i+1], target[i:i+1], mask[i:i+1], sites[i:i+1])
        # site a=1，site b=9；站点等权为 5，不按样本数加权为 11/3。
        self.assertEqual(together.compute(), 5)
        self.assertEqual(split.compute(), 5)
        missing = ValidationMSE(2, expected_sites=["a", "b"])
        missing.update(prediction[:1], target[:1], mask[:1], ["a"])
        with self.assertRaisesRegex(ValueError, "站点"):
            missing.compute()
        mask[-1, -1] = False
        incomplete = ValidationMSE(2, expected_sites=sites)
        incomplete.update(prediction, target, mask, sites)
        with self.assertRaisesRegex(ValueError, "horizon"):
            incomplete.compute()

    def test_diagnostics_keep_step_one_and_isolate_future_teacher(self):
        model, batch = self.fixture()
        model.set_training_stage("B")
        model.eval()
        with torch.no_grad():
            production = model.rollout(batch, 2).y
        outcomes = {kind: diagnostic_rollout(model, batch, kind) for kind in ("fixed_z", "fixed_y", "teacher_z")}
        for outcome in outcomes.values():
            torch.testing.assert_close(outcome[:, 0], production[:, 0], atol=1e-6, rtol=1e-5)
        batch.supervision.future_groups["assets"]["values"].add_(100)
        for kind in ("fixed_z", "fixed_y"):
            torch.testing.assert_close(outcomes[kind], diagnostic_rollout(model, batch, kind))
        teacher = diagnostic_rollout(model, batch, "teacher_z")
        self.assertGreater((teacher[:, 1] - outcomes["teacher_z"][:, 1]).abs().max().item(), 1e-7)


if __name__ == "__main__":
    unittest.main()
