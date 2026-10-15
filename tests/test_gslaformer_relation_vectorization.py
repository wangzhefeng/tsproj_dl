"""关系 mask/bias 向量化的数值、参数梯度与输入梯度等价性。"""
import copy
import unittest

import torch

from layers.gslaformer.relations import RelationAttention


def reference_mask(layer, nodes, edges, effective):
    # 逐边显式 oracle：query=接收方，key=发送方，不调用被测 mask builder。
    f = len(edges)
    allowed = torch.eye(nodes, dtype=torch.bool).expand(f, -1, -1).clone()
    bias = torch.zeros(f, layer.heads, nodes, nodes)
    for frame, items in enumerate(edges):
        for src, dst, kind in items:
            allowed[frame, dst, src] = True
            bias[frame, :, dst, src] = bias[frame, :, dst, src] + layer.direction_bias[kind, 0]
            if layer.relation_types[kind]["reverse_message"]:
                allowed[frame, src, dst] = True
                bias[frame, :, src, dst] = bias[frame, :, src, dst] + layer.direction_bias[kind, 1]
    for frame in range(f):
        for receiver in range(nodes):
            for sender in range(nodes):
                if not effective[frame, receiver] or not effective[frame, sender]:
                    allowed[frame, receiver, sender] = receiver == sender
    additive = torch.where(allowed, 0., -torch.inf).unsqueeze(1)
    return additive, bias * allowed.unsqueeze(1)


class RelationVectorizationTests(unittest.TestCase):
    def test_values_and_gradients_match_explicit_edge_oracle(self):
        torch.manual_seed(2023)
        layer = RelationAttention([{"reverse_message": True}, {"reverse_message": False}], 8, 2, dropout=0)
        reference = copy.deepcopy(layer)
        effective = torch.tensor([[True, True, False, True], [True, True, True, True]])
        edges = [[(0, 1, 0), (0, 1, 1), (2, 1, 0)], [(0, 1, 1), (1, 3, 0)]]
        actual_mask, actual_bias = layer.build_masks(4, edges, effective)
        expected_mask, expected_bias = reference_mask(reference, 4, edges, effective)
        torch.testing.assert_close(actual_mask, expected_mask)
        torch.testing.assert_close(actual_bias, expected_bias)
        x = torch.randn(2, 4, 8, requires_grad=True)
        other = x.detach().clone().requires_grad_(True)
        actual = layer(x, edges, effective)
        expected = other
        merged = (expected_mask + expected_bias).reshape(4, 4, 4)
        for block in reference.layers:
            expected = block(expected, src_mask=merged)
        expected = expected * effective.unsqueeze(-1)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
        (expected * probe).sum().backward()
        torch.testing.assert_close(x.grad, other.grad, atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(layer.direction_bias.grad, reference.direction_bias.grad, atol=1e-6, rtol=1e-5)
        self.assertGreater(layer.direction_bias.grad.abs().max().item(), 0)

    def test_no_edges_has_only_self_and_zero_bias(self):
        layer = RelationAttention([{"reverse_message": False}], 8, 2)
        mask, bias = layer.build_masks(3, [[], []], torch.ones(2, 3, dtype=torch.bool))
        torch.testing.assert_close(torch.isfinite(mask[:, 0]), torch.eye(3, dtype=torch.bool).expand(2, -1, -1))
        self.assertEqual(bias.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
