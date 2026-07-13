"""Attention Bias Loss（src/models/limb_bias.py 追加分）の単体テスト。

docs/DESIGN_attn_bias_loss.md §8 のテスト計画に対応する:
  - build_limb_pair_masks: 相互排他・メタ除外・下三角
  - compute_attn_bias_loss: 3 モードの hinge 損失
  - bias_loss_weight_sched: 3 種スケジューラの境界値
  - AttentionWeightCapture: 勾配が QKV 重みへ到達すること
"""
import sys
import unittest
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.limb_bias import (
    build_limb_pair_masks,
    build_limb_bias_mask,
    compute_attn_bias_loss,
    bias_loss_weight_sched,
    AttentionWeightCapture,
    LIMB_META, LIMB_HAND, LIMB_RFOOT, LIMB_HAND_LFOOT,
)
from src.models.position_bias import (
    build_position_bias_mask,
    build_position_pair_masks,
    compute_attn_pos_bias_loss,
)


class BuildLimbPairMasksTests(unittest.TestCase):
    def setUp(self):
        # [META, HAND, HAND, RFOOT, HAND_LFOOT]
        self.limb_ids = torch.tensor(
            [[LIMB_META, LIMB_HAND, LIMB_HAND, LIMB_RFOOT, LIMB_HAND_LFOOT]]
        )

    def test_meta_excluded_and_mutually_exclusive(self):
        M_limb, M_other = build_limb_pair_masks(self.limb_ids, causal=False)
        # メタ(idx0)関与ペアは両集合で 0
        self.assertEqual(M_limb[0, 0, :].sum().item(), 0)
        self.assertEqual(M_limb[0, :, 0].sum().item(), 0)
        self.assertEqual(M_other[0, 0, :].sum().item(), 0)
        self.assertEqual(M_other[0, :, 0].sum().item(), 0)
        # 相互排他
        self.assertEqual((M_limb * M_other).sum().item(), 0)

    def test_same_and_cross_limb_classification(self):
        M_limb, M_other = build_limb_pair_masks(self.limb_ids, causal=False)
        self.assertEqual(M_limb[0, 1, 2].item(), 1)   # HAND-HAND
        self.assertEqual(M_other[0, 1, 3].item(), 1)  # HAND-RFOOT
        self.assertEqual(M_limb[0, 4, 1].item(), 1)   # HAND_LFOOT-HAND（同一扱い）
        self.assertEqual(M_other[0, 4, 3].item(), 1)  # HAND_LFOOT-RFOOT
        self.assertEqual(M_limb[0, 1, 1].item(), 1)   # 自己ペア（非メタ）

    def test_causal_masks_upper_triangle(self):
        M_limb, _ = build_limb_pair_masks(self.limb_ids, causal=True)
        self.assertEqual(M_limb[0, 1, 2].item(), 0)   # 上三角は除外
        self.assertEqual(M_limb[0, 2, 1].item(), 1)   # 下三角は保持

    def test_consistency_with_build_limb_bias_mask(self):
        M_limb, _ = build_limb_pair_masks(self.limb_ids, causal=False)
        M_ref = build_limb_bias_mask(self.limb_ids)
        self.assertTrue(torch.equal(M_limb, M_ref))


class ComputeAttnBiasLossTests(unittest.TestCase):
    def setUp(self):
        self.limb_ids = torch.tensor(
            [[LIMB_META, LIMB_HAND, LIMB_HAND, LIMB_RFOOT, LIMB_HAND_LFOOT]]
        )
        self.M_limb, self.M_other = build_limb_pair_masks(self.limb_ids, causal=False)
        self.modes = ['none', 'none', 'limb_over_other', 'none', 'none', 'none']

    def test_uniform_attention_hinge_equals_margin(self):
        N = 5
        A = torch.full((1, N, N), 1.0 / N)
        loss, metrics = compute_attn_bias_loss(
            {2: A}, self.M_limb, self.M_other, self.modes, margin=0.05)
        self.assertAlmostEqual(metrics['attn_limb/2'], metrics['attn_other/2'], places=5)
        self.assertAlmostEqual(loss.item(), 0.05, places=5)

    def test_limb_concentrated_zero_loss(self):
        A = torch.zeros((1, 5, 5))
        A[0, 1, 2] = 1.0
        A[0, 2, 1] = 1.0
        A[0, 4, 1] = 1.0
        loss, metrics = compute_attn_bias_loss(
            {2: A}, self.M_limb, self.M_other, self.modes, margin=0.05)
        self.assertEqual(metrics['attn_other/2'], 0.0)
        self.assertEqual(loss.item(), 0.0)

    def test_other_over_limb_penalized_when_limb_concentrated(self):
        A = torch.zeros((1, 5, 5))
        A[0, 1, 2] = 1.0
        A[0, 2, 1] = 1.0
        modes_o = ['none', 'none', 'other_over_limb', 'none', 'none', 'none']
        loss, _ = compute_attn_bias_loss(
            {2: A}, self.M_limb, self.M_other, modes_o, margin=0.05)
        self.assertGreater(loss.item(), 0.05)

    def test_all_none_returns_zero(self):
        A = torch.full((1, 5, 5), 0.2)
        modes = ['none'] * 6
        loss, metrics = compute_attn_bias_loss(
            {2: A}, self.M_limb, self.M_other, modes, margin=0.05)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics, {})


class BiasLossWeightSchedTests(unittest.TestCase):
    def test_constant(self):
        self.assertEqual(bias_loss_weight_sched(500, 'constant', 1.0, 1000, 4000, 10000), 0.0)
        self.assertEqual(bias_loss_weight_sched(1000, 'constant', 1.0, 1000, 4000, 10000), 1.0)

    def test_linear_warmup(self):
        self.assertEqual(bias_loss_weight_sched(999, 'linear_warmup', 1.0, 1000, 4000, 10000), 0.0)
        self.assertAlmostEqual(
            bias_loss_weight_sched(3000, 'linear_warmup', 1.0, 1000, 4000, 10000), 0.5, places=9)
        self.assertEqual(bias_loss_weight_sched(5000, 'linear_warmup', 1.0, 1000, 4000, 10000), 1.0)
        self.assertEqual(bias_loss_weight_sched(100000, 'linear_warmup', 1.0, 1000, 4000, 10000), 1.0)

    def test_cyclical_gated_by_no_loss_steps(self):
        self.assertEqual(bias_loss_weight_sched(500, 'cyclical', 1.0, 1000, 4000, 10000), 0.0)

    def test_unknown_sched_raises(self):
        with self.assertRaises(ValueError):
            bias_loss_weight_sched(2000, 'bogus', 1.0, 1000, 4000, 10000)


class AttentionWeightCaptureTests(unittest.TestCase):
    def test_gradient_reaches_qkv_weights(self):
        torch.manual_seed(0)
        layer = torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=False)
        layer.train()
        cap = AttentionWeightCapture({0: layer}).attach()
        cap.enabled = True

        x = torch.randn(5, 1, 16)
        mask = torch.triu(torch.full((5, 5), float('-inf')), diagonal=1)
        layer(x, src_mask=mask)
        maps = cap.pop_maps()

        self.assertIn(0, maps)
        self.assertTrue(maps[0].requires_grad)

        li = torch.tensor([[LIMB_HAND, LIMB_HAND, LIMB_RFOOT, LIMB_HAND, LIMB_RFOOT]])
        M_limb, M_other = build_limb_pair_masks(li, causal=True)
        # margin=1.0 で hinge>0 を保証（attn 差は必ず <1）
        loss, _ = compute_attn_bias_loss({0: maps[0]}, M_limb, M_other, ['limb_over_other'], margin=1.0)
        self.assertGreater(loss.item(), 0.0)
        loss.backward()

        grad = layer.self_attn.in_proj_weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0.0)

        cap.detach()
        self.assertEqual(cap._originals, {})

    def test_disabled_capture_records_nothing(self):
        layer = torch.nn.TransformerEncoderLayer(16, 2, 32, batch_first=False)
        layer.train()
        cap = AttentionWeightCapture({0: layer}).attach()
        cap.enabled = False
        x = torch.randn(5, 1, 16)
        layer(x)
        self.assertEqual(cap.pop_maps(), {})
        cap.detach()


class BuildPositionPairMasksTests(unittest.TestCase):
    def setUp(self):
        # idx0=META(0), idx1&2=同一Group(5), idx3&4=同一Group(8)
        self.pos_ids = torch.tensor([[0, 5, 5, 8, 8]])

    def test_meta_excluded_and_mutually_exclusive(self):
        M_pos, M_other = build_position_pair_masks(self.pos_ids, causal=False)
        # META(idx0)関与ペアは両集合で 0
        self.assertEqual(M_pos[0, 0, :].sum().item(), 0)
        self.assertEqual(M_pos[0, :, 0].sum().item(), 0)
        self.assertEqual(M_other[0, 0, :].sum().item(), 0)
        self.assertEqual(M_other[0, :, 0].sum().item(), 0)
        # 相互排他
        self.assertEqual((M_pos * M_other).sum().item(), 0)

    def test_same_and_other_position_classification(self):
        M_pos, M_other = build_position_pair_masks(self.pos_ids, causal=False)
        self.assertEqual(M_pos[0, 1, 2].item(), 1)    # 同一Group(5)
        self.assertEqual(M_pos[0, 3, 4].item(), 1)    # 同一Group(8)
        self.assertEqual(M_pos[0, 1, 1].item(), 1)    # 自己ペア（非メタ）
        self.assertEqual(M_other[0, 1, 3].item(), 1)  # Group5-Group8
        self.assertEqual(M_other[0, 2, 4].item(), 1)  # Group5-Group8

    def test_causal_masks_upper_triangle(self):
        M_pos, _ = build_position_pair_masks(self.pos_ids, causal=True)
        self.assertEqual(M_pos[0, 1, 2].item(), 0)   # 上三角は除外
        self.assertEqual(M_pos[0, 2, 1].item(), 1)   # 下三角は保持

    def test_consistency_with_build_position_bias_mask(self):
        M_pos, _ = build_position_pair_masks(self.pos_ids, causal=False)
        M_ref = build_position_bias_mask(self.pos_ids)
        self.assertTrue(torch.equal(M_pos, M_ref))


class ComputeAttnPosBiasLossTests(unittest.TestCase):
    def setUp(self):
        self.pos_ids = torch.tensor([[0, 5, 5, 8, 8]])
        self.M_pos, self.M_other = build_position_pair_masks(self.pos_ids, causal=False)
        self.modes = ['none', 'none', 'pos_over_other', 'none', 'none', 'none']

    def test_uniform_attention_hinge_equals_margin(self):
        N = 5
        A = torch.full((1, N, N), 1.0 / N)
        loss, metrics = compute_attn_pos_bias_loss(
            {2: A}, self.M_pos, self.M_other, self.modes, margin=0.05)
        self.assertAlmostEqual(metrics['attn_pos/2'], metrics['attn_otherpos/2'], places=5)
        self.assertAlmostEqual(loss.item(), 0.05, places=5)

    def test_pos_concentrated_zero_loss(self):
        A = torch.zeros((1, 5, 5))
        A[0, 1, 2] = 1.0   # 同一Group
        A[0, 3, 4] = 1.0   # 同一Group
        loss, metrics = compute_attn_pos_bias_loss(
            {2: A}, self.M_pos, self.M_other, self.modes, margin=0.05)
        self.assertEqual(metrics['attn_otherpos/2'], 0.0)
        self.assertEqual(loss.item(), 0.0)

    def test_other_over_pos_penalized_when_pos_concentrated(self):
        A = torch.zeros((1, 5, 5))
        A[0, 1, 2] = 1.0
        A[0, 3, 4] = 1.0
        modes_o = ['none', 'none', 'other_over_pos', 'none', 'none', 'none']
        loss, _ = compute_attn_pos_bias_loss(
            {2: A}, self.M_pos, self.M_other, modes_o, margin=0.05)
        self.assertGreater(loss.item(), 0.05)

    def test_limb_and_none_modes_skipped(self):
        # limb 系モード・none の層は pos 損失に寄与せず metrics も出さない
        A = torch.full((1, 5, 5), 0.2)
        modes = ['limb_over_other', 'other_over_limb', 'none', 'none', 'none', 'none']
        loss, metrics = compute_attn_pos_bias_loss(
            {0: A, 1: A, 2: A}, self.M_pos, self.M_other, modes, margin=0.05)
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics, {})

    def test_only_pos_layers_processed_in_mixed_modes(self):
        # limb 層(idx2)と pos 層(idx3)が混在。pos compute は idx3 のみ処理
        A = torch.full((1, 5, 5), 0.2)
        modes = ['none', 'none', 'limb_over_other', 'pos_over_other', 'none', 'none']
        loss, metrics = compute_attn_pos_bias_loss(
            {2: A, 3: A}, self.M_pos, self.M_other, modes, margin=0.05)
        self.assertIn('attn_pos/3', metrics)
        self.assertNotIn('attn_pos/2', metrics)


if __name__ == '__main__':
    unittest.main()
