import json
import os
import sys
import tempfile
import types
import unittest
import importlib
import importlib.util
from pathlib import Path
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Allow importing generation script even when miditoolkit is unavailable in test env.
if importlib.util.find_spec('miditoolkit') is None:
    sys.modules['miditoolkit'] = types.ModuleType('miditoolkit')


def _load_generation_module_or_skip(testcase):
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        raise unittest.SkipTest('torch is not installed in this environment')

    try:
        mod = importlib.import_module('scripts.generate.generate_drum_from_input')
    except Exception as e:  # pragma: no cover - defensive skip for optional runtime deps
        raise unittest.SkipTest(f'cannot import generation module: {e}')

    return mod


def _torch_or_skip(testcase):
    try:
        import torch
    except ModuleNotFoundError:
        raise unittest.SkipTest('torch is not installed in this environment')
    return torch


class DifficultyResolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load_generation_module_or_skip(cls)

    def setUp(self):
        self.base = {
            's_tech_cls': np.array([1, 6, 2], dtype=np.int64),
            's_indep_cls': np.array([2, 7, 0], dtype=np.int64),
            's_hand_cls': np.array([5, 1, 3], dtype=np.int64),
            's_foot_cls': np.array([2, 3, 4], dtype=np.int64),
            's_move_cls': np.array([7, 2, 1], dtype=np.int64),
        }
        self.bounds = {k: 7 for k in self.gen.DIFFICULTY_KEYS}

    def test_global_shift_mode(self):
        out = self.gen.resolve_target_difficulty_levels(
            self.base,
            difficulty_mode='global_shift',
            bounds=self.bounds,
            difficulty_shift=[1, -2, 3, 0, -10],
        )
        np.testing.assert_array_equal(out['s_tech_cls'], np.array([2, 7, 3], dtype=np.int64))
        np.testing.assert_array_equal(out['s_indep_cls'], np.array([0, 5, 0], dtype=np.int64))
        np.testing.assert_array_equal(out['s_move_cls'], np.array([0, 0, 0], dtype=np.int64))

    def test_upper_cap_mode(self):
        out = self.gen.resolve_target_difficulty_levels(
            self.base,
            difficulty_mode='upper_cap',
            bounds=self.bounds,
            difficulty_cap=[3, 5, 4, 1, 6],
        )
        np.testing.assert_array_equal(out['s_tech_cls'], np.array([1, 3, 2], dtype=np.int64))
        np.testing.assert_array_equal(out['s_indep_cls'], np.array([2, 5, 0], dtype=np.int64))
        np.testing.assert_array_equal(out['s_foot_cls'], np.array([1, 1, 1], dtype=np.int64))

    def test_full_assign_mode(self):
        payload = {
            's_tech_cls': [0, 1, 2],
            's_indep_cls': [2, 2, 2],
            's_hand_cls': [3, 3, 3],
            's_foot_cls': [4, 4, 4],
            's_move_cls': [5, 5, 5],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            p = Path(tmpdir) / 'labels.json'
            with open(p, 'w', encoding='utf-8') as f:
                json.dump(payload, f)

            out = self.gen.resolve_target_difficulty_levels(
                self.base,
                difficulty_mode='full_assign',
                bounds=self.bounds,
                labels_file=str(p),
            )

        np.testing.assert_array_equal(out['s_tech_cls'], np.array([0, 1, 2], dtype=np.int64))
        np.testing.assert_array_equal(out['s_move_cls'], np.array([5, 5, 5], dtype=np.int64))


class WindowAndMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load_generation_module_or_skip(cls)

    def test_window_plan_with_stride_and_short_last_window(self):
        plan = self.gen.build_window_plan(n_bars=8, max_bars=4, window_stride_bars=2)
        self.assertEqual(plan, [(0, 4), (2, 6), (4, 8)])

        plan_short_last = self.gen.build_window_plan(n_bars=5, max_bars=4, window_stride_bars=3)
        self.assertEqual(plan_short_last, [(0, 4), (3, 5)])

    def test_merge_window_token_chunks_preserves_total_bar_count(self):
        total_bars = 5
        w1 = (0, 3, [['<BAR>', 'A'], ['<BAR>', 'B'], ['<BAR>', 'C']])
        w2 = (2, 5, [['<BAR>', 'X'], ['<BAR>', 'Y'], ['<BAR>', 'Z']])

        merged = self.gen.merge_window_token_chunks(total_bars, [w1, w2], merge_policy='keep_first')
        self.assertEqual(self.gen.count_token_bars(merged), total_bars)

        merged_last = self.gen.merge_window_token_chunks(total_bars, [w1, w2], merge_policy='keep_last')
        self.assertEqual(self.gen.count_token_bars(merged_last), total_bars)


class DecoderTerminationGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load_generation_module_or_skip(cls)
        cls.torch = _torch_or_skip(cls)

    class _DummyTokenizer:
        def __init__(self):
            self.token2idx = {'<PAD>': 0, '<BAR>': 1, '<EOS>': 2, 'A': 3}
            self.idx2token = {v: k for k, v in self.token2idx.items()}

    class _DummyModel:
        def __init__(self, vocab_size, torch_mod):
            self.vocab_size = vocab_size
            self._torch = torch_mod

        def generate(self, dec_input, dec_seg_emb, **kwargs):
            return self._torch.zeros((1, self.vocab_size), dtype=self._torch.float32)

    def test_early_eos_does_not_collapse_to_single_bar(self):
        tokenizer = self._DummyTokenizer()
        model = self._DummyModel(vocab_size=len(tokenizer.token2idx), torch_mod=self.torch)
        latents = self.torch.zeros((3, 4), dtype=self.torch.float32)

        # EOS, BAR, EOS, BAR, BAR -> ensures early EOS appears before completion.
        sequence = [
            tokenizer.token2idx['<EOS>'],
            tokenizer.token2idx['<BAR>'],
            tokenizer.token2idx['<EOS>'],
            tokenizer.token2idx['<BAR>'],
            tokenizer.token2idx['<BAR>'],
        ]

        with patch.object(self.gen, 'nucleus', side_effect=sequence):
            generated, _, _ = self.gen.generate_with_difficulty_control(
                model,
                latents,
                s_tech_cls=np.array([0, 0, 0], dtype=np.int64),
                s_indep_cls=np.array([0, 0, 0], dtype=np.int64),
                s_hand_cls=np.array([0, 0, 0], dtype=np.int64),
                s_foot_cls=np.array([0, 0, 0], dtype=np.int64),
                s_move_cls=np.array([0, 0, 0], dtype=np.int64),
                tokenizer=tokenizer,
                device='cpu',
                max_events=256,
                temperature=1.0,
                nucleus_p=0.9,
            )

        generated_tokens = [tokenizer.idx2token[idx] for idx in generated]
        # Initial primer BAR + 3 generated BARs = at least 4 BARs in output sequence.
        self.assertGreaterEqual(generated_tokens.count('<BAR>'), 4)


class VocabularyCompatibilityGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load_generation_module_or_skip(cls)
        cls.torch = _torch_or_skip(cls)

    class _StandardTokenizer:
        def __init__(self):
            self.token2idx = {'<PAD>': 0, '<BAR>': 1, '<EOS>': 2, 'X': 3}
            self.idx2token = {0: '<PAD>', 1: '<BAR>', 2: '<EOS>', 3: 'X'}
            self.vocab_size = len(self.token2idx)

    class _CPTokenizer:
        def __init__(self):
            self.vocab_size = 8
            self.type_token2idx = {'<PAD>': 0, '<BAR>': 1, '<EOS>': 2, '<NOTE>': 3}
            self.idx2type_token = {0: '<PAD>', 1: '<BAR>', 2: '<EOS>', 3: '<NOTE>'}
            self.struct_token2idx = self.type_token2idx
            self.idx2struct_token = self.idx2type_token
            self.event_type2idx = {'<PAD_EVENT>': 0, 'STRUCT': 1, 'NOTE': 2}
            self.idx2event_type = {0: '<PAD_EVENT>', 1: 'STRUCT', 2: 'NOTE'}
            self.limb_token2idx = {'<PAD>': 0, '<NA>': 1, 'HIT': 2}
            self.idx2limb_token = {0: '<PAD>', 1: '<NA>', 2: 'HIT'}
            self.pos_pad_value = 7

    def test_standard_mode_vocab_guard_pass_and_fail(self):
        tok = self._StandardTokenizer()
        ckpt_ok = {
            'token_emb.emb_lookup.weight': self.torch.zeros((4, 16)),
            'dec_out_proj.weight': self.torch.zeros((4, 16)),
            'dec_out_proj.bias': self.torch.zeros((4,)),
        }
        self.gen.validate_vocabulary_compatibility('standard', tok, ckpt_ok)
        self.gen.validate_vocabulary_compatibility('remi', tok, ckpt_ok)

        ckpt_bad = dict(ckpt_ok)
        ckpt_bad['dec_out_proj.bias'] = self.torch.zeros((3,))
        with self.assertRaises(ValueError):
            self.gen.validate_vocabulary_compatibility('standard', tok, ckpt_bad)

    def test_cp_mode_vocab_guard_pass_and_fail(self):
        tok = self._CPTokenizer()
        ckpt_ok = {
            'token_emb.emb_lookup.weight': self.torch.zeros((8, 16)),
            'dec_out_proj.weight': self.torch.zeros((8, 16)),
            'dec_out_proj.bias': self.torch.zeros((8,)),
            'cp_composite_emb.type_emb.weight': self.torch.zeros((4, 8)),
            'cp_composite_emb.pos_emb.weight': self.torch.zeros((8, 8)),
            'cp_composite_emb.hand1_emb.weight': self.torch.zeros((3, 8)),
            'cp_event_type_head.weight': self.torch.zeros((3, 16)),
            'cp_struct_head.weight': self.torch.zeros((4, 16)),
            'cp_pos_head.weight': self.torch.zeros((8, 16)),
            'cp_hand1_head.weight': self.torch.zeros((3, 16)),
        }
        self.gen.validate_vocabulary_compatibility('cp_limb_v1', tok, ckpt_ok)

        tok_bad = self._CPTokenizer()
        del tok_bad.event_type2idx['<PAD_EVENT>']
        with self.assertRaises(ValueError):
            self.gen.validate_vocabulary_compatibility('cp_limb_v1', tok_bad, ckpt_ok)


class OutputWriterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.gen = _load_generation_module_or_skip(cls)

    def test_output_parent_dir_auto_creation_and_debug_suffix(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_mid = Path(tmpdir) / 'a' / 'b' / 'song.mid'
            self.gen.ensure_output_parent_dir(output_mid)
            self.assertTrue(output_mid.parent.exists())
            self.assertEqual(self.gen._resolve_debug_txt_path(output_mid).name, 'song.txt')

    def test_save_standard_or_remi_outputs_writes_same_prefix_debug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_mid = Path(tmpdir) / 'gen.mid'
            tokens = ['<BAR>', 'A', '<EOS>']
            with patch.object(self.gen, 'tokens_to_midi'):
                debug_path = self.gen.save_standard_or_remi_outputs(output_mid, tokens, bpm=120)
            self.assertEqual(debug_path, output_mid.with_suffix('.txt'))
            self.assertTrue(debug_path.exists())

    def test_save_cp_outputs_writes_same_prefix_debug(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_mid = Path(tmpdir) / 'gen_cp.mid'
            cp_data = {
                'cp_type': [1, 3, 2],
                'cp_pos': [0, 1, 0],
                'cp_hand1': [1, 2, 1],
                'cp_hand2': [1, 1, 1],
                'cp_right_foot': [1, 1, 1],
                'cp_left_foot': [1, 1, 1],
            }
            idx2type = {0: '<PAD>', 1: '<BAR>', 2: '<EOS>', 3: '<NOTE>'}
            idx2limb = {0: '<PAD>', 1: '<NA>', 2: 'HIT'}
            with patch.object(self.gen, 'cp_data_to_midi'):
                debug_path = self.gen.save_cp_outputs(output_mid, cp_data, idx2type, idx2limb, bpm=120)
            self.assertEqual(debug_path, output_mid.with_suffix('.txt'))
            self.assertTrue(debug_path.exists())


if __name__ == '__main__':
    unittest.main()
