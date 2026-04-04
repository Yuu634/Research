import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.drum_dataloader import DrumTransformerDataset
    _DATALOADER_IMPORT_ERROR = None
except ModuleNotFoundError as exc:
    DrumTransformerDataset = None
    _DATALOADER_IMPORT_ERROR = exc


class DrumDifficultyAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if DrumTransformerDataset is None:
            raise unittest.SkipTest(f'drum_dataloader import unavailable: {_DATALOADER_IMPORT_ERROR}')

    def _write_vocab(self, path: str) -> None:
        token2idx = {
            '<BAR>': 0,
            '<EOS>': 1,
            '<BEAT_1>': 2,
            '<POS_0>': 3,
            'KICK_HIT_Normal': 4,
        }
        idx2token = {v: k for k, v in token2idx.items()}
        with open(path, 'wb') as f:
            pickle.dump((token2idx, idx2token), f)

    def _difficulty(self, value: int) -> dict:
        return {
            's_tech': value,
            's_indep': value,
            's_hand': value,
            's_foot': value,
            's_move': value,
        }

    def _build_dataset(self, data_dir: str, vocab_path: str, use_difficulty: bool = True):
        return DrumTransformerDataset(
            data_dir=data_dir,
            vocab_path=vocab_path,
            model_enc_seqlen=8,
            model_dec_seqlen=8,
            model_max_bars=4,
            pieces=['0.pkl'],
            pad_to_same=True,
            use_difficulty=use_difficulty,
            tokenization_method='standard',
        )

    def test_bar_level_difficulty_expands_to_token_level_sequence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, 'vocab.pkl')
            self._write_vocab(vocab_path)

            bar_pos = [0, 4, 8]
            tokens = [0, 2, 3, 4, 0, 2, 3, 4]
            difficulty_classes = [self._difficulty(1), self._difficulty(5)]

            with open(os.path.join(tmpdir, '0.pkl'), 'wb') as f:
                pickle.dump((bar_pos, tokens, difficulty_classes), f)

            dset = self._build_dataset(tmpdir, vocab_path, use_difficulty=True)
            sample = dset[0]

            self.assertTrue(np.all(sample['s_tech_cls_seq'][:4] == 1))
            self.assertTrue(np.all(sample['s_tech_cls_seq'][4:8] == 5))
            self.assertEqual(sample['s_tech_cls'].tolist(), [1, 5])
            self.assertEqual(sample['s_indep_cls'].tolist(), [1, 5])

    def test_mismatch_between_bar_count_and_difficulty_count_fails_fast(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, 'vocab.pkl')
            self._write_vocab(vocab_path)

            bar_pos = [0, 4, 8]
            tokens = [0, 2, 3, 4, 0, 2, 3, 4]
            difficulty_classes = [self._difficulty(2)]  # 2 bars に対して 1 label

            with open(os.path.join(tmpdir, '0.pkl'), 'wb') as f:
                pickle.dump((bar_pos, tokens, difficulty_classes), f)

            dset = self._build_dataset(tmpdir, vocab_path, use_difficulty=True)
            with self.assertRaises(ValueError):
                _ = dset[0]

    def test_missing_difficulty_payload_fails_fast_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, 'vocab.pkl')
            self._write_vocab(vocab_path)

            bar_pos = [0, 4, 8]
            tokens = [0, 2, 3, 4, 0, 2, 3, 4]

            with open(os.path.join(tmpdir, '0.pkl'), 'wb') as f:
                pickle.dump((bar_pos, tokens), f)

            dset = self._build_dataset(tmpdir, vocab_path, use_difficulty=True)
            with self.assertRaises(ValueError):
                _ = dset[0]


if __name__ == '__main__':
    unittest.main()
