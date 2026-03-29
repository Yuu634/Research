import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer
from src.tokenizers.drum_tokenizer import DrumTokenizer
from src.utils.tokenization_policy import (
    CP_POS_PAD_VALUE,
    POSITIONS_PER_BEAT,
    classify_velocity_level,
    quantize_position_in_beat,
    quantize_step_in_bar,
)


class TokenizationPolicyTests(unittest.TestCase):
    def test_velocity_classification_boundaries(self):
        self.assertEqual(classify_velocity_level(0), "Ghost")
        self.assertEqual(classify_velocity_level(14), "Ghost")
        self.assertEqual(classify_velocity_level(15), "Normal")
        self.assertEqual(classify_velocity_level(99), "Normal")
        self.assertEqual(classify_velocity_level(100), "Accent")
        self.assertEqual(classify_velocity_level(127), "Accent")

    def test_velocity_classification_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            classify_velocity_level(-1)
        with self.assertRaises(ValueError):
            classify_velocity_level(128)

    def test_velocity_classification_rejects_non_int(self):
        with self.assertRaises(ValueError):
            classify_velocity_level("100")

    def test_position_quantization_boundaries(self):
        self.assertEqual(quantize_position_in_beat(0, 480), 0)
        self.assertEqual(quantize_position_in_beat(20, 480), 1)
        self.assertEqual(quantize_position_in_beat(479, 480), 23)
        self.assertEqual(quantize_position_in_beat(480, 480), 0)

    def test_step_quantization_uses_standard_24_division(self):
        self.assertEqual(quantize_step_in_bar(0, 480), 0)
        self.assertEqual(quantize_step_in_bar(479, 480), 23)
        self.assertEqual(quantize_step_in_bar(480, 480), 24)
        self.assertEqual(quantize_step_in_bar(959, 480), 47)

    def test_shared_constants_for_position_policy(self):
        self.assertEqual(POSITIONS_PER_BEAT, 24)
        self.assertEqual(CP_POS_PAD_VALUE, 24)

    def test_drum_tokenizer_uses_shared_velocity_policy(self):
        tokenizer = DrumTokenizer()
        self.assertEqual(tokenizer._velocity_to_level(14), "Ghost")
        self.assertEqual(tokenizer._velocity_to_level(15), "Normal")
        self.assertEqual(tokenizer._velocity_to_level(100), "Accent")

    def test_cp_tokenizer_uses_shared_pos_pad(self):
        tokenizer = DrumCPTokenizer()
        self.assertEqual(tokenizer.pos_pad_value, CP_POS_PAD_VALUE)


if __name__ == "__main__":
    unittest.main()
