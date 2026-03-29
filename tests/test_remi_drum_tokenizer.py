import os
import sys
import tempfile
import unittest
from pathlib import Path

import miditoolkit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.preprocess.prepare_drum_dataset import RemiDrumTokenizer


class RemiDrumTokenizerTests(unittest.TestCase):
    def _write_midi(self, midi_obj, path):
        midi_obj.dump(path)

    def _make_note(self, velocity, pitch, start, end):
        return miditoolkit.Note(velocity=velocity, pitch=pitch, start=start, end=end)

    def test_remi_tokenizer_ignores_non_drum_tracks(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "mix.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            drum_inst.notes.append(self._make_note(velocity=80, pitch=36, start=0, end=120))
            midi_obj.instruments.append(drum_inst)

            piano_inst = miditoolkit.Instrument(program=0, is_drum=False, name="piano")
            piano_inst.notes.append(self._make_note(velocity=90, pitch=60, start=0, end=120))
            midi_obj.instruments.append(piano_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = RemiDrumTokenizer()
            tokens, _ = tokenizer.midi_to_tokens(midi_path)

            self.assertIn("Note_Pitch_36", tokens)
            self.assertNotIn("Note_Pitch_60", tokens)
            self.assertEqual(tokenizer.last_conversion_info["drum_track_count"], 1)
            self.assertEqual(tokenizer.last_conversion_info["non_drum_tracks_excluded"], 1)

    def test_remi_tokenizer_raises_when_no_drum_track(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "no_drum.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            piano_inst = miditoolkit.Instrument(program=0, is_drum=False, name="piano")
            piano_inst.notes.append(self._make_note(velocity=90, pitch=60, start=0, end=120))
            midi_obj.instruments.append(piano_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = RemiDrumTokenizer()
            with self.assertRaisesRegex(ValueError, "No drum track found"):
                tokenizer.midi_to_tokens(midi_path)

    def test_remi_velocity_bins_follow_shared_three_level_policy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "vel.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            drum_inst.notes.extend(
                [
                    self._make_note(velocity=14, pitch=36, start=0, end=120),
                    self._make_note(velocity=15, pitch=36, start=20, end=140),
                    self._make_note(velocity=100, pitch=36, start=40, end=160),
                ]
            )
            midi_obj.instruments.append(drum_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = RemiDrumTokenizer()
            tokens, _ = tokenizer.midi_to_tokens(midi_path)

            self.assertIn("Note_Velocity_0", tokens)
            self.assertIn("Note_Velocity_16", tokens)
            self.assertIn("Note_Velocity_31", tokens)

    def test_remi_position_quantization_matches_standard_formula(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "pos.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            drum_inst.notes.extend(
                [
                    self._make_note(velocity=80, pitch=36, start=0, end=120),
                    self._make_note(velocity=80, pitch=36, start=20, end=140),
                    self._make_note(velocity=80, pitch=36, start=479, end=600),
                ]
            )
            midi_obj.instruments.append(drum_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = RemiDrumTokenizer()
            tokens, _ = tokenizer.midi_to_tokens(midi_path)
            beat_tokens = [t for t in tokens if t.startswith("<BEAT_")]

            self.assertIn("<BEAT_0>", beat_tokens)
            self.assertIn("<BEAT_1>", beat_tokens)
            self.assertIn("<BEAT_23>", beat_tokens)


if __name__ == "__main__":
    unittest.main()
