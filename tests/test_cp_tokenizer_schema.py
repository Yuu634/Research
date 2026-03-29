import os
import sys
import tempfile
import unittest
from pathlib import Path

import miditoolkit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer


class DrumCpTokenizerSchemaTests(unittest.TestCase):
    def _write_midi(self, midi_obj, path):
        midi_obj.dump(path)

    def _make_note(self, velocity, pitch, start, end):
        return miditoolkit.Note(velocity=velocity, pitch=pitch, start=start, end=end)

    def test_cp_schema_pad_rules_and_positions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "cp.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            # Order matters to make limb assignment deterministic.
            drum_inst.notes.append(self._make_note(velocity=80, pitch=44, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=80, pitch=36, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=80, pitch=38, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=80, pitch=42, start=0, end=120))
            midi_obj.instruments.append(drum_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = DrumCPTokenizer()
            data, _ = tokenizer.midi_to_cp_data(midi_path)

            type_note = tokenizer.type_token2idx["<NOTE>"]
            limb_na = tokenizer.limb_token2idx["<NA>"]

            for idx, cp_type in enumerate(data["cp_type"]):
                if cp_type == type_note:
                    self.assertNotEqual(data["cp_pos"][idx], tokenizer.pos_pad_value)
                    self.assertGreaterEqual(data["cp_pos"][idx], 0)
                    self.assertLess(data["cp_pos"][idx], 24)
                else:
                    self.assertEqual(data["cp_pos"][idx], tokenizer.pos_pad_value)
                    self.assertEqual(data["cp_hand1"][idx], limb_na)
                    self.assertEqual(data["cp_hand2"][idx], limb_na)
                    self.assertEqual(data["cp_right_foot"][idx], limb_na)
                    self.assertEqual(data["cp_left_foot"][idx], limb_na)

    def test_cp_limb_assignment_priorities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "limb.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            drum_inst.notes.append(self._make_note(velocity=80, pitch=44, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=80, pitch=36, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=80, pitch=38, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=80, pitch=42, start=0, end=120))
            midi_obj.instruments.append(drum_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = DrumCPTokenizer()
            data, _ = tokenizer.midi_to_cp_data(midi_path)

            type_note = tokenizer.type_token2idx["<NOTE>"]
            cp_index = data["cp_type"].index(type_note)

            hand1 = tokenizer.idx2limb_token[int(data["cp_hand1"][cp_index])]
            hand2 = tokenizer.idx2limb_token[int(data["cp_hand2"][cp_index])]
            rf = tokenizer.idx2limb_token[int(data["cp_right_foot"][cp_index])]
            lf = tokenizer.idx2limb_token[int(data["cp_left_foot"][cp_index])]

            self.assertEqual(hand1, "SNARE_HIT_Normal")
            self.assertEqual(hand2, "HH_CLOSED_HIT_Normal")
            self.assertEqual(rf, "KICK_HIT_Normal")
            self.assertEqual(lf, "HH_PEDAL_HIT_Normal")

    def test_left_foot_is_inferred_from_hihat_open_close(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "hh_state.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            drum_inst.notes.append(self._make_note(velocity=90, pitch=42, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=90, pitch=42, start=120, end=240))
            drum_inst.notes.append(self._make_note(velocity=90, pitch=46, start=240, end=360))
            midi_obj.instruments.append(drum_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = DrumCPTokenizer()
            data, _ = tokenizer.midi_to_cp_data(midi_path)

            type_note = tokenizer.type_token2idx["<NOTE>"]
            note_indices = [i for i, t in enumerate(data["cp_type"]) if t == type_note]

            self.assertGreaterEqual(len(note_indices), 3)

            lf_closed_first = tokenizer.idx2limb_token[int(data["cp_left_foot"][note_indices[0]])]
            lf_closed_second = tokenizer.idx2limb_token[int(data["cp_left_foot"][note_indices[1]])]
            lf_open = tokenizer.idx2limb_token[int(data["cp_left_foot"][note_indices[2]])]

            self.assertEqual(lf_closed_first, "HH_PEDAL_HIT_Down")
            self.assertEqual(lf_closed_second, "<NONE>")
            self.assertEqual(lf_open, "HH_PEDAL_HIT_Up")

    def test_tambourine_is_assigned_to_hand_slot(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "tambourine.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            drum_inst.notes.append(self._make_note(velocity=90, pitch=54, start=0, end=120))
            midi_obj.instruments.append(drum_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = DrumCPTokenizer()
            data, _ = tokenizer.midi_to_cp_data(midi_path)

            type_note = tokenizer.type_token2idx["<NOTE>"]
            note_indices = [i for i, t in enumerate(data["cp_type"]) if t == type_note]
            self.assertTrue(note_indices)

            hand_tokens = set()
            for note_index in note_indices:
                hand_tokens.add(tokenizer.idx2limb_token[int(data["cp_hand1"][note_index])])
                hand_tokens.add(tokenizer.idx2limb_token[int(data["cp_hand2"][note_index])])

            self.assertIn("TAMBOURINE_HIT_Normal", hand_tokens)

    def test_tambourine_is_not_dropped_with_multiple_hand_notes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "tambourine_mix.mid")
            midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)

            drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
            drum_inst.notes.append(self._make_note(velocity=90, pitch=54, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=90, pitch=38, start=0, end=120))
            drum_inst.notes.append(self._make_note(velocity=90, pitch=42, start=0, end=120))
            midi_obj.instruments.append(drum_inst)

            self._write_midi(midi_obj, midi_path)

            tokenizer = DrumCPTokenizer()
            data, _ = tokenizer.midi_to_cp_data(midi_path)

            type_note = tokenizer.type_token2idx["<NOTE>"]
            note_indices = [i for i, t in enumerate(data["cp_type"]) if t == type_note]
            self.assertTrue(note_indices)

            note_index = note_indices[0]
            hand1 = tokenizer.idx2limb_token[int(data["cp_hand1"][note_index])]
            hand2 = tokenizer.idx2limb_token[int(data["cp_hand2"][note_index])]

            self.assertIn("TAMBOURINE_HIT_Normal", {hand1, hand2})


if __name__ == "__main__":
    unittest.main()
