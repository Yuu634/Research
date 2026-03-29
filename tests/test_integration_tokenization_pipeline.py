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
from src.data.drum_dataloader import DrumTransformerDataset
from src.tokenizers.drum_cp_tokenizer import DrumCPTokenizer
from src.tokenizers.drum_tokenizer import DrumTokenizer
from src.utils.drum_to_midi import DrumToken2MIDI, cp_data_to_midi


class TokenizationPipelineIntegrationTests(unittest.TestCase):
    def _make_note(self, velocity, pitch, start, end):
        return miditoolkit.Note(velocity=velocity, pitch=pitch, start=start, end=end)

    def _write_simple_midi(self, path):
        midi_obj = miditoolkit.MidiFile(ticks_per_beat=480)
        drum_inst = miditoolkit.Instrument(program=0, is_drum=True, name="drums")
        drum_inst.notes.append(self._make_note(velocity=90, pitch=42, start=0, end=120))
        drum_inst.notes.append(self._make_note(velocity=100, pitch=36, start=0, end=120))
        drum_inst.notes.append(self._make_note(velocity=85, pitch=38, start=240, end=360))
        drum_inst.notes.append(self._make_note(velocity=90, pitch=46, start=480, end=600))
        midi_obj.instruments.append(drum_inst)
        midi_obj.dump(path)

    def _save_piece(self, data_dir, payload):
        import pickle

        os.makedirs(data_dir, exist_ok=True)
        piece_path = os.path.join(data_dir, "0.pkl")
        with open(piece_path, "wb") as f:
            pickle.dump(payload, f)
        return piece_path

    def test_standard_preprocess_to_dataset_and_generate_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "standard.mid")
            self._write_simple_midi(midi_path)

            tokenizer = DrumTokenizer()
            tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)
            indices = tokenizer.tokens_to_indices(tokens).tolist()

            vocab_path = os.path.join(tmpdir, "standard_vocab.pkl")
            tokenizer.save_vocab(vocab_path)

            data_dir = os.path.join(tmpdir, "standard_dataset")
            self._save_piece(data_dir, (bar_positions, indices))

            dset = DrumTransformerDataset(
                data_dir,
                vocab_path,
                model_enc_seqlen=64,
                model_dec_seqlen=128,
                model_max_bars=4,
                tokenization_method="standard",
            )
            sample = dset[0]
            self.assertIn("dec_input", sample)
            self.assertGreater(sample["length"], 0)

            converter = DrumToken2MIDI()
            midi_obj = converter.tokens_to_midi(tokens)
            self.assertGreater(len(midi_obj.instruments[0].notes), 0)

    def test_remi_preprocess_to_dataset_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "remi.mid")
            self._write_simple_midi(midi_path)

            tokenizer = RemiDrumTokenizer()
            tokens, bar_positions = tokenizer.midi_to_tokens(midi_path)
            indices = tokenizer.tokens_to_indices(tokens).tolist()

            vocab_path = os.path.join(tmpdir, "remi_vocab.pkl")
            tokenizer.save_vocab(vocab_path)

            data_dir = os.path.join(tmpdir, "remi_dataset")
            self._save_piece(data_dir, (bar_positions, indices))

            dset = DrumTransformerDataset(
                data_dir,
                vocab_path,
                model_enc_seqlen=64,
                model_dec_seqlen=128,
                model_max_bars=4,
                tokenization_method="remi",
            )
            sample = dset[0]
            self.assertIn("dec_input", sample)
            self.assertGreater(sample["length"], 0)

    def test_cp_preprocess_to_dataset_and_generate_smoke(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            midi_path = os.path.join(tmpdir, "cp.mid")
            self._write_simple_midi(midi_path)

            tokenizer = DrumCPTokenizer()
            cp_data, bar_positions = tokenizer.midi_to_cp_data(midi_path)

            vocab_path = os.path.join(tmpdir, "cp_vocab.pkl")
            tokenizer.save_vocab(vocab_path)

            data_dir = os.path.join(tmpdir, "cp_dataset")
            self._save_piece(data_dir, (bar_positions, cp_data))

            dset = DrumTransformerDataset(
                data_dir,
                vocab_path,
                model_enc_seqlen=64,
                model_dec_seqlen=128,
                model_max_bars=4,
                tokenization_method="cp_limb_v1",
            )
            sample = dset[0]
            self.assertIn("cp_type_input", sample)
            self.assertGreater(sample["length"], 0)

            midi_obj = cp_data_to_midi(
                cp_data,
                tokenizer.idx2type_token,
                tokenizer.idx2limb_token,
                os.path.join(tmpdir, "cp_out.mid"),
                bpm=120,
            )
            self.assertGreater(len(midi_obj.instruments[0].notes), 0)


if __name__ == "__main__":
    unittest.main()
