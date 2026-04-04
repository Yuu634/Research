import os
import pickle
import tempfile
import unittest
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.tokenization_mode import (
    CP_REQUIRED_BATCH_KEYS,
    ensure_batch_mode_consistency,
    TOKENIZATION_METADATA_FILENAME,
    ensure_runtime_mode_compatibility,
    read_dataset_tokenization_metadata,
    resolve_artifact_tokenization_method,
    validate_tokenization_method,
    write_dataset_tokenization_metadata,
)


class TokenizationModeValidationTests(unittest.TestCase):
    def _write_pickle(self, path, payload):
        with open(path, "wb") as f:
            pickle.dump(payload, f)

    def test_validate_tokenization_method_accepts_supported_values(self):
        self.assertEqual(validate_tokenization_method("standard"), "standard")
        self.assertEqual(validate_tokenization_method("remi"), "remi")
        self.assertEqual(validate_tokenization_method("cp_limb_v1"), "cp_limb_v1")

    def test_validate_tokenization_method_rejects_invalid_value(self):
        with self.assertRaises(ValueError):
            validate_tokenization_method("unknown_mode")

    def test_resolve_artifact_mode_prefers_vocab_cp_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "vocab.pkl")
            cp_vocab_payload = {
                "tokenization_method": "cp_limb_v1",
                "event_type2idx": {"<PAD_EVENT>": 0, "STRUCT": 1, "CP": 2},
            }
            self._write_pickle(vocab_path, cp_vocab_payload)

            mode, source = resolve_artifact_tokenization_method(vocab_path)
            self.assertEqual(mode, "cp_limb_v1")
            self.assertEqual(source, "vocab")

    def test_resolve_artifact_mode_uses_dataset_metadata_for_tuple_vocab(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(data_dir, exist_ok=True)
            vocab_path = os.path.join(tmpdir, "vocab.pkl")
            standard_or_remi_tuple = ({"<BAR>": 0, "<EOS>": 1}, {0: "<BAR>", 1: "<EOS>"})
            self._write_pickle(vocab_path, standard_or_remi_tuple)

            metadata_path = write_dataset_tokenization_metadata(data_dir, "remi")
            self.assertTrue(metadata_path.endswith(TOKENIZATION_METADATA_FILENAME))

            mode, source = resolve_artifact_tokenization_method(vocab_path, data_dir)
            self.assertEqual(mode, "remi")
            self.assertEqual(source, "dataset_metadata")

    def test_ensure_runtime_mode_compatibility_allows_legacy_standard_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "vocab.pkl")
            standard_or_remi_tuple = ({"<BAR>": 0, "<EOS>": 1}, {0: "<BAR>", 1: "<EOS>"})
            self._write_pickle(vocab_path, standard_or_remi_tuple)

            resolved = ensure_runtime_mode_compatibility(
                requested_mode="standard",
                vocab_path=vocab_path,
                data_dir=None,
            )
            self.assertEqual(resolved, "standard")

    def test_ensure_runtime_mode_compatibility_rejects_ambiguous_nonstandard_without_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            vocab_path = os.path.join(tmpdir, "vocab.pkl")
            standard_or_remi_tuple = ({"<BAR>": 0, "<EOS>": 1}, {0: "<BAR>", 1: "<EOS>"})
            self._write_pickle(vocab_path, standard_or_remi_tuple)

            with self.assertRaises(ValueError):
                ensure_runtime_mode_compatibility(
                    requested_mode="remi",
                    vocab_path=vocab_path,
                    data_dir=None,
                )

    def test_ensure_runtime_mode_compatibility_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(data_dir, exist_ok=True)
            vocab_path = os.path.join(tmpdir, "vocab.pkl")
            standard_or_remi_tuple = ({"<BAR>": 0, "<EOS>": 1}, {0: "<BAR>", 1: "<EOS>"})
            self._write_pickle(vocab_path, standard_or_remi_tuple)
            write_dataset_tokenization_metadata(data_dir, "remi")

            with self.assertRaises(ValueError):
                ensure_runtime_mode_compatibility(
                    requested_mode="standard",
                    vocab_path=vocab_path,
                    data_dir=data_dir,
                )

    def test_write_dataset_tokenization_metadata_accepts_extra_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = os.path.join(tmpdir, "dataset")
            os.makedirs(data_dir, exist_ok=True)

            write_dataset_tokenization_metadata(
                data_dir,
                "remi",
                extra_metadata={
                    "input_scope": "drum_only",
                    "non_drum_tracks_policy": "excluded",
                },
            )

            metadata = read_dataset_tokenization_metadata(data_dir)
            self.assertIsNotNone(metadata)
            self.assertEqual(metadata["tokenization_method"], "remi")
            self.assertEqual(metadata["input_scope"], "drum_only")
            self.assertEqual(metadata["non_drum_tracks_policy"], "excluded")

    def test_ensure_batch_mode_consistency_accepts_cp_payload(self):
        cp_batch = {key: [] for key in CP_REQUIRED_BATCH_KEYS}
        ensure_batch_mode_consistency("cp_limb_v1", cp_batch)

    def test_ensure_batch_mode_consistency_rejects_missing_cp_keys(self):
        cp_batch = {
            "cp_event_type_input": [],
            "cp_event_type_target": [],
            "cp_type_input": [],
        }
        with self.assertRaises(ValueError):
            ensure_batch_mode_consistency("cp_limb_v1", cp_batch)

    def test_ensure_batch_mode_consistency_rejects_cp_keys_in_standard_mode(self):
        standard_batch = {
            "enc_input": [],
            "dec_input": [],
            "dec_target": [],
            "cp_type_input": [],
        }
        with self.assertRaises(ValueError):
            ensure_batch_mode_consistency("standard", standard_batch)


if __name__ == "__main__":
    unittest.main()
