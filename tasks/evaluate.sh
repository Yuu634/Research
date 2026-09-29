#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.evaluate.evaluate \
    --input_midi_dir data/test_dataset \
    --model_path trained_model/checkpoints_drum_standard-lakh1-kl14/params/best_params.pt \
    --difficulty_bounds data/metadata_new/difficulty_bounds_standard-lakh.pkl \
    --difficulty_labels_file config/evaluation_difficulty_labels.json \
    --output_dir output/eval/checkpoints_drum_standard-lakh1-kl14 \
    --use_input_bpm "$@"
