#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if (( $# < 3 )); then
    echo "使用方法: bash tasks/evaluate_ensemble.sh OUTPUT_DIR MODEL1.pt MODEL2.pt [MODEL.pt ...]" >&2
    exit 2
fi
OUTPUT_DIR=$1
shift
python -m scripts.evaluate.evaluate \
    --input_midi_dir "${INPUT_MIDI_DIR:-data/test_dataset}" \
    --model_path "$@" --ensemble_decoding \
    --difficulty_bounds "${DIFFICULTY_BOUNDS:-data/metadata_new/difficulty_bounds_standard-lakh.pkl}" \
    --difficulty_labels_file "${DIFFICULTY_LABELS:-config/evaluation_difficulty_labels.json}" \
    --output_dir "$OUTPUT_DIR" --use_input_bpm
