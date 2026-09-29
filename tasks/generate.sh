#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if (( $# < 2 )); then
    echo "使用方法: bash tasks/generate.sh 入力.mid 出力.mid [生成オプション...]" >&2
    exit 2
fi
INPUT_MIDI=$1
OUTPUT_MIDI=$2
shift 2
python -m scripts.generate.generate_drum_from_input \
    --model_path trained_model/checkpoints_drum_standard-lakh1-kl14/params/best_params.pt \
    --difficulty_bounds data/metadata_new/difficulty_bounds_standard-lakh.pkl \
    --input_midi "$INPUT_MIDI" --output_midi "$OUTPUT_MIDI" \
    --difficulty_mode global_shift --difficulty_shift 0 0 0 0 0 "$@"
