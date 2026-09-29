#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if (( $# < 1 )); then
    echo "使用方法: bash tasks/analysis.sh MODEL_DIR [MODEL_DIR ...]" >&2
    exit 2
fi
for model_dir in "$@"; do
    model_name=$(basename "${model_dir%/}")
    python -m scripts.analysis.inspect_alpha_limb \
        --model_dir "$model_dir" --output_dir "output/alpha_limb/$model_name"
    python -m scripts.analysis.inspect_position_bias \
        --model_dir "$model_dir" --output_dir "output/position_bias/$model_name"
    python -m scripts.analysis.visualize_attention_maps \
        --model_dir "$model_dir" --output_dir "output/attention_maps/$model_name"
    python -m scripts.analysis.qk_limb_bias_ratio \
        --model_dir "$model_dir" --output_dir "output/qk_limb_ratio/$model_name"
done
