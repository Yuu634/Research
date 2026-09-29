#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# 基準設定の入力は E-GMD とフィルタ済み Lakh。引数で入力ディレクトリを変更できる。
if (( $# == 0 )); then
    set -- data/raw/e-gmd data/filtered/lakhmidi
fi
DATASET_DIR=data/processed_new/drum_dataset_standard-lakh
VOCAB_PATH=data/metadata_new/drum_vocab_standard-lakh.pkl
BOUNDS_PATH=data/metadata_new/difficulty_bounds_standard-lakh.pkl

mkdir -p data/metadata_new
python -m scripts.preprocess.prepare_drum_dataset \
    --midi_dir "$@" \
    --output_dir "$DATASET_DIR" \
    --vocab_path "$VOCAB_PATH" \
    --tokenization_method standard \
    --file_extension .midi .mid \
    --train_ratio 0.9 --seed 42
python -m src.utils.compute_difficulty_bounds \
    "$DATASET_DIR" "$VOCAB_PATH" "$BOUNDS_PATH"
python -m scripts.preprocess.prepare_drum_dataset_with_difficulty \
    "$DATASET_DIR" "$VOCAB_PATH" "$BOUNDS_PATH" "${DATASET_DIR}_with_difficulty" \
    --train_ratio 0.9 --seed 42
