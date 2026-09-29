#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# ダウンロード・展開済みの Lakh MIDI からドラムを抽出する。
python -m data.cleaning_lakhmidi \
    --lakhmidi_dir "${1:-data/raw/lmd_full}" \
    --output_dir "${2:-data/filtered/lakhmidi}"
