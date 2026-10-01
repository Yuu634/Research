#!/usr/bin/env bash
# リポジトリ直下で実行。prepare→select→testを個別に確認できる。
set -euo pipefail
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m scripts.train.tune_beam_muse \
  --phase "${1:-smoke}" --output "${2:-output/beam_muse_smoke}" "${@:3}"
