#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.train.train_drum \
    --config config/drum_config_standard-lakh1-kl14.yaml "$@"
