#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m src.utils.plot_loss_curves --trained_model_dir trained_model "$@"
