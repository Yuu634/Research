#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m scripts.tasks generate "$@"
