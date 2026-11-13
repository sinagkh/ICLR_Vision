#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPERIMENT_ROOT/data/env.sh"

"$EXPERIMENT_ROOT/data/prepare_workspace.sh" >/dev/null
cd "$VENDOR_ROOT"
PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" create_datasets/urbancars/predict_car_mask.py --split train
PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" create_datasets/urbancars/predict_car_mask.py --split test
