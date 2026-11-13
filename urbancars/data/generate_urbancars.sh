#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPERIMENT_ROOT/data/env.sh"

"$EXPERIMENT_ROOT/data/prepare_workspace.sh" >/dev/null
cd "$VENDOR_ROOT"
PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" create_datasets/urbancars/gen_urbancars.py --split train
PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" create_datasets/urbancars/gen_urbancars.py --split val
PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" create_datasets/urbancars/gen_urbancars.py --split test
PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" create_datasets/urbancars/urbancars_aug_gen_bg_only.py --data_root=data/urbancars/bg-0.95_co_occur_obj-0.95/train
PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$EXPERIMENT_ROOT/data/verify_setup.py"
