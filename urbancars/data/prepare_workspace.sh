#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "$EXPERIMENT_ROOT/data/env.sh"

mkdir -p "$URBANCARS_RAW_ROOT" "$URBANCARS_EXP_ROOT" "$MPLCONFIGDIR" "$VENDOR_ROOT"
ln -sfn "$URBANCARS_RAW_ROOT" "$VENDOR_ROOT/data"
ln -sfn "$URBANCARS_EXP_ROOT" "$VENDOR_ROOT/exp"

printf 'workspace_ready\n'
printf 'vendor_root=%s\n' "$VENDOR_ROOT"
printf 'raw_root=%s\n' "$URBANCARS_RAW_ROOT"
printf 'exp_root=%s\n' "$URBANCARS_EXP_ROOT"
