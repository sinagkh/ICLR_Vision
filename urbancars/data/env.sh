#!/usr/bin/env bash
# Shared environment for the dataset scripts in this folder.
# Required: DATA_ROOT (raw source datasets and the generated UrbanCars dataset are written here).
# Optional: VENDOR_ROOT (clone of the Whac-A-Mole repository with whac_a_mole.patch applied), PY (python executable).

EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VENDOR_ROOT="${VENDOR_ROOT:-$EXPERIMENT_ROOT/vendor/Whac-A-Mole}"
URBANCARS_RAW_ROOT="${URBANCARS_RAW_ROOT:-${DATA_ROOT:?set DATA_ROOT to the dataset root}}"
URBANCARS_EXP_ROOT="${URBANCARS_EXP_ROOT:-$URBANCARS_RAW_ROOT/exp}"
URBANCARS_MASKFORMER_ROOT="${URBANCARS_MASKFORMER_ROOT:-$VENDOR_ROOT/create_datasets/urbancars/maskformer}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$URBANCARS_EXP_ROOT/mplconfig}"
PY="${PY:-python3}"

export EXPERIMENT_ROOT
export VENDOR_ROOT
export URBANCARS_RAW_ROOT
export URBANCARS_EXP_ROOT
export URBANCARS_MASKFORMER_ROOT
export MPLCONFIGDIR
export PY
export PYTHONPATH="$VENDOR_ROOT${PYTHONPATH:+:$PYTHONPATH}"
