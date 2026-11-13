#!/usr/bin/env bash
# Validation-set DFR (last-layer retraining on group-balanced validation data) on the ERM and IS backbones (ResNet-50,
# seeds 1 / 21 / 42), with the official DFR functions and hyperparameter grids of
# https://github.com/PolinaKirichenko/deep_feature_reweighting (file dfr_evaluate_spurious.py; not included here).
# Also reports DFR(Train). The DFR sampling seed equals the training seed for both backbones of a pair.
set -euo pipefail
source "$(dirname "$0")/env.sh"
export DFR_OFFICIAL_SCRIPT=${DFR_OFFICIAL_SCRIPT:-/path/to/deep_feature_reweighting/dfr_evaluate_spurious.py}

for seed in 1 21 42; do
  $PY $PKG/eval/dfr_official.py --label erm_seed$seed --seed $seed \
    --checkpoint $CKPT_ROOT/erm/seed$seed/checkpoints/erm_best.pt --output $OUT_ROOT/dfr/seed$seed/erm.json
  $PY $PKG/eval/dfr_official.py --label full_is_seed$seed --seed $seed \
    --checkpoint $CKPT_ROOT/full_is/seed$seed/checkpoints/steered_finetuned_best.pt --output $OUT_ROOT/dfr/seed$seed/full_is.json
done

# Robustness evaluation of the DFR(Val) models: refit the same heads (checked against the saved test WGA) and fold them
# into a ResNet-50 checkpoint, then score with eval_robustness.py (needs the probe renders of 14_evaluation.sh, step 1)
# and summarize.
mkdir -p $OUT_ROOT/dfr_fold
for seed in 1 21 42; do
  for base in erm full_is; do $PY $PKG/eval/dfr_refit_fold.py $seed $base; done
  $PY $PKG/eval/eval_robustness.py s${seed}_dfrerm=rn50:$OUT_ROOT/dfr_fold/ckpt/erm_dfrval_s$seed.pt \
    s${seed}_dfris=rn50:$OUT_ROOT/dfr_fold/ckpt/full_is_dfrval_s$seed.pt
done
$PY $PKG/eval/summarize_dfr.py
