#!/usr/bin/env bash
# Interaction Steering (joint BG + CoObj targets) with the staged warm-start schedule. Seeds: 0 1 2.
#
# Per seed:
#   ERM warm start, 20 epochs (lr 1e-3)                    -> $CKPT_ROOT/is/seed$S/erm20
#   audit 1 of erm20/last.pth                               -> $OUT_ROOT/audits/is/seed$S/audit1
#   stage 1: 100 epochs, lr 4e-5, target 0.9 x audited interaction, from erm20/last.pth  -> stage1
#   audit 2 of stage1/best.pth                              -> audit2
#   stage 2:  90 epochs, lr 3e-5, target 0.7, from stage1/best.pth                       -> stage2
#   audit 3 of stage2/best.pth                              -> audit3
#   stage 3:  90 epochs, lr 2e-5, target 0.5, from stage2/best.pth                       -> stage3
# Within a stage, best.pth is selected by paper_slice_mean (mean validation accuracy over the three
# rare-slice groups). The reported model is the stage with the highest selector value (stage 3 for
# all three seeds) at its best epoch; see $CKPT_ROOT/is/seed$S/run_summary.json.
#
# Usage: bash configs/is_staged.sh [SEED ...]        (default: 0 1 2)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0 1 2)

for S in "${SEEDS[@]}"; do
  R=$CKPT_ROOT/is/seed$S
  A=$OUT_ROOT/audits/is/seed$S

  erm20_and_audit "$S"

  run "$R/stage1.log" "$PY" -u train/train_is.py "${IS_ARGS[@]}" "${IS_STAGE1[@]}" --seed "$S" \
    --load_checkpoint "$R/erm20/last.pth" --scores_csv "$A/audit1/audit_scores.csv" --output_dir "$R/stage1"
  audit_train "$R/stage1/best.pth" "$A/audit2"

  run "$R/stage2.log" "$PY" -u train/train_is.py "${IS_ARGS[@]}" "${IS_STAGE2[@]}" --seed "$S" \
    --load_checkpoint "$R/stage1/best.pth" --scores_csv "$A/audit2/audit_scores.csv" --output_dir "$R/stage2"
  audit_train "$R/stage2/best.pth" "$A/audit3"

  run "$R/stage3.log" "$PY" -u train/train_is.py "${IS_ARGS[@]}" "${IS_STAGE3[@]}" --seed "$S" \
    --load_checkpoint "$R/stage2/best.pth" --scores_csv "$A/audit3/audit_scores.csv" --output_dir "$R/stage3"

  select_best_stage "$R" stage1 stage2 stage3
done
