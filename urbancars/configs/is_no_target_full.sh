#!/usr/bin/env bash
# IS without the interaction target at the full staged schedule, compared with IS.
#
# The arm without the target runs the IS stage commands of configs/is_staged.sh with the two interaction-target
# weights set to 0 (--lambda_bg 0 --lambda_co 0); every other term (main-effect and object guards), the schedule,
# the checkpoint selection and the re-audits before stages 2 and 3 are unchanged. Stage 1 starts from the same
# ERM warm start and audit as IS. Requires configs/is_staged.sh for the same seed (ERM start and the IS arm).
#
# Then: benchmark metrics of both arms at the selected stage-3 checkpoint.
#
# Usage: bash configs/is_no_target_full.sh [SEED ...]   (default: 0 1 2)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0 1 2)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

NO_TARGET=(--lambda_bg 0 --lambda_co 0)

for S in "${SEEDS[@]}"; do
  erm20_and_audit "$S"
  I=$CKPT_ROOT/is/seed$S
  R=$CKPT_ROOT/is_no_target/seed$S
  A=$OUT_ROOT/audits/is_no_target/seed$S

  run "$R/stage1.log" "$PY" -u train/train_is.py "${IS_ARGS[@]}" "${IS_STAGE1[@]}" --seed "$S" "${NO_TARGET[@]}" \
    --load_checkpoint "$I/erm20/last.pth" --scores_csv "$OUT_ROOT/audits/is/seed$S/audit1/audit_scores.csv" \
    --output_dir "$R/stage1"
  audit_train "$R/stage1/best.pth" "$A/audit2"

  run "$R/stage2.log" "$PY" -u train/train_is.py "${IS_ARGS[@]}" "${IS_STAGE2[@]}" --seed "$S" "${NO_TARGET[@]}" \
    --load_checkpoint "$R/stage1/best.pth" --scores_csv "$A/audit2/audit_scores.csv" --output_dir "$R/stage2"
  audit_train "$R/stage2/best.pth" "$A/audit3"

  run "$R/stage3.log" "$PY" -u train/train_is.py "${IS_ARGS[@]}" "${IS_STAGE3[@]}" --seed "$S" "${NO_TARGET[@]}" \
    --load_checkpoint "$R/stage2/best.pth" --scores_csv "$A/audit3/audit_scores.csv" --output_dir "$R/stage3"
done

run "$OUT_ROOT/scores/no_target_full.txt" "$PY" eval/score_no_target.py "${SEEDS[@]}"
