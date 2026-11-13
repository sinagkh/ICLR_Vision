#!/usr/bin/env bash
# Single-target ablations: BG-only and CoObj-only steering with the same staged schedule as the joint runs
# (100/90/90 epochs, target 0.9/0.7/0.5 of the audited interaction, re-audit before each stage).
# Seeds: 0 1 2. Each seed starts from the same ERM warm start as the joint IS run
# ($CKPT_ROOT/is/seed$S/erm20/last.pth; created here if missing).
#
# Differences from the joint runs (train/train_is_single_target.py):
#   - only the four corners of the targeted interaction are forwarded in the steering step (--exclusive_interaction);
#   - BG-only measures obj x bg with the co-object present (--bg_delta_mode co_present);
#   - guards and selectors that refer to the other shortcut are off; checkpoints are selected by validation
#     worst-group accuracy of the targeted shortcut (bg_wga or co_wga).
# The reported model is the stage with the highest selector value, at its best epoch
# ($CKPT_ROOT/{bg_only,coobj_only}/seed$S/run_summary.json).
#
# Usage: bash configs/single_target.sh [SEED ...]    (default: 0 1 2)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0 1 2)

ST_ARGS=(
  --backbone resnet50
  --weights IMAGENET1K_V1
  --batch_size 128
  --weight_decay 0.0001
  --momentum 0.9
  --amp
  --freeze_backbone_epochs 0
  --warmup_epochs 10
  --clip_grad_norm 1.0
  --anchors_per_step 48
  --steer_chunk_size 128
  --anchor_strategy ranked_bucket
  --bg_top_pct 40
  --co_top_pct 40
  --three_top_pct 40
  --bg_bucket_min_count 0
  --co_bucket_min_count 0
  --three_bucket_min_count 0
  --bg_bucket_min_frac_of_max 0.1
  --co_bucket_min_frac_of_max 0.1
  --three_bucket_min_frac_of_max 0.0
  --score_band_count 1
  --score_band_expand_ratio 1.0
  --probe_per_bucket 24
  --lambda_three 0.0
  --bg_penalty_mode ref_frac_hinge
  --co_penalty_mode ref_frac_hinge
  --three_penalty_mode zero_l2
  --tau_bg 1.0
  --tau_co 0.5
  --tau_three 0.5
  --three_ref_frac 1.0
  --bg_normalizer none
  --co_normalizer none
  --three_normalizer none
  --normalizer_eps 1e-06
  --anchor_mix_three 0.0
  --lam_obj 0.0
  --lam_obj_guard 0.03
  --obj_guard_floor 6.0
  --lam_obj_aligned 0.0
  --lam_obj_aligned_guard 0.0
  --obj_aligned_floor 0.0
  --lam_obj_ctx_guard 0.0
  --obj_ctx_margin 0.0
  --lam_obj_ctx_reward 0.0
  --lam_obj_conflict_guard 0.0
  --obj_conflict_floor 0.0
  --lam_obj_conflict_rel_guard 0.0
  --obj_conflict_rel_rho 0.0
  --lam_ref_margin_guard 0.0
  --ref_margin_slack 0.0
  --min_obj_mask_frac 0.09
  --probe_every 2
  --save_every 0
)

BG_ONLY_ARGS=(
  --early_stop_metric bg --checkpoint_selector bg_wga
  --lambda_bg 0.002 --lambda_co 0.0 --anchor_mix_bg 1.0 --anchor_mix_co 0.0
  --lam_bg_main 0.00075 --lam_co_main 0.0 --min_obj_effect_bg_anchor 4.0
  --exclusive_interaction bg --bg_delta_mode co_present --co_delta_mode bg_fixed
)
BG_ONLY_AUDIT=(--interactions bg --bg_delta_mode co_present --co_delta_mode bg_fixed)
BG_ONLY_STAGES=("--epochs 100 --lr 4e-05 --bg_ref_frac 0.9 --co_ref_frac 1.0"
                "--epochs 90 --lr 3e-05 --bg_ref_frac 0.7 --co_ref_frac 1.0"
                "--epochs 90 --lr 2e-05 --bg_ref_frac 0.5 --co_ref_frac 1.0")

COOBJ_ONLY_ARGS=(
  --early_stop_metric co_occur_obj --checkpoint_selector co_wga
  --lambda_bg 0.0 --lambda_co 0.0035 --anchor_mix_bg 0.0 --anchor_mix_co 1.0
  --lam_bg_main 0.0 --lam_co_main 0.001 --min_obj_effect_bg_anchor 0.0
  --exclusive_interaction co --bg_delta_mode co_absent --co_delta_mode bg_fixed
)
COOBJ_ONLY_AUDIT=(--interactions co_occur_obj --bg_delta_mode co_absent --co_delta_mode bg_fixed)
COOBJ_ONLY_STAGES=("--epochs 100 --lr 4e-05 --bg_ref_frac 1.0 --co_ref_frac 0.9"
                   "--epochs 90 --lr 3e-05 --bg_ref_frac 1.0 --co_ref_frac 0.7"
                   "--epochs 90 --lr 2e-05 --bg_ref_frac 1.0 --co_ref_frac 0.5")

# Data-loading workers and audit batch size as used per seed (seed 0: 8 / 8; seeds 1-2: 24 / 16).
workers_for() { if [ "$1" = 0 ]; then echo 8; else echo 24; fi; }
audit_batch_for() { if [ "$1" = 0 ]; then echo 8; else echo 16; fi; }

# single_target NAME SEED ; uses ${NAME}_ARGS, ${NAME}_AUDIT, ${NAME}_STAGES
single_target() {
  local name=$1 S=$2 dir
  dir=$(echo "$name" | tr 'A-Z' 'a-z')
  local -n args=${name}_ARGS audit_args=${name}_AUDIT stages=${name}_STAGES
  local R=$CKPT_ROOT/$dir/seed$S A=$OUT_ROOT/audits/$dir/seed$S
  local ckpt=$CKPT_ROOT/is/seed$S/erm20/last.pth k
  for k in 1 2 3; do
    run "$A/audit$k.log" "$PY" -u eval/audit_interactions.py \
      --data_root "$DATA_ROOT" --checkpoint "$ckpt" --backbone resnet50 --weights IMAGENET1K_V1 --split train \
      "${audit_args[@]}" --output_dir "$A/audit$k" --batch_size "$(audit_batch_for "$S")" --amp
    # shellcheck disable=SC2206
    local stage_args=(${stages[$((k - 1))]})
    run "$R/stage$k.log" "$PY" -u train/train_is_single_target.py "${ST_ARGS[@]}" "${args[@]}" "${stage_args[@]}" \
      --num_workers "$(workers_for "$S")" --seed "$S" \
      --load_checkpoint "$ckpt" --scores_csv "$A/audit$k/audit_scores.csv" --output_dir "$R/stage$k"
    ckpt=$R/stage$k/best.pth
  done
  select_best_stage "$R" stage1 stage2 stage3
}

for S in "${SEEDS[@]}"; do
  erm20_and_audit "$S"
  if [ "$S" != 0 ]; then export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1; fi
  single_target BG_ONLY "$S"
  single_target COOBJ_ONLY "$S"
done
