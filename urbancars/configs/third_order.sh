#!/usr/bin/env bash
# Third-order experiment: replaces the final 90-epoch stage of the joint IS run with joint pairwise + third-order
# (object x background x co-object) steering. Seeds: 0 1 2. Requires configs/is_staged.sh for the same seed
# (the run starts from $CKPT_ROOT/is/seed$S/stage2/best.pth with that run's audit3).
#
# Third-order weight: lambda_three = 0.03, chosen on seed 0 by comparing 0.02, 0.03 and 0.04 on validation
# (set THIRD_ORDER_GRID=1 to also run 0.02 and 0.04 for seed 0); seeds 1 and 2 use 0.03.
# Checkpoint: best.pth by paper_slice_mean on validation (best_epoch in train_summary.json).
# Interaction audit: all 1,000 validation images, for the stage-2 start, the joint IS stage-3 model and the
# third-order model; mean |interaction| per term is in each audit_summary.json.
#
# Usage: bash configs/third_order.sh [SEED ...]      (default: 0 1 2)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0 1 2)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

THIRD_ARGS=(
  --backbone resnet50
  --weights IMAGENET1K_V1
  --epochs 90
  --scheduler_t_max 90
  --batch_size 128
  --lr 2e-5
  --weight_decay 1e-4
  --momentum 0.9
  --num_workers 8
  --amp
  --device cuda
  --checkpoint_selector paper_slice_mean
  --freeze_backbone_epochs 0
  --warmup_epochs 10
  --clip_grad_norm 1.0
  --steer_every 1
  --anchors_per_step 48
  --steer_chunk_size 128
  --anchor_strategy ranked_bucket
  --bg_top_pct 40
  --co_top_pct 40
  --three_top_pct 40
  --bg_bucket_min_frac_of_max 0.1
  --co_bucket_min_frac_of_max 0.1
  --three_bucket_min_frac_of_max 0
  --score_band_count 1
  --score_band_expand_ratio 1
  --probe_per_bucket 24
  --pairwise_contrast_mode factorial_orthogonal
  --anchor_mix_bg 0.3333333333333333
  --anchor_mix_co 0.3333333333333333
  --anchor_mix_three 0.3333333333333333
  --lambda_bg 0.002
  --lambda_co 0.0035
  --bg_penalty_mode ref_frac_hinge
  --co_penalty_mode ref_frac_hinge
  --three_penalty_mode ref_frac_hinge
  --tau_bg 1.0
  --tau_co 0.5
  --tau_three 0.5
  --bg_ref_frac 0.5
  --co_ref_frac 0.5
  --three_ref_frac 0.5
  --lam_bg_main 0.00075
  --lam_co_main 0.001
  --obj_guard_mode factorial_mean_face
  --lam_obj_guard 0.03
  --obj_guard_floor 4.0
  --obj_face_floor 4.0
  --obj_face_guard_scale 1.0
  --object_reward_mode factorial_all_context
  --lam_obj 0.01
  --lam_obj_ctx_guard 0
  --lam_obj_ctx_reward 0
  --min_obj_mask_frac 0.09
  --min_obj_effect_bg_anchor 4.0
  --probe_every 1
  --save_every 0
)

# audit_val CHECKPOINT OUTPUT_DIR: interaction audit on the full validation split.
audit_val() {
  run "$2.log" "$PY" -u eval/audit_interactions.py \
    --data_root "$DATA_ROOT" \
    --split val \
    --backbone resnet50 \
    --weights IMAGENET1K_V1 \
    --interactions bg co_occur_obj three_way \
    --max_samples 0 \
    --batch_size 64 \
    --device cuda \
    --amp \
    --corner_examples 16 \
    --neutral_fill imagenet_mean \
    --bg_delta_mode co_absent \
    --co_delta_mode bg_fixed \
    --checkpoint "$1" \
    --output_dir "$2"
}

# third_order SEED LAMBDA OUTPUT_DIR
third_order() {
  local S=$1 lam=$2 out=$3
  run "$out.log" "$PY" -u train/train_is_third_order.py --data_root "$DATA_ROOT" \
    --load_checkpoint "$CKPT_ROOT/is/seed$S/stage2/best.pth" \
    --scores_csv "$OUT_ROOT/audits/is/seed$S/audit3/audit_scores.csv" \
    --output_dir "$out" "${THIRD_ARGS[@]}" --seed "$S" --lambda_three "$lam"
}

for S in "${SEEDS[@]}"; do
  T=$CKPT_ROOT/third_order/seed$S
  V=$OUT_ROOT/audits/third_order/seed$S
  third_order "$S" 0.03 "$T"
  audit_val "$CKPT_ROOT/is/seed$S/stage2/best.pth" "$V/val_stage2_start"
  audit_val "$CKPT_ROOT/is/seed$S/stage3/best.pth" "$V/val_is"
  audit_val "$T/best.pth" "$V/val_third_order"

  if [ "$S" = 0 ] && [ "${THIRD_ORDER_GRID:-0}" = 1 ]; then
    for lam in 0.02 0.04; do
      G=$CKPT_ROOT/third_order_grid/seed0_lambda$lam
      third_order 0 "$lam" "$G"
      audit_val "$G/best.pth" "$OUT_ROOT/audits/third_order_grid/seed0_lambda$lam/val_best"
      audit_val "$G/last.pth" "$OUT_ROOT/audits/third_order_grid/seed0_lambda$lam/val_last"
    done
    audit_val "$T/last.pth" "$V/val_third_order_last"
  fi
done
