#!/usr/bin/env bash
# IS without released masks: bird masks predicted by a COCO Mask R-CNN, and masks from the ERM model's own class
# activation map refined with GrabCut. ResNet-50, seeds 1 / 21 / 42; the ERM models are those of 01_erm_resnet50.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
MR=$OUT_ROOT/maskrcnn_masks

# 1) Mask R-CNN masks (torchvision maskrcnn_resnet50_fpn_v2, COCO_V1 weights): highest-scoring COCO "bird" instance per
#    image, mask threshold 0.5; images without a bird detection have no mask.
$PY $PKG/data/masks_maskrcnn.py --output-root $MR --arrow-cache $DATA_ROOT/hf_cache --batch-size 4 --num-workers 8

for seed in 1 21 42; do
  tau=0.05; if [[ $seed == 42 ]]; then tau=0.04; fi
  A=$OUT_ROOT/anchor_banks/maskrcnn_seed$seed; mkdir -p $A
  # 2) anchor bank from the predicted masks (writes the scores provided as lists/anchor_scores_maskrcnn_masks_seed<s>.csv)
  $PY $PKG/data/audit_with_mask_provider.py --audit-script $PKG/data/audit_interaction.py \
    --arrow-cache $DATA_ROOT/hf_cache --prediction-root $MR \
    --ckpt $CKPT_ROOT/erm/seed$seed/checkpoints/erm_best.pt --backbone resnet50 --split train \
    --anchors 8000 --donor_per_anchor 3 --use_margin --skip_general \
    --save_anchors $A/train_predicted_anchor_set.json --scores_csv $A/train_predicted_opposite.csv \
    --stats_out $A/train_predicted_opposite.json
  $PY $PKG/data/record_ranked_bank.py --scores $A/train_predicted_opposite.csv --output $A/ranked_top55_bank.json --top-pct 55
  # 3) IS with the predicted masks (same arguments as 03_is_resnet50.sh)
  $PY $PKG/train/predicted_mask_adapter.py --trainer-script $PKG/train/train_is.py \
    --arrow-cache $DATA_ROOT/hf_cache --prediction-root $MR \
    --output_dir $CKPT_ROOT/predicted_mask_is/seed$seed --load_checkpoint $CKPT_ROOT/erm/seed$seed/checkpoints/erm_best.pt \
    --backbone resnet50 --epochs 20 --lr 1e-5 --wd 1e-4 --batch_size 128 \
    --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0.05 --tau "$tau" \
    --anchors_per_step 48 --anchor_strategy ranked_top \
    --scores_csv $A/train_predicted_opposite.csv --top_pct 55 \
    --ranked_balance_by y --ranked_seam_l2_max -1 --ranked_ebg_abs_max -1 \
    --ranked_eobj_min -1 --freeze_backbone_epochs 2 --conf_min 0 \
    --min_keep_per_class 1 --eps_obj -1 --tau_bg_min 10 --tau_bg_fac 0 \
    --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 \
    --lam_obj 0.6 --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 \
    --combo_wga_tolerance 0.008 --seed "$seed" --amp --margin_d2 \
    --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors
  # 4) natural test accuracy (WGA, average accuracy, per-example logits); no segmenter call at test time
  for m in erm predicted_mask_is; do
    if [[ $m == erm ]]; then ck=$CKPT_ROOT/erm/seed$seed/checkpoints/erm_best.pt; else ck=$CKPT_ROOT/predicted_mask_is/seed$seed/checkpoints/steered_finetuned_best.pt; fi
    $PY $PKG/eval/evaluate_natural.py --checkpoint $ck --arrow-cache $DATA_ROOT/hf_cache --backbone-utils-dir $PKG/train \
      --split test --output-dir $OUT_ROOT/natural/seed$seed/$m --batch-size 256 --num-workers 8
  done
done

# 5) Interaction on the common test corners with the released CUB masks (2,153 anchors, three donors each; identical
#    renders for every model), IS with predicted masks vs ERM:
$PY $PKG/eval/common_corners.py --output-dir $OUT_ROOT/common_corner_audit --seeds 1 21 42 --cpu-threads 8 \
  --num-workers 4 --anchor-batch-size 16

# 6) Masks from the ERM model's true-class CAM (7x7) refined by GrabCut (no detector, segmenter, CUB mask/box, place or
#    group label). Masks are written in the 224x224 evaluation-crop frame, one set per seed.
for seed in 1 21 42; do
  $PY $PKG/data/masks_cam_grabcut.py --checkpoint $CKPT_ROOT/erm/seed$seed/checkpoints/erm_best.pt \
    --arrow-cache $DATA_ROOT/hf_cache --waterbirds-code $PKG/train --output-root $OUT_ROOT/cam_grabcut_masks \
    --split train --seed $seed
done
