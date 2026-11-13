#!/usr/bin/env bash
# DINOv2 ViT-B/14 (timm vit_base_patch14_dinov2), seeds 1 / 21 / 42: ERM (30 epochs, head-only for the first 5 epochs,
# backbone learning rate x0.1, training augmentation), anchor-bank audit of that ERM model on the training split, and the
# 20-epoch IS fine-tune (batch 64, 24 anchors per step, tau 0.05). ERM reads grodino/waterbirds with datasets.load_dataset;
# the audit and IS read the Waterbirds manifest ($DATA_ROOT/waterbirds_manifest; see README.md).
# Checkpoints: $CKPT_ROOT/dinov2_is/seed<s>/checkpoints/steered_finetuned_best.pt (best validation WGA) and
# steered_finetuned_best_combo.pt.
set -euo pipefail
source "$(dirname "$0")/env.sh"

# seed 1
$PY $PKG/train/train_erm.py --output_dir $CKPT_ROOT/dinov2_erm/seed1 --backbone vit_base_patch14_dinov2 \
    --epochs 30 --lr 0.0001 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --label_smoothing 0.1 \
    --clip_grad_norm 1.0 --freeze_backbone_epochs 5 --backbone_lr_mult 0.1 --seed 1 --train_aug \
    --amp
mkdir -p $OUT_ROOT/anchor_banks/dinov2_seed1
# writes the anchor scores provided as lists/anchor_scores_dinov2_seed1.csv
$PY $PKG/data/audit_interaction.py --ckpt $CKPT_ROOT/dinov2_erm/seed1/checkpoints/erm_best.pt \
    --backbone vit_base_patch14_dinov2 --split train --anchors 8000 --donor_per_anchor 3 --use_margin \
    --skip_general --save_anchors $OUT_ROOT/anchor_banks/dinov2_seed1/train_masked_anchor_set.json \
    --scores_csv $OUT_ROOT/anchor_banks/dinov2_seed1/train_opposite_masked.csv --stats_out $OUT_ROOT/anchor_banks/dinov2_seed1/train_opposite_masked.json
$PY $PKG/train/train_is.py --output_dir $CKPT_ROOT/dinov2_is/seed1 --load_checkpoint $CKPT_ROOT/dinov2_erm/seed1/checkpoints/erm_best.pt \
    --backbone vit_base_patch14_dinov2 --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 64 --image_size 224 \
    --num_workers 8 --mode suppress --lambda_d2 0.05 --tau 0.05 --tau_floor_mult 0.01 --anchors_per_step 24 \
    --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_dinov2_seed1.csv --top_pct 55 \
    --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 1 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors

# seed 21
$PY $PKG/train/train_erm.py --output_dir $CKPT_ROOT/dinov2_erm/seed21 --backbone vit_base_patch14_dinov2 \
    --epochs 30 --lr 0.0001 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --label_smoothing 0.1 \
    --clip_grad_norm 1.0 --freeze_backbone_epochs 5 --backbone_lr_mult 0.1 --seed 21 --train_aug \
    --amp
mkdir -p $OUT_ROOT/anchor_banks/dinov2_seed21
# writes the anchor scores provided as lists/anchor_scores_dinov2_seed21.csv
$PY $PKG/data/audit_interaction.py --ckpt $CKPT_ROOT/dinov2_erm/seed21/checkpoints/erm_best.pt \
    --backbone vit_base_patch14_dinov2 --split train --anchors 8000 --donor_per_anchor 3 --use_margin \
    --skip_general --save_anchors $OUT_ROOT/anchor_banks/dinov2_seed21/train_masked_anchor_set.json \
    --scores_csv $OUT_ROOT/anchor_banks/dinov2_seed21/train_opposite_masked.csv --stats_out $OUT_ROOT/anchor_banks/dinov2_seed21/train_opposite_masked.json
$PY $PKG/train/train_is.py --output_dir $CKPT_ROOT/dinov2_is/seed21 --load_checkpoint $CKPT_ROOT/dinov2_erm/seed21/checkpoints/erm_best.pt \
    --backbone vit_base_patch14_dinov2 --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 64 --image_size 224 \
    --num_workers 8 --mode suppress --lambda_d2 0.05 --tau 0.05 --tau_floor_mult 0.01 --anchors_per_step 24 \
    --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_dinov2_seed21.csv --top_pct 55 \
    --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 21 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors

# seed 42
$PY $PKG/train/train_erm.py --output_dir $CKPT_ROOT/dinov2_erm/seed42 --backbone vit_base_patch14_dinov2 \
    --epochs 30 --lr 0.0001 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --label_smoothing 0.1 \
    --clip_grad_norm 1.0 --freeze_backbone_epochs 5 --backbone_lr_mult 0.1 --seed 42 --train_aug \
    --amp
mkdir -p $OUT_ROOT/anchor_banks/dinov2_seed42
# writes the anchor scores provided as lists/anchor_scores_dinov2_seed42.csv
$PY $PKG/data/audit_interaction.py --ckpt $CKPT_ROOT/dinov2_erm/seed42/checkpoints/erm_best.pt \
    --backbone vit_base_patch14_dinov2 --split train --anchors 8000 --donor_per_anchor 3 --use_margin \
    --skip_general --save_anchors $OUT_ROOT/anchor_banks/dinov2_seed42/train_masked_anchor_set.json \
    --scores_csv $OUT_ROOT/anchor_banks/dinov2_seed42/train_opposite_masked.csv --stats_out $OUT_ROOT/anchor_banks/dinov2_seed42/train_opposite_masked.json
$PY $PKG/train/train_is.py --output_dir $CKPT_ROOT/dinov2_is/seed42 --load_checkpoint $CKPT_ROOT/dinov2_erm/seed42/checkpoints/erm_best.pt \
    --backbone vit_base_patch14_dinov2 --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 64 --image_size 224 \
    --num_workers 8 --mode suppress --lambda_d2 0.05 --tau 0.05 --tau_floor_mult 0.01 --anchors_per_step 24 \
    --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_dinov2_seed42.csv --top_pct 55 \
    --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 42 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors

