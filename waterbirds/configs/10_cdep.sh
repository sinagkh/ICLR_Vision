#!/usr/bin/env bash
# CDEP in the IS setup (train/train_cdep.py, ResNet-50, seeds 1 / 21 / 42): the IS command of 03_is_resnet50.sh with the
# IS terms off (--lambda_d2 0 --guard_weight 0) and the CDEP explanation penalty on the background (complement of the
# anchor's CUB bird mask, dilated by 2 px) with weight --lambda_cdep 10, on the same anchors and schedule.
# Checkpoints: $CKPT_ROOT/cdep/seed<s>/checkpoints/steered_finetuned_best.pt (best validation WGA).
set -euo pipefail
source "$(dirname "$0")/env.sh"

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cdep.py --output_dir $CKPT_ROOT/cdep/seed1 \
    --load_checkpoint $CKPT_ROOT/erm/seed1/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed1.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 1 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cdep 10

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cdep.py --output_dir $CKPT_ROOT/cdep/seed21 \
    --load_checkpoint $CKPT_ROOT/erm/seed21/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed21.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 21 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cdep 10

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cdep.py --output_dir $CKPT_ROOT/cdep/seed42 \
    --load_checkpoint $CKPT_ROOT/erm/seed42/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0 \
    --tau 0.04 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed42.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 42 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cdep 10

