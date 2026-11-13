#!/usr/bin/env bash
# Interaction Steering (IS), ResNet-50: 20-epoch fine-tune of the seed's ERM model (01_erm_resnet50.sh) with the
# anchor bank of the same seed (lists/anchor_scores_resnet50_seed<s>.csv). Suppression tolerance tau = 0.05 for seeds 1
# and 21 and 0.04 for seed 42; all other settings are shared. Images come from the Arrow cache and bird masks from the
# released CUB segmentations (train/is_with_cub_masks.py). The first 2 epochs train the head only (backbone frozen); the
# interaction weight is lambda_d2 * min(1, epoch / 6).
# Checkpoints: $CKPT_ROOT/full_is/seed<s>/checkpoints/steered_finetuned_best.pt (best validation WGA; the IS rows
# of results/waterbirds/) and steered_finetuned_best_combo.pt.
set -euo pipefail
source "$(dirname "$0")/env.sh"

$PY $PKG/train/is_with_cub_masks.py --output_dir $CKPT_ROOT/full_is/seed1 --load_checkpoint $CKPT_ROOT/erm/seed1/checkpoints/erm_best.pt \
    --backbone resnet50 --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 \
    --mode suppress --lambda_d2 0.05 --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top \
    --scores_csv $PKG/lists/anchor_scores_resnet50_seed1.csv --top_pct 55 --ranked_balance_by y \
    --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 --freeze_backbone_epochs 2 \
    --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 \
    --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 --lam_obj_floor 0.3 --ema_obj_beta 0.9 \
    --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 1 --amp --margin_d2 --no-ranked_filter_after_top \
    --no-ranked_on_manifold_only --two_donors

$PY $PKG/train/is_with_cub_masks.py --output_dir $CKPT_ROOT/full_is/seed21 --load_checkpoint $CKPT_ROOT/erm/seed21/checkpoints/erm_best.pt \
    --backbone resnet50 --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 \
    --mode suppress --lambda_d2 0.05 --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top \
    --scores_csv $PKG/lists/anchor_scores_resnet50_seed21.csv --top_pct 55 --ranked_balance_by y \
    --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 --freeze_backbone_epochs 2 \
    --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 \
    --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 --lam_obj_floor 0.3 --ema_obj_beta 0.9 \
    --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 21 --amp --margin_d2 --no-ranked_filter_after_top \
    --no-ranked_on_manifold_only --two_donors

$PY $PKG/train/is_with_cub_masks.py --output_dir $CKPT_ROOT/full_is/seed42 --load_checkpoint $CKPT_ROOT/erm/seed42/checkpoints/erm_best.pt \
    --backbone resnet50 --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 \
    --mode suppress --lambda_d2 0.05 --tau 0.04 --anchors_per_step 48 --anchor_strategy ranked_top \
    --scores_csv $PKG/lists/anchor_scores_resnet50_seed42.csv --top_pct 55 --ranked_balance_by y \
    --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 --freeze_backbone_epochs 2 \
    --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 \
    --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 --lam_obj_floor 0.3 --ema_obj_beta 0.9 \
    --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 42 --amp --margin_d2 --no-ranked_filter_after_top \
    --no-ranked_on_manifold_only --two_donors

# The paper's ResNet-50 IS results (WGA 89.04 ± 1.01) use the same arguments with train/train_is.py on the Waterbirds
# manifest ($DATA_ROOT/waterbirds_manifest), starting from the ERM models of 02_anchor_banks.sh:
#   $PY $PKG/train/train_is.py --output_dir $CKPT_ROOT/is_manifest/seed<s> \
#     --load_checkpoint $CKPT_ROOT/erm_anchor_audit/seed<s>/checkpoints/erm_best.pt <all other arguments as above>
