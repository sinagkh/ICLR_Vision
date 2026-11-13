#!/usr/bin/env bash
# GroupDRO baseline (ResNet-50, 30 epochs from ImageNet weights, eta 0.05, seeds 1 / 21 / 42) and GroupDRO + IS.
# Checkpoint selection by validation WGA throughout.
set -euo pipefail
source "$(dirname "$0")/env.sh"

# 1) GroupDRO: $CKPT_ROOT/groupdro/seed<s>/checkpoints/groupdro_best.pt (also stores the final group weights "w").
for seed in 1 21 42; do
  $PY $PKG/train/train_groupdro.py --backbone resnet50 --output_dir $CKPT_ROOT/groupdro/seed$seed --batch_size 128 \
    --epochs 30 --lr 0.0002 --wd 0.0005 --label_smoothing 0.1 --image_size 224 --num_workers 24 --clip_grad_norm 1.0 \
    --freeze_backbone_epochs 0 --backbone_lr_mult 1.0 --eta 0.05 --seed $seed --train_aug --amp
done

# 2) GroupDRO + IS: 20-epoch fine-tune of the GroupDRO model with the GroupDRO loss on natural images (--groupdro_eta 0.05,
#    group weights initialised from the checkpoint) and the IS terms of 03_is_resnet50.sh (train/train_groupdro_is.py).
#    Checkpoint: $CKPT_ROOT/groupdro_is/seed<s>/checkpoints/steered_finetuned_best.pt (best validation WGA).

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_groupdro_is.py --output_dir $CKPT_ROOT/groupdro_is/seed1 \
    --load_checkpoint $CKPT_ROOT/groupdro/seed1/checkpoints/groupdro_best.pt --backbone resnet50 \
    --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress \
    --lambda_d2 0.05 --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed1.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 1 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --groupdro_eta 0.05

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_groupdro_is.py --output_dir $CKPT_ROOT/groupdro_is/seed21 \
    --load_checkpoint $CKPT_ROOT/groupdro/seed21/checkpoints/groupdro_best.pt --backbone resnet50 \
    --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress \
    --lambda_d2 0.05 --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed21.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 21 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --groupdro_eta 0.05

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_groupdro_is.py --output_dir $CKPT_ROOT/groupdro_is/seed42 \
    --load_checkpoint $CKPT_ROOT/groupdro/seed42/checkpoints/groupdro_best.pt --backbone resnet50 \
    --epochs 20 --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress \
    --lambda_d2 0.05 --tau 0.04 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed42.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 42 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --groupdro_eta 0.05
