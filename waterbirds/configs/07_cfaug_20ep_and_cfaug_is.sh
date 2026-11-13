#!/usr/bin/env bash
# CF-Aug fine-tuned for 20 epochs from the same ERM model on the same edits, without and with the IS terms
# (train/train_cfaug_is.py, ResNet-50, seeds 1 / 21 / 42). Both arms add cross-entropy on the object-present
# counterfactual corners (bird pasted on the two donor backgrounds, labelled with the bird's class; --lambda_cf_ce 1.0,
# scaled by min(1, epoch / 6)). Everything else is the IS command of 03_is_resnet50.sh:
#   cfaug_20ep     CF-Aug only:  --lambda_d2 0    --guard_weight 0
#   cfaug_20ep_is  CF-Aug + IS:  --lambda_d2 0.05 --guard_weight 0.13
# Checkpoints: $CKPT_ROOT/<arm>/seed<s>/checkpoints/steered_finetuned_best.pt (best validation WGA, used for all reported
# numbers) and steered_finetuned_last.pt (epoch 20).
set -euo pipefail
source "$(dirname "$0")/env.sh"

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug_is.py --output_dir $CKPT_ROOT/cfaug_20ep/seed1 \
    --load_checkpoint $CKPT_ROOT/erm/seed1/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed1.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 1 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug_is.py --output_dir $CKPT_ROOT/cfaug_20ep_is/seed1 \
    --load_checkpoint $CKPT_ROOT/erm/seed1/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0.05 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed1.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 1 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug_is.py --output_dir $CKPT_ROOT/cfaug_20ep/seed21 \
    --load_checkpoint $CKPT_ROOT/erm/seed21/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed21.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 21 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug_is.py --output_dir $CKPT_ROOT/cfaug_20ep_is/seed21 \
    --load_checkpoint $CKPT_ROOT/erm/seed21/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0.05 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed21.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 21 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug_is.py --output_dir $CKPT_ROOT/cfaug_20ep/seed42 \
    --load_checkpoint $CKPT_ROOT/erm/seed42/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0 \
    --tau 0.04 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed42.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 42 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug_is.py --output_dir $CKPT_ROOT/cfaug_20ep_is/seed42 \
    --load_checkpoint $CKPT_ROOT/erm/seed42/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0.05 \
    --tau 0.04 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed42.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 42 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

# CF-Aug fine-tune with the stand-alone CF-Aug trainer (train/train_cfaug.py): cross-entropy on the object-present
# corners of all sampled anchors, without the anchor gating of train/train_is.py; the IS-specific arguments are accepted
# and not used. Checkpoints: $CKPT_ROOT/cfaug_20ep_plain_trainer/seed<s>/checkpoints/cfaug_finetuned_best.pt

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug.py --output_dir $CKPT_ROOT/cfaug_20ep_plain_trainer/seed1 \
    --load_checkpoint $CKPT_ROOT/erm/seed1/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0.05 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed1.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 1 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug.py --output_dir $CKPT_ROOT/cfaug_20ep_plain_trainer/seed21 \
    --load_checkpoint $CKPT_ROOT/erm/seed21/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0.05 \
    --tau 0.05 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed21.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 21 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

$PY $PKG/train/cub_mask_adapter.py $PKG/train/train_cfaug.py --output_dir $CKPT_ROOT/cfaug_20ep_plain_trainer/seed42 \
    --load_checkpoint $CKPT_ROOT/erm/seed42/checkpoints/erm_best.pt --backbone resnet50 --epochs 20 \
    --lr 1e-05 --wd 0.0001 --batch_size 128 --image_size 224 --num_workers 8 --mode suppress --lambda_d2 0.05 \
    --tau 0.04 --anchors_per_step 48 --anchor_strategy ranked_top --scores_csv $PKG/lists/anchor_scores_resnet50_seed42.csv \
    --top_pct 55 --ranked_balance_by y --ranked_seam_l2_max -1.0 --ranked_ebg_abs_max -1.0 --ranked_eobj_min -1.0 \
    --freeze_backbone_epochs 2 --conf_min 0.0 --min_keep_per_class 1 --eps_obj -1.0 --tau_bg_min 10.0 \
    --tau_bg_fac 0.0 --tau_bg_guard_min 0.5 --tau_bg_guard_fac 0.02 --guard_weight 0.13 --lam_obj 0.6 \
    --lam_obj_floor 0.3 --ema_obj_beta 0.9 --max_box_frac 0.85 --combo_wga_tolerance 0.008 --seed 42 \
    --amp --margin_d2 --no-ranked_filter_after_top --no-ranked_on_manifold_only --two_donors --lambda_cf_ce 1.0

