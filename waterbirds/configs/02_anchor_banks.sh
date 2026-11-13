#!/usr/bin/env bash
# Anchor banks (ResNet-50). For each seed, a 30-epoch ERM model is audited on the training split: every
# training image with a bird mask is an anchor, three opposite-background donors per anchor, margin-based Δ².
# The resulting scores CSVs are lists/anchor_scores_resnet50_seed{1,21,42}.csv; every ResNet-50 fine-tuning run of a seed
# (IS, IS variants, CF-Aug 20 ep, CF-Aug + IS, GroupDRO + IS, CDEP) reads the same CSV. The trainer keeps the top 55% of
# anchors by |Δ²| (all quality filters disabled) and samples them class-balanced (--anchor_strategy ranked_top --top_pct 55
# --ranked_balance_by y). data/record_ranked_bank.py writes the selected membership for inspection.
#
# The audit reads the Waterbirds manifest ($DATA_ROOT/waterbirds_manifest; see README.md). The ERM models audited here were trained with train/train_erm.py (same settings as 01_erm_resnet50.sh, images
# loaded with datasets.load_dataset("grodino/waterbirds")).
set -euo pipefail
source "$(dirname "$0")/env.sh"
for seed in 1 21 42; do
  $PY $PKG/train/train_erm.py --output_dir $CKPT_ROOT/erm_anchor_audit/seed$seed --backbone resnet50 --epochs 30 \
    --lr 0.0002 --wd 0.0005 --batch_size 128 --image_size 224 --num_workers 8 --label_smoothing 0.1 \
    --clip_grad_norm 1.0 --seed $seed --amp
  A=$OUT_ROOT/anchor_banks/resnet50_seed$seed; mkdir -p $A
  $PY $PKG/data/audit_interaction.py --ckpt $CKPT_ROOT/erm_anchor_audit/seed$seed/checkpoints/erm_best.pt \
    --backbone resnet50 --split train --anchors 8000 --donor_per_anchor 3 --use_margin --skip_general \
    --save_anchors $A/train_masked_anchor_set.json --scores_csv $A/train_opposite_masked.csv \
    --stats_out $A/train_opposite_masked.json
  $PY $PKG/data/record_ranked_bank.py --scores $A/train_opposite_masked.csv --output $A/ranked_top55_bank.json --top-pct 55
done
# The same audit reads the Arrow images and the released CUB masks when run through the mask-provider wrapper:
#   $PY $PKG/data/audit_with_mask_provider.py --audit-script $PKG/data/audit_interaction.py \
#     --arrow-cache $DATA_ROOT/hf_cache --prediction-root none --mask-provider ground_truth \
#     --official-waterbirds-root $DATA_ROOT/waterbird_complete95_forest2water2 --segmentation-root $DATA_ROOT/segmentations \
#     <audit arguments as above>
