#!/usr/bin/env bash
# ERM baseline: ResNet-50, 300 epochs, lr 1e-3. Seeds: 0 1 2.
# The reported checkpoint is best.pth, selected by validation worst-group accuracy on the
# BG+CoObj-conflicting groups (--early_stop_metric both); its epoch is best_epoch in train_summary.json.
#
# Usage: bash configs/erm_baseline.sh [SEED ...]     (default: 0 1 2)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(0 1 2)

for S in "${SEEDS[@]}"; do
  R=$CKPT_ROOT/erm300/seed$S
  run "$R.log" "$PY" -u train/train_erm.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$R" \
    --backbone resnet50 \
    --weights IMAGENET1K_V1 \
    --epochs 300 \
    --batch_size 128 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --momentum 0.9 \
    --num_workers 8 \
    --seed "$S" \
    --device cuda \
    --early_stop_metric both \
    --amp \
    --save_every 0
done
