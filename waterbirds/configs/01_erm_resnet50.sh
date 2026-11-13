#!/usr/bin/env bash
# ERM, ResNet-50 (ImageNet-1k V2 weights), 30 epochs, seeds 1 / 21 / 42.
# Validation worst-group accuracy (WGA) selects the checkpoint: $CKPT_ROOT/erm/seed<s>/checkpoints/erm_best.pt
# These ERM models initialise IS, IS without target, CF-Aug (20 ep), CF-Aug + IS and CDEP.
set -euo pipefail
source "$(dirname "$0")/env.sh"
for seed in 1 21 42; do
  $PY $PKG/train/erm_arrow_adapter.py --trainer-script $PKG/train/train_erm.py --arrow-cache $DATA_ROOT/hf_cache \
    --output_dir $CKPT_ROOT/erm/seed$seed --backbone resnet50 --epochs 30 --lr 0.0002 \
    --wd 0.0005 --batch_size 128 --image_size 224 --num_workers 8 \
    --label_smoothing 0.1 --clip_grad_norm 1.0 --seed $seed --amp --no-train_aug
done
