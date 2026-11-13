#!/usr/bin/env bash
# Full CF-Aug: ResNet-50 trained for 50 epochs from ImageNet weights on natural images plus labelled edits (for every
# image of a batch, its bird pasted with its mask on a cleansed random donor background; cross-entropy with the bird's
# class, --lambda_cf_ce 1.0). Seeds 1 / 21 / 42. Data: the Waterbirds manifest
# ($DATA_ROOT/waterbirds_manifest; see README.md).
# Checkpoint: $CKPT_ROOT/full_cfaug/seed<s>/cfaug_mixed/checkpoints/cfaug_mixed_best.pt (best validation WGA).
set -euo pipefail
source "$(dirname "$0")/env.sh"

$PY $PKG/train/train_cfaug_full.py --output_dir $CKPT_ROOT/full_cfaug/seed1/cfaug_mixed \
    --backbone resnet50 --epochs 50 --lr 0.0002 --wd 0.0005 --batch_size 128 --image_size 224 --num_workers 24 \
    --label_smoothing 0.1 --clip_grad_norm 1.0 --seed 1 --lambda_cf_ce 1.0 --donor_policy random --amp --train_aug

$PY $PKG/train/train_cfaug_full.py --output_dir $CKPT_ROOT/full_cfaug/seed21/cfaug_mixed \
    --backbone resnet50 --epochs 50 --lr 0.0002 --wd 0.0005 --batch_size 128 --image_size 224 --num_workers 24 \
    --label_smoothing 0.1 --clip_grad_norm 1.0 --seed 21 --lambda_cf_ce 1.0 --donor_policy random --amp --train_aug

$PY $PKG/train/train_cfaug_full.py --output_dir $CKPT_ROOT/full_cfaug/seed42/cfaug_mixed \
    --backbone resnet50 --epochs 50 --lr 0.0002 --wd 0.0005 --batch_size 128 --image_size 224 --num_workers 24 \
    --label_smoothing 0.1 --clip_grad_norm 1.0 --seed 42 --lambda_cf_ce 1.0 --donor_policy random --amp --train_aug

