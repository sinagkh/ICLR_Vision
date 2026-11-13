#!/usr/bin/env bash
# ExMap, ResNet-50, seeds 1 / 21 / 42 (paper Table 13). Each seed trains its own ERM model (30 epochs, training
# augmentation; $CKPT_ROOT/exmap_erm/seed<s>/checkpoints/erm_best.pt, best validation WGA). train/train_exmap.py then
# computes LRP heatmaps on the validation split, clusters each class into pseudo-groups and retrains the last layer on
# the frozen ERM features. Test results: $CKPT_ROOT/exmap/seed<s>/train_summary.json. Needs zennit (requirements.txt).
set -euo pipefail
source "$(dirname "$0")/env.sh"

for seed in 1 21 42; do
  $PY $PKG/train/train_erm.py --backbone resnet50 --batch_size 128 --epochs 30 --lr 0.0002 --wd 0.0005 --optimizer adamw \
    --momentum 0.9 --image_size 224 --num_workers 8 --label_smoothing 0.1 --clip_grad_norm 1.0 --freeze_backbone_epochs 0 \
    --backbone_lr_mult 1.0 --output_dir $CKPT_ROOT/exmap_erm/seed$seed --seed $seed --train_aug --amp
  $PY $PKG/train/train_exmap.py --backbone resnet50 --erm_checkpoint $CKPT_ROOT/exmap_erm/seed$seed/checkpoints/erm_best.pt \
    --output_dir $CKPT_ROOT/exmap/seed$seed --batch_size 128 --cam_batch_size 8 --dfr_tune_retrains 5 --dfr_eval_retrains 20 \
    --dfr_outer_repeats 5 --image_size 224 --num_workers 8 --downsize 112 --clusters_class_0 2 --clusters_class_1 2 \
    --seed $seed --auto --amp
done
