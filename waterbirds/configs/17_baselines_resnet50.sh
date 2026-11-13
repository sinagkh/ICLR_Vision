#!/usr/bin/env bash
# Baselines without training group labels, ResNet-50 (ImageNet-1k V2 weights), seeds 1 / 21 / 42 (paper Table 10):
#   Focal Loss, CVaR and Spectral Decoupling: 30 epochs, train/train_focal_cvar_sd.py
#   JTT: a 1-epoch first model; its misclassified training images get loss weight 100 in a 30-epoch second model
#        (weights normalised per batch; train/train_jtt.py)
# Checkpoint selection by validation WGA; test results in each run's train_summary.json. GroupDRO: 09_groupdro_and_groupdro_is.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"

for seed in 1 21 42; do
  for method in focal cvar sd; do
    $PY $PKG/train/train_focal_cvar_sd.py --method $method --backbone resnet50 --output_dir $CKPT_ROOT/$method/seed$seed \
      --batch_size 128 --epochs 30 --lr 0.0002 --wd 0.0005 --label_smoothing 0.1 --image_size 224 --num_workers 24 \
      --clip_grad_norm 1.0 --freeze_backbone_epochs 0 --backbone_lr_mult 1.0 --sd_alpha 0.001 --cvar_alpha 0.2 \
      --focal_gamma 2.0 --focal_alpha -1.0 --seed $seed --train_aug --amp
  done
  $PY $PKG/train/train_jtt.py --backbone resnet50 --output_dir $CKPT_ROOT/jtt/seed$seed --batch_size 128 --lr 0.0002 \
    --wd 0.0005 --label_smoothing 0.1 --image_size 224 --num_workers 24 --clip_grad_norm 1.0 --freeze_backbone_epochs 0 \
    --backbone_lr_mult 1.0 --jtt_stage1_epochs 1 --jtt_stage2_epochs 30 --jtt_lambda 100.0 --seed $seed \
    --no-jtt_expand_dataset --train_aug --amp
done
