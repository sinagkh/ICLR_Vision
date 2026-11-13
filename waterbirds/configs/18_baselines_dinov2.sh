#!/usr/bin/env bash
# Baselines, DINOv2 ViT-B/14 (timm vit_base_patch14_dinov2), seeds 1 / 21 / 42 (paper Table 11). All runs train the head
# only for the first 5 epochs and use backbone learning rate x0.1 and training augmentation.
#   Focal Loss, CVaR and Spectral Decoupling: 30 epochs, train/train_focal_cvar_sd.py
#   JTT: a 2-epoch first model; its misclassified training images get loss weight 50 in a 20-epoch second model
#        (weights normalised per batch; train/train_jtt.py)
#   GroupDRO (training group labels): 30 epochs, eta 0.1, train/train_groupdro.py
# Checkpoint selection by validation WGA; test results in each run's train_summary.json.
set -euo pipefail
source "$(dirname "$0")/env.sh"

for seed in 1 21 42; do
  for method in focal cvar sd; do
    $PY $PKG/train/train_focal_cvar_sd.py --method $method --backbone vit_base_patch14_dinov2 \
      --output_dir $CKPT_ROOT/dinov2_$method/seed$seed --batch_size 128 --epochs 30 --lr 0.0001 --wd 0.0001 \
      --label_smoothing 0.1 --image_size 224 --num_workers 24 --clip_grad_norm 1.0 --freeze_backbone_epochs 5 \
      --backbone_lr_mult 0.1 --sd_alpha 0.001 --cvar_alpha 0.2 --focal_gamma 2.0 --focal_alpha -1.0 --seed $seed \
      --train_aug --amp
  done
  $PY $PKG/train/train_jtt.py --backbone vit_base_patch14_dinov2 --output_dir $CKPT_ROOT/dinov2_jtt/seed$seed \
    --batch_size 128 --lr 0.0001 --wd 0.0001 --optimizer adamw --momentum 0.9 --scheduler cosine --label_smoothing 0.1 \
    --image_size 224 --num_workers 24 --clip_grad_norm 1.0 --freeze_backbone_epochs 5 --backbone_lr_mult 0.1 \
    --jtt_stage1_epochs 2 --jtt_stage2_epochs 20 --jtt_lambda 50.0 --seed $seed --no-jtt_expand_dataset --train_aug --amp
  $PY $PKG/train/train_groupdro.py --backbone vit_base_patch14_dinov2 --output_dir $CKPT_ROOT/dinov2_groupdro/seed$seed \
    --batch_size 128 --epochs 30 --lr 0.0001 --wd 0.0001 --label_smoothing 0.1 --image_size 224 --num_workers 24 \
    --clip_grad_norm 1.0 --freeze_backbone_epochs 5 --backbone_lr_mult 0.1 --eta 0.1 --seed $seed --train_aug --amp
done
