#!/usr/bin/env bash
# Commands for the Colored MNIST runs behind results/colored_mnist/table9.csv (paper Table 9).
# Run from the folder that contains colored_mnist/.
#
# Environment (see README.md):
#   DATA_ROOT  MNIST is downloaded by torchvision to $DATA_ROOT/MNIST on first use
#   CKPT_ROOT  trained models: $CKPT_ROOT/colored_mnist/lam<λ>_seed7.pt
#   OUT_ROOT   test accuracies: $OUT_ROOT/colored_mnist/lam<λ>_seed7.json
# One run per setting, seed 7: ERM (λ = 0) and Steered (first-order color penalty, λ = 0.5, 0.7, 0.9).
# Training: 10 epochs, batch size 128, Adam with learning rate 1e-3, label-color agreement ρ = 0.95.
# Test: ρ = 0.95 (IID) and ρ = 0.05 (OOD).
set -euo pipefail
: "${DATA_ROOT:?}" "${CKPT_ROOT:?}" "${OUT_ROOT:?}"
PY=${PY:-python}

# ---- 1. Training (one GPU, under a minute per run) ----
$PY colored_mnist/train.py --lam 0 --epochs 10 --batch 128 --lr 1e-3 --train_corr 0.95 --seed 7 --out $CKPT_ROOT/colored_mnist/lam0_seed7.pt
$PY colored_mnist/train.py --lam 0.5 --epochs 10 --batch 128 --lr 1e-3 --train_corr 0.95 --seed 7 --out $CKPT_ROOT/colored_mnist/lam0.5_seed7.pt
$PY colored_mnist/train.py --lam 0.7 --epochs 10 --batch 128 --lr 1e-3 --train_corr 0.95 --seed 7 --out $CKPT_ROOT/colored_mnist/lam0.7_seed7.pt
$PY colored_mnist/train.py --lam 0.9 --epochs 10 --batch 128 --lr 1e-3 --train_corr 0.95 --seed 7 --out $CKPT_ROOT/colored_mnist/lam0.9_seed7.pt

# ---- 2. Evaluation: IID and OOD test accuracy ----
$PY colored_mnist/evaluate.py --ckpt $CKPT_ROOT/colored_mnist/lam0_seed7.pt --train_corr 0.95 --ood_corr 0.05 --test_seed 8 --out $OUT_ROOT/colored_mnist/lam0_seed7.json
$PY colored_mnist/evaluate.py --ckpt $CKPT_ROOT/colored_mnist/lam0.5_seed7.pt --train_corr 0.95 --ood_corr 0.05 --test_seed 8 --out $OUT_ROOT/colored_mnist/lam0.5_seed7.json
$PY colored_mnist/evaluate.py --ckpt $CKPT_ROOT/colored_mnist/lam0.7_seed7.pt --train_corr 0.95 --ood_corr 0.05 --test_seed 8 --out $OUT_ROOT/colored_mnist/lam0.7_seed7.json
$PY colored_mnist/evaluate.py --ckpt $CKPT_ROOT/colored_mnist/lam0.9_seed7.pt --train_corr 0.95 --ood_corr 0.05 --test_seed 8 --out $OUT_ROOT/colored_mnist/lam0.9_seed7.json

# ---- 3. Table ----
$PY colored_mnist/make_table.py results/colored_mnist
