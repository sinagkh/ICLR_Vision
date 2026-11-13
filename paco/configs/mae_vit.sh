#!/usr/bin/env bash
# PACO with an MAE-pretrained ViT-B/16 (timm vit_base_patch16_224.mae), seeds 0, 1, 2:
#   ERM from the MAE weights (train/train_erm_vit.py: MAE fine-tuning recipe, 20 epochs, checkpoint by validation top-1),
#   anchor banks by the rule of the ResNet-50 study (data/build_anchor_bank.py), then plain fine-tuning, IS without the
#   synergy target and IS with the settings of commands.sh, and the spoof test. Writes results/paco/mae_vit.csv.
# Run from the folder that contains paco/, after step 1 of commands.sh (crop cache). Needs timm (requirements.txt).
set -euo pipefail
: "${DATA_ROOT:?}" "${CKPT_ROOT:?}" "${OUT_ROOT:?}"
PY=${PY:-python}
export PACO_BACKBONE=vit_mae
C=$CKPT_ROOT/mae_vit; O=$OUT_ROOT/mae_vit
for s in 0 1 2; do
  $PY paco/train/train_erm_vit.py --seed $s --out-dir $C/erm_seed$s
  $PY paco/data/build_anchor_bank.py --checkpoint $C/erm_seed$s/best_by_val_top1.pt --out $O/banks/anchor_bank_seed$s.csv
  for arm in ce_only notarget is; do
    $PY paco/train/train.py --arm $arm --seed $s --erm-checkpoint $C/erm_seed$s/best_by_val_top1.pt \
      --anchor-bank $O/banks/anchor_bank_seed$s.csv --out-dir $C/runs/${arm}_s$s
  done
  CK="erm_s$s=$C/erm_seed$s/best_by_val_top1.pt ce_only_s$s=$C/runs/ce_only_s$s/last.pt notarget_s$s=$C/runs/notarget_s$s/last.pt is_s$s=$C/runs/is_s$s/last.pt"
  $PY paco/eval/save_logits.py --checkpoints $CK --out-dir $O/logits
  OUT_ROOT=$O $PY paco/eval/spoof_test.py $CK
done
OUT_ROOT=$O $PY paco/eval/spoof_metrics.py
OUT_ROOT=$O $PY paco/eval/make_mae_table.py results/paco
