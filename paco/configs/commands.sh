#!/usr/bin/env bash
# Commands for every PACO run and evaluation behind results/paco. Run from the folder that contains paco/.
#
# Environment (see README.md):
#   COCO_ROOT  folder with the COCO 2017 images (train2017/, val2017/)
#   PACO_ROOT  folder with paco_lvis_v1_train.json, paco_lvis_v1_val.json, paco_lvis_v1_test.json
#   DATA_ROOT  the crop cache is written to $DATA_ROOT/paco_cache/<split>
#   CKPT_ROOT  ERM starting checkpoints: $CKPT_ROOT/erm_seed{0,1,2}/best_by_val_top1.pt
#   OUT_ROOT   training runs, logits, spoof scores and metric files
# Every training run takes the defaults of train/train.py except the flags shown; configs/run_settings.csv lists the
# resulting hyperparameters of each run. All results use the final (epoch-12) checkpoint, last.pt.
set -euo pipefail
: "${COCO_ROOT:?}" "${PACO_ROOT:?}" "${DATA_ROOT:?}" "${CKPT_ROOT:?}" "${OUT_ROOT:?}"
PY=${PY:-python}

# ---- 1. Crop cache (CPU) ----
$PY paco/data/build_cache.py --manifest paco/lists/train_instances.csv --out-dir $DATA_ROOT/paco_cache/train
$PY paco/data/build_cache.py --manifest paco/lists/val_instances.csv --out-dir $DATA_ROOT/paco_cache/val
$PY paco/data/build_cache.py --manifest paco/lists/test_instances.csv --out-dir $DATA_ROOT/paco_cache/test
$PY paco/data/build_cache.py --manifest paco/lists/part_absent.csv --out-dir $DATA_ROOT/paco_cache/part_absent
$PY paco/data/build_cache.py --manifest paco/lists/unrelated.csv --out-dir $DATA_ROOT/paco_cache/unrelated
$PY paco/data/build_cache.py --manifest paco/lists/lookalike.csv --out-dir $DATA_ROOT/paco_cache/lookalike

# ---- 2. Training (one GPU; 12 epochs from the seed-matched ERM checkpoint) ----
# Plain fine-tuning
$PY paco/train/train.py --arm ce_only --seed 0 --out-dir $OUT_ROOT/runs/ce_only_s0
$PY paco/train/train.py --arm ce_only --seed 1 --out-dir $OUT_ROOT/runs/ce_only_s1
$PY paco/train/train.py --arm ce_only --seed 2 --out-dir $OUT_ROOT/runs/ce_only_s2
# PartDrop
$PY paco/train/train.py --arm partdrop_ft --seed 0 --out-dir $OUT_ROOT/runs/partdrop_ft_s0
$PY paco/train/train.py --arm partdrop_ft --seed 1 --out-dir $OUT_ROOT/runs/partdrop_ft_s1
$PY paco/train/train.py --arm partdrop_ft --seed 2 --out-dir $OUT_ROOT/runs/partdrop_ft_s2
# Outlier exposure
$PY paco/train/train.py --arm oe_ft --seed 0 --out-dir $OUT_ROOT/runs/oe_ft_s0
$PY paco/train/train.py --arm oe_ft --seed 1 --out-dir $OUT_ROOT/runs/oe_ft_s1
$PY paco/train/train.py --arm oe_ft --seed 2 --out-dir $OUT_ROOT/runs/oe_ft_s2
# IS without the synergy target
$PY paco/train/train.py --arm notarget --seed 0 --out-dir $OUT_ROOT/runs/notarget_s0
$PY paco/train/train.py --arm notarget --seed 1 --out-dir $OUT_ROOT/runs/notarget_s1
$PY paco/train/train.py --arm notarget --seed 2 --out-dir $OUT_ROOT/runs/notarget_s2
# IS with the sign reversed
$PY paco/train/train.py --arm redund --seed 0 --out-dir $OUT_ROOT/runs/redund_s0
$PY paco/train/train.py --arm redund --seed 1 --out-dir $OUT_ROOT/runs/redund_s1
$PY paco/train/train.py --arm redund --seed 2 --out-dir $OUT_ROOT/runs/redund_s2
# IS with synergy on random regions
$PY paco/train/train.py --arm randmask --seed 0 --out-dir $OUT_ROOT/runs/randmask_s0
$PY paco/train/train.py --arm randmask --seed 1 --out-dir $OUT_ROOT/runs/randmask_s1
$PY paco/train/train.py --arm randmask --seed 2 --out-dir $OUT_ROOT/runs/randmask_s2
# IS
$PY paco/train/train.py --arm is --seed 0 --out-dir $OUT_ROOT/runs/is_s0
$PY paco/train/train.py --arm is --seed 1 --out-dir $OUT_ROOT/runs/is_s1
$PY paco/train/train.py --arm is --seed 2 --out-dir $OUT_ROOT/runs/is_s2
# PartDrop + IS
$PY paco/train/train_compose.py --arm partdrop_is --seed 0 --out-dir $OUT_ROOT/runs/partdrop_is_s0
$PY paco/train/train_compose.py --arm partdrop_is --seed 1 --out-dir $OUT_ROOT/runs/partdrop_is_s1
$PY paco/train/train_compose.py --arm partdrop_is --seed 2 --out-dir $OUT_ROOT/runs/partdrop_is_s2
# Outlier exposure + IS
$PY paco/train/train_compose.py --arm oe_is --seed 0 --out-dir $OUT_ROOT/runs/oe_is_s0
$PY paco/train/train_compose.py --arm oe_is --seed 1 --out-dir $OUT_ROOT/runs/oe_is_s1
$PY paco/train/train_compose.py --arm oe_is --seed 2 --out-dir $OUT_ROOT/runs/oe_is_s2
# IS, K=0.25
$PY paco/train/train.py --arm is --seed 0 --K 0.25 --out-dir $OUT_ROOT/runs/K0p25_s0
$PY paco/train/train.py --arm is --seed 1 --K 0.25 --out-dir $OUT_ROOT/runs/K0p25_s1
$PY paco/train/train.py --arm is --seed 2 --K 0.25 --out-dir $OUT_ROOT/runs/K0p25_s2
# IS, K=1
$PY paco/train/train.py --arm is --seed 0 --K 1.0 --out-dir $OUT_ROOT/runs/K1_s0
$PY paco/train/train.py --arm is --seed 1 --K 1.0 --out-dir $OUT_ROOT/runs/K1_s1
$PY paco/train/train.py --arm is --seed 2 --K 1.0 --out-dir $OUT_ROOT/runs/K1_s2
# IS, K=2
$PY paco/train/train.py --arm is --seed 0 --K 2.0 --out-dir $OUT_ROOT/runs/K2_s0
$PY paco/train/train.py --arm is --seed 1 --K 2.0 --out-dir $OUT_ROOT/runs/K2_s1
$PY paco/train/train.py --arm is --seed 2 --K 2.0 --out-dir $OUT_ROOT/runs/K2_s2
# IS, guard weights x0.5
$PY paco/train/train.py --arm is --seed 0 --lambda-order 0.01 --lambda-wrong 0.005 --lambda-stab 0.0025 --out-dir $OUT_ROOT/runs/gHalf_s0
$PY paco/train/train.py --arm is --seed 1 --lambda-order 0.01 --lambda-wrong 0.005 --lambda-stab 0.0025 --out-dir $OUT_ROOT/runs/gHalf_s1
$PY paco/train/train.py --arm is --seed 2 --lambda-order 0.01 --lambda-wrong 0.005 --lambda-stab 0.0025 --out-dir $OUT_ROOT/runs/gHalf_s2
# IS, guard weights x2
$PY paco/train/train.py --arm is --seed 0 --lambda-order 0.04 --lambda-wrong 0.02 --lambda-stab 0.01 --out-dir $OUT_ROOT/runs/gDouble_s0
$PY paco/train/train.py --arm is --seed 1 --lambda-order 0.04 --lambda-wrong 0.02 --lambda-stab 0.01 --out-dir $OUT_ROOT/runs/gDouble_s1
$PY paco/train/train.py --arm is --seed 2 --lambda-order 0.04 --lambda-wrong 0.02 --lambda-stab 0.01 --out-dir $OUT_ROOT/runs/gDouble_s2

# ---- 3. Logits on the test / val / evaluation sets, and single-part spoof scores (one GPU) ----
$PY paco/eval/save_logits.py --checkpoints ce_only_s0=$OUT_ROOT/runs/ce_only_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py ce_only_s0=$OUT_ROOT/runs/ce_only_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints ce_only_s1=$OUT_ROOT/runs/ce_only_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py ce_only_s1=$OUT_ROOT/runs/ce_only_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints ce_only_s2=$OUT_ROOT/runs/ce_only_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py ce_only_s2=$OUT_ROOT/runs/ce_only_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints partdrop_ft_s0=$OUT_ROOT/runs/partdrop_ft_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py partdrop_ft_s0=$OUT_ROOT/runs/partdrop_ft_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints partdrop_ft_s1=$OUT_ROOT/runs/partdrop_ft_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py partdrop_ft_s1=$OUT_ROOT/runs/partdrop_ft_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints partdrop_ft_s2=$OUT_ROOT/runs/partdrop_ft_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py partdrop_ft_s2=$OUT_ROOT/runs/partdrop_ft_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints oe_ft_s0=$OUT_ROOT/runs/oe_ft_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py oe_ft_s0=$OUT_ROOT/runs/oe_ft_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints oe_ft_s1=$OUT_ROOT/runs/oe_ft_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py oe_ft_s1=$OUT_ROOT/runs/oe_ft_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints oe_ft_s2=$OUT_ROOT/runs/oe_ft_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py oe_ft_s2=$OUT_ROOT/runs/oe_ft_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints notarget_s0=$OUT_ROOT/runs/notarget_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py notarget_s0=$OUT_ROOT/runs/notarget_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints notarget_s1=$OUT_ROOT/runs/notarget_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py notarget_s1=$OUT_ROOT/runs/notarget_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints notarget_s2=$OUT_ROOT/runs/notarget_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py notarget_s2=$OUT_ROOT/runs/notarget_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints redund_s0=$OUT_ROOT/runs/redund_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py redund_s0=$OUT_ROOT/runs/redund_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints redund_s1=$OUT_ROOT/runs/redund_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py redund_s1=$OUT_ROOT/runs/redund_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints redund_s2=$OUT_ROOT/runs/redund_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py redund_s2=$OUT_ROOT/runs/redund_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints randmask_s0=$OUT_ROOT/runs/randmask_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py randmask_s0=$OUT_ROOT/runs/randmask_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints randmask_s1=$OUT_ROOT/runs/randmask_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py randmask_s1=$OUT_ROOT/runs/randmask_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints randmask_s2=$OUT_ROOT/runs/randmask_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py randmask_s2=$OUT_ROOT/runs/randmask_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints is_s0=$OUT_ROOT/runs/is_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py is_s0=$OUT_ROOT/runs/is_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints is_s1=$OUT_ROOT/runs/is_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py is_s1=$OUT_ROOT/runs/is_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints is_s2=$OUT_ROOT/runs/is_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py is_s2=$OUT_ROOT/runs/is_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints partdrop_is_s0=$OUT_ROOT/runs/partdrop_is_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py partdrop_is_s0=$OUT_ROOT/runs/partdrop_is_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints partdrop_is_s1=$OUT_ROOT/runs/partdrop_is_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py partdrop_is_s1=$OUT_ROOT/runs/partdrop_is_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints partdrop_is_s2=$OUT_ROOT/runs/partdrop_is_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py partdrop_is_s2=$OUT_ROOT/runs/partdrop_is_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints oe_is_s0=$OUT_ROOT/runs/oe_is_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py oe_is_s0=$OUT_ROOT/runs/oe_is_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints oe_is_s1=$OUT_ROOT/runs/oe_is_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py oe_is_s1=$OUT_ROOT/runs/oe_is_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints oe_is_s2=$OUT_ROOT/runs/oe_is_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py oe_is_s2=$OUT_ROOT/runs/oe_is_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints K0p25_s0=$OUT_ROOT/runs/K0p25_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K0p25_s0=$OUT_ROOT/runs/K0p25_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints K0p25_s1=$OUT_ROOT/runs/K0p25_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K0p25_s1=$OUT_ROOT/runs/K0p25_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints K0p25_s2=$OUT_ROOT/runs/K0p25_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K0p25_s2=$OUT_ROOT/runs/K0p25_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints K1_s0=$OUT_ROOT/runs/K1_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K1_s0=$OUT_ROOT/runs/K1_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints K1_s1=$OUT_ROOT/runs/K1_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K1_s1=$OUT_ROOT/runs/K1_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints K1_s2=$OUT_ROOT/runs/K1_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K1_s2=$OUT_ROOT/runs/K1_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints K2_s0=$OUT_ROOT/runs/K2_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K2_s0=$OUT_ROOT/runs/K2_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints K2_s1=$OUT_ROOT/runs/K2_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K2_s1=$OUT_ROOT/runs/K2_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints K2_s2=$OUT_ROOT/runs/K2_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py K2_s2=$OUT_ROOT/runs/K2_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints gHalf_s0=$OUT_ROOT/runs/gHalf_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py gHalf_s0=$OUT_ROOT/runs/gHalf_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints gHalf_s1=$OUT_ROOT/runs/gHalf_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py gHalf_s1=$OUT_ROOT/runs/gHalf_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints gHalf_s2=$OUT_ROOT/runs/gHalf_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py gHalf_s2=$OUT_ROOT/runs/gHalf_s2/last.pt
$PY paco/eval/save_logits.py --checkpoints gDouble_s0=$OUT_ROOT/runs/gDouble_s0/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py gDouble_s0=$OUT_ROOT/runs/gDouble_s0/last.pt
$PY paco/eval/save_logits.py --checkpoints gDouble_s1=$OUT_ROOT/runs/gDouble_s1/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py gDouble_s1=$OUT_ROOT/runs/gDouble_s1/last.pt
$PY paco/eval/save_logits.py --checkpoints gDouble_s2=$OUT_ROOT/runs/gDouble_s2/last.pt --out-dir $OUT_ROOT/logits
$PY paco/eval/spoof_test.py gDouble_s2=$OUT_ROOT/runs/gDouble_s2/last.pt

# ---- 4. Spoof renderings: colour-harmonized with a feathered edge (blend), and shifted (shift) ----
$PY paco/eval/spoof_renderings.py blend ce_only_s0=$OUT_ROOT/runs/ce_only_s0/last.pt
$PY paco/eval/spoof_renderings.py blend ce_only_s1=$OUT_ROOT/runs/ce_only_s1/last.pt
$PY paco/eval/spoof_renderings.py blend ce_only_s2=$OUT_ROOT/runs/ce_only_s2/last.pt
$PY paco/eval/spoof_renderings.py blend notarget_s0=$OUT_ROOT/runs/notarget_s0/last.pt
$PY paco/eval/spoof_renderings.py blend notarget_s1=$OUT_ROOT/runs/notarget_s1/last.pt
$PY paco/eval/spoof_renderings.py blend notarget_s2=$OUT_ROOT/runs/notarget_s2/last.pt
$PY paco/eval/spoof_renderings.py blend is_s0=$OUT_ROOT/runs/is_s0/last.pt
$PY paco/eval/spoof_renderings.py blend is_s1=$OUT_ROOT/runs/is_s1/last.pt
$PY paco/eval/spoof_renderings.py blend is_s2=$OUT_ROOT/runs/is_s2/last.pt
$PY paco/eval/spoof_renderings.py blend partdrop_ft_s0=$OUT_ROOT/runs/partdrop_ft_s0/last.pt
$PY paco/eval/spoof_renderings.py blend partdrop_ft_s1=$OUT_ROOT/runs/partdrop_ft_s1/last.pt
$PY paco/eval/spoof_renderings.py blend partdrop_ft_s2=$OUT_ROOT/runs/partdrop_ft_s2/last.pt
$PY paco/eval/spoof_renderings.py blend oe_ft_s0=$OUT_ROOT/runs/oe_ft_s0/last.pt
$PY paco/eval/spoof_renderings.py blend oe_ft_s1=$OUT_ROOT/runs/oe_ft_s1/last.pt
$PY paco/eval/spoof_renderings.py blend oe_ft_s2=$OUT_ROOT/runs/oe_ft_s2/last.pt
$PY paco/eval/spoof_renderings.py blend oe_is_s0=$OUT_ROOT/runs/oe_is_s0/last.pt
$PY paco/eval/spoof_renderings.py blend oe_is_s1=$OUT_ROOT/runs/oe_is_s1/last.pt
$PY paco/eval/spoof_renderings.py blend oe_is_s2=$OUT_ROOT/runs/oe_is_s2/last.pt
$PY paco/eval/spoof_renderings.py blend partdrop_is_s0=$OUT_ROOT/runs/partdrop_is_s0/last.pt
$PY paco/eval/spoof_renderings.py blend partdrop_is_s1=$OUT_ROOT/runs/partdrop_is_s1/last.pt
$PY paco/eval/spoof_renderings.py blend partdrop_is_s2=$OUT_ROOT/runs/partdrop_is_s2/last.pt
$PY paco/eval/spoof_renderings.py shift ce_only_s0=$OUT_ROOT/runs/ce_only_s0/last.pt
$PY paco/eval/spoof_renderings.py shift ce_only_s1=$OUT_ROOT/runs/ce_only_s1/last.pt
$PY paco/eval/spoof_renderings.py shift ce_only_s2=$OUT_ROOT/runs/ce_only_s2/last.pt
$PY paco/eval/spoof_renderings.py shift notarget_s0=$OUT_ROOT/runs/notarget_s0/last.pt
$PY paco/eval/spoof_renderings.py shift notarget_s1=$OUT_ROOT/runs/notarget_s1/last.pt
$PY paco/eval/spoof_renderings.py shift notarget_s2=$OUT_ROOT/runs/notarget_s2/last.pt
$PY paco/eval/spoof_renderings.py shift is_s0=$OUT_ROOT/runs/is_s0/last.pt
$PY paco/eval/spoof_renderings.py shift is_s1=$OUT_ROOT/runs/is_s1/last.pt
$PY paco/eval/spoof_renderings.py shift is_s2=$OUT_ROOT/runs/is_s2/last.pt
$PY paco/eval/spoof_renderings.py shift partdrop_ft_s0=$OUT_ROOT/runs/partdrop_ft_s0/last.pt
$PY paco/eval/spoof_renderings.py shift partdrop_ft_s1=$OUT_ROOT/runs/partdrop_ft_s1/last.pt
$PY paco/eval/spoof_renderings.py shift partdrop_ft_s2=$OUT_ROOT/runs/partdrop_ft_s2/last.pt
$PY paco/eval/spoof_renderings.py shift oe_ft_s0=$OUT_ROOT/runs/oe_ft_s0/last.pt
$PY paco/eval/spoof_renderings.py shift oe_ft_s1=$OUT_ROOT/runs/oe_ft_s1/last.pt
$PY paco/eval/spoof_renderings.py shift oe_ft_s2=$OUT_ROOT/runs/oe_ft_s2/last.pt
$PY paco/eval/spoof_renderings.py shift oe_is_s0=$OUT_ROOT/runs/oe_is_s0/last.pt
$PY paco/eval/spoof_renderings.py shift oe_is_s1=$OUT_ROOT/runs/oe_is_s1/last.pt
$PY paco/eval/spoof_renderings.py shift oe_is_s2=$OUT_ROOT/runs/oe_is_s2/last.pt
$PY paco/eval/spoof_renderings.py shift partdrop_is_s0=$OUT_ROOT/runs/partdrop_is_s0/last.pt
$PY paco/eval/spoof_renderings.py shift partdrop_is_s1=$OUT_ROOT/runs/partdrop_is_s1/last.pt
$PY paco/eval/spoof_renderings.py shift partdrop_is_s2=$OUT_ROOT/runs/partdrop_is_s2/last.pt

# ---- 5. Metrics and tables (CPU) ----
$PY paco/eval/spoof_metrics.py
$PY paco/eval/hidden_part_test.py $OUT_ROOT/hidden_part.json
$PY paco/eval/make_tables.py results/paco
