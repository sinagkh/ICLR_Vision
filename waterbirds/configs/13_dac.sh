#!/usr/bin/env bash
# DaC (official code, https://github.com/fhn98/DaC, commit 8f41e1e9f3cf506acb09a21822323c86550dea72, unmodified; not
# included here), ResNet-50, seeds 1 / 21 / 42, with the settings of the repository README. The mask stage uses batch size 32
# instead of the README's 128 to fit a 24 GB GPU; the CAM computation runs with the model in eval mode and all thresholds
# are per image, so the masks do not depend on the batch size. Extra packages: grad-cam==1.5.2, ttach==0.0.3, kneed==0.8.5.
# The DaC model is converted to a plain torchvision ResNet-50 state dict and scored with eval_robustness.py and
# guard_analysis.py in 14_evaluation.sh.
set -euo pipefail
source "$(dirname "$0")/env.sh"
DAC_REPO=${DAC_REPO:-/path/to/DaC}
DAC_DATA=$OUT_ROOT/dac/data       # official Waterbirds folder (images + metadata.csv); DaC writes its masks here
DAC_RUNS=$OUT_ROOT/dac/runs
mkdir -p $DAC_DATA $DAC_RUNS $CKPT_ROOT/dac
for f in $DATA_ROOT/waterbird_complete95_forest2water2/*; do ln -sfn "$f" $DAC_DATA/; done

for s in 1 21 42; do
  (cd $DAC_REPO && $PY main.py --experiment ERM --dataset WaterBirds --data_path $DAC_DATA --optimizer SGD --lr 1e-3 \
    --weight_decay 1e-3 --num_epochs 100 --batch_size 128 --save_path $DAC_RUNS --seed $s)
  E=$DAC_RUNS/seed${s}_epochs100_opt-SGD_bs128_lr0.001_wd0.001_best.model
  (cd $DAC_REPO && $PY adaptive_mask.py --dataset WaterBirds --data_path $DAC_DATA --model_path $E --batch_size 32 --seed $s)
  (cd $DAC_REPO && $PY main.py --experiment DaC --dataset WaterBirds --data_path $DAC_DATA --mask_path $DAC_DATA/masks_seed$s/ \
    --model_path $E --save_path $DAC_RUNS --optimizer Adam --scheduler StepLr --step_size 5 --gamma 0.5 --lr 5e-3 \
    --weight_decay 0 --num_epochs 20 --alpha 10 --quantile 0.6 --batch_size 64 --seed $s)
  $PY -c "
import torch,sys
sd=torch.load(sys.argv[1],map_location='cpu'); sd={k[len('model.'):]:v for k,v in sd.items()}; torch.save({'model':sd},sys.argv[2])" \
    $DAC_RUNS/seed${s}_alpha10.0_lt0.6_bs64_best.model $CKPT_ROOT/dac/dac_s${s}_eval.pt
done
