#!/usr/bin/env bash
# Evaluation of all ResNet-50 models (inference only). Every model is scored on identical inputs:
#   natural test set               WGA, group accuracies (average accuracy = group accuracies weighted by test group sizes)
#   object-only corruption         bird pixels blurred (radius 2/4/8), downsampled (x4/8/16) or contrast-reduced (0.2/0.1);
#                                  corrupted BG-gap = majority-group minus minority-group accuracy, mean over blur 8,
#                                  downsample x16 and contrast 0.1
#   birdless probe                 bird removed (inpainted) and conflicting donor background alone; Only-BG accuracy =
#                                  share of these birdless images classified as their background's class
# Model names are free; eval_robustness.py writes $OUT_ROOT/eval/results_<names>.json, and eval/summarize_metrics.py turns
# these files into the CSV columns of results/waterbirds/.
set -euo pipefail
source "$(dirname "$0")/env.sh"
C=$CKPT_ROOT; B=checkpoints/steered_finetuned_best.pt

# 1) Fixed birdless-probe renders (one donor per test image, seeded per image; $OUT_ROOT/birdless_probe/). This is the
#    `prepare` step of eval/birdless_probe.py. The donor assignment it fixes is lists/birdless_probe_ids.json.
(cd $PKG/eval && $PY - <<'PYEOF'
import torch, birdless_probe as b
torch.set_num_threads(4); torch.set_num_interop_threads(1)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
b.DEST.mkdir(parents=True, exist_ok=False)
module = b.source_module(); base = module.HFWaterbirds('test', train_aug=False, backbone='resnet50', image_size=224)
b.prepare(module, base)
PYEOF
)

# 2) Natural, object-corruption and birdless-probe scores (eval_robustness.py), per seed and model group.
for s in 1 21 42; do
  $PY $PKG/eval/eval_robustness.py s${s}_erm=rn50:$C/erm/seed$s/checkpoints/erm_best.pt s${s}_is=rn50:$C/full_is/seed$s/$B \
    s${s}_is_without_target=rn50:$C/no_target/seed$s/$B s${s}_full_cfaug=cf:$C/full_cfaug/seed$s/cfaug_mixed/checkpoints/cfaug_mixed_best.pt \
    s${s}_cfaug_20ep_plain_trainer=rn50:$C/cfaug_20ep_plain_trainer/seed$s/checkpoints/cfaug_finetuned_best.pt
  $PY $PKG/eval/eval_robustness.py s${s}_cfaug_20ep=rn50:$C/cfaug_20ep/seed$s/$B s${s}_cfaug_20ep_is=rn50:$C/cfaug_20ep_is/seed$s/$B
  $PY $PKG/eval/eval_robustness.py s${s}_groupdro=rn50:$C/groupdro/seed$s/checkpoints/groupdro_best.pt
  $PY $PKG/eval/eval_robustness.py s${s}_groupdro_is=rn50:$C/groupdro_is/seed$s/$B
  $PY $PKG/eval/eval_robustness.py s${s}_cdep=rn50:$C/cdep/seed$s/$B
  $PY $PKG/eval/eval_robustness.py s${s}_dac=rn50:$C/dac/dac_s${s}_eval.pt
  for v in tau_x0.5 tau_x2 guard_x0.5 guard_x2; do
    $PY $PKG/eval/eval_robustness.py s${s}_$v=rn50:$C/$v/seed$s/$B
  done
done
$PY $PKG/eval/eval_robustness.py s1_placebo_mask=rn50:$C/placebo_mask/seed1/$B

# 3) Background main effect Δ¹ (e_bg_alone = m10 - m11 on the bird-on-own-background test images) and the other guard
#    quantities, on the same probe renders.
mkdir -p $OUT_ROOT/guards
for s in 1 21 42; do
  extra=""; if [[ $s == 1 ]]; then extra="guard_x0.5=rn50:$C/guard_x0.5/seed1/$B guard_x2=rn50:$C/guard_x2/seed1/$B placebo_mask=rn50:$C/placebo_mask/seed1/$B"; fi
  $PY $PKG/eval/guard_analysis.py $OUT_ROOT/guards/seed$s.json erm=rn50:$C/erm/seed$s/checkpoints/erm_best.pt \
    is=rn50:$C/full_is/seed$s/$B is_without_target=rn50:$C/no_target/seed$s/$B \
    cfaug_20ep_plain_trainer=rn50:$C/cfaug_20ep_plain_trainer/seed$s/checkpoints/cfaug_finetuned_best.pt $extra
  $PY $PKG/eval/guard_analysis.py $OUT_ROOT/guards/cdep_s$s.json cdep=rn50:$C/cdep/seed$s/$B
  $PY $PKG/eval/guard_analysis.py $OUT_ROOT/guards/dac_s$s.json dac=rn50:$C/dac/dac_s${s}_eval.pt
  $PY $PKG/eval/guard_analysis.py $OUT_ROOT/guards/groupdro_s$s.json groupdro=rn50:$C/groupdro/seed$s/checkpoints/groupdro_best.pt
done

# 4) Object-only corruption at all strengths for ERM, IS, IS without target (seed 1) and Full CF-Aug
#    (blur 2/4/8/16, downsample x4/8/16/32, contrast 0.4/0.2/0.1/0.05) -> $OUT_ROOT/degradation/results.json
mkdir -p $OUT_ROOT/degradation
$PY $PKG/eval/degradation.py

# 5) Unseen scenes (Places365 validation scenes of 28 categories not used by Waterbirds; see README.md).
mkdir -p $OUT_ROOT/unseen_scenes
for s in 1 21 42; do
  $PY $PKG/eval/unseen_scenes.py $s $OUT_ROOT/unseen_scenes/results_s$s.json erm=rn50:$C/erm/seed$s/checkpoints/erm_best.pt \
    is=rn50:$C/full_is/seed$s/$B is_without_target=rn50:$C/no_target/seed$s/$B \
    cfaug_20ep_plain_trainer=rn50:$C/cfaug_20ep_plain_trainer/seed$s/checkpoints/cfaug_finetuned_best.pt \
    full_cfaug=cf:$C/full_cfaug/seed$s/cfaug_mixed/checkpoints/cfaug_mixed_best.pt \
    groupdro=rn50:$C/groupdro/seed$s/checkpoints/groupdro_best.pt
done
