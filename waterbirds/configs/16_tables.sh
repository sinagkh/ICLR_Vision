#!/usr/bin/env bash
# Result tables: writes per_seed.csv, mean_over_seeds.csv, unseen_scenes.csv and script_reported_test_accuracy.csv to
# results/waterbirds/ (or $RES) from the outputs of 14_evaluation.sh and 15_dfr.sh and the Full CF-Aug training summary.
# Each per_seed.csv row is MODEL:SEED=<eval_robustness.py output>:<key>[=<guard_analysis.py output>:<key>]; the guard
# file gives the background main effect Δ¹ (bg_delta1).
set -euo pipefail
source "$(dirname "$0")/env.sh"
RES=${RES:-$PKG/../results/waterbirds}
E=$OUT_ROOT/eval; G=$OUT_ROOT/guards
A() { echo "$E/results_s$1_erm_s$1_is_s$1_is_without_target_s$1_full_cfaug_s$1_cfaug_20ep_plain_trainer.json"; }
B() { echo "$E/results_s$1_cfaug_20ep_s$1_cfaug_20ep_is.json"; }
D() { echo "$E/results_s$1_dfrerm_s$1_dfris.json"; }

specs=()
for s in 1 21 42; do specs+=("ERM:$s=$(A $s):s${s}_erm=$G/seed$s.json:erm"); done
for s in 1 21 42; do specs+=("IS:$s=$(A $s):s${s}_is=$G/seed$s.json:is"); done
for s in 1 21 42; do specs+=("IS without target:$s=$(A $s):s${s}_is_without_target=$G/seed$s.json:is_without_target"); done
for s in 1 21 42; do specs+=("CF-Aug (last 20 epochs):$s=$(B $s):s${s}_cfaug_20ep"); done
for s in 1 21 42; do specs+=("CF-Aug + IS (last 20 epochs):$s=$(B $s):s${s}_cfaug_20ep_is"); done
for s in 1 21 42; do specs+=("Full CF-Aug:$s=$(A $s):s${s}_full_cfaug"); done
for s in 1 21 42; do specs+=("GroupDRO:$s=$E/results_s${s}_groupdro.json:s${s}_groupdro=$G/groupdro_s$s.json:groupdro"); done
for s in 1 21 42; do specs+=("GroupDRO + IS:$s=$E/results_s${s}_groupdro_is.json:s${s}_groupdro_is"); done
for s in 1 21 42; do specs+=("CDEP:$s=$E/results_s${s}_cdep.json:s${s}_cdep=$G/cdep_s$s.json:cdep"); done
for s in 1 21 42; do specs+=("DaC:$s=$E/results_s${s}_dac.json:s${s}_dac=$G/dac_s$s.json:dac"); done
for s in 1 21 42; do specs+=("ERM + DFR (Val):$s=$(D $s):s${s}_dfrerm"); done
for s in 1 21 42; do specs+=("IS + DFR (Val):$s=$(D $s):s${s}_dfris"); done
for v in "tau_x0.5:IS (tolerance x0.5)" "tau_x2:IS (tolerance x2)" "guard_x0.5:IS (guard weights x0.5)" "guard_x2:IS (guard weights x2)"; do
  run=${v%%:*}; name=${v#*:}
  for s in 1 21 42; do
    spec="$name:$s=$E/results_s${s}_$run.json:s${s}_$run"
    if [[ $run == guard_* && $s == 1 ]]; then spec+="=$G/seed1.json:$run"; fi
    specs+=("$spec")
  done
done
specs+=("IS (placebo masks):1=$E/results_s1_placebo_mask.json:s1_placebo_mask=$G/seed1.json:placebo_mask")

$PY $PKG/eval/summarize_metrics.py "$RES" "${specs[@]}"
$PY $PKG/eval/write_tables.py "$RES"
