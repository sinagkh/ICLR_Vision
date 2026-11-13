#!/usr/bin/env bash
# Shared settings and helpers, sourced by every experiment config in this folder.
#
# Required environment variables (absolute paths):
#   DATA_ROOT  dataset root produced by the data/ scripts (contains urbancars/bg-0.95_co_occur_obj-0.95/train, ...)
#   OUT_ROOT   audits, stress-test outputs and scores are written here
# Optional:
#   CKPT_ROOT  training runs (checkpoints, history.jsonl, logs); default $OUT_ROOT/runs
#   PY         python executable; default python3
#   DRY=1      print the commands instead of running them
#
# All commands run from the urbancars/ folder of this package, with that folder on PYTHONPATH (for common.py).

set -euo pipefail

PKG_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DATA_ROOT:?set DATA_ROOT to the generated dataset root}"
: "${OUT_ROOT:?set OUT_ROOT to an output folder}"
CKPT_ROOT="${CKPT_ROOT:-$OUT_ROOT/runs}"
PY="${PY:-python3}"
DRY="${DRY:-0}"
export DATA_ROOT OUT_ROOT CKPT_ROOT
export PYTHONPATH="$PKG_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$PKG_ROOT"

# run LOGFILE CMD...: run a command, keeping a copy of its output in LOGFILE.
run() {
  local log=$1; shift
  if [ "$DRY" = 1 ]; then printf '%q ' "$@"; printf '\n'; return 0; fi
  mkdir -p "$(dirname "$log")"
  "$@" 2>&1 | tee "$log"
}

# Interaction audit on the training split. This is the audit run before every steering stage of the
# joint (BG + CoObj) runs: it scores every training image under the counterfactual corners of the
# checkpoint that the next stage starts from.
# audit_train CHECKPOINT OUTPUT_DIR
audit_train() {
  local ckpt=$1 out=$2
  run "$out.log" "$PY" -u eval/audit_interactions.py \
    --data_root "$DATA_ROOT" \
    --checkpoint "$ckpt" \
    --backbone resnet50 \
    --weights IMAGENET1K_V1 \
    --split train \
    --interactions bg co_occur_obj three_way \
    --output_dir "$out" \
    --batch_size 8 \
    --amp
}

# ERM warm start (20 epochs) for one seed, followed by the audit of its last checkpoint.
# Outputs: $CKPT_ROOT/is/seed$SEED/erm20 and $OUT_ROOT/audits/is/seed$SEED/audit1. Skipped if both exist.
# erm20_and_audit SEED
erm20_and_audit() {
  local s=$1
  local R=$CKPT_ROOT/is/seed$s A=$OUT_ROOT/audits/is/seed$s
  if [ "$DRY" != 1 ] && [ -f "$R/erm20/last.pth" ] && [ -f "$A/audit1/audit_scores.csv" ]; then
    echo "seed $s: ERM warm start and its audit already exist"; return 0
  fi
  run "$R/erm20.log" "$PY" -u train/train_erm.py \
    --data_root "$DATA_ROOT" \
    --output_dir "$R/erm20" \
    --backbone resnet50 \
    --weights IMAGENET1K_V1 \
    --epochs 20 \
    --batch_size 128 \
    --lr 0.001 \
    --weight_decay 0.0001 \
    --momentum 0.9 \
    --num_workers 8 \
    --seed "$s" \
    --device cuda \
    --early_stop_metric both \
    --amp \
    --save_every 0
  audit_train "$R/erm20/last.pth" "$A/audit1"
}

# Steering arguments shared by every stage and seed of the joint IS runs (train/train_is.py).
# Per stage only --load_checkpoint, --scores_csv, --output_dir, --epochs, --lr, --bg_ref_frac,
# --co_ref_frac and --seed change.
IS_ARGS=(
  --backbone resnet50
  --weights IMAGENET1K_V1
  --batch_size 128
  --weight_decay 0.0001
  --momentum 0.9
  --num_workers 8
  --amp
  --early_stop_metric both
  --checkpoint_selector paper_slice_mean
  --freeze_backbone_epochs 0
  --warmup_epochs 10
  --clip_grad_norm 1.0
  --anchors_per_step 48
  --steer_chunk_size 128
  --anchor_strategy ranked_bucket
  --bg_top_pct 40
  --co_top_pct 40
  --three_top_pct 40
  --bg_bucket_min_count 0
  --co_bucket_min_count 0
  --three_bucket_min_count 0
  --bg_bucket_min_frac_of_max 0.1
  --co_bucket_min_frac_of_max 0.1
  --three_bucket_min_frac_of_max 0.0
  --score_band_count 1
  --score_band_expand_ratio 1.0
  --probe_per_bucket 24
  --lambda_bg 0.002
  --lambda_co 0.0035
  --lambda_three 0.0
  --bg_penalty_mode ref_frac_hinge
  --co_penalty_mode ref_frac_hinge
  --three_penalty_mode zero_l2
  --tau_bg 1.0
  --tau_co 0.5
  --tau_three 0.5
  --three_ref_frac 1.0
  --bg_normalizer none
  --co_normalizer none
  --three_normalizer none
  --normalizer_eps 1e-06
  --anchor_mix_bg 0.5
  --anchor_mix_co 0.5
  --anchor_mix_three 0.0
  --lam_obj 0.0
  --lam_obj_guard 0.03
  --obj_guard_floor 6.0
  --lam_bg_main 0.00075
  --lam_co_main 0.001
  --lam_obj_aligned 0.0
  --lam_obj_aligned_guard 0.0
  --obj_aligned_floor 0.0
  --lam_obj_ctx_guard 0.01
  --obj_ctx_margin 0.0
  --lam_obj_ctx_reward 0.01
  --lam_obj_conflict_guard 0.0
  --obj_conflict_floor 0.0
  --lam_obj_conflict_rel_guard 0.0
  --obj_conflict_rel_rho 0.0
  --lam_ref_margin_guard 0.0
  --ref_margin_slack 0.0
  --min_obj_mask_frac 0.09
  --min_obj_effect_bg_anchor 4.0
  --probe_every 2
  --save_every 0
)

# Stage schedule of the staged warm start: epochs, learning rate, and target level of the audited interaction.
IS_STAGE1=(--epochs 100 --lr 4e-05 --bg_ref_frac 0.9 --co_ref_frac 0.9)
IS_STAGE2=(--epochs 90 --lr 3e-05 --bg_ref_frac 0.7 --co_ref_frac 0.7)
IS_STAGE3=(--epochs 90 --lr 2e-05 --bg_ref_frac 0.5 --co_ref_frac 0.5)

# select_best_stage RUN_DIR STAGE...: writes RUN_DIR/run_summary.json with the stage whose validation
# selector value is highest; that stage's best checkpoint and best epoch are the reported ones.
select_best_stage() {
  local run_dir=$1; shift
  if [ "$DRY" = 1 ]; then echo "select_best_stage $run_dir $*"; return 0; fi
  RUN_DIR="$run_dir" "$PY" - "$@" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
stage_summaries = []
best_stage = None
for stage_name in sys.argv[1:]:
    data = json.loads((run_dir / stage_name / "train_summary.json").read_text())
    entry = {
        "stage_name": stage_name,
        "best_metric_name": data.get("best_metric_name"),
        "best_metric_value": data.get("best_metric_value"),
        "best_epoch": data.get("best_epoch"),
        "best_checkpoint": data.get("artifacts", {}).get("best_checkpoint"),
        "last_checkpoint": data.get("artifacts", {}).get("last_checkpoint"),
        "best_test_metrics": data.get("best_test_metrics", {}),
    }
    stage_summaries.append(entry)
    if best_stage is None or float(entry["best_metric_value"]) > float(best_stage["best_metric_value"]):
        best_stage = entry
summary = {
    "best_stage": best_stage["stage_name"],
    "best_epoch": best_stage["best_epoch"],
    "best_metric_name": best_stage["best_metric_name"],
    "best_metric_value": best_stage["best_metric_value"],
    "best_checkpoint": best_stage["best_checkpoint"],
    "best_test_metrics": best_stage["best_test_metrics"],
    "stage_summaries": stage_summaries,
}
(run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
m = best_stage["best_test_metrics"]
print(f"{run_dir}: {best_stage['stage_name']} epoch {best_stage['best_epoch']}  "
      f"ID {100 * m['test_id_acc']:.2f}  BG {100 * m['test_bg_gap']:.2f}  "
      f"CoObj {100 * m['test_co_occur_obj_gap']:.2f}  BG+CoObj {100 * m['test_both_gap']:.2f}")
PY
}
