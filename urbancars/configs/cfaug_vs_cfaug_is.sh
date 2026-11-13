#!/usr/bin/env bash
# CF-Aug vs CF-Aug + IS, 20 epochs of the first steering stage, seeds 1 2 3.
#
# Both arms start from the seed's ERM warm start ($CKPT_ROOT/is/seed$S/erm20/last.pth) and use the anchors
# of its audit (audit1), i.e. exactly the start of IS stage 1. Both run the IS stage-1 command (lr 4e-5 with a
# cosine schedule over 100 epochs; interaction-target weights ramped up over the first 10 epochs) and are
# stopped after epoch 20; the epoch-20 checkpoint is kept as epoch20.pth and the benchmark numbers are read
# from row 20 of history.jsonl.
#   labels (CF-Aug):     all steering terms off; cross-entropy with the car label on the four car-present
#                        counterfactual corners of every anchor, weight 1 (--lam_label_ce 1.0).
#   labis  (CF-Aug + IS): the full IS stage-1 objective plus the same label loss (--lam_label_ce 1.0).
# Later options on the command line override earlier ones (argparse), so the arm settings are appended to the
# IS stage-1 arguments.
#
# Then: stress test (Corrupted BG-Gap) of the epoch-20 checkpoints, and scoring.
# Seed 3 has no IS run; its ERM warm start and audit are created here with the same commands.
#
# Usage: bash configs/cfaug_vs_cfaug_is.sh [SEED ...]   (default: 1 2 3)

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"
SEEDS=("$@"); [ ${#SEEDS[@]} -gt 0 ] || SEEDS=(1 2 3)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

LABELS_ARM=(--lambda_bg 0 --lambda_co 0 --lam_bg_main 0 --lam_co_main 0 --lam_obj_guard 0 --lam_obj_ctx_guard 0
            --lam_obj_ctx_reward 0 --lam_label_ce 1.0)
LABIS_ARM=(--lam_label_ce 1.0)

# run_first_20_epochs OUTPUT_DIR CMD...: runs CMD and stops it once epoch 20 is logged and its last.pth is
# saved; copies that checkpoint to OUTPUT_DIR/epoch20.pth.
run_first_20_epochs() {
  local out=$1; shift
  if [ "$DRY" = 1 ]; then printf '%q ' "$@"; printf '   # stopped after epoch 20\n'; return 0; fi
  mkdir -p "$out"
  "$@" > "$out/run.log" 2>&1 &
  local pid=$! n i
  while kill -0 "$pid" 2>/dev/null; do
    n=$(wc -l < "$out/history.jsonl" 2>/dev/null || echo 0)
    if [ "$n" -ge 20 ]; then
      # history.jsonl is appended before last.pth is saved; wait until last.pth holds epoch 20
      for i in $(seq 1 60); do [ "$out/last.pth" -nt "$out/history.jsonl" ] && sleep 20 && break; sleep 5; done
      kill "$pid" || true; break
    fi
    sleep 15
  done
  while kill -0 "$pid" 2>/dev/null; do sleep 5; done
  cp "$out/last.pth" "$out/epoch20.pth"
  "$PY" -c "import sys, torch; print(sys.argv[1], 'epoch', torch.load(sys.argv[1], map_location='cpu', weights_only=False)['epoch'])" "$out/epoch20.pth"
}

for S in "${SEEDS[@]}"; do
  erm20_and_audit "$S"
  I=$CKPT_ROOT/is/seed$S
  C=$CKPT_ROOT/cfaug20/seed$S
  STAGE1=("${IS_ARGS[@]}" "${IS_STAGE1[@]}" --seed "$S"
          --load_checkpoint "$I/erm20/last.pth" --scores_csv "$OUT_ROOT/audits/is/seed$S/audit1/audit_scores.csv")

  run_first_20_epochs "$C/labels" "$PY" -u train/train_is.py "${STAGE1[@]}" --output_dir "$C/labels" "${LABELS_ARM[@]}"
  run_first_20_epochs "$C/labis"  "$PY" -u train/train_is.py "${STAGE1[@]}" --output_dir "$C/labis"  "${LABIS_ARM[@]}"

  run "$OUT_ROOT/stress/cfaug20_seed$S.log" "$PY" eval/stress_test.py --out "$OUT_ROOT/stress/cfaug20_seed$S.json" \
    erm="$I/erm20/last.pth" labels="$C/labels/epoch20.pth" labis="$C/labis/epoch20.pth"

  run "$OUT_ROOT/scores/cfaug20_seed$S.txt" "$PY" eval/score_cfaug.py "$C" "$OUT_ROOT/stress/cfaug20_seed$S.json"
done
