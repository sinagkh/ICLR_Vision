#!/usr/bin/env python3
"""Record the top-55% ranked anchor-bank membership implied by an audit scores CSV."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--top-pct", type=float, default=55.0)
    args = ap.parse_args()
    with args.scores.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r.get("has_mask", 1)) == 1]
    # The IS runs disable on-manifold and all quality filters, then rank by
    # absolute margin interaction and takes Python round(N * 55%).
    rows.sort(key=lambda r: float(r["d2_abs"]), reverse=True)
    n = max(1, round(len(rows) * args.top_pct / 100.0))
    selected = rows[:n]
    payload = {
        "source_scores": str(args.scores.resolve()),
        "definition": "top 55% by d2_abs; has_mask=1; no on-manifold/seam/e_bg/e_obj filter; class-balanced sampling occurs during training",
        "base_count": len(rows),
        "selected_count": len(selected),
        "selected_by_label": {str(y): sum(int(r["y"]) == y for r in selected) for y in (0, 1)},
        "selected_indices_sorted_by_rank": [int(r["idx"]) for r in selected],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2) + "\n"
    args.output.write_text(text)
    print(json.dumps({**{k: v for k, v in payload.items() if k != "selected_indices_sorted_by_rank"},
                      "sha256": hashlib.sha256(text.encode()).hexdigest()}, indent=2))


if __name__ == "__main__": main()
