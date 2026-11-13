#!/usr/bin/env python3
"""Anchor bank of a seed's ERM model: training objects the ERM model classifies correctly whose Δ² of the
true-class margin (mean-colour fill) is below K; per class, the `per-class` objects with the lowest Δ².
Writes the bank CSV (rows of lists/train_instances.csv, per class by increasing Δ²) and an .npz with all Δ².
usage: build_anchor_bank.py --checkpoint ERM.pt --out BANK.csv [--K 0.5 --per-class 512]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paco.src import steering as rv  # noqa: E402
from paco.src.common import LISTS, CachedCorners, load_model, normalize_gpu, read_rows  # noqa: E402


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--per-class", type=int, default=512)
    args = ap.parse_args()
    device = torch.device("cuda")
    rows = read_rows(LISTS / "train_instances.csv"); y = np.array([int(r["class_index"]) for r in rows])
    model = load_model(args.checkpoint, device)
    ds = CachedCorners("train", list(range(len(rows))), y.tolist(), "mean_bg", seed=0, deterministic=True)
    M = {k: [] for k in ("x11", "x10", "x01", "x00")}; pred = []
    for bt in DataLoader(ds, batch_size=128, num_workers=16):
        yb = bt["y"].to(device)
        for k in M:
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(normalize_gpu(bt[k], device)).float()
            M[k].append(rv.true_class_margin(z, yb).cpu())
            if k == "x11":
                pred.append(z.argmax(1).cpu())
    m = {k: torch.cat(v).numpy() for k, v in M.items()}; pred = torch.cat(pred).numpy()
    d2 = m["x11"] - m["x10"] - m["x01"] + m["x00"]
    cand = (pred == y) & (d2 < args.K)
    keep = []
    for c in range(int(y.max()) + 1):
        idx = np.where(cand & (y == c))[0]
        keep += idx[np.argsort(d2[idx], kind="stable")][: args.per_class].tolist()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n"); w.writerow(["index", "class_index", "class_name", "image_id", "object_instance_id"])
        for i in keep:
            r = rows[i]; w.writerow([i, r["class_index"], r["class_name"], r["image_id"], r["object_instance_id"]])
    np.savez_compressed(args.out.with_suffix(".npz"), d2=d2, pred=pred, y=y, **{f"m_{k}": v for k, v in m.items()})
    print(f"candidates {int(cand.sum())}, kept {len(keep)} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
