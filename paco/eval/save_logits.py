#!/usr/bin/env python3
"""Save the logits of each checkpoint on the PACO evaluation sets.

Per checkpoint NAME=path, writes <out-dir>/NAME.npz with z_<set> for the natural crops of every set in SETS and
z_test_x10 / z_test_x01 / z_test_x00 for the test crops with part B, part A, or both removed (mean-colour fill).
Top-1, the per-class 95% thresholds and the hidden-part test are computed from these files.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paco.src.common import LISTS, CachedCorners, CachedNatural, load_model, normalize_gpu, read_rows  # noqa: E402

SETS = ["test", "val", "lookalike", "part_absent", "unrelated"]


@torch.inference_mode()
def natural_logits(model, split, device):
    out = []
    for x, _, _ in DataLoader(CachedNatural(split), batch_size=256, num_workers=8):
        with torch.autocast("cuda", dtype=torch.float16):
            out.append(model(normalize_gpu(x, device)).float().cpu())
    return torch.cat(out).numpy()


@torch.inference_mode()
def corner_logits(model, device):
    rows = read_rows(LISTS / "test_instances.csv")
    ds = CachedCorners("test", list(range(len(rows))), [int(r["class_index"]) for r in rows], "mean_bg", seed=0, deterministic=True)
    out = {k: [] for k in ["x10", "x01", "x00"]}
    for bt in DataLoader(ds, batch_size=128, num_workers=16):
        for k in out:
            with torch.autocast("cuda", dtype=torch.float16):
                out[k].append(model(normalize_gpu(bt[k], device)).float().cpu())
    return {k: torch.cat(v).numpy() for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True, help="NAME=path, e.g. is_s0=$OUT_ROOT/runs/is_s0/last.pt")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    for spec in args.checkpoints:
        name, path = spec.split("=", 1)
        out = args.out_dir / f"{name}.npz"
        if out.exists():
            print("skip", name)
            continue
        model = load_model(Path(path), device)
        res = {f"z_{s}": natural_logits(model, s, device) for s in SETS}
        res.update({f"z_test_{k}": v for k, v in corner_logits(model, device).items()})
        np.savez_compressed(out, checkpoint=str(path), **res)
        print("saved", name, flush=True)


if __name__ == "__main__":
    main()
