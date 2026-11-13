#!/usr/bin/env python3
"""
Official DFR (Kirichenko et al., 2023) on saved Waterbirds checkpoints.

The tuning / refitting functions and hyperparameter grids are loaded verbatim
from the official dfr_evaluate_spurious.py of
https://github.com/PolinaKirichenko/deep_feature_reweighting (not included here; pass its
path with --official_script or $DFR_OFFICIAL_SCRIPT). Only feature extraction is local:
penultimate features from our checkpoint with the evaluation transform.

Reports DFR(Val) (last layer refit on group-balanced validation data) and
DFR(Train) (refit on group-balanced training subsets, tuned on validation).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from backbone_utils import forward_with_features
from train_jtt import HFWaterbirds, build_backbone

OFFICIAL = Path(os.environ.get("DFR_OFFICIAL_SCRIPT", "dfr_evaluate_spurious.py"))
OFFICIAL_FUNCS = {
    "dfr_on_validation_tune",
    "dfr_on_validation_eval",
    "dfr_train_subset_tune",
    "dfr_train_subset_eval",
}
TRAIN_COUNTS = np.array([3498, 184, 56, 1057], dtype=np.float64)  # g = 2*y + place


def load_official_namespace() -> dict:
    """Exec the official constants and DFR functions without running the script body."""
    tree = ast.parse(OFFICIAL.read_text())
    keep = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in OFFICIAL_FUNCS:
            keep.append(node)
        elif isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id in {"C_OPTIONS", "CLASS_WEIGHT_OPTIONS", "REG"} for t in node.targets
        ):
            keep.append(node)
    ns = {"np": np, "LogisticRegression": LogisticRegression, "StandardScaler": StandardScaler}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(OFFICIAL), "exec"), ns)
    missing = OFFICIAL_FUNCS - ns.keys()
    if missing:
        raise RuntimeError(f"Official DFR functions not found: {sorted(missing)}")
    return ns


@torch.no_grad()
def extract(model, backbone: str, split: str, device, batch_size: int, num_workers: int):
    ds = HFWaterbirds(split, train_aug=False, backbone=backbone)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    feats, logits, ys, gs = [], [], [], []
    for x, y, g, _ in dl:
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            lo, fe = forward_with_features(model, backbone, x.to(device, non_blocking=True))
        feats.append(fe.float().cpu().numpy())
        logits.append(lo.float().cpu())
        ys.append(y.numpy())
        gs.append(g.numpy())
    return np.vstack(feats), torch.cat(logits), np.concatenate(ys), np.concatenate(gs)


def summarize(test_accs, test_mean_acc) -> dict:
    accs = np.asarray(test_accs, dtype=np.float64)
    return {
        "test_group_accs": accs.tolist(),
        "test_wga": float(accs.min()),
        "test_mean_acc": float(test_mean_acc),
        "test_weighted_avg": float((accs * TRAIN_COUNTS).sum() / TRAIN_COUNTS.sum()),
    }


def main() -> None:
    global OFFICIAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--label", required=True)
    ap.add_argument("--seed", type=int, required=True, help="DFR sampling seed (same for paired models)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=16)
    ap.add_argument("--official_script", type=str, default=str(OFFICIAL),
                    help="Path to dfr_evaluate_spurious.py from the official DFR repository")
    args = ap.parse_args()

    OFFICIAL = Path(args.official_script)
    ns = load_official_namespace()
    device = torch.device("cuda")
    model = build_backbone(args.backbone).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    emb, ys, gs, base = {}, {}, {}, {}
    for split, key in [("train", "train"), ("validation", "val"), ("test", "test")]:
        emb[key], logits, ys[key], gs[key] = extract(model, args.backbone, split, device, args.batch_size, args.num_workers)
        pred = logits.argmax(1).numpy()
        accs = [float((pred == ys[key])[gs[key] == g].mean()) for g in range(4)]
        base[key] = {"group_accs": accs, "wga": min(accs), "mean_acc": float((pred == ys[key]).mean())}
    base["test"]["weighted_avg"] = float((np.array(base["test"]["group_accs"]) * TRAIN_COUNTS).sum() / TRAIN_COUNTS.sum())
    print(f"[base] {args.label} test WGA={base['test']['wga']:.4f} mean={base['test']['mean_acc']:.4f}", flush=True)

    # Official defaults: balance_dfr_val=True, notrain_dfr_val=True (no training data in DFR(Val)).
    np.random.seed(args.seed)
    c, w1, w2 = ns["dfr_on_validation_tune"](emb, ys, gs, balance_val=True, add_train=False)
    test_accs, test_mean, _ = ns["dfr_on_validation_eval"](c, w1, w2, emb, ys, gs, balance_val=True, add_train=False)
    dfr_val = {"best_hypers": [c, w1, w2], **summarize(test_accs, test_mean)}
    print(f"[DFR-Val] {args.label} C={c} test WGA={dfr_val['test_wga']:.4f} mean={dfr_val['test_mean_acc']:.4f}", flush=True)

    np.random.seed(args.seed)
    c, w1, w2 = ns["dfr_train_subset_tune"](emb, ys, gs, learn_class_weights=False)
    test_accs, test_mean, _ = ns["dfr_train_subset_eval"](c, w1, w2, emb, ys, gs)
    dfr_train = {"best_hypers": [c, w1, w2], **summarize(test_accs, test_mean)}
    print(f"[DFR-Train] {args.label} C={c} test WGA={dfr_train['test_wga']:.4f} mean={dfr_train['test_mean_acc']:.4f}", flush=True)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump({"label": args.label, "checkpoint": args.checkpoint, "backbone": args.backbone,
                   "dfr_seed": args.seed, "base": base, "dfr_val": dfr_val, "dfr_train": dfr_train},
                  handle, indent=2, default=float)
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
