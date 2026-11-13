#!/usr/bin/env python3
"""PACO trainer with the composition arms (label-based baseline + IS).

Fine-tuning arms start from the seed-matched ERM checkpoint and anchor bank (d2 = Delta^2, the second-order
difference of the true-class margin over the four part-presence corners):
  is            : IS. CE + stability + target [K - d2]_+^2 (20% warmup) + order guard + wrong-class guard
  notarget      : IS without the synergy target. As `is` with target weight 0
  redund        : IS with the sign reversed. As `is` with target [d2 + K]_+^2
  randmask      : IS with synergy on random regions. As `is` with area-matched random regions inside the object instead of parts
  partdrop_ft   : PartDrop. CE + stability + CE(object label) on the three edited corners (weight 1)
  oe_ft         : Outlier exposure. CE + stability + CE(uniform) on the three edited corners (weight 1)
  ce_only       : Plain fine-tuning. CE + stability; corners forwarded with zero loss
Composition arms (this file only):
  partdrop_is   : PartDrop + IS. The partdrop_ft label loss plus the IS terms of arm `is`
  oe_is         : Outlier exposure + IS. The oe_ft label loss plus the IS terms of arm `is`
All arms of train.py are also available here; the reported results train them with train.py.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paco.src import steering as rv  # noqa: E402
from paco.src.common import CACHE, LISTS, CachedCorners, CachedNatural, load_model, normalize_gpu, read_rows, worker_init  # noqa: E402

CORNERS = ["x11", "x10", "x01", "x00"]


def margins_true(z, y):
    return rv.true_class_margin(z, y)


@torch.inference_mode()
def validate(model, device, val_y, val_corners_loader):
    model.eval()
    zs = []
    for x, _, _ in DataLoader(CachedNatural("val"), batch_size=256, num_workers=6):
        with torch.autocast("cuda", dtype=torch.float16):
            zs.append(model(normalize_gpu(x, device)).float())
    z = torch.cat(zs)
    y = torch.tensor(val_y, device=device)
    top1 = float((z.argmax(1) == y).float().mean())
    m = {k: [] for k in CORNERS}
    for bt in val_corners_loader:
        yy = bt["y"].to(device)
        for k in CORNERS:
            with torch.autocast("cuda", dtype=torch.float16):
                m[k].append(margins_true(model(normalize_gpu(bt[k], device)).float(), yy))
    m = {k: torch.cat(v) for k, v in m.items()}
    d2 = m["x11"] - m["x10"] - m["x01"] + m["x00"]
    corner_top = (m["x11"] > torch.maximum(torch.maximum(m["x10"], m["x01"]), m["x00"])).float().mean()
    model.train()
    return {"val_top1": top1, "val_d2_meanbg": float(d2.mean()), "val_frac_d2_gt_K": float((d2 > 0.5).float().mean()), "val_corner_top": float(corner_top)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, help="partdrop_is | oe_is (see module docstring)")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--erm-checkpoint", type=Path, default=None, help="default: $CKPT_ROOT/erm_seed<seed>/best_by_val_top1.pt")
    ap.add_argument("--anchor-bank", type=Path, default=None, help="default: lists/anchor_bank_seed<seed>.csv")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--anchors", type=int, default=32)
    ap.add_argument("--K", type=float, default=0.5)
    ap.add_argument("--lambda-d2", type=float, default=0.05)
    ap.add_argument("--lambda-order", type=float, default=0.02)
    ap.add_argument("--lambda-wrong", type=float, default=0.01)
    ap.add_argument("--lambda-stab", type=float, default=0.005)
    ap.add_argument("--order-q", type=float, default=0.25)
    ap.add_argument("--warmup-frac", type=float, default=0.2)
    ap.add_argument("--label-weight", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early after this many steps (quick checks only)")
    args = ap.parse_args()
    arm = args.arm
    epochs = args.epochs or 12
    lr = args.lr or 1e-5
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    train_rows = read_rows(LISTS / "train_instances.csv")
    train_y = [int(r["class_index"]) for r in train_rows]
    val_rows = read_rows(LISTS / "val_instances.csv")
    val_y = [int(r["class_index"]) for r in val_rows]
    val_corners = DataLoader(CachedCorners("val", list(range(len(val_y))), val_y, "mean_bg", seed=0, deterministic=True), batch_size=128, num_workers=6)
    erm_path = args.erm_checkpoint or Path(os.environ.get("CKPT_ROOT", "checkpoints")) / f"erm_seed{args.seed}" / "best_by_val_top1.pt"
    bank_path = args.anchor_bank or LISTS / f"anchor_bank_seed{args.seed}.csv"
    cfg = {**vars(args), "out_dir": str(out), "epochs": epochs, "lr": lr, "erm_checkpoint": str(erm_path), "code": "train/train_compose.py + src/steering.py"}
    (out / "config.json").write_text(json.dumps(cfg, indent=1, default=str))

    model = load_model(erm_path, device)
    model.train()
    erm = load_model(erm_path, device)
    for p in erm.parameters():
        p.requires_grad_(False)
    loader = DataLoader(CachedNatural("train", train_y), batch_size=args.batch, shuffle=True, num_workers=8, persistent_workers=True, prefetch_factor=4)
    bank = read_rows(bank_path)
    bank_idx = [int(r["index"]) for r in bank]
    bank_y = [int(r["class_index"]) for r in bank]
    assert all(train_y[i] == y for i, y in zip(bank_idx, bank_y))
    anchor_ds = CachedCorners("train", bank_idx, bank_y, "mixed", seed=args.seed, random_masks=(arm == "randmask"))
    anchor_loader = DataLoader(anchor_ds, batch_size=args.anchors, shuffle=True, num_workers=16, worker_init_fn=worker_init, persistent_workers=True, prefetch_factor=4, drop_last=True)
    anchor_iter = itertools.chain.from_iterable(itertools.repeat(anchor_loader))
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda")
    total_steps = epochs * len(loader)
    warm = max(1, int(round(args.warmup_frac * total_steps)))
    best, step, t0 = -1.0, 0, time.time()
    log = open(out / "metrics_epoch.jsonl", "w")
    for epoch in range(1, epochs + 1):
        sums, n_steps = {}, 0
        for xb, yb, *_ in loader:
            y = yb.to(device)
            x = normalize_gpu(xb, device)
            with torch.autocast("cuda", dtype=torch.float16):
                z = model(x)
            z = z.float()
            task = F.cross_entropy(z, y)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                z_ref = erm(x).float()
            stab = (z - z_ref).pow(2).sum(1).mean()
            bt = next(anchor_iter)
            ya = bt["y"].to(device)
            X = torch.cat([normalize_gpu(bt[k], device) for k in CORNERS])
            with torch.autocast("cuda", dtype=torch.float16):
                zc = model(X)
            zc = zc.float()
            n = ya.numel()
            Z = {k: zc[i * n:(i + 1) * n] for i, k in enumerate(CORNERS)}
            m = {k: margins_true(Z[k], ya) for k in CORNERS}
            d2 = m["x11"] - m["x10"] - m["x01"] + m["x00"]
            order = sum(F.relu(args.order_q - (m["x11"] - m[k])).pow(2) for k in ["x10", "x01", "x00"]).mean()
            masked = Z["x11"].detach().clone()
            masked.scatter_(1, ya.view(-1, 1), float("-inf"))
            wi = masked.topk(min(5, masked.size(1) - 1), dim=1).indices
            wd2 = Z["x11"].gather(1, wi) - Z["x10"].gather(1, wi) - Z["x01"].gather(1, wi) + Z["x00"].gather(1, wi)
            wrong = F.relu(wd2).pow(2).mean(1).mean()
            w = min(1.0, (step + 1) / warm)
            parts = {"task": task, "stab": stab, "d2_mean": d2.detach().mean(), "order": order, "wrong": wrong}
            loss = task + args.lambda_stab * stab
            if arm in ("is", "randmask"):
                tgt = F.relu(args.K - d2).pow(2).mean()
                loss = loss + w * args.lambda_d2 * tgt + args.lambda_order * order + args.lambda_wrong * wrong
                parts["target"] = tgt
            elif arm == "redund":
                tgt = F.relu(d2 + args.K).pow(2).mean()
                loss = loss + w * args.lambda_d2 * tgt + args.lambda_order * order + args.lambda_wrong * wrong
                parts["target"] = tgt
            elif arm == "notarget":
                loss = loss + args.lambda_order * order + args.lambda_wrong * wrong
            elif arm == "ce_only":
                pass
            elif arm == "partdrop_ft":
                ze = torch.cat([Z["x10"], Z["x01"], Z["x00"]])
                lab = F.cross_entropy(ze, ya.repeat(3))
                loss = loss + args.label_weight * lab
                parts["label"] = lab
            elif arm in ("oe_is", "partdrop_is"):
                # label-based training on the part-removed corners (as oe_ft / partdrop_ft) + the IS terms of arm "is"
                ze = torch.cat([Z["x10"], Z["x01"], Z["x00"]])
                lab = (-F.log_softmax(ze, dim=1).mean(1).mean()) if arm == "oe_is" else F.cross_entropy(ze, ya.repeat(3))
                tgt = F.relu(args.K - d2).pow(2).mean()
                loss = loss + args.label_weight * lab + w * args.lambda_d2 * tgt + args.lambda_order * order + args.lambda_wrong * wrong
                parts["label"] = lab
                parts["target"] = tgt
            elif arm == "oe_ft":
                ze = torch.cat([Z["x10"], Z["x01"], Z["x00"]])
                lab = -F.log_softmax(ze, dim=1).mean(1).mean()
                loss = loss + args.label_weight * lab
                parts["label"] = lab
            else:
                raise ValueError(arm)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            n_steps += 1
            parts["loss"] = loss.detach()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            if args.max_steps and step >= args.max_steps:
                break
        val = validate(model, device, val_y, val_corners)
        rec = {"epoch": epoch, "step": step, "minutes": round((time.time() - t0) / 60, 2), **{f"train_{k}": v / max(n_steps, 1) for k, v in sums.items()}, **val}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
        if val["val_top1"] > best:
            best = val["val_top1"]
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": cfg, "val_metrics": val}, out / "best_by_val_top1.pt")
        if args.max_steps and step >= args.max_steps:
            break
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": cfg, "val_metrics": val}, out / "last.pt")
    (out / "done.json").write_text(json.dumps({"best_val_top1": best, "minutes": round((time.time() - t0) / 60, 2)}))


if __name__ == "__main__":
    main()
