#!/usr/bin/env python3
"""ERM for the MAE-pretrained ViT-B/16 on PACO: the MAE fine-tuning recipe scaled to 20 epochs.

AdamW, lr 2.5e-4 (base 5e-4 x 128/256), layer-wise lr decay 0.65, weight decay 0.05 (none on norms, biases, tokens),
drop path 0.1, 2 warm-up epochs then cosine, label smoothing 0.1, batch 128, horizontal flip, AMP.
Writes <out-dir>/best_by_val_top1.pt (best validation top-1) and metrics_epoch.jsonl.
usage: train_erm_vit.py --seed S --out-dir DIR
"""
from __future__ import annotations

import argparse
import json
import math
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
from paco.src.common import LISTS, CachedNatural, normalize_gpu, read_rows  # noqa: E402


def param_groups(model, wd, decay):
    n = len(model.blocks) + 1
    groups = {}
    for name, p in model.named_parameters():
        if name.startswith(("cls_token", "pos_embed", "patch_embed", "reg_token")):
            layer = 0
        elif name.startswith("blocks."):
            layer = int(name.split(".")[1]) + 1
        else:
            layer = n
        no_wd = p.ndim == 1 or name in ("cls_token", "pos_embed")
        g = groups.setdefault((layer, no_wd), {"params": [], "lr_scale": decay ** (n - layer), "weight_decay": 0.0 if no_wd else wd})
        g["params"].append(p)
    return list(groups.values())


@torch.inference_mode()
def top1(model, loader, device):
    model.eval(); correct = n = 0
    for x, y, _ in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            z = model(normalize_gpu(x, device))
        correct += int((z.argmax(1).cpu() == y).sum()); n += len(y)
    model.train()
    return 100.0 * correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2.5e-4)
    ap.add_argument("--layer-decay", type=float, default=0.65)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--drop-path", type=float, default=0.1)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--max-steps", type=int, default=None, help="stop early after this many steps (quick checks only)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda"); torch.backends.cudnn.benchmark = True
    out = args.out_dir; out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "out_dir": str(out), "backbone": "vit_base_patch16_224.mae"}, indent=1, default=str))

    train_y = [int(r["class_index"]) for r in read_rows(LISTS / "train_instances.csv")]
    val_y = [int(r["class_index"]) for r in read_rows(LISTS / "val_instances.csv")]
    loader = DataLoader(CachedNatural("train", train_y), batch_size=args.batch, shuffle=True, num_workers=8, drop_last=True, persistent_workers=True, prefetch_factor=4)
    val_loader = DataLoader(CachedNatural("val", val_y), batch_size=256, num_workers=6)

    model = rv.build_vit_mae(5, pretrained=True, drop_path_rate=args.drop_path).to(device)
    model.set_grad_checkpointing(True)
    groups = param_groups(model, args.wd, args.layer_decay)
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.999))
    scaler = torch.amp.GradScaler("cuda")
    total = args.epochs * len(loader); warm = args.warmup_epochs * len(loader)
    best, step, t0 = -1.0, 0, time.time()
    log = open(out / "metrics_epoch.jsonl", "w")
    model.train()
    for epoch in range(1, args.epochs + 1):
        losses = []
        for x, y, _ in loader:
            lr = args.lr * (step + 1) / warm if step < warm else 1e-6 + (args.lr - 1e-6) * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
            for g in opt.param_groups:
                g["lr"] = lr * g["lr_scale"]
            xb = normalize_gpu(x, device); yb = y.to(device)
            flip = torch.rand(len(yb), device=device) < 0.5
            xb = torch.where(flip[:, None, None, None], xb.flip(3), xb)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = F.cross_entropy(model(xb).float(), yb, label_smoothing=0.1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss)); step += 1
            if args.max_steps and step >= args.max_steps:
                break
        val = top1(model, val_loader, device)
        rec = {"epoch": epoch, "step": step, "minutes": round((time.time() - t0) / 60, 2), "train_loss": float(np.mean(losses)), "val_top1": val,
               "max_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        log.write(json.dumps(rec) + "\n"); log.flush(); print(json.dumps(rec), flush=True)
        if val > best:
            best = val
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_top1": val}, out / "best_by_val_top1.pt")
        if args.max_steps and step >= args.max_steps:
            break
    (out / "done.json").write_text(json.dumps({"best_val_top1": best, "minutes": round((time.time() - t0) / 60, 2)}))


if __name__ == "__main__":
    main()
