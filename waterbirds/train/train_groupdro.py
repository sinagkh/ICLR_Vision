#!/usr/bin/env python3
"""
GroupDRO on Waterbirds.

- Online group weights w over the four bird x background groups: each step minimises the per-group mean losses of the
  groups present in the batch, weighted by w renormalised over those groups, then updates w <- w * exp(eta * loss)
  and renormalises
- Tracks best checkpoint by validation worst-group accuracy (WGA)
- Saves outputs under --output_dir/checkpoints/groupdro_best.pt (with the final group weights "w")
- Writes train_summary.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from tqdm import tqdm

from backbone_utils import (
    build_image_transforms,
    set_backbone_trainable,
    split_backbone_and_head_params,
)


class HFWaterbirds(Dataset):
    """Returns (x, y, g) with g = 2*y + bg (g in {0..3})."""

    def __init__(self, split: str, image_size: int = 224, train_aug: bool = False, backbone: str = "resnet50"):
        self.ds = load_dataset("grodino/waterbirds", split=split)
        feats = self.ds.features
        self.img_k = "image" if "image" in feats else next(k for k in feats if "image" in k.lower())
        self.label_k = next(
            k for k, v in feats.items()
            if ("label" in k or "bird" in k) and str(getattr(v, "dtype", v)).startswith("int")
        )
        self.bg_k = next(
            k for k, v in feats.items()
            if ("place" in k or "background" in k or k in ["bg", "water", "land"])
            and str(getattr(v, "dtype", v)).startswith("int")
        )
        self.tf = build_image_transforms(backbone=backbone, image_size=image_size, train_aug=train_aug)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        ex = self.ds[i]
        pil = ex[self.img_k].convert("RGB")
        y = int(ex[self.label_k])
        bg = int(ex[self.bg_k])
        g = 2 * y + bg
        return self.tf(pil), y, g


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return contextlib.nullcontext()


def build_backbone(name: str, num_classes: int = 2, image_size: int = 224):
    n = name.lower()
    if n == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if n == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if n == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if n == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    if n == "vit_base_patch14_dinov2":
        return timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=num_classes, img_size=image_size)
    raise ValueError(f"Unknown backbone: {name}")


@torch.no_grad()
def worst_group_metrics(logits: torch.Tensor, ys: torch.Tensor, groups: torch.Tensor, num_groups: int = 4):
    pred = logits.argmax(1).cpu().numpy()
    ys = ys.cpu().numpy()
    groups = groups.cpu().numpy()
    accs = []
    total_correct = 0
    total_count = 0
    for g in range(num_groups):
        idx = groups == g
        if idx.sum() == 0:
            continue
        correct = int((pred[idx] == ys[idx]).sum())
        count = int(idx.sum())
        accs.append((g, correct / count, count))
        total_correct += correct
        total_count += count
    if not accs:
        return 0.0, 0.0, {}
    worst = min(accs, key=lambda t: t[1])[1]
    avg = float(total_correct / max(1, total_count))
    details = {int(g): (float(a), int(n)) for g, a, n in accs}
    return avg, float(worst), details


def evaluate(model: nn.Module, dl: DataLoader, device: torch.device):
    model.eval()
    logits_all, ys_all, groups_all = [], [], []
    with torch.inference_mode(), autocast_context(device, enabled=device.type == "cuda"):
        for x, y, g in dl:
            x = x.to(device, non_blocking=True)
            logits_all.append(model(x).cpu())
            ys_all.append(y)
            groups_all.append(g)
    logits = torch.cat(logits_all)
    ys = torch.cat(ys_all)
    groups = torch.cat(groups_all)
    return worst_group_metrics(logits, ys, groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--eta", type=float, default=0.05, help="GroupDRO weight update rate")
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap.add_argument("--train_aug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = HFWaterbirds("train", image_size=args.image_size, train_aug=args.train_aug, backbone=args.backbone)
    val_ds = HFWaterbirds("validation", image_size=args.image_size, train_aug=False, backbone=args.backbone)
    test_ds = HFWaterbirds("test", image_size=args.image_size, train_aug=False, backbone=args.backbone)

    def mk_loader(ds: Dataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=(args.num_workers > 0),
            prefetch_factor=(2 if args.num_workers > 0 else None),
        )

    tr = mk_loader(train_ds, True)
    va = mk_loader(val_ds, False)
    te = mk_loader(test_ds, False)

    model = build_backbone(args.backbone, image_size=args.image_size).to(device)
    backbone_params, head_params = split_backbone_and_head_params(model)
    if head_params and backbone_params and args.backbone_lr_mult != 1.0:
        opt = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
                {"params": head_params, "lr": args.lr},
            ],
            weight_decay=args.wd,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    ce = nn.CrossEntropyLoss(reduction="none", label_smoothing=args.label_smoothing)

    w = torch.ones(4, device=device) / 4.0
    best_wga = -1.0
    best_epoch = None
    best_val_avg = None
    best_val_details = None
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        set_backbone_trainable(model, train_backbone=(epoch > args.freeze_backbone_epochs))
        pbar = tqdm(tr, desc=f"Epoch {epoch}/{args.epochs}", ncols=120, leave=False)
        for x, y, g in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device)
            g = g.to(device)
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(device.type == "cuda" and args.amp)):
                logits = model(x)
                l_i = ce(logits, y)
                present = torch.unique(g)
                loss_g = []
                wg = w[present] / w[present].sum()
                for gj in present:
                    loss_g.append(l_i[g == gj].mean())
                loss_g = torch.stack(loss_g) if loss_g else torch.tensor(0.0, device=device)
                loss = (wg * loss_g).sum() if loss_g.ndim > 0 else torch.tensor(0.0, device=device)

            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(opt)
            scaler.update()

            with torch.no_grad():
                if loss_g.ndim > 0:
                    delta = torch.zeros_like(w)
                    for j, gj in enumerate(present):
                        delta[gj] = loss_g[j]
                    w = w * torch.exp(args.eta * delta)
                    w = w / w.sum().clamp(min=1e-12)

            pbar.set_postfix(loss=f"{float(loss):.3f}")

        va_avg, va_wga, va_det = evaluate(model, va, device)
        print(f"[VAL] avg={va_avg:.3f} worst={va_wga:.3f} groups={va_det}")
        if va_wga > best_wga:
            best_wga = va_wga
            best_epoch = epoch
            best_val_avg = va_avg
            best_val_details = va_det
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args), "w": w.detach().cpu().tolist()},
                ckpt_dir / "groupdro_best.pt",
            )
        sched.step()

    ck = torch.load(ckpt_dir / "groupdro_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    te_avg, te_wga, te_det = evaluate(model, te, device)
    elapsed = time.time() - start
    print(f"[TEST] avg={te_avg:.3f} worst={te_wga:.3f} groups={te_det} epoch={best_epoch}  (time {elapsed/60:.1f} min)")

    summary = {
        "output_dir": str(output_dir),
        "best_wga_epoch": best_epoch,
        "best_val_wga": best_wga,
        "best_checkpoint": {
            "path": str((ckpt_dir / "groupdro_best.pt").resolve()),
            "epoch": best_epoch,
            "val_average_accuracy": best_val_avg,
            "val_worst_group_accuracy": best_wga,
            "val_groups": best_val_details,
            "test_average_accuracy": te_avg,
            "test_worst_group_accuracy": te_wga,
            "test_groups": te_det,
        },
        "args": vars(args),
    }
    summary_path = output_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[saved] summary -> {summary_path}")


if __name__ == "__main__":
    main()
