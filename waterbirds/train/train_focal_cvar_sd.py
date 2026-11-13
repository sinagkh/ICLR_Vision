#!/usr/bin/env python3
"""
Focal Loss, CVaR and Spectral Decoupling on Waterbirds (no group labels in training).

  --method focal  focal loss (Lin et al., 2017): mean of (1 - p_y)^gamma * CE
  --method cvar   CVaR (Levy et al., 2020): mean CE of the top alpha fraction of the batch
  --method sd     Spectral Decoupling (Pezeshki et al., 2021): CE + sd_alpha * mean squared logit norm

- Tracks best checkpoint by validation worst-group accuracy (WGA)
- Saves outputs under --output_dir/checkpoints/<method>_best.pt
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
from typing import Callable

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
    avg = float(total_correct / max(1, total_count))
    worst = float(min(accs, key=lambda t: t[1])[1])
    details = {int(g): (float(a), int(n)) for g, a, n in accs}
    return avg, worst, details


class HFWaterbirds(Dataset):
    """Returns (x, y, group, idx) with group = 2*y + bg."""

    def __init__(self, split: str, image_size: int = 224, train_aug: bool = False, backbone: str = "resnet50"):
        self.ds = load_dataset("grodino/waterbirds", split=split)
        features = self.ds.features
        self.img_k = "image" if "image" in features else next((k for k in features if "image" in k.lower()), None)
        self.label_k = next(
            (k for k, v in features.items()
             if ("label" in k or "bird" in k) and str(getattr(v, "dtype", v)).startswith("int")),
            None,
        )
        self.bg_k = next(
            (k for k, v in features.items()
             if ("place" in k or "background" in k or k in ["bg", "water", "land"])
             and str(getattr(v, "dtype", v)).startswith("int")),
            None,
        )
        if not (self.img_k and self.label_k and self.bg_k):
            raise RuntimeError(f"Could not infer keys from dataset features: {list(features.keys())}")
        self.tf = build_image_transforms(backbone=backbone, image_size=image_size, train_aug=train_aug)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, i: int):
        ex = self.ds[i]
        img = ex[self.img_k].convert("RGB")
        x = self.tf(img)
        y = int(ex[self.label_k])
        bg = int(ex[self.bg_k])
        group = 2 * y + bg
        return x, y, group, i


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def build_backbone(name: str, num_classes: int = 2, image_size: int = 224):
    name = name.lower()
    if name == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "vit_base_patch14_dinov2":
        return timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=num_classes, img_size=image_size)
    raise ValueError(f"Unknown backbone: {name}")


def build_optimizer(model: nn.Module, lr: float, wd: float, backbone_lr_mult: float):
    backbone_params, head_params = split_backbone_and_head_params(model)
    if head_params and backbone_params and backbone_lr_mult != 1.0:
        return torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": lr * backbone_lr_mult},
                {"params": head_params, "lr": lr},
            ],
            weight_decay=wd,
        )
    return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_y, all_g = [], [], []
    with autocast_context(device, enabled=(device.type == "cuda")):
        for x, y, g, _ in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y)
            all_g.append(g)
    logits = torch.cat(all_logits)
    ys = torch.cat(all_y)
    groups = torch.cat(all_g)
    return worst_group_metrics(logits, ys, groups)


def ce_per_sample(logits: torch.Tensor, y: torch.Tensor, label_smoothing: float = 0.0):
    return nn.functional.cross_entropy(logits, y, reduction="none", label_smoothing=label_smoothing)


def spectral_decoupling_penalty(logits: torch.Tensor):
    return (logits ** 2).sum(dim=1).mean()


def focal_loss_from_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    gamma: float = 2.0,
    alpha: float = -1.0,
    label_smoothing: float = 0.0,
):
    ce = ce_per_sample(logits, y, label_smoothing=label_smoothing)
    pt = torch.softmax(logits, dim=1).gather(1, y.view(-1, 1)).squeeze(1).clamp(1e-6, 1 - 1e-6)
    weights = (1.0 - pt) ** gamma
    if alpha >= 0.0:
        alpha_t = torch.where(y == 1, torch.full_like(pt, alpha), torch.full_like(pt, 1.0 - alpha))
        weights = weights * alpha_t
    return (weights * ce).mean()


def cvar_top_alpha_loss(logits: torch.Tensor, y: torch.Tensor, alpha: float = 0.2, label_smoothing: float = 0.0):
    losses = ce_per_sample(logits, y, label_smoothing=label_smoothing)
    k = max(1, int(np.ceil(alpha * losses.numel())))
    vals, _ = torch.topk(losses, k=k, largest=True, sorted=False)
    return vals.mean()


def write_summary(
    output_dir: Path,
    args: argparse.Namespace,
    ckpt_name: str,
    best_epoch: int | None,
    best_val_avg: float | None,
    best_val_wga: float | None,
    best_val_groups,
    test_avg: float,
    test_wga: float,
    test_groups,
) -> None:
    summary = {
        "output_dir": str(output_dir),
        "best_wga_epoch": best_epoch,
        "best_val_wga": best_val_wga,
        "best_checkpoint": {
            "path": str((output_dir / "checkpoints" / ckpt_name).resolve()),
            "epoch": best_epoch,
            "val_average_accuracy": best_val_avg,
            "val_worst_group_accuracy": best_val_wga,
            "val_groups": best_val_groups,
            "test_average_accuracy": test_avg,
            "test_worst_group_accuracy": test_wga,
            "test_groups": test_groups,
        },
        "args": vars(args),
    }
    summary_path = output_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[saved] summary -> {summary_path}")


def train_single_phase(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    loss_fn_builder: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ckpt_name: str,
) -> None:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    opt = build_optimizer(model, args.lr, args.wd, args.backbone_lr_mult)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    best_wga = -1.0
    best_epoch = None
    best_val_avg = None
    best_val_groups = None
    start = time.time()
    epoch_bar = tqdm(range(1, args.epochs + 1), desc=f"Epochs[{args.method}]", ncols=100, leave=False)

    for epoch in epoch_bar:
        model.train()
        set_backbone_trainable(model, train_backbone=(epoch > args.freeze_backbone_epochs))
        run_loss, steps = 0.0, 0
        for x, y, _, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(device.type == "cuda" and args.amp)):
                logits = model(x)
                loss = loss_fn_builder(logits, y)
            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(opt)
            scaler.update()
            run_loss += float(loss.item())
            steps += 1
        epoch_bar.set_postfix(loss=f"{run_loss / max(1, steps):.3f}")

        va_avg, va_worst, va_det = evaluate(model, val_loader, device)
        print(f"[VAL] avg={va_avg:.3f} worst={va_worst:.3f} groups={va_det}")
        if va_worst > best_wga:
            best_wga = va_worst
            best_epoch = epoch
            best_val_avg = va_avg
            best_val_groups = va_det
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "args": vars(args)},
                ckpt_dir / ckpt_name,
            )
        sched.step()

    ckpt = torch.load(ckpt_dir / ckpt_name, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    te_avg, te_worst, te_det = evaluate(model, test_loader, device)
    elapsed = (time.time() - start) / 60.0
    print(f"[TEST] avg={te_avg:.4f} worst={te_worst:.4f} groups={te_det} epoch={best_epoch}  (time {elapsed:.1f} min)")
    write_summary(output_dir, args, ckpt_name, best_epoch, best_val_avg, best_wga, best_val_groups, te_avg, te_worst, te_det)


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    tr = HFWaterbirds("train", image_size=args.image_size, train_aug=args.train_aug, backbone=args.backbone)
    va = HFWaterbirds("validation", image_size=args.image_size, train_aug=False, backbone=args.backbone)
    te = HFWaterbirds("test", image_size=args.image_size, train_aug=False, backbone=args.backbone)
    train_loader = make_loader(tr, args.batch_size, True, args.num_workers)
    val_loader = make_loader(va, args.batch_size, False, args.num_workers)
    test_loader = make_loader(te, args.batch_size, False, args.num_workers)

    model = build_backbone(args.backbone, num_classes=2, image_size=args.image_size).to(device)
    ckpt_name = f"{args.method}_best.pt"

    if args.method == "sd":
        ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

        def loss_fn(logits, y):
            return ce(logits, y) + args.sd_alpha * spectral_decoupling_penalty(logits)
    elif args.method == "cvar":
        def loss_fn(logits, y):
            return cvar_top_alpha_loss(logits, y, alpha=args.cvar_alpha, label_smoothing=args.label_smoothing)
    else:
        def loss_fn(logits, y):
            return focal_loss_from_logits(
                logits, y, gamma=args.focal_gamma, alpha=args.focal_alpha, label_smoothing=args.label_smoothing
            )

    train_single_phase(model, train_loader, val_loader, test_loader, device, args, loss_fn, ckpt_name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", type=str, required=True, choices=["focal", "cvar", "sd"])
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "vit_base_patch14_dinov2"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap.add_argument("--train_aug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--sd_alpha", type=float, default=1e-4, help="Spectral Decoupling penalty weight")
    ap.add_argument("--cvar_alpha", type=float, default=0.2, help="CVaR fraction of highest-loss examples")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.25, help="focal class weight for y=1; negative disables it")
    ap.add_argument("--seed", type=int, default=1)
    main(ap.parse_args())
