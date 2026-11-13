#!/usr/bin/env python3
"""
JTT for Waterbirds.

Stage 1: ERM on augmented train
Mining : find errors on clean train
Stage 2: reweighted ERM on augmented train

It supports the same backbones and output conventions as the ERM / steering
scripts.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
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
        acc = correct / count
        accs.append((g, acc, count))
        total_correct += correct
        total_count += count
    if not accs:
        return 0.0, 0.0, {}
    avg = float(total_correct / max(1, total_count))
    worst = float(min(accs, key=lambda t: t[1])[1])
    details = {int(g): (float(a), int(n)) for g, a, n in accs}
    return avg, worst, details


class HFWaterbirds(Dataset):
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


class JTTUpsampledDataset(Dataset):
    """Dataset-level upsampling to match the public JTT implementation."""

    def __init__(self, base: Dataset, error_indices: set[int], upweight_factor: int):
        if upweight_factor < 0:
            raise ValueError(f"upweight_factor must be non-negative, got {upweight_factor}")
        ordered_errors = sorted(int(i) for i in error_indices)
        self.base = base
        self.indices = list(range(len(base))) + ordered_errors * upweight_factor

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[self.indices[i]]


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
    if name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    if name == "convnext_tiny":
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    if name == "vit_b_16":
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    if name == "vit_base_patch14_dinov2":
        return timm.create_model("vit_base_patch14_dinov2", pretrained=True, num_classes=num_classes, img_size=image_size)
    raise ValueError(f"Unknown backbone: {name}")


def build_optimizer(
    model: nn.Module,
    lr: float,
    wd: float,
    backbone_lr_mult: float,
    optimizer: str = "adamw",
    momentum: float = 0.9,
):
    backbone_params, head_params = split_backbone_and_head_params(model)
    if head_params and backbone_params and backbone_lr_mult != 1.0:
        param_groups = [
            {"params": backbone_params, "lr": lr * backbone_lr_mult},
            {"params": head_params, "lr": lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": lr}]
    if optimizer == "sgd":
        return torch.optim.SGD(param_groups, lr=lr, momentum=momentum, weight_decay=wd)
    return torch.optim.AdamW(param_groups, lr=lr, weight_decay=wd)


def build_scheduler(opt: torch.optim.Optimizer, epochs: int, lr: float, scheduler: str):
    if scheduler == "none":
        return None
    if scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.1)
    raise ValueError(f"Unknown scheduler: {scheduler}")


@torch.no_grad()
def evaluate(model: nn.Module, dl: DataLoader, device: torch.device):
    model.eval()
    all_logits, all_y, all_g = [], [], []
    with autocast_context(device, enabled=(device.type == "cuda")):
        for x, y, g, _ in dl:
            x = x.to(device, non_blocking=True)
            all_logits.append(model(x).cpu())
            all_y.append(y)
            all_g.append(g)
    return worst_group_metrics(torch.cat(all_logits), torch.cat(all_y), torch.cat(all_g))


def normalized_weighted_loss(per_sample: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (weights * per_sample).sum() / weights.sum().clamp_min(1e-12)


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


def main(args: argparse.Namespace) -> None:
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

    tr_aug = HFWaterbirds("train", image_size=args.image_size, train_aug=args.train_aug, backbone=args.backbone)
    tr_clean = HFWaterbirds("train", image_size=args.image_size, train_aug=False, backbone=args.backbone)
    va = HFWaterbirds("validation", image_size=args.image_size, train_aug=False, backbone=args.backbone)
    te = HFWaterbirds("test", image_size=args.image_size, train_aug=False, backbone=args.backbone)

    train_loader_s1 = make_loader(tr_aug, args.batch_size, True, args.num_workers)
    clean_loader = make_loader(tr_clean, args.batch_size, False, args.num_workers)
    val_loader = make_loader(va, args.batch_size, False, args.num_workers)
    test_loader = make_loader(te, args.batch_size, False, args.num_workers)

    print(f"--- [Seed {args.seed}] JTT Stage 1 ---")
    model1 = build_backbone(args.backbone, image_size=args.image_size).to(device)
    if args.stage1_checkpoint:
        ckpt = torch.load(args.stage1_checkpoint, map_location=device, weights_only=False)
        state = ckpt.get("model", ckpt)
        model1.load_state_dict(state)
        print(f"[stage1] loaded checkpoint: {args.stage1_checkpoint}")
    else:
        opt1 = build_optimizer(model1, args.lr, args.wd, args.backbone_lr_mult, args.optimizer, args.momentum)
        sched1 = build_scheduler(opt1, args.jtt_stage1_epochs, args.lr, args.scheduler)
        scaler1 = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))
        ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
        for epoch in range(1, args.jtt_stage1_epochs + 1):
            model1.train()
            set_backbone_trainable(model1, train_backbone=(epoch > args.freeze_backbone_epochs))
            for x, y, _, _ in train_loader_s1:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                opt1.zero_grad(set_to_none=True)
                with autocast_context(device, enabled=(device.type == "cuda" and args.amp)):
                    loss = ce(model1(x), y)
                scaler1.scale(loss).backward()
                if args.clip_grad_norm > 0:
                    scaler1.unscale_(opt1)
                    nn.utils.clip_grad_norm_(model1.parameters(), args.clip_grad_norm)
                scaler1.step(opt1)
                scaler1.update()
            if sched1 is not None:
                sched1.step()

    print(f"--- [Seed {args.seed}] JTT Mining on clean train split ---")
    model1.eval()
    mis_idx: set[int] = set()
    mis_groups: dict[int, int] = {}
    with torch.inference_mode(), autocast_context(device, enabled=(device.type == "cuda" and args.amp)):
        for x, y, g, idx in tqdm(clean_loader, desc="Mining", ncols=100, leave=False):
            x = x.to(device, non_blocking=True)
            pred = model1(x).argmax(1).cpu()
            wrong = pred != y
            for k, gk, w in zip(idx.tolist(), g.tolist(), wrong.tolist()):
                if w:
                    mis_idx.add(k)
                    mis_groups[gk] = mis_groups.get(gk, 0) + 1
    print(f"Found {len(mis_idx)} misclassified examples out of {len(tr_clean)}.")
    print(f"Error-set group counts: {dict(sorted(mis_groups.items()))}")
    with (output_dir / "jtt_error_set.json").open("w", encoding="utf-8") as handle:
        json.dump({"indices": sorted(mis_idx), "group_counts": mis_groups}, handle)

    print(f"--- [Seed {args.seed}] JTT Stage 2 (lambda={args.jtt_lambda}) ---")
    if args.jtt_expand_dataset:
        upweight_factor = int(round(args.jtt_lambda))
        if abs(upweight_factor - args.jtt_lambda) > 1e-6:
            raise ValueError("--jtt_expand_dataset requires an integer-valued --jtt_lambda")
        stage2_ds = JTTUpsampledDataset(tr_aug, mis_idx, upweight_factor)
        train_loader_s2 = make_loader(stage2_ds, args.batch_size, True, args.num_workers)
        print(
            f"Stage-2 dataset expansion: {len(tr_aug)} natural + "
            f"{len(mis_idx)} errors x {upweight_factor} = {len(stage2_ds)} samples/epoch."
        )
    else:
        train_loader_s2 = make_loader(tr_aug, args.batch_size, True, args.num_workers)
        print("Stage-2 weighted-loss surrogate: using the natural dataset length.")

    model2 = build_backbone(args.backbone, image_size=args.image_size).to(device)
    opt2 = build_optimizer(model2, args.lr, args.wd, args.backbone_lr_mult, args.optimizer, args.momentum)
    sched2 = build_scheduler(opt2, args.jtt_stage2_epochs, args.lr, args.scheduler)
    scaler2 = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    def jtt_loss(logits: torch.Tensor, y: torch.Tensor, idx_batch: torch.Tensor):
        per_sample = nn.functional.cross_entropy(logits, y, reduction="none", label_smoothing=args.label_smoothing)
        if args.jtt_expand_dataset:
            return per_sample.mean()
        weights = torch.ones_like(per_sample)
        for i, k in enumerate(idx_batch.tolist()):
            if k in mis_idx:
                weights[i] = args.jtt_lambda
        return normalized_weighted_loss(per_sample, weights)

    best_wga = -1.0
    best_epoch = None
    best_val_avg = None
    best_val_groups = None
    ckpt_name = "jtt_best.pt"

    for epoch in range(1, args.jtt_stage2_epochs + 1):
        model2.train()
        set_backbone_trainable(model2, train_backbone=(epoch > args.freeze_backbone_epochs))
        for x, y, _, idx in train_loader_s2:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt2.zero_grad(set_to_none=True)
            with autocast_context(device, enabled=(device.type == "cuda" and args.amp)):
                loss = jtt_loss(model2(x), y, idx)
            scaler2.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler2.unscale_(opt2)
                nn.utils.clip_grad_norm_(model2.parameters(), args.clip_grad_norm)
            scaler2.step(opt2)
            scaler2.update()

        va_avg, va_worst, va_det = evaluate(model2, val_loader, device)
        print(f"[VAL] avg={va_avg:.3f} worst={va_worst:.3f} groups={va_det}")
        if va_worst > best_wga:
            best_wga = va_worst
            best_epoch = epoch
            best_val_avg = va_avg
            best_val_groups = va_det
            torch.save({"model": model2.state_dict(), "epoch": epoch, "args": vars(args)}, ckpt_dir / ckpt_name)
        if sched2 is not None:
            sched2.step()

    ckpt = torch.load(ckpt_dir / ckpt_name, map_location=device, weights_only=False)
    model2.load_state_dict(ckpt["model"])
    te_avg, te_worst, te_det = evaluate(model2, test_loader, device)
    print(f"[TEST] avg={te_avg:.4f} worst={te_worst:.4f} groups={te_det} epoch={best_epoch}")
    write_summary(output_dir, args, ckpt_name, best_epoch, best_val_avg, best_wga, best_val_groups, te_avg, te_worst, te_det)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backbone",
        type=str,
        default="resnet50",
        choices=["resnet50", "resnet18", "convnext_tiny", "vit_b_16", "vit_base_patch14_dinov2"],
    )
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["cosine", "none"])
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap.add_argument("--train_aug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output_dir", type=str, default=".")
    ap.add_argument("--stage1_checkpoint", type=str, default=None)
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--label_smoothing", type=float, default=0.0)
    ap.add_argument("--jtt_stage1_epochs", type=int, default=5)
    ap.add_argument("--jtt_stage2_epochs", type=int, default=10)
    ap.add_argument("--jtt_lambda", type=float, default=10.0)
    ap.add_argument("--jtt_expand_dataset", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, default=1)
    main(ap.parse_args())
