#!/usr/bin/env python3
"""
From-scratch counterfactual augmentation baseline for Waterbirds.

A stricter control:
  - start from the standard ImageNet-pretrained backbone,
  - train on natural images with ordinary CE,
  - additionally train on object-present edited images x01 with ordinary CE,
  - do not use the finite-difference objective.

The edited images reuse the same masked Waterbirds edit pipeline (train_cfaug.py) and
random donor backgrounds, but the optimization is standard supervised training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from backbone_utils import set_backbone_trainable, split_backbone_and_head_params


HELPER_PATH = Path(__file__).resolve().with_name("train_cfaug.py")
_SPEC = importlib.util.spec_from_file_location("wb_cfaug_helpers", HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load helper module from {HELPER_PATH}")
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

HFWaterbirds = _HELPERS.HFWaterbirds
collate_with_pil = _HELPERS.collate_with_pil
build_donor_pools = _HELPERS.build_donor_pools
cleanse_donor = _HELPERS.cleanse_donor
hist_match = _HELPERS.hist_match
threshold_mask = _HELPERS.threshold_mask
mask_crop = _HELPERS.mask_crop
mask_bbox = _HELPERS.mask_bbox
paste_with_segment_mask = _HELPERS.paste_with_segment_mask
set_seed = _HELPERS.set_seed
get_device = _HELPERS.get_device
build_backbone = _HELPERS.build_backbone
worst_group_metrics = _HELPERS.worst_group_metrics


def make_loader(ds, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
        collate_fn=collate_with_pil,
    )


def evaluate(model, dl, device):
    model.eval()
    all_logits, all_y, all_g = [], [], []
    with torch.inference_mode():
        for x, y, g, _, _, _ in dl:
            x = x.to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                logits = model(x)
            all_logits.append(logits.cpu())
            all_y.append(y)
            all_g.append(g)
    logits = torch.cat(all_logits)
    ys = torch.cat(all_y)
    groups = torch.cat(all_g)
    return worst_group_metrics(logits, ys, groups)


def choose_donor(pool_all: list[int], pool_same: list[int], pool_opp: list[int], donor_policy: str, anchor_idx: int) -> int | None:
    if donor_policy == "same":
        pool = pool_same
    elif donor_policy == "opposite":
        pool = pool_opp
    elif donor_policy == "random":
        pool = [j for j in pool_all if j != anchor_idx]
    else:
        raise ValueError(f"Unknown donor_policy: {donor_policy}")
    if not pool:
        return None
    return random.choice(pool)


def build_cf_batch(train_ds, donor_pools, pils, ys, bgs, idxs, donor_policy: str, max_box_frac: float):
    x_cf_list = []
    y_cf_list = []

    tf_train = train_ds.tf
    pool_all = donor_pools.get("all", [])

    for pil_img, y, bg, idx in zip(pils, ys.tolist(), bgs.tolist(), idxs.tolist()):
        anchor_mask = train_ds.load_mask(int(idx))
        box = mask_bbox(anchor_mask) if anchor_mask is not None else None
        if box is None:
            box = train_ds.box_for_index(int(idx))
        if box is None or anchor_mask is None:
            continue
        w = box[2] - box[0]
        h = box[3] - box[1]
        if (w * h) > max_box_frac * (pil_img.width * pil_img.height):
            continue

        same_pool = donor_pools.get(int(bg), [])
        opp_pool = donor_pools.get(1 - int(bg), [])
        donor_idx = choose_donor(pool_all, same_pool, opp_pool, donor_policy, int(idx))
        if donor_idx is None:
            continue

        _, _, _, donor_pil, _, _ = train_ds[donor_idx]
        donor_mask = train_ds.load_mask(donor_idx)
        if donor_mask is None:
            continue

        donor_clean = cleanse_donor(donor_pil, donor_mask)
        donor_matched = hist_match(donor_clean, pil_img)
        obj_crop = pil_img.crop(tuple(box))
        segment_mask = threshold_mask(mask_crop(anchor_mask, box), threshold=127)
        x01_pil = paste_with_segment_mask(donor_matched, obj_crop, tuple(box), segment_mask, blur_radius=0)

        x_cf_list.append(tf_train(x01_pil))
        y_cf_list.append(int(y))

    if not x_cf_list:
        return None, None
    x_cf = torch.stack(x_cf_list, dim=0)
    y_cf = torch.as_tensor(y_cf_list, dtype=torch.long)
    return x_cf, y_cf


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50", "resnet18", "convnext_tiny", "vit_base_patch14_dinov2"])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap.add_argument("--train_aug", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--data_root", type=str, default=str(_HELPERS._DEFAULT_REAL_ROOT))
    ap.add_argument("--lambda_cf_ce", type=float, default=1.0)
    ap.add_argument("--donor_policy", type=str, default="random", choices=["random", "same", "opposite"])
    ap.add_argument("--max_box_frac", type=float, default=0.45)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}, CUDA:{torch.cuda.is_available()} | cudnn.benchmark=True")
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

    train_ds = HFWaterbirds("train", image_size=args.image_size, train_aug=args.train_aug, data_root=args.data_root, backbone=args.backbone)
    val_ds = HFWaterbirds("validation", image_size=args.image_size, train_aug=False, data_root=args.data_root, backbone=args.backbone)
    test_ds = HFWaterbirds("test", image_size=args.image_size, train_aug=False, data_root=args.data_root, backbone=args.backbone)
    donor_pools = build_donor_pools(train_ds)

    train_loader = make_loader(train_ds, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    model = build_backbone(args.backbone, num_classes=2, image_size=args.image_size).to(device)
    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    backbone_params, head_params = split_backbone_and_head_params(model)
    if head_params and backbone_params and args.backbone_lr_mult != 1.0:
        param_groups = [
            {"params": backbone_params, "lr": args.lr * args.backbone_lr_mult},
            {"params": head_params, "lr": args.lr},
        ]
    else:
        param_groups = [{"params": model.parameters(), "lr": args.lr}]

    if args.optimizer == "sgd":
        opt = torch.optim.SGD(param_groups, lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    else:
        opt = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    best_wga = -1.0
    best_epoch = None
    best_val_avg = None
    best_val_details = None
    start = time.time()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", ncols=120, leave=False)
    for epoch in epoch_bar:
        model.train()
        freeze_backbone = (epoch <= args.freeze_backbone_epochs)
        set_backbone_trainable(model, train_backbone=not freeze_backbone)

        running_loss = 0.0
        running_nat = 0.0
        running_cf = 0.0
        running_kept = 0
        steps = 0

        for x_nat, y, _, pils, bgs, idxs in train_loader:
            x_nat = x_nat.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            x_cf, y_cf = build_cf_batch(
                train_ds=train_ds,
                donor_pools=donor_pools,
                pils=pils,
                ys=y.cpu(),
                bgs=bgs,
                idxs=idxs,
                donor_policy=args.donor_policy,
                max_box_frac=args.max_box_frac,
            )

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda" and args.amp)):
                logits_nat = model(x_nat)
                loss_nat = ce(logits_nat, y)
                if x_cf is not None:
                    x_cf = x_cf.to(device, non_blocking=True)
                    y_cf = y_cf.to(device, non_blocking=True)
                    logits_cf = model(x_cf)
                    loss_cf = ce(logits_cf, y_cf)
                else:
                    loss_cf = torch.zeros((), device=device)
                loss = loss_nat + (args.lambda_cf_ce * loss_cf)

            scaler.scale(loss).backward()
            if args.clip_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            scaler.step(opt)
            scaler.update()

            running_loss += float(loss.item())
            running_nat += float(loss_nat.item())
            running_cf += float(loss_cf.item())
            running_kept += 0 if y_cf is None else int(y_cf.numel())
            steps += 1

        epoch_bar.set_postfix(
            loss=f"{running_loss / max(1, steps):.3f}",
            nat=f"{running_nat / max(1, steps):.3f}",
            cf=f"{running_cf / max(1, steps):.3f}",
            kept=f"{running_kept / max(1, steps):.1f}",
        )

        va_avg, va_worst, va_det = evaluate(model, val_loader, device)
        print(f"[VAL] avg={va_avg:.3f} worst={va_worst:.3f} groups={va_det}")
        if va_worst > best_wga:
            best_wga = va_worst
            best_epoch = epoch
            best_val_avg = va_avg
            best_val_details = va_det
            torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, ckpt_dir / "cfaug_mixed_best.pt")
        torch.save({"model": model.state_dict(), "epoch": epoch, "args": vars(args)}, ckpt_dir / "cfaug_mixed_last.pt")
        sched.step()

    ckpt = torch.load(ckpt_dir / "cfaug_mixed_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    te_avg, te_worst, te_det = evaluate(model, test_loader, device)
    elapsed = time.time() - start
    print(f"[TEST] avg={te_avg:.3f} worst={te_worst:.3f} groups={te_det} epoch={best_epoch} (time {elapsed/60:.1f} min)")

    summary = {
        "output_dir": str(output_dir),
        "best_wga_epoch": best_epoch,
        "best_val_wga": best_wga,
        "best_val_avg": best_val_avg,
        "best_val_groups": {str(k): {"acc": v[0], "count": v[1]} for k, v in (best_val_details or {}).items()},
        "best_test_avg": te_avg,
        "best_test_wga": te_worst,
        "best_test_groups": {str(k): {"acc": v[0], "count": v[1]} for k, v in te_det.items()},
        "best_checkpoint": {
            "path": str((ckpt_dir / "cfaug_mixed_best.pt").resolve()),
            "epoch": best_epoch,
        },
        "elapsed_sec": elapsed,
        "args": vars(args),
    }
    summary_path = output_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[saved] summary -> {summary_path}")


if __name__ == "__main__":
    main()
