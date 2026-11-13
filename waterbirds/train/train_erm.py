#!/usr/bin/env python3
"""
train_erm.py  (CUDA-optimized ERM baseline)

- Fine-tunes an ImageNet-pretrained backbone (default: ResNet-50)
- Tracks Average Accuracy and Worst-Group Accuracy (WGA) on val+test
- Mixed precision (torch.amp) + GradScaler for speed
- Saves the best checkpoint by VAL WGA to ./checkpoints/erm_best.pt
- Prints per-group accuracies and counts
"""

import argparse, os, random, time, json
from pathlib import Path
import contextlib

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import timm
from datasets import load_dataset
from tqdm import tqdm

from backbone_utils import (
    build_image_transforms,
    build_pil_transforms,
    set_backbone_trainable,
    split_backbone_and_head_params,
)

# --------------------------- utilities ---------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def worst_group_metrics(logits: torch.Tensor, ys: torch.Tensor, groups: torch.Tensor, num_groups: int = 4):
    """Return (avg_acc, worst_group_acc, details_dict)."""
    pred = logits.argmax(1).cpu().numpy()
    ys = ys.cpu().numpy(); groups = groups.cpu().numpy()
    accs = []
    total_correct = 0
    total_count = 0
    for g in range(num_groups):
        idx = (groups == g)
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

# --------------------------- dataset ---------------------------

class HFWaterbirds(Dataset):
    """
    Wrap HuggingFace 'grodino/waterbirds' with torchvision transforms.
    Assumes integer label (bird type) and integer background indicator present.
    """
    def __init__(self, split: str, image_size: int = 224, train_aug: bool = False, backbone: str = "resnet50"):
        self.ds = load_dataset("grodino/waterbirds", split=split)
        self.features = self.ds.features

        # infer keys
        self.img_k = 'image' if 'image' in self.features else next((k for k in self.features if 'image' in k.lower()), None)
        self.label_k = next((k for k, v in self.features.items()
                             if ('label' in k or 'bird' in k) and str(getattr(v, 'dtype', v)).startswith('int')), None)
        self.bg_k = next((k for k, v in self.features.items()
                          if ('place' in k or 'background' in k or k in ['bg','water','land']) and str(getattr(v, 'dtype', v)).startswith('int')), None)

        if not (self.img_k and self.label_k and self.bg_k):
            raise RuntimeError(f"Could not infer keys from dataset features: {list(self.features.keys())}")

        self.tf = build_image_transforms(backbone=backbone, image_size=image_size, train_aug=train_aug)
        self.tf_pil = build_pil_transforms(backbone=backbone, image_size=image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        img = ex[self.img_k].convert('RGB')
        x = self.tf(img)
        y = int(ex[self.label_k])      # 0=landbird, 1=waterbird
        bg = int(ex[self.bg_k])        # 0=land, 1=water
        group = 2*y + bg               # {0..3}
        return x, y, group

def make_loaders(bs: int, size: int, num_workers: int, backbone: str, train_aug: bool):
    tr = HFWaterbirds('train', image_size=size, train_aug=train_aug, backbone=backbone)
    va = HFWaterbirds('validation', image_size=size, train_aug=False, backbone=backbone)
    te = HFWaterbirds('test', image_size=size, train_aug=False, backbone=backbone)
    loader = lambda d, sh: DataLoader(
        d, batch_size=bs, shuffle=sh, num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0), prefetch_factor=(2 if num_workers > 0 else None)
    )
    return loader(tr, True), loader(va, False), loader(te, False)

# --------------------------- model ---------------------------

def build_backbone(name: str, num_classes: int = 2, image_size: int = 224):
    name = name.lower()
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    elif name == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    elif name == 'vit_b_16':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m
    elif name == 'vit_base_patch14_dinov2':
        return timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=num_classes, img_size=image_size)
    else:
        raise ValueError(f"Unknown backbone: {name}")

# --------------------------- training / eval ---------------------------

def autocast_context(device):
    if device.type == 'cuda':
        return torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True)
    # CPU/MPS paths don’t benefit much; use a no-op context
    return contextlib.nullcontext()

def evaluate(model, dl, device):
    model.eval()
    all_logits, all_y, all_g = [], [], []
    with torch.inference_mode(), autocast_context(device):
        for x, y, g in dl:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            all_logits.append(logits.cpu()); all_y.append(y); all_g.append(g)
    logits = torch.cat(all_logits); ys = torch.cat(all_y); groups = torch.cat(all_g)
    avg, worst, details = worst_group_metrics(logits, ys, groups)
    return avg, worst, details

def main(args):
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}, CUDA:{torch.cuda.is_available()} | cudnn.benchmark=True")
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    train_loader, val_loader, test_loader = make_loaders(
        args.batch_size,
        args.image_size,
        args.num_workers,
        backbone=args.backbone,
        train_aug=args.train_aug,
    )
    model = build_backbone(args.backbone, num_classes=2, image_size=args.image_size).to(device)

    # Loss / Optim / Sched
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
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda' and args.amp))

    output_dir = Path(args.output_dir).resolve() if args.output_dir else Path.cwd()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_wga = -1.0
    best_epoch = None
    best_val_avg = None
    best_val_details = None
    start = time.time()

    epoch_bar = tqdm(range(1, args.epochs + 1), desc="Epochs", ncols=100, leave=False)
    for epoch in epoch_bar:
        model.train()
        freeze_backbone = (epoch <= args.freeze_backbone_epochs)
        set_backbone_trainable(model, train_backbone=not freeze_backbone)
        running_loss = 0.0
        steps = 0
        for x, y, g in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if device.type == 'cuda' and args.amp:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
                    logits = model(x)
                    loss = ce(logits, y)
                scaler.scale(loss).backward()
                if args.clip_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(opt); scaler.update()
            else:
                logits = model(x)
                loss = ce(logits, y)
                loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                opt.step()
            running_loss += float(loss.item())
            steps += 1
        epoch_bar.set_postfix(loss=f"{running_loss/max(1,steps):.3f}")

        # Validate
        va_avg, va_worst, va_det = evaluate(model, val_loader, device)
        print(f"[VAL] avg={va_avg:.3f} worst={va_worst:.3f} groups={va_det}")
        if va_worst > best_wga:
            best_wga = va_worst
            best_epoch = epoch
            best_val_avg = va_avg
            best_val_details = va_det
            torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, ckpt_dir / "erm_best.pt")
        torch.save({'model': model.state_dict(), 'epoch': epoch, 'args': vars(args)}, ckpt_dir / "erm_last.pt")
        sched.step()

    # Test best checkpoint
    ckpt = torch.load(ckpt_dir / "erm_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    te_avg, te_worst, te_det = evaluate(model, test_loader, device)
    elapsed = time.time() - start
    print(f"[TEST] avg={te_avg:.3f} worst={te_worst:.3f} groups={te_det} epoch={best_epoch}  (time {elapsed/60:.1f} min)")

    summary = {
        "output_dir": str(output_dir),
        "best_wga_epoch": best_epoch,
        "best_val_wga": best_wga,
        "best_checkpoint": {
            "path": str((ckpt_dir / "erm_best.pt").resolve()),
            "epoch": best_epoch,
            "val_average_accuracy": best_val_avg,
            "val_worst_group_accuracy": best_wga,
            "val_groups": best_val_details,
            "test_average_accuracy": te_avg,
            "test_worst_group_accuracy": te_worst,
            "test_groups": te_det,
        },
        "args": vars(args),
    }
    summary_path = output_dir / "train_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"[saved] summary -> {summary_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="resnet50",
                    choices=["resnet50", "resnet18", "convnext_tiny", "vit_b_16", "vit_base_patch14_dinov2"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap.add_argument("--train_aug", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--amp", action="store_true", default=True,
                    help="Use mixed precision on CUDA (enabled by default).")
    ap.add_argument("--output_dir", type=str, default=None,
                    help="Optional run directory. Checkpoints and train_summary.json are written here.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    main(args)
