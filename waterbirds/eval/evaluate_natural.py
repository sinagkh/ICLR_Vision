#!/usr/bin/env python3
"""Natural-image Waterbirds evaluation with per-example outputs; no segmenter call."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import torchvision.models as models
from datasets import Dataset as ArrowDataset
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EvalData(Dataset):
    def __init__(self, arrow_path: Path, transform):
        self.ds = ArrowDataset.from_file(str(arrow_path))
        self.transform = transform
    def __len__(self): return len(self.ds)
    def __getitem__(self, i):
        ex = self.ds[int(i)]
        y, bg = int(ex["label"]), int(ex["place"])
        return self.transform(ex["image"].convert("RGB")), y, bg, int(i)


def arrow_file(cache: Path, split: str):
    found = sorted(cache.glob(f"grodino___waterbirds/default/*/*/waterbirds-{split}.arrow"))
    if len(found) != 1: raise RuntimeError(found)
    return found[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--arrow-cache", type=Path, required=True)
    ap.add_argument("--backbone-utils-dir", type=Path, required=True)
    ap.add_argument("--split", choices=["validation", "test"], default="test")
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.backbone_utils_dir.resolve()))
    from backbone_utils import build_image_transforms
    transform = build_image_transforms("resnet50", 224, train_aug=False)
    ds = EvalData(arrow_file(args.arrow_cache, args.split), transform)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                        pin_memory=True, persistent_workers=args.num_workers > 0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights=None); model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint.get("model", checkpoint)); model.to(device).eval()
    rows = []
    with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for x, y, bg, idx in loader:
            logits = model(x.to(device, non_blocking=True)).float().cpu()
            probs = logits.softmax(1); pred = logits.argmax(1)
            for j in range(len(y)):
                group = 2 * int(y[j]) + int(bg[j])
                rows.append({"split": args.split, "index": int(idx[j]), "label": int(y[j]),
                             "background": int(bg[j]), "group": group,
                             "logit_0": float(logits[j, 0]), "logit_1": float(logits[j, 1]),
                             "prob_1": float(probs[j, 1]), "prediction": int(pred[j]),
                             "correct": int(pred[j] == y[j])})
    per_group = {}
    for g in range(4):
        subset = [r for r in rows if r["group"] == g]
        per_group[str(g)] = {"accuracy": sum(r["correct"] for r in subset) / len(subset), "n": len(subset)}
    metrics = {
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "split": args.split,
        "n": len(rows),
        "average_accuracy": sum(r["correct"] for r in rows) / len(rows),
        "worst_group_accuracy": min(x["accuracy"] for x in per_group.values()),
        "groups": per_group,
        "natural_inference_calls_segmenter": False
    }
    csv_path = args.output_dir / f"{args.split}_per_example.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (args.output_dir / f"{args.split}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
