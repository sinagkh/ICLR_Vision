#!/usr/bin/env python3
"""Generate one COCO bird instance mask per Waterbirds image (highest-scoring COCO "bird" instance).

This script never reads Waterbirds class, place/group, CUB box, or CUB mask metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from datasets import Dataset as ArrowDataset
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)
from torchvision.transforms.functional import pil_to_tensor


class ImagesOnly(Dataset):
    def __init__(self, split: str, arrow_cache: Path):
        matches = sorted(arrow_cache.glob(f"grodino___waterbirds/default/*/*/waterbirds-{split}.arrow"))
        if len(matches) != 1:
            raise RuntimeError(f"Expected one cached Arrow file for {split}, found: {matches}")
        self.arrow_path = matches[0].resolve()
        self.ds = ArrowDataset.from_file(str(self.arrow_path))

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        image = self.ds[index]["image"].convert("RGB")
        return pil_to_tensor(image).float().div_(255.0), int(index), image.size


def collate(batch):
    images, indices, sizes = zip(*batch)
    return list(images), list(indices), list(sizes)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--arrow-cache", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn_v2(weights=weights).to(device).eval()

    weight_name = Path(weights.url).name
    weight_path = Path(torch.hub.get_dir()) / "checkpoints" / weight_name
    run_started = time.time()
    all_counts = {}

    for split in args.splits:
        split_started = time.time()
        mask_dir = output_root / split / "masks"
        mask_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = output_root / split / "predictions.jsonl"
        dataset = ImagesOnly(split, args.arrow_cache.resolve())
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
            collate_fn=collate,
        )
        counts = {"total": len(dataset), "valid": 0, "no_bird_detection": 0, "empty_mask": 0}
        with metadata_path.open("w", encoding="utf-8") as out, torch.inference_mode():
            for images, indices, sizes in loader:
                predictions = model([image.to(device, non_blocking=True) for image in images])
                for pred, index, size in zip(predictions, indices, sizes):
                    labels = pred["labels"]
                    bird_indices = torch.nonzero(labels == 16, as_tuple=False).flatten()
                    row = {
                        "split": split,
                        "index": index,
                        "image_width": int(size[0]),
                        "image_height": int(size[1]),
                        "model": "torchvision.maskrcnn_resnet50_fpn_v2.COCO_V1",
                        "detector_default_score_threshold": 0.05,
                        "mask_threshold": 0.5,
                        "status": None,
                        "confidence": None,
                        "box_xyxy": None,
                        "mask_path": None,
                        "mask_area_pixels": 0,
                        "mask_area_fraction": 0.0,
                    }
                    if bird_indices.numel() == 0:
                        row["status"] = "no_bird_detection"
                        counts["no_bird_detection"] += 1
                    else:
                        candidate_scores = pred["scores"][bird_indices]
                        selected = int(bird_indices[int(torch.argmax(candidate_scores).item())].item())
                        mask = pred["masks"][selected, 0] >= 0.5
                        area = int(mask.sum().item())
                        row["confidence"] = float(pred["scores"][selected].item())
                        row["box_xyxy"] = [float(v) for v in pred["boxes"][selected].tolist()]
                        row["mask_area_pixels"] = area
                        row["mask_area_fraction"] = area / float(size[0] * size[1])
                        if area == 0:
                            row["status"] = "empty_mask"
                            counts["empty_mask"] += 1
                        else:
                            rel = Path(split) / "masks" / f"{index:06d}.png"
                            Image.fromarray(mask.cpu().numpy().astype(np.uint8) * 255, mode="L").save(output_root / rel)
                            row["status"] = "valid"
                            row["mask_path"] = rel.as_posix()
                            counts["valid"] += 1
                    out.write(json.dumps(row, sort_keys=True) + "\n")
        counts["coverage"] = counts["valid"] / counts["total"]
        counts["seconds"] = time.time() - split_started
        all_counts[split] = counts
        print(json.dumps({"completed_split": split, **counts}, sort_keys=True), flush=True)

    provenance = {
        "model": "torchvision.maskrcnn_resnet50_fpn_v2.COCO_V1",
        "weights_url": weights.url,
        "weights_path": str(weight_path.resolve()),
        "weights_sha256": sha256(weight_path),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "python": platform.python_version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "selection_rule": "highest-score COCO bird (label 16), built-in score threshold 0.05, mask threshold 0.5",
        "counts": all_counts,
        "total_seconds": time.time() - run_started,
        "environment": {k: os.environ.get(k) for k in ["HF_HOME", "HF_DATASETS_OFFLINE"]},
    }
    (output_root / "generation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
