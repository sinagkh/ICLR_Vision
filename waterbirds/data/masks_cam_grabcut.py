#!/usr/bin/env python3
"""Generate Waterbirds masks from the seed-matched ERM classifier only.

No detector, segmenter, CUB mask/box, place label, or group label is loaded.
Rule: true-class 7x7 CAM; mean+2std sure foreground; Otsu component containing the CAM peak, dilated by one cell;
five GrabCut iterations seeded with these regions (image border fixed as background); fallback to the dilated CAM component.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import scipy.ndimage as ndi
import torch
from torch import nn
from torchvision import models
from datasets import Dataset as ArrowDataset
from PIL import Image
from skimage.filters import threshold_otsu
from torch.utils.data import DataLoader, Dataset


def arrow_file(cache: Path, split: str) -> Path:
    matches = sorted(cache.glob(f"grodino___waterbirds/default/*/*/waterbirds-{split}.arrow"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Arrow file for {split}, found {matches}")
    return matches[0].resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CamRows(Dataset):
    def __init__(self, cache: Path, split: str, transform, pil_transform):
        self.arrow_path = arrow_file(cache, split)
        self.ds = ArrowDataset.from_file(str(self.arrow_path))
        self.transform = transform
        self.pil_transform = pil_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        row = self.ds[int(index)]
        image = row["image"].convert("RGB")
        # Only image and ordinary task label are accessed. In particular the
        # Arrow row's `place` field is never read.
        label = int(row["label"])
        crop = np.asarray(self.pil_transform(image), dtype=np.uint8)
        return self.transform(image), label, int(index), crop


def collate(rows):
    xs, ys, indices, crops = zip(*rows)
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long), list(indices), list(crops)


def component_containing(mask: np.ndarray, point_yx: tuple[int, int]) -> np.ndarray:
    labels, count = ndi.label(mask.astype(bool), structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    label = int(labels[point_yx])
    if label == 0:
        sizes = ndi.sum(mask, labels, index=np.arange(1, count + 1))
        label = int(np.argmax(sizes)) + 1
    return labels == label


def make_mask(job):
    index, crop, cam7 = job
    cv2.setNumThreads(1)
    cam7 = np.asarray(cam7, dtype=np.float32)
    max_pos = tuple(int(v) for v in np.unravel_index(int(np.argmax(cam7)), cam7.shape))
    mean, std = float(cam7.mean()), float(cam7.std())
    sure7 = cam7 >= (mean + 2.0 * std)
    if not sure7.any():
        sure7[max_pos] = True
    if float(cam7.max() - cam7.min()) <= 1e-8:
        probable7 = sure7.copy()
    else:
        try:
            probable7 = cam7 >= float(threshold_otsu(cam7))
        except ValueError:
            probable7 = sure7.copy()
    probable7 |= sure7
    probable7 = component_containing(probable7, max_pos)
    probable7 = ndi.binary_dilation(probable7, structure=np.ones((3, 3), dtype=bool), iterations=1)

    sure = cv2.resize(sure7.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    probable = cv2.resize(probable7.astype(np.uint8), (224, 224), interpolation=cv2.INTER_NEAREST).astype(bool)
    init = np.full((224, 224), cv2.GC_PR_BGD, dtype=np.uint8)
    init[probable] = cv2.GC_PR_FGD
    init[sure] = cv2.GC_FGD
    # The border is an image-only background seed. Do this after foreground
    # assignment so it is unambiguously fixed if a CAM touches the crop edge.
    init[0, :] = cv2.GC_BGD
    init[-1, :] = cv2.GC_BGD
    init[:, 0] = cv2.GC_BGD
    init[:, -1] = cv2.GC_BGD
    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)
    status = "valid_grabcut"
    try:
        cv2.grabCut(
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR), init, None,
            bg_model, fg_model, 5, cv2.GC_INIT_WITH_MASK,
        )
        foreground = (init == cv2.GC_FGD) | (init == cv2.GC_PR_FGD)
        peak224 = (min(223, int((max_pos[0] + 0.5) * 224 / 7)),
                   min(223, int((max_pos[1] + 0.5) * 224 / 7)))
        foreground = component_containing(foreground, peak224)
        if not foreground.any():
            raise RuntimeError("empty GrabCut foreground")
    except Exception:
        foreground = probable
        status = "valid_cam_fallback"

    area = int(foreground.sum())
    ys, xs = np.nonzero(foreground)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    return index, foreground.astype(np.uint8) * 255, status, area, bbox


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--arrow-cache", type=Path, required=True)
    ap.add_argument("--waterbirds-code", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=12)
    ap.add_argument("--refine-workers", type=int, default=16)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(args.waterbirds_code.resolve()))
    from backbone_utils import build_image_transforms, build_pil_transforms

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True

    # Do not invoke a pretrained model constructor here. The complete
    # seed-matched ERM state is loaded below; weights=None guarantees this
    # script cannot accidentally download or import any additional weights.
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    features = None
    def hook(_module, _inputs, output):
        nonlocal features
        features = output
    handle = model.layer4.register_forward_hook(hook)

    dataset = CamRows(
        args.arrow_cache.resolve(), args.split,
        build_image_transforms("resnet50", 224, train_aug=False),
        build_pil_transforms("resnet50", 224),
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None,
        collate_fn=collate,
    )

    split_root = args.output_root.resolve() / f"seed{args.seed}" / args.split
    mask_dir = split_root / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    records = {}
    counts = {"valid_grabcut": 0, "valid_cam_fallback": 0}
    area_fracs = []

    with ThreadPoolExecutor(max_workers=args.refine_workers) as pool, torch.inference_mode():
        for x, y, indices, crops in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            _ = model(x)
            if features is None:
                raise RuntimeError("ResNet feature hook did not run")
            weights = model.fc.weight[y].unsqueeze(-1).unsqueeze(-1)
            cams = torch.relu((features * weights).sum(dim=1))
            lo = cams.amin(dim=(1, 2), keepdim=True)
            hi = cams.amax(dim=(1, 2), keepdim=True)
            cams = ((cams - lo) / (hi - lo).clamp_min(1e-8)).float().cpu().numpy()
            jobs = [(int(i), crop, cam) for i, crop, cam in zip(indices, crops, cams)]
            for index, mask, status, area, bbox in pool.map(make_mask, jobs):
                rel = Path(f"seed{args.seed}") / args.split / "masks" / f"{index:06d}.png"
                Image.fromarray(mask, mode="L").save(args.output_root.resolve() / rel, optimize=True)
                frac = area / float(224 * 224)
                counts[status] += 1
                area_fracs.append(frac)
                records[index] = {
                    "split": args.split,
                    "index": index,
                    "seed": args.seed,
                    "label_source": "ordinary_waterbirds_task_label",
                    "model": "seed_matched_waterbirds_erm_resnet50_true_class_cam",
                    "coordinate_space": "eval_crop_224",
                    "status": "valid",
                    "refinement_status": status,
                    "confidence": None,
                    "box_xyxy": bbox,
                    "mask_path": rel.as_posix(),
                    "mask_area_pixels": area,
                    "mask_area_fraction": frac,
                }
    handle.remove()
    if sorted(records) != list(range(len(dataset))):
        raise RuntimeError("Generated records do not cover the Arrow rows exactly")
    metadata = split_root / "predictions.jsonl"
    with metadata.open("w", encoding="utf-8") as out:
        for index in range(len(dataset)):
            out.write(json.dumps(records[index], sort_keys=True) + "\n")

    stats = np.asarray(area_fracs, dtype=np.float64)
    provenance = {
        "experiment": "Waterbirds self-localized CAM+GrabCut masks",
        "seed": args.seed,
        "split": args.split,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint.resolve()),
        "arrow_file": str(dataset.arrow_path),
        "arrow_file_sha256": sha256(dataset.arrow_path),
        "uses_cub_masks_or_boxes": False,
        "uses_pretrained_detector_or_segmenter": False,
        "uses_place_or_group_labels": False,
        "uses_task_labels": True,
        "common_backbone_initialization": "ImageNet classification initialization inherited through the seed-matched ERM",
        "mask_rule": "true-class 7x7 CAM; mean+2std sure FG; Otsu max component; one-cell dilation; 5-iteration CAM-seeded GrabCut; max-component fallback",
        "counts": counts,
        "total": len(dataset),
        "coverage": len(records) / len(dataset),
        "area_fraction": {
            "mean": float(stats.mean()), "std": float(stats.std()),
            "min": float(stats.min()), "p05": float(np.quantile(stats, 0.05)),
            "median": float(np.median(stats)), "p95": float(np.quantile(stats, 0.95)),
            "max": float(stats.max()),
        },
        "seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "opencv": cv2.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "batch_size": args.batch_size,
        "loader_workers": args.num_workers,
        "refine_workers": args.refine_workers,
        "environment": {key: os.environ.get(key) for key in ("HF_HOME", "HF_DATASETS_OFFLINE")},
    }
    (split_root / "generation_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
