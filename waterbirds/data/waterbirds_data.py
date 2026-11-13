#!/usr/bin/env python3
"""Arrow-backed Waterbirds data providers: natural images, predicted masks, and released CUB masks."""
from __future__ import annotations

import json
import csv
from pathlib import Path
from typing import Optional

from datasets import Dataset as ArrowDataset
from PIL import Image
from torch.utils.data import Dataset


def _arrow_file(cache: Path, split: str) -> Path:
    matches = sorted(cache.glob(f"grodino___waterbirds/default/*/*/waterbirds-{split}.arrow"))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one cached Arrow file for {split}, found {matches}")
    return matches[0].resolve()


class ArrowWaterbirds(Dataset):
    """Drop-in natural Waterbirds dataset for train/train_erm.py."""

    arrow_cache: Path | None = None
    build_image_transforms = None
    build_pil_transforms = None

    def __init__(self, split: str, image_size: int = 224, train_aug: bool = False, backbone: str = "resnet50", **_):
        if self.arrow_cache is None or self.build_image_transforms is None:
            raise RuntimeError("ArrowWaterbirds class configuration was not installed")
        self.split = split
        self.ds = ArrowDataset.from_file(str(_arrow_file(self.arrow_cache, split)))
        self.tf = self.build_image_transforms(backbone=backbone, image_size=image_size, train_aug=train_aug)
        self.tf_pil = self.build_pil_transforms(backbone=backbone, image_size=image_size)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index: int):
        ex = self.ds[int(index)]
        image = ex["image"].convert("RGB")
        y = int(ex["label"])
        bg = int(ex["place"])
        return self.tf(image), y, 2 * y + bg


class PredictedMaskWaterbirds(Dataset):
    """Drop-in replacement for the manifest-backed dataset of train/train_is.py, with predicted masks."""

    arrow_cache: Path | None = None
    prediction_root: Path | None = None
    build_image_transforms = None
    build_pil_transforms = None

    def __init__(self, split: str, image_size: int = 224, train_aug: bool = False, backbone: str = "resnet50", **_):
        if self.arrow_cache is None or self.prediction_root is None or self.build_image_transforms is None:
            raise RuntimeError("PredictedMaskWaterbirds class configuration was not installed")
        canonical = "validation" if split == "eval" else split
        self.split = canonical
        self.ds = ArrowDataset.from_file(str(_arrow_file(self.arrow_cache, canonical)))
        metadata_path = self.prediction_root / canonical / "predictions.jsonl"
        records = [json.loads(line) for line in metadata_path.read_text().splitlines() if line.strip()]
        if len(records) != len(self.ds) or [r["index"] for r in records] != list(range(len(self.ds))):
            raise RuntimeError(f"Prediction metadata does not align with {canonical} Arrow rows")
        self.predictions = records
        self.rows = []
        for i in range(len(self.ds)):
            ex = self.ds[i]
            y, bg = int(ex["label"]), int(ex["place"])
            pred = records[i]
            self.rows.append({
                "label": y,
                "background": bg,
                "group": 2 * y + bg,
                "mask_available": int(pred["status"] == "valid"),
                "row_index": i,
                "prediction_confidence": pred["confidence"],
                "prediction_status": pred["status"],
                "mask_area_fraction": pred["mask_area_fraction"],
            })
        self.tf = self.build_image_transforms(backbone=backbone, image_size=image_size, train_aug=train_aug)
        self.tf_pil = self.build_pil_transforms(backbone=backbone, image_size=image_size)

    def __len__(self):
        return len(self.ds)

    def has_mask(self, index: int) -> bool:
        return self.predictions[int(index)]["status"] == "valid"

    def load_mask(self, index: int) -> Optional[Image.Image]:
        pred = self.predictions[int(index)]
        if pred["status"] != "valid" or not pred["mask_path"]:
            return None
        path = self.prediction_root / pred["mask_path"]
        with Image.open(path) as mask:
            return self.tf_pil(mask.convert("L"))

    def box_for_index(self, index: int):
        # Deliberately never use CUB boxes or detector boxes as a fallback. The
        # renderer computes its box directly from the predicted mask.
        return None

    def __getitem__(self, index: int):
        ex = self.ds[int(index)]
        image = ex["image"].convert("RGB")
        y, bg = int(ex["label"]), int(ex["place"])
        group = 2 * y + bg
        return self.tf(image), y, group, self.tf_pil(image), bg, int(index)


class GroundTruthMaskWaterbirds(Dataset):
    """Released CUB masks for the Arrow images (matched through the official Waterbirds metadata.csv)."""

    arrow_cache: Path | None = None
    official_waterbirds_root: Path | None = None
    segmentation_root: Path | None = None
    build_image_transforms = None
    build_pil_transforms = None

    def __init__(self, split: str, image_size: int = 224, train_aug: bool = False, backbone: str = "resnet50", **_):
        canonical = "validation" if split == "eval" else split
        split_code = {"train": 0, "validation": 1, "test": 2}[canonical]
        self.ds = ArrowDataset.from_file(str(_arrow_file(self.arrow_cache, canonical)))
        with (self.official_waterbirds_root / "metadata.csv").open(newline="") as f:
            all_rows = list(csv.DictReader(f))
        meta = [r for r in all_rows if int(r["split"]) == split_code]
        if len(meta) != len(self.ds):
            raise RuntimeError(f"Official metadata mismatch for {canonical}: {len(meta)} vs {len(self.ds)}")
        self.meta = meta
        self.rows = []
        for i, (ex, row) in enumerate(zip(self.ds, meta)):
            y, bg = int(ex["label"]), int(ex["place"])
            if y != int(row["y"]) or bg != int(row["place"]):
                raise RuntimeError(f"Official metadata row mismatch at {canonical}:{i}")
            mask_path = self.segmentation_root / Path(row["img_filename"]).with_suffix(".png")
            self.rows.append({"label": y, "background": bg, "group": 2 * y + bg,
                              "mask_available": int(mask_path.exists()), "resolved_mask_path": str(mask_path)})
        self.tf = self.build_image_transforms(backbone=backbone, image_size=image_size, train_aug=train_aug)
        self.tf_pil = self.build_pil_transforms(backbone=backbone, image_size=image_size)

    def __len__(self): return len(self.ds)
    def has_mask(self, index: int) -> bool: return bool(self.rows[int(index)]["mask_available"])
    def load_mask(self, index: int) -> Optional[Image.Image]:
        if not self.has_mask(index): return None
        with Image.open(self.rows[int(index)]["resolved_mask_path"]) as mask:
            return self.tf_pil(mask.convert("L"))
    def box_for_index(self, index: int): return None
    def __getitem__(self, index: int):
        ex = self.ds[int(index)]; image = ex["image"].convert("RGB")
        y, bg = int(ex["label"]), int(ex["place"])
        return self.tf(image), y, 2 * y + bg, self.tf_pil(image), bg, int(index)


class PredictedMaskAuditWaterbirds(PredictedMaskWaterbirds):
    def __getitem__(self, index: int):
        x, y, group, pil, bg, _ = super().__getitem__(index)
        return x, y, group, pil, bg


class GroundTruthMaskAuditWaterbirds(GroundTruthMaskWaterbirds):
    def __getitem__(self, index: int):
        x, y, group, pil, bg, _ = super().__getitem__(index)
        return x, y, group, pil, bg


def configure_dataset_class(cls, arrow_cache: Path, original_module, prediction_root: Path | None = None):
    cls.arrow_cache = arrow_cache.resolve()
    cls.build_image_transforms = staticmethod(original_module.build_image_transforms)
    cls.build_pil_transforms = staticmethod(original_module.build_pil_transforms)
    if prediction_root is not None:
        cls.prediction_root = prediction_root.resolve()
    return cls


def configure_ground_truth_class(cls, arrow_cache: Path, original_module, official_root: Path, segmentation_root: Path):
    configure_dataset_class(cls, arrow_cache, original_module)
    cls.official_waterbirds_root = official_root.resolve()
    cls.segmentation_root = segmentation_root.resolve()
    return cls
