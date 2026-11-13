"""Locate a COCO image from a PACO-LVIS image entry."""
from __future__ import annotations

from pathlib import Path


def resolve_coco_image_path(coco_root: str | Path, file_name: str, image_id: int, split: str | None = None) -> Path:
    root = Path(coco_root)
    candidates = [root / file_name]
    if split:
        candidates.append(root / f"{split}2017" / f"{int(image_id):012d}.jpg")
    candidates.extend(
        [
            root / "train2017" / f"{int(image_id):012d}.jpg",
            root / "val2017" / f"{int(image_id):012d}.jpg",
            root / "images" / "train2017" / f"{int(image_id):012d}.jpg",
            root / "images" / "val2017" / f"{int(image_id):012d}.jpg",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]

