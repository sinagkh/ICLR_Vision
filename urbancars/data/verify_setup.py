from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
VENDOR_ROOT = Path(os.environ.get("VENDOR_ROOT", ROOT / "vendor" / "Whac-A-Mole"))
RAW_ROOT = Path(os.environ.get("URBANCARS_RAW_ROOT", ROOT / "data" / "raw"))
EXP_ROOT = Path(os.environ.get("URBANCARS_EXP_ROOT", ROOT / "data" / "exp"))


def _check(path: Path) -> dict[str, object]:
    return {
        "path": str(path),
        "exists": path.exists(),
        "is_dir": path.is_dir() if path.exists() else False,
        "is_file": path.is_file() if path.exists() else False,
    }


def _count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob(pattern))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-only", action="store_true")
    args = parser.parse_args()

    payload: dict[str, object] = {
        "vendor_root": _check(VENDOR_ROOT),
        "raw_root": _check(RAW_ROOT),
        "exp_root": _check(EXP_ROOT),
        "links": {
            "vendor_data_link": str(VENDOR_ROOT / "data"),
            "vendor_exp_link": str(VENDOR_ROOT / "exp"),
        },
        "sources": {
            "stanford_cars_train": _check(RAW_ROOT / "stanford_cars" / "cars_train"),
            "stanford_cars_test": _check(RAW_ROOT / "stanford_cars" / "cars_test"),
            "stanford_devkit": _check(RAW_ROOT / "stanford_cars" / "devkit"),
            "lvis_train_json": _check(RAW_ROOT / "lvis" / "lvis_v1_train.json"),
            "lvis_val_json": _check(RAW_ROOT / "lvis" / "lvis_v1_val.json"),
            "coco_train2017": _check(RAW_ROOT / "coco" / "train2017"),
            "coco_val2017": _check(RAW_ROOT / "coco" / "val2017"),
            "places_root": _check(RAW_ROOT / "places"),
            "maskformer_repo": _check(VENDOR_ROOT / "create_datasets" / "urbancars" / "maskformer"),
            "maskformer_weights": _check(EXP_ROOT / "weights" / "maskformer_panoptic_swin_large.pkl"),
        },
        "counts": {
            "stanford_train_jpg": _count_files(RAW_ROOT / "stanford_cars" / "cars_train", "*.jpg"),
            "stanford_test_jpg": _count_files(RAW_ROOT / "stanford_cars" / "cars_test", "*.jpg"),
            "coco_train_jpg": _count_files(RAW_ROOT / "coco" / "train2017", "*.jpg"),
            "coco_val_jpg": _count_files(RAW_ROOT / "coco" / "val2017", "*.jpg"),
        },
    }

    if not args.workspace_only:
        payload["generated"] = {
            "stanford_train_masks": _count_files(RAW_ROOT / "stanford_cars" / "mask" / "cars_train", "*.png"),
            "stanford_test_masks": _count_files(RAW_ROOT / "stanford_cars" / "mask" / "cars_test", "*.png"),
            "urbancars_train_jpg": _count_files(RAW_ROOT / "urbancars" / "bg-0.95_co_occur_obj-0.95" / "train", "*.jpg"),
            "urbancars_val_jpg": _count_files(RAW_ROOT / "urbancars" / "bg-0.5_co_occur_obj-0.5" / "val", "*.jpg"),
            "urbancars_test_jpg": _count_files(RAW_ROOT / "urbancars" / "bg-0.5_co_occur_obj-0.5" / "test", "*.jpg"),
            "urbancars_train_bg_source_png": _count_files(RAW_ROOT / "urbancars" / "bg-0.95_co_occur_obj-0.95" / "train", "*_bg_source.png"),
            "urbancars_val_bg_source_png": _count_files(RAW_ROOT / "urbancars" / "bg-0.5_co_occur_obj-0.5" / "val", "*_bg_source.png"),
            "urbancars_test_bg_source_png": _count_files(RAW_ROOT / "urbancars" / "bg-0.5_co_occur_obj-0.5" / "test", "*_bg_source.png"),
        }

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
