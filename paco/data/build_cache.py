#!/usr/bin/env python3
"""Cache 224px object crops and crop-space masks for a PACO instance list.

Image paths in the lists are relative to --coco-root ($COCO_ROOT) and annotation files relative to
--paco-root ($PACO_ROOT). Write each split to $DATA_ROOT/paco_cache/<split> (see the README).

masks.npy bit flags: 1 = object mask, 2 = part group A (dilated 1px), 4 = part group B (dilated 1px).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paco.src import steering as rv  # noqa: E402

B = rv.CornerBuilder()
ROWS: list[dict] = []
COCO_ROOT = Path(".")
PACO_ROOT = Path(".")


def work(i: int):
    r = ROWS[i]
    anns = rv.ann_index(str(PACO_ROOT / r["annotation_json"]))
    h, w = int(float(r["image_height"])), int(float(r["image_width"]))
    obj = rv.union_refs(anns, json.loads(r["object_mask_ref"]), h, w)
    with Image.open(COCO_ROOT / r["image_path"]) as im:
        im = im.convert("RGB")
    box = B.crop_box(r, obj)
    crop = np.asarray(B.natural(im, box), dtype=np.uint8)
    m = rv.resize_bool_mask(obj[box[1]:box[3], box[0]:box[2]], B.size).astype(np.uint8)
    if r.get("part_a_mask_ref"):
        a = rv.union_refs(anns, json.loads(r["part_a_mask_ref"]), h, w)
        b = rv.union_refs(anns, json.loads(r["part_b_mask_ref"]), h, w)
        m |= B.crop_mask(a, box).astype(np.uint8) << 1
        m |= B.crop_mask(b, box).astype(np.uint8) << 2
    return i, crop, m, box


def main() -> None:
    global ROWS, COCO_ROOT, PACO_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--coco-root", type=Path, default=Path(os.environ.get("COCO_ROOT", ".")), help="folder with COCO train2017/ and val2017/")
    ap.add_argument("--paco-root", type=Path, default=Path(os.environ.get("PACO_ROOT", ".")), help="folder with paco_lvis_v1_{train,val,test}.json")
    args = ap.parse_args()
    COCO_ROOT, PACO_ROOT = args.coco_root, args.paco_root
    ROWS = list(csv.DictReader(open(args.manifest)))
    for aj in sorted({r["annotation_json"] for r in ROWS}):
        rv.ann_index(str(PACO_ROOT / aj))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    n = len(ROWS)
    crops = np.lib.format.open_memmap(args.out_dir / "crops.npy", mode="w+", dtype=np.uint8, shape=(n, 224, 224, 3))
    masks = np.lib.format.open_memmap(args.out_dir / "masks.npy", mode="w+", dtype=np.uint8, shape=(n, 224, 224))
    boxes = np.zeros((n, 4), dtype=np.int32)
    with Pool(args.workers) as pool:
        for k, (i, c, m, box) in enumerate(pool.imap_unordered(work, range(n), chunksize=16)):
            crops[i], masks[i], boxes[i] = c, m, box
            if k % 2000 == 0:
                print(f"{k}/{n}", flush=True)
    crops.flush()
    masks.flush()
    np.save(args.out_dir / "boxes.npy", boxes)
    (args.out_dir / "source.json").write_text(json.dumps({"manifest": str(args.manifest), "n": n}))
    print("done", n)


if __name__ == "__main__":
    main()
