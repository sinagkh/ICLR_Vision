#!/usr/bin/env python3
"""Build the held-out PACO evaluation sets.

All objects come from the PACO-LVIS val and test annotation files; none of their images appear in the
training list. Sets:
  part_absent - objects of the five trained classes that the main lists skip because one of the named part
                groups is not annotated on the instance (e.g. hidden by another object, the image border or the
                viewpoint), restricted to instances with at least one annotated part. Used by the hidden-part test.
  unrelated   - objects from every other PACO category (capped per category). Used as spoof carriers.
  lookalike   - objects from a fixed family of related categories of three trained classes (chair, laptop,
                bottle), typed by which of the named part groups PACO annotates on the instance.
Size filters match the main lists: object area >= 0.002 of the image, padded crop <= 0.95 of the image.
Image paths are written relative to --coco-root and annotation files relative to --paco-root.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from paco.src.coco_paths import resolve_coco_image_path  # noqa: E402
from paco.src.common import LISTS  # noqa: E402
from paco.src import steering as rv  # noqa: E402

FAMILIES = {
    "chair": ["stool", "bench"],
    "laptop_computer": ["cellular_telephone", "telephone", "television_set"],
    "bottle": ["cup", "mug", "glass_(drink_container)", "can", "jar", "vase", "soap", "kettle", "carton"],
}
TRAINED = ["bicycle", "chair", "laptop_computer", "car_(automobile)", "bottle"]
UNRELATED_CAP = 60
MIN_OBJ_FRAC = 0.002
MAX_CROP_FRAC = 0.95


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco-root", default=os.environ.get("COCO_ROOT", "."), help="folder with COCO train2017/ and val2017/")
    ap.add_argument("--paco-root", default=os.environ.get("PACO_ROOT", "."), help="folder with paco_lvis_v1_{val,test}.json")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_map = json.loads((LISTS / "part_pair_map.json").read_text())["part_pairs"]
    pairs = {p["object_category_name"]: p for p in pair_map}
    class_index = {n: i for i, n in enumerate(json.loads((LISTS / "class_map.json").read_text())["class_names"])}
    # Part-name aliases of each trained class, as resolved for that class (names after the colon).
    group_names = {
        c: {
            "A": {n.split(":", 1)[1] for n in pairs[c]["part_a_part_names"]},
            "B": {n.split(":", 1)[1] for n in pairs[c]["part_b_part_names"]},
        }
        for c in pairs
    }
    family_of = {o: c for c, objs in FAMILIES.items() for o in objs}
    rows = {"lookalike": [], "part_absent": [], "unrelated": []}
    for split in ("val", "test"):
        ann_path = Path(args.paco_root) / f"paco_lvis_v1_{split}.json"
        data = json.loads(ann_path.read_text())
        cats = {int(c["id"]): c for c in data["categories"]}
        images = {int(i["id"]): i for i in data["images"]}
        by_obj = defaultdict(list)
        for a in data["annotations"]:
            by_obj[int(a.get("obj_ann_id", a["id"]))].append(a)
        for obj_id in sorted(by_obj):
            anns = by_obj[obj_id]
            obj_anns = [a for a in anns if cats[int(a["category_id"])].get("supercategory") == "OBJECT"]
            if not obj_anns:
                continue
            oa = obj_anns[0]
            oname = cats[int(oa["category_id"])]["name"]
            img = images[int(oa["image_id"])]
            w, h = int(img["width"]), int(img["height"])
            if float(oa.get("area", 0.0)) / max(w * h, 1) < MIN_OBJ_FRAC:
                continue
            part_names = sorted({cats[int(a["category_id"])]["name"].split(":", 1)[1] for a in anns if cats[int(a["category_id"])].get("supercategory") == "PART"})
            if oname in TRAINED:
                c = oname
                has_a = bool(set(part_names) & group_names[c]["A"])
                has_b = bool(set(part_names) & group_names[c]["B"])
                if has_a and has_b:
                    continue  # already in the main manifest population
                if not part_names:
                    continue  # no part annotation on this instance: absence is not established
                kind = "absent_A" if (has_b and not has_a) else "absent_B" if (has_a and not has_b) else "absent_both"
                set_name, extra = "part_absent", {"class_name": c, "class_index": class_index[c], "absent_type": kind, "lookalike_for": "", "lookalike_type": ""}
            elif oname in family_of:
                c = family_of[oname]
                has_a = bool(set(part_names) & group_names[c]["A"])
                has_b = bool(set(part_names) & group_names[c]["B"])
                kind = "two_part" if (has_a and has_b) else "single_A" if has_a else "single_B" if has_b else "family_no_target_part"
                set_name, extra = "lookalike", {"class_name": "", "class_index": -1, "absent_type": "", "lookalike_for": c, "lookalike_type": kind}
            else:
                set_name, extra = "unrelated", {"class_name": "", "class_index": -1, "absent_type": "", "lookalike_for": "", "lookalike_type": ""}
            mask = rv.decode_segmentation(oa["segmentation"], h, w)
            if not mask.any():
                continue
            box = rv.mask_crop_box(mask, 0.15, w, h)
            if (box[2] - box[0]) * (box[3] - box[1]) / (w * h) > MAX_CROP_FRAC or box[2] <= box[0] or box[3] <= box[1]:
                continue
            image_path = resolve_coco_image_path(args.coco_root, str(img.get("file_name", "")), int(img["id"]), split)
            if not Path(image_path).exists():
                continue
            rows[set_name].append(
                {
                    "eval_set": set_name,
                    "split": split,
                    "annotation_json": f"paco_lvis_v1_{split}.json",
                    "image_id": int(img["id"]),
                    "image_path": str(Path(image_path).relative_to(args.coco_root)),
                    "image_width": w,
                    "image_height": h,
                    "object_instance_id": int(obj_id),
                    "object_category_name": oname,
                    "object_mask_ref": json.dumps([int(oa["id"])]),
                    "part_names": json.dumps(part_names),
                    **extra,
                }
            )
    # cap unrelated objects per category, deterministic by (split, object id)
    capped, seen = [], Counter()
    for r in sorted(rows["unrelated"], key=lambda r: (r["object_category_name"], r["split"], r["object_instance_id"])):
        if seen[r["object_category_name"]] < UNRELATED_CAP:
            capped.append(r)
            seen[r["object_category_name"]] += 1
    rows["unrelated"] = capped
    fields = list(rows["lookalike"][0].keys())
    summary = {}
    for name, rr in rows.items():
        with (args.out_dir / f"{name}.csv").open("w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rr)
        if name == "lookalike":
            summary[name] = {f"{c}|{t}": n for (c, t), n in sorted(Counter((r["lookalike_for"], r["lookalike_type"]) for r in rr).items())}
            summary[name + "_by_category"] = {f"{r0}|{r1}|{r2}": n for (r0, r1, r2), n in sorted(Counter((r["lookalike_for"], r["object_category_name"], r["lookalike_type"]) for r in rr).items())}
        elif name == "part_absent":
            summary[name] = {f"{c}|{t}": n for (c, t), n in sorted(Counter((r["class_name"], r["absent_type"]) for r in rr).items())}
        else:
            summary[name] = {"total": len(rr), "categories": len(seen)}
    (args.out_dir / "eval_sets_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
