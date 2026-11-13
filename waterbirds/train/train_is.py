#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_is.py

Fine-tunes a Waterbirds ERM checkpoint with Δ²(Object×Background) steering that:
  • builds four photorealistic corners via dataset masks, mask inpainting, and donor cleansing.
  • filters anchors with quality gates (object > ε, background < τ) and label-free ranked banks if desired.
  • supports AMP toggles, margin/nomargin Δ², suppress/encourage modes with configurable |Δ²| target K, and guard-weighted hinge penalties.
  • balances anchors per class and exposes `--anchor_strategy` to replay audit-selected anchors.

Commands: see configs/ (e.g. configs/03_is_resnet50.sh).
"""

import argparse, csv, os, random, math, sys, json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Tuple, List, Optional
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
import timm
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageChops
from tqdm import tqdm

from backbone_utils import build_image_transforms, build_pil_transforms, set_backbone_trainable

# ----------------------------- Dataset -----------------------------

_DEFAULT_REAL_ROOT = Path(os.environ.get("DATA_ROOT", ".")) / "waterbirds_manifest"

class HFWaterbirds(Dataset):
    """
    Manifest-backed Waterbirds dataset (images, labels, places, bird masks).

    Returns:
      x: tensor [C,H,W] (aug or eval transform)
      y: int label {0,1}
      group: 2*y + bg  (g∈{0..3})
      pil_img: center-cropped PIL (for editing)
      bg: int {0,1}
      idx: int index inside this split
    """
    def __init__(self, split: str, image_size: int = 224, train_aug: bool = False, data_root: str | Path | None = None, backbone: str = "resnet50"):
        self.split = split
        root = Path(data_root) if data_root is not None else _DEFAULT_REAL_ROOT
        split_map = {
            "train": "waterbirds_train.csv",
            "validation": "waterbirds_eval.csv",
            "eval": "waterbirds_eval.csv",
            "test": "waterbirds_test.csv",
        }
        if split not in split_map:
            raise ValueError(f"Unknown split '{split}'. Expected one of {sorted(split_map)}.")
        self.csv_path = (root / split_map[split]).resolve()
        self.base_dir = self.csv_path.parent
        self.rows: list[dict[str, object]] = []
        with self.csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row_index, raw_row in enumerate(reader):
                row = dict(raw_row)
                bird_box_raw = row.get("bird_box")
                if isinstance(bird_box_raw, str) and bird_box_raw:
                    try:
                        row["bird_box"] = json.loads(bird_box_raw)
                    except json.JSONDecodeError:
                        row["bird_box"] = None
                else:
                    row["bird_box"] = None
                row["label"] = int(row["label"])
                row["background"] = int(row["background"])
                row["group"] = int(row["group"])
                row["mask_available"] = int(row.get("mask_available", "0") or 0)
                row["row_index"] = row_index
                row["resolved_image_path"] = str((self.base_dir / str(row["image_path"])).resolve())
                mask_value = str(row.get("bird_mask_path", "") or "")
                row["resolved_mask_path"] = str((self.base_dir / mask_value).resolve()) if mask_value else ""
                self.rows.append(row)
        self.tf = build_image_transforms(backbone=backbone, image_size=image_size, train_aug=train_aug)
        self.tf_pil = build_pil_transforms(backbone=backbone, image_size=image_size)

    def __len__(self): return len(self.rows)

    def has_mask(self, i: int) -> bool:
        row = self.rows[int(i)]
        return bool(int(row.get("mask_available", 0))) and bool(row.get("resolved_mask_path"))

    def load_mask(self, i: int) -> Optional[Image.Image]:
        row = self.rows[int(i)]
        mask_path = str(row.get("resolved_mask_path", "") or "")
        if not mask_path:
            return None
        try:
            with Image.open(mask_path) as mask:
                return self.tf_pil(mask.convert("L"))
        except FileNotFoundError:
            return None

    def box_for_index(self, i: int) -> Optional[List[int]]:
        row = self.rows[int(i)]
        box = row.get("bird_box")
        if not isinstance(box, list) or len(box) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def __getitem__(self, i: int):
        row = self.rows[int(i)]
        with Image.open(str(row["resolved_image_path"])) as image:
            pil = image.convert('RGB')
        y = int(row["label"])
        bg = int(row["background"])
        g = int(row["group"])
        return self.tf(pil), y, g, self.tf_pil(pil), bg, int(i)

def collate_with_pil(batch):
    xs, ys, gs, pils, bgs, idxs = zip(*batch)
    xs = torch.stack(xs, dim=0)
    ys = torch.as_tensor(ys, dtype=torch.long)
    gs = torch.as_tensor(gs, dtype=torch.long)
    bgs = torch.as_tensor(bgs, dtype=torch.long)
    idxs = torch.as_tensor(idxs, dtype=torch.long)
    return xs, ys, gs, list(pils), bgs, idxs

# -------------------------- Donors & Masks -------------------------

def build_donor_pools(ds: HFWaterbirds) -> Dict[object, List[int]]:
    idx_by_bg = defaultdict(list)
    all_valid: List[int] = []
    for i, row in enumerate(ds.rows):
        if not ds.has_mask(i):
            continue
        idx = int(i)
        idx_by_bg[int(row["background"])].append(idx)
        all_valid.append(idx)
    idx_by_bg["all"] = all_valid
    return idx_by_bg

def feathered_paste(dst: Image.Image, src_crop: Image.Image, box: Tuple[int,int,int,int], feather: int = 4):
    x1,y1,x2,y2 = box
    w, h = x2-x1, y2-y1
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rectangle([0,0,w,h], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    if src_crop.size != (w,h):
        src_crop = src_crop.resize((w,h), Image.BICUBIC)
    out = dst.copy()
    out.paste(src_crop, (x1,y1), mask)
    return out

def blur_inside_box(pil_img: Image.Image, box: Optional[List[int]], radius=5):
    if box is None:
        return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    x1,y1,x2,y2 = box
    region = pil_img.crop((x1,y1,x2,y2)).filter(ImageFilter.GaussianBlur(radius=radius))
    return feathered_paste(pil_img, region, (x1,y1,x2,y2), feather=3)

def swap_background_feathered(pil_img: Image.Image, donor_pil: Image.Image, box: Optional[List[int]]):
    if box is None:
        W,H = pil_img.size; x=W//2
        out = donor_pil.copy(); out.paste(pil_img.crop((0,0,x,H)), (0,0)); return out
    x1,y1,x2,y2 = box
    crop = pil_img.crop((x1,y1,x2,y2))
    return feathered_paste(donor_pil, crop, (x1,y1,x2,y2), feather=4)

def hist_match(src: Image.Image, ref: Image.Image):
    src_gray, ref_gray = ImageOps.grayscale(src), ImageOps.grayscale(ref)
    src_np = np.asarray(src_gray).ravel(); ref_np = np.asarray(ref_gray).ravel()
    s_vals, s_idx, s_cnt = np.unique(src_np, return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(ref_np, return_counts=True)
    s_q = np.cumsum(s_cnt)/src_np.size; r_q = np.cumsum(r_cnt)/ref_np.size
    interp = np.interp(s_q, r_q, r_vals)
    matched = interp[s_idx].reshape(src_gray.size[1], src_gray.size[0]).astype(np.uint8)
    matched_rgb = Image.merge("RGB", [Image.fromarray(matched)]*3)
    return Image.blend(src, matched_rgb, alpha=0.25)

def _box_iou(a, b, eps=1e-6):
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + eps
    return inter / union

# ---------- Helpers: masks, fills, and donor cleansing ----------

def _try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

CV2 = _try_import_cv2()
if CV2 is None:
    print("[warn] OpenCV not available; GrabCut falls back to feathered rectangles.")

def ring_stats(pil: Image.Image, box, ring=8):
    x1,y1,x2,y2 = box; W,H = pil.size
    x1r=max(0,x1-ring); y1r=max(0,y1-ring); x2r=min(W,x2+ring); y2r=min(H,y2+ring)
    arr = np.asarray(pil.crop((x1r,y1r,x2r,y2r))).astype(np.float32)
    mean = arr.mean(axis=(0,1)); std = arr.std(axis=(0,1))
    return mean, std

def fill_box_mean(pil: Image.Image, box, jitter_std=0.05):
    mean, std = ring_stats(pil, box)
    x1,y1,x2,y2 = box; w,h = x2-x1, y2-y1
    base = np.clip(np.random.normal(mean, np.maximum(1.0, std*jitter_std), size=(h,w,3)), 0, 255).astype(np.uint8)
    patch = Image.fromarray(base)
    return feathered_paste(pil, patch, box, feather=6)

def _border_l2(a, b, bwidth=2):
    # expect HxWx3 uint8 arrays of same shape
    e  = 0.0
    e += np.mean((a[:bwidth,:,:]-b[:bwidth,:,:])**2)
    e += np.mean((a[-bwidth:,:,:]-b[-bwidth:,:,:])**2)
    e += np.mean((a[:, :bwidth,:]-b[:, :bwidth,:])**2)
    e += np.mean((a[:, -bwidth:,:]-b[:, -bwidth:,:])**2)
    return e

def sample_bg_patch(pil: Image.Image, box, tries=12):
    W,H = pil.size; x1,y1,x2,y2 = box; w,h = x2-x1, y2-y1
    full = np.asarray(pil)
    bx = full[y1:y2, x1:x2]
    best, best_cost = None, 1e9

    def _overlap(a, b):
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)

    for _ in range(tries):
        max_rx = max(0, W - w)
        max_ry = max(0, H - h)
        rx = random.randint(0, max_rx)
        ry = random.randint(0, max_ry)
        if _overlap((rx, ry, rx+w, ry+h), (x1, y1, x2, y2)):
            continue
        cand = full[ry:ry+h, rx:rx+w]
        cost = _border_l2(cand, bx, bwidth=2)
        if cost < best_cost:
            best, best_cost = (rx,ry), cost
    if best is None:
        return None
    rx,ry = best
    return pil.crop((rx,ry,rx+w,ry+h))

def fill_box_patch(pil: Image.Image, box):
    p = sample_bg_patch(pil, box)
    if p is None:  # fallback
        return fill_box_mean(pil, box)
    return feathered_paste(pil, p, box, feather=8)

def mask_bbox(mask: Optional[Image.Image]):
    if mask is None:
        return None
    bbox = mask.convert("L").getbbox()
    if bbox is None:
        return None
    x1, y1, x2, y2 = [int(v) for v in bbox]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]

def threshold_mask(mask: Image.Image, threshold: int = 127):
    t = max(0, min(255, int(threshold)))
    return mask.convert("L").point(lambda px: 255 if int(px) >= t else 0)

def mask_crop(mask: Image.Image, box):
    return mask.crop(tuple(box)).convert("L")

def expand_mask(mask: Image.Image, radius: int = 3):
    if radius <= 0:
        return mask.convert("L")
    size = max(3, (int(radius) * 2) + 1)
    return mask.convert("L").filter(ImageFilter.MaxFilter(size=size))

def inpaint_with_mask(pil: Image.Image, mask: Image.Image, radius: int = 4, expand_radius: int = 3, threshold: int = 127):
    mask_l = expand_mask(threshold_mask(mask, threshold=threshold), radius=expand_radius)
    box = mask_bbox(mask_l)
    if box is None:
        return pil.copy()
    if CV2 is not None:
        cv = CV2
        img = np.asarray(pil.convert("RGB"))[:, :, ::-1].copy()
        mask_np = (np.asarray(mask_l) > 0).astype(np.uint8) * 255
        inpainted = cv.inpaint(img, mask_np, float(radius), cv.INPAINT_TELEA)
        return Image.fromarray(inpainted[:, :, ::-1])
    filled = fill_box_patch(pil, box)
    out = pil.copy()
    out.paste(filled.crop(tuple(box)), tuple(box), mask_crop(mask_l, box))
    return out

def paste_with_segment_mask(dst: Image.Image, src_crop: Image.Image, box, segment_mask: Image.Image, blur_radius: int = 0):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    alpha = segment_mask.convert("L")
    if alpha.size != (w, h):
        alpha = alpha.resize((w, h), Image.NEAREST)
    if blur_radius > 0:
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    out = dst.copy()
    if src_crop.size != (w, h):
        src_crop = src_crop.resize((w, h), Image.BICUBIC)
    out.paste(src_crop, (x1, y1), alpha)
    return out

def grabcut_soft_mask(pil: Image.Image, box, iters=3):
    """Return soft mask (PIL 'L') for object inside box; fallback to box feather if cv2 missing."""
    x1,y1,x2,y2 = box
    w,h = x2-x1, y2-y1
    if CV2 is None or w<=1 or h<=1:
        # rectangle mask with feather
        mask = Image.new("L", (w,h), 255).filter(ImageFilter.GaussianBlur(radius=3))
        return mask
    import numpy as np
    cv = CV2
    img = np.array(pil)[:,:,::-1]  # RGB->BGR
    mask = np.zeros(img.shape[:2], np.uint8)
    rect = (x1,y1,w,h)
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)
    try:
        cv.grabCut(img, mask, rect, bgdModel, fgdModel, iters, cv.GC_INIT_WITH_RECT)
    except Exception:
        mask = np.zeros((h,w), np.uint8); mask[:,:] = 255
        return Image.fromarray(mask).filter(ImageFilter.GaussianBlur(radius=3))
    # foreground probabilities
    m = np.where((mask==cv.GC_FGD) | (mask==cv.GC_PR_FGD), 255, 0).astype(np.uint8)
    return Image.fromarray(m).filter(ImageFilter.GaussianBlur(radius=2))

def paste_with_mask(dst: Image.Image, src_crop: Image.Image, box, soft_mask: Image.Image, feather=4):
    # combine the soft mask with a blurred rectangular alpha
    x1,y1,x2,y2 = box; w,h = x2-x1, y2-y1
    rect = Image.new("L", (w,h), 255).filter(ImageFilter.GaussianBlur(radius=feather))
    alpha = ImageChops.lighter(soft_mask, rect) if 'ImageChops' in globals() else rect
    out = dst.copy()
    if src_crop.size != (w,h):
        src_crop = src_crop.resize((w,h), Image.BICUBIC)
    out.paste(src_crop, (x1,y1), alpha)
    return out

def cleanse_donor(donor_pil: Image.Image, donor_mask: Optional[Image.Image]):
    if donor_mask is None:
        return donor_pil
    return inpaint_with_mask(donor_pil, donor_mask, radius=4, expand_radius=3, threshold=127)

# ---------------------------- Utils/Eval ---------------------------

def _set_bn_eval(model):
    touched = []
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)) and m.training:
            m.eval(); touched.append(m)
    return touched

def _restore_bn_train(layers):
    for m in layers: m.train()

def set_seed(s: int):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    elif torch.backends.mps.is_available(): return torch.device("mps")
    else: return torch.device("cpu")

def build_backbone(name: str, num_classes: int = 2, image_size: int = 224):
    name = name.lower()
    if name == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    elif name == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, num_classes); return m
    elif name == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes); return m
    elif name == 'vit_base_patch14_dinov2':
        return timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=num_classes, img_size=image_size)
    else:
        raise ValueError(name)
def load_anchor_bank_from_scores(
    scores_csv: str,
    strategy: str = "ranked_top",   # "ranked_top" or "ranked_stratified"
    top_pct: float = 20.0,
    seed: int = 42,
    on_manifold_only: bool = True,
    seam_l2_max: float = 60.0,
    e_bg_abs_max: float | None = None,
    e_obj_min: float | None = None,
    filter_after_top: bool = False,
    balance_by: str = "none",
):
    """
    Reads the audit CSV (from data/audit_interaction.py --scores_csv) and returns
    a dict with index lists to sample anchors label-free.

    Returns:
      {
        "all": [idx1, idx2, ...]          # for ranked_top
        "qbins": [[...Q1...],[...Q2...], ...]  # for ranked_stratified (abs Δ² quartiles)
      }
    """
    import csv
    rng = random.Random(seed)
    def count_by_key(records: list[dict[str, float]], key: str) -> dict[str, int]:
        counts: dict[str, int] = {}
        for rec in records:
            value = rec.get(key)
            if value is None:
                continue
            counts[str(int(value))] = counts.get(str(int(value)), 0) + 1
        return counts

    rows = []
    with open(scores_csv, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            has_region = r.get("has_mask", r.get("mask_available", r.get("has_box", "1")))
            if int(has_region) == 0:
                continue
            if on_manifold_only and int(r.get("on_manifold", 1)) == 0:
                continue
            idx = int(r["idx"])
            d2_field = r.get("d2_abs") or r.get("d2_mean") or "0.0"
            d2_abs = float(d2_field)
            rows.append({
                "idx": idx,
                "d2_abs": d2_abs,
                "y": int(r.get("y", "0")),
                "group": int(r.get("group", "0")),
                "seam_l2": float(r.get("seam_l2", 0.0)),
                "e_bg": float(r.get("e_bg", "nan")),
                "e_obj": float(r.get("e_obj", "nan")),
            })
    if not rows:
        return {
            "all": [],
            "qbins": [[], [], [], []],
            "_rng": random.Random(seed),
            "_stats": {
                "strategy": strategy,
                "filter_after_top": bool(filter_after_top),
                "balance_by": balance_by,
                "base_count": 0,
                "quality_pass_count": 0,
                "selected_count": 0,
            },
        }

    def passes_quality_filters(rec: dict[str, float]) -> bool:
        if seam_l2_max is not None and rec["seam_l2"] > seam_l2_max:
            return False
        if e_bg_abs_max is not None:
            bg_val = rec["e_bg"]
            if not math.isfinite(bg_val) or abs(bg_val) > e_bg_abs_max:
                return False
        if e_obj_min is not None:
            obj_val = rec["e_obj"]
            if not math.isfinite(obj_val) or obj_val < e_obj_min:
                return False
        return True

    # sort by |Δ²| descending
    rows.sort(key=lambda rec: rec["d2_abs"], reverse=True)

    stats = {
        "strategy": strategy,
        "filter_after_top": bool(filter_after_top),
        "balance_by": balance_by,
        "base_count": len(rows),
        "base_by_y": count_by_key(rows, "y"),
    }

    if filter_after_top:
        source_n = max(1, int(round(len(rows) * (top_pct / 100.0))))
        source_rows = rows[:source_n]
        filtered_rows = [rec for rec in source_rows if passes_quality_filters(rec)]
        stats["top_source_count"] = len(source_rows)
        stats["top_source_by_y"] = count_by_key(source_rows, "y")
        stats["quality_pass_count"] = len(filtered_rows)
    else:
        filtered_rows = [rec for rec in rows if passes_quality_filters(rec)]
        stats["quality_pass_count"] = len(filtered_rows)
    stats["quality_pass_by_y"] = count_by_key(filtered_rows, "y")

    if strategy == "ranked_top":
        if filter_after_top:
            selected_rows = filtered_rows
        else:
            n = max(1, int(round(len(filtered_rows) * (top_pct / 100.0))))
            selected_rows = filtered_rows[:n]
        top = [rec["idx"] for rec in selected_rows]
        rng.shuffle(top)
        stats["selected_count"] = len(top)
        stats["selected_by_y"] = count_by_key(selected_rows, "y")
        by_y: dict[int, list[int]] = {0: [], 1: []}
        for rec in selected_rows:
            by_y.setdefault(int(rec["y"]), []).append(int(rec["idx"]))
        return {
            "all": top,
            "qbins": [[], [], [], []],
            "by_y": by_y,
            "_balance_by": balance_by,
            "_rng": random.Random(seed),
            "_stats": stats,
        }

    elif strategy == "ranked_stratified":
        # split by quartiles of |Δ²|
        rows_for_bins = filtered_rows
        if not rows_for_bins:
            return {"all": [], "qbins": [[], [], [], []], "_rng": random.Random(seed), "_stats": {**stats, "selected_count": 0}}
        d2s = np.array([rec["d2_abs"] for rec in rows_for_bins], dtype=np.float32)
        qs = np.quantile(d2s, [0.25,0.5,0.75])
        bins = [[],[],[],[]]
        for rec in rows_for_bins:
            idx, v = rec["idx"], rec["d2_abs"]
            if v <= qs[0]: bins[0].append(idx)
            elif v <= qs[1]: bins[1].append(idx)
            elif v <= qs[2]: bins[2].append(idx)
            else: bins[3].append(idx)
        for b in bins: rng.shuffle(b)
        stats["selected_count"] = sum(len(b) for b in bins)
        return {"all": [], "qbins": bins, "_rng": random.Random(seed), "_stats": stats}

    else:
        raise ValueError(f"Unknown strategy: {strategy}")
def pick_anchors_from_ranked_bank(
    k: int,
    bank: dict,
    ds: HFWaterbirds,
):
    """
    Samples k anchors using the bank returned by load_anchor_bank_from_scores.
    Returns a list of (pil_img, y, bg, idx) tuples (y/bg are not used by the sampler).
    """
    rng = bank.get("_rng")
    if not isinstance(rng, random.Random):
        rng = random.Random()
        bank["_rng"] = rng
    picked = []
    if bank["all"]:
        balance_by = bank.get("_balance_by", "none")
        if balance_by == "y" and isinstance(bank.get("by_y"), dict):
            by_y = bank["by_y"]
            pool0 = list(by_y.get(0, []))
            pool1 = list(by_y.get(1, []))
            if pool0 and pool1:
                target0 = k // 2
                target1 = k - target0
                picked0 = rng.sample(pool0, min(target0, len(pool0))) if target0 > 0 else []
                picked1 = rng.sample(pool1, min(target1, len(pool1))) if target1 > 0 else []
                picked = picked0 + picked1
                if len(picked) < k:
                    used = set(picked)
                    remaining = [idx for idx in bank["all"] if idx not in used]
                    need = min(k - len(picked), len(remaining))
                    if need > 0:
                        picked += rng.sample(remaining, need)
                rng.shuffle(picked)
            else:
                pool = bank["all"]
                take = min(k, len(pool))
                picked = rng.sample(pool, take) if take > 0 else []
        else:
            pool = bank["all"]
            take = min(k, len(pool))
            picked = rng.sample(pool, take) if take > 0 else []
    else:
        # stratified: draw ~k/4 from each quartile (with fallback)
        qbins = bank["qbins"]
        per = max(1, k // 4)
        sel = []
        for q in range(4):
            take = min(per, len(qbins[q]))
            if take > 0:
                sel += rng.sample(qbins[q], take)
        if len(sel) < k:
            # top up from remaining bins
            flat = [idx for q in qbins for idx in q]
            need = min(k - len(sel), len(flat))
            if need > 0:
                sel += rng.sample(flat, need)
        picked = sel[:k]

    anchors = []
    for j in picked:
        if not ds.has_mask(j):
            continue
        _, yj, _, pilj, bgj, idxj = ds[j]
        anchors.append((pilj, int(yj), int(bgj), int(idxj)))
    return anchors

def worst_group_metrics(logits, ys, groups, num_groups=4):
    pred = logits.argmax(1).cpu().numpy()
    ys = ys.cpu().numpy(); groups = groups.cpu().numpy()
    accs = []
    total_correct = 0
    total_count = 0
    for g in range(num_groups):
        idx = (groups==g)
        if idx.sum()==0: continue
        correct = int((pred[idx]==ys[idx]).sum())
        count = int(idx.sum())
        accs.append((g, correct / count, count))
        total_correct += correct
        total_count += count
    if not accs: return 0.0, 0.0, {}
    worst = min(accs, key=lambda t:t[1])[1]
    avg = float(total_correct / max(1, total_count))
    details = {int(g):(float(a), int(n)) for g,a,n in accs}
    return avg, float(worst), details

def evaluate(model, dl, device, amp_ctx=nullcontext):
    model.eval()
    all_logits, all_y, all_g = [], [], []
    with torch.inference_mode():
        for x,y,g,_,_,_ in dl:
            x = x.to(device, non_blocking=True)
            with amp_ctx():
                all_logits.append(model(x).cpu()); all_y.append(y); all_g.append(g)
    logits = torch.cat(all_logits); ys = torch.cat(all_y); groups = torch.cat(all_g)
    return worst_group_metrics(logits, ys, groups)

def val_pred1_rate(model, dl, device, amp_ctx=nullcontext) -> float:
    model.eval()
    ones = total = 0
    with torch.inference_mode():
        for x,_,_,_,_,_ in dl:
            x = x.to(device, non_blocking=True)
            with amp_ctx():
                p = model(x).argmax(1)
            ones += int((p==1).sum()); total += int(p.numel())
    return ones / max(1, total)

def build_index_maps(ds: HFWaterbirds):
    idx_by_class = {0: [], 1: []}
    idx_by_group = {0: [], 1: [], 2: [], 3: []}
    for i, row in enumerate(ds.rows):
        if not ds.has_mask(i):
            continue
        idx_by_class[int(row["label"])].append(int(i))
        idx_by_group[int(row["group"])].append(int(i))
    return idx_by_class, idx_by_group

def pick_balanced_anchors_from_bank(
    k: int,
    worst_by_class: Dict[int, int],
    group_bias: float,
    idx_by_class: Dict[int, List[int]],
    idx_by_group: Dict[int, List[int]],
    ds: HFWaterbirds,
):
    k0 = k // 2; k1 = k - k0
    def pick_for_class(cls: int, take: int):
        if take <= 0: return []
        worst_g = worst_by_class.get(cls, 2*cls)
        focus = idx_by_group.get(worst_g, [])
        rest  = list({*idx_by_class[cls]} - set(focus))
        t_focus = min(int(round(group_bias * take)), len(focus))
        sel = []
        if t_focus > 0: sel += random.sample(focus, t_focus)
        need = take - len(sel)
        if need > 0:
            pool = rest if len(rest) > 0 else focus
            if len(pool) > 0:
                sel += random.sample(pool, min(need, len(pool)))
        return sel

    idxs = pick_for_class(0, k0) + pick_for_class(1, k1)
    anchors = []
    for j in idxs:
        _, yj, _, pilj, bgj, idxj = ds[j]
        anchors.append((pilj, int(yj), int(bgj), int(idxj)))
    return anchors

# -------------------- Δ² penalty (training-time) -------------------

def d2_penalty(
    model: nn.Module,
    device,
    tf_tensor,  # eval-style tensor transform
    anchors: List[Tuple[Image.Image,int,int,int]],  # (pil, y, bg, idx)
    donor_pools: Dict[object,List[int]],
    donor_ds: HFWaterbirds,
    donor_policy: str,
    mode: str, tau_base: float, K: float, two_donors: bool,
    conf_min: float,
    min_keep_per_class: int, margin_d2: bool,
    eps_obj: float,
    tau_floor_mult: float,
    tau_bg_min: float,
    tau_bg_fac: float,
    tau_bg_guard_min: float,
    tau_bg_guard_fac: float,
    max_box_frac: float,
    amp_ctx=nullcontext,
):
    if len(anchors) == 0:
        z = torch.tensor(0.0, device=device)
        zero = torch.tensor(0.0, device=device)
        return z, z, z, z, z, z, z, 0, 0, 0, zero, zero

    x00_list=[]; x10_list=[]; x01a_list=[]; x11a_list=[]; x01b_list=[]; x11b_list=[]; Y_list=[]
    for (pil_img, y, bg, idx) in anchors:
        anchor_mask = donor_ds.load_mask(int(idx))
        box = mask_bbox(anchor_mask)
        if box is None:
            box = donor_ds.box_for_index(int(idx))
        if box is None:
            continue
        w = box[2]-box[0]; h = box[3]-box[1]
        if (w*h) > max_box_frac * (pil_img.width * pil_img.height):
            continue
        if anchor_mask is None:
            continue

        if donor_policy == "opposite":
            pool = donor_pools.get(1 - bg, [])
        elif donor_policy == "same":
            pool = donor_pools.get(bg, [])
        elif donor_policy == "random":
            pool = [j for j in donor_pools.get("all", []) if j != int(idx)]
        else:
            raise ValueError(f"Unknown donor_policy: {donor_policy}")
        if len(pool) == 0: continue
        # ensure two distinct donors if requested and available
        if two_donors and len(pool) >= 2:
            d_idx_a, d_idx_b = random.sample(pool, 2)
        else:
            choice = random.choice(pool)
            d_idx_a = d_idx_b = choice

        _,_,_, donor_pil_a,_,_ = donor_ds[d_idx_a]
        _,_,_, donor_pil_b,_,_ = donor_ds[d_idx_b]
        donor_mask_a = donor_ds.load_mask(d_idx_a)
        donor_mask_b = donor_ds.load_mask(d_idx_b)
        if donor_mask_a is None or donor_mask_b is None:
            continue

        # ----- Mask-based corners with true dataset bird masks -----
        x00_pil = pil_img
        x10_pil = inpaint_with_mask(pil_img, anchor_mask, radius=4, expand_radius=3, threshold=127)

        # donor cleansing, then color-match to source
        donor_a_clean = cleanse_donor(donor_pil_a, donor_mask_a)
        donor_b_clean = cleanse_donor(donor_pil_b, donor_mask_b)
        donor_a = hist_match(donor_a_clean, pil_img)
        donor_b = hist_match(donor_b_clean, pil_img)

        # object crop + exact segment mask from the dataset
        obj_crop = pil_img.crop(tuple(box))
        segment_mask = threshold_mask(mask_crop(anchor_mask, box), threshold=127)

        # paste object onto background-B
        x01a_pil = paste_with_segment_mask(donor_a, obj_crop, tuple(box), segment_mask, blur_radius=0)
        x01b_pil = paste_with_segment_mask(donor_b, obj_crop, tuple(box), segment_mask, blur_radius=0)

        # no-bird on background-B
        x11a_pil = donor_a_clean
        x11b_pil = donor_b_clean

        x00_list.append(tf_tensor(x00_pil)); x10_list.append(tf_tensor(x10_pil))
        x01a_list.append(tf_tensor(x01a_pil)); x11a_list.append(tf_tensor(x11a_pil))
        x01b_list.append(tf_tensor(x01b_pil)); x11b_list.append(tf_tensor(x11b_pil))
        Y_list.append(y)

    if not x00_list:
        z = torch.tensor(0.0, device=device)
        zero = torch.tensor(0.0, device=device)
        return z, z, z, z, z, z, z, 0, 0, 0, zero, zero

    X00 = torch.stack(x00_list, 0).to(device); X10 = torch.stack(x10_list, 0).to(device)
    X01a= torch.stack(x01a_list,0).to(device); X11a= torch.stack(x11a_list,0).to(device)
    X01b= torch.stack(x01b_list,0).to(device); X11b= torch.stack(x11b_list,0).to(device)
    Y   = torch.as_tensor(Y_list, device=device, dtype=torch.long)

    # freeze BN stats while doing Δ² forwards
    touched = _set_bn_eval(model)
    with amp_ctx():
        logits00 = model(X00); logits10 = model(X10)
        logits01a= model(X01a); logits11a= model(X11a)
        logits01b= model(X01b); logits11b= model(X11b)
    _restore_bn_train(touched)

    # ----- Confidence gating with per-class minimum and balancing -----
    with torch.no_grad():
        prob00 = torch.softmax(logits00.float(), dim=1)[torch.arange(len(Y)), Y]
        # initial threshold
        keep = prob00 >= conf_min
        # ensure per-class minimum
        for cls in (0, 1):
            cls_mask = (Y == cls)
            have = int((keep & cls_mask).sum())
            if have < min_keep_per_class and cls_mask.any():
                idx_cls = torch.nonzero(cls_mask, as_tuple=False).squeeze(1)
                topk = min(min_keep_per_class - have, idx_cls.numel())
                if topk > 0:
                    _, top_rel = prob00[idx_cls].topk(topk)
                    keep[idx_cls[top_rel]] = True
        # balance kept per class (top-k by prob within each class)
        idx0 = torch.nonzero(Y==0, as_tuple=False).squeeze(1)
        idx1 = torch.nonzero(Y==1, as_tuple=False).squeeze(1)
        k_all = int(keep.sum().item())
        if k_all > 1 and idx0.numel()>0 and idx1.numel()>0:
            k0 = min(k_all//2, idx0.numel()); k1 = min(k_all - k0, idx1.numel())
            k0 = max(k0, 1); k1 = max(k1, 1)
            _, top0 = prob00[idx0].topk(k0); _, top1 = prob00[idx1].topk(k1)
            new_keep = torch.zeros_like(keep, dtype=torch.bool)
            new_keep[idx0[top0]] = True; new_keep[idx1[top1]] = True
            keep = new_keep
        if keep.sum() == 0:
            topk = min(2, prob00.numel()); _, top = prob00.topk(topk); keep[top] = True

    if keep.sum() == 0:
        z = torch.tensor(0.0, device=device)
        zero = torch.tensor(0.0, device=device)
        return z, z, z, z, z, z, z, 0, 0, 0, zero, zero

    kept = keep.nonzero(as_tuple=False).squeeze(1)
    kept_count = int(kept.numel()); kept_cls0 = int((Y[keep]==0).sum()); kept_cls1 = int((Y[keep]==1).sum())

    # clamp to safe FP range
    logits00 = logits00[keep].float().clamp_(-20,20); logits10 = logits10[keep].float().clamp_(-20,20)
    logits01a= logits01a[keep].float().clamp_(-20,20); logits11a= logits11a[keep].float().clamp_(-20,20)
    logits01b= logits01b[keep].float().clamp_(-20,20); logits11b= logits11b[keep].float().clamp_(-20,20)
    Y = Y[keep]

    idx = torch.arange(len(Y), device=Y.device); other = 1 - Y
    # Use margins → scale-invariant Δ² and guards
    m00 = logits00[idx, Y] - logits00[idx, other]
    m10 = logits10[idx, Y] - logits10[idx, other]
    m01a= logits01a[idx, Y] - logits01a[idx, other]
    m11a= logits11a[idx, Y] - logits11a[idx, other]
    m01b= logits01b[idx, Y] - logits01b[idx, other]
    m11b= logits11b[idx, Y] - logits11b[idx, other]

    scale = torch.max(
        torch.tensor(1.0, device=logits00.device),
        torch.stack([logits00.abs(), logits10.abs(), logits01a.abs(), logits11a.abs(), logits01b.abs(), logits11b.abs()]).amax()
    )
    tau_bg = max(tau_bg_min, tau_bg_fac*float(scale.detach().item()))
    tau_bg_guard = max(tau_bg_guard_min, tau_bg_guard_fac*float(scale.detach().item()))

    # ----- Quality gates: reject anchors that look off-manifold -----
    e_obj = (m00 - m10)                      # object should help
    m01m  = 0.5*(m01a + m01b)                # background shouldn't
    e_bg  = (m00 - m01m)

    obj_ok = (e_obj > eps_obj)                  # ε object margin
    bg_ok  = (e_bg.abs() < tau_bg)

    keep_mask = obj_ok & bg_ok
    if keep_mask.sum() > 0:
        m00, m10, m01a, m11a, m01b, m11b = [t[keep_mask] for t in (m00,m10,m01a,m11a,m01b,m11b)]
        e_obj = e_obj[keep_mask]; e_bg = e_bg[keep_mask]; m01m = m01m[keep_mask]
    else:
        # drop this mini-batch of anchors; it will contribute 0
        z = torch.tensor(0.0, device=device)
        zero = torch.tensor(0.0, device=device)
        return z, z, z, z, e_obj.detach().mean(), torch.tensor(0.,device=device), torch.tensor(0.,device=device), 0, 0, 0, zero, zero

    d2a = (m11a - m01a) - (m10 - m00)
    d2b = (m11b - m01b) - (m10 - m00)
    d2  = 0.5*(d2a + d2b)

    # scale-aware τ floor
    tau = max(tau_base, tau_floor_mult*float(scale.detach().item()))

    if mode == "suppress":
        p_d2 = torch.clamp(d2.abs() - tau, min=0).pow(2).mean()
    elif mode == "encourage":
        p_d2 = ((d2.abs()+1e-6).log() - math.log(max(K, 1e-6))).pow(2).mean()
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # guards on margins (object should matter; background should not)
    e_obj = (m00 - m10)
    m_obj = torch.clamp(torch.quantile(e_obj.detach(), 0.25), min=0.2)
    p_obj = torch.clamp(m_obj - e_obj, min=0).pow(2).mean()          # small-object penalty
    p_obj_hi = torch.clamp(e_obj - 6.0, min=0).pow(2).mean()         # absurdly large margins
    p_bg   = torch.clamp(e_bg.abs() - tau_bg_guard, min=0).pow(2).mean()

    d2_abs = d2.abs().detach()
    d2_abs_mean = d2_abs.mean()
    d2_abs_p90  = torch.quantile(d2_abs, 0.90)
    e_obj_mean  = e_obj.detach().mean()
    e_bg_abs_mean = e_bg.abs().detach().mean()

    tau_eff = torch.tensor(float(tau), device=device)
    return p_d2, p_obj, p_bg, p_obj_hi, e_obj_mean, d2_abs_mean, d2_abs_p90, kept_count, kept_cls0, kept_cls1, e_bg_abs_mean, tau_eff

# ------------------------------- Main ------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="resnet50", choices=["resnet50","resnet18","convnext_tiny","vit_base_patch14_dinov2"])
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output_dir", type=str, default=".",
                    help="Run output directory. Checkpoints are saved under <output_dir>/checkpoints.")
    ap.add_argument("--data_root", type=str, default=str(_DEFAULT_REAL_ROOT),
                    help="Prepared Waterbirds directory containing waterbirds_{train,eval,test}.csv and mask paths")

    # Load ERM
    ap.add_argument("--load_checkpoint", type=str, default="checkpoints/erm_best.pt")

    # Δ² steering params
    ap.add_argument("--mode", type=str, default="suppress", choices=["suppress","encourage"])
    ap.add_argument("--lambda_d2", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--tau_floor_mult", type=float, default=0.01,
                    help="Effective suppress threshold is max(tau, tau_floor_mult * logit_scale).")
    ap.add_argument("--anchors_per_step", type=int, default=32)
    ap.add_argument("--two_donors", action=argparse.BooleanOptionalAction, default=True,
                    help="Use two distinct donors per anchor when constructing Δ² corners")
    ap.add_argument("--donor_policy", type=str, default="opposite",
                    choices=["opposite", "same", "random"],
                    help="How donor images are selected when constructing the Waterbirds corners.")

    # ===== anchor selection mode & inputs =====
    ap.add_argument("--anchor_strategy", type=str, default="balanced",
                    choices=["balanced","ranked_top","ranked_stratified"],
                    help="balanced uses labels; ranked_* are label-free from audit CSV")
    ap.add_argument("--scores_csv", type=str, default=None,
                    help="CSV from data/audit_interaction.py --scores_csv (required for ranked_*).")
    ap.add_argument("--top_pct", type=float, default=20.0, help="Top %% of |Δ²| for ranked_top")
    ap.add_argument("--ranked_filter_after_top", action=argparse.BooleanOptionalAction, default=False,
                    help="When loading ranked anchors, first take the top_pct |Δ²| source pool and only then apply seam/e_bg/e_obj filters.")
    ap.add_argument("--ranked_balance_by", type=str, default="none", choices=["none", "y"],
                    help="Optional balancing rule when sampling from ranked_top banks.")
    ap.add_argument("--ranked_on_manifold_only", action=argparse.BooleanOptionalAction, default=True,
                    help="When loading ranked anchors, require on_manifold==1")
    ap.add_argument("--ranked_seam_l2_max", type=float, default=60.0,
                    help="Drop ranked anchors whose seam_l2 exceeds this value (set None to disable)")
    ap.add_argument("--ranked_ebg_abs_max", type=float, default=-1.0,
                    help="Optional direct-background filter when loading ranked anchors; negative disables it")
    ap.add_argument("--ranked_eobj_min", type=float, default=-1.0,
                    help="Optional object-effect floor when loading ranked anchors; negative disables it")

    # existing controls
    ap.add_argument("--group_bias", type=float, default=0.7)
    ap.add_argument("--guard_weight", type=float, default=0.30)
    ap.add_argument("--conf_min", type=float, default=0.50)
    ap.add_argument("--min_keep_per_class", type=int, default=4)
    ap.add_argument("--det_cache_frac", type=float, default=0.0,
                    help="Ignored: mask-based corner construction does not use detector boxes")
    ap.add_argument("--box_cache_train", type=str, default=None,
                    help="Ignored: mask-based corner construction does not use detector boxes")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=2)
    ap.add_argument("--margin_d2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--K", type=float, default=0.0, help="Target |Δ²| if mode=encourage")
    ap.add_argument("--eps_obj", type=float, default=0.20,
                    help="Minimum (m00-m10) margin gain required for anchors to be kept")
    ap.add_argument("--tau_bg_min", type=float, default=0.10,
                    help="Floor for allowable background effect |m00-m01| (gate)")
    ap.add_argument("--tau_bg_fac", type=float, default=0.02,
                    help="Scale factor applied to logits magnitude for background tolerance (gate)")
    ap.add_argument("--tau_bg_guard_min", type=float, default=0.20,
                    help="Floor for allowable background effect used in the guard penalty")
    ap.add_argument("--tau_bg_guard_fac", type=float, default=0.02,
                    help="Scale factor for guard background tolerance")
    ap.add_argument("--max_box_frac", type=float, default=0.45,
                    help="Skip anchors whose mask bbox covers more than this fraction of the image")
    ap.add_argument("--lam_obj", type=float, default=0.0,
                    help="Positive reward weight for large e_obj margins")
    ap.add_argument("--lam_obj_floor", type=float, default=0.0,
                    help="Penalty weight for e_obj falling below its EMA floor")
    ap.add_argument("--ema_obj_beta", type=float, default=0.9,
                    help="Decay for the e_obj EMA reference used by lam_obj_floor")
    ap.add_argument("--save_combo_checkpoint", action=argparse.BooleanOptionalAction, default=True,
                    help="Also save a secondary checkpoint that keeps val worst-group close to the best while preferring lower Δ².")
    ap.add_argument("--combo_wga_tolerance", type=float, default=0.01,
                    help="Best-combo checkpoint must have val worst-group within this tolerance of the best-so-far WGA.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "train_summary.json"
    amp_enabled = (device.type == 'cuda' and args.amp)
    if amp_enabled:
        def amp_ctx():
            return torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        amp_ctx = lambda: nullcontext()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    print(f"Device: {device}, AMP={amp_enabled}")

    # Data
    train_ds = HFWaterbirds('train', image_size=args.image_size, train_aug=True, data_root=args.data_root, backbone=args.backbone)
    val_ds   = HFWaterbirds('validation', image_size=args.image_size, train_aug=False, data_root=args.data_root, backbone=args.backbone)
    test_ds  = HFWaterbirds('test', image_size=args.image_size, train_aug=False, data_root=args.data_root, backbone=args.backbone)

    def mk_loader(ds, shuffle):
        loader_kwargs = dict(
            batch_size=args.batch_size,
            shuffle=shuffle,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_with_pil,
        )
        if args.num_workers > 0:
            loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(ds, **loader_kwargs)
    train_loader, val_loader, test_loader = mk_loader(train_ds, True), mk_loader(val_ds, False), mk_loader(test_ds, False)

    # Donors & indices
    donor_pools_train = build_donor_pools(train_ds)
    idx_by_class, idx_by_group = build_index_maps(train_ds)

    # ===== label-free bank from audited CSV (if requested) =====
    ranked_bank = None
    if args.anchor_strategy in {"ranked_top","ranked_stratified"}:
        assert args.scores_csv is not None, "--scores_csv is required for ranked_* strategies"
        seam_cap = args.ranked_seam_l2_max
        if seam_cap is not None and seam_cap < 0:
            seam_cap = None
        e_bg_abs_max = args.ranked_ebg_abs_max if args.ranked_ebg_abs_max >= 0 else None
        e_obj_min = args.ranked_eobj_min if args.ranked_eobj_min >= 0 else None
        ranked_bank = load_anchor_bank_from_scores(
            args.scores_csv,
            strategy=("ranked_top" if args.anchor_strategy=="ranked_top" else "ranked_stratified"),
            top_pct=args.top_pct,
            seed=args.seed,
            filter_after_top=bool(args.ranked_filter_after_top),
            balance_by=args.ranked_balance_by,
            on_manifold_only=bool(args.ranked_on_manifold_only),
            seam_l2_max=seam_cap,
            e_bg_abs_max=e_bg_abs_max,
            e_obj_min=e_obj_min,
        )
        n_all = len(ranked_bank["all"]) if ranked_bank["all"] else sum(len(b) for b in ranked_bank["qbins"])
        ranked_stats = ranked_bank.get("_stats") or {}
        details = []
        if ranked_stats.get("base_count") is not None:
            details.append(f"base={ranked_stats['base_count']}")
        if ranked_stats.get("top_source_count") is not None:
            details.append(f"top_source={ranked_stats['top_source_count']}")
        if ranked_stats.get("quality_pass_count") is not None:
            details.append(f"quality_pass={ranked_stats['quality_pass_count']}")
        if ranked_stats.get("selected_by_y"):
            details.append(f"selected_by_y={ranked_stats['selected_by_y']}")
        details.append(f"filter_stage={'post-top' if ranked_stats.get('filter_after_top') else 'pre-top'}")
        if ranked_stats.get("balance_by"):
            details.append(f"balance_by={ranked_stats['balance_by']}")
        detail_str = f" ({', '.join(details)})" if details else ""
        print(f"[ranked] loaded {n_all} label-free anchor indices from {args.scores_csv}{detail_str}")
        ranked_indices = ranked_bank["all"] if ranked_bank["all"] else [idx for bin_idx in ranked_bank["qbins"] for idx in bin_idx]
        if ranked_indices:
            max_idx = max(ranked_indices)
            if max_idx >= len(train_ds):
                raise ValueError("scores_csv appears to come from a different split. Re-run the auditor with --split train.")

    if args.box_cache_train:
        print("[mask] ignoring --box_cache_train because this trainer uses dataset masks, not detector boxes")
    if args.det_cache_frac > 0:
        print("[mask] ignoring --det_cache_frac because this trainer uses dataset masks, not detector boxes")

    # Model
    model = build_backbone(args.backbone, image_size=args.image_size).to(device)
    if os.path.exists(args.load_checkpoint):
        print(f"Loading ERM checkpoint: {args.load_checkpoint}")
        ckpt = torch.load(args.load_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
    else:
        print(f"WARNING: Checkpoint not found at {args.load_checkpoint}. Starting from scratch.")

    ce = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr*0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    best_ckpt_path = ckpt_dir / "steered_finetuned_best.pt"
    combo_ckpt_path = ckpt_dir / "steered_finetuned_best_combo.pt"
    best_wga = -1.0
    best_wga_epoch = None
    best_combo_d2m = float("inf")
    best_combo_wga = -1.0
    best_combo_epoch = None
    eobj_ema: Optional[float] = None
    worst_by_class = {0: 1, 1: 2}  # reasonable init

    def lam_d2_now(ep): return args.lambda_d2 * min(1.0, ep/6.0)

    # ---------------------------- Training ----------------------------
    epoch_bar = tqdm(range(1, args.epochs+1), desc="Epochs", ncols=140, leave=False)
    for epoch in epoch_bar:
        model.train()

        freeze_backbone = (epoch <= args.freeze_backbone_epochs)
        set_backbone_trainable(model, train_backbone=not freeze_backbone)

        lam = lam_d2_now(epoch)
        epoch_metrics = {}
        epoch_log = {"steps": 0, "loss": 0.0, "kept": 0.0, "eobj": 0.0, "ebg": 0.0, "d2m": 0.0, "tau_eff": 0.0}

        for x,y,g,pil,bg,idx in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            # CE
            with amp_ctx():
                logits = model(x)
            loss_ce = ce(logits, y)

            # Δ² steering (only after unfreezing)
            if not freeze_backbone:
                k = min(args.anchors_per_step, len(train_ds))
                if ranked_bank is not None:
                    anchors = pick_anchors_from_ranked_bank(k, ranked_bank, train_ds)
                else:
                    anchors = pick_balanced_anchors_from_bank(
                        k=k, worst_by_class=worst_by_class, group_bias=args.group_bias,
                        idx_by_class=idx_by_class, idx_by_group=idx_by_group, ds=train_ds,
                    )

                p_d2, p_obj, p_bg, p_obj_hi, eobj_mean, d2_abs_mean, d2_abs_p90, kept, cls0, cls1, e_bg_abs_mean, tau_eff = d2_penalty(
                    model=model, device=device, tf_tensor=val_ds.tf,
                    anchors=anchors, donor_pools=donor_pools_train, donor_ds=train_ds,
                    donor_policy=args.donor_policy,
                    mode=args.mode,
                    tau_base=args.tau, K=args.K, two_donors=bool(args.two_donors),
                    conf_min=args.conf_min,
                    min_keep_per_class=args.min_keep_per_class, margin_d2=args.margin_d2,
                    eps_obj=args.eps_obj, tau_floor_mult=args.tau_floor_mult,
                    tau_bg_min=args.tau_bg_min, tau_bg_fac=args.tau_bg_fac,
                    tau_bg_guard_min=args.tau_bg_guard_min, tau_bg_guard_fac=args.tau_bg_guard_fac,
                    max_box_frac=args.max_box_frac,
                    amp_ctx=amp_ctx,
                )
                if eobj_ema is None:
                    eobj_ema = float(eobj_mean)
                else:
                    eobj_ema = args.ema_obj_beta * eobj_ema + (1 - args.ema_obj_beta) * float(eobj_mean)
                ema_floor = torch.tensor(eobj_ema, device=eobj_mean.device)
                p_obj_floor = torch.clamp(ema_floor - eobj_mean, min=0).pow(2)

                loss = (
                    loss_ce
                    + lam * p_d2
                    + args.guard_weight * (p_bg + p_obj + 0.5*p_obj_hi)
                    - args.lam_obj * eobj_mean
                    + args.lam_obj_floor * p_obj_floor
                )
                epoch_metrics = {
                    "loss": f"{loss.item():.3f}",
                    "d2": f"{(lam*p_d2).item():.4f}",
                    "d2m": f"{float(d2_abs_mean):.3f}",
                    "tau": f"{float(tau_eff):.3f}",
                    "eobj": f"{float(eobj_mean):.3f}",
                    "ebg": f"{float(e_bg_abs_mean):.3f}",
                    "ank": f"{kept:02d}({cls0}/{cls1})",
                    "mode": ("ranked" if ranked_bank is not None else "balanced") + f"/{args.mode}",
                }
                epoch_log["steps"] += 1
                epoch_log["loss"] += float(loss.item())
                epoch_log["kept"] += float(kept)
                epoch_log["eobj"] += float(eobj_mean)
                epoch_log["ebg"] += float(e_bg_abs_mean)
                epoch_log["d2m"] += float(d2_abs_mean)
                epoch_log["tau_eff"] += float(tau_eff)
            else:
                loss = loss_ce
                epoch_metrics = {"loss": f"{loss.item():.3f}", "mode": "head-only"}

            if amp_enabled:
                scaler.scale(loss).backward()
                if args.clip_grad_norm>0:
                    scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(opt); scaler.update()
            else:
                loss.backward()
                if args.clip_grad_norm>0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                opt.step()

        if epoch_metrics:
            epoch_bar.set_postfix(epoch_metrics)

        if epoch_log["steps"] > 0:
            steps = epoch_log["steps"]
            tqdm.write(
                f"[TRAIN] epoch={epoch:02d} "
                f"loss={epoch_log['loss']/steps:.3f} "
                f"d2m={epoch_log['d2m']/steps:.3f} "
                f"tau={epoch_log['tau_eff']/steps:.3f} "
                f"eobj={epoch_log['eobj']/steps:.3f} "
                f"ebg={epoch_log['ebg']/steps:.3f} "
                f"kept={epoch_log['kept']/steps:.1f}"
            )
        else:
            tqdm.write(f"[TRAIN] epoch={epoch:02d} head-only (backbone frozen)")

        # VAL
        model.eval()
        va_avg, va_worst, va_det = evaluate(model, val_loader, device, amp_ctx)
        pred1 = val_pred1_rate(model, val_loader, device, amp_ctx)
        tqdm.write(f"[VAL] epoch={epoch:02d} avg={va_avg:.3f} worst={va_worst:.3f} pred1_rate={pred1:.3f} groups={va_det}")

        if va_det:
            worst_by_class[0] = min([0,1], key=lambda g: va_det.get(g, (1.0,0))[0])
            worst_by_class[1] = min([2,3], key=lambda g: va_det.get(g, (1.0,0))[0])

        # save best by worst-group
        if va_worst > best_wga:
            best_wga = va_worst
            best_wga_epoch = epoch
            torch.save(
                {
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'args': vars(args),
                    'metric': {
                        'val_worst_group_accuracy': float(va_worst),
                        'val_average_accuracy': float(va_avg),
                    },
                },
                best_ckpt_path,
            )

        if args.save_combo_checkpoint and epoch_log["steps"] > 0:
            steps = epoch_log["steps"]
            epoch_d2m = epoch_log["d2m"] / steps
            combo_wga_floor = best_wga - args.combo_wga_tolerance
            combo_eligible = va_worst >= combo_wga_floor
            combo_better = (
                combo_eligible
                and (
                    epoch_d2m < best_combo_d2m - 1e-12
                    or (
                        abs(epoch_d2m - best_combo_d2m) <= 1e-12
                        and va_worst > best_combo_wga
                    )
                )
            )
            if combo_better:
                best_combo_d2m = epoch_d2m
                best_combo_wga = va_worst
                best_combo_epoch = epoch
                torch.save(
                    {
                        'model': model.state_dict(),
                        'epoch': epoch,
                        'args': vars(args),
                        'combo_metric': {
                            'val_worst_group_accuracy': float(va_worst),
                            'val_average_accuracy': float(va_avg),
                            'train_d2_abs_mean': float(epoch_d2m),
                            'combo_wga_tolerance': float(args.combo_wga_tolerance),
                            'best_wga_so_far': float(best_wga),
                        },
                    },
                    combo_ckpt_path
                )
        sched.step()

    # TEST — fixed best-ckpt name
    tqdm.write("\n--- Final Test Evaluation ---")
    best_summary = None
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model'])
        best_epoch = ckpt.get('epoch')
        best_metric = ckpt.get('metric', {})
    else:
        tqdm.write("No best checkpoint saved; testing last state.")
        best_epoch = None
        best_metric = {}

    te_avg, te_worst, te_det = evaluate(model, test_loader, device, amp_ctx)
    best_summary = {
        "path": str(best_ckpt_path),
        "epoch": best_epoch,
        "val_average_accuracy": float(best_metric.get("val_average_accuracy", float("nan"))),
        "val_worst_group_accuracy": float(best_metric.get("val_worst_group_accuracy", float("nan"))),
        "test_average_accuracy": float(te_avg),
        "test_worst_group_accuracy": float(te_worst),
        "test_groups": te_det,
    }
    tqdm.write(f"[TEST] avg={te_avg:.4f} worst={te_worst:.4f} groups={te_det} epoch={best_epoch}")

    combo_summary = None
    if args.save_combo_checkpoint and combo_ckpt_path.exists():
        combo_ckpt = torch.load(combo_ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(combo_ckpt['model'])
        combo_te_avg, combo_te_worst, combo_te_det = evaluate(model, test_loader, device, amp_ctx)
        combo_metric = combo_ckpt.get("combo_metric", {})
        combo_summary = {
            "path": str(combo_ckpt_path),
            "epoch": combo_ckpt.get("epoch"),
            "val_average_accuracy": float(combo_metric.get("val_average_accuracy", float("nan"))),
            "val_worst_group_accuracy": float(combo_metric.get("val_worst_group_accuracy", float("nan"))),
            "train_d2_abs_mean": float(combo_metric.get("train_d2_abs_mean", float("nan"))),
            "test_average_accuracy": float(combo_te_avg),
            "test_worst_group_accuracy": float(combo_te_worst),
            "test_groups": combo_te_det,
        }
        tqdm.write(
            f"[TEST combo] avg={combo_te_avg:.4f} worst={combo_te_worst:.4f} "
            f"groups={combo_te_det} epoch={combo_ckpt.get('epoch')}"
        )

    summary = {
        "output_dir": str(output_dir),
        "best_wga_epoch": best_wga_epoch,
        "best_val_wga": float(best_wga) if best_wga >= 0 else None,
        "best_combo_epoch": best_combo_epoch,
        "best_combo_train_d2_abs_mean": (float(best_combo_d2m) if math.isfinite(best_combo_d2m) else None),
        "best_combo_val_wga": float(best_combo_wga) if best_combo_wga >= 0 else None,
        "best_checkpoint": best_summary,
        "combo_checkpoint": combo_summary,
        "ranked_bank_stats": (ranked_bank.get("_stats") if ranked_bank is not None else None),
        "args": vars(args),
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    tqdm.write(f"[saved] summary -> {summary_path}")

if __name__ == "__main__":
    main()
