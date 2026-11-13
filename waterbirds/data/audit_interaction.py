
#!/usr/bin/env python3
"""
audit_interaction.py — Δ²(Object×Background) auditor + anchor bank builder.

Mirrors the masked training pipeline of train/train_is.py:
  • reads the prepared data manifests with dataset masks rather than running detection.
  • corners use mask inpainting, donor cleansing with donor masks, and exact mask-based bird re-paste.
  • every anchor logs Δ², e_obj, e_bg, seam L2, has_mask, and on_manifold for ranked-bank reuse.
  • writes the single-policy CSV/stat outputs read by the ranked-bank loader in train/train_is.py.
"""

import argparse, csv, json, os, random, math, sys
from pathlib import Path
from collections import defaultdict, Counter
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.models as models
import timm
from PIL import Image, ImageFilter, ImageOps, ImageDraw, ImageChops

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
from backbone_utils import build_image_transforms, build_pil_transforms

# ------------------ dataset wrapper (same as training) ------------------

_DEFAULT_REAL_ROOT = Path(os.environ.get("DATA_ROOT", ".")) / "waterbirds_manifest"

class HFWaterbirds(Dataset):
    def __init__(self, split: str, image_size: int = 224, data_root: str | Path | None = None, backbone: str = "resnet50"):
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
        self.tf = build_image_transforms(backbone=backbone, image_size=image_size, train_aug=False)
        self.tf_pil = build_pil_transforms(backbone=backbone, image_size=image_size)

    def __len__(self): return len(self.rows)

    def has_mask(self, i: int) -> bool:
        row = self.rows[int(i)]
        return bool(int(row.get("mask_available", 0))) and bool(row.get("resolved_mask_path"))

    def load_mask(self, i: int) -> Image.Image | None:
        row = self.rows[int(i)]
        mask_path = str(row.get("resolved_mask_path", "") or "")
        if not mask_path:
            return None
        try:
            with Image.open(mask_path) as mask:
                return self.tf_pil(mask.convert("L"))
        except FileNotFoundError:
            return None

    def box_for_index(self, i: int) -> list[int] | None:
        row = self.rows[int(i)]
        box = row.get("bird_box")
        if not isinstance(box, list) or len(box) != 4:
            return None
        x1, y1, x2, y2 = [int(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    def __getitem__(self, i):
        row = self.rows[int(i)]
        with Image.open(str(row["resolved_image_path"])) as image:
            pil = image.convert("RGB")
        y = int(row["label"])
        bg = int(row["background"])
        group = int(row["group"])
        return self.tf(pil), y, group, self.tf_pil(pil), bg

# ------------------ donor pools & (optional) box cache ------------------

def build_donor_pools(ds: HFWaterbirds):
    idx_by_bg = defaultdict(list)
    all_valid = []
    for i, row in enumerate(ds.rows):
        if not ds.has_mask(i):
            continue
        idx = int(i)
        idx_by_bg[int(row["background"])].append(idx)
        all_valid.append(idx)
    idx_by_bg["all"] = all_valid
    return idx_by_bg

def load_box_cache(path: Path):
    if path and path.exists():
        with open(path, 'r') as f:
            raw = json.load(f)
        # keys are str indices, values are [x1,y1,x2,y2] or None
        return {int(k): v for k, v in raw.items()}
    return {}

def save_box_cache(path: Path, cache: dict):
    if not path: return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in cache.items()}, f)

def resolve_box_cache_path(raw, split: str):
    if raw is None:
        return Path("caches") / f"box_{split}.json"
    raw = raw.strip()
    if raw.lower() == "none":
        return None
    base = Path(raw)
    if base.suffix:
        return base.with_name(f"{base.stem}_{split}{base.suffix}")
    return base / f"box_{split}.json"

# ---------- Background embedding (label-free) and simple k-means ----------

def _bg_crop_for_embed(pil_img: Image.Image, box, pad_ratio: float = 0.06):
    """
    Return a background-only view (mask the bird box) for embedding.
    We fill the box with the surround mean to avoid injecting edges.
    """
    W, H = pil_img.size
    img = pil_img.copy()
    if box is None:
        return img
    x1,y1,x2,y2 = box
    # inflate box a bit to cover feathers
    w = x2 - x1; h = y2 - y1
    dx = int(pad_ratio * w); dy = int(pad_ratio * h)
    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy)
    x2 = min(W, x2 + dx); y2 = min(H, y2 + dy)
    # fill region with local mean color
    region = img.crop((x1,y1,x2,y2))
    mean_color = tuple(np.array(region.resize((1,1), Image.BILINEAR)).reshape(-1))
    draw = Image.new("RGB", (x2-x1, y2-y1), mean_color)
    img.paste(draw, (x1,y1))
    return img

def _bg_embed_hsv_hist(pil_img: Image.Image, bins: int = 16):
    """
    Simple label-free background embedding: HSV histogram (H,S,V) concatenated.
    Robust, ~48 dims by default.
    """
    hsv = pil_img.convert("HSV")
    arr = np.asarray(hsv)
    Hch, Sch, Vch = arr[:,:,0].ravel(), arr[:,:,1].ravel(), arr[:,:,2].ravel()
    h_hist, _ = np.histogram(Hch, bins=bins, range=(0,255), density=True)
    s_hist, _ = np.histogram(Sch, bins=bins, range=(0,255), density=True)
    v_hist, _ = np.histogram(Vch, bins=bins, range=(0,255), density=True)
    emb = np.concatenate([h_hist, s_hist, v_hist]).astype(np.float32)
    # L2 normalize
    n = np.linalg.norm(emb) + 1e-8
    return emb / n

def _kmeans_np(X: np.ndarray, k: int = 2, iters: int = 25, seed: int = 123):
    """
    Tiny k-means in NumPy (L2) to avoid external deps.
    X: [N,D]
    Returns: (centroids [k,D], labels [N])
    """
    rng = np.random.RandomState(seed)
    N, D = X.shape
    if N < k:
        # trivial fallback: fewer points than clusters
        cents = np.pad(X.copy(), ((0, k-N), (0,0)), mode='edge')[:k]
        labels = np.arange(N) % max(1,k)
        return cents, labels
    # init with k random points
    idx = rng.choice(N, size=k, replace=False)
    C = X[idx].copy()
    for _ in range(iters):
        # assign
        d2 = ((X[:,None,:] - C[None,:,:])**2).sum(-1)   # [N,k]
        lab = d2.argmin(1)
        # update
        C_new = np.zeros_like(C)
        for j in range(k):
            mask = (lab == j)
            if mask.any():
                C_new[j] = X[mask].mean(0)
            else:
                # re-seed empty centroid
                C_new[j] = X[rng.randint(0,N)]
        if np.allclose(C_new, C, atol=1e-5):
            C = C_new; break
        C = C_new
    return C, lab

def build_bg_clusters(
    ds: HFWaterbirds,
    max_items: int = None,
    bins: int = 16,
    k_clusters: int = 2,
    seed: int = 123,
):
    """
    Compute label-free background embeddings and k-means clusters for the split.
    Uses dataset position i as the 'idx'.
    Returns:
      {'ids': [...], 'emb': X, 'centroids': C, 'labels': lab}
    """
    ids, embs = [], []
    it_range = range(len(ds)) if max_items is None else range(min(max_items, len(ds)))
    for i in it_range:
        if not ds.has_mask(i):
            continue
        _, _, _, pil, _ = ds[i]
        idx = i  # dataset position is our id
        box = mask_bbox(ds.load_mask(i)) or ds.box_for_index(i)

        bg_only = _bg_crop_for_embed(pil, box)
        emb = _bg_embed_hsv_hist(bg_only, bins=bins)
        ids.append(int(idx)); embs.append(emb)

    if not embs:
        return {
            'ids': [],
            'emb': np.zeros((0,48), np.float32),
            'centroids': np.zeros((0,48), np.float32),
            'labels': np.zeros((0,), np.int64)
        }

    X = np.stack(embs, 0)
    cents, labels = _kmeans_np(X, k=k_clusters, iters=30, seed=seed)
    return {'ids': ids, 'emb': X, 'centroids': cents, 'labels': labels}
# ------------------ counterfactual constructors ------------------

def blur_inside_box(pil_img: Image.Image, box, radius=9):
    if box is None:
        return pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
    x1,y1,x2,y2 = box
    region = pil_img.crop((x1,y1,x2,y2)).filter(ImageFilter.GaussianBlur(radius=radius))
    out = pil_img.copy()
    out.paste(region, (x1,y1))
    return out

def swap_background(pil_img: Image.Image, donor_pil: Image.Image, box):
    """Keep object box from pil_img; fill outside box with donor background."""
    if box is None:
        # fallback: simple half cutmix
        W,H = pil_img.size
        out = donor_pil.copy()
        out.paste(pil_img.crop((0,0,W//2,H)), (0,0))
        return out
    x1,y1,x2,y2 = box
    out = donor_pil.copy()
    out.paste(pil_img.crop((x1,y1,x2,y2)), (x1,y1))
    return out

def hist_match_brightness(src: Image.Image, ref: Image.Image, alpha=0.25):
    """Fast brightness-only histogram match; blend to keep color."""
    src_gray = ImageOps.grayscale(src)
    ref_gray = ImageOps.grayscale(ref)
    s = np.asarray(src_gray).ravel()
    r = np.asarray(ref_gray).ravel()
    s_vals, s_idx, s_cnt = np.unique(s, return_inverse=True, return_counts=True)
    r_vals, r_cnt = np.unique(r, return_counts=True)
    s_q = np.cumsum(s_cnt).astype(np.float64) / s.size
    r_q = np.cumsum(r_cnt).astype(np.float64) / r.size
    interp = np.interp(s_q, r_q, r_vals)
    matched = interp[s_idx].reshape(src_gray.size[1], src_gray.size[0]).astype(np.uint8)
    matched_rgb = Image.merge("RGB", [Image.fromarray(matched)]*3)
    return Image.blend(src, matched_rgb, alpha=alpha)

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

def feathered_paste(dst: Image.Image, src_crop: Image.Image, box, feather: int = 4):
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

# ---------- Helpers: masks, fills, and donor cleansing ----------

def _try_import_cv2():
    try:
        import cv2
        return cv2
    except Exception:
        return None

CV2 = _try_import_cv2()
if CV2 is None:
    print("[warn] OpenCV not available; auditors will use feathered rectangle masks.")

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
        rx = random.randint(0, max_rx); ry = random.randint(0, max_ry)
        if _overlap((rx,ry,rx+w,ry+h), (x1,y1,x2,y2)):
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

def mask_bbox(mask: Image.Image | None):
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

def cleanse_donor(donor_pil: Image.Image, donor_mask: Image.Image | None):
    if donor_mask is None:
        return donor_pil.copy()
    return inpaint_with_mask(donor_pil, donor_mask, radius=4, expand_radius=3, threshold=127)

# ------------------ anchors I/O ------------------

def save_anchors(path: Path, anchors):
    """anchors: list of dicts {idx, y, bg, group, box, donors:[idx,...]}"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(anchors, f)

def class_margin(logits: torch.Tensor, y: int):
    other = 1 - y
    return logits[:, y] - logits[:, other]

def load_anchors(path: Path):
    with open(path, 'r') as f:
        return json.load(f)
# ---------- Donor sampling with multiple policies ----------

def _index_of_id(ds: HFWaterbirds, idx_value: int) -> int:
    """Map dataset 'idx' (returned in __getitem__) back to position 0..len(ds)-1."""
    # linear map; HFWaterbirds returns idx=i, so it's identity in this wrapper
    return idx_value

def sample_donors_general(
    policy: str,
    anchor_idx: int,       # dataset-level idx (not position)
    anchor_bg: int,
    ds: HFWaterbirds,
    donor_pools_by_bg: dict,     # {0:[idx...], 1:[idx...]}
    cluster_info: dict = None,   # from build_bg_clusters (for clustered policies)
    num_donors: int = 3,
    seed: int = 123,
):
    """
    policy ∈ {"opposite","same","random","cluster_near","cluster_far"}
    Returns: list[int] donor idx values (dataset-level ids), no self-donor.
    """
    rng = random.Random(seed + anchor_idx * 97)
    N = len(ds)

    # helper: convert cluster labels to a dict idx->cluster
    idx_to_cluster = {}
    cents = None
    if cluster_info and cluster_info.get('ids'):
        ids = cluster_info['ids']
        labs = cluster_info['labels']
        idx_to_cluster = {int(i): int(l) for i,l in zip(ids, labs)}
        cents = cluster_info['centroids']

    pool = donor_pools_by_bg.get("all", list(range(N)))  # default pool is every valid donor
    # exclude self
    pool = [j for j in pool if j != _index_of_id(ds, anchor_idx)]

    if policy == "opposite":
        pool = donor_pools_by_bg.get(1 - anchor_bg, [])
        pool = [p for p in pool if p != anchor_idx]
    elif policy == "same":
        pool = donor_pools_by_bg.get(anchor_bg, [])
        pool = [p for p in pool if p != anchor_idx]
    elif policy == "random":
        pass  # keep full pool (minus self)
    elif policy in ("cluster_near","cluster_far"):
        if not idx_to_cluster:
            # fallback: random if no clusters available
            pass
        else:
            # pick donors from nearest or farthest centroid to anchor's cluster
            # get anchor's cluster
            a_c = idx_to_cluster.get(anchor_idx, None)
            if a_c is None or cents is None:
                pass
            else:
                # compute centroid distances
                c_anchor = cents[a_c]
                dists = np.sqrt(((cents - c_anchor[None,:])**2).sum(-1))  # [k]
                if policy == "cluster_near":
                    # include same cluster first (excluding self), then nearest other
                    near_order = np.argsort(dists)  # 0 == same cluster
                    chosen_clusters = [int(c) for c in near_order[:max(1, len(near_order))]]
                else:
                    far_order = np.argsort(-dists)
                    chosen_clusters = [int(c) for c in far_order[:max(1, len(far_order))]]
                # gather all idx in those clusters
                idxs = [i for i,lab in idx_to_cluster.items() if lab in chosen_clusters and i != anchor_idx]
                if idxs:
                    pool = idxs
                else:
                    # fallback to random pool
                    pass
    else:
        raise ValueError(f"Unknown donor policy: {policy}")

    if not pool:
        return []

    # sample with replacement to guarantee num_donors
    donors = [rng.choice(pool) for _ in range(num_donors)]
    return donors

# ------------------ Δ² audit core ------------------

@torch.no_grad()
def d2_audit(
    model, ds: HFWaterbirds, device,
    sample_anchors=400, donor_per_anchor=3, seed=123,
    box_cache_path: Path=None, min_score=0.5,
    save_anchors_path: Path=None, load_anchors_path: Path=None,
    scores_csv: Path=None,
    donor_policy: str = "opposite",
    use_margin: bool = False,
    max_box_frac: float = 0.45,
):
    rng = random.Random(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    model.eval()
    tf = ds.tf
    donor_pools = build_donor_pools(ds)

    anchors = []
    if load_anchors_path:
        anchors = load_anchors(load_anchors_path)
    else:
        idxs = list(range(len(ds))); rng.shuffle(idxs)
        for idx in idxs:
            if len(anchors) >= sample_anchors: break
            _, y, group, pil_img, bg = ds[idx]
            anchor_mask = ds.load_mask(idx)
            box = mask_bbox(anchor_mask) if anchor_mask is not None else None
            if box is None or anchor_mask is None:
                continue
            w = box[2]-box[0]; h = box[3]-box[1]
            if (w*h) > max_box_frac * (pil_img.width * pil_img.height):
                continue
            donor_idxs = sample_donors_general(
                policy=donor_policy,
                anchor_idx=idx,
                anchor_bg=int(bg),
                ds=ds,
                donor_pools_by_bg=donor_pools,
                num_donors=donor_per_anchor,
                seed=seed,
            )
            if not donor_idxs:
                continue
            anchors.append({"idx": idx, "y": int(y), "bg": int(bg), "group": int(group), "donors": donor_idxs})
        if save_anchors_path is not None:
            save_anchors(save_anchors_path, anchors)

    writer = None; csv_f = None
    if scores_csv is not None:
        scores_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        csv_f = open(scores_csv, "w", newline="")
        writer = csv.writer(csv_f)
        writer.writerow(["idx","y","group","bg","has_mask","has_box","donors","d2_mean","d2_abs","e_obj","e_bg","on_manifold","seam_l2"])

    if device.type=='cuda':
        amp_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        amp_ctx = lambda: nullcontext()

    d2_vals, e_obj_vals, e_bg_vals, errs = [], [], [], []
    realism = Counter()
    per_group = {g: {"errs": [], "abs_d2": [], "e_bg": [], "e_obj": []} for g in [0,1,2,3]}

    for a in anchors:
        idx = a["idx"]; y = a["y"]; group = a["group"]; bg = a["bg"]
        _, _, _, pil_img, _ = ds[idx]
        anchor_mask = ds.load_mask(idx)
        box = mask_bbox(anchor_mask) if anchor_mask is not None else None
        has_mask = int(anchor_mask is not None and box is not None)
        realism.update({'has_mask' if has_mask else 'no_mask': 1})
        if not has_mask:
            if writer is not None:
                nan = float('nan')
                writer.writerow([idx, y, group, bg, 0, 0, len(a["donors"]), nan, nan, nan, nan, 0, nan])
            continue
        w = box[2]-box[0]; h = box[3]-box[1]
        if (w*h) > max_box_frac * (pil_img.width * pil_img.height):
            realism.update({'off_manifold': 1})
            continue

        # Prepare x00/x10 once
        x00_pil = pil_img
        x10_pil = inpaint_with_mask(pil_img, anchor_mask, radius=4, expand_radius=3, threshold=127)
        x00_patch = np.asarray(x00_pil.crop(box))
        x10_patch = np.asarray(x10_pil.crop(box))
        seam_scores = [_border_l2(x10_patch, x00_patch)]
        obj_crop = pil_img.crop(tuple(box))
        segment_mask = threshold_mask(mask_crop(anchor_mask, box), threshold=127)

        X00 = tf(x00_pil).unsqueeze(0).to(device)
        X10 = tf(x10_pil).unsqueeze(0).to(device)

        with amp_ctx():
            logits00 = model(X00); logits10 = model(X10)
        pred = int(logits00.argmax(1).item())
        err_item = 1 if pred != y else 0
        errs.append(err_item)

        # Build donor batches
        X01_list, X11_list = [], []
        for d_idx in a["donors"]:
            _, _, _, donor_pil, _ = ds[d_idx]
            donor_mask = ds.load_mask(d_idx)
            if donor_mask is None:
                continue
            donor_clean = cleanse_donor(donor_pil, donor_mask)
            donor_adj = hist_match(donor_clean, pil_img)
            x01_pil = paste_with_segment_mask(donor_adj, obj_crop, tuple(box), segment_mask, blur_radius=0)
            x11_pil = donor_clean
            seam_scores.append(_border_l2(np.asarray(x11_pil.crop(box)), np.asarray(donor_clean.crop(box))))
            X01_list.append(tf(x01_pil)); X11_list.append(tf(x11_pil))

        if not X01_list:
            continue

        X01 = torch.stack(X01_list, 0).to(device)
        X11 = torch.stack(X11_list, 0).to(device)
        with amp_ctx():
            logits01 = model(X01)  # [D,2]
            logits11 = model(X11)  # [D,2]
        if use_margin:
            z00 = class_margin(logits00, y)
            z10 = class_margin(logits10, y)
            z01 = class_margin(logits01, y)
            z11 = class_margin(logits11, y)
        else:
            z00 = logits00[:, y]
            z10 = logits10[:, y]
            z01 = logits01[:, y]
            z11 = logits11[:, y]

        # Compute per-donor Δ² then average
        d2_per = (z11 - z01) - (z10 - z00)  # broadcast
        e_obj = (z00 - z10)                 # constant across donors
        e_bg_per = (z00 - z01)

        d2_mean = float(d2_per.mean().item())
        eobj_mean = float(e_obj.mean().item())
        ebg_mean  = float(e_bg_per.mean().item())

        # quality flags for downstream filtering
        scale = torch.tensor(1.0, device=logits00.device)
        for logits in (logits00, logits10, logits01, logits11):
            scale = torch.max(scale, logits.abs().amax())
        tau_bg = max(0.10, 0.02*float(scale.detach().item()))
        on_manifold = int((e_obj.mean().item() > 0.20) and (abs(e_bg_per.mean().item()) < tau_bg))
        seam_l2 = float(np.mean(seam_scores)) if seam_scores else 0.0

        d2_vals.append(d2_mean)
        e_obj_vals.append(eobj_mean)
        e_bg_vals.append(ebg_mean)

        per_group[group]["errs"].append(err_item)
        per_group[group]["abs_d2"].append(abs(d2_mean))
        per_group[group]["e_bg"].append(ebg_mean)
        per_group[group]["e_obj"].append(eobj_mean)

        if writer is not None:
            writer.writerow([idx, y, group, bg, 1, 1, len(a["donors"]), d2_mean, abs(d2_mean), eobj_mean, ebg_mean, on_manifold, seam_l2])

    if csv_f is not None:
        csv_f.close()

    if not d2_vals:
        return None

    arr = np.array(d2_vals); eobj = np.array(e_obj_vals); ebg = np.array(e_bg_vals); err = np.array(errs[:len(arr)])
    abs_arr = np.abs(arr)

    def pb_corr_binary_cont(b, x):
        b = (b - b.mean()) / (b.std() + 1e-8)
        x = (x - x.mean()) / (x.std() + 1e-8)
        return float(np.mean(b * x))

    overall_corr = pb_corr_binary_cont(err, np.abs(arr)) if len(arr) > 2 else 0.0

    per_group_stats = {}
    for g, d in per_group.items():
        if len(d["errs"]) >= 3 and np.std(d["abs_d2"]) > 0:
            pg_corr = pb_corr_binary_cont(np.array(d["errs"]), np.array(d["abs_d2"]))
        else:
            pg_corr = 0.0
        per_group_stats[g] = {
            "n": len(d["errs"]),
            "corr_|d2|_vs_error": pg_corr,
            "d2_mean": float(np.mean(d["abs_d2"])) if d["abs_d2"] else 0.0,
            "e_bg_mean": float(np.mean(d["e_bg"])) if d["e_bg"] else 0.0,
            "e_obj_mean": float(np.mean(d["e_obj"])) if d["e_obj"] else 0.0,
        }

    stats = {
        "n": int(len(arr)),
        "donor_policy": str(donor_policy),
        "d2_mean": float(arr.mean()),
        "d2_median": float(np.median(arr)),
        "d2_p10": float(np.percentile(arr, 10)),
        "d2_p90": float(np.percentile(arr, 90)),
        "d2_abs_mean": float(abs_arr.mean()),
        "d2_abs_p90":  float(np.percentile(abs_arr, 90)),
        "eobj_mean": float(eobj.mean()),
        "ebg_mean": float(ebg.mean()),
        "realism_stats": dict(realism),
        "corr_|d2|_vs_error": float(overall_corr),
        "donor_per_anchor": int(donor_per_anchor),
        "per_group": per_group_stats,
    }
    return stats

@torch.no_grad()
def d2_audit_general(
    model,
    ds: HFWaterbirds,
    device,
    donor_policies = ("opposite","same","random","cluster_near","cluster_far"),
    sample_anchors=400,
    donors_per_policy=3,
    seed=123,
    min_score=0.5,
    box_cache_path: Path=None,
    save_anchor_ids_path: Path=None,
    load_anchor_ids_path: Path=None,
    per_policy_csv: Path=None,
    k_clusters: int = 2,
    use_margin: bool = False,
    max_box_frac: float = 0.45,
):
    rng = random.Random(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    model.eval(); tf = ds.tf
    donor_pools = build_donor_pools(ds)

    # anchors
    if load_anchor_ids_path:
        with open(load_anchor_ids_path, "r") as f:
            anchor_ids = json.load(f)
    else:
        idxs = list(range(len(ds))); rng.shuffle(idxs)
        anchor_ids = [int(pos) for pos in idxs[:sample_anchors]]
        if save_anchor_ids_path:
            Path(save_anchor_ids_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_anchor_ids_path, "w") as f: json.dump(anchor_ids, f)

    # clusters only if needed
    need_clusters = any(p in ("cluster_near","cluster_far") for p in donor_policies)
    cluster_info = build_bg_clusters(
        ds=ds,
        k_clusters=k_clusters,
        seed=seed,
    ) if need_clusters else None

    writer = None; csv_f = None
    if per_policy_csv is not None:
        per_policy_csv.parent.mkdir(parents=True, exist_ok=True)
        import csv
        csv_f = open(per_policy_csv, "w", newline="")
        writer = csv.writer(csv_f)
        writer.writerow(["anchor_idx","y","group","bg","policy","donors_used","d2_mean","d2_abs","e_obj","e_bg","has_mask","has_box","on_manifold","seam_l2"])

    per_policy_vals = {p: {"abs_d2": [], "e_bg": [], "e_obj": [], "n": 0} for p in donor_policies}
    policy_wins = {p: 0 for p in donor_policies}
    n_valid_anchors = 0

    if device.type=='cuda':
        amp_ctx = lambda: torch.amp.autocast("cuda", dtype=torch.float16)
    else:
        amp_ctx = lambda: nullcontext()

    for idx in anchor_ids:
        _, y, group, pil_img, bg = ds[idx]

        # 1) sample donors for all policies first (cheap, label-free)
        donors_by_policy = {}
        for policy in donor_policies:
            donors_by_policy[policy] = sample_donors_general(
                policy=policy, anchor_idx=idx, anchor_bg=int(bg),
                ds=ds, donor_pools_by_bg=donor_pools,
                cluster_info=cluster_info, num_donors=donors_per_policy, seed=seed,
            )

        # if no donors for any policy, skip (saves detection work)
        if not any(donors_by_policy[p] for p in donor_policies):
            continue

        anchor_mask = ds.load_mask(idx)
        box = mask_bbox(anchor_mask) if anchor_mask is not None else None
        if box is None or anchor_mask is None:
            continue
        w = box[2]-box[0]; h = box[3]-box[1]
        if (w*h) > max_box_frac * (pil_img.width * pil_img.height):
            continue

        # base x00/x10 once
        x00_pil = pil_img
        x10_pil = inpaint_with_mask(pil_img, anchor_mask, radius=4, expand_radius=3, threshold=127)
        x00_patch = np.asarray(x00_pil.crop(box))
        x10_patch = np.asarray(x10_pil.crop(box))
        seam_base = _border_l2(x10_patch, x00_patch)
        obj_crop = pil_img.crop(tuple(box))
        segment_mask = threshold_mask(mask_crop(anchor_mask, box), threshold=127)

        X00 = tf(x00_pil).unsqueeze(0).to(device)
        X10 = tf(x10_pil).unsqueeze(0).to(device)
        with amp_ctx():
            logits00 = model(X00); logits10 = model(X10)
        if use_margin:
            z00 = class_margin(logits00, y)
            z10 = class_margin(logits10, y)
        else:
            z00 = logits00[:, y]
            z10 = logits10[:, y]

        per_pol_d2_abs_mean = {}
        any_valid = False

        # 3) per-policy batched donors
        for policy in donor_policies:
            donors = donors_by_policy[policy]
            if not donors:
                continue

            X01_list, X11_list = [], []
            seam_scores = [seam_base]
            for d_idx in donors:
                _, _, _, donor_pil, _ = ds[d_idx]
                donor_mask = ds.load_mask(d_idx)
                if donor_mask is None:
                    continue
                donor_clean = cleanse_donor(donor_pil, donor_mask)
                donor_adj = hist_match(donor_clean, pil_img)
                x01_pil = paste_with_segment_mask(donor_adj, obj_crop, tuple(box), segment_mask, blur_radius=0)
                x11_pil = donor_clean
                seam_scores.append(_border_l2(np.asarray(x11_pil.crop(box)), np.asarray(donor_clean.crop(box))))
                X01_list.append(tf(x01_pil)); X11_list.append(tf(x11_pil))

            if not X01_list:
                continue

            X01 = torch.stack(X01_list, 0).to(device)
            X11 = torch.stack(X11_list, 0).to(device)
            with amp_ctx():
                logits01 = model(X01); logits11 = model(X11)
            if use_margin:
                z01 = class_margin(logits01, y)
                z11 = class_margin(logits11, y)
            else:
                z01 = logits01[:, y]; z11 = logits11[:, y]

            d2_per = (z11 - z01) - (z10 - z00)
            e_obj  = (z00 - z10)                  # constant across donors
            e_bg   = (z00 - z01)

            d2_mean = float(d2_per.mean().item())
            e_obj_mean = float(e_obj.mean().item())
            e_bg_mean  = float(e_bg.mean().item())
            scale = torch.tensor(1.0, device=logits00.device)
            for logits in (logits00, logits10, logits01, logits11):
                scale = torch.max(scale, logits.abs().amax())
            tau_bg = max(0.10, 0.02*float(scale.detach().item()))
            on_manifold = int((e_obj.mean().item() > 0.20) and (abs(e_bg.mean().item()) < tau_bg))
            seam_l2 = float(np.mean(seam_scores)) if seam_scores else 0.0

            any_valid = True
            per_policy_vals[policy]["abs_d2"].append(abs(d2_mean))
            per_policy_vals[policy]["e_bg"].append(e_bg_mean)
            per_policy_vals[policy]["e_obj"].append(e_obj_mean)
            per_policy_vals[policy]["n"] += 1
            per_pol_d2_abs_mean[policy] = abs(d2_mean)

            if writer is not None:
                writer.writerow([idx, int(y), int(group), int(bg), policy, len(donors),
                                 d2_mean, abs(d2_mean), e_obj_mean, e_bg_mean, 1, 1, on_manifold, seam_l2])

        if any_valid:
            n_valid_anchors += 1
            best_pol = max(per_pol_d2_abs_mean.items(), key=lambda kv: kv[1])[0]
            policy_wins[best_pol] += 1

    if csv_f is not None:
        csv_f.close()

    def _mean(x): return float(np.mean(x)) if len(x) else 0.0
    def _p90(x):  return float(np.percentile(x, 90)) if len(x) else 0.0

    per_policy_stats = {
        p: {
            "n": int(d["n"]),
            "d2_abs_mean": _mean(d["abs_d2"]),
            "d2_abs_p90":  _p90(d["abs_d2"]),
            "e_bg_mean":   _mean(d["e_bg"]),
            "e_obj_mean":  _mean(d["e_obj"]),
        } for p, d in per_policy_vals.items()
    }
    wins_frac = {p: (policy_wins[p] / max(1, n_valid_anchors)) for p in donor_policies}

    return {
        "overall": {"anchors": int(n_valid_anchors), "donors_per_policy": int(donors_per_policy), "k_clusters": int(k_clusters)},
        "per_policy": per_policy_stats,
        "policy_wins": wins_frac,
    }


# ------------------ model loader ------------------

def load_model(ckpt_path, backbone, device, image_size: int = 224):
    if backbone == 'resnet50':
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif backbone == 'resnet18':
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 2)
    elif backbone == 'convnext_tiny':
        m = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
    elif backbone == 'vit_b_16':
        m = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
    elif backbone == 'vit_base_patch14_dinov2':
        m = timm.create_model('vit_base_patch14_dinov2', pretrained=True, num_classes=2, img_size=image_size)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = ckpt.get('model', ckpt)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()

# ------------------ main ------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--ckpt", type=str, default="checkpoints/erm_best.pt")
    ap.add_argument("--split", type=str, default="validation", choices=["train","validation","test"])
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--anchors", type=int, default=400)
    ap.add_argument("--donor_per_anchor", type=int, default=3)
    ap.add_argument("--data_root", type=str, default=str(_DEFAULT_REAL_ROOT),
                    help="Prepared real Waterbirds directory containing waterbirds_{train,eval,test}.csv and mask paths")
    ap.add_argument("--min_score", type=float, default=0.5)
    ap.add_argument("--use_margin", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--max_box_frac", type=float, default=0.45,
                    help="Skip anchors whose mask bbox covers more than this fraction of the image")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--save_anchors", type=str, default=None)
    ap.add_argument("--load_anchors", type=str, default=None)
    ap.add_argument("--box_cache", type=str, default=None)
    ap.add_argument("--stats_out", type=str, default=None)
    ap.add_argument("--scores_csv", type=str, default=None)
    # Aliases
    ap.add_argument("--out_csv",  dest="scores_csv", type=str, default=None)
    ap.add_argument("--out_json", dest="stats_out",  type=str, default=None)
    # multi-policy controls
    ap.add_argument("--donor_policies", type=str, default="opposite,same,random,cluster_far,cluster_near")
    ap.add_argument("--legacy_donor_policy", type=str, default="opposite", choices=["opposite","same","random"])
    ap.add_argument("--per_policy_csv", type=str, default=None)
    ap.add_argument("--save_anchor_ids", type=str, default=None)
    ap.add_argument("--load_anchor_ids", type=str, default=None)
    ap.add_argument("--k_clusters", type=int, default=2)

    # Toggles
    ap.add_argument("--skip_general", action="store_true", default=False)
    ap.add_argument("--skip_legacy", action="store_true", default=False)

    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    ds = HFWaterbirds(args.split, image_size=args.image_size, data_root=args.data_root, backbone=args.backbone)
    _ = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    model = load_model(args.ckpt, args.backbone, device, image_size=args.image_size)
    box_cache_path = resolve_box_cache_path(args.box_cache, args.split)
    if args.box_cache:
        print("[mask] ignoring --box_cache because the masked auditor uses dataset masks, not detector boxes")
    if args.min_score != 0.5:
        print("[mask] ignoring --min_score because the masked auditor does not run detection")

    # ---------- General donor audit ----------
    if not args.skip_general:
        policies = [s.strip() for s in args.donor_policies.split(",") if s.strip()]
        gen_stats = d2_audit_general(
            model, ds, device,
            donor_policies=tuple(policies),
            sample_anchors=args.anchors,
            donors_per_policy=args.donor_per_anchor,
            seed=args.seed,
            min_score=args.min_score,
            box_cache_path=box_cache_path,
            per_policy_csv=Path(args.per_policy_csv) if args.per_policy_csv else None,
            save_anchor_ids_path=Path(args.save_anchor_ids) if args.save_anchor_ids else None,
            load_anchor_ids_path=Path(args.load_anchor_ids) if args.load_anchor_ids else None,
            k_clusters=args.k_clusters,
            use_margin=bool(args.use_margin),
            max_box_frac=args.max_box_frac,
        )
        print("[general-audit]", json.dumps(gen_stats, indent=2))

    # ---------- Single-policy audit ----------
    if not args.skip_legacy:
        stats = d2_audit(
            model, ds, device,
            sample_anchors=args.anchors,
            donor_per_anchor=args.donor_per_anchor,
            seed=args.seed,
            box_cache_path=box_cache_path,
            min_score=args.min_score,
            save_anchors_path=Path(args.save_anchors) if args.save_anchors else None,
            load_anchors_path=Path(args.load_anchors) if args.load_anchors else None,
            scores_csv=Path(args.scores_csv) if args.scores_csv else None,
            donor_policy=args.legacy_donor_policy,
            use_margin=bool(args.use_margin),
            max_box_frac=args.max_box_frac,
        )
        if stats is None:
            print("No Δ² stats (detector failed on all anchors).")
            return

        print("Δ²(Object×Background) stats:", stats)
        if args.stats_out:
            outp = Path(args.stats_out)
            outp.parent.mkdir(parents=True, exist_ok=True)
            with open(outp, 'w') as f:
                json.dump(stats, f, indent=2)
            print(f"[saved] stats JSON -> {outp}")

if __name__ == "__main__":
    main()
