"""Core PACO utilities: object-crop construction, part masks, removal fills, the ResNet-50 classifier,
and the true-class margin used by the interaction terms."""
from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchvision
from PIL import Image, ImageFilter
from pycocotools import mask as mask_utils
from scipy import ndimage

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ----------------------------------------------------------------------------- annotations
@lru_cache(maxsize=4)
def ann_index(annotation_json: str) -> dict[int, dict[str, Any]]:
    data = json.loads(Path(annotation_json).read_text(encoding="utf-8"))
    return {int(a["id"]): a for a in data["annotations"]}


def decode_segmentation(seg: Any, height: int, width: int) -> np.ndarray:
    if isinstance(seg, list):
        rles = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rles)
    elif isinstance(seg, dict) and isinstance(seg.get("counts"), list):
        rle = mask_utils.frPyObjects(seg, height, width)
    else:
        rle = seg
    return mask_utils.decode(rle).astype(bool)


def union_refs(anns: dict[int, dict[str, Any]], refs: list[int], height: int, width: int) -> np.ndarray:
    out = np.zeros((height, width), dtype=bool)
    for ref in refs:
        out |= decode_segmentation(anns[int(ref)].get("segmentation"), height, width)
    return out


# ----------------------------------------------------------------------------- geometry
def mask_crop_box(mask: np.ndarray, pad_frac: float, width: int, height: int) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    x1, x2 = float(xs.min()), float(xs.max() + 1)
    y1, y2 = float(ys.min()), float(ys.max() + 1)
    pad = pad_frac * max(x2 - x1, y2 - y1)
    return (
        max(0, int(round(x1 - pad))),
        max(0, int(round(y1 - pad))),
        min(width, int(round(x2 + pad))),
        min(height, int(round(y2 + pad))),
    )


def manifest_crop_box(row: dict[str, Any]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [int(round(float(v))) for v in json.loads(row["crop_bbox"])]
    return x1, y1, x2, y2


def resize_bool_mask(mask: np.ndarray, size: int) -> np.ndarray:
    img = Image.fromarray(mask.astype(np.uint8) * 255)
    return np.asarray(img.resize((size, size), Image.Resampling.NEAREST)) > 127


def dilate(mask: np.ndarray, iters: int) -> np.ndarray:
    if iters <= 0:
        return mask
    return ndimage.binary_dilation(mask, structure=np.ones((3, 3), dtype=bool), iterations=int(iters))


# ----------------------------------------------------------------------------- fills
def fill_removed(
    crop: Image.Image | np.ndarray,
    remove: np.ndarray,
    *,
    mode: str,
    rng: random.Random,
    obj_mask: np.ndarray | None = None,
    blur_radius: float = 9.0,
    mean_source: str = "keep",
    mixed_modes: tuple[str, ...] = ("mean_bg", "blur", "black", "noise"),
) -> np.ndarray:
    """Return a float32 HxWx3 array in [0, 255] with `remove` pixels filled.

    black = 0; mean_bg = unrounded mean colour of the pixels that are kept; blur = PIL Gaussian blur
    radius 9; mixed = uniform choice per corner among mean_bg / blur / black / uniform noise.
    """
    if mode == "mixed":
        mode = rng.choice(list(mixed_modes))
    base = np.asarray(crop, dtype=np.float32).copy()
    if not remove.any():
        return base
    if mode == "black":
        base[remove] = 0.0
    elif mode == "mean_bg":
        if mean_source == "bg" and obj_mask is not None and (~obj_mask).any():
            src = ~obj_mask
        else:
            src = ~remove if (~remove).any() else np.ones_like(remove)
        base[remove] = base[src].reshape(-1, 3).mean(axis=0)
    elif mode == "blur":
        img = crop if isinstance(crop, Image.Image) else Image.fromarray(np.asarray(crop).astype(np.uint8))
        blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=blur_radius)), dtype=np.float32)
        base[remove] = blurred[remove]
    elif mode == "noise":
        nrng = np.random.default_rng(rng.getrandbits(32))
        base[remove] = nrng.integers(0, 256, size=(int(remove.sum()), 3)).astype(np.float32)
    else:
        raise ValueError(mode)
    return base


def to_tensor(img: Image.Image | np.ndarray) -> torch.Tensor:
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).copy())


# ----------------------------------------------------------------------------- corner builder
class CornerBuilder:
    """Presence convention: x11 = natural (A and B present), x10 = A present / B removed,
    x01 = A removed / B present, x00 = both removed."""

    def __init__(
        self,
        image_size: int = 224,
        pad_frac: float = 0.15,
        fill_mode: str = "mean_bg",
        mask_dilate: int = 1,
        crop_source: str = "mask",
        blur_radius: float = 9.0,
        mean_source: str = "keep",
        mixed_modes: tuple[str, ...] = ("mean_bg", "blur", "black", "noise"),
        resample: str = "bilinear",
    ) -> None:
        self.size = int(image_size)
        self.pad_frac = float(pad_frac)
        self.fill_mode = fill_mode
        self.mask_dilate = int(mask_dilate)
        self.crop_source = crop_source
        self.blur_radius = float(blur_radius)
        self.mean_source = mean_source
        self.mixed_modes = tuple(mixed_modes)
        self.resample = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC}[resample]

    def crop_box(self, row: dict[str, Any], obj_mask: np.ndarray | None) -> tuple[int, int, int, int]:
        width, height = int(float(row["image_width"])), int(float(row["image_height"]))
        if self.crop_source == "mask" and obj_mask is not None and obj_mask.any():
            return mask_crop_box(obj_mask, self.pad_frac, width, height)
        return manifest_crop_box(row)

    def natural(self, image: Image.Image, box: tuple[int, int, int, int]) -> Image.Image:
        return image.crop(box).resize((self.size, self.size), self.resample)

    def crop_mask(self, mask: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x1, y1, x2, y2 = box
        return dilate(resize_bool_mask(mask[y1:y2, x1:x2], self.size), self.mask_dilate)

    def corners(
        self,
        image: Image.Image,
        box: tuple[int, int, int, int],
        mask_a: np.ndarray,
        mask_b: np.ndarray,
        rng: random.Random,
        obj_mask: np.ndarray | None = None,
        fill_mode: str | None = None,
    ) -> dict[str, np.ndarray]:
        mode = fill_mode or self.fill_mode
        nat = self.natural(image, box)
        ma = self.crop_mask(mask_a, box)
        mb = self.crop_mask(mask_b, box)
        mo = self.crop_mask(obj_mask, box) if obj_mask is not None else None
        kw = dict(obj_mask=mo, blur_radius=self.blur_radius, mean_source=self.mean_source, mixed_modes=self.mixed_modes)
        return {
            "x11": np.asarray(nat, dtype=np.float32),
            "x10": fill_removed(nat, mb, mode=mode, rng=rng, **kw),
            "x01": fill_removed(nat, ma, mode=mode, rng=rng, **kw),
            "x00": fill_removed(nat, ma | mb, mode=mode, rng=rng, **kw),
        }


# ----------------------------------------------------------------------------- model + scores
def build_resnet50(num_classes: int, weights: str | None = None) -> torch.nn.Module:
    w = None if weights in (None, "none") else getattr(torchvision.models.ResNet50_Weights, weights)
    model = torchvision.models.resnet50(weights=w)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def build_vit_mae(num_classes: int, pretrained: bool = False, drop_path_rate: float = 0.0) -> torch.nn.Module:
    """ViT-B/16 with MAE pretraining (timm vit_base_patch16_224.mae), average-pooled tokens + LayerNorm head."""
    import timm
    return timm.create_model("vit_base_patch16_224.mae", pretrained=pretrained, num_classes=num_classes,
                             global_pool="avg", drop_path_rate=drop_path_rate)


def build_backbone(num_classes: int) -> torch.nn.Module:
    """Backbone chosen by $PACO_BACKBONE (resnet50 | vit_mae), without pretrained weights (filled by load_state)."""
    import os
    if os.environ.get("PACO_BACKBONE", "resnet50") == "vit_mae":
        return build_vit_mae(num_classes)
    return build_resnet50(num_classes)


def load_state(model: torch.nn.Module, path: str | Path) -> dict[str, Any]:
    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    sd = ck["model_state_dict"] if isinstance(ck, dict) and "model_state_dict" in ck else ck
    sd = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    return ck if isinstance(ck, dict) else {}


def true_class_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    z = logits.float()
    zy = z.gather(1, target.view(-1, 1)).squeeze(1)
    masked = z.clone()
    masked.scatter_(1, target.view(-1, 1), float("-inf"))
    return zy - masked.max(dim=1).values


def class_margins(logits: torch.Tensor) -> torch.Tensor:
    """m_c(x) = z_c - max_{c' != c} z_c' for every class c (N x C)."""
    z = logits.float()
    top2 = z.topk(2, dim=1).values
    other = torch.where(z >= top2[:, :1], top2[:, 1:2], top2[:, :1])
    return z - other
