"""Shared data and model utilities for the PACO experiments (cached crops, part-removal corners, model loading)."""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import steering as rv

# Crop cache written by data/build_cache.py: $DATA_ROOT/paco_cache/<split>/{crops,masks,boxes}.npy
CACHE = Path(os.environ.get("DATA_ROOT", ".")) / "paco_cache"
# Instance lists, anchor banks and spoof pairs shipped with the package.
LISTS = Path(__file__).resolve().parents[1] / "lists"
MEAN = torch.tensor(rv.IMAGENET_MEAN).view(1, 3, 1, 1) * 255.0
STD = torch.tensor(rv.IMAGENET_STD).view(1, 3, 1, 1) * 255.0


def normalize_gpu(x_uint8_or_float_nhwc: torch.Tensor, device: torch.device) -> torch.Tensor:
    x = x_uint8_or_float_nhwc.to(device, non_blocking=True).float().permute(0, 3, 1, 2)
    x = (x - MEAN.to(device)) / STD.to(device)
    return x.contiguous(memory_format=torch.channels_last)


def read_rows(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


class CachedNatural(Dataset):
    def __init__(self, split: str, labels: list[int] | None = None):
        self.crops = np.load(CACHE / split / "crops.npy", mmap_mode="r")
        self.labels = labels if labels is not None else [-1] * len(self.crops)

    def __len__(self) -> int:
        return len(self.crops)

    def __getitem__(self, i: int):
        return torch.from_numpy(np.array(self.crops[i])), int(self.labels[i]), i


def random_blob(obj: np.ndarray, n: int, rng: random.Random) -> np.ndarray:
    """n object pixels nearest to a random object pixel (area-matched random region inside the object)."""
    ys, xs = np.nonzero(obj)
    out = np.zeros_like(obj, dtype=bool)
    if len(ys) == 0 or n <= 0:
        return out
    n = min(int(n), len(ys))
    c = rng.randrange(len(ys))
    d = (ys - ys[c]) ** 2 + (xs - xs[c]) ** 2
    sel = np.argpartition(d, n - 1)[:n]
    out[ys[sel], xs[sel]] = True
    return out


class CachedCorners(Dataset):
    """Four corners (presence convention: x11 natural, x10 = B removed, x01 = A removed, x00 = both removed)."""

    def __init__(self, split: str, indices: list[int], labels: list[int], fill_mode: str, seed: int, random_masks: bool = False, deterministic: bool = False):
        self.split, self.indices, self.labels = split, list(indices), list(labels)
        self.fill_mode, self.seed, self.random_masks, self.deterministic = fill_mode, int(seed), bool(random_masks), bool(deterministic)
        self.crops = np.load(CACHE / split / "crops.npy", mmap_mode="r")
        self.masks = np.load(CACHE / split / "masks.npy", mmap_mode="r")
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, j: int):
        i = self.indices[j]
        rng = random.Random(self.seed * 1_000_003 + j) if self.deterministic else self.rng
        crop = np.asarray(self.crops[i])
        mk = np.asarray(self.masks[i])
        a, b = (mk & 2) > 0, (mk & 4) > 0
        if self.random_masks:
            obj = (mk & 1) > 0
            a, b = random_blob(obj, int(a.sum()), rng), random_blob(obj, int(b.sum()), rng)
        f = lambda rem: torch.from_numpy(rv.fill_removed(crop, rem, mode=self.fill_mode, rng=rng))
        return {
            "x11": torch.from_numpy(crop.astype(np.float32)),
            "x10": f(b),
            "x01": f(a),
            "x00": f(a | b),
            "y": int(self.labels[j]),
            "i": i,
        }


def worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    base = torch.initial_seed() % (2**31)
    if hasattr(ds, "rng"):
        ds.rng = random.Random(base + worker_id)
    np.random.seed((base + worker_id) % (2**32))


def load_model(path: str | Path, device: torch.device, num_classes: int = 5) -> torch.nn.Module:
    m = rv.build_backbone(num_classes)
    rv.load_state(m, path)
    return m.to(device).to(memory_format=torch.channels_last).eval()
