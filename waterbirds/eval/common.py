#!/usr/bin/env python3
"""Small helpers shared by the evaluation scripts: file hashing, JSON output, an append-only event log,
the exact uint8 -> normalized-tensor lookup used by the cached renders, and checkpoint locations.

Locations come from environment variables (see README):
  DATA_ROOT  datasets (hf_cache/, waterbird_complete95_forest2water2/, segmentations/, places365/)
  CKPT_ROOT  trained checkpoints, one folder per model and seed (e.g. full_is/seed1/checkpoints/steered_finetuned_best.pt)
  OUT_ROOT   evaluation outputs and render caches
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os

import torch

PKG = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data_root"))
CKPT = Path(os.environ.get("CKPT_ROOT", "checkpoints"))
OUT = Path(os.environ.get("OUT_ROOT", "outputs"))
LISTS = PKG / "lists"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def save(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def event(**data):
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "commands.jsonl").open("a") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **data}) + "\n")


def lookup_table():
    # Exactly the float32 ToTensor + Normalize operations used by torchvision.
    return (torch.arange(256, dtype=torch.float32)[None, :] / 255
            - torch.tensor([.485, .456, .406])[:, None]) / torch.tensor([.229, .224, .225])[:, None]


def checkpoint(seed, arm):
    if arm == "erm": return CKPT / f"erm/seed{seed}/checkpoints/erm_best.pt"
    return CKPT / f"{arm}/seed{seed}/checkpoints/steered_finetuned_best.pt"
