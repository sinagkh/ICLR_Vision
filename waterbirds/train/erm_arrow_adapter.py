#!/usr/bin/env python3
"""Run train/train_erm.py on the Waterbirds images read from the local Hugging Face Arrow cache."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from waterbirds_data import ArrowWaterbirds, configure_dataset_class


def load_module(path: Path):
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location("erm_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-script", type=Path, required=True)
    ap.add_argument("--arrow-cache", type=Path, required=True)
    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--optimizer", default="adamw")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--freeze_backbone_epochs", type=int, default=0)
    ap.add_argument("--backbone_lr_mult", type=float, default=1.0)
    ap.add_argument("--train_aug", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    trainer_script = args.trainer_script
    arrow_cache = args.arrow_cache
    delattr(args, "trainer_script")
    delattr(args, "arrow_cache")
    module = load_module(trainer_script)
    module.HFWaterbirds = configure_dataset_class(ArrowWaterbirds, arrow_cache, module)
    module.main(args)


if __name__ == "__main__":
    main()
