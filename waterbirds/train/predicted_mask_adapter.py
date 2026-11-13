#!/usr/bin/env python3
"""Minimal adapter around train/train_is.py: replaces only the mask/data provider (predicted masks instead of CUB masks)."""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from waterbirds_data import PredictedMaskWaterbirds, configure_dataset_class


def load_module(path: Path):
    sys.path.insert(0, str(path.resolve().parent))
    spec = importlib.util.spec_from_file_location("is_trainer", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trainer-script", type=Path, required=True)
    ap.add_argument("--arrow-cache", type=Path, required=True)
    ap.add_argument("--prediction-root", type=Path, required=True)
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--clip_grad_norm", type=float, default=1.0)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--data_root", default="ignored_for_arrow_adapter")
    ap.add_argument("--load_checkpoint", required=True)
    ap.add_argument("--mode", default="suppress", choices=["suppress", "encourage"])
    ap.add_argument("--lambda_d2", type=float, default=0.05)
    ap.add_argument("--tau", type=float, default=0.10)
    ap.add_argument("--tau_floor_mult", type=float, default=0.01)
    ap.add_argument("--anchors_per_step", type=int, default=32)
    ap.add_argument("--two_donors", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--donor_policy", default="opposite", choices=["opposite", "same", "random"])
    ap.add_argument("--anchor_strategy", default="balanced", choices=["balanced", "ranked_top", "ranked_stratified"])
    ap.add_argument("--scores_csv")
    ap.add_argument("--top_pct", type=float, default=20.0)
    ap.add_argument("--ranked_filter_after_top", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--ranked_balance_by", default="none", choices=["none", "y"])
    ap.add_argument("--ranked_on_manifold_only", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--ranked_seam_l2_max", type=float, default=60.0)
    ap.add_argument("--ranked_ebg_abs_max", type=float, default=-1.0)
    ap.add_argument("--ranked_eobj_min", type=float, default=-1.0)
    ap.add_argument("--group_bias", type=float, default=0.7)
    ap.add_argument("--guard_weight", type=float, default=0.30)
    ap.add_argument("--conf_min", type=float, default=0.50)
    ap.add_argument("--min_keep_per_class", type=int, default=4)
    ap.add_argument("--det_cache_frac", type=float, default=0.0)
    ap.add_argument("--box_cache_train")
    ap.add_argument("--freeze_backbone_epochs", type=int, default=2)
    ap.add_argument("--margin_d2", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--K", type=float, default=0.0)
    ap.add_argument("--eps_obj", type=float, default=0.20)
    ap.add_argument("--tau_bg_min", type=float, default=0.10)
    ap.add_argument("--tau_bg_fac", type=float, default=0.02)
    ap.add_argument("--tau_bg_guard_min", type=float, default=0.20)
    ap.add_argument("--tau_bg_guard_fac", type=float, default=0.02)
    ap.add_argument("--max_box_frac", type=float, default=0.45)
    ap.add_argument("--lam_obj", type=float, default=0.0)
    ap.add_argument("--lam_obj_floor", type=float, default=0.0)
    ap.add_argument("--ema_obj_beta", type=float, default=0.9)
    ap.add_argument("--save_combo_checkpoint", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--combo_wga_tolerance", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main():
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--trainer-script", type=Path, required=True)
    wrapper.add_argument("--arrow-cache", type=Path, required=True)
    wrapper.add_argument("--prediction-root", type=Path, required=True)
    known, trainer_argv = wrapper.parse_known_args()
    trainer_script, arrow_cache, prediction_root = known.trainer_script, known.arrow_cache, known.prediction_root
    module = load_module(trainer_script)
    # Use the deterministic non-OpenCV inpainting fallback, as in all reported runs.
    module.CV2 = None
    module.HFWaterbirds = configure_dataset_class(PredictedMaskWaterbirds, arrow_cache, module, prediction_root)
    sys.argv = [sys.argv[0], *trainer_argv]
    module.main()


if __name__ == "__main__":
    main()
