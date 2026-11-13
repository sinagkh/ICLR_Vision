#!/usr/bin/env python3
"""Run train/train_is.py on the Waterbirds images (Hugging Face Arrow cache) with the released CUB bird masks.

Data paths (see README): $DATA_ROOT/hf_cache (Arrow files of grodino/waterbirds),
$DATA_ROOT/waterbird_complete95_forest2water2 (official metadata.csv), $DATA_ROOT/segmentations (CUB-200-2011 masks).
usage: is_with_cub_masks.py [train_is.py arguments...]
"""
import os
from pathlib import Path
import sys
import importlib.util

PKG = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ["DATA_ROOT"])
sys.path.insert(0, str(PKG / "data"))
sys.path.insert(0, str(PKG / "train"))
from waterbirds_data import GroundTruthMaskWaterbirds, configure_ground_truth_class


def load_training_module():
    spec = importlib.util.spec_from_file_location("is_trainer", PKG / "train" / "train_is.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CV2 = None
    module.HFWaterbirds = configure_ground_truth_class(
        GroundTruthMaskWaterbirds, DATA_ROOT / "hf_cache", module,
        DATA_ROOT / "waterbird_complete95_forest2water2",
        DATA_ROOT / "segmentations")
    return module


if __name__ == "__main__":
    load_training_module().main()
