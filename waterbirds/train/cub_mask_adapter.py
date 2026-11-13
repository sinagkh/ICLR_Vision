#!/usr/bin/env python3
"""Run any of the fine-tuning trainers in train/ on the Waterbirds images (Hugging Face Arrow cache) with the released
CUB bird masks; same data provider as is_with_cub_masks.py.
usage: cub_mask_adapter.py <trainer.py> [trainer args...]"""
import importlib.util, os, sys
from pathlib import Path
PKG = Path(__file__).resolve().parents[1]; DATA_ROOT = Path(os.environ['DATA_ROOT'])
sys.path.insert(0, str(PKG / 'data')); sys.path.insert(0, str(PKG / 'train'))
from waterbirds_data import GroundTruthMaskWaterbirds, configure_ground_truth_class
trainer = Path(sys.argv[1]).resolve()
spec = importlib.util.spec_from_file_location('finetune_trainer', trainer)
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.CV2 = None
module.HFWaterbirds = configure_ground_truth_class(
    GroundTruthMaskWaterbirds, DATA_ROOT / 'hf_cache', module,
    DATA_ROOT / 'waterbird_complete95_forest2water2',
    DATA_ROOT / 'segmentations')
sys.argv = [str(trainer)] + sys.argv[2:]
module.main()
