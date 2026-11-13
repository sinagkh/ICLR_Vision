#!/usr/bin/env python3
"""Run data/audit_interaction.py with a different mask provider (predicted masks, or the released CUB masks read via the Arrow cache)."""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from waterbirds_data import (GroundTruthMaskAuditWaterbirds, PredictedMaskAuditWaterbirds,
                               configure_dataset_class, configure_ground_truth_class)


def main():
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--audit-script", type=Path, required=True)
    wrapper.add_argument("--arrow-cache", type=Path, required=True)
    wrapper.add_argument("--prediction-root", type=Path, required=True)
    wrapper.add_argument("--mask-provider", choices=["predicted", "ground_truth"], default="predicted")
    wrapper.add_argument("--official-waterbirds-root", type=Path)
    wrapper.add_argument("--segmentation-root", type=Path)
    known, remaining = wrapper.parse_known_args()
    sys.path.insert(0, str(known.audit_script.resolve().parent))
    spec = importlib.util.spec_from_file_location("interaction_audit", known.audit_script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Use the non-OpenCV inpainting fallback, as in all reported runs.
    module.CV2 = None
    if known.mask_provider == "predicted":
        module.HFWaterbirds = configure_dataset_class(
            PredictedMaskAuditWaterbirds, known.arrow_cache, module, known.prediction_root
        )
    else:
        if known.official_waterbirds_root is None or known.segmentation_root is None:
            raise ValueError("ground_truth provider requires official and segmentation roots")
        module.HFWaterbirds = configure_ground_truth_class(
            GroundTruthMaskAuditWaterbirds, known.arrow_cache, module,
            known.official_waterbirds_root, known.segmentation_root
        )
    sys.argv = [sys.argv[0], *remaining]
    module.main()


if __name__ == "__main__":
    main()
