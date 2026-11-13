from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    BG_NAMES,
    CO_OCCUR_OBJ_NAMES,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    OBJ_NAMES,
    UrbanCarsClassificationDataset,
    UrbanCarsSample,
    build_bg_conditional_variants,
    build_classifier,
    build_co_conditional_variants,
    build_interaction_variants,
    compose_from_components,
    build_joint_conditional_variants,
    enumerate_urbancars_samples,
    evaluate_group_metrics,
    extract_sample_components,
    get_transforms,
    group_samples_by_factors,
    load_binary_mask,
    pick_paired_sample,
    ratio_folder_for_split,
    serialize_json,
)


VARIANT_ORDER = [
    "bg_anchor",
    "co_bg_anchor",
    "obj_bg_anchor",
    "full_anchor",
    "bg_donor",
    "co_bg_donor",
    "obj_bg_donor",
    "full_donor",
]

CTX_BUNDLE_DONOR_VARIANT_ORDER = [
    "ctx_bundle_donor",
    "full_bundle_donor",
]
BG_ONLY_VARIANT_ORDER = [
    "bg_anchor",
    "obj_bg_anchor",
    "bg_donor",
    "obj_bg_donor",
]
BG_CO_PRESENT_VARIANT_ORDER = [
    "co_bg_anchor",
    "full_anchor",
    "co_bg_donor",
    "full_donor",
]
CO_ONLY_FIXED_BG_VARIANT_ORDER = [
    "bg_only",
    "bg_co",
    "obj_bg",
    "full",
]
CO_ONLY_NO_BG_VARIANT_ORDER = [
    "empty",
    "co_only",
    "obj_only",
    "obj_co",
]
# The shared 8-corner order above matches the names used in the UrbanCars
# sanity-check panels and audit code:
#   bg_anchor     = background only on the anchor background
#   co_bg_anchor  = co-object + background on the anchor background
#   obj_bg_anchor = object + background on the anchor background
#   full_anchor   = object + co-object + background on the anchor background
#   bg_donor      = background only on the donor background
#   co_bg_donor   = co-object + background on the donor background
#   obj_bg_donor  = object + background on the donor background
#   full_donor    = object + co-object + background on the donor background


@dataclass
class AnchorBank:
    name: str
    selected_indices: list[int]
    by_bucket: dict[str, list[int]]
    probe_indices: list[int]
    probe_by_bucket: dict[str, list[int]]
    score_by_idx: dict[int, float]
    reference_score_by_idx: dict[int, float]
    summary: dict[str, object]


def empty_anchor_bank(name: str) -> AnchorBank:
    return AnchorBank(
        name=name,
        selected_indices=[],
        by_bucket={},
        probe_indices=[],
        probe_by_bucket={},
        score_by_idx={},
        reference_score_by_idx={},
        summary={
            "interaction": name,
            "strategy": "disabled",
            "selected_count": 0,
            "selected_by_bucket": {},
            "probe_count": 0,
            "probe_by_bucket": {},
        },
    )


def _score_band_select_rows(
    rows: list[dict[str, object]],
    *,
    keep_n: int,
    band_count: int,
    expand_ratio: float,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    keep_n = min(max(0, keep_n), len(rows))
    if keep_n <= 0:
        return [], {
            "band_count": 0,
            "candidate_count": 0,
            "selected_by_band": {},
            "candidate_by_band": {},
        }

    band_count = max(1, int(band_count))
    expand_ratio = max(1.0, float(expand_ratio))
    candidate_n = min(len(rows), max(keep_n, int(math.ceil(keep_n * expand_ratio))))
    candidates = list(rows[:candidate_n])

    if band_count <= 1 or candidate_n <= keep_n:
        return list(candidates[:keep_n]), {
            "band_count": 1,
            "candidate_count": candidate_n,
            "selected_by_band": {"band_0": keep_n},
            "candidate_by_band": {"band_0": candidate_n},
        }

    bands: list[list[dict[str, object]]] = []
    candidate_by_band: dict[str, int] = {}
    for band_idx in range(band_count):
        start = int(round(band_idx * candidate_n / band_count))
        end = int(round((band_idx + 1) * candidate_n / band_count))
        band_rows = list(candidates[start:end])
        if not band_rows:
            continue
        bands.append(band_rows)
        candidate_by_band[f"band_{len(bands) - 1}"] = len(band_rows)

    if not bands:
        return list(candidates[:keep_n]), {
            "band_count": 1,
            "candidate_count": candidate_n,
            "selected_by_band": {"band_0": keep_n},
            "candidate_by_band": {"band_0": candidate_n},
        }

    selected: list[dict[str, object]] = []
    selected_by_band = {f"band_{idx}": 0 for idx in range(len(bands))}
    while len(selected) < keep_n and any(band for band in bands):
        for band_idx, band_rows in enumerate(bands):
            if len(selected) >= keep_n:
                break
            if not band_rows:
                continue
            selected.append(band_rows.pop(0))
            selected_by_band[f"band_{band_idx}"] += 1

    return selected, {
        "band_count": len(candidate_by_band),
        "candidate_count": candidate_n,
        "selected_by_band": selected_by_band,
        "candidate_by_band": candidate_by_band,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune an UrbanCars ERM checkpoint with multi-interaction steering."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--load_checkpoint", type=Path, required=True)
    parser.add_argument("--scores_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--early_stop_metric", choices=["bg", "co_occur_obj", "both"], default="both")
    parser.add_argument(
        "--checkpoint_selector",
        choices=[
            "both_wga",
            "bg_wga",
            "co_wga",
            "both_slice_acc",
            "bg_slice_acc",
            "co_slice_acc",
            "paper_slice_mean",
            "paper_gap_mean",
        ],
        default="both_wga",
        help="Validation objective used to select the saved best checkpoint.",
    )
    parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--clip_grad_norm", type=float, default=1.0)
    parser.add_argument("--steer_every", type=int, default=1)
    parser.add_argument("--anchors_per_step", type=int, default=16)
    parser.add_argument(
        "--steer_chunk_size",
        type=int,
        default=128,
        help="Max number of steering variant images to forward per chunk. <=0 disables chunking.",
    )
    parser.add_argument(
        "--anchor_strategy",
        type=str,
        choices=["ranked_bucket", "ranked_top", "random"],
        default="ranked_bucket",
    )
    parser.add_argument("--bg_top_pct", type=float, default=20.0)
    parser.add_argument("--co_top_pct", type=float, default=20.0)
    parser.add_argument("--three_top_pct", type=float, default=20.0)
    parser.add_argument("--bg_bucket_min_count", type=int, default=0)
    parser.add_argument("--co_bucket_min_count", type=int, default=0)
    parser.add_argument("--three_bucket_min_count", type=int, default=0)
    parser.add_argument("--bg_bucket_min_frac_of_max", type=float, default=0.0)
    parser.add_argument("--co_bucket_min_frac_of_max", type=float, default=0.0)
    parser.add_argument("--three_bucket_min_frac_of_max", type=float, default=0.0)
    parser.add_argument(
        "--score_band_count",
        type=int,
        default=1,
        help="If >1, diversify ranked-bank selection across score bands within the retained top-score pool.",
    )
    parser.add_argument(
        "--score_band_expand_ratio",
        type=float,
        default=1.0,
        help="Expand the ranked candidate pool before score-band diversification is applied.",
    )
    parser.add_argument("--probe_per_bucket", type=int, default=24)
    parser.add_argument("--lambda_bg", type=float, default=0.05)
    parser.add_argument("--lambda_co", type=float, default=0.03)
    parser.add_argument("--lambda_three", type=float, default=0.0)
    parser.add_argument(
        "--lambda_ctx_bundle",
        type=float,
        default=0.0,
        help="Auxiliary penalty on an object x bundled-context interaction that toggles background and co-object together.",
    )
    parser.add_argument(
        "--bg_penalty_mode",
        type=str,
        choices=["hinge_l2", "zero_l2", "ref_frac_hinge"],
        default="hinge_l2",
    )
    parser.add_argument(
        "--co_penalty_mode",
        type=str,
        choices=["hinge_l2", "zero_l2", "ref_frac_hinge"],
        default="hinge_l2",
    )
    parser.add_argument(
        "--three_penalty_mode",
        type=str,
        choices=["hinge_l2", "zero_l2", "ref_frac_hinge"],
        default="hinge_l2",
    )
    parser.add_argument(
        "--ctx_bundle_penalty_mode",
        type=str,
        choices=["hinge_l2", "zero_l2", "ref_frac_hinge"],
        default="ref_frac_hinge",
    )
    parser.add_argument("--tau_bg", type=float, default=1.0)
    parser.add_argument("--tau_co", type=float, default=0.5)
    parser.add_argument("--tau_three", type=float, default=0.5)
    parser.add_argument("--tau_ctx_bundle", type=float, default=0.5)
    parser.add_argument("--bg_ref_frac", type=float, default=1.0)
    parser.add_argument("--co_ref_frac", type=float, default=1.0)
    parser.add_argument("--three_ref_frac", type=float, default=1.0)
    parser.add_argument("--ctx_bundle_ref_frac", type=float, default=1.0)
    parser.add_argument(
        "--ctx_bundle_ref_mode",
        type=str,
        choices=["max", "mean", "l2"],
        default="l2",
        help="How to derive the bundled-context reference scale from the bg/co reference magnitudes.",
    )
    parser.add_argument(
        "--ctx_bundle_donor_mode",
        type=str,
        choices=["paired_opposite", "random_paired_opposite", "random_independent_opposite"],
        default="paired_opposite",
        help=(
            "How to construct the bundled-context donor: deterministic paired opposite context, "
            "randomly sampled paired opposite context, or random independent opposite background "
            "and co-object donors."
        ),
    )
    parser.add_argument(
        "--bg_delta_mode",
        type=str,
        choices=["co_absent", "co_present"],
        default="co_absent",
        help=(
            "How to define the object x background interaction: either in the co-absent slice "
            "(default) or with the co-object held present across all bg corners."
        ),
    )
    parser.add_argument(
        "--co_delta_mode",
        type=str,
        choices=["bg_fixed", "bg_absent"],
        default="bg_fixed",
        help=(
            "How to define the object x co-object interaction: either at fixed background "
            "(default) or in a background-absent 4-corner panel."
        ),
    )
    parser.add_argument(
        "--exclusive_interaction",
        type=str,
        choices=["none", "bg", "co"],
        default="none",
        help=(
            "Restrict steering to a single interaction family by only forwarding the matching "
            "four corners during the steering step."
        ),
    )
    parser.add_argument(
        "--bg_normalizer",
        type=str,
        choices=["none", "obj_aligned", "obj_conflict", "context_main_max", "obj_plus_ctx"],
        default="none",
    )
    parser.add_argument(
        "--co_normalizer",
        type=str,
        choices=["none", "obj_aligned", "obj_conflict", "context_main_max", "obj_plus_ctx"],
        default="none",
    )
    parser.add_argument(
        "--three_normalizer",
        type=str,
        choices=["none", "obj_aligned", "obj_conflict", "context_main_max", "obj_plus_ctx"],
        default="none",
    )
    parser.add_argument("--normalizer_eps", type=float, default=1e-6)
    parser.add_argument("--anchor_mix_bg", type=float, default=0.5)
    parser.add_argument("--anchor_mix_co", type=float, default=0.5)
    parser.add_argument("--anchor_mix_three", type=float, default=0.0)
    parser.add_argument("--lam_obj", type=float, default=0.0)
    parser.add_argument("--lam_obj_guard", type=float, default=0.0)
    parser.add_argument("--obj_guard_floor", type=float, default=0.0)
    parser.add_argument("--lam_bg_main", type=float, default=0.0)
    parser.add_argument("--lam_co_main", type=float, default=0.0)
    parser.add_argument("--lam_obj_aligned", type=float, default=0.0)
    parser.add_argument("--lam_obj_aligned_guard", type=float, default=0.0)
    parser.add_argument("--obj_aligned_floor", type=float, default=0.0)
    parser.add_argument("--lam_obj_ctx_guard", type=float, default=0.0)
    parser.add_argument("--obj_ctx_margin", type=float, default=0.0)
    parser.add_argument("--lam_obj_ctx_reward", type=float, default=0.0)
    parser.add_argument("--lam_obj_conflict_guard", type=float, default=0.0)
    parser.add_argument("--obj_conflict_floor", type=float, default=0.0)
    parser.add_argument(
        "--lam_obj_conflict_rel_guard",
        type=float,
        default=0.0,
        help="Penalty weight for keeping object effect in conflicting contexts above a relative fraction of aligned-context object effect.",
    )
    parser.add_argument(
        "--obj_conflict_rel_rho",
        type=float,
        default=0.0,
        help="Relative floor coefficient: target conflict object effect >= rho * aligned object effect.",
    )
    parser.add_argument("--lam_ref_margin_guard", type=float, default=0.0)
    parser.add_argument("--ref_margin_slack", type=float, default=0.0)
    parser.add_argument(
        "--min_obj_mask_frac",
        type=float,
        default=0.0,
        help="Drop anchor-bank candidates whose object-mask area fraction is below this threshold.",
    )
    parser.add_argument(
        "--min_obj_effect_bg_anchor",
        type=float,
        default=0.0,
        help="Drop anchor-bank candidates whose audit-time object effect on the anchor background is below this threshold.",
    )
    parser.add_argument("--steer_during_freeze", action="store_true")
    parser.add_argument(
        "--recovery_only",
        action="store_true",
        help="Disable interaction-suppression terms for this stage and keep CE plus guards/rewards only.",
    )
    parser.add_argument("--probe_every", type=int, default=1)
    parser.add_argument("--save_every", type=int, default=0)
    return parser.parse_args()


def choose_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_ROOT / f"steered_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def append_jsonl(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def object_mask_fraction(sample: UrbanCarsSample) -> float | None:
    if sample.obj_mask_path is None or not sample.obj_mask_path.exists():
        return None
    mask = load_binary_mask(sample.obj_mask_path)
    return float(mask.mean())


SELECTOR_SPECS: dict[str, tuple[str, callable]] = {
    "both_wga": ("val_both_worst_group_acc", lambda m: float(m["val_both_worst_group_acc"])),
    "bg_wga": ("val_bg_worst_group_acc", lambda m: float(m["val_bg_worst_group_acc"])),
    "co_wga": ("val_co_occur_obj_worst_group_acc", lambda m: float(m["val_co_occur_obj_worst_group_acc"])),
    "both_slice_acc": (
        "val_bg_absent_co_occur_obj_absent_acc",
        lambda m: float(m["val_bg_absent_co_occur_obj_absent_acc"]),
    ),
    "bg_slice_acc": (
        "val_bg_absent_co_occur_obj_present_acc",
        lambda m: float(m["val_bg_absent_co_occur_obj_present_acc"]),
    ),
    "co_slice_acc": (
        "val_bg_present_co_occur_obj_absent_acc",
        lambda m: float(m["val_bg_present_co_occur_obj_absent_acc"]),
    ),
    "paper_slice_mean": (
        "val_paper_slice_mean",
        lambda m: float(
            (
                m["val_bg_absent_co_occur_obj_present_acc"]
                + m["val_bg_present_co_occur_obj_absent_acc"]
                + m["val_bg_absent_co_occur_obj_absent_acc"]
            )
            / 3.0
        ),
    ),
    "paper_gap_mean": (
        "val_paper_gap_mean",
        lambda m: float((m["val_bg_gap"] + m["val_co_occur_obj_gap"] + m["val_both_gap"]) / 3.0),
    ),
}


def compute_selector_value(metrics: dict[str, float], selector: str) -> float:
    try:
        _, fn = SELECTOR_SPECS[selector]
    except KeyError as exc:
        raise ValueError(f"Unknown checkpoint selector: {selector}") from exc
    return fn(metrics)


def summarize_epoch_selection(record: dict[str, object]) -> dict[str, object]:
    return {
        "epoch": int(record["epoch"]),
        "selector_value": float(record["selector_value"]),
        "val_both_wga": float(record["val_both_worst_group_acc"]),
        "val_bg_wga": float(record["val_bg_worst_group_acc"]),
        "val_co_wga": float(record["val_co_occur_obj_worst_group_acc"]),
        "val_bg_gap": float(record["val_bg_gap"]),
        "val_co_gap": float(record["val_co_occur_obj_gap"]),
        "val_both_gap": float(record["val_both_gap"]),
        "val_id_acc": float(record["val_id_acc"]),
        "val_both_slice_acc": float(record["val_bg_absent_co_occur_obj_absent_acc"]),
        "val_bg_slice_acc": float(record["val_bg_absent_co_occur_obj_present_acc"]),
        "val_co_slice_acc": float(record["val_bg_present_co_occur_obj_absent_acc"]),
        "test_both_wga": float(record["test_both_worst_group_acc"]),
        "test_bg_wga": float(record["test_bg_worst_group_acc"]),
        "test_co_wga": float(record["test_co_occur_obj_worst_group_acc"]),
        "test_bg_gap": float(record["test_bg_gap"]),
        "test_co_gap": float(record["test_co_occur_obj_gap"]),
        "test_both_gap": float(record["test_both_gap"]),
        "test_id_acc": float(record["test_id_acc"]),
        "test_both_slice_acc": float(record["test_id_acc"] + record["test_both_gap"]),
        "test_bg_slice_acc": float(record["test_id_acc"] + record["test_bg_gap"]),
        "test_co_slice_acc": float(record["test_id_acc"] + record["test_co_occur_obj_gap"]),
        "probe_bg_abs_mean": float(record.get("probe_bg_abs_mean", float("nan"))),
        "probe_co_abs_mean": float(record.get("probe_co_abs_mean", float("nan"))),
        "probe_three_abs_mean": float(record.get("probe_three_abs_mean", float("nan"))),
        "probe_bg_main_abs_mean": float(record.get("probe_bg_main_abs_mean", float("nan"))),
        "probe_co_main_abs_mean": float(record.get("probe_co_main_abs_mean", float("nan"))),
        "probe_obj_aligned_abs_mean": float(record.get("probe_obj_aligned_abs_mean", float("nan"))),
        "probe_obj_conflict_abs_mean": float(record.get("probe_obj_conflict_abs_mean", float("nan"))),
        "probe_ctxmax_abs_mean": float(record.get("probe_ctxmax_abs_mean", float("nan"))),
        "probe_objctx_mean": float(record.get("probe_objctx_mean", float("nan"))),
    }


def build_selection_diagnostics(epoch_records: list[dict[str, object]]) -> dict[str, object]:
    diagnostics: dict[str, object] = {}
    for selector, (metric_name, _) in SELECTOR_SPECS.items():
        best_record = max(
            epoch_records,
            key=lambda record: (
                float(record[f"selector_{selector}"]),
                -int(record["epoch"]),
            ),
        )
        diagnostics[selector] = {
            "metric_name": metric_name,
            **summarize_epoch_selection(best_record),
        }
    return diagnostics


def set_random_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_backbone_trainable(model: torch.nn.Module, *, train_backbone: bool) -> None:
    for name, param in model.named_parameters():
        if name.startswith("fc."):
            param.requires_grad = True
        else:
            param.requires_grad = train_backbone


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
) -> None:
    state = torch.load(checkpoint_path, map_location="cpu")
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        return
    if "model" in state:
        model.load_state_dict(state["model"])
        return
    if "classifier" in state:
        model.load_state_dict(state["classifier"])
        return
    raise KeyError(
        "Unsupported checkpoint format. Expected one of: model_state_dict, model, classifier."
    )


def target_margin(logits: torch.Tensor, obj_ids: torch.Tensor) -> torch.Tensor:
    other = 1 - obj_ids
    row = torch.arange(logits.size(0), device=logits.device)
    return logits[row, obj_ids] - logits[row, other]


def summarize_values(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"count": 0}
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "abs_mean": float(np.abs(arr).mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "p10": float(np.percentile(arr, 10)),
        "median": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
    }


def summarize_overlap(indices_a: Iterable[int], indices_b: Iterable[int]) -> dict[str, int]:
    set_a = set(indices_a)
    set_b = set(indices_b)
    return {
        "count_a": len(set_a),
        "count_b": len(set_b),
        "overlap": len(set_a & set_b),
        "union": len(set_a | set_b),
    }


def unique_preserve(items: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def bucket_for_interaction(sample: UrbanCarsSample, interaction_name: str) -> str:
    if interaction_name == "bg":
        return f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
    if interaction_name == "co_occur_obj":
        return "aligned" if sample.obj_id == sample.co_occur_obj_id else "conflict"
    if interaction_name == "three_way":
        return (
            f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
            f"__co_{'aligned' if sample.obj_id == sample.co_occur_obj_id else 'conflict'}"
        )
    raise ValueError(f"Unknown interaction name: {interaction_name}")


def random_anchor_bank(
    name: str,
    samples: list[UrbanCarsSample],
    *,
    probe_per_bucket: int,
    seed: int,
) -> AnchorBank:
    rng = random.Random(seed)
    by_bucket: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        bucket = bucket_for_interaction(sample, name)
        by_bucket.setdefault(bucket, []).append(idx)
    selected_indices = unique_preserve(idx for bucket in by_bucket.values() for idx in bucket)
    probe_by_bucket: dict[str, list[int]] = {}
    for bucket, indices in sorted(by_bucket.items()):
        choices = list(indices)
        rng.shuffle(choices)
        probe_by_bucket[bucket] = choices[: min(probe_per_bucket, len(choices))]
    probe_indices = unique_preserve(idx for bucket in probe_by_bucket.values() for idx in bucket)
    summary = {
        "interaction": name,
        "strategy": "random",
        "selected_count": len(selected_indices),
        "selected_by_bucket": {bucket: len(indices) for bucket, indices in sorted(by_bucket.items())},
        "probe_count": len(probe_indices),
        "probe_by_bucket": {bucket: len(indices) for bucket, indices in sorted(probe_by_bucket.items())},
    }
    return AnchorBank(
        name=name,
        selected_indices=selected_indices,
        by_bucket=by_bucket,
        probe_indices=probe_indices,
        probe_by_bucket=probe_by_bucket,
        score_by_idx={idx: 0.0 for idx in selected_indices},
        reference_score_by_idx={idx: 0.0 for idx in selected_indices},
        summary=summary,
    )


def load_anchor_bank_from_scores(
    scores_csv: Path,
    interaction_name: str,
    *,
    image_path_to_idx: dict[str, int],
    obj_mask_frac_by_idx: dict[int, float | None] | None,
    top_pct: float,
    strategy: str,
    probe_per_bucket: int,
    min_obj_mask_frac: float,
    min_obj_effect_bg_anchor: float,
    bucket_min_count: int,
    bucket_min_frac_of_max: float,
    score_band_count: int,
    score_band_expand_ratio: float,
) -> AnchorBank:
    abs_field = {
        "bg": "abs_delta_bg",
        "co_occur_obj": "abs_delta_co_occur_obj",
        "three_way": "abs_delta_three_way",
    }[interaction_name]
    bucket_field_fn = {
        "bg": lambda row: f"source_bg_{'aligned' if int(row['bg_aligned']) == 1 else 'conflict'}",
        "co_occur_obj": lambda row: "aligned" if int(row["co_occur_obj_aligned"]) == 1 else "conflict",
        "three_way": (
            lambda row: (
                f"source_bg_{'aligned' if int(row['bg_aligned']) == 1 else 'conflict'}"
                f"__co_{'aligned' if int(row['co_occur_obj_aligned']) == 1 else 'conflict'}"
            )
        ),
    }[interaction_name]

    rows: list[dict[str, object]] = []
    filtered_low_mask = 0
    filtered_low_obj_effect = 0
    with scores_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            if abs_field not in raw or raw[abs_field] in {"", None}:
                continue
            image_path = str(Path(str(raw["image_path"])).resolve())
            if image_path not in image_path_to_idx:
                continue
            idx = image_path_to_idx[image_path]
            mask_frac = None if obj_mask_frac_by_idx is None else obj_mask_frac_by_idx.get(idx)
            if min_obj_mask_frac > 0.0 and (mask_frac is None or mask_frac < min_obj_mask_frac):
                filtered_low_mask += 1
                continue
            obj_effect_bg_anchor = float(raw.get("obj_effect_bg_anchor", 0.0) or 0.0)
            if min_obj_effect_bg_anchor > 0.0 and obj_effect_bg_anchor < min_obj_effect_bg_anchor:
                filtered_low_obj_effect += 1
                continue
            score = float(raw[abs_field])
            bucket = bucket_field_fn(raw)
            rows.append(
                {
                    "idx": idx,
                    "score": score,
                    "bucket": bucket,
                    "obj_mask_frac": mask_frac,
                    "obj_effect_bg_anchor": obj_effect_bg_anchor,
                }
            )
    if not rows:
        raise ValueError(f"No usable rows found in {scores_csv} for interaction={interaction_name}")

    by_bucket_all: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_bucket_all.setdefault(str(row["bucket"]), []).append(row)
    for bucket_rows in by_bucket_all.values():
        bucket_rows.sort(key=lambda row: float(row["score"]), reverse=True)

    if strategy == "ranked_bucket":
        selected_rows: list[dict[str, object]] = []
        selected_by_bucket: dict[str, list[int]] = {}
        selected_by_bucket_band: dict[str, dict[str, int]] = {}
        candidate_by_bucket_band: dict[str, dict[str, int]] = {}
        base_keep_by_bucket = {
            bucket: max(1, int(round(len(bucket_rows) * (top_pct / 100.0))))
            for bucket, bucket_rows in by_bucket_all.items()
        }
        max_base_keep = max(base_keep_by_bucket.values())
        for bucket, bucket_rows in sorted(by_bucket_all.items()):
            keep_n = base_keep_by_bucket[bucket]
            if bucket_min_count > 0:
                keep_n = max(keep_n, bucket_min_count)
            if bucket_min_frac_of_max > 0.0:
                keep_n = max(keep_n, int(round(max_base_keep * bucket_min_frac_of_max)))
            keep_n = min(len(bucket_rows), keep_n)
            keep_rows, band_info = _score_band_select_rows(
                bucket_rows,
                keep_n=keep_n,
                band_count=score_band_count,
                expand_ratio=score_band_expand_ratio,
            )
            selected_rows.extend(keep_rows)
            selected_by_bucket[bucket] = [int(row["idx"]) for row in keep_rows]
            selected_by_bucket_band[bucket] = {
                key: int(value) for key, value in band_info["selected_by_band"].items()
            }
            candidate_by_bucket_band[bucket] = {
                key: int(value) for key, value in band_info["candidate_by_band"].items()
            }
    elif strategy == "ranked_top":
        all_rows_sorted = sorted(rows, key=lambda row: float(row["score"]), reverse=True)
        keep_n = max(1, int(round(len(all_rows_sorted) * (top_pct / 100.0))))
        selected_rows, band_info = _score_band_select_rows(
            all_rows_sorted,
            keep_n=keep_n,
            band_count=score_band_count,
            expand_ratio=score_band_expand_ratio,
        )
        selected_by_bucket = {}
        selected_by_bucket_band = {"all": {key: int(value) for key, value in band_info["selected_by_band"].items()}}
        candidate_by_bucket_band = {"all": {key: int(value) for key, value in band_info["candidate_by_band"].items()}}
        base_keep_by_bucket = {}
        for row in selected_rows:
            selected_by_bucket.setdefault(str(row["bucket"]), []).append(int(row["idx"]))
    else:
        raise ValueError(f"Unsupported bank strategy: {strategy}")

    probe_by_bucket: dict[str, list[int]] = {}
    for bucket, indices in sorted(selected_by_bucket.items()):
        probe_by_bucket[bucket] = list(indices[: min(probe_per_bucket, len(indices))])

    selected_indices = unique_preserve(int(row["idx"]) for row in selected_rows)
    probe_indices = unique_preserve(idx for bucket in probe_by_bucket.values() for idx in bucket)
    score_by_idx = {int(row["idx"]): float(row["score"]) for row in selected_rows}
    reference_score_by_idx = {int(row["idx"]): float(row["score"]) for row in rows}
    summary = {
        "interaction": interaction_name,
        "strategy": strategy,
        "top_pct": float(top_pct),
        "csv": str(scores_csv),
        "min_obj_mask_frac": float(min_obj_mask_frac),
        "min_obj_effect_bg_anchor": float(min_obj_effect_bg_anchor),
        "bucket_min_count": int(bucket_min_count),
        "bucket_min_frac_of_max": float(bucket_min_frac_of_max),
        "score_band_count": int(score_band_count),
        "score_band_expand_ratio": float(score_band_expand_ratio),
        "source_count": len(rows),
        "filtered_low_mask": int(filtered_low_mask),
        "filtered_low_obj_effect_bg_anchor": int(filtered_low_obj_effect),
        "source_by_bucket": {bucket: len(bucket_rows) for bucket, bucket_rows in sorted(by_bucket_all.items())},
        "base_selected_by_bucket": {
            bucket: int(base_keep_by_bucket.get(bucket, len(indices)))
            for bucket, indices in sorted(selected_by_bucket.items())
        },
        "selected_count": len(selected_indices),
        "selected_by_bucket": {bucket: len(indices) for bucket, indices in sorted(selected_by_bucket.items())},
        "selected_extra_by_bucket": {
            bucket: int(len(indices) - base_keep_by_bucket.get(bucket, len(indices)))
            for bucket, indices in sorted(selected_by_bucket.items())
        },
        "selected_by_bucket_band": selected_by_bucket_band,
        "candidate_by_bucket_band": candidate_by_bucket_band,
        "probe_count": len(probe_indices),
        "probe_by_bucket": {bucket: len(indices) for bucket, indices in sorted(probe_by_bucket.items())},
        "selected_abs_score_summary": summarize_values(score_by_idx.values()),
        "selected_obj_mask_frac_summary": summarize_values(
            row["obj_mask_frac"] for row in selected_rows if row["obj_mask_frac"] is not None
        ),
        "selected_obj_effect_bg_anchor_summary": summarize_values(
            row["obj_effect_bg_anchor"] for row in selected_rows
        ),
    }
    return AnchorBank(
        name=interaction_name,
        selected_indices=selected_indices,
        by_bucket=selected_by_bucket,
        probe_indices=probe_indices,
        probe_by_bucket=probe_by_bucket,
        score_by_idx=score_by_idx,
        reference_score_by_idx=reference_score_by_idx,
        summary=summary,
    )


def sample_indices_from_bank(
    bank: AnchorBank,
    *,
    k: int,
    rng: random.Random,
) -> list[int]:
    if k <= 0 or not bank.selected_indices:
        return []
    bucket_names = sorted(name for name, indices in bank.by_bucket.items() if indices)
    if not bucket_names:
        return rng.sample(bank.selected_indices, k=min(k, len(bank.selected_indices)))

    sampled: list[int] = []
    base = k // len(bucket_names)
    rem = k % len(bucket_names)
    for bucket_idx, bucket_name in enumerate(bucket_names):
        want = base + (1 if bucket_idx < rem else 0)
        if want <= 0:
            continue
        pool = bank.by_bucket[bucket_name]
        if len(pool) >= want:
            sampled.extend(rng.sample(pool, want))
        else:
            sampled.extend(pool)
    if len(sampled) < k:
        remaining_pool = [idx for idx in bank.selected_indices if idx not in sampled]
        fill_n = min(k - len(sampled), len(remaining_pool))
        if fill_n > 0:
            sampled.extend(rng.sample(remaining_pool, fill_n))
    rng.shuffle(sampled)
    return sampled


def build_variant_batch(
    anchor_indices: list[int],
    *,
    samples: list[UrbanCarsSample],
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    transform,
) -> tuple[torch.Tensor, torch.Tensor, list[UrbanCarsSample]]:
    tensors: list[torch.Tensor] = []
    obj_ids: list[int] = []
    anchor_samples: list[UrbanCarsSample] = []
    for anchor_idx in anchor_indices:
        sample = samples[anchor_idx]
        variants = build_joint_conditional_variants(sample, grouped_samples)
        anchor_samples.append(sample)
        for key in VARIANT_ORDER:
            tensors.append(transform(variants[key]))
            obj_ids.append(sample.obj_id)
    batch = torch.stack(tensors, dim=0)
    obj_tensor = torch.tensor(obj_ids, dtype=torch.long)
    return batch, obj_tensor, anchor_samples


def build_bg_only_variant_batch(
    anchor_indices: list[int],
    *,
    samples: list[UrbanCarsSample],
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    transform,
) -> tuple[torch.Tensor, torch.Tensor, list[UrbanCarsSample]]:
    tensors: list[torch.Tensor] = []
    obj_ids: list[int] = []
    anchor_samples: list[UrbanCarsSample] = []
    for anchor_idx in anchor_indices:
        sample = samples[anchor_idx]
        variants = build_bg_conditional_variants(sample, grouped_samples)
        anchor_samples.append(sample)
        for key in BG_ONLY_VARIANT_ORDER:
            tensors.append(transform(variants[key]))
            obj_ids.append(sample.obj_id)
    batch = torch.stack(tensors, dim=0)
    obj_tensor = torch.tensor(obj_ids, dtype=torch.long)
    return batch, obj_tensor, anchor_samples


def build_bg_copresent_variant_batch(
    anchor_indices: list[int],
    *,
    samples: list[UrbanCarsSample],
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    transform,
) -> tuple[torch.Tensor, torch.Tensor, list[UrbanCarsSample]]:
    tensors: list[torch.Tensor] = []
    obj_ids: list[int] = []
    anchor_samples: list[UrbanCarsSample] = []
    for anchor_idx in anchor_indices:
        sample = samples[anchor_idx]
        variants = build_joint_conditional_variants(sample, grouped_samples)
        anchor_samples.append(sample)
        for key in BG_CO_PRESENT_VARIANT_ORDER:
            tensors.append(transform(variants[key]))
            obj_ids.append(sample.obj_id)
    batch = torch.stack(tensors, dim=0)
    obj_tensor = torch.tensor(obj_ids, dtype=torch.long)
    return batch, obj_tensor, anchor_samples


def build_co_fixedbg_variant_batch(
    anchor_indices: list[int],
    *,
    samples: list[UrbanCarsSample],
    transform,
) -> tuple[torch.Tensor, torch.Tensor, list[UrbanCarsSample]]:
    tensors: list[torch.Tensor] = []
    obj_ids: list[int] = []
    anchor_samples: list[UrbanCarsSample] = []
    for anchor_idx in anchor_indices:
        sample = samples[anchor_idx]
        variants = build_co_conditional_variants(sample)
        anchor_samples.append(sample)
        for key in CO_ONLY_FIXED_BG_VARIANT_ORDER:
            tensors.append(transform(variants[key]))
            obj_ids.append(sample.obj_id)
    batch = torch.stack(tensors, dim=0)
    obj_tensor = torch.tensor(obj_ids, dtype=torch.long)
    return batch, obj_tensor, anchor_samples


def build_co_nobg_variant_batch(
    anchor_indices: list[int],
    *,
    samples: list[UrbanCarsSample],
    transform,
) -> tuple[torch.Tensor, torch.Tensor, list[UrbanCarsSample]]:
    tensors: list[torch.Tensor] = []
    obj_ids: list[int] = []
    anchor_samples: list[UrbanCarsSample] = []
    for anchor_idx in anchor_indices:
        sample = samples[anchor_idx]
        variants = build_interaction_variants(sample)
        anchor_samples.append(sample)
        for key in CO_ONLY_NO_BG_VARIANT_ORDER:
            tensors.append(transform(variants[key]))
            obj_ids.append(sample.obj_id)
    batch = torch.stack(tensors, dim=0)
    obj_tensor = torch.tensor(obj_ids, dtype=torch.long)
    return batch, obj_tensor, anchor_samples


def sample_grouped_candidate(
    candidates: list[UrbanCarsSample],
    *,
    rng: random.Random | None,
    description: str,
) -> UrbanCarsSample:
    if not candidates:
        raise KeyError(f"No UrbanCars candidates found for {description}")
    if rng is None:
        return random.choice(candidates)
    return candidates[rng.randrange(len(candidates))]


def collect_grouped_candidates(
    anchor: UrbanCarsSample,
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    *,
    bg_id: int | None = None,
    co_occur_obj_id: int | None = None,
) -> list[UrbanCarsSample]:
    bg_ids = [bg_id] if bg_id is not None else [0, 1]
    co_ids = [co_occur_obj_id] if co_occur_obj_id is not None else [0, 1]
    candidates: list[UrbanCarsSample] = []
    for candidate_bg_id in bg_ids:
        for candidate_co_id in co_ids:
            candidates.extend(
                grouped_samples.get(
                    (anchor.split, anchor.obj_id, candidate_bg_id, candidate_co_id),
                    [],
                )
            )
    return candidates


def build_ctx_bundle_donor_batch(
    anchor_indices: list[int],
    *,
    samples: list[UrbanCarsSample],
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    transform,
    ctx_bundle_donor_mode: str = "paired_opposite",
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    tensors: list[torch.Tensor] = []
    obj_ids: list[int] = []
    for anchor_idx in anchor_indices:
        sample = samples[anchor_idx]
        anchor_components = extract_sample_components(sample)
        if ctx_bundle_donor_mode == "paired_opposite":
            donor_bg_sample = pick_paired_sample(
                sample,
                grouped_samples,
                bg_id=1 - sample.bg_id,
                co_occur_obj_id=1 - sample.co_occur_obj_id,
            )
            donor_co_sample = donor_bg_sample
        elif ctx_bundle_donor_mode == "random_paired_opposite":
            donor_candidates = collect_grouped_candidates(
                sample,
                grouped_samples,
                bg_id=1 - sample.bg_id,
                co_occur_obj_id=1 - sample.co_occur_obj_id,
            )
            donor_bg_sample = sample_grouped_candidate(
                donor_candidates,
                rng=rng,
                description=(
                    f"split={sample.split}, obj={sample.obj_id}, "
                    f"bg={1 - sample.bg_id}, co={1 - sample.co_occur_obj_id}"
                ),
            )
            donor_co_sample = donor_bg_sample
        elif ctx_bundle_donor_mode == "random_independent_opposite":
            donor_bg_candidates = collect_grouped_candidates(
                sample,
                grouped_samples,
                bg_id=1 - sample.bg_id,
                co_occur_obj_id=None,
            )
            donor_co_candidates = collect_grouped_candidates(
                sample,
                grouped_samples,
                bg_id=None,
                co_occur_obj_id=1 - sample.co_occur_obj_id,
            )
            donor_bg_sample = sample_grouped_candidate(
                donor_bg_candidates,
                rng=rng,
                description=(
                    f"split={sample.split}, obj={sample.obj_id}, any co, bg={1 - sample.bg_id}"
                ),
            )
            donor_co_sample = sample_grouped_candidate(
                donor_co_candidates,
                rng=rng,
                description=(
                    f"split={sample.split}, obj={sample.obj_id}, any bg, co={1 - sample.co_occur_obj_id}"
                ),
            )
        else:
            raise ValueError(f"Unsupported ctx_bundle_donor_mode: {ctx_bundle_donor_mode}")
        donor_background = extract_sample_components(donor_bg_sample)["background"]
        donor_components = extract_sample_components(donor_co_sample)
        donor_ctx = compose_from_components(
            background=donor_background,
            co_rgb=donor_components["co_rgb"],
            co_alpha=donor_components["co_alpha"],
        )
        donor_full = compose_from_components(
            background=donor_background,
            obj_rgb=anchor_components["obj_rgb"],
            obj_alpha=anchor_components["obj_alpha"],
            co_rgb=donor_components["co_rgb"],
            co_alpha=donor_components["co_alpha"],
        )
        donor_variants = {
            "ctx_bundle_donor": donor_ctx,
            "full_bundle_donor": donor_full,
        }
        for key in CTX_BUNDLE_DONOR_VARIANT_ORDER:
            tensors.append(transform(donor_variants[key].clip(0, 255).astype(np.uint8)))
            obj_ids.append(sample.obj_id)
    batch = torch.stack(tensors, dim=0)
    obj_tensor = torch.tensor(obj_ids, dtype=torch.long)
    return batch, obj_tensor


def combine_reference_abs(
    bg_ref_abs: torch.Tensor,
    co_ref_abs: torch.Tensor,
    *,
    mode: str,
) -> torch.Tensor:
    if mode == "max":
        return torch.maximum(bg_ref_abs, co_ref_abs)
    if mode == "mean":
        return 0.5 * (bg_ref_abs + co_ref_abs)
    if mode == "l2":
        return torch.sqrt(bg_ref_abs.pow(2) + co_ref_abs.pow(2))
    raise ValueError(f"Unsupported ctx_bundle_ref_mode: {mode}")


def interaction_bundle(
    model: torch.nn.Module,
    *,
    anchor_indices: list[int],
    samples: list[UrbanCarsSample],
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    transform,
    device: torch.device,
    amp_enabled: bool,
    steer_chunk_size: int,
    include_ctx_bundle: bool = False,
    ctx_bundle_donor_mode: str = "paired_opposite",
    bg_delta_mode: str = "co_absent",
    co_delta_mode: str = "bg_fixed",
    exclusive_interaction: str = "none",
    rng: random.Random | None = None,
) -> dict[str, object] | None:
    if not anchor_indices:
        return None

    if exclusive_interaction == "bg":
        if bg_delta_mode == "co_absent":
            batch_cpu, obj_ids_cpu, anchor_samples = build_bg_only_variant_batch(
                anchor_indices,
                samples=samples,
                grouped_samples=grouped_samples,
                transform=transform,
            )
            variant_count = len(BG_ONLY_VARIANT_ORDER)
        elif bg_delta_mode == "co_present":
            batch_cpu, obj_ids_cpu, anchor_samples = build_bg_copresent_variant_batch(
                anchor_indices,
                samples=samples,
                grouped_samples=grouped_samples,
                transform=transform,
            )
            variant_count = len(BG_CO_PRESENT_VARIANT_ORDER)
        else:
            raise ValueError(f"Unsupported bg_delta_mode for exclusive bg: {bg_delta_mode}")
        batch = batch_cpu.to(device, non_blocking=True)
        obj_ids = obj_ids_cpu.to(device, non_blocking=True)
        if steer_chunk_size <= 0 or steer_chunk_size >= batch.size(0):
            with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                logits = model(batch)
        else:
            logits_chunks: list[torch.Tensor] = []
            for start in range(0, batch.size(0), steer_chunk_size):
                end = min(start + steer_chunk_size, batch.size(0))
                with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                    logits_chunks.append(model(batch[start:end]))
            logits = torch.cat(logits_chunks, dim=0)
        margins = target_margin(logits, obj_ids).view(len(anchor_indices), variant_count)
        if bg_delta_mode == "co_absent":
            bg_anchor = margins[:, 0]
            obj_bg_anchor = margins[:, 1]
            bg_donor = margins[:, 2]
            obj_bg_donor = margins[:, 3]
            obj_effect_anchor = obj_bg_anchor - bg_anchor
            obj_effect_donor = obj_bg_donor - bg_donor
            bg_main_anchor = bg_anchor
            bg_main_donor = bg_donor
        else:
            co_bg_anchor = margins[:, 0]
            full_anchor = margins[:, 1]
            co_bg_donor = margins[:, 2]
            full_donor = margins[:, 3]
            obj_effect_anchor = full_anchor - co_bg_anchor
            obj_effect_donor = full_donor - co_bg_donor
            bg_main_anchor = co_bg_anchor
            bg_main_donor = co_bg_donor
        anchor_bg_is_aligned = torch.tensor(
            [sample.obj_id == sample.bg_id for sample in anchor_samples],
            device=device,
            dtype=torch.bool,
        )
        bg_delta = torch.where(
            anchor_bg_is_aligned,
            obj_effect_anchor - obj_effect_donor,
            obj_effect_donor - obj_effect_anchor,
        )
        bg_main = torch.where(
            anchor_bg_is_aligned,
            bg_main_anchor - bg_main_donor,
            bg_main_donor - bg_main_anchor,
        )
        obj_effect_aligned = torch.where(anchor_bg_is_aligned, obj_effect_anchor, obj_effect_donor)
        obj_effect_conflict = torch.where(anchor_bg_is_aligned, obj_effect_donor, obj_effect_anchor)
        co_is_aligned = [sample.obj_id == sample.co_occur_obj_id for sample in anchor_samples]
        zeros = torch.zeros_like(bg_delta)
        context_main_max = bg_main.abs()
        obj_vs_ctx_gap = obj_effect_aligned - context_main_max
        bg_buckets = [
            f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
            for sample in anchor_samples
        ]
        co_buckets = ["aligned" if flag else "conflict" for flag in co_is_aligned]
        three_buckets = [
            (
                f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
                f"__co_{'aligned' if sample.obj_id == sample.co_occur_obj_id else 'conflict'}"
            )
            for sample in anchor_samples
        ]
        return {
            "anchor_indices": anchor_indices,
            "bg_delta": bg_delta,
            "co_delta": zeros,
            "three_delta": zeros,
            "bg_main": bg_main,
            "co_main": zeros,
            "obj_effect": 0.5 * (obj_effect_anchor + obj_effect_donor),
            "obj_effect_aligned": obj_effect_aligned,
            "obj_effect_conflict": obj_effect_conflict,
            "context_main_max": context_main_max,
            "obj_vs_ctx_gap": obj_vs_ctx_gap,
            "bg_buckets": bg_buckets,
            "co_buckets": co_buckets,
            "three_buckets": three_buckets,
            "ctx_bundle_delta": None,
            "ctx_bundle_buckets": three_buckets,
        }

    if exclusive_interaction == "co":
        if co_delta_mode == "bg_absent":
            batch_cpu, obj_ids_cpu, anchor_samples = build_co_nobg_variant_batch(
                anchor_indices,
                samples=samples,
                transform=transform,
            )
            variant_count = len(CO_ONLY_NO_BG_VARIANT_ORDER)
        elif co_delta_mode == "bg_fixed":
            batch_cpu, obj_ids_cpu, anchor_samples = build_co_fixedbg_variant_batch(
                anchor_indices,
                samples=samples,
                transform=transform,
            )
            variant_count = len(CO_ONLY_FIXED_BG_VARIANT_ORDER)
        else:
            raise ValueError(f"Unsupported co_delta_mode for exclusive co: {co_delta_mode}")
        batch = batch_cpu.to(device, non_blocking=True)
        obj_ids = obj_ids_cpu.to(device, non_blocking=True)
        if steer_chunk_size <= 0 or steer_chunk_size >= batch.size(0):
            with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                logits = model(batch)
        else:
            logits_chunks: list[torch.Tensor] = []
            for start in range(0, batch.size(0), steer_chunk_size):
                end = min(start + steer_chunk_size, batch.size(0))
                with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                    logits_chunks.append(model(batch[start:end]))
            logits = torch.cat(logits_chunks, dim=0)
        margins = target_margin(logits, obj_ids).view(len(anchor_indices), variant_count)
        if co_delta_mode == "bg_absent":
            empty = margins[:, 0]
            co_only = margins[:, 1]
            obj_only = margins[:, 2]
            obj_co = margins[:, 3]
            co_delta = obj_only - empty - obj_co + co_only
            co_main_raw = co_only - empty
            obj_effect_left = obj_only - empty
            obj_effect_right = obj_co - co_only
        else:
            bg_only = margins[:, 0]
            bg_co = margins[:, 1]
            obj_bg = margins[:, 2]
            full = margins[:, 3]
            co_delta = obj_bg - bg_only - full + bg_co
            co_main_raw = bg_co - bg_only
            obj_effect_left = obj_bg - bg_only
            obj_effect_right = full - bg_co
        co_is_aligned_tensor = torch.tensor(
            [sample.obj_id == sample.co_occur_obj_id for sample in anchor_samples],
            device=device,
            dtype=torch.bool,
        )
        co_main = torch.where(co_is_aligned_tensor, co_main_raw, -co_main_raw)
        obj_effect = 0.5 * (obj_effect_left + obj_effect_right)
        zeros = torch.zeros_like(co_delta)
        context_main_max = co_main.abs()
        obj_vs_ctx_gap = obj_effect - context_main_max
        bg_buckets = [
            f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
            for sample in anchor_samples
        ]
        co_buckets = ["aligned" if flag else "conflict" for flag in co_is_aligned_tensor.tolist()]
        three_buckets = [
            (
                f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
                f"__co_{'aligned' if sample.obj_id == sample.co_occur_obj_id else 'conflict'}"
            )
            for sample in anchor_samples
        ]
        return {
            "anchor_indices": anchor_indices,
            "bg_delta": zeros,
            "co_delta": co_delta,
            "three_delta": zeros,
            "bg_main": zeros,
            "co_main": co_main,
            "obj_effect": obj_effect,
            "obj_effect_aligned": obj_effect,
            "obj_effect_conflict": obj_effect,
            "context_main_max": context_main_max,
            "obj_vs_ctx_gap": obj_vs_ctx_gap,
            "bg_buckets": bg_buckets,
            "co_buckets": co_buckets,
            "three_buckets": three_buckets,
            "ctx_bundle_delta": None,
            "ctx_bundle_buckets": three_buckets,
        }

    batch_cpu, obj_ids_cpu, anchor_samples = build_variant_batch(
        anchor_indices,
        samples=samples,
        grouped_samples=grouped_samples,
        transform=transform,
    )
    batch = batch_cpu.to(device, non_blocking=True)
    obj_ids = obj_ids_cpu.to(device, non_blocking=True)

    if steer_chunk_size <= 0 or steer_chunk_size >= batch.size(0):
        with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
            logits = model(batch)
    else:
        logits_chunks: list[torch.Tensor] = []
        for start in range(0, batch.size(0), steer_chunk_size):
            end = min(start + steer_chunk_size, batch.size(0))
            with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                logits_chunks.append(model(batch[start:end]))
        logits = torch.cat(logits_chunks, dim=0)
    margins = target_margin(logits, obj_ids).view(len(anchor_indices), len(VARIANT_ORDER))

    bg_anchor = margins[:, 0]
    co_bg_anchor = margins[:, 1]
    obj_bg_anchor = margins[:, 2]
    full_anchor = margins[:, 3]
    bg_donor = margins[:, 4]
    co_bg_donor = margins[:, 5]
    obj_bg_donor = margins[:, 6]
    full_donor = margins[:, 7]

    if bg_delta_mode == "co_absent":
        obj_effect_anchor = obj_bg_anchor - bg_anchor
        obj_effect_donor = obj_bg_donor - bg_donor
        bg_main_anchor = bg_anchor
        bg_main_donor = bg_donor
    elif bg_delta_mode == "co_present":
        obj_effect_anchor = full_anchor - co_bg_anchor
        obj_effect_donor = full_donor - co_bg_donor
        bg_main_anchor = co_bg_anchor
        bg_main_donor = co_bg_donor
    else:
        raise ValueError(f"Unsupported bg_delta_mode: {bg_delta_mode}")

    anchor_bg_is_aligned = torch.tensor(
        [sample.obj_id == sample.bg_id for sample in anchor_samples],
        device=device,
        dtype=torch.bool,
    )
    co_is_aligned = [sample.obj_id == sample.co_occur_obj_id for sample in anchor_samples]
    co_is_aligned_tensor = torch.tensor(
        co_is_aligned,
        device=device,
        dtype=torch.bool,
    )

    # Object x background:
    #   Delta_bg_anchor = f(obj_bg_anchor) - f(bg_anchor)
    #   Delta_bg_donor  = f(obj_bg_donor)  - f(bg_donor)
    #   delta_bg_raw    = Delta_bg_donor - Delta_bg_anchor
    #
    # We standardize sign so positive always means:
    #   object effect on aligned background > object effect on conflicting background.
    # Therefore:
    #   if anchor background is aligned:
    #       delta_bg = [f(obj_bg_anchor) - f(bg_anchor)]
    #                - [f(obj_bg_donor)  - f(bg_donor)]
    #   else:
    #       delta_bg = [f(obj_bg_donor)  - f(bg_donor)]
    #                - [f(obj_bg_anchor) - f(bg_anchor)]
    bg_delta = torch.where(
        anchor_bg_is_aligned,
        obj_effect_anchor - obj_effect_donor,
        obj_effect_donor - obj_effect_anchor,
    )
    bg_main = torch.where(
        anchor_bg_is_aligned,
        bg_main_anchor - bg_main_donor,
        bg_main_donor - bg_main_anchor,
    )

    # Object x co-object at fixed anchor background:
    #   delta_co
    #     = [f(obj_bg_anchor) - f(bg_anchor)]
    #     - [f(full_anchor) - f(co_bg_anchor)]
    #     = f(obj_bg_anchor) - f(bg_anchor) - f(full_anchor) + f(co_bg_anchor)
    #
    # This matches the 4-corner panel exactly as:
    #   ("bg+obj" - "bg only") - ("bg+obj+co" - "bg+co")
    if co_delta_mode == "bg_fixed":
        co_delta = obj_bg_anchor - bg_anchor - full_anchor + co_bg_anchor
        co_main_raw = co_bg_anchor - bg_anchor
    else:
        raise ValueError(
            "co_delta_mode=bg_absent requires exclusive_interaction=co to avoid background corners."
        )
    co_main = torch.where(
        co_is_aligned_tensor,
        co_main_raw,
        -co_main_raw,
    )
    three_anchor = co_delta
    three_donor = obj_bg_donor - bg_donor - full_donor + co_bg_donor
    obj_effect_aligned = torch.where(anchor_bg_is_aligned, obj_effect_anchor, obj_effect_donor)
    obj_effect_conflict = torch.where(anchor_bg_is_aligned, obj_effect_donor, obj_effect_anchor)
    context_main_max = torch.maximum(bg_main.abs(), co_main.abs())
    obj_vs_ctx_gap = obj_effect_aligned - context_main_max

    # Three-way interaction:
    #   delta_three_anchor
    #     = [f(obj_bg_anchor) - f(bg_anchor)]
    #     - [f(full_anchor) - f(co_bg_anchor)]
    #   delta_three_donor
    #     = [f(obj_bg_donor) - f(bg_donor)]
    #     - [f(full_donor) - f(co_bg_donor)]
    #
    # Then, exactly as for delta_bg, we standardize sign to aligned-minus-conflict:
    #   if anchor background is aligned:
    #       delta_three = delta_three_anchor - delta_three_donor
    #   else:
    #       delta_three = delta_three_donor - delta_three_anchor
    three_delta = torch.where(
        anchor_bg_is_aligned,
        three_anchor - three_donor,
        three_donor - three_anchor,
    )

    ctx_bundle_delta = None
    if include_ctx_bundle:
        ctx_batch_cpu, ctx_obj_ids_cpu = build_ctx_bundle_donor_batch(
            anchor_indices,
            samples=samples,
            grouped_samples=grouped_samples,
            transform=transform,
            ctx_bundle_donor_mode=ctx_bundle_donor_mode,
            rng=rng,
        )
        ctx_batch = ctx_batch_cpu.to(device, non_blocking=True)
        ctx_obj_ids = ctx_obj_ids_cpu.to(device, non_blocking=True)
        if steer_chunk_size <= 0 or steer_chunk_size >= ctx_batch.size(0):
            with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                ctx_logits = model(ctx_batch)
        else:
            ctx_chunks: list[torch.Tensor] = []
            for start in range(0, ctx_batch.size(0), steer_chunk_size):
                end = min(start + steer_chunk_size, ctx_batch.size(0))
                with torch.cuda.amp.autocast(enabled=amp_enabled and device.type == "cuda"):
                    ctx_chunks.append(model(ctx_batch[start:end]))
            ctx_logits = torch.cat(ctx_chunks, dim=0)
        ctx_margins = target_margin(ctx_logits, ctx_obj_ids).view(
            len(anchor_indices), len(CTX_BUNDLE_DONOR_VARIANT_ORDER)
        )
        ctx_bundle_donor = ctx_margins[:, 0]
        full_bundle_donor = ctx_margins[:, 1]
        ctx_bundle_delta = (full_anchor - co_bg_anchor) - (full_bundle_donor - ctx_bundle_donor)

    bg_buckets = [
        f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
        for sample in anchor_samples
    ]
    co_buckets = ["aligned" if flag else "conflict" for flag in co_is_aligned]
    three_buckets = [
        (
            f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
            f"__co_{'aligned' if sample.obj_id == sample.co_occur_obj_id else 'conflict'}"
        )
        for sample in anchor_samples
    ]

    return {
        "anchor_indices": anchor_indices,
        "bg_delta": bg_delta,
        "co_delta": co_delta,
        "three_delta": three_delta,
        "bg_main": bg_main,
        "co_main": co_main,
        "obj_effect": 0.5 * (obj_effect_anchor + obj_effect_donor),
        "obj_effect_aligned": obj_effect_aligned,
        "obj_effect_conflict": obj_effect_conflict,
        "context_main_max": context_main_max,
        "obj_vs_ctx_gap": obj_vs_ctx_gap,
        "bg_buckets": bg_buckets,
        "co_buckets": co_buckets,
        "three_buckets": three_buckets,
        "ctx_bundle_delta": ctx_bundle_delta,
        "ctx_bundle_buckets": three_buckets,
    }


def anchor_chunk_slices(anchor_indices: list[int], steer_chunk_size: int) -> list[list[int]]:
    if not anchor_indices:
        return []
    if steer_chunk_size <= 0:
        return [anchor_indices]
    anchors_per_chunk = max(1, steer_chunk_size // len(VARIANT_ORDER))
    return [
        anchor_indices[start : start + anchors_per_chunk]
        for start in range(0, len(anchor_indices), anchors_per_chunk)
    ]


def tensor_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float()
    if values.numel() == 0:
        return {"count": 0}
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "abs_mean": float(values.abs().mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def summarize_by_bucket(
    values: torch.Tensor,
    buckets: list[str],
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(buckets)):
        indices = [idx for idx, label in enumerate(buckets) if label == bucket]
        if not indices:
            continue
        bucket_values = values[indices]
        out[bucket] = tensor_summary(bucket_values)
    return out


def evaluate_probe_quantity(
    model: torch.nn.Module,
    *,
    bank: AnchorBank,
    samples: list[UrbanCarsSample],
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    transform,
    device: torch.device,
    amp_enabled: bool,
    steer_chunk_size: int,
    bg_delta_mode: str,
    co_delta_mode: str,
    exclusive_interaction: str,
    value_key: str,
    bucket_key: str,
) -> dict[str, object]:
    if not bank.probe_indices:
        return {"count": 0}
    with torch.no_grad():
        bundle = interaction_bundle(
            model,
            anchor_indices=bank.probe_indices,
            samples=samples,
            grouped_samples=grouped_samples,
            transform=transform,
            device=device,
            amp_enabled=amp_enabled,
            steer_chunk_size=steer_chunk_size,
            bg_delta_mode=bg_delta_mode,
            co_delta_mode=co_delta_mode,
            exclusive_interaction=exclusive_interaction,
        )
    assert bundle is not None
    values = bundle[value_key]
    buckets = bundle[bucket_key]
    return {
        "overall": tensor_summary(values),
        "by_bucket": summarize_by_bucket(values, buckets),
    }


def evaluate_probe_bank(
    model: torch.nn.Module,
    *,
    bank: AnchorBank,
    samples: list[UrbanCarsSample],
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    transform,
    device: torch.device,
    amp_enabled: bool,
    steer_chunk_size: int,
    bg_delta_mode: str,
    co_delta_mode: str,
    exclusive_interaction: str,
) -> dict[str, object]:
    key_map = {
        "bg": ("bg_delta", "bg_buckets"),
        "co_occur_obj": ("co_delta", "co_buckets"),
        "three_way": ("three_delta", "three_buckets"),
    }
    value_key, bucket_key = key_map[bank.name]
    return evaluate_probe_quantity(
        model,
        bank=bank,
        samples=samples,
        grouped_samples=grouped_samples,
        transform=transform,
        device=device,
        amp_enabled=amp_enabled,
        steer_chunk_size=steer_chunk_size,
        bg_delta_mode=bg_delta_mode,
        co_delta_mode=co_delta_mode,
        exclusive_interaction=exclusive_interaction,
        value_key=value_key,
        bucket_key=bucket_key,
    )


def flatten_probe_summary(probe_summary: dict[str, object], prefix: str) -> dict[str, float]:
    overall = probe_summary.get("overall", {})
    if not isinstance(overall, dict):
        return {}
    flattened: dict[str, float] = {}
    for metric_name in ["mean", "abs_mean", "std", "min", "max"]:
        metric_value = overall.get(metric_name)
        if isinstance(metric_value, (int, float)):
            flattened[f"{prefix}_{metric_name}"] = float(metric_value)
    by_bucket = probe_summary.get("by_bucket", {})
    if isinstance(by_bucket, dict):
        for bucket_name, bucket_summary in sorted(by_bucket.items()):
            if not isinstance(bucket_summary, dict):
                continue
            bucket_prefix = f"{prefix}_{bucket_name}"
            count_value = bucket_summary.get("count")
            if isinstance(count_value, (int, float)):
                flattened[f"{bucket_prefix}_count"] = float(count_value)
            for metric_name in ["mean", "abs_mean", "std", "min", "max"]:
                metric_value = bucket_summary.get(metric_name)
                if isinstance(metric_value, (int, float)):
                    flattened[f"{bucket_prefix}_{metric_name}"] = float(metric_value)
    return flattened


def interaction_penalty(
    values: torch.Tensor,
    *,
    tau: float,
    mode: str,
    reference_abs: torch.Tensor | None = None,
    ref_frac: float = 1.0,
    normalizer: torch.Tensor | None = None,
    normalizer_eps: float = 1e-6,
) -> torch.Tensor:
    if mode == "hinge_l2":
        base = torch.clamp(values.abs() - tau, min=0)
    elif mode == "zero_l2":
        base = values
    elif mode == "ref_frac_hinge":
        if reference_abs is None:
            raise ValueError("reference_abs is required when mode=ref_frac_hinge")
        base = torch.clamp(values.abs() - (reference_abs * ref_frac), min=0)
    else:
        raise ValueError(f"Unsupported penalty mode: {mode}")
    if normalizer is not None:
        base = base / torch.clamp(normalizer.detach(), min=normalizer_eps)
    return base.pow(2).mean()


def penalty_normalizer(
    kind: str,
    *,
    obj_effect_aligned: torch.Tensor,
    obj_effect_conflict: torch.Tensor,
    context_main_max: torch.Tensor,
) -> torch.Tensor | None:
    if kind == "none":
        return None
    if kind == "obj_aligned":
        return obj_effect_aligned.abs()
    if kind == "obj_conflict":
        return obj_effect_conflict.abs()
    if kind == "context_main_max":
        return context_main_max.abs()
    if kind == "obj_plus_ctx":
        return obj_effect_aligned.abs() + context_main_max.abs()
    raise ValueError(f"Unsupported normalizer kind: {kind}")


def main() -> None:
    args = parse_args()
    start_time = time.time()
    output_dir = choose_output_dir(args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    set_random_seeds(args.seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    rng = random.Random(args.seed)

    train_dataset = UrbanCarsClassificationDataset(
        data_root=args.data_root,
        split="train",
        transform=get_transforms(is_training=True),
    )
    val_dataset = UrbanCarsClassificationDataset(
        data_root=args.data_root,
        split="val",
        transform=get_transforms(is_training=False),
    )
    test_dataset = UrbanCarsClassificationDataset(
        data_root=args.data_root,
        split="test",
        transform=get_transforms(is_training=False),
    )
    steering_transform = get_transforms(is_training=False)

    all_train_samples = enumerate_urbancars_samples(args.data_root, "train", include_auxiliary_assets=True)
    grouped_train_samples = group_samples_by_factors(all_train_samples)
    image_path_to_idx = {
        str(sample.image_path.resolve()): idx
        for idx, sample in enumerate(all_train_samples)
    }
    obj_mask_frac_by_idx = {
        idx: object_mask_fraction(sample)
        for idx, sample in enumerate(all_train_samples)
    }

    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    eval_loader_kwargs = dict(loader_kwargs)
    eval_loader_kwargs["shuffle"] = False
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    model = build_classifier(args.backbone, num_classes=2, weights_name=args.weights).to(device)
    load_model_checkpoint(model, args.load_checkpoint)
    ref_model: torch.nn.Module | None = None
    if args.lam_ref_margin_guard > 0:
        ref_model = build_classifier(args.backbone, num_classes=2, weights_name=args.weights).to(device)
        load_model_checkpoint(ref_model, args.load_checkpoint)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    if args.exclusive_interaction == "bg":
        active_bank_names = {"bg"}
    elif args.exclusive_interaction == "co":
        active_bank_names = {"co_occur_obj"}
    else:
        active_bank_names = {"bg", "co_occur_obj", "three_way"}

    if args.anchor_strategy == "random":
        bg_bank = (
            random_anchor_bank("bg", all_train_samples, probe_per_bucket=args.probe_per_bucket, seed=args.seed)
            if "bg" in active_bank_names
            else empty_anchor_bank("bg")
        )
        co_bank = (
            random_anchor_bank(
                "co_occur_obj",
                all_train_samples,
                probe_per_bucket=args.probe_per_bucket,
                seed=args.seed + 1,
            )
            if "co_occur_obj" in active_bank_names
            else empty_anchor_bank("co_occur_obj")
        )
        three_bank = (
            random_anchor_bank(
                "three_way",
                all_train_samples,
                probe_per_bucket=args.probe_per_bucket,
                seed=args.seed + 2,
            )
            if "three_way" in active_bank_names
            else empty_anchor_bank("three_way")
        )
    else:
        bg_bank = (
            load_anchor_bank_from_scores(
                args.scores_csv,
                "bg",
                image_path_to_idx=image_path_to_idx,
                obj_mask_frac_by_idx=obj_mask_frac_by_idx,
                top_pct=args.bg_top_pct,
                strategy=args.anchor_strategy,
                probe_per_bucket=args.probe_per_bucket,
                min_obj_mask_frac=args.min_obj_mask_frac,
                min_obj_effect_bg_anchor=args.min_obj_effect_bg_anchor,
                bucket_min_count=args.bg_bucket_min_count,
                bucket_min_frac_of_max=args.bg_bucket_min_frac_of_max,
                score_band_count=args.score_band_count,
                score_band_expand_ratio=args.score_band_expand_ratio,
            )
            if "bg" in active_bank_names
            else empty_anchor_bank("bg")
        )
        co_bank = (
            load_anchor_bank_from_scores(
                args.scores_csv,
                "co_occur_obj",
                image_path_to_idx=image_path_to_idx,
                obj_mask_frac_by_idx=obj_mask_frac_by_idx,
                top_pct=args.co_top_pct,
                strategy=args.anchor_strategy,
                probe_per_bucket=args.probe_per_bucket,
                min_obj_mask_frac=args.min_obj_mask_frac,
                min_obj_effect_bg_anchor=args.min_obj_effect_bg_anchor,
                bucket_min_count=args.co_bucket_min_count,
                bucket_min_frac_of_max=args.co_bucket_min_frac_of_max,
                score_band_count=args.score_band_count,
                score_band_expand_ratio=args.score_band_expand_ratio,
            )
            if "co_occur_obj" in active_bank_names
            else empty_anchor_bank("co_occur_obj")
        )
        three_bank = (
            load_anchor_bank_from_scores(
                args.scores_csv,
                "three_way",
                image_path_to_idx=image_path_to_idx,
                obj_mask_frac_by_idx=obj_mask_frac_by_idx,
                top_pct=args.three_top_pct,
                strategy=args.anchor_strategy,
                probe_per_bucket=args.probe_per_bucket,
                min_obj_mask_frac=args.min_obj_mask_frac,
                min_obj_effect_bg_anchor=args.min_obj_effect_bg_anchor,
                bucket_min_count=args.three_bucket_min_count,
                bucket_min_frac_of_max=args.three_bucket_min_frac_of_max,
                score_band_count=args.score_band_count,
                score_band_expand_ratio=args.score_band_expand_ratio,
            )
            if "three_way" in active_bank_names
            else empty_anchor_bank("three_way")
        )

    banks = {"bg": bg_bank, "co_occur_obj": co_bank, "three_way": three_bank}
    selected_sets = {name: set(bank.selected_indices) for name, bank in banks.items()}
    serialize_json(
        {
            **{name: bank.summary for name, bank in banks.items()},
            "_overlap": {
                "bg__co_occur_obj": summarize_overlap(
                    selected_sets["bg"], selected_sets["co_occur_obj"]
                ),
                "bg__three_way": summarize_overlap(
                    selected_sets["bg"], selected_sets["three_way"]
                ),
                "co_occur_obj__three_way": summarize_overlap(
                    selected_sets["co_occur_obj"], selected_sets["three_way"]
                ),
                "all_three_overlap": len(
                    selected_sets["bg"]
                    & selected_sets["co_occur_obj"]
                    & selected_sets["three_way"]
                ),
            },
        },
        output_dir / "anchor_bank_summary.json",
    )
    serialize_json(
        {
            name: {
                "probe_indices": bank.probe_indices,
                "probe_by_bucket": bank.probe_by_bucket,
            }
            for name, bank in banks.items()
        },
        output_dir / "probe_indices.json",
    )

    history_path = output_dir / "history.jsonl"
    probe_history_path = output_dir / "probe_history.jsonl"
    metric_name = SELECTOR_SPECS[args.checkpoint_selector][0]
    best_metric = float("-inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] = {}
    best_probe_metrics: dict[str, object] = {}
    selection_diagnostics: dict[str, object] = {}
    epoch_records: list[dict[str, object]] = []

    config = {
        **{
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "data_root": str(args.data_root),
        "load_checkpoint": str(args.load_checkpoint),
        "scores_csv": str(args.scores_csv),
        "output_dir": str(output_dir),
        "device_resolved": str(device),
        "amp_enabled": amp_enabled,
        "train_ratio_folder": ratio_folder_for_split("train"),
        "eval_ratio_folder": ratio_folder_for_split("val"),
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
    }
    serialize_json(config, output_dir / "config.json")
    (output_dir / "train_command.txt").write_text(
        " ".join(shlex.quote(token) for token in ["python", *sys.argv]),
        encoding="utf-8",
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        freeze_backbone = epoch <= args.freeze_backbone_epochs
        set_backbone_trainable(model, train_backbone=not freeze_backbone)

        if args.steer_during_freeze:
            warmup_epoch = epoch
        else:
            warmup_epoch = max(0, epoch - args.freeze_backbone_epochs)
        warmup_scale = min(1.0, warmup_epoch / max(1, args.warmup_epochs))
        lam_bg = args.lambda_bg * warmup_scale
        lam_co = args.lambda_co * warmup_scale
        lam_three = args.lambda_three * warmup_scale
        lam_ctx_bundle = args.lambda_ctx_bundle * warmup_scale
        if args.recovery_only:
            lam_bg = 0.0
            lam_co = 0.0
            lam_three = 0.0
            lam_ctx_bundle = 0.0

        running = {
            "steps": 0,
            "total_loss": 0.0,
            "ce_loss": 0.0,
            "bg_penalty": 0.0,
            "co_penalty": 0.0,
            "three_penalty": 0.0,
            "ctx_bundle_penalty": 0.0,
            "bg_main_penalty": 0.0,
            "co_main_penalty": 0.0,
            "obj_effect": 0.0,
            "obj_effect_aligned": 0.0,
            "obj_effect_conflict": 0.0,
            "obj_guard_penalty": 0.0,
            "obj_aligned_guard_penalty": 0.0,
            "obj_ctx_guard_penalty": 0.0,
            "obj_conflict_guard_penalty": 0.0,
            "obj_conflict_rel_guard_penalty": 0.0,
            "ref_margin_guard_penalty": 0.0,
            "bg_abs_mean": 0.0,
            "co_abs_mean": 0.0,
            "three_abs_mean": 0.0,
            "ctx_bundle_abs_mean": 0.0,
            "bg_main_abs_mean": 0.0,
            "co_main_abs_mean": 0.0,
            "bg_mean": 0.0,
            "co_mean": 0.0,
            "three_mean": 0.0,
            "ctx_bundle_mean": 0.0,
            "bg_main_mean": 0.0,
            "co_main_mean": 0.0,
            "bg_sq_mean": 0.0,
            "co_sq_mean": 0.0,
            "three_sq_mean": 0.0,
            "ctx_bundle_sq_mean": 0.0,
            "bg_active_frac": 0.0,
            "co_active_frac": 0.0,
            "three_active_frac": 0.0,
            "ctx_bundle_active_frac": 0.0,
            "bg_target_abs_mean": 0.0,
            "co_target_abs_mean": 0.0,
            "three_target_abs_mean": 0.0,
            "ctx_bundle_target_abs_mean": 0.0,
            "obj_guard_active_frac": 0.0,
            "obj_aligned_guard_active_frac": 0.0,
            "obj_ctx_guard_active_frac": 0.0,
            "obj_conflict_guard_active_frac": 0.0,
            "obj_conflict_rel_guard_active_frac": 0.0,
            "obj_conflict_rel_target_mean": 0.0,
            "ref_margin_guard_active_frac": 0.0,
            "obj_vs_ctx_gap_mean": 0.0,
            "context_main_max_mean": 0.0,
            "curr_margin": 0.0,
            "ref_margin": 0.0,
            "margin_gap": 0.0,
            "anchor_count": 0.0,
            "unique_anchor_count": 0.0,
            "sampled_from_bg": 0.0,
            "sampled_from_co": 0.0,
            "sampled_from_three": 0.0,
            "sampled_bg_score": 0.0,
            "sampled_co_score": 0.0,
            "sampled_three_score": 0.0,
            "sampled_overlap_bg_co": 0.0,
            "sampled_overlap_bg_three": 0.0,
            "sampled_overlap_co_three": 0.0,
        }

        progress = tqdm(train_loader, dynamic_ncols=True, desc=f"epoch {epoch}/{args.epochs}")
        mix_sum = args.anchor_mix_bg + args.anchor_mix_co + args.anchor_mix_three
        if mix_sum <= 0:
            raise ValueError("anchor_mix_* values must sum to a positive number.")

        for step_idx, batch in enumerate(progress, start=1):
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"][:, 0].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss_ce = F.cross_entropy(logits, labels)
            curr_margin = target_margin(logits, labels)
            if ref_model is not None:
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp_enabled):
                        ref_logits = ref_model(images)
                    ref_margin = target_margin(ref_logits, labels)
                ref_margin_guard_penalty = torch.clamp(
                    (ref_margin - args.ref_margin_slack) - curr_margin,
                    min=0,
                ).pow(2).mean()
            else:
                ref_margin = None
                ref_margin_guard_penalty = torch.zeros((), device=device, dtype=loss_ce.dtype)
            base_loss = loss_ce + args.lam_ref_margin_guard * ref_margin_guard_penalty

            loss_value = float(base_loss.item())
            epoch_stats = {}
            use_steering = (not freeze_backbone) or args.steer_during_freeze
            steering_terms_active = any(
                value > 0.0
                for value in [
                    lam_bg,
                    lam_co,
                    lam_three,
                    lam_ctx_bundle,
                    args.lam_bg_main,
                    args.lam_co_main,
                    args.lam_obj,
                    args.lam_obj_guard,
                    args.lam_obj_aligned,
                    args.lam_obj_aligned_guard,
                    args.lam_obj_ctx_guard,
                    args.lam_obj_ctx_reward,
                    args.lam_obj_conflict_guard,
                    args.lam_obj_conflict_rel_guard,
                ]
            )
            do_step = True
            if use_steering and steering_terms_active and (step_idx % max(1, args.steer_every) == 0):
                k_total = min(args.anchors_per_step, len(all_train_samples))
                raw_bg = k_total * (args.anchor_mix_bg / mix_sum)
                raw_co = k_total * (args.anchor_mix_co / mix_sum)
                k_bg = int(round(raw_bg))
                k_co = int(round(raw_co))
                k_three = max(0, k_total - k_bg - k_co)

                bg_indices = sample_indices_from_bank(bg_bank, k=k_bg, rng=rng)
                co_indices = sample_indices_from_bank(co_bank, k=k_co, rng=rng)
                three_indices = sample_indices_from_bank(three_bank, k=k_three, rng=rng)
                anchor_indices = unique_preserve(bg_indices + co_indices + three_indices)

                total_anchor_count = len(anchor_indices)
                if total_anchor_count > 0:
                    weighted_stats = {
                        "bg_penalty": 0.0,
                        "co_penalty": 0.0,
                        "three_penalty": 0.0,
                        "ctx_bundle_penalty": 0.0,
                        "bg_main_penalty": 0.0,
                        "co_main_penalty": 0.0,
                        "obj_effect": 0.0,
                        "obj_effect_aligned": 0.0,
                        "obj_effect_conflict": 0.0,
                        "obj_guard_penalty": 0.0,
                        "obj_aligned_guard_penalty": 0.0,
                        "obj_ctx_guard_penalty": 0.0,
                        "obj_conflict_guard_penalty": 0.0,
                        "obj_conflict_rel_guard_penalty": 0.0,
                        "bg_abs_mean": 0.0,
                        "co_abs_mean": 0.0,
                        "three_abs_mean": 0.0,
                        "ctx_bundle_abs_mean": 0.0,
                        "bg_main_abs_mean": 0.0,
                        "co_main_abs_mean": 0.0,
                        "bg_mean": 0.0,
                        "co_mean": 0.0,
                        "three_mean": 0.0,
                        "ctx_bundle_mean": 0.0,
                        "bg_main_mean": 0.0,
                        "co_main_mean": 0.0,
                        "bg_sq_mean": 0.0,
                        "co_sq_mean": 0.0,
                        "three_sq_mean": 0.0,
                        "ctx_bundle_sq_mean": 0.0,
                        "bg_active_frac": 0.0,
                        "co_active_frac": 0.0,
                        "three_active_frac": 0.0,
                        "ctx_bundle_active_frac": 0.0,
                        "bg_target_abs_mean": 0.0,
                        "co_target_abs_mean": 0.0,
                        "three_target_abs_mean": 0.0,
                        "ctx_bundle_target_abs_mean": 0.0,
                        "obj_guard_active_frac": 0.0,
                        "obj_aligned_guard_active_frac": 0.0,
                        "obj_ctx_guard_active_frac": 0.0,
                        "obj_conflict_guard_active_frac": 0.0,
                        "obj_conflict_rel_guard_active_frac": 0.0,
                        "obj_conflict_rel_target_mean": 0.0,
                        "obj_vs_ctx_gap_mean": 0.0,
                        "context_main_max_mean": 0.0,
                    }
                    steering_objective_value = 0.0

                    if amp_enabled:
                        scaler.scale(base_loss).backward()
                    else:
                        base_loss.backward()

                    for anchor_chunk in anchor_chunk_slices(anchor_indices, args.steer_chunk_size):
                        chunk_size = len(anchor_chunk)
                        if chunk_size == 0:
                            continue
                        chunk_weight = chunk_size / total_anchor_count
                        bundle = interaction_bundle(
                            model,
                            anchor_indices=anchor_chunk,
                            samples=all_train_samples,
                            grouped_samples=grouped_train_samples,
                            transform=steering_transform,
                            device=device,
                            amp_enabled=amp_enabled,
                            steer_chunk_size=0,
                            include_ctx_bundle=lam_ctx_bundle > 0.0,
                            ctx_bundle_donor_mode=args.ctx_bundle_donor_mode,
                            bg_delta_mode=args.bg_delta_mode,
                            co_delta_mode=args.co_delta_mode,
                            exclusive_interaction=args.exclusive_interaction,
                            rng=rng,
                        )
                        if bundle is None:
                            continue

                        bg_delta = bundle["bg_delta"]
                        co_delta = bundle["co_delta"]
                        three_delta = bundle["three_delta"]
                        ctx_bundle_delta = bundle.get("ctx_bundle_delta")
                        bg_main = bundle["bg_main"]
                        co_main = bundle["co_main"]
                        obj_effect = bundle["obj_effect"]
                        obj_effect_aligned = bundle["obj_effect_aligned"]
                        obj_effect_conflict = bundle["obj_effect_conflict"]
                        context_main_max = bundle["context_main_max"]
                        obj_vs_ctx_gap = bundle["obj_vs_ctx_gap"]
                        bg_ref_abs = torch.tensor(
                            [bg_bank.reference_score_by_idx.get(idx, 0.0) for idx in anchor_chunk],
                            device=device,
                            dtype=bg_delta.dtype,
                        )
                        co_ref_abs = torch.tensor(
                            [co_bank.reference_score_by_idx.get(idx, 0.0) for idx in anchor_chunk],
                            device=device,
                            dtype=co_delta.dtype,
                        )
                        three_ref_abs = torch.tensor(
                            [three_bank.reference_score_by_idx.get(idx, 0.0) for idx in anchor_chunk],
                            device=device,
                            dtype=three_delta.dtype,
                        )
                        ctx_bundle_ref_abs = combine_reference_abs(
                            bg_ref_abs,
                            co_ref_abs,
                            mode=args.ctx_bundle_ref_mode,
                        )
                        bg_normalizer = penalty_normalizer(
                            args.bg_normalizer,
                            obj_effect_aligned=obj_effect_aligned,
                            obj_effect_conflict=obj_effect_conflict,
                            context_main_max=context_main_max,
                        )
                        co_normalizer = penalty_normalizer(
                            args.co_normalizer,
                            obj_effect_aligned=obj_effect_aligned,
                            obj_effect_conflict=obj_effect_conflict,
                            context_main_max=context_main_max,
                        )
                        three_normalizer = penalty_normalizer(
                            args.three_normalizer,
                            obj_effect_aligned=obj_effect_aligned,
                            obj_effect_conflict=obj_effect_conflict,
                            context_main_max=context_main_max,
                        )

                        bg_penalty = interaction_penalty(
                            bg_delta,
                            tau=args.tau_bg,
                            mode=args.bg_penalty_mode,
                            reference_abs=bg_ref_abs,
                            ref_frac=args.bg_ref_frac,
                            normalizer=bg_normalizer,
                            normalizer_eps=args.normalizer_eps,
                        )
                        co_penalty = interaction_penalty(
                            co_delta,
                            tau=args.tau_co,
                            mode=args.co_penalty_mode,
                            reference_abs=co_ref_abs,
                            ref_frac=args.co_ref_frac,
                            normalizer=co_normalizer,
                            normalizer_eps=args.normalizer_eps,
                        )
                        three_penalty = interaction_penalty(
                            three_delta,
                            tau=args.tau_three,
                            mode=args.three_penalty_mode,
                            reference_abs=three_ref_abs,
                            ref_frac=args.three_ref_frac,
                            normalizer=three_normalizer,
                            normalizer_eps=args.normalizer_eps,
                        )
                        if ctx_bundle_delta is None:
                            ctx_bundle_penalty = torch.zeros_like(three_penalty)
                        else:
                            ctx_bundle_penalty = interaction_penalty(
                                ctx_bundle_delta,
                                tau=args.tau_ctx_bundle,
                                mode=args.ctx_bundle_penalty_mode,
                                reference_abs=ctx_bundle_ref_abs,
                                ref_frac=args.ctx_bundle_ref_frac,
                                normalizer=None,
                                normalizer_eps=args.normalizer_eps,
                            )
                        bg_main_penalty = bg_main.pow(2).mean()
                        co_main_penalty = co_main.pow(2).mean()
                        obj_guard_penalty = torch.clamp(
                            args.obj_guard_floor - obj_effect,
                            min=0,
                        ).pow(2).mean()
                        obj_aligned_guard_penalty = torch.clamp(
                            args.obj_aligned_floor - obj_effect_aligned,
                            min=0,
                        ).pow(2).mean()
                        obj_ctx_guard_penalty = torch.clamp(
                            args.obj_ctx_margin - obj_vs_ctx_gap,
                            min=0,
                        ).pow(2).mean()
                        obj_conflict_guard_penalty = torch.clamp(
                            args.obj_conflict_floor - obj_effect_conflict,
                            min=0,
                        ).pow(2).mean()
                        obj_conflict_rel_target = torch.maximum(
                            torch.full_like(obj_effect_conflict, args.obj_conflict_floor),
                            obj_effect_aligned.detach() * args.obj_conflict_rel_rho,
                        )
                        obj_conflict_rel_guard_penalty = torch.clamp(
                            obj_conflict_rel_target - obj_effect_conflict,
                            min=0,
                        ).pow(2).mean()
                        chunk_objective = (
                            lam_bg * bg_penalty
                            + lam_co * co_penalty
                            + lam_three * three_penalty
                            + lam_ctx_bundle * ctx_bundle_penalty
                            + args.lam_bg_main * bg_main_penalty
                            + args.lam_co_main * co_main_penalty
                            + args.lam_obj_guard * obj_guard_penalty
                            + args.lam_obj_aligned_guard * obj_aligned_guard_penalty
                            + args.lam_obj_ctx_guard * obj_ctx_guard_penalty
                            + args.lam_obj_conflict_guard * obj_conflict_guard_penalty
                            + args.lam_obj_conflict_rel_guard * obj_conflict_rel_guard_penalty
                            - args.lam_obj * obj_effect.mean()
                            - args.lam_obj_aligned * obj_effect_aligned.mean()
                            - args.lam_obj_ctx_reward * obj_vs_ctx_gap.mean()
                        )
                        scaled_chunk_objective = chunk_objective * chunk_weight

                        if amp_enabled:
                            scaler.scale(scaled_chunk_objective).backward()
                        else:
                            scaled_chunk_objective.backward()

                        steering_objective_value += float(scaled_chunk_objective.detach().item())
                        weighted_stats["bg_penalty"] += float(bg_penalty.detach().item()) * chunk_weight
                        weighted_stats["co_penalty"] += float(co_penalty.detach().item()) * chunk_weight
                        weighted_stats["three_penalty"] += float(three_penalty.detach().item()) * chunk_weight
                        weighted_stats["ctx_bundle_penalty"] += float(
                            ctx_bundle_penalty.detach().item()
                        ) * chunk_weight
                        weighted_stats["bg_main_penalty"] += float(bg_main_penalty.detach().item()) * chunk_weight
                        weighted_stats["co_main_penalty"] += float(co_main_penalty.detach().item()) * chunk_weight
                        weighted_stats["obj_effect"] += float(obj_effect.detach().mean().item()) * chunk_weight
                        weighted_stats["obj_effect_aligned"] += float(obj_effect_aligned.detach().mean().item()) * chunk_weight
                        weighted_stats["obj_effect_conflict"] += float(obj_effect_conflict.detach().mean().item()) * chunk_weight
                        weighted_stats["obj_guard_penalty"] += float(obj_guard_penalty.detach().item()) * chunk_weight
                        weighted_stats["obj_aligned_guard_penalty"] += float(obj_aligned_guard_penalty.detach().item()) * chunk_weight
                        weighted_stats["obj_ctx_guard_penalty"] += float(obj_ctx_guard_penalty.detach().item()) * chunk_weight
                        weighted_stats["obj_conflict_guard_penalty"] += float(obj_conflict_guard_penalty.detach().item()) * chunk_weight
                        weighted_stats["obj_conflict_rel_guard_penalty"] += float(
                            obj_conflict_rel_guard_penalty.detach().item()
                        ) * chunk_weight
                        weighted_stats["bg_abs_mean"] += float(bg_delta.detach().abs().mean().item()) * chunk_weight
                        weighted_stats["co_abs_mean"] += float(co_delta.detach().abs().mean().item()) * chunk_weight
                        weighted_stats["three_abs_mean"] += float(three_delta.detach().abs().mean().item()) * chunk_weight
                        if ctx_bundle_delta is not None:
                            weighted_stats["ctx_bundle_abs_mean"] += float(
                                ctx_bundle_delta.detach().abs().mean().item()
                            ) * chunk_weight
                        weighted_stats["bg_main_abs_mean"] += float(bg_main.detach().abs().mean().item()) * chunk_weight
                        weighted_stats["co_main_abs_mean"] += float(co_main.detach().abs().mean().item()) * chunk_weight
                        weighted_stats["bg_mean"] += float(bg_delta.detach().mean().item()) * chunk_weight
                        weighted_stats["co_mean"] += float(co_delta.detach().mean().item()) * chunk_weight
                        weighted_stats["three_mean"] += float(three_delta.detach().mean().item()) * chunk_weight
                        if ctx_bundle_delta is not None:
                            weighted_stats["ctx_bundle_mean"] += float(
                                ctx_bundle_delta.detach().mean().item()
                            ) * chunk_weight
                        weighted_stats["bg_main_mean"] += float(bg_main.detach().mean().item()) * chunk_weight
                        weighted_stats["co_main_mean"] += float(co_main.detach().mean().item()) * chunk_weight
                        weighted_stats["bg_sq_mean"] += float(bg_delta.detach().pow(2).mean().item()) * chunk_weight
                        weighted_stats["co_sq_mean"] += float(co_delta.detach().pow(2).mean().item()) * chunk_weight
                        weighted_stats["three_sq_mean"] += float(three_delta.detach().pow(2).mean().item()) * chunk_weight
                        if ctx_bundle_delta is not None:
                            weighted_stats["ctx_bundle_sq_mean"] += float(
                                ctx_bundle_delta.detach().pow(2).mean().item()
                            ) * chunk_weight
                        weighted_stats["bg_active_frac"] += float(
                            (bg_delta.detach().abs() > args.tau_bg).float().mean().item()
                        ) * chunk_weight
                        weighted_stats["co_active_frac"] += float(
                            (co_delta.detach().abs() > args.tau_co).float().mean().item()
                        ) * chunk_weight
                        three_target = (
                            three_ref_abs.detach() * args.three_ref_frac
                            if args.three_penalty_mode == "ref_frac_hinge"
                            else torch.full_like(three_ref_abs, args.tau_three)
                        )
                        weighted_stats["three_active_frac"] += float(
                            (three_delta.detach().abs() > three_target).float().mean().item()
                        ) * chunk_weight
                        if ctx_bundle_delta is not None:
                            ctx_bundle_target = (
                                ctx_bundle_ref_abs.detach() * args.ctx_bundle_ref_frac
                                if args.ctx_bundle_penalty_mode == "ref_frac_hinge"
                                else torch.full_like(ctx_bundle_ref_abs, args.tau_ctx_bundle)
                            )
                            weighted_stats["ctx_bundle_active_frac"] += float(
                                (ctx_bundle_delta.detach().abs() > ctx_bundle_target).float().mean().item()
                            ) * chunk_weight
                        weighted_stats["bg_target_abs_mean"] += float(
                            (bg_ref_abs.detach().mean().item() * args.bg_ref_frac)
                        ) * chunk_weight
                        weighted_stats["co_target_abs_mean"] += float(
                            (co_ref_abs.detach().mean().item() * args.co_ref_frac)
                        ) * chunk_weight
                        weighted_stats["three_target_abs_mean"] += float(
                            (three_ref_abs.detach().mean().item() * args.three_ref_frac)
                        ) * chunk_weight
                        weighted_stats["ctx_bundle_target_abs_mean"] += float(
                            (ctx_bundle_ref_abs.detach().mean().item() * args.ctx_bundle_ref_frac)
                        ) * chunk_weight
                        weighted_stats["obj_guard_active_frac"] += float(
                            (obj_effect.detach() < args.obj_guard_floor).float().mean().item()
                        ) * chunk_weight
                        weighted_stats["obj_aligned_guard_active_frac"] += float(
                            (obj_effect_aligned.detach() < args.obj_aligned_floor).float().mean().item()
                        ) * chunk_weight
                        weighted_stats["obj_ctx_guard_active_frac"] += float(
                            (obj_vs_ctx_gap.detach() < args.obj_ctx_margin).float().mean().item()
                        ) * chunk_weight
                        weighted_stats["obj_conflict_guard_active_frac"] += float(
                            (obj_effect_conflict.detach() < args.obj_conflict_floor).float().mean().item()
                        ) * chunk_weight
                        weighted_stats["obj_conflict_rel_guard_active_frac"] += float(
                            (obj_effect_conflict.detach() < obj_conflict_rel_target.detach()).float().mean().item()
                        ) * chunk_weight
                        weighted_stats["obj_conflict_rel_target_mean"] += float(
                            obj_conflict_rel_target.detach().mean().item()
                        ) * chunk_weight
                        weighted_stats["obj_vs_ctx_gap_mean"] += float(obj_vs_ctx_gap.detach().mean().item()) * chunk_weight
                        weighted_stats["context_main_max_mean"] += float(context_main_max.detach().mean().item()) * chunk_weight

                    loss_value += steering_objective_value
                    epoch_stats = {
                        **weighted_stats,
                        "anchor_count": float(total_anchor_count),
                        "unique_anchor_count": float(total_anchor_count),
                        "sampled_from_bg": float(len(bg_indices)),
                        "sampled_from_co": float(len(co_indices)),
                        "sampled_from_three": float(len(three_indices)),
                        "sampled_bg_score": float(
                            np.mean([bg_bank.score_by_idx[idx] for idx in bg_indices])
                        )
                        if bg_indices
                        else 0.0,
                        "sampled_co_score": float(
                            np.mean([co_bank.score_by_idx[idx] for idx in co_indices])
                        )
                        if co_indices
                        else 0.0,
                        "sampled_three_score": float(
                            np.mean([three_bank.score_by_idx[idx] for idx in three_indices])
                        )
                        if three_indices
                        else 0.0,
                        "sampled_overlap_bg_co": float(len(set(bg_indices) & set(co_indices))),
                        "sampled_overlap_bg_three": float(len(set(bg_indices) & set(three_indices))),
                        "sampled_overlap_co_three": float(len(set(co_indices) & set(three_indices))),
                    }
                    do_step = False

            if amp_enabled and do_step:
                scaler.scale(base_loss).backward()
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            elif amp_enabled:
                if args.clip_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            elif do_step:
                base_loss.backward()
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()
            else:
                if args.clip_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
                optimizer.step()

            running["steps"] += 1
            running["total_loss"] += loss_value
            running["ce_loss"] += float(loss_ce.item())
            running["ref_margin_guard_penalty"] += float(ref_margin_guard_penalty.detach().item())
            running["curr_margin"] += float(curr_margin.detach().mean().item())
            running["ref_margin"] += float(ref_margin.detach().mean().item()) if ref_margin is not None else 0.0
            running["margin_gap"] += float(
                ((ref_margin - curr_margin).detach().mean().item()) if ref_margin is not None else 0.0
            )
            running["ref_margin_guard_active_frac"] += float(
                (
                    ((ref_margin - args.ref_margin_slack) > curr_margin).float().mean().item()
                    if ref_margin is not None
                    else 0.0
                )
            )
            for key, value in epoch_stats.items():
                running[key] += float(value)

            steps = max(running["steps"], 1)
            postfix = {
                "loss": f"{running['total_loss'] / steps:.3f}",
                "ce": f"{running['ce_loss'] / steps:.3f}",
                "valsel": args.early_stop_metric,
            }
            if not freeze_backbone and epoch_stats:
                postfix.update(
                    {
                        "bg|d|": f"{running['bg_abs_mean'] / steps:.2f}",
                        "co|d|": f"{running['co_abs_mean'] / steps:.2f}",
                        "3w|d|": f"{running['three_abs_mean'] / steps:.2f}",
                        "ctx|d|": f"{running['ctx_bundle_abs_mean'] / steps:.2f}",
                        "bgm": f"{running['bg_main_abs_mean'] / steps:.2f}",
                        "com": f"{running['co_main_abs_mean'] / steps:.2f}",
                        "anc": f"{int(round(running['anchor_count'] / steps))}",
                    }
                )
            progress.set_postfix(postfix)

        scheduler.step()
        train_steps = max(running["steps"], 1)

        val_metrics = evaluate_group_metrics(
            model,
            val_loader,
            device=device,
            amp=amp_enabled,
            split="val",
            bg_ratio=0.95,
            co_ratio=0.95,
        )
        test_metrics = evaluate_group_metrics(
            model,
            test_loader,
            device=device,
            amp=amp_enabled,
            split="test",
            bg_ratio=0.95,
            co_ratio=0.95,
        )

        probe_metrics: dict[str, object] = {}
        if epoch % max(1, args.probe_every) == 0:
            model.eval()
            probe_metrics = {
                "bg": evaluate_probe_bank(
                    model,
                    bank=bg_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                ),
                "co_occur_obj": evaluate_probe_bank(
                    model,
                    bank=co_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                ),
                "three_way": evaluate_probe_bank(
                    model,
                    bank=three_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                ),
                "bg_main": evaluate_probe_quantity(
                    model,
                    bank=bg_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                    value_key="bg_main",
                    bucket_key="bg_buckets",
                ),
                "co_main": evaluate_probe_quantity(
                    model,
                    bank=co_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                    value_key="co_main",
                    bucket_key="co_buckets",
                ),
                "obj_effect_aligned": evaluate_probe_quantity(
                    model,
                    bank=bg_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                    value_key="obj_effect_aligned",
                    bucket_key="bg_buckets",
                ),
                "obj_effect_conflict": evaluate_probe_quantity(
                    model,
                    bank=bg_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                    value_key="obj_effect_conflict",
                    bucket_key="bg_buckets",
                ),
                "context_main_max": evaluate_probe_quantity(
                    model,
                    bank=bg_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                    value_key="context_main_max",
                    bucket_key="bg_buckets",
                ),
                "obj_vs_ctx_gap": evaluate_probe_quantity(
                    model,
                    bank=bg_bank,
                    samples=all_train_samples,
                    grouped_samples=grouped_train_samples,
                    transform=steering_transform,
                    device=device,
                    amp_enabled=amp_enabled,
                    steer_chunk_size=args.steer_chunk_size,
                    bg_delta_mode=args.bg_delta_mode,
                    co_delta_mode=args.co_delta_mode,
                    exclusive_interaction=args.exclusive_interaction,
                    value_key="obj_vs_ctx_gap",
                    bucket_key="bg_buckets",
                ),
            }
            append_jsonl(
                probe_history_path,
                {
                    "epoch": epoch,
                    "probe_metrics": probe_metrics,
                },
            )

        flat_probe = {}
        if probe_metrics:
            flat_probe.update(flatten_probe_summary(probe_metrics["bg"], "probe_bg"))
            flat_probe.update(flatten_probe_summary(probe_metrics["co_occur_obj"], "probe_co"))
            flat_probe.update(flatten_probe_summary(probe_metrics["three_way"], "probe_three"))
            flat_probe.update(flatten_probe_summary(probe_metrics["bg_main"], "probe_bg_main"))
            flat_probe.update(flatten_probe_summary(probe_metrics["co_main"], "probe_co_main"))
            flat_probe.update(flatten_probe_summary(probe_metrics["obj_effect_aligned"], "probe_obj_aligned"))
            flat_probe.update(flatten_probe_summary(probe_metrics["obj_effect_conflict"], "probe_obj_conflict"))
            flat_probe.update(flatten_probe_summary(probe_metrics["context_main_max"], "probe_ctxmax"))
            flat_probe.update(flatten_probe_summary(probe_metrics["obj_vs_ctx_gap"], "probe_objctx"))

        record = {
            "epoch": epoch,
            "lr": float(scheduler.get_last_lr()[0]),
            "freeze_backbone": int(freeze_backbone),
            "recovery_only": int(args.recovery_only),
            "lam_bg": float(lam_bg),
            "lam_co": float(lam_co),
            "lam_three": float(lam_three),
            "lam_ctx_bundle": float(lam_ctx_bundle),
            "train_loss": running["total_loss"] / train_steps,
            "train_ce_loss": running["ce_loss"] / train_steps,
            "train_bg_penalty": running["bg_penalty"] / train_steps,
            "train_co_penalty": running["co_penalty"] / train_steps,
            "train_three_penalty": running["three_penalty"] / train_steps,
            "train_ctx_bundle_penalty": running["ctx_bundle_penalty"] / train_steps,
            "train_bg_main_penalty": running["bg_main_penalty"] / train_steps,
            "train_co_main_penalty": running["co_main_penalty"] / train_steps,
            "train_obj_effect_mean": running["obj_effect"] / train_steps,
            "train_obj_effect_aligned_mean": running["obj_effect_aligned"] / train_steps,
            "train_obj_effect_conflict_mean": running["obj_effect_conflict"] / train_steps,
            "train_obj_guard_penalty": running["obj_guard_penalty"] / train_steps,
            "train_obj_aligned_guard_penalty": running["obj_aligned_guard_penalty"] / train_steps,
            "train_obj_ctx_guard_penalty": running["obj_ctx_guard_penalty"] / train_steps,
            "train_obj_conflict_guard_penalty": running["obj_conflict_guard_penalty"] / train_steps,
            "train_obj_conflict_rel_guard_penalty": running["obj_conflict_rel_guard_penalty"] / train_steps,
            "train_ref_margin_guard_penalty": running["ref_margin_guard_penalty"] / train_steps,
            "train_curr_margin_mean": running["curr_margin"] / train_steps,
            "train_ref_margin_mean": running["ref_margin"] / train_steps,
            "train_margin_gap_mean": running["margin_gap"] / train_steps,
            "train_bg_delta_mean": running["bg_mean"] / train_steps,
            "train_bg_abs_mean": running["bg_abs_mean"] / train_steps,
            "train_co_delta_mean": running["co_mean"] / train_steps,
            "train_co_abs_mean": running["co_abs_mean"] / train_steps,
            "train_three_delta_mean": running["three_mean"] / train_steps,
            "train_three_abs_mean": running["three_abs_mean"] / train_steps,
            "train_ctx_bundle_delta_mean": running["ctx_bundle_mean"] / train_steps,
            "train_ctx_bundle_abs_mean": running["ctx_bundle_abs_mean"] / train_steps,
            "train_bg_main_mean": running["bg_main_mean"] / train_steps,
            "train_bg_main_abs_mean": running["bg_main_abs_mean"] / train_steps,
            "train_co_main_mean": running["co_main_mean"] / train_steps,
            "train_co_main_abs_mean": running["co_main_abs_mean"] / train_steps,
            "train_bg_sq_mean": running["bg_sq_mean"] / train_steps,
            "train_co_sq_mean": running["co_sq_mean"] / train_steps,
            "train_three_sq_mean": running["three_sq_mean"] / train_steps,
            "train_ctx_bundle_sq_mean": running["ctx_bundle_sq_mean"] / train_steps,
            "train_bg_active_frac": running["bg_active_frac"] / train_steps,
            "train_co_active_frac": running["co_active_frac"] / train_steps,
            "train_three_active_frac": running["three_active_frac"] / train_steps,
            "train_ctx_bundle_active_frac": running["ctx_bundle_active_frac"] / train_steps,
            "train_bg_target_abs_mean": running["bg_target_abs_mean"] / train_steps,
            "train_co_target_abs_mean": running["co_target_abs_mean"] / train_steps,
            "train_three_target_abs_mean": running["three_target_abs_mean"] / train_steps,
            "train_ctx_bundle_target_abs_mean": running["ctx_bundle_target_abs_mean"] / train_steps,
            "train_obj_guard_active_frac": running["obj_guard_active_frac"] / train_steps,
            "train_obj_aligned_guard_active_frac": running["obj_aligned_guard_active_frac"] / train_steps,
            "train_obj_ctx_guard_active_frac": running["obj_ctx_guard_active_frac"] / train_steps,
            "train_obj_conflict_guard_active_frac": running["obj_conflict_guard_active_frac"] / train_steps,
            "train_obj_conflict_rel_guard_active_frac": running["obj_conflict_rel_guard_active_frac"] / train_steps,
            "train_obj_conflict_rel_target_mean": running["obj_conflict_rel_target_mean"] / train_steps,
            "train_ref_margin_guard_active_frac": running["ref_margin_guard_active_frac"] / train_steps,
            "train_obj_vs_ctx_gap_mean": running["obj_vs_ctx_gap_mean"] / train_steps,
            "train_context_main_max_mean": running["context_main_max_mean"] / train_steps,
            "train_anchor_count": running["anchor_count"] / train_steps,
            "train_unique_anchor_count": running["unique_anchor_count"] / train_steps,
            "train_sampled_from_bg": running["sampled_from_bg"] / train_steps,
            "train_sampled_from_co": running["sampled_from_co"] / train_steps,
            "train_sampled_from_three": running["sampled_from_three"] / train_steps,
            "train_sampled_bg_score": running["sampled_bg_score"] / train_steps,
            "train_sampled_co_score": running["sampled_co_score"] / train_steps,
            "train_sampled_three_score": running["sampled_three_score"] / train_steps,
            "train_sampled_overlap_bg_co": running["sampled_overlap_bg_co"] / train_steps,
            "train_sampled_overlap_bg_three": running["sampled_overlap_bg_three"] / train_steps,
            "train_sampled_overlap_co_three": running["sampled_overlap_co_three"] / train_steps,
            **flat_probe,
            **val_metrics,
            **test_metrics,
        }
        record["val_paper_slice_mean"] = float(
            (
                record["val_bg_absent_co_occur_obj_present_acc"]
                + record["val_bg_present_co_occur_obj_absent_acc"]
                + record["val_bg_absent_co_occur_obj_absent_acc"]
            )
            / 3.0
        )
        record["val_paper_gap_mean"] = float(
            (
                record["val_bg_gap"]
                + record["val_co_occur_obj_gap"]
                + record["val_both_gap"]
            )
            / 3.0
        )
        record["test_paper_slice_mean"] = float(
            (
                (record["test_id_acc"] + record["test_bg_gap"])
                + (record["test_id_acc"] + record["test_co_occur_obj_gap"])
                + (record["test_id_acc"] + record["test_both_gap"])
            )
            / 3.0
        )
        record["test_paper_gap_mean"] = float(
            (
                record["test_bg_gap"]
                + record["test_co_occur_obj_gap"]
                + record["test_both_gap"]
            )
            / 3.0
        )
        for selector in SELECTOR_SPECS:
            record[f"selector_{selector}"] = compute_selector_value(val_metrics, selector)
        record["selector_name"] = args.checkpoint_selector
        record["selector_value"] = float(record[f"selector_{args.checkpoint_selector}"])
        append_jsonl(history_path, record)
        epoch_records.append(record)

        metric_value = float(record["selector_value"])
        last_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "best_metric_name": metric_name,
            "best_metric_value": best_metric,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "best_test_metrics": best_test_metrics,
            "best_probe_metrics": best_probe_metrics,
            "selection_diagnostics": selection_diagnostics,
            "probe_metrics": probe_metrics,
        }
        save_checkpoint(output_dir / "last.pth", last_state)
        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", last_state)
        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_test_metrics = dict(test_metrics)
            best_probe_metrics = probe_metrics
            best_state = dict(last_state)
            best_state["best_metric_value"] = best_metric
            best_state["best_epoch"] = best_epoch
            best_state["best_val_metrics"] = best_val_metrics
            best_state["best_test_metrics"] = best_test_metrics
            best_state["best_probe_metrics"] = best_probe_metrics
            save_checkpoint(output_dir / "best.pth", best_state)

        probe_msg = ""
        if flat_probe:
            probe_msg = (
                f" probe_bg|d|={flat_probe.get('probe_bg_abs_mean', float('nan')):.2f}"
                f" probe_co|d|={flat_probe.get('probe_co_abs_mean', float('nan')):.2f}"
                f" probe_3w|d|={flat_probe.get('probe_three_abs_mean', float('nan')):.2f}"
                f" probe_bgm={flat_probe.get('probe_bg_main_abs_mean', float('nan')):.2f}"
                f" probe_com={flat_probe.get('probe_co_main_abs_mean', float('nan')):.2f}"
                f" probe_objA={flat_probe.get('probe_obj_aligned_abs_mean', float('nan')):.2f}"
                f" probe_objC={flat_probe.get('probe_obj_conflict_abs_mean', float('nan')):.2f}"
                f" probe_ctxmax={flat_probe.get('probe_ctxmax_abs_mean', float('nan')):.2f}"
                f" probe_gap={flat_probe.get('probe_objctx_mean', float('nan')):.2f}"
            )
        print(
            f"[epoch {epoch:03d}] "
            f"loss={record['train_loss']:.4f} "
            f"bg|d|={record['train_bg_abs_mean']:.3f} "
            f"co|d|={record['train_co_abs_mean']:.3f} "
            f"3w|d|={record['train_three_abs_mean']:.3f} "
            f"ctx|d|={record['train_ctx_bundle_abs_mean']:.3f} "
            f"bgm={record['train_bg_main_abs_mean']:.3f} "
            f"com={record['train_co_main_abs_mean']:.3f} "
            f"objA={record['train_obj_effect_aligned_mean']:.3f} "
            f"objC={record['train_obj_effect_conflict_mean']:.3f} "
            f"gap={record['train_obj_vs_ctx_gap_mean']:.3f} "
            f"{args.checkpoint_selector}={metric_value * 100:.2f} "
            f"val_both_wga={val_metrics['val_both_worst_group_acc'] * 100:.2f} "
            f"test_both_wga={test_metrics['test_both_worst_group_acc'] * 100:.2f} "
            f"test_id_acc={test_metrics['test_id_acc'] * 100:.2f}"
            f"{probe_msg}"
        )

    if epoch_records:
        selection_diagnostics = build_selection_diagnostics(epoch_records)

    summary = {
        "config": config,
        "best_metric_name": metric_name,
        "best_metric_value": best_metric,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "best_probe_metrics": best_probe_metrics,
        "selection_diagnostics": selection_diagnostics,
        "artifacts": {
            "best_checkpoint": str(output_dir / "best.pth"),
            "last_checkpoint": str(output_dir / "last.pth"),
            "history": str(history_path),
            "probe_history": str(probe_history_path),
            "anchor_bank_summary": str(output_dir / "anchor_bank_summary.json"),
            "probe_indices": str(output_dir / "probe_indices.json"),
        },
        "finished_at_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": time.time() - start_time,
    }
    serialize_json(summary, output_dir / "train_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
