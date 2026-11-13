#!/usr/bin/env python3
"""
Waterbirds diagnostic suite for object-vs-background behavior.

This evaluator conditions key probes on the true conflict groups (`bg != y`)
and separates three questions:

1. Necessity: does removing the bird reduce support for the original class?
2. Abstention / hallucination: once the bird is gone, does the model become
   uncertain, or does it confidently follow background?
3. Override: on a conflict-only donor background, does adding the bird make the
   model return to the bird class?
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "train"))
HELPER_PATH = Path(__file__).resolve().parents[1] / "train" / "train_cfaug.py"
_SPEC = importlib.util.spec_from_file_location("wb_cfaug_helpers", HELPER_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Could not load helper module from {HELPER_PATH}")
_HELPERS = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_HELPERS)

HFWaterbirds = _HELPERS.HFWaterbirds
inpaint_with_mask = _HELPERS.inpaint_with_mask
build_donor_pools = _HELPERS.build_donor_pools
cleanse_donor = _HELPERS.cleanse_donor
hist_match = _HELPERS.hist_match
build_backbone = _HELPERS.build_backbone
set_seed = _HELPERS.set_seed
get_device = _HELPERS.get_device
mask_bbox = _HELPERS.mask_bbox
threshold_mask = _HELPERS.threshold_mask
mask_crop = _HELPERS.mask_crop
paste_with_segment_mask = _HELPERS.paste_with_segment_mask


class _BaseWBSubset(Dataset):
    def __init__(self, base_ds: HFWaterbirds, indices=None):
        self.base = base_ds
        self.indices = list(indices) if indices is not None else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.base)

    def resolve_i(self, i: int) -> int:
        return self.indices[i] if self.indices is not None else i


class NaturalDataset(_BaseWBSubset):
    def __getitem__(self, i):
        i = self.resolve_i(i)
        x, y, _, _, bg, idx = self.base[i]
        return x, int(y), int(bg), int(idx)


class BackgroundOnlyNaturalDataset(_BaseWBSubset):
    def __getitem__(self, i):
        i = self.resolve_i(i)
        _, y, _, pil_img, bg, idx = self.base[i]
        mask = self.base.load_mask(int(idx))
        if mask is None:
            out = pil_img
        else:
            out = inpaint_with_mask(pil_img, mask, radius=4, expand_radius=3, threshold=127)
        return self.base.tf(out), int(y), int(bg), int(idx)


class _DonorBase(_BaseWBSubset):
    def __init__(self, base_ds: HFWaterbirds, seed: int, indices=None):
        super().__init__(base_ds, indices=indices)
        self.seed = int(seed)
        self.donor_pools = build_donor_pools(base_ds)

    def choose_donor_for_bg(self, desired_bg: int, anchor_idx: int):
        pool = [j for j in self.donor_pools.get(int(desired_bg), []) if j != anchor_idx]
        if not pool:
            return None
        rng = random.Random(self.seed * 1000003 + anchor_idx * 1009 + int(desired_bg) * 97)
        start = rng.randrange(len(pool))
        for off in range(len(pool)):
            cand = int(pool[(start + off) % len(pool)])
            if self.base.load_mask(cand) is not None:
                return cand
        return None


class ConflictBackgroundOnlyDataset(_DonorBase):
    """
    Conflict-only donor backgrounds with no bird present.

    By construction, donor_bg == 1 - y.
    """

    def __getitem__(self, i):
        i = self.resolve_i(i)
        _, y, _, pil_img, _bg, idx = self.base[i]
        desired_bg = 1 - int(y)
        donor_idx = self.choose_donor_for_bg(desired_bg, int(idx))
        if donor_idx is None:
            out = pil_img
            donor_bg = desired_bg
        else:
            _, _, _, donor_pil, donor_bg, _ = self.base[donor_idx]
            donor_mask = self.base.load_mask(int(donor_idx))
            if donor_mask is None:
                out = pil_img
                donor_bg = desired_bg
            else:
                donor_clean = cleanse_donor(donor_pil, donor_mask)
                out = hist_match(donor_clean, pil_img)
        return self.base.tf(out), int(y), int(donor_bg), int(idx)


class ConflictOverridePairDataset(_DonorBase):
    """
    Conflict-only donor backgrounds, scored both without and with the anchor bird.
    """

    def __getitem__(self, i):
        i = self.resolve_i(i)
        _, y, _, pil_img, _bg, idx = self.base[i]
        desired_bg = 1 - int(y)
        donor_idx = self.choose_donor_for_bg(desired_bg, int(idx))
        mask = self.base.load_mask(int(idx))
        box = mask_bbox(mask) if mask is not None else None
        if box is None:
            box = self.base.box_for_index(int(idx))

        if donor_idx is None or mask is None or box is None:
            bg_only = pil_img
            with_bird = pil_img
            donor_bg = desired_bg
        else:
            _, _, _, donor_pil, donor_bg, _ = self.base[donor_idx]
            donor_mask = self.base.load_mask(int(donor_idx))
            if donor_mask is None:
                bg_only = pil_img
                with_bird = pil_img
                donor_bg = desired_bg
            else:
                donor_clean = cleanse_donor(donor_pil, donor_mask)
                bg_only = hist_match(donor_clean, pil_img)
                obj_crop = pil_img.crop(tuple(box))
                obj_mask = threshold_mask(mask_crop(mask, box), threshold=127)
                with_bird = paste_with_segment_mask(bg_only, obj_crop, tuple(box), obj_mask, blur_radius=0)

        return (
            self.base.tf(bg_only),
            self.base.tf(with_bird),
            int(y),
            int(donor_bg),
            int(idx),
        )


def make_loader(ds: Dataset, batch_size: int, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def load_model(checkpoint: Path, backbone: str, image_size: int, device):
    model = build_backbone(backbone, num_classes=2, image_size=image_size).to(device)
    obj = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "model" in obj:
        model.load_state_dict(obj["model"])
    else:
        model.load_state_dict(obj)
    model.eval()
    return model


@torch.inference_mode()
def score_dataset(model, dl, device):
    rows = []
    for x, y, bg_signal, idx in dl:
        x = x.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        bg_dev = bg_signal.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits = model(x).float()
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(1)
        ent = -(probs * probs.clamp_min(1e-8).log()).sum(dim=1)
        for i in range(x.size(0)):
            rows.append(
                {
                    "idx": int(idx[i]),
                    "y": int(y[i]),
                    "bg_signal": int(bg_signal[i]),
                    "pred": int(pred[i].cpu()),
                    "correct": int(pred[i].cpu() == y[i]),
                    "target_prob": float(probs[i, y_dev[i]].cpu()),
                    "bg_prob": float(probs[i, bg_dev[i]].cpu()),
                    "max_prob": float(probs[i].max().cpu()),
                    "entropy": float(ent[i].cpu()),
                }
            )
    rows.sort(key=lambda r: r["idx"])
    return rows


@torch.inference_mode()
def score_override_pairs(model, dl, device):
    rows = []
    for x_bg, x_with, y, donor_bg, idx in dl:
        x_bg = x_bg.to(device, non_blocking=True)
        x_with = x_with.to(device, non_blocking=True)
        y_dev = y.to(device, non_blocking=True)
        donor_bg_dev = donor_bg.to(device, non_blocking=True)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            logits_bg = model(x_bg).float()
            logits_with = model(x_with).float()
        probs_bg = torch.softmax(logits_bg, dim=1)
        probs_with = torch.softmax(logits_with, dim=1)
        pred_bg = logits_bg.argmax(1)
        pred_with = logits_with.argmax(1)
        for i in range(x_bg.size(0)):
            rows.append(
                {
                    "idx": int(idx[i]),
                    "y": int(y[i]),
                    "bg_signal": int(donor_bg[i]),
                    "bg_only_pred": int(pred_bg[i].cpu()),
                    "with_bird_pred": int(pred_with[i].cpu()),
                    "bg_only_target_prob": float(probs_bg[i, y_dev[i]].cpu()),
                    "with_bird_target_prob": float(probs_with[i, y_dev[i]].cpu()),
                    "bg_only_bg_prob": float(probs_bg[i, donor_bg_dev[i]].cpu()),
                    "with_bird_bg_prob": float(probs_with[i, donor_bg_dev[i]].cpu()),
                    "bg_only_max_prob": float(probs_bg[i].max().cpu()),
                    "with_bird_max_prob": float(probs_with[i].max().cpu()),
                }
            )
    rows.sort(key=lambda r: r["idx"])
    return rows


def summarize_rows(rows):
    return {
        "acc": float(np.mean([r["correct"] for r in rows])) if rows else 0.0,
        "target_prob_mean": float(np.mean([r["target_prob"] for r in rows])) if rows else 0.0,
        "bg_prob_mean": float(np.mean([r["bg_prob"] for r in rows])) if rows else 0.0,
        "max_prob_mean": float(np.mean([r["max_prob"] for r in rows])) if rows else 0.0,
        "entropy_mean": float(np.mean([r["entropy"] for r in rows])) if rows else 0.0,
        "pred_matches_bg_rate": float(np.mean([r["pred"] == r["bg_signal"] for r in rows])) if rows else 0.0,
        "n": len(rows),
    }


def filter_conflict(rows):
    return [r for r in rows if r["y"] != r["bg_signal"]]


def paired_removal_summary(natural_rows, removed_rows):
    removed_by_idx = {r["idx"]: r for r in removed_rows}
    pairs = []
    for nr in natural_rows:
        rr = removed_by_idx[nr["idx"]]
        pairs.append((nr, rr))

    def _mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    all_target_drops = [nr["target_prob"] - rr["target_prob"] for nr, rr in pairs]
    conflict_pairs = [(nr, rr) for nr, rr in pairs if nr["y"] != nr["bg_signal"]]
    conflict_target_drops = [nr["target_prob"] - rr["target_prob"] for nr, rr in conflict_pairs]
    correct_to_bg_flip = [
        int((nr["pred"] == nr["y"]) and (rr["pred"] == rr["bg_signal"]))
        for nr, rr in conflict_pairs
    ]

    removed_conflict_rows = [rr for _, rr in conflict_pairs]
    removed_conflict_summary = summarize_rows(removed_conflict_rows)

    return {
        "all_target_drop_mean": _mean(all_target_drops),
        "conflict_target_drop_mean": _mean(conflict_target_drops),
        "conflict_correct_to_bg_flip_rate": _mean(correct_to_bg_flip),
        "conflict_removed_bg_match_rate": removed_conflict_summary["pred_matches_bg_rate"],
        "conflict_removed_bg_prob_mean": removed_conflict_summary["bg_prob_mean"],
        "conflict_removed_max_prob_mean": removed_conflict_summary["max_prob_mean"],
        "conflict_removed_entropy_mean": removed_conflict_summary["entropy_mean"],
        "conflict_n": removed_conflict_summary["n"],
    }


def override_summary(rows):
    def _mean(vals):
        return float(np.mean(vals)) if vals else 0.0

    return {
        "bg_only_bg_match_rate": _mean([r["bg_only_pred"] == r["bg_signal"] for r in rows]),
        "bg_only_bg_prob_mean": _mean([r["bg_only_bg_prob"] for r in rows]),
        "bg_only_max_prob_mean": _mean([r["bg_only_max_prob"] for r in rows]),
        "with_bird_target_rate": _mean([r["with_bird_pred"] == r["y"] for r in rows]),
        "with_bird_bg_match_rate": _mean([r["with_bird_pred"] == r["bg_signal"] for r in rows]),
        "strict_override_success_rate": _mean(
            [(r["bg_only_pred"] == r["bg_signal"]) and (r["with_bird_pred"] == r["y"]) for r in rows]
        ),
        "target_prob_lift_mean": _mean([r["with_bird_target_prob"] - r["bg_only_target_prob"] for r in rows]),
        "bg_prob_drop_mean": _mean([r["bg_only_bg_prob"] - r["with_bird_bg_prob"] for r in rows]),
    }


def family_name(model_name: str) -> str:
    if "_seed" in model_name:
        return model_name.rsplit("_seed", 1)[0]
    return model_name


def mean_summary(rows):
    if not rows:
        return {}
    out = {"family": rows[0]["family"], "n_models": len(rows)}
    numeric_keys = [k for k, v in rows[0].items() if k not in {"model", "family"} and isinstance(v, (int, float))]
    for key in numeric_keys:
        vals = [float(r[key]) for r in rows]
        out[key] = float(np.mean(vals))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="Repeated model=checkpoint.pt entries")
    ap.add_argument("--backbone", type=str, default="resnet50")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_samples", type=int, default=0)
    ap.add_argument("--data_root", type=str, default=str(_HELPERS._DEFAULT_REAL_ROOT))
    ap.add_argument("--output_dir", type=str, required=True)
    args = ap.parse_args()

    set_seed(args.seed)
    device = get_device()
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_specs = {}
    for item in args.models:
        if "=" not in item:
            raise SystemExit(f"Bad --models entry: {item}")
        name, path = item.split("=", 1)
        model_specs[name] = Path(path).resolve()

    base_ds = HFWaterbirds("test", image_size=args.image_size, train_aug=False, data_root=args.data_root, backbone=args.backbone)
    subset_indices = list(range(min(args.max_samples, len(base_ds)))) if args.max_samples > 0 else None

    natural_loader = make_loader(NaturalDataset(base_ds, subset_indices), args.batch_size, args.num_workers)
    removed_loader = make_loader(BackgroundOnlyNaturalDataset(base_ds, subset_indices), args.batch_size, args.num_workers)
    conflict_bg_only_loader = make_loader(
        ConflictBackgroundOnlyDataset(base_ds, seed=args.seed, indices=subset_indices),
        args.batch_size,
        args.num_workers,
    )
    override_loader = make_loader(
        ConflictOverridePairDataset(base_ds, seed=args.seed, indices=subset_indices),
        max(1, args.batch_size // 2),
        args.num_workers,
    )

    per_model_rows = []
    detailed = {}
    for model_name, ckpt in model_specs.items():
        model = load_model(ckpt, args.backbone, args.image_size, device)

        natural_rows = score_dataset(model, natural_loader, device)
        removed_rows = score_dataset(model, removed_loader, device)
        conflict_bg_only_rows = score_dataset(model, conflict_bg_only_loader, device)
        override_rows = score_override_pairs(model, override_loader, device)

        natural_conflict = summarize_rows(filter_conflict(natural_rows))
        removal = paired_removal_summary(natural_rows, removed_rows)
        conflict_bg_only = summarize_rows(conflict_bg_only_rows)
        override = override_summary(override_rows)

        row = {
            "model": model_name,
            "family": family_name(model_name),
            "natural_conflict_acc": natural_conflict["acc"],
            "natural_conflict_target_prob_mean": natural_conflict["target_prob_mean"],
            "removal_all_target_drop_mean": removal["all_target_drop_mean"],
            "removal_conflict_target_drop_mean": removal["conflict_target_drop_mean"],
            "removal_conflict_correct_to_bg_flip_rate": removal["conflict_correct_to_bg_flip_rate"],
            "removal_conflict_bg_match_rate": removal["conflict_removed_bg_match_rate"],
            "removal_conflict_bg_prob_mean": removal["conflict_removed_bg_prob_mean"],
            "removal_conflict_max_prob_mean": removal["conflict_removed_max_prob_mean"],
            "removal_conflict_entropy_mean": removal["conflict_removed_entropy_mean"],
            "conflict_bg_only_bg_match_rate": conflict_bg_only["pred_matches_bg_rate"],
            "conflict_bg_only_bg_prob_mean": conflict_bg_only["bg_prob_mean"],
            "conflict_bg_only_max_prob_mean": conflict_bg_only["max_prob_mean"],
            "conflict_bg_only_entropy_mean": conflict_bg_only["entropy_mean"],
            "override_bg_only_bg_match_rate": override["bg_only_bg_match_rate"],
            "override_bg_only_bg_prob_mean": override["bg_only_bg_prob_mean"],
            "override_bg_only_max_prob_mean": override["bg_only_max_prob_mean"],
            "override_with_bird_target_rate": override["with_bird_target_rate"],
            "override_with_bird_bg_match_rate": override["with_bird_bg_match_rate"],
            "override_strict_success_rate": override["strict_override_success_rate"],
            "override_target_prob_lift_mean": override["target_prob_lift_mean"],
            "override_bg_prob_drop_mean": override["bg_prob_drop_mean"],
            "n_conflict_natural": natural_conflict["n"],
            "n_conflict_removed": removal["conflict_n"],
            "n_conflict_bg_only": conflict_bg_only["n"],
        }
        per_model_rows.append(row)
        detailed[model_name] = {
            "natural_conflict": natural_conflict,
            "removal": removal,
            "conflict_bg_only": conflict_bg_only,
            "override": override,
        }

    families = defaultdict(list)
    for row in per_model_rows:
        families[row["family"]].append(row)
    family_rows = [mean_summary(rows) for _, rows in sorted(families.items())]

    payload = {
        "seed": args.seed,
        "backbone": args.backbone,
        "models": {k: str(v) for k, v in model_specs.items()},
        "per_model": per_model_rows,
        "per_family_mean": family_rows,
        "detailed": detailed,
    }
    (output_dir / "clean_diagnostics_eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if per_model_rows:
        header = list(per_model_rows[0].keys())
        with (output_dir / "per_model_summary.csv").open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in per_model_rows:
                f.write(",".join(str(row[h]) for h in header) + "\n")

    if family_rows:
        header = list(family_rows[0].keys())
        with (output_dir / "per_family_mean_summary.csv").open("w", encoding="utf-8") as f:
            f.write(",".join(header) + "\n")
            for row in family_rows:
                f.write(",".join(str(row[h]) for h in header) + "\n")

    print(json.dumps({"per_model": per_model_rows, "per_family_mean": family_rows}, indent=2))


if __name__ == "__main__":
    main()
