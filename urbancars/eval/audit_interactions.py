from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common import (
    BG_NAMES,
    CO_OCCUR_OBJ_NAMES,
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    OBJ_NAMES,
    UrbanCarsSample,
    build_classifier,
    build_bg_conditional_variants,
    build_co_conditional_variants,
    build_interaction_variants,
    build_joint_conditional_variants,
    build_three_way_conditional_variants,
    enumerate_urbancars_samples,
    first_sample_per_composition,
    get_transforms,
    group_samples_by_factors,
    make_panel,
    serialize_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit object-background and object-co-object interactions on UrbanCars."
    )
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument(
        "--interactions",
        nargs="+",
        default=["bg", "co_occur_obj"],
        choices=["bg", "co_occur_obj", "three_way"],
    )
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--corner_examples", type=int, default=8)
    parser.add_argument(
        "--neutral_fill",
        type=str,
        default="imagenet_mean",
        choices=["imagenet_mean", "black"],
    )
    parser.add_argument(
        "--bg_delta_mode",
        type=str,
        choices=["co_absent", "co_present"],
        default="co_absent",
        help=(
            "How to define the object x background audit: in the co-absent slice "
            "(default) or with the co-object held present across all bg corners."
        ),
    )
    parser.add_argument(
        "--co_delta_mode",
        type=str,
        choices=["bg_fixed", "bg_absent"],
        default="bg_fixed",
        help=(
            "How to define the object x co-object audit: at fixed background "
            "(default) or in a background-absent 4-corner panel."
        ),
    )
    return parser.parse_args()


def choose_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_ROOT / f"interaction_audit_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_model(
    checkpoint_path: Path,
    *,
    backbone: str,
    weights: str,
    device: torch.device,
) -> torch.nn.Module:
    state = torch.load(checkpoint_path, map_location="cpu")
    model = build_classifier(backbone, num_classes=2, weights_name=weights)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "classifier" in state:
        model.load_state_dict(state["classifier"])
    else:
        raise KeyError(
            "Unsupported checkpoint format. Expected model_state_dict or classifier in checkpoint."
        )
    return model.to(device).eval()


def target_margin(logits: torch.Tensor, obj_ids: list[int]) -> torch.Tensor:
    obj = torch.tensor(obj_ids, device=logits.device, dtype=torch.long)
    other = 1 - obj
    return logits[torch.arange(logits.size(0), device=logits.device), obj] - logits[
        torch.arange(logits.size(0), device=logits.device), other
    ]


def summarize_values(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64)
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


def save_corner_examples(
    samples: list[UrbanCarsSample],
    *,
    interactions: list[str],
    grouped_samples,
    output_dir: Path,
    bg_delta_mode: str,
    co_delta_mode: str,
) -> dict[str, int]:
    corner_root = output_dir / "corner_examples"
    bg_dir = corner_root / "bg"
    obj_dir = corner_root / "obj"
    co_dir = corner_root / "co_occur_obj"
    co_only_dir = corner_root / "co_occur_obj_only"
    bg_dir.mkdir(parents=True, exist_ok=True)
    obj_dir.mkdir(parents=True, exist_ok=True)
    co_dir.mkdir(parents=True, exist_ok=True)
    co_only_dir.mkdir(parents=True, exist_ok=True)

    counts = {"bg": 0, "obj": 0, "co_occur_obj": 0, "co_occur_obj_only": 0}
    for sample in samples:
        meta = (
            f"{sample.composition_name} | stem={sample.stem} | "
            f"obj={OBJ_NAMES[sample.obj_id]} bg={BG_NAMES[sample.bg_id]} "
            f"co={CO_OCCUR_OBJ_NAMES[sample.co_occur_obj_id]}"
        )

        if "bg" in interactions or "three_way" in interactions:
            if bg_delta_mode == "co_absent":
                bg_variants = build_bg_conditional_variants(sample, grouped_samples)
                bg_panel = make_panel(
                    [
                        bg_variants["bg_anchor"],
                        bg_variants["bg_donor"],
                        bg_variants["obj_bg_anchor"],
                        bg_variants["obj_bg_donor"],
                    ],
                    [
                        f"bg only | {BG_NAMES[sample.bg_id]}",
                        f"bg only | {BG_NAMES[1 - sample.bg_id]}",
                        f"obj + bg | {BG_NAMES[sample.bg_id]}",
                        f"obj + bg | {BG_NAMES[1 - sample.bg_id]}",
                    ],
                    title=f"Object x Background corners | {meta}",
                )
            else:
                joint_variants = build_joint_conditional_variants(sample, grouped_samples)
                bg_panel = make_panel(
                    [
                        joint_variants["co_bg_anchor"],
                        joint_variants["co_bg_donor"],
                        joint_variants["full_anchor"],
                        joint_variants["full_donor"],
                    ],
                    [
                        f"bg + co | {BG_NAMES[sample.bg_id]}",
                        f"bg + co | {BG_NAMES[1 - sample.bg_id]}",
                        f"bg + co + obj | {BG_NAMES[sample.bg_id]}",
                        f"bg + co + obj | {BG_NAMES[1 - sample.bg_id]}",
                    ],
                    title=f"Object x Background corners (co fixed) | {meta}",
                )
            bg_panel.save(bg_dir / f"{sample.composition_name}_{sample.stem}.png")
            counts["bg"] += 1

        if co_delta_mode == "bg_fixed":
            co_variants = build_co_conditional_variants(sample)
            obj_panel = make_panel(
                [co_variants["bg_only"], co_variants["obj_bg"]],
                ["bg only", "bg + obj"],
                title=f"Object-only intervention | {meta}",
                columns=2,
            )
            co_only_panel = make_panel(
                [co_variants["bg_only"], co_variants["bg_co"]],
                ["bg only", "bg + co"],
                title=f"Co-object-only intervention | {meta}",
                columns=2,
            )
            co_panel_images = [
                co_variants["bg_only"],
                co_variants["bg_co"],
                co_variants["obj_bg"],
                co_variants["full"],
            ]
            co_panel_labels = ["bg only", "bg + co", "bg + obj", "bg + obj + co"]
        else:
            co_variants = build_interaction_variants(sample)
            obj_panel = make_panel(
                [co_variants["empty"], co_variants["obj_only"]],
                ["empty", "obj only"],
                title=f"Object-only intervention (no bg) | {meta}",
                columns=2,
            )
            co_only_panel = make_panel(
                [co_variants["empty"], co_variants["co_only"]],
                ["empty", "co only"],
                title=f"Co-object-only intervention (no bg) | {meta}",
                columns=2,
            )
            co_panel_images = [
                co_variants["empty"],
                co_variants["co_only"],
                co_variants["obj_only"],
                co_variants["obj_co"],
            ]
            co_panel_labels = ["empty", "co only", "obj only", "obj + co"]

        obj_panel.save(obj_dir / f"{sample.composition_name}_{sample.stem}.png")
        counts["obj"] += 1

        co_only_panel.save(co_only_dir / f"{sample.composition_name}_{sample.stem}.png")
        counts["co_occur_obj_only"] += 1

        co_panel = make_panel(
            co_panel_images,
            co_panel_labels,
            title=(
                f"Object x Co-object corners{' (no bg)' if co_delta_mode == 'bg_absent' else ''} | {meta}"
            ),
        )
        co_panel.save(co_dir / f"{sample.composition_name}_{sample.stem}.png")
        counts["co_occur_obj"] += 1

    return counts


def main() -> None:
    args = parse_args()
    output_dir = choose_output_dir(args.output_dir)
    device = torch.device("cpu" if args.device == "cuda" and not torch.cuda.is_available() else args.device)
    amp_enabled = args.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    all_samples = enumerate_urbancars_samples(args.data_root, args.split, include_auxiliary_assets=True)
    grouped_samples = group_samples_by_factors(all_samples)
    samples = all_samples
    if args.max_samples > 0:
        samples = samples[: args.max_samples]

    requires_background_assets = False
    if "bg" in args.interactions or "three_way" in args.interactions:
        requires_background_assets = True
    elif "co_occur_obj" in args.interactions and args.co_delta_mode == "bg_fixed":
        requires_background_assets = True
    if requires_background_assets:
        missing_background_asset = [
            sample for sample in samples
            if sample.bg_source_path is None and sample.bg_only_path is None
        ]
        if missing_background_asset:
            raise FileNotFoundError(
                "A background asset is required for these conditional audits. "
                f"Missing for split={args.split}; generate the dataset with data/generate_urbancars.sh first."
            )

    corner_samples = first_sample_per_composition(samples, limit=min(args.corner_examples, len(samples)))
    corner_counts = save_corner_examples(
        corner_samples,
        interactions=args.interactions,
        grouped_samples=grouped_samples,
        output_dir=output_dir,
        bg_delta_mode=args.bg_delta_mode,
        co_delta_mode=args.co_delta_mode,
    )

    config = {
        **{
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "data_root": str(args.data_root),
        "output_dir": str(output_dir),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "device_resolved": str(device),
        "amp_enabled": amp_enabled,
        "num_samples_selected": len(samples),
        "num_corner_examples_saved": corner_counts,
    }
    serialize_json(config, output_dir / "config.json")

    if args.checkpoint is None:
        summary = {
            "config": config,
            "message": "Corner examples generated without model scoring.",
            "corner_examples_root": str(output_dir / "corner_examples"),
            "finished_at_utc": datetime.utcnow().isoformat() + "Z",
        }
        serialize_json(summary, output_dir / "audit_summary.json")
        print(json.dumps(summary, indent=2))
        return

    model = load_model(
        args.checkpoint,
        backbone=args.backbone,
        weights=args.weights,
        device=device,
    )
    transform = get_transforms(is_training=False)

    rows: list[dict[str, object]] = []
    interaction_values: dict[str, list[float]] = defaultdict(list)
    interaction_values_by_bucket: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    with torch.no_grad():
        for sample in samples:
            if args.co_delta_mode == "bg_fixed":
                co_variants = build_co_conditional_variants(sample)
            else:
                co_variants = build_interaction_variants(sample)
            bg_variants = (
                build_bg_conditional_variants(sample, grouped_samples)
                if "bg" in args.interactions and args.bg_delta_mode == "co_absent"
                else {}
            )
            joint_variants = (
                build_joint_conditional_variants(sample, grouped_samples)
                if ("bg" in args.interactions and args.bg_delta_mode == "co_present")
                else {}
            )
            three_way_variants = (
                build_three_way_conditional_variants(sample, grouped_samples)
                if "three_way" in args.interactions
                else {}
            )

            ordered_images: list[tuple[str, Image.Image]] = []
            seen_keys: set[str] = set()

            def add_image(key: str, image: Image.Image) -> None:
                if key in seen_keys:
                    return
                ordered_images.append((key, image))
                seen_keys.add(key)

            if "co_occur_obj" in args.interactions:
                if args.co_delta_mode == "bg_fixed":
                    add_image("bg_only", co_variants["bg_only"])
                    add_image("bg_co", co_variants["bg_co"])
                    add_image("obj_bg", co_variants["obj_bg"])
                    add_image("full", co_variants["full"])
                else:
                    add_image("empty", co_variants["empty"])
                    add_image("co_only", co_variants["co_only"])
                    add_image("obj_only", co_variants["obj_only"])
                    add_image("obj_co", co_variants["obj_co"])
            if "bg" in args.interactions:
                if args.bg_delta_mode == "co_absent":
                    add_image("bg_anchor", bg_variants["bg_anchor"])
                    add_image("bg_donor", bg_variants["bg_donor"])
                    add_image("obj_bg_anchor", bg_variants["obj_bg_anchor"])
                    add_image("obj_bg_donor", bg_variants["obj_bg_donor"])
                else:
                    add_image("co_bg_anchor", joint_variants["co_bg_anchor"])
                    add_image("co_bg_donor", joint_variants["co_bg_donor"])
                    add_image("full_anchor", joint_variants["full_anchor"])
                    add_image("full_donor", joint_variants["full_donor"])
            if "three_way" in args.interactions:
                add_image("bg_anchor", three_way_variants["bg_anchor"])
                add_image("bg_donor", three_way_variants["bg_donor"])
                add_image("obj_bg_anchor", three_way_variants["obj_bg_anchor"])
                add_image("obj_bg_donor", three_way_variants["obj_bg_donor"])
                add_image("co_bg_anchor", three_way_variants["co_bg_anchor"])
                add_image("co_bg_donor", three_way_variants["co_bg_donor"])
                add_image("full_anchor", three_way_variants["full_anchor"])
                add_image("full_donor", three_way_variants["full_donor"])

            ordered_keys = [key for key, _ in ordered_images]
            tensors = [transform(image) for _, image in ordered_images]
            batch = torch.stack(tensors, dim=0).to(device)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(batch)
            margins = target_margin(logits, [sample.obj_id] * batch.size(0)).cpu().tolist()
            margin_by_key = {key: margins[idx] for idx, key in enumerate(ordered_keys)}

            row = {
                "image_path": str(sample.image_path),
                "composition": sample.composition_name,
                "stem": sample.stem,
                "obj": OBJ_NAMES[sample.obj_id],
                "bg": BG_NAMES[sample.bg_id],
                "bg_donor": BG_NAMES[1 - sample.bg_id],
                "co_occur_obj": CO_OCCUR_OBJ_NAMES[sample.co_occur_obj_id],
                "bg_aligned": int(sample.obj_id == sample.bg_id),
                "co_occur_obj_aligned": int(sample.obj_id == sample.co_occur_obj_id),
            }

            if "bg" in args.interactions:
                if args.bg_delta_mode == "co_absent":
                    bg_effect_anchor = margin_by_key["obj_bg_anchor"] - margin_by_key["bg_anchor"]
                    bg_effect_donor = margin_by_key["obj_bg_donor"] - margin_by_key["bg_donor"]
                    bg_anchor_key = "bg_anchor"
                    bg_donor_key = "bg_donor"
                    obj_bg_anchor_key = "obj_bg_anchor"
                    obj_bg_donor_key = "obj_bg_donor"
                else:
                    bg_effect_anchor = margin_by_key["full_anchor"] - margin_by_key["co_bg_anchor"]
                    bg_effect_donor = margin_by_key["full_donor"] - margin_by_key["co_bg_donor"]
                    bg_anchor_key = "co_bg_anchor"
                    bg_donor_key = "co_bg_donor"
                    obj_bg_anchor_key = "full_anchor"
                    obj_bg_donor_key = "full_donor"
                bg_delta_anchor_to_donor = bg_effect_donor - bg_effect_anchor
                anchor_bg_is_aligned = sample.obj_id == sample.bg_id
                # Standardize sign so positive always means:
                # object effect on aligned background > object effect on conflicting background.
                bg_delta = (
                    bg_effect_anchor - bg_effect_donor
                    if anchor_bg_is_aligned
                    else bg_effect_donor - bg_effect_anchor
                )
                row["target_margin_bg_anchor"] = margin_by_key[bg_anchor_key]
                row["target_margin_bg_donor"] = margin_by_key[bg_donor_key]
                row["target_margin_obj_bg_anchor"] = margin_by_key[obj_bg_anchor_key]
                row["target_margin_obj_bg_donor"] = margin_by_key[obj_bg_donor_key]
                row["obj_effect_bg_anchor"] = bg_effect_anchor
                row["obj_effect_bg_donor"] = bg_effect_donor
                row["delta_bg_anchor_to_donor"] = bg_delta_anchor_to_donor
                row["abs_delta_bg_anchor_to_donor"] = abs(bg_delta_anchor_to_donor)
                row["delta_bg"] = bg_delta
                row["abs_delta_bg"] = abs(bg_delta)
                row["bg_delta_mode"] = args.bg_delta_mode
                interaction_values["bg"].append(bg_delta)
                bucket = f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
                interaction_values_by_bucket["bg"][bucket].append(bg_delta)

            if "co_occur_obj" in args.interactions:
                if args.co_delta_mode == "bg_fixed":
                    co_delta = (
                        margin_by_key["obj_bg"]
                        - margin_by_key["bg_only"]
                        - margin_by_key["full"]
                        + margin_by_key["bg_co"]
                    )
                    row["target_margin_bg_only"] = margin_by_key["bg_only"]
                    row["target_margin_bg_co"] = margin_by_key["bg_co"]
                    row["target_margin_obj_bg"] = margin_by_key["obj_bg"]
                    row["target_margin_full"] = margin_by_key["full"]
                else:
                    co_delta = (
                        margin_by_key["obj_only"]
                        - margin_by_key["empty"]
                        - margin_by_key["obj_co"]
                        + margin_by_key["co_only"]
                    )
                    row["target_margin_empty"] = margin_by_key["empty"]
                    row["target_margin_co_only"] = margin_by_key["co_only"]
                    row["target_margin_obj_only"] = margin_by_key["obj_only"]
                    row["target_margin_obj_co"] = margin_by_key["obj_co"]
                row["delta_co_occur_obj"] = co_delta
                row["abs_delta_co_occur_obj"] = abs(co_delta)
                row["co_delta_mode"] = args.co_delta_mode
                interaction_values["co_occur_obj"].append(co_delta)
                bucket = "aligned" if sample.obj_id == sample.co_occur_obj_id else "conflict"
                interaction_values_by_bucket["co_occur_obj"][bucket].append(co_delta)

            if "three_way" in args.interactions:
                three_way_anchor = (
                    margin_by_key["obj_bg_anchor"]
                    - margin_by_key["bg_anchor"]
                    - margin_by_key["full_anchor"]
                    + margin_by_key["co_bg_anchor"]
                )
                three_way_donor = (
                    margin_by_key["obj_bg_donor"]
                    - margin_by_key["bg_donor"]
                    - margin_by_key["full_donor"]
                    + margin_by_key["co_bg_donor"]
                )
                three_way_anchor_to_donor = three_way_donor - three_way_anchor
                three_way = (
                    three_way_anchor - three_way_donor
                    if sample.obj_id == sample.bg_id
                    else three_way_donor - three_way_anchor
                )
                row["target_margin_co_bg_anchor"] = margin_by_key["co_bg_anchor"]
                row["target_margin_co_bg_donor"] = margin_by_key["co_bg_donor"]
                row["target_margin_full_anchor"] = margin_by_key["full_anchor"]
                row["target_margin_full_donor"] = margin_by_key["full_donor"]
                row["delta_three_way_anchor_to_donor"] = three_way_anchor_to_donor
                row["abs_delta_three_way_anchor_to_donor"] = abs(three_way_anchor_to_donor)
                row["delta_three_way"] = three_way
                row["abs_delta_three_way"] = abs(three_way)
                interaction_values["three_way"].append(three_way)
                bucket = (
                    f"source_bg_{'aligned' if sample.obj_id == sample.bg_id else 'conflict'}"
                    f"__co_{'aligned' if sample.obj_id == sample.co_occur_obj_id else 'conflict'}"
                )
                interaction_values_by_bucket["three_way"][bucket].append(three_way)

            rows.append(row)

    fieldnames = sorted({key for row in rows for key in row})
    csv_path = output_dir / "audit_scores.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "config": config,
        "artifacts": {
            "scores_csv": str(csv_path),
            "corner_examples_root": str(output_dir / "corner_examples"),
        },
        "interactions": {},
        "finished_at_utc": datetime.utcnow().isoformat() + "Z",
    }
    for interaction_name, values in interaction_values.items():
        summary["interactions"][interaction_name] = {
            "overall": summarize_values(values),
            "by_bucket": {
                bucket: summarize_values(bucket_values)
                for bucket, bucket_values in sorted(interaction_values_by_bucket[interaction_name].items())
            },
        }

    serialize_json(summary, output_dir / "audit_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
