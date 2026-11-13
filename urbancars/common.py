from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torchvision.models as tv_models
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(os.environ["DATA_ROOT"]) if os.environ.get("DATA_ROOT") else ROOT / "data" / "raw"
DEFAULT_OUTPUT_ROOT = ROOT / "runs"

OBJ_NAMES = ["urban", "country"]
BG_NAMES = ["urban", "country"]
CO_OCCUR_OBJ_NAMES = ["urban", "country"]
NAME_TO_ID = {name: idx for idx, name in enumerate(OBJ_NAMES)}

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_COMPOSITION_RE = re.compile(
    r"^obj-(?P<obj>[^_]+)_bg-(?P<bg>[^_]+)_co_occur_obj-(?P<co>[^/]+)$"
)


@dataclass(frozen=True)
class UrbanCarsSample:
    image_path: Path
    split: str
    ratio_folder: str
    obj_id: int
    bg_id: int
    co_occur_obj_id: int
    obj_mask_path: Path | None
    co_occur_obj_mask_path: Path | None
    bg_source_path: Path | None
    bg_only_path: Path | None

    @property
    def composition_name(self) -> str:
        return self.image_path.parent.name

    @property
    def stem(self) -> str:
        return self.image_path.stem

    @property
    def stem_index(self) -> int | None:
        try:
            return int(self.stem)
        except ValueError:
            return None


def ratio_folder_for_split(split: str) -> str:
    if split == "train":
        return "bg-0.95_co_occur_obj-0.95"
    if split in {"val", "test"}:
        return "bg-0.5_co_occur_obj-0.5"
    raise ValueError(f"Unsupported split: {split}")


def parse_composition_name(name: str) -> tuple[int, int, int]:
    match = _COMPOSITION_RE.match(name)
    if match is None:
        raise ValueError(f"Could not parse composition directory name: {name}")
    obj_id = NAME_TO_ID[match.group("obj")]
    bg_id = NAME_TO_ID[match.group("bg")]
    co_id = NAME_TO_ID[match.group("co")]
    return obj_id, bg_id, co_id


def enumerate_urbancars_samples(
    data_root: Path | str = DEFAULT_DATA_ROOT,
    split: str = "train",
    *,
    include_auxiliary_assets: bool = True,
) -> list[UrbanCarsSample]:
    data_root = Path(data_root)
    ratio_folder = ratio_folder_for_split(split)
    split_root = data_root / "urbancars" / ratio_folder / split
    if not split_root.exists():
        raise FileNotFoundError(f"UrbanCars split root does not exist: {split_root}")

    samples: list[UrbanCarsSample] = []
    for composition_dir in sorted(p for p in split_root.iterdir() if p.is_dir()):
        obj_id, bg_id, co_id = parse_composition_name(composition_dir.name)
        for image_path in sorted(composition_dir.glob("*.jpg")):
            stem = image_path.stem
            obj_mask_path = composition_dir / f"{stem}_mask.png"
            co_mask_path = composition_dir / f"{stem}_co_occur_obj_mask.png"
            bg_source_path = composition_dir / f"{stem}_bg_source.png"
            bg_only_path = composition_dir / f"{stem}_bg_only.png"
            samples.append(
                UrbanCarsSample(
                    image_path=image_path,
                    split=split,
                    ratio_folder=ratio_folder,
                    obj_id=obj_id,
                    bg_id=bg_id,
                    co_occur_obj_id=co_id,
                    obj_mask_path=obj_mask_path if include_auxiliary_assets and obj_mask_path.exists() else None,
                    co_occur_obj_mask_path=co_mask_path if include_auxiliary_assets and co_mask_path.exists() else None,
                    bg_source_path=bg_source_path if include_auxiliary_assets and bg_source_path.exists() else None,
                    bg_only_path=bg_only_path if include_auxiliary_assets and bg_only_path.exists() else None,
                )
            )
    return samples


class UrbanCarsClassificationDataset(Dataset):
    def __init__(
        self,
        data_root: Path | str = DEFAULT_DATA_ROOT,
        split: str = "train",
        *,
        transform=None,
        return_path: bool = False,
    ) -> None:
        self.samples = enumerate_urbancars_samples(data_root, split, include_auxiliary_assets=False)
        self.transform = transform
        self.return_path = return_path

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(
            [sample.obj_id, sample.bg_id, sample.co_occur_obj_id],
            dtype=torch.long,
        )
        payload = {
            "image": image,
            "label": label,
        }
        if self.return_path:
            payload["image_path"] = str(sample.image_path)
        return payload


def build_classifier(backbone: str, num_classes: int, weights_name: str = "IMAGENET1K_V1"):
    supported = {
        "resnet18": tv_models.ResNet18_Weights,
        "resnet34": tv_models.ResNet34_Weights,
        "resnet50": tv_models.ResNet50_Weights,
        "resnet101": tv_models.ResNet101_Weights,
    }
    if backbone not in supported:
        raise ValueError(
            f"Unsupported backbone: {backbone}. Supported backbones: {sorted(supported)}"
        )
    weight_enum = None
    if weights_name.lower() != "none":
        weight_enum = supported[backbone][weights_name]
    try:
        model = getattr(tv_models, backbone)(weights=weight_enum)
    except Exception as exc:
        if weights_name.lower() != "none":
            raise RuntimeError(
                f"Could not load torchvision weights {weights_name} for {backbone}. "
                "Make sure the weights are cached locally or network access is available, "
                "or rerun with --weights none."
            ) from exc
        raise
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    return model


def get_transforms(is_training: bool):
    normalize = T.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist())
    if is_training:
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize,
            ]
        )
    return T.Compose(
        [
            T.ToTensor(),
            normalize,
        ]
    )


def evaluate_group_metrics(
    model: torch.nn.Module,
    loader,
    *,
    device: torch.device,
    amp: bool,
    split: str,
    bg_ratio: float,
    co_ratio: float,
) -> dict[str, float]:
    model.eval()
    correct_cube = torch.zeros((2, 2, 2), dtype=torch.float64)
    count_cube = torch.zeros((2, 2, 2), dtype=torch.float64)
    total_correct = []
    total_bg_correct = []
    total_co_correct = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp and device.type == "cuda"):
                logits = model(images)
            pred = logits.argmax(dim=1)

            obj = labels[:, 0]
            bg = labels[:, 1]
            co = labels[:, 2]
            correct = pred == obj
            total_correct.append(correct.cpu())
            total_bg_correct.append((pred == bg).cpu())
            total_co_correct.append((pred == co).cpu())

            flat_index = (obj * 4 + bg * 2 + co).cpu()
            correct_cube.view(-1).index_add_(0, flat_index, correct.cpu().to(torch.float64))
            count_cube.view(-1).index_add_(
                0,
                flat_index,
                torch.ones_like(flat_index, dtype=torch.float64),
            )

    group_acc = correct_cube / count_cube.clamp_min(1.0)
    metrics: dict[str, float] = {}

    absent_present = ["absent", "present"]
    bg_ratio_list = [1.0 - bg_ratio, bg_ratio]
    co_ratio_list = [1.0 - co_ratio, co_ratio]
    weighted_group_acc = 0.0

    for bg_shortcut in range(2):
        for co_shortcut in range(2):
            values = []
            for obj_id in range(2):
                bg_id = obj_id if bg_shortcut == 1 else 1 - obj_id
                co_id = obj_id if co_shortcut == 1 else 1 - obj_id
                values.append(group_acc[obj_id, bg_id, co_id].item())
            acc = float(np.mean(values))
            key = (
                f"{split}_bg_{absent_present[bg_shortcut]}"
                f"_co_occur_obj_{absent_present[co_shortcut]}_acc"
            )
            metrics[key] = acc
            weighted_group_acc += acc * bg_ratio_list[bg_shortcut] * co_ratio_list[co_shortcut]

    bg_gap = metrics[f"{split}_bg_absent_co_occur_obj_present_acc"] - weighted_group_acc
    co_gap = metrics[f"{split}_bg_present_co_occur_obj_absent_acc"] - weighted_group_acc
    both_gap = metrics[f"{split}_bg_absent_co_occur_obj_absent_acc"] - weighted_group_acc

    metrics[f"{split}_id_acc"] = weighted_group_acc
    metrics[f"{split}_bg_gap"] = bg_gap
    metrics[f"{split}_co_occur_obj_gap"] = co_gap
    metrics[f"{split}_both_gap"] = both_gap

    bg_wga = (correct_cube.sum(dim=2) / count_cube.sum(dim=2).clamp_min(1.0)).min().item()
    co_wga = (correct_cube.sum(dim=1) / count_cube.sum(dim=1).clamp_min(1.0)).min().item()
    both_wga = group_acc.min().item()
    metrics[f"{split}_bg_worst_group_acc"] = bg_wga
    metrics[f"{split}_co_occur_obj_worst_group_acc"] = co_wga
    metrics[f"{split}_both_worst_group_acc"] = both_wga

    total_correct_t = torch.cat(total_correct, dim=0).float()
    total_bg_t = torch.cat(total_bg_correct, dim=0).float()
    total_co_t = torch.cat(total_co_correct, dim=0).float()
    metrics[f"{split}_cue_obj_acc"] = total_correct_t.mean().item()
    metrics[f"{split}_cue_bg_acc"] = total_bg_t.mean().item()
    metrics[f"{split}_cue_co_occur_obj_acc"] = total_co_t.mean().item()

    for obj_id in range(2):
        for bg_id in range(2):
            for co_id in range(2):
                key = (
                    f"{split}_obj_{OBJ_NAMES[obj_id]}_bg_{BG_NAMES[bg_id]}"
                    f"_co_occur_obj_{CO_OCCUR_OBJ_NAMES[co_id]}_acc"
                )
                metrics[key] = group_acc[obj_id, bg_id, co_id].item()

    return metrics


def load_binary_mask(path: Path) -> np.ndarray:
    mask = Image.open(path).convert("L")
    mask_arr = np.asarray(mask, dtype=np.float32)
    return (mask_arr > 127.0).astype(np.float32)[..., None]


def load_rgb(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32)


def make_neutral_canvas(size: tuple[int, int], fill_mode: str = "imagenet_mean") -> np.ndarray:
    width, height = size
    if fill_mode == "imagenet_mean":
        color = (IMAGENET_MEAN * 255.0).round().astype(np.uint8)
    elif fill_mode == "black":
        color = np.zeros((3,), dtype=np.uint8)
    else:
        raise ValueError(f"Unsupported fill mode: {fill_mode}")
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = color
    return canvas.astype(np.float32)


def alpha_composite(foreground: np.ndarray, alpha: np.ndarray, background: np.ndarray) -> np.ndarray:
    alpha = np.clip(alpha, 0.0, 1.0)
    return foreground * alpha + background * (1.0 - alpha)


def extract_sample_components(sample: UrbanCarsSample) -> dict[str, np.ndarray]:
    if sample.obj_mask_path is None or sample.co_occur_obj_mask_path is None:
        raise FileNotFoundError(f"Missing mask assets for sample: {sample.image_path}")

    full = load_rgb(sample.image_path)
    obj_alpha = load_binary_mask(sample.obj_mask_path)
    co_alpha_raw = load_binary_mask(sample.co_occur_obj_mask_path)
    co_alpha = co_alpha_raw * (1.0 - obj_alpha)
    if sample.bg_source_path is not None:
        bg = load_rgb(sample.bg_source_path)
    elif sample.bg_only_path is not None:
        bg = load_rgb(sample.bg_only_path)
    else:
        bg = full.copy()
    obj_rgb = full * obj_alpha
    co_rgb = full * co_alpha
    return {
        "full": full,
        "background": bg,
        "obj_alpha": obj_alpha,
        "obj_rgb": obj_rgb,
        "co_alpha": co_alpha,
        "co_rgb": co_rgb,
    }


def compose_from_components(
    *,
    background: np.ndarray,
    obj_rgb: np.ndarray | None = None,
    obj_alpha: np.ndarray | None = None,
    co_rgb: np.ndarray | None = None,
    co_alpha: np.ndarray | None = None,
) -> np.ndarray:
    image = background.copy()
    if co_rgb is not None and co_alpha is not None:
        image = alpha_composite(co_rgb, co_alpha, image)
    if obj_rgb is not None and obj_alpha is not None:
        image = alpha_composite(obj_rgb, obj_alpha, image)
    return image


def group_samples_by_factors(
    samples: Iterable[UrbanCarsSample],
) -> dict[tuple[str, int, int, int], list[UrbanCarsSample]]:
    buckets: dict[tuple[str, int, int, int], list[UrbanCarsSample]] = {}
    for sample in samples:
        key = (
            sample.split,
            sample.obj_id,
            sample.bg_id,
            sample.co_occur_obj_id,
        )
        buckets.setdefault(key, []).append(sample)
    return buckets


def pick_paired_sample(
    anchor: UrbanCarsSample,
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
    *,
    bg_id: int,
    co_occur_obj_id: int | None = None,
) -> UrbanCarsSample:
    if co_occur_obj_id is None:
        co_occur_obj_id = anchor.co_occur_obj_id
    candidates = grouped_samples.get(
        (anchor.split, anchor.obj_id, bg_id, co_occur_obj_id),
        [],
    )
    if not candidates:
        raise KeyError(
            "No paired UrbanCars sample found for "
            f"split={anchor.split}, obj={anchor.obj_id}, bg={bg_id}, co={co_occur_obj_id}"
        )
    if anchor.stem_index is not None:
        same_stem = [candidate for candidate in candidates if candidate.stem == anchor.stem]
        if same_stem:
            return same_stem[0]
        return candidates[anchor.stem_index % len(candidates)]
    return candidates[0]


def build_interaction_variants(
    sample: UrbanCarsSample,
    *,
    fill_mode: str = "imagenet_mean",
) -> dict[str, Image.Image]:
    if sample.obj_mask_path is None or sample.co_occur_obj_mask_path is None:
        raise FileNotFoundError(f"Missing mask assets for sample: {sample.image_path}")
    full = load_rgb(sample.image_path)
    obj_alpha = load_binary_mask(sample.obj_mask_path)
    co_alpha_raw = load_binary_mask(sample.co_occur_obj_mask_path)
    co_alpha = co_alpha_raw * (1.0 - obj_alpha)

    height, width = full.shape[:2]
    empty = make_neutral_canvas((width, height), fill_mode=fill_mode)
    if sample.bg_source_path is not None:
        bg = load_rgb(sample.bg_source_path)
    elif sample.bg_only_path is not None:
        bg = load_rgb(sample.bg_only_path)
    else:
        bg = empty.copy()

    obj_only = alpha_composite(full, obj_alpha, empty)
    co_only = alpha_composite(full, co_alpha, empty)
    bg_only = bg.copy()
    obj_bg = alpha_composite(full, obj_alpha, bg)
    bg_co = alpha_composite(full, co_alpha, bg)
    obj_co = alpha_composite(full, obj_alpha, co_only)

    return {
        "empty": Image.fromarray(np.clip(empty, 0, 255).astype(np.uint8)),
        "bg_only": Image.fromarray(np.clip(bg_only, 0, 255).astype(np.uint8)),
        "obj_only": Image.fromarray(np.clip(obj_only, 0, 255).astype(np.uint8)),
        "co_only": Image.fromarray(np.clip(co_only, 0, 255).astype(np.uint8)),
        "obj_bg": Image.fromarray(np.clip(obj_bg, 0, 255).astype(np.uint8)),
        "bg_co": Image.fromarray(np.clip(bg_co, 0, 255).astype(np.uint8)),
        "obj_co": Image.fromarray(np.clip(obj_co, 0, 255).astype(np.uint8)),
        "full": Image.fromarray(np.clip(full, 0, 255).astype(np.uint8)),
    }


def build_co_conditional_variants(sample: UrbanCarsSample) -> dict[str, Image.Image]:
    components = extract_sample_components(sample)
    bg = components["background"]
    bg_only = bg
    bg_co = compose_from_components(
        background=bg,
        co_rgb=components["co_rgb"],
        co_alpha=components["co_alpha"],
    )
    obj_bg = compose_from_components(
        background=bg,
        obj_rgb=components["obj_rgb"],
        obj_alpha=components["obj_alpha"],
    )
    full = compose_from_components(
        background=bg,
        obj_rgb=components["obj_rgb"],
        obj_alpha=components["obj_alpha"],
        co_rgb=components["co_rgb"],
        co_alpha=components["co_alpha"],
    )
    return {
        "bg_only": Image.fromarray(np.clip(bg_only, 0, 255).astype(np.uint8)),
        "bg_co": Image.fromarray(np.clip(bg_co, 0, 255).astype(np.uint8)),
        "obj_bg": Image.fromarray(np.clip(obj_bg, 0, 255).astype(np.uint8)),
        "full": Image.fromarray(np.clip(full, 0, 255).astype(np.uint8)),
    }


def build_bg_conditional_variants(
    sample: UrbanCarsSample,
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
) -> dict[str, Image.Image]:
    anchor_components = extract_sample_components(sample)
    anchor_bg = anchor_components["background"]
    donor_bg_sample = pick_paired_sample(
        sample,
        grouped_samples,
        bg_id=1 - sample.bg_id,
    )
    donor_bg = extract_sample_components(donor_bg_sample)["background"]

    obj_on_anchor = compose_from_components(
        background=anchor_bg,
        obj_rgb=anchor_components["obj_rgb"],
        obj_alpha=anchor_components["obj_alpha"],
    )
    obj_on_donor = compose_from_components(
        background=donor_bg,
        obj_rgb=anchor_components["obj_rgb"],
        obj_alpha=anchor_components["obj_alpha"],
    )

    return {
        "bg_anchor": Image.fromarray(np.clip(anchor_bg, 0, 255).astype(np.uint8)),
        "bg_donor": Image.fromarray(np.clip(donor_bg, 0, 255).astype(np.uint8)),
        "obj_bg_anchor": Image.fromarray(np.clip(obj_on_anchor, 0, 255).astype(np.uint8)),
        "obj_bg_donor": Image.fromarray(np.clip(obj_on_donor, 0, 255).astype(np.uint8)),
    }


def build_three_way_conditional_variants(
    sample: UrbanCarsSample,
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
) -> dict[str, Image.Image]:
    anchor_components = extract_sample_components(sample)
    anchor_bg = anchor_components["background"]
    donor_bg_sample = pick_paired_sample(
        sample,
        grouped_samples,
        bg_id=1 - sample.bg_id,
    )
    donor_bg = extract_sample_components(donor_bg_sample)["background"]

    def render(background: np.ndarray, *, obj_present: bool, co_present: bool) -> np.ndarray:
        return compose_from_components(
            background=background,
            obj_rgb=anchor_components["obj_rgb"] if obj_present else None,
            obj_alpha=anchor_components["obj_alpha"] if obj_present else None,
            co_rgb=anchor_components["co_rgb"] if co_present else None,
            co_alpha=anchor_components["co_alpha"] if co_present else None,
        )

    variants = {
        "bg_anchor": render(anchor_bg, obj_present=False, co_present=False),
        "co_bg_anchor": render(anchor_bg, obj_present=False, co_present=True),
        "obj_bg_anchor": render(anchor_bg, obj_present=True, co_present=False),
        "full_anchor": render(anchor_bg, obj_present=True, co_present=True),
        "bg_donor": render(donor_bg, obj_present=False, co_present=False),
        "co_bg_donor": render(donor_bg, obj_present=False, co_present=True),
        "obj_bg_donor": render(donor_bg, obj_present=True, co_present=False),
        "full_donor": render(donor_bg, obj_present=True, co_present=True),
    }
    return {
        key: Image.fromarray(np.clip(value, 0, 255).astype(np.uint8))
        for key, value in variants.items()
    }


def build_joint_conditional_variants(
    sample: UrbanCarsSample,
    grouped_samples: dict[tuple[str, int, int, int], list[UrbanCarsSample]],
) -> dict[str, Image.Image]:
    """
    Build the shared 8-corner set used by UrbanCars steering.

    The "anchor" background is the sample's own background. The "donor"
    background is chosen from the opposite background label while keeping
    the object and co-occurring object labels fixed.
    """
    anchor_components = extract_sample_components(sample)
    anchor_bg = anchor_components["background"]
    donor_bg_sample = pick_paired_sample(
        sample,
        grouped_samples,
        bg_id=1 - sample.bg_id,
    )
    donor_bg = extract_sample_components(donor_bg_sample)["background"]

    def render(background: np.ndarray, *, obj_present: bool, co_present: bool) -> np.ndarray:
        return compose_from_components(
            background=background,
            obj_rgb=anchor_components["obj_rgb"] if obj_present else None,
            obj_alpha=anchor_components["obj_alpha"] if obj_present else None,
            co_rgb=anchor_components["co_rgb"] if co_present else None,
            co_alpha=anchor_components["co_alpha"] if co_present else None,
        )

    variants = {
        "bg_anchor": render(anchor_bg, obj_present=False, co_present=False),
        "co_bg_anchor": render(anchor_bg, obj_present=False, co_present=True),
        "obj_bg_anchor": render(anchor_bg, obj_present=True, co_present=False),
        "full_anchor": render(anchor_bg, obj_present=True, co_present=True),
        "bg_donor": render(donor_bg, obj_present=False, co_present=False),
        "co_bg_donor": render(donor_bg, obj_present=False, co_present=True),
        "obj_bg_donor": render(donor_bg, obj_present=True, co_present=False),
        "full_donor": render(donor_bg, obj_present=True, co_present=True),
    }
    return {
        key: Image.fromarray(np.clip(value, 0, 255).astype(np.uint8))
        for key, value in variants.items()
    }


def make_panel(
    images: list[Image.Image],
    labels: list[str],
    *,
    title: str,
    columns: int = 2,
    label_height: int = 24,
    title_height: int = 30,
    padding: int = 8,
) -> Image.Image:
    if len(images) != len(labels):
        raise ValueError("images and labels must have the same length")
    if not images:
        raise ValueError("At least one image is required")

    tile_width, tile_height = images[0].size
    rows = (len(images) + columns - 1) // columns
    width = padding + columns * (tile_width + padding)
    height = title_height + padding + rows * (tile_height + label_height + padding)
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((padding, 8), title, fill=(0, 0, 0), font=font)

    for index, (image, label) in enumerate(zip(images, labels)):
        row = index // columns
        col = index % columns
        x = padding + col * (tile_width + padding)
        y = title_height + padding + row * (tile_height + label_height + padding)
        canvas.paste(image, (x, y))
        draw.text((x, y + tile_height + 4), label, fill=(0, 0, 0), font=font)
    return canvas


def serialize_json(payload: dict, path: Path) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def first_sample_per_composition(samples: Iterable[UrbanCarsSample], limit: int) -> list[UrbanCarsSample]:
    selected: list[UrbanCarsSample] = []
    seen: set[str] = set()
    for sample in samples:
        if sample.composition_name in seen:
            continue
        seen.add(sample.composition_name)
        selected.append(sample)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for sample in samples:
            if sample in selected:
                continue
            selected.append(sample)
            if len(selected) >= limit:
                break
    return selected
