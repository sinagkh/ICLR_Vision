from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset
from PIL import Image


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("DATA_ROOT", ROOT / "raw")) / "stanford_cars"
DEFAULT_HF_HOME = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
DATASET_NAME = "tanganke/stanford_cars"


def _export_split(
    *,
    split: str,
    output_root: Path,
    hf_home: Path,
    force: bool,
) -> dict[str, object]:
    dataset = load_dataset(
        DATASET_NAME,
        split=split,
        cache_dir=str(hf_home),
    )
    image_dir = output_root / f"cars_{split}"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / f"hf_{split}_manifest.jsonl"
    filter_dir = output_root / "filters"
    filter_dir.mkdir(parents=True, exist_ok=True)
    filter_path = filter_dir / f"{split}_all.txt"

    class_names = dataset.features["label"].names
    manifest_rows: list[dict[str, object]] = []
    filter_names: list[str] = []

    for index, row in enumerate(dataset, start=1):
        filename = f"{index:05d}.jpg"
        image_path = image_dir / filename
        image = row["image"]
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected PIL image for split={split}, got {type(image)!r}")
        rgb = image.convert("RGB")
        if force or not image_path.exists():
            rgb.save(image_path, format="JPEG", quality=95)
        width, height = rgb.size
        label = int(row["label"])
        manifest_rows.append(
            {
                "filename": filename,
                "image_path": str(image_path),
                "label": label,
                "class_name": str(class_names[label]),
                "width": width,
                "height": height,
            }
        )
        filter_names.append(filename)

    manifest_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in manifest_rows) + "\n",
        encoding="utf-8",
    )
    filter_path.write_text("\n".join(filter_names) + "\n", encoding="utf-8")
    return {
        "split": split,
        "row_count": len(manifest_rows),
        "image_dir": str(image_dir),
        "manifest_path": str(manifest_path),
        "filter_path": str(filter_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Stanford Cars from the Hugging Face mirror into the UrbanCars workspace.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--hf-home", default=str(DEFAULT_HF_HOME))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root).resolve()
    hf_home = Path(args.hf_home).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    summary = {
        "dataset": DATASET_NAME,
        "output_root": str(output_root),
        "hf_home": str(hf_home),
        "splits": [],
    }

    class_names: list[str] | None = None
    for split in ("train", "test"):
        split_summary = _export_split(
            split=split,
            output_root=output_root,
            hf_home=hf_home,
            force=args.force,
        )
        summary["splits"].append(split_summary)
        if class_names is None:
            dataset = load_dataset(DATASET_NAME, split=split, cache_dir=str(hf_home))
            class_names = list(dataset.features["label"].names)

    if class_names is None:
        class_names = []
    (output_root / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (output_root / "hf_export_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
