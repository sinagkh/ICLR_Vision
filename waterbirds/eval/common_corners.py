#!/usr/bin/env python3
"""Common-corner interaction audit on the fixed 2,153 test anchors (three donors each) with the released CUB masks.

No training or checkpoint selection. Uses the renderer of data/audit_interaction.py with the CUB-mask provider.
Both fill RNGs are seeded per anchor (123 + 97 * idx), so every model is scored on identical corner pixels; pixel
equality with the sequential auditor is verified on four anchors, and independent corners are batched. ERM is
scored on the identical renders as a reference. `load_model` is also used by the other evaluation scripts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import random
import shlex
import sys
import time
from datetime import datetime, timezone
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import models

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / "data"))
from waterbirds_data import GroundTruthMaskAuditWaterbirds, configure_ground_truth_class

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data_root"))
CKPT = Path(os.environ.get("CKPT_ROOT", "checkpoints"))
ANCHORS = PKG / "lists/common_audit_anchors_test.json"
SUBSET = PKG / "lists/common_audit_anchors_test_subset192.json"
REFERENCE_CSV = PKG / "lists/common_audit_reference_erm_seed1.csv"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    with Path(path).open("w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def load_renderer():
    sys.path.insert(0, str(PKG / "train"))
    spec = importlib.util.spec_from_file_location("interaction_audit", PKG / "data" / "audit_interaction.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CV2 = None  # The reported runs used the non-OpenCV fill.
    return module


class FixedCorners(Dataset):
    def __init__(self, anchors, renderer):
        self.anchors = anchors
        self.r = renderer
        cls = configure_ground_truth_class(
            GroundTruthMaskAuditWaterbirds, DATA_ROOT / "hf_cache", renderer,
            DATA_ROOT / "waterbird_complete95_forest2water2",
            DATA_ROOT / "segmentations",
        )
        self.ds = cls("test", image_size=224, train_aug=False, backbone="resnet50")

    def __len__(self):
        return len(self.anchors)

    def __getitem__(self, index):
        # Freeze the fill realization independent of workers, model, and seed.
        random_state, numpy_state = random.getstate(), np.random.get_state()
        fill_seed = 123 + 97 * self.anchors[index]["idx"]
        random.seed(fill_seed)
        np.random.seed(fill_seed)
        try:
            return self.render(index)
        finally:
            random.setstate(random_state)
            np.random.set_state(numpy_state)

    def render(self, index):
        a = self.anchors[index]
        _, y, group, anchor, bg = self.ds[a["idx"]]
        if (y, group, bg) != (a["y"], a["group"], a["bg"]):
            raise ValueError(f"Anchor identity mismatch: {a['idx']}")
        mask = self.ds.load_mask(a["idx"])
        box = self.r.mask_bbox(mask)
        if box is None or (box[2] - box[0]) * (box[3] - box[1]) > 0.45 * anchor.width * anchor.height:
            raise ValueError(f"Anchor is not eligible: {a['idx']}")
        x10 = self.r.inpaint_with_mask(anchor, mask, radius=4, expand_radius=3, threshold=127)
        crop = anchor.crop(tuple(box))
        segment = self.r.threshold_mask(self.r.mask_crop(mask, box), threshold=127)
        present, absent = [], []
        for donor_id in a["donors"]:
            _, _, _, donor, _ = self.ds[donor_id]
            donor_mask = self.ds.load_mask(donor_id)
            if donor_mask is None:
                raise ValueError(f"Missing donor mask: {donor_id}")
            clean = self.r.cleanse_donor(donor, donor_mask)
            adjusted = self.r.hist_match(clean, anchor)
            present.append(self.r.paste_with_segment_mask(adjusted, crop, tuple(box), segment, blur_radius=0))
            absent.append(clean)  # Same asymmetric photometric adjustment as the training corners.
        images = [anchor, x10, *present, *absent]
        if len(images) != 8:
            raise ValueError("The audit must have three donors per anchor")
        return torch.stack([self.ds.tf(x) for x in images]), a["idx"], a["y"]


def load_model(path, device):
    # Every parameter is loaded strictly; no download or pretrained initialization.
    model = models.resnet50(weights=None, num_classes=2)
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model", state), strict=True)
    return model.to(device=device, memory_format=torch.channels_last).eval()


def summarize(logits, target):
    score = logits[:, target] - logits[:, 1 - target]
    interaction = (score[5:8] - score[2:5]) - (score[1] - score[0])
    mean = float(interaction.mean())
    return {"d2_mean": mean, "d2_abs": abs(mean),
            "e_obj": float(score[0] - score[1]),
            "e_bg": float((score[0] - score[2:5]).mean())}


class RecordingModel(torch.nn.Module):
    """Record the auditor's inputs without scoring a model."""
    def __init__(self):
        super().__init__()
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.cpu().clone())
        return torch.zeros((len(x), 2), device=x.device)


def validate_renderer(dataset, out):
    checks = []
    for i in range(4):
        anchor = dataset.anchors[i]
        path = out / f"validation_anchor_{i}.json"
        save_json(path, [anchor])
        fill_seed = 123 + 97 * anchor["idx"]
        recorder = RecordingModel()
        random.seed(fill_seed)
        dataset.r.d2_audit(recorder, dataset.ds, torch.device("cpu"), seed=fill_seed,
                           load_anchors_path=path, use_margin=True)
        expected = torch.cat(recorder.inputs)
        actual, idx, _ = dataset[i]
        difference = float((expected-actual).abs().max())
        if difference != 0.0 or not torch.equal(actual, dataset[i][0]):
            raise RuntimeError("Corner pixels differ from the seeded sequential renderer")
        checks.append({"idx": idx, "sequential_renderer_max_pixel_difference": difference,
                       "repeated_render_exact": True,
                       "tensor_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
    save_json(out / "renderer_validation.json", checks)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 21, 42])
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--anchor-batch-size", type=int, default=16)
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    if (out / "run_manifest.json").exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {out}")
    with (out / "commands.log").open("x") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()}\n{shlex.join([sys.executable, *sys.argv])}\n")
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    anchors = json.loads(ANCHORS.read_text())
    subset = json.loads(SUBSET.read_text())
    with REFERENCE_CSV.open(newline="") as f:
        reference = list(csv.DictReader(f))
    if len(anchors) != 2153 or [a["idx"] for a in anchors] != [int(r["idx"]) for r in reference]:
        raise ValueError("Anchor list does not exactly match the reference audit rows")
    keyed = {a["idx"]: a for a in anchors}
    if len(keyed) != 2153 or any(keyed[a["idx"]] != a for a in subset):
        raise ValueError("The 192-anchor subset is not an exact member set of the full audit")
    renderer = load_renderer()
    checkpoint_paths = {f"{condition}/seed{s}": CKPT / f"{condition}/seed{s}/checkpoints/{filename}"
                        for condition, filename in [("predicted_mask_is", "steered_finetuned_best.pt"),
                                                    ("erm", "erm_best.pt")]
                        for s in args.seeds}
    dtype = "cuda float16 autocast" if device.type == "cuda" else "CPU float32"
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "device": str(device),
                "torch": torch.__version__, "dtype": dtype, "training": False,
                "checkpoint_selection": "existing validation-WGA-selected checkpoints; unchanged",
                "mask_source": "released CUB masks, evaluation only", "n": 2153,
                "donors_per_anchor": 3, "row_statistic": "absolute value of signed donor-mean margin interaction",
                "renderer": "data/audit_interaction.py; CV2=None; histogram matching as in training",
                "fill_rng": "Python and NumPy seeded per anchor with 123 + 97 * idx; identical across models and seeds",
                "anchor_manifest": str(ANCHORS), "anchor_sha256": digest(ANCHORS),
                "subset_sha256": digest(SUBSET), "script_sha256": digest(__file__),
                "renderer_sha256": digest(PKG / "data" / "audit_interaction.py"),
                "seeds": args.seeds, "settings": {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
                "precision_note": "GPU audits use float16 autocast. Implementation validation uses float32.",
                "checks": {"exact_reference_row_order": True, "exact_subset_stencils": True},
                "checkpoints": {key: {"path": str(path), "sha256": digest(path)}
                                for key, path in checkpoint_paths.items()}}
    # Write the run manifest before scoring any model.
    save_json(out / "run_manifest.json", manifest)
    dataset = FixedCorners(anchors, renderer)
    check_set = FixedCorners(subset[:4], renderer)
    print(json.dumps({"renderer_validation": validate_renderer(check_set, out)}), flush=True)
    results = {}
    reference_render_hashes = None
    for key in checkpoint_paths:
        condition, seed_name = key.split("/")
        seed = int(seed_name[4:])
        model = load_model(manifest["checkpoints"][key]["path"], device)
        started = time.perf_counter()
        checks = []
        with torch.inference_mode():
            for i in range(len(check_set)):
                x, idx, y = check_set[i]
                x = x.to(device=device, memory_format=torch.channels_last)
                batched = model(x)
                sequential = torch.cat([model(x[j:j+1]) for j in range(len(x))])
                drift = float((batched-sequential).abs().max())
                checks.append({"idx": idx, "batch_vs_single_max_logit_difference": drift})
                if drift > 0.0002:
                    save_json(out / f"{condition}_{seed_name}_failed_validation.json", checks)
                    raise RuntimeError("Batched-versus-single inference validation failed")
        save_json(out / f"{condition}_{seed_name}_validation.json", checks)
        print(json.dumps({"model":key,"validation":checks,"seconds":time.perf_counter()-started}),flush=True)
        loader = DataLoader(dataset, batch_size=args.anchor_batch_size, shuffle=False,
                            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
                            pin_memory=device.type == "cuda")
        seed_dir = out / condition / seed_name
        seed_dir.mkdir(parents=True)
        rows = []
        render_hashes = []
        with (seed_dir / "scores.csv").open("x", newline="") as f, torch.inference_mode():
            writer = csv.DictWriter(f, fieldnames=["idx", "y", "group", "bg", "donors", "d2_mean", "d2_abs", "e_obj", "e_bg"])
            writer.writeheader()
            for batch_no, (x, indices, ys) in enumerate(loader):
                render_hashes.extend(hashlib.sha256(v.numpy().tobytes()).hexdigest() for v in x)
                amp = torch.autocast("cuda", dtype=torch.float16) if device.type == "cuda" else nullcontext()
                with amp:
                    logits = model(x.flatten(0,1).to(device=device, memory_format=torch.channels_last, non_blocking=True)).cpu().reshape(-1,8,2)
                for index, y, z in zip(indices.tolist(), ys.tolist(), logits):
                    a = keyed[index]
                    row = {"idx":index,"y":y,"group":a["group"],"bg":a["bg"],"donors":len(a["donors"]),**summarize(z,y)}
                    rows.append(row)
                    writer.writerow(row)
                if batch_no % 25 == 0:
                    f.flush()
                    print(json.dumps({"model":key,"completed":len(rows),"n":len(dataset),"seconds":time.perf_counter()-started}),flush=True)
        if [r["idx"] for r in rows] != [a["idx"] for a in anchors]:
            raise RuntimeError("Completed audit row identity/order mismatch")
        values = np.asarray([r["d2_abs"] for r in rows])
        if reference_render_hashes is None:
            reference_render_hashes = render_hashes
            save_json(out / "corner_tensor_hashes.json", [{"idx": a["idx"], "sha256": h} for a,h in zip(anchors, render_hashes)])
        elif render_hashes != reference_render_hashes:
            raise RuntimeError("Rendered tensors differ between model/seed conditions")
        stats = {"seed":seed,"n":len(rows),"mean":float(values.mean()),"median":float(np.median(values)),
                 "condition":condition,"all_corner_tensors_identical_across_conditions":True,
                 "p90":float(np.quantile(values,0.9)),"elapsed_seconds":time.perf_counter()-started,
                 "scores_sha256":digest(seed_dir / "scores.csv")}
        save_json(seed_dir / "stats.json", stats)
        results[key] = stats
        print(json.dumps({"completed_seed":stats}),flush=True)
        del model, loader
    aggregates = {}
    for condition in ("predicted_mask_is", "erm"):
        values = [r["mean"] for r in results.values() if r["condition"] == condition]
        aggregates[condition] = {"mean":float(np.mean(values)),
                                 "sample_sd":float(np.std(values,ddof=1)) if len(values)>1 else None}
    save_json(out / "summary.json", {"per_seed":results,"aggregates":aggregates,
                                      "n_per_seed":2153,"n_seeds":len(args.seeds),"device":str(device),"dtype":dtype})


if __name__ == "__main__":
    main()
