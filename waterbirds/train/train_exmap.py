#!/usr/bin/env python3
"""
ExMap (Chakraborty et al., 2024) on Waterbirds, ResNet-50.

1. load a seed-matched ERM checkpoint,
2. compute LRP explanation heatmaps (zennit, ExMap's ResNet rules) on the validation split,
3. cluster the heatmaps of each class into pseudo-groups with spectral clustering, the number of clusters chosen by the
   eigengap heuristic (--auto), and
4. retrain the last layer DFR-style: L1 logistic regression on frozen ERM features of pseudo-group-balanced validation
   samples, C chosen by pseudo-group worst accuracy on held-out validation halves, coefficients averaged over
   --dfr_eval_retrains fits and test group accuracies averaged over --dfr_outer_repeats repetitions.

Pseudo-groups take the place of group labels in the balancing and in the choice of C; the true groups only enter the
reported validation/test accuracies and the cluster diagnostics.

Outputs: --output_dir/checkpoints/exmap_best.pt (pseudo-groups, chosen C, cluster diagnostics) and train_summary.json.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
from tqdm import tqdm

from scipy.sparse import csgraph
from scipy.sparse.linalg import lobpcg
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.manifold._spectral_embedding import _set_diag
from sklearn.preprocessing import StandardScaler

import zennit
import zennit.attribution
import zennit.composites
import zennit.core
import zennit.torchvision
import zennit.types

from backbone_utils import forward_with_features
from train_focal_cvar_sd import HFWaterbirds, worst_group_metrics


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=enabled)
    return contextlib.nullcontext()


class GammaSymmetric(zennit.core.BasicHook):
    """ExMap's convolution rule for numerically stable ResNet LRP."""

    def __init__(self, gamma: float = 0.25):
        self.gamma = gamma

        def gradient_map(out_grad, outputs):
            positive = outputs[0] + outputs[1]
            negative = outputs[2] + outputs[3]
            alpha = 1 + gamma * (positive >= negative.abs())
            beta = 1 + gamma * (positive < negative.abs())
            out = positive * alpha + negative * beta
            factors = [alpha, alpha, beta, beta]
            return [fac * out_grad / zennit.core.stabilize(out) for fac in factors]

        super().__init__(
            input_modifiers=[
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
                lambda input: input.clamp(min=0),
                lambda input: input.clamp(max=0),
            ],
            param_modifiers=[
                lambda param, _: param.clamp(min=0),
                lambda param, name: param.clamp(max=0) if name != "bias" else torch.zeros_like(param),
                lambda param, _: param.clamp(max=0),
                lambda param, name: param.clamp(min=0) if name != "bias" else torch.zeros_like(param),
            ],
            output_modifiers=[lambda output: output] * 4,
            gradient_mapper=gradient_map,
            reducer=(
                lambda inputs, gradients: (
                    (inputs[0] * gradients[0] + inputs[1] * gradients[1])
                    + (inputs[2] * gradients[2] + inputs[3] * gradients[3])
                )
            ),
        )


class IndexedWaterbirds(Dataset):
    """Waterbirds split returning (x, y, group, index)."""

    def __init__(self, split: str, image_size: int, train_aug: bool, backbone: str):
        self.base = HFWaterbirds(split, image_size=image_size, train_aug=train_aug, backbone=backbone)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, i: int):
        x, y, g, _ = self.base[i]
        return x, int(y), int(g), int(i)


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers: int) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def build_backbone(name: str, num_classes: int = 2):
    if name.lower() == "resnet50":
        m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    raise ValueError(f"Unknown backbone: {name}")


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = build_backbone(args.backbone).to(device)
    ckpt = torch.load(args.erm_checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def evaluate_logits(logits: torch.Tensor, labels: torch.Tensor, groups: torch.Tensor):
    return worst_group_metrics(logits.cpu(), labels.cpu(), groups.cpu())


@torch.no_grad()
def collect_features(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    backbone: str,
    amp: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    feats_all: list[torch.Tensor] = []
    logits_all: list[torch.Tensor] = []
    ys_all: list[torch.Tensor] = []
    gs_all: list[torch.Tensor] = []
    idx_all: list[torch.Tensor] = []
    for x, y, g, idx in tqdm(loader, desc="Features", ncols=100, leave=False):
        x = x.to(device, non_blocking=True)
        with autocast_context(device, enabled=(device.type == "cuda" and amp)):
            logits, feats = forward_with_features(model, backbone, x)
        feats_all.append(feats.detach().float().cpu())
        logits_all.append(logits.detach().float().cpu())
        ys_all.append(y.cpu())
        gs_all.append(g.cpu())
        idx_all.append(idx.cpu())
    return (
        torch.cat(feats_all),
        torch.cat(logits_all),
        torch.cat(ys_all).long(),
        torch.cat(gs_all).long(),
        torch.cat(idx_all).long(),
    )


def normalize_rows(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x = x - x.mean(axis=1, keepdims=True)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(denom, eps)


def eigen_decomposition(affinity_matrix, seed: int, bot_k: int = 12) -> np.ndarray:
    """Eigenvalues of the normalised graph Laplacian (ExMap's eigengap helper)."""
    random_state = np.random.RandomState(seed)
    laplacian, diag = csgraph.laplacian(affinity_matrix, normed=True, return_diag=True)
    laplacian = _set_diag(laplacian, 1, True)
    k = min(bot_k + 1, max(2, laplacian.shape[0] - 1))
    init = random_state.standard_normal(size=(laplacian.shape[0], k)).astype(laplacian.dtype)
    init[:, 0] = diag.ravel()
    eigenvalues, _ = lobpcg(laplacian, init, tol=None, largest=False, maxiter=2000)
    return np.sort(eigenvalues)


def spectral_cluster_np(x: np.ndarray, k: int, args: argparse.Namespace, seed: int) -> np.ndarray:
    n = len(x)
    if k <= 1 or n <= 1:
        return np.zeros(n, dtype=np.int64)
    n_neighbors = max(1, min(n - 1, int(np.log(n))))
    pipeline = SpectralClustering(
        n_clusters=k,
        affinity="nearest_neighbors",
        n_neighbors=n_neighbors,
        n_jobs=-1,
        random_state=seed,
        assign_labels="cluster_qr",
        eigen_solver="lobpcg",
    )
    data_for_fit = x
    if args.auto:
        affinity_matrix = pipeline.fit(x).affinity_matrix_
        eigv = eigen_decomposition(affinity_matrix, seed=seed)
        if len(eigv) > 2:
            upper = min(10, len(eigv) - 1)
            k = int(np.diff(eigv)[1:upper].argmax() + 2) if upper > 1 else 2
            k = max(2, min(k, n))
        pipeline = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            n_neighbors=n_neighbors,
            n_jobs=-1,
            random_state=seed,
            assign_labels="cluster_qr",
            eigen_solver="lobpcg",
        )
        data_for_fit = affinity_matrix
    return pipeline.fit_predict(data_for_fit).astype(np.int64)


def cluster_diagnostics(clusters: np.ndarray, true_groups: np.ndarray) -> dict[str, object]:
    rows: dict[str, object] = {}
    majority = 0
    for c in sorted(np.unique(clusters).tolist()):
        mask = clusters == c
        counts = np.bincount(true_groups[mask].astype(np.int64), minlength=4)
        majority += int(counts.max())
        rows[str(int(c))] = counts.astype(int).tolist()
    rows["majority_alignment"] = float(majority / max(1, len(clusters)))
    return rows


def cluster_pseudo_groups(
    heatmaps: np.ndarray,
    labels: np.ndarray,
    true_groups: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, object]]:
    """Per-class clustering of the heatmaps; pseudo-group ids are offset so that they are unique across classes."""
    heatmaps = normalize_rows(heatmaps)
    pseudo = np.zeros(len(labels), dtype=np.int64)
    offset = 0
    cluster_info: dict[str, object] = {"type": "local", "classes": {}}
    for y_value, k in [(0, args.clusters_class_0), (1, args.clusters_class_1)]:
        mask = labels == y_value
        cl = spectral_cluster_np(heatmaps[mask], k=k, args=args, seed=args.seed + 17 * (y_value + 1))
        pseudo[mask] = cl + offset
        cluster_info["classes"][str(y_value)] = cluster_diagnostics(cl, true_groups[mask])
        offset += int(cl.max()) + 1
    return pseudo, cluster_info


def compute_lrp_heatmaps(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if device.type != "cuda":
        raise SystemExit("ExMap LRP heatmaps require a CUDA device.")

    model.eval()
    heatmaps: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    preds: list[np.ndarray] = []
    idxs: list[np.ndarray] = []

    pbar = tqdm(loader, desc="ExMap LRP heatmaps", ncols=100, leave=False)
    for x, y, g, idx in pbar:
        x = x.to(device, non_blocking=True).float()
        with torch.no_grad():
            out = model(x)
        pred = out.argmax(dim=1)
        torch.cuda.empty_cache()

        canonizer = zennit.torchvision.ResNetCanonizer()
        composite = zennit.composites.EpsilonAlpha2Beta1(canonizers=[canonizer])
        layer_map = []
        for layer_type, rule in composite.layer_map:
            if layer_type == zennit.types.Convolution:
                layer_map.append((layer_type, GammaSymmetric(gamma=4)))
            else:
                layer_map.append((layer_type, rule))
        composite.layer_map = layer_map

        target = torch.eye(2, device=device)[pred]
        with zennit.attribution.Gradient(model=model, composite=composite) as attributor:
            _, relevance = attributor(x, target)
            relevance.detach_()

        if args.downsize > 0:
            relevance = torch.nn.functional.interpolate(
                relevance,
                size=(args.downsize, args.downsize),
                mode="bilinear",
                align_corners=False,
            )
        heatmaps.append(relevance.reshape(relevance.shape[0], -1).detach().cpu().numpy())
        ys.append(y.numpy())
        gs.append(g.numpy())
        preds.append(pred.detach().cpu().numpy())
        idxs.append(idx.numpy())
        del x, out, pred, target, relevance

    return (
        np.concatenate(heatmaps, axis=0),
        np.concatenate(ys, axis=0).astype(np.int64),
        np.concatenate(gs, axis=0).astype(np.int64),
        np.concatenate(preds, axis=0).astype(np.int64),
        np.concatenate(idxs, axis=0).astype(np.int64),
    )


C_OPTIONS = [1.0, 0.7, 0.3, 0.1, 0.07, 0.03, 0.01]


def fit_logreg(x_train: np.ndarray, y_train: np.ndarray, c_value: float, seed: int) -> LogisticRegression:
    clf = LogisticRegression(penalty="l1", C=c_value, solver="liblinear", random_state=seed, max_iter=1000)
    clf.fit(x_train, y_train)
    return clf


def pseudo_group_acc(preds: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    accs = []
    for group in sorted(np.unique(groups).tolist()):
        mask = groups == group
        if mask.any():
            accs.append(float((preds[mask] == y[mask]).mean()))
    return min(accs) if accs else 0.0


def balanced_numpy_indices(groups: np.ndarray, rng: np.random.Generator, min_group_floor: int = 1) -> np.ndarray:
    buckets = [np.where(groups == g)[0] for g in sorted(np.unique(groups).tolist())]
    usable = [b for b in buckets if len(b) >= min_group_floor]
    if not usable:
        raise ValueError("No usable pseudo-groups for DFR balancing")
    min_count = min(len(b) for b in usable)
    chosen = []
    for bucket in usable:
        local = bucket.copy()
        rng.shuffle(local)
        chosen.append(local[:min_count])
    out = np.concatenate(chosen)
    rng.shuffle(out)
    return out.astype(np.int64)


def dfr_sklearn_tune(
    feats: np.ndarray,
    y: np.ndarray,
    pseudo: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, float]]]:
    """Choose C: fit on a pseudo-group-balanced half of validation, score pseudo-group worst accuracy on the other half."""
    rng = np.random.default_rng(args.seed)
    scores = {float(c): 0.0 for c in C_OPTIONS}
    trials: list[dict[str, float]] = []
    for retrain_id in range(args.dfr_tune_retrains):
        idx = np.arange(len(y))
        rng.shuffle(idx)
        half = len(idx) // 2
        tune_idx = idx[:half]
        fit_pool = idx[half:]
        fit_bal = fit_pool[balanced_numpy_indices(pseudo[fit_pool], rng, min_group_floor=1)]

        scaler = StandardScaler()
        x_fit = scaler.fit_transform(feats[fit_bal])
        x_tune = scaler.transform(feats[tune_idx])
        for c_value in C_OPTIONS:
            clf = fit_logreg(x_fit, y[fit_bal], c_value, seed=args.seed + retrain_id)
            preds = clf.predict(x_tune)
            score = pseudo_group_acc(preds, y[tune_idx], pseudo[tune_idx])
            scores[float(c_value)] += score
            trials.append({"c": float(c_value), "trial": float(retrain_id), "tune_pseudo_wga": float(score)})
    best_c = max(scores, key=scores.get)
    return float(best_c), trials


def eval_group_details(preds: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, float, dict[int, tuple[float, int]]]:
    details: dict[int, tuple[float, int]] = {}
    total_correct = 0
    total_count = 0
    for group in sorted(np.unique(groups).tolist()):
        mask = groups == group
        count = int(mask.sum())
        correct = int((preds[mask] == y[mask]).sum())
        acc = float(correct / max(1, count))
        details[int(group)] = (acc, count)
        total_correct += correct
        total_count += count
    avg = float(total_correct / max(1, total_count))
    wga = float(min(acc for acc, _ in details.values())) if details else 0.0
    return avg, wga, details


def dfr_sklearn_eval(
    val_feats: torch.Tensor,
    val_y: torch.Tensor,
    val_g_true: torch.Tensor,
    pseudo_groups: torch.Tensor,
    test_feats: torch.Tensor,
    test_y: torch.Tensor,
    test_g: torch.Tensor,
    args: argparse.Namespace,
) -> tuple[dict[str, object], dict[str, object]]:
    """Fit the averaged L1 logistic-regression head; test group accuracies are averaged over the outer repetitions."""
    feats_val = val_feats.numpy().astype(np.float64)
    y_val = val_y.numpy().astype(np.int64)
    g_val = val_g_true.numpy().astype(np.int64)
    pseudo = pseudo_groups.numpy().astype(np.int64)
    feats_test = test_feats.numpy().astype(np.float64)
    y_test = test_y.numpy().astype(np.int64)
    g_test = test_g.numpy().astype(np.int64)

    best_c, tune_trials = dfr_sklearn_tune(feats_val, y_val, pseudo, args)
    outer_results = []
    test_group_accs = []
    val_group_accs = []
    rng = np.random.default_rng(args.seed + 555)
    for outer in range(args.dfr_outer_repeats):
        scaler = StandardScaler()
        scaler.fit(feats_val)
        coefs = []
        intercepts = []
        for retrain_id in range(args.dfr_eval_retrains):
            for _ in range(20):
                fit_idx = balanced_numpy_indices(pseudo, rng, min_group_floor=1)
                if len(np.unique(y_val[fit_idx])) == len(np.unique(y_val)):
                    break
            x_fit = scaler.transform(feats_val[fit_idx])
            clf = fit_logreg(x_fit, y_val[fit_idx], best_c, seed=args.seed + outer * 1000 + retrain_id)
            coefs.append(clf.coef_.copy())
            intercepts.append(clf.intercept_.copy())

        template = LogisticRegression(penalty="l1", C=best_c, solver="liblinear", random_state=args.seed, max_iter=1000)
        n_classes = len(np.unique(y_val))
        template.fit(scaler.transform(feats_val[:n_classes]), np.arange(n_classes))
        template.coef_ = np.mean(coefs, axis=0)
        template.intercept_ = np.mean(intercepts, axis=0)
        val_preds = template.predict(scaler.transform(feats_val))
        test_preds = template.predict(scaler.transform(feats_test))
        val_avg, val_wga, val_det = eval_group_details(val_preds, y_val, g_val)
        test_avg, test_wga, test_det = eval_group_details(test_preds, y_test, g_test)
        outer_results.append({
            "outer": outer,
            "val_average_accuracy": val_avg,
            "val_worst_group_accuracy": val_wga,
            "test_average_accuracy": test_avg,
            "test_worst_group_accuracy": test_wga,
        })
        val_group_accs.append([val_det[g][0] for g in sorted(val_det)])
        test_group_accs.append([test_det[g][0] for g in sorted(test_det)])

    test_group_acc_arr = np.asarray(test_group_accs, dtype=np.float64)
    val_group_acc_arr = np.asarray(val_group_accs, dtype=np.float64)
    test_counts = {int(g): int((g_test == g).sum()) for g in sorted(np.unique(g_test).tolist())}
    val_counts = {int(g): int((g_val == g).sum()) for g in sorted(np.unique(g_val).tolist())}
    test_groups = {
        int(g): (float(test_group_acc_arr[:, i].mean()), test_counts[int(g)])
        for i, g in enumerate(sorted(test_counts))
    }
    val_groups = {
        int(g): (float(val_group_acc_arr[:, i].mean()), val_counts[int(g)])
        for i, g in enumerate(sorted(val_counts))
    }
    test_avg = float(sum(test_groups[g][0] * test_groups[g][1] for g in test_groups) / max(1, sum(test_counts.values())))
    val_avg = float(sum(val_groups[g][0] * val_groups[g][1] for g in val_groups) / max(1, sum(val_counts.values())))
    info: dict[str, object] = {
        "solver": "sklearn_l1",
        "best_c": best_c,
        "tune_trials": tune_trials,
        "outer_results": outer_results,
        "val_average_accuracy": val_avg,
        "val_worst_group_accuracy": float(min(acc for acc, _ in val_groups.values())),
        "val_groups": val_groups,
        "test_average_accuracy": test_avg,
        "test_worst_group_accuracy": float(min(acc for acc, _ in test_groups.values())),
        "test_groups": test_groups,
    }
    head_info = {"kind": "sklearn_l1_averaged", "best_c": best_c}
    return head_info, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="resnet50", choices=["resnet50"])
    ap.add_argument("--erm_checkpoint", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--batch_size", type=int, default=128, help="batch size for feature extraction")
    ap.add_argument("--cam_batch_size", type=int, default=16, help="batch size for the LRP heatmaps")
    ap.add_argument("--dfr_tune_retrains", type=int, default=5)
    ap.add_argument("--dfr_eval_retrains", type=int, default=20)
    ap.add_argument("--dfr_outer_repeats", type=int, default=5)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 8))
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--downsize", type=int, default=112, help="heatmap side length before clustering")
    ap.add_argument("--auto", action="store_true", default=True, help="choose the number of clusters by the eigengap")
    ap.add_argument("--no-auto", dest="auto", action="store_false")
    ap.add_argument("--clusters_class_0", type=int, default=2, help="clusters for class 0 when --no-auto")
    ap.add_argument("--clusters_class_1", type=int, default=2, help="clusters for class 1 when --no-auto")
    ap.add_argument("--seed", type=int, default=1)
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
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()

    model = load_model(args, device)
    val_ds = IndexedWaterbirds("validation", image_size=args.image_size, train_aug=False, backbone=args.backbone)
    test_ds = IndexedWaterbirds("test", image_size=args.image_size, train_aug=False, backbone=args.backbone)
    val_explain_loader = make_loader(val_ds, args.cam_batch_size, False, args.num_workers)
    val_loader = make_loader(val_ds, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_ds, args.batch_size, False, args.num_workers)

    heatmaps, y_exp, g_exp, pred_exp, idx_exp = compute_lrp_heatmaps(model, val_explain_loader, device, args)
    pseudo_exp, cluster_info = cluster_pseudo_groups(heatmaps, y_exp, g_exp, args)

    order = np.argsort(idx_exp)
    pseudo_val = torch.from_numpy(pseudo_exp[order]).long()
    y_exp = y_exp[order]
    g_exp = g_exp[order]
    pred_exp = pred_exp[order]
    if not np.array_equal(np.arange(len(val_ds)), idx_exp[order]):
        raise RuntimeError("Validation pseudo-label indices are not contiguous after sorting")

    val_feats, base_val_logits, val_y, val_g, _ = collect_features(model, val_loader, device, args.backbone, args.amp)
    test_feats, base_test_logits, test_y, test_g, _ = collect_features(model, test_loader, device, args.backbone, args.amp)
    base_val_avg, base_val_wga, base_val_det = evaluate_logits(base_val_logits, val_y, val_g)
    base_test_avg, base_test_wga, base_test_det = evaluate_logits(base_test_logits, test_y, test_g)

    head_info, dfr_info = dfr_sklearn_eval(val_feats, val_y, val_g, pseudo_val, test_feats, test_y, test_g, args)
    torch.save(
        {
            "base_checkpoint": str(Path(args.erm_checkpoint).resolve()),
            "dfr_head": head_info,
            "args": vars(args),
            "cluster_info": cluster_info,
            "pseudo_groups": pseudo_val.cpu(),
        },
        ckpt_dir / "exmap_best.pt",
    )

    pseudo_counts = np.bincount(pseudo_val.numpy()).astype(int).tolist()
    summary = {
        "output_dir": str(output_dir),
        "best_val_wga": dfr_info["val_worst_group_accuracy"],
        "base_erm_checkpoint": str(Path(args.erm_checkpoint).resolve()),
        "base_erm": {
            "val_average_accuracy": base_val_avg,
            "val_worst_group_accuracy": base_val_wga,
            "val_groups": base_val_det,
            "test_average_accuracy": base_test_avg,
            "test_worst_group_accuracy": base_test_wga,
            "test_groups": base_test_det,
        },
        "exmap": {
            "pseudo_group_type": "local",
            "pseudo_group_counts": pseudo_counts,
            "cluster_info": cluster_info,
            "explanation_accuracy": float((pred_exp == y_exp).mean()),
        },
        "best_checkpoint": {
            "path": str((ckpt_dir / "exmap_best.pt").resolve()),
            "val_average_accuracy": dfr_info["val_average_accuracy"],
            "val_worst_group_accuracy": dfr_info["val_worst_group_accuracy"],
            "val_groups": dfr_info["val_groups"],
            "test_average_accuracy": dfr_info["test_average_accuracy"],
            "test_worst_group_accuracy": dfr_info["test_worst_group_accuracy"],
            "test_groups": dfr_info["test_groups"],
        },
        "dfr": dfr_info,
        "args": vars(args),
        "elapsed_minutes": (time.time() - start) / 60.0,
    }
    with (output_dir / "train_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary["best_checkpoint"], indent=2))
    print(f"[saved] summary -> {output_dir / 'train_summary.json'}")


if __name__ == "__main__":
    main()
