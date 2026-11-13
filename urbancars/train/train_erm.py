from __future__ import annotations

import argparse
import json
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import (
    DEFAULT_DATA_ROOT,
    DEFAULT_OUTPUT_ROOT,
    UrbanCarsClassificationDataset,
    build_classifier,
    evaluate_group_metrics,
    get_transforms,
    ratio_folder_for_split,
    serialize_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an ERM baseline on UrbanCars.")
    parser.add_argument("--data_root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output_dir", type=Path)
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V1")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument(
        "--early_stop_metric",
        type=str,
        choices=["bg", "co_occur_obj", "both"],
        default="both",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def choose_output_dir(requested: Path | None) -> Path:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        return requested
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_dir = DEFAULT_OUTPUT_ROOT / f"erm_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path: Path, payload: dict) -> None:
    torch.save(payload, path)


def main() -> None:
    args = parse_args()
    start_time = time.time()
    output_dir = choose_output_dir(args.output_dir)

    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    amp_enabled = args.amp and device.type == "cuda"

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"

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
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    history_path = output_dir / "history.jsonl"
    metric_name = f"val_{args.early_stop_metric}_worst_group_acc"
    best_metric = float("-inf")
    best_epoch = 0
    best_val_metrics: dict[str, float] = {}
    best_test_metrics: dict[str, float] = {}
    start_epoch = 1

    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scaler.load_state_dict(state["scaler_state_dict"])
        best_metric = float(state.get("best_metric_value", best_metric))
        best_epoch = int(state.get("best_epoch", best_epoch))
        best_val_metrics = dict(state.get("best_val_metrics", {}))
        best_test_metrics = dict(state.get("best_test_metrics", {}))
        start_epoch = int(state["epoch"]) + 1

    config = {
        **{
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        },
        "data_root": str(args.data_root),
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

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_examples = 0
        progress = tqdm(train_loader, dynamic_ncols=True, desc=f"epoch {epoch}/{args.epochs}")
        for batch in progress:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"][:, 0].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logits = model(images)
                loss = F.cross_entropy(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            batch_size = images.size(0)
            running_examples += batch_size
            running_loss += loss.item() * batch_size
            progress.set_postfix(loss=f"{running_loss / max(running_examples, 1):.4f}")

        train_loss = running_loss / max(running_examples, 1)
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

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **val_metrics,
            **test_metrics,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        metric_value = float(val_metrics[metric_name])
        last_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "best_metric_name": metric_name,
            "best_metric_value": best_metric,
            "best_epoch": best_epoch,
            "best_val_metrics": best_val_metrics,
            "best_test_metrics": best_test_metrics,
        }
        save_checkpoint(output_dir / "last.pth", last_state)

        if args.save_every > 0 and epoch % args.save_every == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", last_state)

        if metric_value > best_metric:
            best_metric = metric_value
            best_epoch = epoch
            best_val_metrics = dict(val_metrics)
            best_test_metrics = dict(test_metrics)
            best_state = dict(last_state)
            best_state["best_metric_value"] = best_metric
            best_state["best_epoch"] = best_epoch
            best_state["best_val_metrics"] = best_val_metrics
            best_state["best_test_metrics"] = best_test_metrics
            save_checkpoint(output_dir / "best.pth", best_state)

        print(
            f"[epoch {epoch:03d}] "
            f"loss={train_loss:.4f} "
            f"val_both_wga={val_metrics['val_both_worst_group_acc'] * 100:.2f} "
            f"test_both_wga={test_metrics['test_both_worst_group_acc'] * 100:.2f} "
            f"test_id_acc={test_metrics['test_id_acc'] * 100:.2f}"
        )

    summary = {
        "config": config,
        "best_metric_name": metric_name,
        "best_metric_value": best_metric,
        "best_epoch": best_epoch,
        "best_val_metrics": best_val_metrics,
        "best_test_metrics": best_test_metrics,
        "artifacts": {
            "best_checkpoint": str(output_dir / "best.pth"),
            "last_checkpoint": str(output_dir / "last.pth"),
            "history": str(history_path),
        },
        "finished_at_utc": datetime.utcnow().isoformat() + "Z",
        "elapsed_seconds": time.time() - start_time,
    }
    serialize_json(summary, output_dir / "train_summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
