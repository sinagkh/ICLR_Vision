#!/usr/bin/env python3
"""IID and OOD test accuracy (%) of a trained Colored MNIST model, written to a JSON file.

Both test sets are built from the 10,000 MNIST test images with the same seed, so they share the labels and the
label noise; the label-color agreement is ρ = --train_corr in the IID set and ρ = --ood_corr in the OOD set.
The dataset is read from $DATA_ROOT.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dataset import ColoredMNIST
from model import SmallCNN


@torch.no_grad()
def accuracy(net, dataset, device):
    correct = 0
    for x, y, _ in DataLoader(dataset, batch_size=256):
        correct += (net(x.to(device)).argmax(1).cpu() == y).sum().item()
    return 100 * correct / len(dataset)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", type=Path, required=True, help="model weights written by train.py")
    p.add_argument("--train_corr", type=float, default=0.95, help="ρ of the IID test set")
    p.add_argument("--ood_corr", type=float, default=0.05, help="ρ of the OOD test set")
    p.add_argument("--test_seed", type=int, default=8, help="seed of the test-set label noise and colors")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True, help="JSON file for the accuracies")
    args = p.parse_args()

    net = SmallCNN().to(args.device)
    net.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    net.eval()

    root = os.environ["DATA_ROOT"]
    iid = ColoredMNIST(root, train=False, corr=args.train_corr, seed=args.test_seed)
    ood = ColoredMNIST(root, train=False, corr=args.ood_corr, seed=args.test_seed)
    result = {
        "iid_acc": accuracy(net, iid, args.device),
        "ood_acc": accuracy(net, ood, args.device),
        "n_test": len(iid),
        "train_corr": args.train_corr,
        "ood_corr": args.ood_corr,
        "test_seed": args.test_seed,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{args.ckpt.name}: IID {result['iid_acc']:.2f}  OOD {result['ood_acc']:.2f}")


if __name__ == "__main__":
    main()
