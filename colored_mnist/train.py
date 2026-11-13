#!/usr/bin/env python3
"""Train the Colored MNIST classifier with cross-entropy plus λ times the first-order color penalty.

The color intervention T swaps the red and green channels, which recolors both the digit and the background.
The first-order color effect on the true-class logit is Δ¹(x) = f_y(T(x)) − f_y(x); the penalty steers it toward
zero with d² / (|d| + 0.01), a smooth version of |d|. λ = 0 is ERM: the penalty is still computed and printed
but does not enter the gradient.

Training uses a seeded random 90% of the MNIST training images. The dataset is read from $DATA_ROOT.
"""
import argparse
import os
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from dataset import ColoredMNIST
from model import SmallCNN

SMOOTH = 1e-2


def swap_red_green(x):
    return x[:, [1, 0, 2]]


def color_effect_penalty(net, x, y):
    idx = torch.arange(len(y), device=x.device)
    f_orig = net(x)[idx, y]
    f_swap = net(swap_red_green(x))[idx, y]
    d = f_swap - f_orig
    return torch.mean(d.abs() ** 2 / (d.abs() + SMOOTH))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lam", type=float, required=True, help="weight λ of the first-order color penalty (0 = ERM)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate")
    p.add_argument("--train_corr", type=float, default=0.95, help="label-color agreement ρ in training")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, required=True, help="where to save the model weights (.pt)")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    full = ColoredMNIST(os.environ["DATA_ROOT"], train=True, corr=args.train_corr, seed=args.seed)
    n_train = int(0.9 * len(full))
    train_set, _ = random_split(full, [n_train, len(full) - n_train],
                                generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)

    net = SmallCNN().to(args.device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    ce = nn.CrossEntropyLoss()

    for ep in range(1, args.epochs + 1):
        net.train()
        sum_ce, sum_pen = 0.0, 0.0
        for x, y, _ in loader:
            x, y = x.to(args.device), y.to(args.device)
            loss_ce = ce(net(x), y)
            loss_pen = color_effect_penalty(net, x, y)
            loss = loss_ce + args.lam * loss_pen
            opt.zero_grad()
            loss.backward()
            opt.step()
            sum_ce += loss_ce.item() * x.size(0)
            sum_pen += loss_pen.item() * x.size(0)
        print(f"epoch {ep}/{args.epochs}  CE {sum_ce / len(train_set):.4f}  "
              f"color penalty {sum_pen / len(train_set):.4f}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
