#!/usr/bin/env python3
"""Write table9.csv (paper Table 9, Colored MNIST) from the evaluation files $OUT_ROOT/colored_mnist/lam<λ>_seed7.json
written by evaluate.py. Every row is a single run with seed 7.
usage: make_table.py OUT_DIR"""
import csv
import json
import os
import sys
from pathlib import Path

SEED = 7
ROWS = [("ERM", "0"), ("Steered", "0.5"), ("Steered", "0.7"), ("Steered", "0.9")]


def main(dst):
    src = Path(os.environ["OUT_ROOT"]) / "colored_mnist"
    rows = []
    for method, lam in ROWS:
        r = json.loads((src / f"lam{lam}_seed{SEED}.json").read_text())
        rows.append([method, f"{float(lam):.1f}", SEED, f"{r['iid_acc']:.2f}", f"{r['ood_acc']:.2f}"])
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    with open(dst / "table9.csv", "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["method", "lambda", "seed", "iid_acc", "ood_acc"])
        w.writerows(rows)
    print("wrote", dst / "table9.csv")


if __name__ == "__main__":
    main(sys.argv[1])
