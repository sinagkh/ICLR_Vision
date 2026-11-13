#!/usr/bin/env python3
"""Write results/paco/mae_vit.csv from the spoof metrics of the MAE ViT-B/16 study ($OUT_ROOT/spoof_metrics.json,
written by eval/spoof_metrics.py with OUT_ROOT set to the study's output folder; see configs/mae_vit.sh).
usage: make_mae_table.py OUT_DIR"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('OUT_ROOT', '.'))
ARMS = [('ERM', 'erm'), ('Plain fine-tuning', 'ce_only'), ('IS without the synergy target', 'notarget'), ('IS', 'is')]
SEEDS = ('0', '1', '2')


def main(dst):
    M = json.loads((OUT / 'spoof_metrics.json').read_text())['original']
    rows = []
    for name, arm in ARMS:
        vals = {s: [M[arm][s]['top1'], M[arm][s]['single'], M[arm][s]['auroc']] for s in SEEDS}
        rows += [[name, arm, s] + [round(v, d) for v, d in zip(vals[s], (3, 3, 6))] for s in SEEDS]
        rows.append([name, arm, 'mean'] + [round(float(v), d) for v, d in zip(np.mean([vals[s] for s in SEEDS], axis=0), (3, 3, 6))])
    dst = Path(dst); dst.mkdir(parents=True, exist_ok=True)
    with open(dst / 'mae_vit.csv', 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n'); w.writerow(['method', 'arm', 'seed', 'top1_pct', 'spoof_fpr95_pct', 'auroc']); w.writerows(rows)
    print('wrote', dst / 'mae_vit.csv')


if __name__ == '__main__':
    main(sys.argv[1])
