#!/usr/bin/env python3
"""Write the PACO result tables (CSV, per seed and seed mean).

Inputs, all under $OUT_ROOT:
  spoof_metrics.json         from eval/spoof_metrics.py
  hidden_part.json           from eval/hidden_part_test.py
  runs/<tag>_s<seed>/metrics_epoch.jsonl   training logs (validation top-1 at the final epoch)
usage: make_tables.py OUT_DIR
"""
import csv, json, os, sys
from pathlib import Path
import numpy as np

OUT = Path(os.environ.get('OUT_ROOT', '.'))
SEEDS = ('0', '1', '2')
MAIN = [('Plain fine-tuning', 'ce_only'), ('PartDrop', 'partdrop_ft'), ('PartDrop + IS', 'partdrop_is'), ('Outlier exposure', 'oe_ft'),
        ('Outlier exposure + IS', 'oe_is'), ('IS without the synergy target', 'notarget'), ('IS with the sign reversed', 'redund'),
        ('IS with synergy on random regions', 'randmask'), ('IS', 'is')]
RENDER = [('Plain fine-tuning', 'ce_only'), ('IS without the synergy target', 'notarget'), ('IS', 'is')]
SENS = [('K=0.25', 'K0p25', '--K 0.25'), ('K=0.5 (reported IS)', 'is', ''), ('K=1', 'K1', '--K 1.0'), ('K=2', 'K2', '--K 2.0'),
        ('guard weights x0.5', 'gHalf', '--lambda-order 0.01 --lambda-wrong 0.005 --lambda-stab 0.0025'),
        ('guard weights x2', 'gDouble', '--lambda-order 0.04 --lambda-wrong 0.02 --lambda-stab 0.01')]
K_PICK = [('0.25', 'K0p25'), ('0.5', 'is'), ('1', 'K1'), ('2', 'K2')]
HIDDEN = [('Plain fine-tuning', 'Plain fine-tuning'), ('IS without the synergy target', 'IS without the synergy target'), ('IS', 'IS'),
          ('PartDrop', 'PartDrop'), ('PartDrop + IS', 'PartDrop + IS')]


def write(path, header, rows):
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh, lineterminator='\n')
        w.writerow(header)
        w.writerows(rows)
    print('wrote', path)


def with_mean(prefix, per_seed, digits):
    """per_seed: {seed: [values]} -> rows prefix + [seed] + values, then a 'mean' row over seeds."""
    rows = [prefix + [s] + [round(v, d) for v, d in zip(per_seed[s], digits)] for s in SEEDS]
    mean = np.mean([per_seed[s] for s in SEEDS], axis=0)
    rows.append(prefix + ['mean'] + [round(float(v), d) for v, d in zip(mean, digits)])
    return rows


def val_top1(tag, seed):
    hist = [json.loads(l) for l in open(OUT / 'runs' / f'{tag}_s{seed}' / 'metrics_epoch.jsonl')]
    return 100 * hist[-1]['val_top1']


def main():
    dst = Path(sys.argv[1])
    dst.mkdir(parents=True, exist_ok=True)
    S = json.loads((OUT / 'spoof_metrics.json').read_text())
    H = json.loads((OUT / 'hidden_part.json').read_text())['rows']
    orig = S['original']

    rows = []
    for name, arm in MAIN:
        rows += with_mean([name, arm], {s: [orig[arm][s]['top1'], orig[arm][s]['single'], orig[arm][s]['auroc']] for s in SEEDS}, (3, 3, 6))
    write(dst / 'main_table.csv', ['method', 'arm', 'seed', 'top1_pct', 'spoof_fpr95_pct', 'auroc'], rows)

    rows = []
    for name, arm in RENDER:
        rows += with_mean([name, arm], {s: [S[r][arm][s]['single'] for r in ('original', 'blend', 'shift')] for s in SEEDS}, (3, 3, 3))
    write(dst / 'spoof_renderings.csv', ['method', 'arm', 'seed', 'fpr95_original_pct', 'fpr95_harmonized_feathered_pct', 'fpr95_shifted_pct'], rows)

    rows = []
    for setting, tag, flags in SENS:
        rows += with_mean([setting, flags], {s: [orig[tag][s]['top1'], orig[tag][s]['single'], val_top1(tag, s)] for s in SEEDS}, (3, 3, 3))
    write(dst / 'sensitivity.csv', ['setting', 'train_flags_beyond_arm_is', 'seed', 'top1_pct', 'spoof_fpr95_pct', 'val_top1_pct'], rows)

    rows, picked = [], []
    for s in SEEDS:
        vals = [(val_top1(tag, s), k, tag) for k, tag in K_PICK]
        best = max(v for v, _, _ in vals)
        v, k, tag = next(x for x in vals if x[0] == best)   # first K (in increasing order) with the best validation top-1
        picked.append(orig[tag][s]['single'])
        rows.append([s, k, round(v, 3), round(orig[tag][s]['single'], 3)])
    rows.append(['mean', '', '', round(float(np.mean(picked)), 3)])
    write(dst / 'sensitivity_k_validation_pick.csv', ['seed', 'picked_K', 'val_top1_pct', 'spoof_fpr95_pct'], rows)

    rows = []
    for name, key in HIDDEN:
        arm = dict(MAIN)[name]
        rows += with_mean([name, arm], {s: [H[f'{key}|s{s}']['recognized_at_95']] for s in SEEDS}, (3,))
    write(dst / 'hidden_part.csv', ['method', 'arm', 'seed', 'recognized_at_95_threshold_pct'], rows)


if __name__ == '__main__':
    main()
