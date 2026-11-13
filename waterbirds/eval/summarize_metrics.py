#!/usr/bin/env python3
"""Turn eval_robustness.py (and guard_analysis.py) outputs into the CSV files of results/waterbirds/.

usage: summarize_metrics.py OUT_DIR MODEL:SEED=RESULTS_JSON:KEY[=GUARDS_JSON:GUARD_KEY] ...

  MODEL:SEED     row label, e.g. IS:1
  RESULTS_JSON   a results_*.json written by eval_robustness.py; KEY is the model name used on its command line
  GUARDS_JSON    optional output of guard_analysis.py for the same checkpoint; GUARD_KEY is the model name used there

Columns (accuracies in %, bg_delta1 in logits):
  wga                  worst-group accuracy on the natural test set
  avg_test_acc         test accuracy = group accuracies weighted by the test group sizes (2255 / 2255 / 642 / 642)
  corrupted_bg_gap     majority-group minus minority-group accuracy (per image) under object-only corruption,
                       mean over blur radius 8, downsample x16 and contrast 0.1
  only_bg_acc          share of birdless probe images (bird removed, or conflicting donor background alone) that are
                       classified as the class of their background
  bg_delta1            background main effect without the bird, m10 - m11 (true-class margins), mean over the test
                       images whose bird is on its own background
  acc_<class>_<place>  natural test accuracy of each group
  bg_gap_<corruption>  the corrupted BG-gap for each of the three corruptions
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

N = {'y0_bg0': 2255, 'y0_bg1': 2255, 'y1_bg0': 642, 'y1_bg1': 642}
STRONG = ('blur_8', 'lowres_16', 'contrast_0.1')
GROUP_NAMES = {'y0_bg0': 'acc_landbird_land', 'y0_bg1': 'acc_landbird_water', 'y1_bg0': 'acc_waterbird_land', 'y1_bg1': 'acc_waterbird_water'}
GAP_NAMES = {'blur_8': 'bg_gap_blur8', 'lowres_16': 'bg_gap_downsample16', 'contrast_0.1': 'bg_gap_contrast0.1'}
COLUMNS = ['wga', 'avg_test_acc', 'corrupted_bg_gap', 'only_bg_acc', 'bg_delta1'] + list(GROUP_NAMES.values()) + list(GAP_NAMES.values())


def metrics(v):
    g = v['none_0']['groups']
    row = {'wga': 100 * v['none_0']['wga'],
           'avg_test_acc': 100 * sum(N[k] * g[k] for k in N) / sum(N.values()),
           'corrupted_bg_gap': float(np.mean([100 * (v[c]['majority'] - v[c]['minority']) for c in STRONG])),
           'only_bg_acc': 100 * v['decomposition']['birdless_follows_background'],
           'bg_delta1': None}
    row.update({GROUP_NAMES[k]: 100 * g[k] for k in N})
    row.update({GAP_NAMES[c]: 100 * (v[c]['majority'] - v[c]['minority']) for c in STRONG})
    return row


def parse(spec):
    label, rest = spec.split('=', 1)
    model, seed = label.rsplit(':', 1)
    parts = rest.split('=')
    res_path, key = parts[0].rsplit(':', 1)
    row = {'model': model, 'seed': int(seed), **metrics(json.load(open(res_path))[key])}
    if len(parts) > 1:
        g_path, g_key = parts[1].rsplit(':', 1)
        row['bg_delta1'] = json.load(open(g_path))[g_key]['e_bg_alone']['mean']
    return row


def fmt(x):
    if x is None:
        return ''
    return f'{x:.4f}' if isinstance(x, float) else str(x)


def write(rows, out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'per_seed.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(['model', 'seed'] + COLUMNS)
        for r in rows:
            w.writerow([r['model'], r['seed']] + [fmt(r[c]) for c in COLUMNS])
    models = list(dict.fromkeys(r['model'] for r in rows))
    with open(out_dir / 'mean_over_seeds.csv', 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['model', 'n_seeds', 'seeds'] + [h for c in COLUMNS for h in (c, c + '_sd')])
        for m in models:
            rr = [r for r in rows if r['model'] == m]
            line = [m, len(rr), ' '.join(str(r['seed']) for r in rr)]
            for c in COLUMNS:
                vals = [r[c] for r in rr]
                if any(x is None for x in vals):
                    line += ['', '']
                else:
                    line += [fmt(float(np.mean(vals))), fmt(float(np.std(vals, ddof=1))) if len(vals) > 1 else '']
            w.writerow(line)


def main(argv):
    out_dir, specs = argv[0], argv[1:]
    rows = [parse(s) for s in specs]
    write(rows, out_dir)
    for m in dict.fromkeys(r['model'] for r in rows):
        rr = [r for r in rows if r['model'] == m]
        print(f"{m:20s} n={len(rr)} " + ' '.join(f"{c} {np.mean([r[c] for r in rr]):6.2f}" for c in COLUMNS[:4]))


if __name__ == '__main__':
    main(sys.argv[1:])
