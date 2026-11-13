#!/usr/bin/env python3
"""Write unseen_scenes.csv and script_reported_test_accuracy.csv of results/waterbirds/.

  unseen_scenes.csv                  Corrupted BG-Gap on unseen Places365 scenes ($OUT_ROOT/unseen_scenes, 14_evaluation.sh)
  script_reported_test_accuracy.csv  test WGA and accuracy from the Full CF-Aug trainer
                                     ($CKPT_ROOT/full_cfaug/seed<s>/cfaug_mixed/train_summary.json) and from the official
                                     DFR evaluation ($OUT_ROOT/dfr/seed<s>/{erm,full_is}.json, 15_dfr.sh)
usage: write_tables.py OUT_DIR"""
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

OUT_ROOT = Path(os.environ['OUT_ROOT']); CKPT_ROOT = Path(os.environ['CKPT_ROOT'])
SEEDS = (1, 21, 42)


def write(path, header, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n'); w.writerow(header); w.writerows(rows)
    print('wrote', path)


def main(out_dir):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for name, key in (('ERM', 'erm'), ('IS', 'is'), ('CF-Aug (last 20 epochs; stand-alone trainer)', 'cfaug_20ep_plain_trainer')):
        vals = [100 * json.load(open(OUT_ROOT / 'unseen_scenes' / f'results_s{s}.json'))[key]['corrupted_bg_gap'] for s in SEEDS]
        rows += [[name, s, f'{v:.4f}'] for s, v in zip(SEEDS, vals)] + [[name, 'mean', f'{np.mean(vals):.4f}']]
    write(out_dir / 'unseen_scenes.csv', ['model', 'seed', 'corrupted_bg_gap'], rows)

    rows = {}
    for s in SEEDS:
        d = json.load(open(CKPT_ROOT / 'full_cfaug' / f'seed{s}' / 'cfaug_mixed' / 'train_summary.json'))
        rows.setdefault('Full CF-Aug', []).append((s, 'train/train_cfaug_full.py test evaluation', 100 * d['best_test_wga'], 100 * d['best_test_avg']))
        for base, name in (('erm', 'ERM'), ('full_is', 'IS')):
            d = json.load(open(OUT_ROOT / 'dfr' / f'seed{s}' / f'{base}.json'))
            for part, kind, how in (('dfr_val', 'Val', 'validation'), ('dfr_train', 'Train', 'training subsets')):
                rows.setdefault(f'{name} + DFR ({kind})', []).append(
                    (s, f'eval/dfr_official.py (DFR on {how})', 100 * d[part]['test_wga'], 100 * d[part]['test_mean_acc']))
    order = ['Full CF-Aug', 'ERM + DFR (Val)', 'IS + DFR (Val)', 'ERM + DFR (Train)', 'IS + DFR (Train)']
    out = [[m, s, how, f'{w:.4f}', f'{a:.4f}'] for m in order for s, how, w, a in rows[m]]
    out += [[m, 'mean', rows[m][0][1], f'{np.mean([r[2] for r in rows[m]]):.4f}', f'{np.mean([r[3] for r in rows[m]]):.4f}'] for m in order]
    write(out_dir / 'script_reported_test_accuracy.csv', ['model', 'seed', 'evaluated_by', 'wga', 'test_acc'], out)


if __name__ == '__main__':
    main(sys.argv[1])
