#!/usr/bin/env python3
"""Hidden-part test: recognition of held-out objects of the trained classes on which one named part group is not
annotated (lists/part_absent.csv; 1,543 objects from PACO-LVIS val and test images).
  recognized_at_95: share of these objects whose class margin reaches the class's 95% threshold, i.e. the 5th percentile of the
      margin of clean test positives (the same threshold as the spoof test), macro over classes (%)
  top1: top-1 accuracy on these objects, macro over classes (%)
Reads $OUT_ROOT/logits/<arm>_s<seed>.npz (save_logits.py). Usage: hidden_part_test.py OUT_JSON"""
import csv, json, os, sys
from pathlib import Path
import numpy as np
LISTS = Path(__file__).resolve().parents[1] / 'lists'; LG = Path(os.environ.get('OUT_ROOT', '.')) / 'logits'
NAMES = ["bicycle", "chair", "laptop_computer", "car_(automobile)", "bottle"]
y_test = np.array([int(r['class_index']) for r in csv.DictReader(open(LISTS / 'test_instances.csv'))])
y_abs = np.array([int(r['class_index']) for r in csv.DictReader(open(LISTS / 'part_absent.csv'))])
def margins(z):
    s = np.sort(z, 1); top, sec = s[:, -1:], s[:, -2:-1]
    return z - np.where(z >= top, sec, top)
ARMS = [('Plain fine-tuning', 'ce_only'), ('IS without the synergy target', 'notarget'), ('IS', 'is'),
        ('PartDrop', 'partdrop_ft'), ('PartDrop + IS', 'partdrop_is'), ('Outlier exposure', 'oe_ft'), ('Outlier exposure + IS', 'oe_is'),
        ('IS with the sign reversed', 'redund'), ('IS with synergy on random regions', 'randmask')]
R = {}
for name, arm in ARMS:
    for s in (0, 1, 2):
        d = np.load(LG / f'{arm}_s{s}.npz'); zt, za = d['z_test'].astype(np.float64), d['z_part_absent'].astype(np.float64)
        mt, ma = margins(zt), margins(za)
        m1 = {NAMES[k]: float((ma[y_abs == k, k] >= np.percentile(mt[y_test == k, k], 5)).mean()) for k in range(5)}
        m2 = {NAMES[k]: float((za[y_abs == k].argmax(1) == k).mean()) for k in range(5)}
        R[(name, s)] = dict(recognized_at_95=100 * np.mean(list(m1.values())), top1=100 * np.mean(list(m2.values())), test_top1=100 * float((zt.argmax(1) == y_test).mean()),
                            recognized_at_95_per_class={k: 100 * v for k, v in m1.items()}, top1_per_class={k: 100 * v for k, v in m2.items()})
print(f"{'method':34s} {'recog@95':>8s} {'(s0 / s1 / s2)':>22s} {'top-1':>8s} {'(s0 / s1 / s2)':>22s} {'test top1':>9s}")
for name, _ in ARMS:
    r = [R[(name, s)] for s in (0, 1, 2)]
    f = lambda k: f"{np.mean([x[k] for x in r]):8.1f} ({' / '.join(f'{x[k]:.1f}' for x in r):>20s})"
    print(f"{name:34s} {f('recognized_at_95')} {f('top1')} {np.mean([x['test_top1'] for x in r]):9.1f}")
print('\nrecognized at the 95% threshold, per class (3-seed mean)')
for name, _ in ARMS:
    print(f"{name:34s} " + ' '.join(f"{c[:7]} {np.mean([R[(name, s)]['recognized_at_95_per_class'][c] for s in (0, 1, 2)]):5.1f}" for c in NAMES))
json.dump({'rows': {f'{k[0]}|s{k[1]}': v for k, v in R.items()}}, open(sys.argv[1], 'w'), indent=1)
