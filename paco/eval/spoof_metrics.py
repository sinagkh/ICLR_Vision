#!/usr/bin/env python3
"""Spoof metrics from the saved score files.

For NAME = <arm>_s<seed>, reads $OUT_ROOT/logits/NAME.npz (save_logits.py) and $OUT_ROOT/spoof_scores/<rendering>/NAME.npz
(spoof_test.py, spoof_renderings.py). Per class, the threshold tau is the 5% quantile of the true-class margin of clean
test positives (95% of genuine test objects are accepted). Reported per checkpoint:
  single : single-part spoof FPR95, the mean acceptance (margin >= tau) of +A and +B spoofs, macro over classes (%)
  none / A / B / AB / R : acceptance in each paste condition (%)
  auroc  : AUROC between clean test positives and single-part (+A, +B) spoofs by the same margin, macro over classes
  top1   : test top-1 accuracy (%)
  d2     : mean test Delta^2 of the true-class margin over the part-removal corners (mean-colour fill)
Writes $OUT_ROOT/spoof_metrics.json and prints seed means per rendering."""
import glob, json, os, sys
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from paco.src.common import LISTS, read_rows  # noqa: E402
OUT = Path(os.environ.get('OUT_ROOT', '.')); LG = OUT / 'logits'; SC = OUT / 'spoof_scores'
ytest = np.array([int(r['class_index']) for r in read_rows(LISTS / 'test_instances.csv')])
def margin(z, y):
    zc = z[np.arange(len(y)), y].copy(); z2 = z.copy(); z2[np.arange(len(y)), y] = -np.inf
    return zc - z2.max(1)
def score(name, sdir):
    f = sdir / f'{name}.npz'; lg = LG / f'{name}.npz'
    if not f.exists() or not lg.exists(): return None
    d = np.load(f); s, y = d['s'], d['y']; L = np.load(lg); st = margin(L['z_test'], ytest)
    acc = {k: [] for k in ['none', 'A', 'B', 'AB', 'R']}; au = []
    for c in range(5):
        tau = float(np.quantile(st[ytest == c], 0.05)); m = y == c; sc = s[m]
        for i, k in enumerate(['none', 'A', 'B', 'AB']): acc[k].append(float((sc[:, i] >= tau).mean()))
        hasR = d['r_area_ratio'][m] > 0.5
        acc['R'].append(float((sc[hasR, 4] >= tau).mean()) if hasR.any() else np.nan)
        pos, neg = st[ytest == c], np.concatenate([sc[:, 1], sc[:, 2]])
        au.append(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    r = {k: 100 * float(np.nanmean(v)) for k, v in acc.items()}; r['single'] = (r['A'] + r['B']) / 2; r['auroc'] = float(np.mean(au))
    m_ = lambda k: margin(L[k], ytest)
    r['top1'] = 100 * float((L['z_test'].argmax(1) == ytest).mean()); r['d2'] = float((m_('z_test') - m_('z_test_x10') - m_('z_test_x01') + m_('z_test_x00')).mean())
    return r
REND = {'original': SC / 'original', 'blend': SC / 'blend', 'shift': SC / 'shift'}
ARMS = ['erm', 'ce_only', 'partdrop_ft', 'partdrop_is', 'oe_ft', 'oe_is', 'notarget', 'redund', 'randmask', 'is', 'K0p25', 'K1', 'K2', 'gHalf', 'gDouble']
R = {rn: {a: {s: score(f'{a}_s{s}', sd) for s in (0, 1, 2)} for a in ARMS} for rn, sd in REND.items()}
R = {rn: {a: {s: v for s, v in d.items() if v is not None} for a, d in ra.items()} for rn, ra in R.items()}
(OUT / 'spoof_metrics.json').write_text(json.dumps(R, indent=1))
for rn, ra in R.items():
    print(f'\n== rendering: {rn}')
    print(f"{'arm':18s} {'single':>7s} {'+AB':>6s} {'none':>6s} {'+R':>6s} {'AUROC':>6s} {'top1':>6s} {'testD2':>7s} seeds")
    for a, d in ra.items():
        if not d: continue
        mm = lambda k: np.mean([v[k] for v in d.values()])
        print(f"{a:18s} {mm('single'):7.1f} {mm('AB'):6.1f} {mm('none'):6.1f} {mm('R'):6.1f} {mm('auroc'):6.3f} {mm('top1'):6.1f} {mm('d2'):+7.2f} {len(d)}")
