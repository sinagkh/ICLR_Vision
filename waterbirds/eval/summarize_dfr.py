#!/usr/bin/env python3
"""Summarize eval_robustness.py results for ERM + DFR(Val) and IS + DFR(Val) (3 seeds), with the same scoring as for all
other models: WGA, average test accuracy (group accuracies weighted by the test group sizes), corrupted BG-Gap (mean over
blur 8, lowres 16, contrast 0.1 of majority − minority per-image accuracy) and Only-BG accuracy
(decomposition['birdless_follows_background'])."""
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT as OUT_ROOT  # noqa: E402
E = OUT_ROOT / 'eval'
N = {'y0_bg0': 2255, 'y0_bg1': 2255, 'y1_bg0': 642, 'y1_bg1': 642}
STRONG = ('blur_8', 'lowres_16', 'contrast_0.1')


def row(v):
    g = v['none_0']['groups']
    avg = 100 * sum(N[k] * g[k] for k in N) / sum(N.values())
    gap = float(np.mean([100 * (v[c]['majority'] - v[c]['minority']) for c in STRONG]))
    maj = float(np.mean([100 * v[c]['majority'] for c in STRONG])); mino = float(np.mean([100 * v[c]['minority'] for c in STRONG]))
    return dict(wga=100 * v['none_0']['wga'], avg=avg, gap=gap, aligned=maj, conflicting=mino,
                onlybg=100 * v['decomposition']['birdless_follows_background'], mean_group_deg=float(np.mean([100 * v[c]['mean_group'] for c in STRONG])))


out = {}
for arm in ('dfrerm', 'dfris'):
    rs = []
    for s in (1, 21, 42):
        d = json.load(open(E / f'results_s{s}_dfrerm_s{s}_dfris.json'))
        rs.append(row(d[f's{s}_{arm}']))
    out[arm] = rs
    keys = rs[0].keys()
    print(f"{arm:7s} " + '  '.join(f"{k} {np.mean([r[k] for r in rs]):6.2f} ({' / '.join(f'{r[k]:.1f}' for r in rs)})" for k in keys))
json.dump(out, open(OUT_ROOT / 'dfr_fold' / 'summary.json', 'w'), indent=1)
