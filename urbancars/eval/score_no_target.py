#!/usr/bin/env python3
"""IS vs IS without target at the full staged schedule: test benchmark metrics (in %) of the stage-3 checkpoint
selected on validation (best.pth), per seed and averaged over seeds.
usage: score_no_target.py SEED [SEED ...]
  Reads $CKPT_ROOT/is/seed<s>/stage3 (IS) and $CKPT_ROOT/is_no_target/seed<s>/stage3 (IS without target)."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

CKPT = Path(os.environ['CKPT_ROOT'])
SEEDS = [int(s) for s in sys.argv[1:]]
ARMS = (('is_no_target', 'IS without target'), ('is', 'IS'))
COLS = ['id_acc', 'bg_gap', 'co_occur_obj_gap', 'both_gap']


def selected(seed, arm):
    d = CKPT / arm / f'seed{seed}' / 'stage3'
    h = [json.loads(l) for l in open(d / 'history.jsonl')]
    e = torch.load(d / 'best.pth', map_location='cpu', weights_only=False)['epoch']
    x = h[e - 1]
    assert x['epoch'] == e
    return [100 * x['test_' + c] for c in COLS]


R = {(s, a): selected(s, a) for s in SEEDS for a, _ in ARMS}
print(f"{'method':18s} {'seed':>4s} {'ID acc':>7s} {'BG Gap':>7s} {'CoObj Gap':>9s} {'BG+CoObj Gap':>12s}")
for a, name in ARMS:
    for s in SEEDS:
        print(f"{name:18s} {s:4d} " + ' '.join(f'{v:{w}.2f}' for v, w in zip(R[(s, a)], (7, 7, 9, 12))))
    m = np.mean([R[(s, a)] for s in SEEDS], axis=0)
    print(f"{name:18s} mean " + ' '.join(f'{v:{w}.2f}' for v, w in zip(m, (7, 7, 9, 12))))
