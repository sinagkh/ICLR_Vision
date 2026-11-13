#!/usr/bin/env python3
"""CF-Aug vs CF-Aug + IS for one seed: test benchmark metrics at epoch 20 and the Corrupted BG-Gap (in %).
Arms: labels = CF-Aug, labis = CF-Aug + IS.
usage: score_cfaug.py SEEDDIR STRESS_JSON
  SEEDDIR: directory with one subdirectory per arm, each holding its history.jsonl
  STRESS_JSON: output of eval/stress_test.py for the epoch-20 checkpoints"""
import json
import sys
from pathlib import Path

seeddir = Path(sys.argv[1])
models = json.load(open(sys.argv[2]))['models']
cols = ['id_acc', 'bg_gap', 'co_occur_obj_gap', 'both_gap']
print(f"{'method':12s} {'ID acc':>7s} {'BG Gap':>7s} {'CoObj Gap':>9s} {'BG+CoObj Gap':>12s} {'Corrupted BG-Gap':>16s}")
for arm, name in (('labels', 'CF-Aug'), ('labis', 'CF-Aug + IS')):
    r = [json.loads(l) for l in open(seeddir / arm / 'history.jsonl')][19]
    assert r['epoch'] == 20
    v = [100 * r['test_' + c] for c in cols] + [100 * models[arm]['corrupted_bg_gap']]
    print(f"{name:12s} {v[0]:7.2f} {v[1]:7.2f} {v[2]:9.2f} {v[3]:12.2f} {v[4]:16.2f}")
