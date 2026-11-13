#!/usr/bin/env python3
"""Score Waterbirds checkpoints on all robustness tests, so that every model is evaluated on identical inputs.

Tests (all on the official test split; the fixed probe renders of eval/birdless_probe.py where edits are needed):
  natural      : WGA, mean-group accuracy, minority/majority accuracy
  weakened bird: WGA under bird-only blur / low resolution / low contrast (eval/degradation.py)
  decomposition: natural minority margin = background-alone (bird inpainted out) + bird term;
                 synthetic conflicts (bird pasted on conflicting donor) by background-salience quintile;
                 birdless images: share following the background; bird-vs-birdless AUROC
usage: eval_robustness.py NAME=KIND:PATH ...   (KIND = rn50 | cf)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'eval')); sys.path.insert(0, str(PKG / 'train')); sys.path.insert(0, str(PKG / 'data'))
import birdless_probe as rsb  # noqa: E402
from common_corners import load_model as load_rn50  # noqa: E402
import degradation as deg  # noqa: E402  (eval/degradation.py)
from common import OUT as OUT_ROOT  # noqa: E402

OUT = OUT_ROOT / 'eval'
CONDS = [('none', 0), ('blur', 2), ('blur', 4), ('blur', 8), ('lowres', 4), ('lowres', 8), ('lowres', 16), ('contrast', 0.2), ('contrast', 0.1)]
D = PKG / 'lists'


def margin(z, y):
    return z[np.arange(len(y)), y] - z[np.arange(len(y)), 1 - y]


def auroc(pos, neg):
    allv = np.r_[pos, neg]
    r = np.argsort(np.argsort(allv, kind='mergesort'), kind='mergesort') + 1.0
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def main():
    torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda')
    OUT.mkdir(parents=True, exist_ok=True)
    module = rsb.source_module()
    base = module.HFWaterbirds('test', train_aug=False, backbone='resnet50', image_size=224)
    models = {}
    for spec in sys.argv[1:]:
        name, rest = spec.split('=', 1)
        kind, path = rest.split(':', 1)
        m = load_rn50(path, dev) if kind == 'rn50' else module.load_model(Path(path), 'resnet50', 224, dev).to(memory_format=torch.channels_last)
        models[name] = m.eval()
    res = {k: {'checkpoint': s.split('=', 1)[1]} for k, s in zip(models, sys.argv[1:])}
    mean, std = deg.MEAN.to(dev), deg.STD.to(dev)
    # natural + weakened-bird conditions
    for kind, level in CONDS:
        logits = {k: [] for k in models}; ys, bgs = [], []
        for x, y, bg, _ in DataLoader(deg.Rendered(base, kind, level), batch_size=128, num_workers=16):
            xb = ((x.to(dev).float().permute(0, 3, 1, 2) / 255 - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                for k, m in models.items():
                    logits[k].append(m(xb).float().cpu())
            ys.append(y); bgs.append(bg)
        y = torch.cat(ys).numpy(); bg = torch.cat(bgs).numpy()
        for k in models:
            z = torch.cat(logits[k]).numpy(); c = z.argmax(1) == y
            g = {f'y{a}_bg{b}': float(c[(y == a) & (bg == b)].mean()) for a in (0, 1) for b in (0, 1)}
            res[k][f'{kind}_{level}'] = {'wga': min(g.values()), 'mean_group': float(np.mean(list(g.values()))), 'groups': g,
                                         'minority': float(c[y != bg].mean()), 'majority': float(c[y == bg].mean())}
            if kind == 'none':
                res[k]['_natural_logits'] = z
        print(kind, level, ' '.join(f"{k}:{100 * res[k][f'{kind}_{level}']['wga']:.1f}" for k in models), flush=True)
    # decomposition on the fixed probe renders
    ids_y = y; ids_bg = bg
    ed = {k: [] for k in models}
    with torch.inference_mode():
        for x, _ in DataLoader(rsb.Cached(), batch_size=16, num_workers=8):
            xb = x.flatten(0, 1).to(dev).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16):
                for k, m in models.items():
                    ed[k].append(m(xb).float().cpu().reshape(-1, 3, 2).numpy())
    M = ids_bg == ids_y; C = ~M
    erm_don = None
    import csv
    r = sorted(csv.DictReader(open(D / 'erm_seed1_donor_only.csv')), key=lambda x: int(x['idx']))
    p = np.clip(np.array([float(x['target_prob']) for x in r]), 1e-6, 1 - 1e-6)
    erm_don = np.log(p / (1 - p))
    q = np.array_split(np.argsort(erm_don[M]), 5)
    for k in models:
        e = np.concatenate(ed[k]); nat = margin(res[k].pop('_natural_logits'), ids_y)
        rem, don, wb = margin(e[:, 0], ids_y), margin(e[:, 1], ids_y), margin(e[:, 2], ids_y)
        res[k]['decomposition'] = {
            'natural_minority': {'margin': float(nat[C].mean()), 'bg_alone': float(rem[C].mean()), 'bird': float((nat - rem)[C].mean()), 'acc': float((nat[C] > 0).mean())},
            'synthetic_conflict': {'margin': float(wb[M].mean()), 'bg_alone': float(don[M].mean()), 'bird': float((wb - don)[M].mean()), 'acc': float((wb[M] > 0).mean())},
            'synthetic_conflict_acc_by_salience_quintile': [float((wb[M][i] > 0).mean()) for i in q],
            'bg_alone_by_salience_quintile': [float(don[M][i].mean()) for i in q],
            'birdless_follows_background': float(np.r_[np.where(ids_bg == ids_y, rem > 0, rem < 0), don < 0].mean()),
            'bird_vs_birdless_auroc': auroc(np.abs(nat), np.abs(np.r_[rem, don])),
            'interaction_d2_matching_mean': float(((nat - rem) - (wb - don))[M].mean()),
        }
    (OUT / f"results_{'_'.join(models)}.json").write_text(json.dumps(res, indent=1))
    for k in models:
        d = res[k]['decomposition']
        print(k, 'WGA', round(100 * res[k]['none_0']['wga'], 1), 'minority margin', round(d['natural_minority']['margin'], 2), '= bg', round(d['natural_minority']['bg_alone'], 2), '+ bird', round(d['natural_minority']['bird'], 2),
              '| birdless follows bg', round(100 * d['birdless_follows_background'], 1), '| salience-Q1 acc', round(100 * d['synthetic_conflict_acc_by_salience_quintile'][0], 1), flush=True)


if __name__ == '__main__':
    main()
