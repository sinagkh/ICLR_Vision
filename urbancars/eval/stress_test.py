#!/usr/bin/env python3
"""UrbanCars stress test: context reliance when the car is degraded.

Every test car (all 1000) is rendered on its class-aligned and class-conflicting background and co-object
(2 x 2 contexts, components taken from the same-stem test samples), with the car degraded by operators no model
was trained with (blur / lowres / contrast on car pixels only). Car-free context renders are added.
The Corrupted BG-Gap of a model is `corrupted_bg_gap` in the output JSON: accuracy with background and co-object
both aligned with the car class minus accuracy with both conflicting, averaged over the 8 degraded conditions.
usage: stress_test.py --out OUT.json [--max_cars N] NAME=path.pth ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT.parent))
from common import (DEFAULT_DATA_ROOT, IMAGENET_MEAN, IMAGENET_STD, build_classifier, compose_from_components,  # noqa: E402
                    enumerate_urbancars_samples, extract_sample_components, group_samples_by_factors, pick_paired_sample)

CONDS = [('none', 0), ('blur', 2), ('blur', 4), ('blur', 8), ('lowres', 4), ('lowres', 8), ('lowres', 16), ('contrast', 0.2), ('contrast', 0.1)]
STRONGEST = ['blur_8', 'lowres_16', 'contrast_0.1']
COMBOS = ['aa', 'ca', 'ac', 'cc']  # (bg, co): a = aligned with the car class, c = conflicting


def degrade(a: np.ndarray, m: np.ndarray, kind: str, level) -> np.ndarray:
    """Degrade the pixels inside the car mask m (blur, low resolution or reduced contrast)."""
    if kind == 'none' or not m.any():
        return a
    img = Image.fromarray(np.clip(a, 0, 255).round().astype(np.uint8)); a = np.asarray(img).astype(np.float32)
    if kind == 'blur':
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=level))).astype(np.float32)
    elif kind == 'lowres':
        ys, xs = np.nonzero(m); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = img.crop((x0, y0, x1, y1)); w, h = crop.size
        small = crop.resize((max(1, w // level), max(1, h // level)), Image.Resampling.BILINEAR)
        b = a.copy(); b[y0:y1, x0:x1] = np.asarray(small.resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32)
    elif kind == 'contrast':
        mu = a[m].mean(axis=0); b = mu + level * (a - mu)
    else:
        raise ValueError(kind)
    out = a.copy(); out[m] = b[m]
    return np.clip(out, 0, 255).round()


def load(path, dev):
    m = build_classifier('resnet50', 2, 'none')
    ck = torch.load(path, map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict', ck.get('model', ck)) if isinstance(ck, dict) else ck
    m.load_state_dict({k.replace('module.', ''): v for k, v in sd.items()})
    return m.to(dev).eval().to(memory_format=torch.channels_last), (ck.get('epoch') if isinstance(ck, dict) else None)


def contexts(a, grouped):
    ca = extract_sample_components(a); y = a.obj_id
    def bg_of(bid):
        return ca['background'] if a.bg_id == bid else extract_sample_components(pick_paired_sample(a, grouped, bg_id=bid))['background']
    def co_of(cid):
        if a.co_occur_obj_id == cid:
            return ca['co_rgb'], ca['co_alpha']
        c = extract_sample_components(pick_paired_sample(a, grouped, bg_id=a.bg_id, co_occur_obj_id=cid))
        return c['co_rgb'], c['co_alpha']
    bgs = {'a': bg_of(y), 'c': bg_of(1 - y)}; cos = {'a': co_of(y), 'c': co_of(1 - y)}
    return ca, bgs, cos


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', type=Path, required=True); ap.add_argument('--max_cars', type=int, default=0)
    ap.add_argument('ckpts', nargs='+'); args = ap.parse_args()
    dev = torch.device('cuda'); torch.backends.cudnn.benchmark = True
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(1, 3, 1, 1).to(dev); std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(1, 3, 1, 1).to(dev)
    test = enumerate_urbancars_samples(DEFAULT_DATA_ROOT, 'test', include_auxiliary_assets=True); grouped = group_samples_by_factors(test)
    anchors = [s for s in test if s.obj_mask_path is not None and s.co_occur_obj_mask_path is not None]
    if args.max_cars:
        anchors = anchors[:: max(1, len(anchors) // args.max_cars)][: args.max_cars]
    models, epochs = {}, {}
    for spec in args.ckpts:
        k, p = spec.split('=', 1); models[k], epochs[k] = load(p, dev)
    nC = len(CONDS)
    M = {k: np.full((len(anchors), nC * 4 + 2), np.nan, np.float32) for k in models}
    buf, rows = [], []
    def flush():
        x = torch.from_numpy(np.stack([b for b, _ in buf])).to(dev).permute(0, 3, 1, 2).float().div(255).sub(mean).div(std).contiguous(memory_format=torch.channels_last)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            for k, m in models.items():
                z = m(x).float().cpu().numpy()
                for (i, col, y), zz in zip(rows, z):
                    M[k][i, col] = zz[y] - zz[1 - y]
        buf.clear(); rows.clear()
    for i, a in enumerate(anchors):
        ca, bgs, cos = contexts(a, grouped); y = a.obj_id; car = ca['obj_alpha'][..., 0] > 0.5 if ca['obj_alpha'].ndim == 3 else ca['obj_alpha'] > 0.5
        for j, cmb in enumerate(COMBOS):
            co_rgb, co_a = cos[cmb[1]]
            base = compose_from_components(background=bgs[cmb[0]], obj_rgb=ca['obj_rgb'], obj_alpha=ca['obj_alpha'], co_rgb=co_rgb, co_alpha=co_a)
            for c, (kind, lvl) in enumerate(CONDS):
                buf.append((degrade(base, car, kind, lvl).astype(np.uint8), None)); rows.append((i, c * 4 + j, y))
        for j, cmb in enumerate(('aa', 'cc')):
            co_rgb, co_a = cos[cmb[1]]
            buf.append((np.clip(compose_from_components(background=bgs[cmb[0]], co_rgb=co_rgb, co_alpha=co_a), 0, 255).round().astype(np.uint8), None)); rows.append((i, nC * 4 + j, y))
        if len(buf) >= 190:
            flush()
        if i % 100 == 0:
            print('car', i, flush=True)
    if buf:
        flush()
    res = {'n_cars': len(anchors), 'epochs': epochs, 'models': {}}
    for k in models:
        Z = M[k]; r = {}
        for c, (kind, lvl) in enumerate(CONDS):
            name = f'{kind}_{lvl}'; m4 = Z[:, c * 4: c * 4 + 4]; acc = (m4 > 0).mean(0)
            r[name] = {'acc': dict(zip(COMBOS, acc.round(4).tolist())), 'margin': dict(zip(COMBOS, m4.mean(0).round(3).tolist())),
                       'gap_both': float(acc[0] - acc[3]), 'gap_bg': float(acc[0] - acc[1]), 'gap_co': float(acc[0] - acc[2]),
                       'pull_both': float((m4[:, 3] - m4[:, 0]).mean()), 'pull_bg': float((m4[:, 1] - m4[:, 0]).mean()), 'pull_co': float((m4[:, 2] - m4[:, 0]).mean()),
                       'compounding': float((m4[:, 3] - m4[:, 1] - m4[:, 2] + m4[:, 0]).mean())}
        r['corrupted_bg_gap_strongest'] = float(np.mean([r[s]['gap_both'] for s in STRONGEST]))
        r['corrupted_bg_gap'] = float(np.mean([r[f'{kk}_{ll}']['gap_both'] for kk, ll in CONDS if kk != 'none']))
        r['carfree_context_effect'] = float((Z[:, nC * 4 + 1] - Z[:, nC * 4]).mean())
        res['models'][k] = r
    args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(json.dumps(res, indent=1))
    np.savez_compressed(args.out.with_suffix('.npz'), **{k: v for k, v in M.items()})
    for k, r in res['models'].items():
        print(k, 'ep', epochs[k], 'none acc', r['none_0']['acc'], 'pull_both %.2f' % r['none_0']['pull_both'], 'corrupted BG-Gap %.3f' % r['corrupted_bg_gap'], 'strongest %.3f' % r['corrupted_bg_gap_strongest'],
              'carfree %.2f' % r['carfree_context_effect'], flush=True)


if __name__ == '__main__':
    main()
