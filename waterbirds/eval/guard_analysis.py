#!/usr/bin/env python3
"""Where each model sits relative to the IS guard bounds, on held-out test anchors (fixed probe renders of eval/birdless_probe.py).
Corners per test image: m00 natural, m10 bird removed, m01 bird pasted on a conflicting donor, m11 donor alone (margins w.r.t. the true label).
  e_obj_own = m00 - m10 (bird effect on its own background)    e_obj_don = m01 - m11 (bird effect on the donor background)
  e_bg_bird = m00 - m01 (background effect with the bird present: what the p_bg guard bounds by tau_bg_guard = 0.5)
  e_bg_alone = m10 - m11 (background main effect without the bird)    d2 = e_obj_own - e_obj_don (the interaction target, tau 0.05)
Inference only. usage: guard_analysis.py OUT.json NAME=rn50:PATH ..."""
import json, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'eval')); sys.path.insert(0, str(PKG / 'train')); sys.path.insert(0, str(PKG / 'data'))
import birdless_probe as rsb
from common_corners import load_model as load_rn50
import degradation as deg

def margin(z, y): return z[np.arange(len(y)), y] - z[np.arange(len(y)), 1 - y]

def main():
    out = Path(sys.argv[1]); dev = torch.device('cuda')
    module = rsb.source_module(); base = module.HFWaterbirds('test', train_aug=False, backbone='resnet50', image_size=224)
    models = {s.split('=', 1)[0]: load_rn50(s.split('=', 1)[1].split(':', 1)[1], dev).eval() for s in sys.argv[2:]}
    mean, std = deg.MEAN.to(dev), deg.STD.to(dev)
    nat = {k: [] for k in models}; ys, bgs = [], []
    with torch.inference_mode():
        for x, y, bg, _ in DataLoader(deg.Rendered(base, 'none', 0), batch_size=64, num_workers=12):
            xb = ((x.to(dev).float().permute(0, 3, 1, 2) / 255 - mean) / std).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16):
                for k, m in models.items(): nat[k].append(m(xb).float().cpu().numpy())
            ys.append(y.numpy()); bgs.append(bg.numpy())
        y = np.concatenate(ys); bg = np.concatenate(bgs)
        ed = {k: [] for k in models}
        for x, _ in DataLoader(rsb.Cached(), batch_size=16, num_workers=8):
            xb = x.flatten(0, 1).to(dev).contiguous(memory_format=torch.channels_last)
            with torch.autocast('cuda', dtype=torch.float16):
                for k, m in models.items(): ed[k].append(m(xb).float().cpu().reshape(-1, 3, 2).numpy())
    aligned = bg == y   # training-like anchors: bird on its own (class-matching) background, donor background conflicting
    res = {'n_aligned': int(aligned.sum())}
    for k in models:
        z = np.concatenate(nat[k]); e = np.concatenate(ed[k])
        m00, m10, m11, m01 = margin(z, y), margin(e[:, 0], y), margin(e[:, 1], y), margin(e[:, 2], y)
        q = {n: v[aligned] for n, v in dict(e_obj_own=m00 - m10, e_obj_don=m01 - m11, e_bg_bird=m00 - m01, e_bg_alone=m10 - m11, d2=(m00 - m10) - (m01 - m11), m00=m00, m01=m01).items()}
        r = {n: {'mean': float(v.mean()), 'median': float(np.median(v)), 'q25': float(np.quantile(v, 0.25))} for n, v in q.items()}
        r['frac_bg_guard_violated'] = float((np.abs(q['e_bg_bird']) > 0.5).mean())       # |e_bg| > 0.5 -> p_bg active
        r['frac_obj_below_0p2'] = float((q['e_obj_own'] < 0.2).mean())                    # below the floor's minimum
        r['frac_obj_above_6'] = float((q['e_obj_own'] > 6.0).mean())                      # p_obj_hi active
        r['frac_d2_target_violated'] = float((np.abs(q['d2']) > 0.05).mean())
        r['mean_bg_guard_excess'] = float(np.clip(np.abs(q['e_bg_bird']) - 0.5, 0, None).mean())
        res[k] = r
        print(f"{k:12s} e_obj own {r['e_obj_own']['mean']:+.2f} don {r['e_obj_don']['mean']:+.2f} | e_bg bird {r['e_bg_bird']['mean']:+.2f} alone {r['e_bg_alone']['mean']:+.2f} | d2 {r['d2']['mean']:+.2f} (|d2|>.05 {r['frac_d2_target_violated']:.2f}) | bg-guard violated {r['frac_bg_guard_violated']:.2f} excess {r['mean_bg_guard_excess']:.2f} | obj<0.2 {r['frac_obj_below_0p2']:.2f} >6 {r['frac_obj_above_6']:.2f}", flush=True)
    out.write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
