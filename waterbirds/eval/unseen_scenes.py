#!/usr/bin/env python3
"""Unseen-scene tests on Places365 validation scenes of 28 categories not used by Waterbirds (100 images each; list in
lists/places365_unseen_scenes.csv). (1) birdless real scenes: scene-following = mean of P(waterbird | water scene) and
P(landbird | land scene). (2) every test bird pasted (CUB mask, dilated 2 px) on one random water and one random land scene,
natural or degraded by eval/degradation.py::degrade; Corrupted BG-Gap = aligned minus conflicting accuracy, averaged over
blur 8, lowres 16 and contrast 0.1.
usage: unseen_scenes.py SEED OUT.json NAME=KIND:PATH ...  (KIND rn50 | cf)"""
import csv, json, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset
PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'eval')); sys.path.insert(0, str(PKG / 'train')); sys.path.insert(0, str(PKG / 'data'))
import birdless_probe as rsb
from common_corners import load_model as load_rn50
import degradation as deg
from common import DATA_ROOT
P = DATA_ROOT / 'places365'
CONDS = [('none', 0), ('blur', 8), ('lowres', 16), ('contrast', 0.1)]
rows = list(csv.DictReader(open(PKG / 'lists' / 'places365_unseen_scenes.csv')))
SCENE_T = np.array([1 if r['type'] == 'water' else 0 for r in rows])   # Waterbirds convention: place 1 = water, label 1 = waterbird

def scene(k):
    return np.asarray(Image.open(P / 'val_256' / rows[k]['file']).convert('RGB').resize((224, 224), Image.Resampling.BILINEAR))

class Scenes(Dataset):
    def __len__(self): return len(rows)
    def __getitem__(self, k): return torch.from_numpy(scene(k).copy())

class Composites(Dataset):
    def __init__(self, base, pairs): self.base, self.pairs = base, pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self, j):
        i, k = self.pairs[j]
        _, y, _, pil, bg, idx = self.base[int(i)]
        mk = self.base.load_mask(int(idx))
        m = ndimage.binary_dilation(np.asarray(mk.convert('L').resize(pil.size, Image.Resampling.NEAREST)) > 127, iterations=2)
        sc = scene(int(k)); out = []
        for kind, lvl in CONDS:
            b = np.asarray(deg.degrade(pil.convert('RGB'), m, kind, lvl))
            out.append(np.where(m[..., None], b, sc))
        return torch.from_numpy(np.stack(out)), int(y), int(SCENE_T[k])

def main():
    seed, out = int(sys.argv[1]), Path(sys.argv[2]); dev = torch.device('cuda')
    module = rsb.source_module(); base = module.HFWaterbirds('test', train_aug=False, backbone='resnet50', image_size=224)
    models = {}
    for spec in sys.argv[3:]:
        name, rest = spec.split('=', 1); kind, path = rest.split(':', 1)
        models[name] = (load_rn50(Path(path), dev) if kind == 'rn50' else module.load_model(Path(path), 'resnet50', 224, dev).to(memory_format=torch.channels_last)).eval()
    mean, std = deg.MEAN.to(dev), deg.STD.to(dev)
    norm = lambda x: ((x.to(dev).float().permute(0, 3, 1, 2) / 255 - mean) / std).contiguous(memory_format=torch.channels_last)
    res = {k: {} for k in models}
    # 1) birdless real scenes
    P1 = {k: [] for k in models}
    with torch.inference_mode():
        for x in DataLoader(Scenes(), batch_size=128, num_workers=8):
            xb = norm(x)
            with torch.autocast('cuda', dtype=torch.float16):
                for k, m in models.items(): P1[k].append(m(xb).float().argmax(1).cpu().numpy())
    cat = np.array([r['category'] for r in rows])
    for k in models:
        pred = np.concatenate(P1[k])
        pw = float((pred[SCENE_T == 1] == 1).mean()); pl = float((pred[SCENE_T == 0] == 0).mean())
        res[k]['scene_following'] = 0.5 * (pw + pl); res[k]['p_waterbird_on_water'] = pw; res[k]['p_landbird_on_land'] = pl
        res[k]['per_category_follow'] = {c: float((pred[cat == c] == SCENE_T[cat == c][0]).mean()) for c in sorted(set(cat))}
    # 2) birds pasted on unseen scenes: each test bird on one random water and one random land scene (same pairs for every model and seed)
    rng = np.random.default_rng(20261107); water, land = np.where(SCENE_T == 1)[0], np.where(SCENE_T == 0)[0]
    pairs = [(i, int(rng.choice(water))) for i in range(len(base))] + [(i, int(rng.choice(land))) for i in range(len(base))]
    Z = {k: [] for k in models}; Y, T = [], []
    with torch.inference_mode():
        for x, y, t in DataLoader(Composites(base, pairs), batch_size=16, num_workers=12):
            xb = norm(x.flatten(0, 1))
            with torch.autocast('cuda', dtype=torch.float16):
                for k, m in models.items(): Z[k].append(m(xb).float().argmax(1).cpu().numpy().reshape(len(y), len(CONDS)))
            Y.append(y.numpy()); T.append(t.numpy())
    Y, T = np.concatenate(Y), np.concatenate(T); match = Y == T
    for k in models:
        pred = np.concatenate(Z[k]); c = pred == Y[:, None]
        per = {}
        for ci, (kind, lvl) in enumerate(CONDS):
            g = {f'y{a}_scene{b}': float(c[(Y == a) & (T == b), ci].mean()) for a in (0, 1) for b in (0, 1)}
            per[f'{kind}_{lvl}'] = dict(aligned=float(c[match, ci].mean()), conflict=float(c[~match, ci].mean()), wga=min(g.values()), groups=g)
        res[k]['composites'] = per
        res[k]['corrupted_bg_gap'] = float(np.mean([per[f'{a}_{b}']['aligned'] - per[f'{a}_{b}']['conflict'] for a, b in CONDS[1:]]))
        print(f"s{seed} {k:10s} scene-following {100*res[k]['scene_following']:5.1f} (water {100*res[k]['p_waterbird_on_water']:5.1f} land {100*res[k]['p_landbird_on_land']:5.1f}) | natural bird: conflict {100*per['none_0']['conflict']:5.1f} wga {100*per['none_0']['wga']:5.1f} | corrupted BG-Gap {100*res[k]['corrupted_bg_gap']:5.1f}", flush=True)
    out.write_text(json.dumps(res, indent=1))

if __name__ == '__main__':
    main()
