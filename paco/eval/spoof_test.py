#!/usr/bin/env python3
"""Single-part spoof test.

Donors are test objects whose two named part groups each cover at least 1% of the crop; each donor is paired with
three carriers from the `unrelated` set (lists/spoof_pairs.npz, recreated with the same seed if missing). For each
pair, donor pixels under a mask are copied onto the carrier at the same crop coordinates, in five conditions:
none (carrier alone), +A, +B, +AB, and +R (a random region of the donor object outside both parts, area-matched
to part A). Saves, per checkpoint, $OUT_ROOT/spoof_scores/original/NAME.npz with the donor-class margin s
(pairs x conditions).
usage: spoof_test.py NAME=checkpoint [NAME=checkpoint ...]
"""
import json, os, random, sys
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from paco.src.common import CACHE, LISTS, load_model, normalize_gpu, read_rows, random_blob  # noqa: E402
OUT = Path(os.environ.get('OUT_ROOT', '.')) / 'spoof_scores'
CONDS = ['none', 'A', 'B', 'AB', 'R']

def build_pairs():
    p = LISTS / 'spoof_pairs.npz'
    if p.exists():
        return dict(np.load(p))
    masks = np.load(CACHE / 'test' / 'masks.npy', mmap_mode='r')
    rows = read_rows(LISTS / 'test_instances.csv'); y = np.array([int(r['class_index']) for r in rows])
    n_unrel = len(np.load(CACHE / 'unrelated' / 'crops.npy', mmap_mode='r'))
    thr = 0.01 * 224 * 224
    keep = [i for i in range(len(rows)) if ((masks[i] & 2) > 0).sum() >= thr and ((masks[i] & 4) > 0).sum() >= thr]
    rng = np.random.default_rng(20261107)
    donor, carrier = [], []
    for i in keep:
        for j in rng.choice(n_unrel, 3, replace=False):
            donor.append(i); carrier.append(int(j))
    d = dict(donor=np.array(donor), carrier=np.array(carrier), y=y[np.array(donor)], n_donors=np.array(len(keep)))
    np.savez(p, **d)
    return d

class Pairs(Dataset):
    def __init__(self, pairs):
        self.d, self.c, self.y = pairs['donor'], pairs['carrier'], pairs['y']
        self.dc = np.load(CACHE / 'test' / 'crops.npy', mmap_mode='r'); self.dm = np.load(CACHE / 'test' / 'masks.npy', mmap_mode='r')
        self.cc = np.load(CACHE / 'unrelated' / 'crops.npy', mmap_mode='r')
    def __len__(self): return len(self.d)
    def __getitem__(self, k):
        i, j = int(self.d[k]), int(self.c[k])
        don, mk, car = np.asarray(self.dc[i]), np.asarray(self.dm[i]), np.asarray(self.cc[j])
        a, b, obj = (mk & 2) > 0, (mk & 4) > 0, (mk & 1) > 0
        r = random_blob(obj & ~a & ~b, int(a.sum()), random.Random(20261107 * 1000 + k))
        out = []
        for m in (None, a, b, a | b, r):
            x = car.copy()
            if m is not None: x[m] = don[m]
            out.append(x)
        return torch.from_numpy(np.stack(out)), int(self.y[k]), float(r.sum() / max(1, a.sum()))

def margin(z, y):
    zc = z[np.arange(len(y)), y].copy(); z2 = z.copy(); z2[np.arange(len(y)), y] = -np.inf
    return zc - z2.max(1)

@torch.inference_mode()
def main():
    pairs = build_pairs(); dev = torch.device('cuda')
    specs = [s.split('=', 1) for s in sys.argv[1:]]
    for name, path in specs:
        f = OUT / 'original' / f'{name}.npz'
        if f.exists(): print('skip', name); continue
        f.parent.mkdir(parents=True, exist_ok=True)
        model = load_model(Path(path), dev); S = []; ys = []; rr = []
        for x, y, r in DataLoader(Pairs(pairs), batch_size=48, num_workers=10):
            xb = normalize_gpu(x.flatten(0, 1), dev)
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(xb).float().cpu().numpy().reshape(len(y), len(CONDS), -1)
            S.append(np.stack([margin(z[:, c], y.numpy()) for c in range(len(CONDS))], 1)); ys.append(y.numpy()); rr.append(r.numpy())
        np.savez_compressed(f, s=np.concatenate(S), y=np.concatenate(ys), r_area_ratio=np.concatenate(rr), checkpoint=path)
        print('saved', name, flush=True); del model; torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
