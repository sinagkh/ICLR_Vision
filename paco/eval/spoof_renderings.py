#!/usr/bin/env python3
"""Single-part spoof test under two further renderings. Same 7,320 donor-carrier pairs and the same five conditions
(none/+A/+B/+AB/+R) as spoof_test.py; only the way donor pixels are put on the carrier changes.
  blend: donor colours partly matched to the carrier under the mask, pasted with a feathered (Gaussian, sigma 3 px) alpha.
  shift: hard paste, but every pasted region of the pair is translated by one random offset (same for all conditions),
         so parts appear at positions where they never were on the donor.
Saves $OUT_ROOT/spoof_scores/<VARIANT>/NAME.npz.
usage: spoof_renderings.py VARIANT NAME=checkpoint [NAME=checkpoint ...]"""
import os, random, sys
from pathlib import Path
import numpy as np, torch
from scipy.ndimage import gaussian_filter
from torch.utils.data import DataLoader, Dataset
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
from paco.src.common import CACHE, LISTS, load_model, normalize_gpu, random_blob  # noqa: E402
OUT = Path(os.environ.get('OUT_ROOT', '.')) / 'spoof_scores'
CONDS = ['none', 'A', 'B', 'AB', 'R']

def harmonize(don, car, m, mix=0.5):
    d = don.astype(np.float32).copy()
    if m.sum() < 2: return d
    for ch in range(3):
        dv, cv = d[..., ch][m], car[..., ch][m].astype(np.float32)
        z = (dv - dv.mean()) / (dv.std() + 1e-6) * (cv.std() + 1e-6) + cv.mean()
        d[..., ch][m] = (1 - mix) * dv + mix * z
    return d

def offset(union, rng):
    ys, xs = np.nonzero(union)
    if len(ys) == 0: return 0, 0
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); H, W = union.shape
    for _ in range(50):
        dy, dx = rng.randint(-y0, H - 1 - y1), rng.randint(-x0, W - 1 - x1)
        if max(abs(dy), abs(dx)) >= 32: return dy, dx
    return rng.randint(-y0, H - 1 - y1), rng.randint(-x0, W - 1 - x1)

def shift(a, dy, dx):
    out = np.zeros_like(a); H, W = a.shape[:2]
    ys, yd = (slice(0, H - dy), slice(dy, H)) if dy >= 0 else (slice(-dy, H), slice(0, H + dy))
    xs, xd = (slice(0, W - dx), slice(dx, W)) if dx >= 0 else (slice(-dx, W), slice(0, W + dx))
    out[yd, xd] = a[ys, xs]; return out

class Pairs(Dataset):
    def __init__(self, pairs, variant):
        self.d, self.c, self.y, self.v = pairs['donor'], pairs['carrier'], pairs['y'], variant
        self.dc = np.load(CACHE / 'test' / 'crops.npy', mmap_mode='r'); self.dm = np.load(CACHE / 'test' / 'masks.npy', mmap_mode='r')
        self.cc = np.load(CACHE / 'unrelated' / 'crops.npy', mmap_mode='r')
    def __len__(self): return len(self.d)
    def __getitem__(self, k):
        i, j = int(self.d[k]), int(self.c[k])
        don, mk, car = np.asarray(self.dc[i]), np.asarray(self.dm[i]), np.asarray(self.cc[j])
        a, b, obj = (mk & 2) > 0, (mk & 4) > 0, (mk & 1) > 0
        r = random_blob(obj & ~a & ~b, int(a.sum()), random.Random(20261107 * 1000 + k))   # same R region as in spoof_test.py
        if self.v == 'shift':
            dy, dx = offset(a | b | r, random.Random(20261108 * 1000 + k))
            don, a, b, r = shift(don, dy, dx), shift(a, dy, dx), shift(b, dy, dx), shift(r, dy, dx)
        out = []
        for m in (None, a, b, a | b, r):
            x = car.copy()
            if m is not None:
                if self.v == 'blend':
                    d = harmonize(don, car, m); al = np.clip(gaussian_filter(m.astype(np.float32), 3.0) * 1.5, 0, 1)[..., None]
                    x = np.clip(al * d + (1 - al) * car.astype(np.float32), 0, 255).astype(np.uint8)
                else:
                    x[m] = don[m]
            out.append(x)
        return torch.from_numpy(np.stack(out)), int(self.y[k]), float(r.sum() / max(1, a.sum()))

def margin(z, y):
    zc = z[np.arange(len(y)), y].copy(); z2 = z.copy(); z2[np.arange(len(y)), y] = -np.inf
    return zc - z2.max(1)

@torch.inference_mode()
def main():
    variant = sys.argv[1]; assert variant in ('blend', 'shift')
    pairs = dict(np.load(LISTS / 'spoof_pairs.npz')); dev = torch.device('cuda')
    for name, path in [s.split('=', 1) for s in sys.argv[2:]]:
        f = OUT / variant / f'{name}.npz'
        if f.exists(): print('skip', name); continue
        f.parent.mkdir(parents=True, exist_ok=True)
        model = load_model(Path(path), dev); S = []; ys = []; rr = []
        for x, y, r in DataLoader(Pairs(pairs, variant), batch_size=48, num_workers=12):
            xb = normalize_gpu(x.flatten(0, 1), dev)
            with torch.autocast('cuda', dtype=torch.float16):
                z = model(xb).float().cpu().numpy().reshape(len(y), len(CONDS), -1)
            S.append(np.stack([margin(z[:, c], y.numpy()) for c in range(len(CONDS))], 1)); ys.append(y.numpy()); rr.append(r.numpy())
        np.savez_compressed(f, s=np.concatenate(S), y=np.concatenate(ys), r_area_ratio=np.concatenate(rr), checkpoint=path)
        print('saved', variant, name, flush=True); del model; torch.cuda.empty_cache()

if __name__ == '__main__':
    main()
