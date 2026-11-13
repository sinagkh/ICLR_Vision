#!/usr/bin/env python3
"""Object-only corruption test: blur, low resolution (downsample) or reduced contrast applied to the bird pixels only
(released CUB mask, dilated by 2 px); background pixels unchanged. Inference only on existing checkpoints.
`degrade`, `Rendered`, `MEAN` and `STD` are reused by eval_robustness.py, guard_analysis.py and unseen_scenes.py."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from scipy import ndimage
from torch.utils.data import DataLoader, Dataset

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG / 'eval')); sys.path.insert(0, str(PKG / 'train')); sys.path.insert(0, str(PKG / 'data'))
import birdless_probe as rsb  # noqa: E402
from common_corners import load_model as load_rn50  # noqa: E402
from common import checkpoint, OUT as OUT_ROOT, CKPT  # noqa: E402

OUT = OUT_ROOT / 'degradation'
CF = CKPT / 'full_cfaug'
CONDS = [('none', 0)] + [('blur', r) for r in (2, 4, 8, 16)] + [('lowres', f) for f in (4, 8, 16, 32)] + [('contrast', a) for a in (0.4, 0.2, 0.1, 0.05)]
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def degrade(img: Image.Image, m: np.ndarray, kind: str, level) -> Image.Image:
    if kind == 'none' or not m.any():
        return img
    a = np.asarray(img).astype(np.float32)
    if kind == 'blur':
        b = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=level))).astype(np.float32)
    elif kind == 'lowres':
        ys, xs = np.nonzero(m)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        crop = img.crop((x0, y0, x1, y1))
        w, h = crop.size
        small = crop.resize((max(1, w // level), max(1, h // level)), Image.Resampling.BILINEAR)
        up = np.asarray(small.resize((w, h), Image.Resampling.BILINEAR)).astype(np.float32)
        b = a.copy(); b[y0:y1, x0:x1] = up
    elif kind == 'contrast':
        mu = a[m].mean(axis=0)
        b = mu + level * (a - mu)
    else:
        raise ValueError(kind)
    out = a.copy(); out[m] = b[m]
    return Image.fromarray(np.clip(out, 0, 255).round().astype(np.uint8))


class Rendered(Dataset):
    def __init__(self, base, kind, level):
        self.base, self.kind, self.level = base, kind, level

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        _, y, _, pil, bg, idx = self.base[i]
        mask = self.base.load_mask(int(idx))
        if mask is None:
            m = np.zeros((pil.size[1], pil.size[0]), dtype=bool); has = 0
        else:
            m = np.asarray(mask.convert('L').resize(pil.size, Image.Resampling.NEAREST)) > 127
            m = ndimage.binary_dilation(m, iterations=2); has = 1
        out = degrade(pil.convert('RGB'), m, self.kind, self.level)
        return torch.from_numpy(np.asarray(out).copy()), int(y), int(bg), has


def models(module, dev):
    specs = [(f'erm_s{s}', 'rn50', checkpoint(s, 'erm')) for s in (1, 21, 42)]
    specs += [(f'is_s{s}', 'rn50', checkpoint(s, 'full_is')) for s in (1, 21, 42)]
    specs += [('notarget_s1', 'rn50', checkpoint(1, 'no_target'))]
    specs += [(f'cfaug_s{s}', 'cf', CF / f'seed{s}/cfaug_mixed/checkpoints/cfaug_mixed_best.pt') for s in (1, 21, 42)]
    out = {}
    for name, kind, path in specs:
        m = load_rn50(path, dev) if kind == 'rn50' else module.load_model(path, 'resnet50', 224, dev).to(memory_format=torch.channels_last)
        out[name] = m.eval()
    return out


def main():
    torch.backends.cudnn.benchmark = True
    dev = torch.device('cuda')
    module = rsb.source_module()
    base = module.HFWaterbirds('test', train_aug=False, backbone='resnet50', image_size=224)
    # check: GPU normalization of the uint8 PIL equals the dataset transform
    x0, _, _, pil0, _, _ = base[0]
    mine = ((torch.from_numpy(np.asarray(pil0.convert('RGB')).copy()).float().permute(2, 0, 1)[None] / 255 - MEAN) / STD)[0]
    assert tuple(mine.shape) == tuple(x0.shape), (mine.shape, x0.shape)
    err = float((mine - x0).abs().max())
    print('normalization check max abs diff', err, flush=True)
    assert err < 1e-4
    M = models(module, dev)
    results = {}
    for kind, level in CONDS:
        t0 = time.time()
        logits = {k: [] for k in M}
        ys, bgs, has = [], [], []
        for x, y, bg, h in DataLoader(Rendered(base, kind, level), batch_size=128, num_workers=16):
            xb = ((x.to(dev).float().permute(0, 3, 1, 2) / 255 - MEAN.to(dev)) / STD.to(dev)).contiguous(memory_format=torch.channels_last)
            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                for k, m in M.items():
                    logits[k].append(m(xb).float().cpu())
            ys.append(y); bgs.append(bg); has.append(h)
        y = torch.cat(ys).numpy(); bg = torch.cat(bgs).numpy(); h = torch.cat(has).numpy()
        key = f'{kind}_{level}'
        results[key] = {'n': int(len(y)), 'n_without_mask': int((h == 0).sum()), 'models': {}}
        for k in M:
            z = torch.cat(logits[k]).numpy()
            correct = z.argmax(1) == y
            g = {f'y{a}_bg{b}': float(correct[(y == a) & (bg == b)].mean()) for a in (0, 1) for b in (0, 1)}
            margin = z[np.arange(len(y)), y] - z[np.arange(len(y)), 1 - y]
            results[key]['models'][k] = {
                'groups': g, 'wga': min(g.values()), 'mean_group': float(np.mean(list(g.values()))),
                'minority': float(correct[y != bg].mean()), 'majority': float(correct[y == bg].mean()),
                'minority_margin': float(margin[y != bg].mean()), 'majority_margin': float(margin[y == bg].mean()),
            }
        print(key, f'{time.time() - t0:.0f}s', ' '.join(f"{k}:{100 * results[key]['models'][k]['wga']:.1f}" for k in M), flush=True)
        (OUT / 'results.json').write_text(json.dumps(results, indent=1))


if __name__ == '__main__':
    main()
