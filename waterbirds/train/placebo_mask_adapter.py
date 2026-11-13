#!/usr/bin/env python3
"""Placebo-mask IS. Same trainer, objective, donors and schedule as IS, but every anchor's
object mask is replaced by the mask of another training image (similar size, 0.5-2x, where possible) that overlaps the anchor's own bird by < 10% of its area.
Donor cleansing keeps the correct masks. The anchor index is swapped only inside d2_penalty, where it is used solely to load the mask.
usage: placebo_mask_adapter.py [train_is.py arguments...]"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from is_with_cub_masks import load_training_module

module = load_training_module()
out_dir = Path(sys.argv[sys.argv.index('--output_dir') + 1]); out_dir.mkdir(parents=True, exist_ok=True)
ds = module.HFWaterbirds('train', image_size=224, train_aug=False, backbone='resnet50')
masks = np.stack([np.asarray(ds.load_mask(i)) > 127 for i in range(len(ds))])
area = masks.reshape(len(ds), -1).sum(1)
rng = np.random.default_rng(20261107)
MAP, overlaps = {}, []
flat = masks.reshape(len(ds), -1).astype(np.float32)
for i in range(len(ds)):
    ovs = (flat @ flat[i]) / np.maximum(area, 1)   # share of each candidate mask that lies on anchor i's bird
    ovs[i] = np.inf; ovs[area == 0] = np.inf
    order = rng.permutation(len(ds)); ratio = area[order] / max(area[i], 1)
    ok = order[(ovs[order] < 0.10) & (ratio >= 0.5) & (ratio <= 2.0)]   # prefer a mask of similar size
    if not len(ok):
        ok = order[ovs[order] < 0.10]
    j = int(ok[0]) if len(ok) else int(np.argmin(ovs))   # last resort: least-overlapping mask
    MAP[i] = j; overlaps.append(float(ovs[j]))
(out_dir / 'placebo_map.json').write_text(json.dumps({'n': len(MAP), 'mean_overlap': float(np.mean(overlaps)), 'n_over_0.10': int((np.array(overlaps) >= 0.10).sum()), 'map': MAP}))
print(f'placebo map: {len(MAP)}/{len(ds)} anchors, mean overlap {np.mean(overlaps):.4f}, over 0.10: {int((np.array(overlaps) >= 0.10).sum())}', flush=True)

_orig = module.d2_penalty
def placebo_d2_penalty(*a, **kw):
    kw['anchors'] = [(p, y, bg, MAP[int(i)]) for (p, y, bg, i) in kw['anchors']]
    return _orig(*a, **kw)
module.d2_penalty = placebo_d2_penalty

if __name__ == '__main__':
    module.main()
