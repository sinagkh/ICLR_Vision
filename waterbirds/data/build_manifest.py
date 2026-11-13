#!/usr/bin/env python3
"""Build the Waterbirds manifest read by train/train_is.py, train/train_cfaug_full.py and data/audit_interaction.py.

One CSV per split ($DATA_ROOT/waterbirds_manifest/waterbirds_{train,eval,test}.csv), one row per image of the
grodino/waterbirds split in the same order, pointing to the official Waterbirds image and its CUB-200-2011 bird mask.
Rows are matched to the official metadata.csv in split order, and labels and places are checked row by row.
usage: build_manifest.py   (reads $DATA_ROOT/hf_cache, $DATA_ROOT/waterbird_complete95_forest2water2, $DATA_ROOT/segmentations)
"""
import csv
import os
from pathlib import Path

from datasets import Dataset as ArrowDataset

from waterbirds_data import _arrow_file

ROOT = Path(os.environ.get('DATA_ROOT', '.')).resolve()
OFFICIAL = ROOT / 'waterbird_complete95_forest2water2'
SEGMENTATIONS = ROOT / 'segmentations'
OUT = ROOT / 'waterbirds_manifest'
SPLITS = (('train', 0, 'waterbirds_train.csv'), ('validation', 1, 'waterbirds_eval.csv'), ('test', 2, 'waterbirds_test.csv'))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OFFICIAL / 'metadata.csv').open(newline='') as f:
        metadata = list(csv.DictReader(f))
    for split, code, name in SPLITS:
        ds = ArrowDataset.from_file(str(_arrow_file(ROOT / 'hf_cache', split))).select_columns(['label', 'place'])
        meta = [r for r in metadata if int(r['split']) == code]
        if len(meta) != len(ds):
            raise RuntimeError(f'{split}: {len(meta)} metadata rows vs {len(ds)} images')
        rows = []
        for i, (ex, m) in enumerate(zip(ds, meta)):
            y, bg = int(ex['label']), int(ex['place'])
            if y != int(m['y']) or bg != int(m['place']):
                raise RuntimeError(f'{split}: label/place mismatch at row {i}')
            mask = SEGMENTATIONS / Path(m['img_filename']).with_suffix('.png')
            rows.append({'image_path': os.path.relpath(OFFICIAL / m['img_filename'], OUT),
                         'label': y, 'background': bg, 'group': 2 * y + bg,
                         'mask_available': int(mask.exists()),
                         'bird_mask_path': os.path.relpath(mask, OUT) if mask.exists() else ''})
        with (OUT / name).open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f'{OUT / name}: {len(rows)} rows, {sum(r["mask_available"] for r in rows)} with masks')


if __name__ == '__main__':
    main()
