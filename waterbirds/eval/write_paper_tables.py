#!/usr/bin/env python3
"""Write paper_table10_11.csv and paper_table13_exmap.csv of results/waterbirds/ from the runs' train_summary.json files.

usage: write_paper_tables.py OUT_DIR TABLE|BACKBONE|MODEL|SEED=RUN_DIR[:ENTRY] ...

  TABLE      table10_11 or table13_exmap (the output file is paper_<TABLE>.csv)
  RUN_DIR    output directory of a training run; its train_summary.json holds the test group accuracies of the
             selected checkpoint under ENTRY (default best_checkpoint)

Columns (accuracies in %):
  wga                  worst-group test accuracy
  avg_acc              test accuracy = group accuracies weighted by the test group sizes (2255 / 2255 / 642 / 642)
  delta_wga            wga minus the wga of ERM with the same backbone and seed (the ERM rows of the same table)
  acc_<class>_<place>  test accuracy of each group (bird class x background)
  checkpoint           checkpoint file the row is computed from
One row per seed, then the mean and the sample standard deviation over seeds.
"""
import csv
import json
import statistics
import sys
from pathlib import Path

GROUPS = ['acc_landbird_land', 'acc_landbird_water', 'acc_waterbird_land', 'acc_waterbird_water']  # group = 2*y + bg
METRICS = ['wga', 'avg_acc', 'delta_wga'] + GROUPS
HEADER = ['backbone', 'model', 'seed'] + METRICS + ['checkpoint']


def parse(spec):
    label, run = spec.split('=', 1)
    table, backbone, model, seed = label.split('|')
    entry = 'best_checkpoint'
    if ':' in run and run.rsplit(':', 1)[1].endswith('_checkpoint'):
        run, entry = run.rsplit(':', 1)
    return table, backbone, model, int(seed), Path(run), entry


def read_run(run, entry):
    ck = json.load(open(run / 'train_summary.json'))[entry]
    acc_n = [ck['test_groups'][str(g)] for g in range(4)]
    row = {name: 100 * a for name, (a, _) in zip(GROUPS, acc_n)}
    row['wga'] = min(row[name] for name in GROUPS)
    row['avg_acc'] = 100 * sum(a * n for a, n in acc_n) / sum(n for _, n in acc_n)
    row['checkpoint'] = Path(ck['path']).name
    return row


def main(out_dir, specs):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    for spec in specs:
        table, backbone, model, seed, run, entry = parse(spec)
        tables.setdefault(table, {}).setdefault((backbone, model), {})[seed] = read_run(run, entry)
    for table, models in tables.items():
        rows = []
        for (backbone, model), by_seed in models.items():
            erm = models[(backbone, 'ERM')]
            for seed, r in by_seed.items():
                r['delta_wga'] = r['wga'] - erm[seed]['wga']
            rows += [[backbone, model, seed] + [f'{r[m]:.6f}' for m in METRICS] + [r['checkpoint']] for seed, r in by_seed.items()]
            for name, stat in (('mean', statistics.mean), ('sd', statistics.stdev)):
                rows.append([backbone, model, name] + [f'{stat([r[m] for r in by_seed.values()]):.6f}' for m in METRICS] + [''])
        path = out_dir / f'paper_{table}.csv'
        with open(path, 'w', newline='') as f:
            w = csv.writer(f, lineterminator='\n'); w.writerow(HEADER); w.writerows(rows)
        print('wrote', path)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
