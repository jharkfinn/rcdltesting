"""Aggregates finished runs into a markdown table: mean ± sample std of final test accuracy over seeds.

Usage: python -m experiments.summarize results/gpu [--configs experiments/configs/*.json]
Reference columns come from the 'notebook_acc' / 'notebook_tuned' / 'paper_acc' fields of the config files.
"""

import argparse
import glob
import json
import os

import numpy as np


def main():
  p = argparse.ArgumentParser()
  p.add_argument('root')
  p.add_argument('--configs', nargs='*', default=sorted(glob.glob('experiments/configs/*.json')))
  args = p.parse_args()
  refs = {}
  for c in args.configs:
    for name, exp in json.load(open(c))['experiments'].items():
      refs[name] = exp
  rows = {}
  for s in sorted(glob.glob(os.path.join(args.root, '*', 'seed*', 'summary.json'))):
    name = s.split(os.sep)[-3]
    r = json.load(open(s))
    rows.setdefault(name, []).append((r['final_test_acc'], r['final_train_loss_avg100'], r['elapsed']))
  print('| experiment | seeds | test acc % (mean ± std) | min-max | train loss (last 100) | sec/run | notebook | paper |')
  print('|---|---|---|---|---|---|---|---|')
  for name, vals in rows.items():
    a = np.array([v[0] for v in vals]) * 100
    ref = refs.get(name.split('_seed')[0], {})
    nb = ref.get('notebook_acc', ref.get('notebook_tuned', ''))
    nb = f'{nb * 100:.2f}' if isinstance(nb, float) else nb
    sd = f'{a.std(ddof=1):.2f}' if len(a) > 1 else '-'
    print(f'| {name} | {len(a)} | {a.mean():.2f} ± {sd} | {a.min():.2f}-{a.max():.2f} | '
          f'{np.mean([v[1] for v in vals]):.3f} | {np.mean([v[2] for v in vals]):.0f} | {nb} | {ref.get("paper_acc", "")} |')


if __name__ == '__main__':
  main()
