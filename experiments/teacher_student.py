"""Teacher-student regression runs and grids (paper Fig. 3).

Usage: python -m experiments.teacher_student experiments/configs/teacher_student.json --only gws_sweep --out results/ts
Config format: {"experiments": {name: {"base": {TSConfig fields}, "grid": {field: [values], ...}, "eval_every": int}}}.
Each grid point runs once and writes <out>/<name>/<point>/result.json; a summary sorted by final test MSE is printed.
"""

import argparse
import dataclasses as dc
import itertools
import json
import os

import torch

from rcdl.teacher_student import TSConfig, run_teacher_student


def main():
  p = argparse.ArgumentParser()
  p.add_argument('config')
  p.add_argument('--only', nargs='*', default=None)
  p.add_argument('--out', default='results/ts')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--set', nargs='*', default=[], help='TSConfig overrides key=value (JSON values)')
  args = p.parse_args()
  spec = json.load(open(args.config))['experiments']
  overrides = {k: json.loads(v) for k, v in (kv.split('=', 1) for kv in args.set)}
  for name in args.only or list(spec):
    exp = spec[name]
    grid = exp.get('grid', {})
    keys = list(grid)
    summary = []
    for values in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
      point = dict(zip(keys, values))
      cfg = TSConfig(**(exp['base'] | point | overrides))
      tag = '_'.join(f'{k}{v:g}' if isinstance(v, (int, float)) else f'{k}{v}' for k, v in point.items()) or 'run'
      out_dir = os.path.join(args.out, name, tag)
      if os.path.exists(os.path.join(out_dir, 'result.json')):
        r = json.load(open(os.path.join(out_dir, 'result.json')))
      else:
        print(f'=== {name} {tag} {dc.asdict(cfg)}', flush=True)
        r = run_teacher_student(cfg, out_dir, eval_every=exp.get('eval_every', 1000), device=args.device)
      summary.append((r['final_test_mse'], r['final_rel_mse'], tag))
    for mse, rel, tag in sorted(summary):
      print(f'{name} {tag}: test_mse={mse:.3e} rel={rel:.4f}', flush=True)


if __name__ == '__main__':
  main()
