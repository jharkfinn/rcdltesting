"""Synthetic-fact runs (article's biography experiment, minimal version; see rcdl/facts.py).

Usage: python -m experiments.facts experiments/configs/facts.json --out results/facts [--only name ...]
Config format: {"experiments": {name: {FactsConfig fields}}}. Writes <out>/<name>/result.json and prints a summary
(plateau end = first evaluation step with loss below 0.9 ln K; final fact accuracy).
"""

import argparse
import json
import os

import torch

from rcdl.facts import FactsConfig, run_facts


def main():
  p = argparse.ArgumentParser()
  p.add_argument('config')
  p.add_argument('--only', nargs='*', default=None)
  p.add_argument('--out', default='results/facts')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  args = p.parse_args()
  spec = json.load(open(args.config))['experiments']
  rows = []
  for name in args.only or list(spec):
    out_dir = os.path.join(args.out, name)
    if os.path.exists(os.path.join(out_dir, 'result.json')):
      r = json.load(open(os.path.join(out_dir, 'result.json')))
    else:
      r = run_facts(FactsConfig(**spec[name]), out_dir, args.device)
    rows.append((name, r['plateau_end_step'], r['final_acc'], r['params']))
  for name, plateau, acc, params in rows:
    print(f'{name}: plateau_end={plateau} final_acc={acc:.4f} trained_params={params:,}', flush=True)


if __name__ == '__main__':
  main()
