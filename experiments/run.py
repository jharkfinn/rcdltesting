"""Runs one or more named CIFAR-10 experiments from a JSON config file, for one or more seeds.

Usage:
  python -m experiments.run experiments/configs/repro_1epoch.json --only gws rbf --seeds 1 2 3 --out results
  python -m experiments.run experiments/configs/repro_1epoch.json --only gws --set eta=0.0 --tag eta0

Config file format: {"defaults": {...run options...}, "experiments": {name: {"cfg": {...RCDLConfig fields...},
"run": {...run options...}}}}. Run options: eval_every, eval_subset, eval_batch_size, input_grad, max_steps.
Results go to <out>/<name>[_<tag>]/seed<k>/.
"""

import argparse
import dataclasses as dc
import json
import os

import torch

from rcdl.data import load_cifar10_tensors
from rcdl.layers import RCDLConfig
from rcdl.train import run_rcdl_experiment


def _parse_value(text: str):
  try:
    return json.loads(text)
  except json.JSONDecodeError:
    return text


def main():
  p = argparse.ArgumentParser()
  p.add_argument('config')
  p.add_argument('--only', nargs='*', default=None, help='experiment names to run (default: all)')
  p.add_argument('--seeds', nargs='*', type=int, default=None, help='overrides global_init_seed')
  p.add_argument('--set', nargs='*', default=[], help='cfg overrides key=value (JSON values)')
  p.add_argument('--run-set', nargs='*', default=[], help='run-option overrides key=value')
  p.add_argument('--tag', default='')
  p.add_argument('--out', default='results')
  p.add_argument('--data', default='data')
  p.add_argument('--source', default='auto')
  p.add_argument('--threads', type=int, default=None)
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--skip-done', action='store_true', help='skip runs whose summary.json exists')
  args = p.parse_args()

  if args.threads:
    torch.set_num_threads(args.threads)
  with open(args.config) as f:
    spec = json.load(f)
  data, train_u8 = load_cifar10_tensors(args.data, args.source, args.device)
  names = args.only or list(spec['experiments'])
  overrides = dict(kv.split('=', 1) for kv in args.set)
  run_overrides = dict(kv.split('=', 1) for kv in args.run_set)
  for name in names:
    exp = spec['experiments'][name]
    cfg_kwargs = dict(exp['cfg']) | {k: _parse_value(v) for k, v in overrides.items()}
    run_kwargs = dict(spec.get('defaults', {})) | exp.get('run', {}) | {k: _parse_value(v) for k, v in run_overrides.items()}
    for seed in (args.seeds or [cfg_kwargs.get('global_init_seed', 1)]):
      cfg = RCDLConfig(**(cfg_kwargs | {'global_init_seed': seed}))
      out_dir = os.path.join(args.out, name + (f'_{args.tag}' if args.tag else ''), f'seed{seed}')
      if args.skip_done and os.path.exists(os.path.join(out_dir, 'summary.json')):
        print(f'skip {out_dir}')
        continue
      print('=' * 72 + f'\n{name} seed={seed} -> {out_dir}\n{dc.asdict(cfg)}\n{run_kwargs}\n' + '=' * 72, flush=True)
      run_rcdl_experiment(cfg, data, out_dir=out_dir, train_u8=train_u8, **run_kwargs)


if __name__ == '__main__':
  main()
