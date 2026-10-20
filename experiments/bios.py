"""Synthetic-biography runs on the Gemma 3 style decoder (article's language experiment; see rcdl/bios.py).

Usage: python -m experiments.bios experiments/configs/bios.json --out results/bios [--only name ...] [--set k=v ...]
       [--ckpt_dir DIR --ckpt_every STEPS]
Config format: {"defaults": {BiosConfig fields}, "experiments": {name: {BiosConfig fields}}}. Writes
<out>/<name>/result.json and prints a summary (plateau end = first evaluation with attribute loss below 0.9 times the
no-knowledge baseline; steps to 50% / 90% attribute accuracy; final accuracy).
"""

import argparse
import dataclasses as dc
import json
import os

import torch

from rcdl.bios import BiosConfig, run_bios


def main():
  p = argparse.ArgumentParser()
  p.add_argument('config')
  p.add_argument('--only', nargs='*', default=None)
  p.add_argument('--out', default='results/bios')
  p.add_argument('--set', nargs='*', default=[], help='BiosConfig overrides key=value (JSON values)')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--ckpt_dir', default=None, help='keep a resumable pre-training checkpoint of each run here '
                 '(<ckpt_dir>/ckpt_bios_<name>_*.py, rcdl/ckpt.py)')
  p.add_argument('--ckpt_every', type=int, default=1000)
  args = p.parse_args()
  spec = json.load(open(args.config))
  defaults, exps = spec.get('defaults', {}), spec['experiments']
  overrides = {k: json.loads(v) for k, v in (s.split('=', 1) for s in args.set)}
  rows = []
  for name in args.only or list(exps):
    out_dir = os.path.join(args.out, name)
    cfg = BiosConfig(**(defaults | exps[name] | overrides))
    done = os.path.join(out_dir, 'result.json')
    r = json.load(open(done)) if os.path.exists(done) else None
    defaults_now = {f.name: f.default for f in dc.fields(BiosConfig)}
    if r is not None and any(r['cfg'].get(k, defaults_now[k]) != v for k, v in dc.asdict(cfg).items()):
      # stored result of a different configuration (fields added later count with their default): rerun
      print(f'{name}: stored result has a different configuration, rerunning', flush=True)
      r = None
    if r is None:
      r = run_bios(cfg, out_dir, args.device, ckpt=os.path.join(args.ckpt_dir, f'ckpt_bios_{name}') if args.ckpt_dir else None,
                   ckpt_every=args.ckpt_every)
    rows.append((name, r))
  for name, r in rows:
    print(f"{name}: plateau_end={r['plateau_end_step']} acc50={r['acc50_step']} acc90={r['acc90_step']} "
          f"final_acc={r['final_attr_acc']:.4f} trained_params={r['trained_params']:,}"
          + (f" old_acc_after_ft={r['old_acc_after_finetune']:.4f} new_acc={r['new_acc_after_finetune']:.4f}"
             if 'old_acc_after_finetune' in r else ''), flush=True)


if __name__ == '__main__':
  main()
