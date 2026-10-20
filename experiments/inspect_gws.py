"""Trains one single-pass GWS model and reports how its stored rows are used (supports the β/η symmetry argument).

For each layer: share of rows whose weight w was clipped to 0; quantiles of w; and, over test queries q, the share of
the denominator Σ_j w_j e^{z_j(q)} held by the B initial rows D_0 (if this share is near 0, scaling all new w and ṽ
jointly leaves the output unchanged, so only β/η matters).

Usage: python -m experiments.inspect_gws [--config experiments/configs/repro_1epoch.json --name gws --seed 1]
"""

import argparse
import json
import math
import os

import torch

from rcdl.data import load_cifar10_tensors
from rcdl.layers import RCDLConfig, rms_norm
from rcdl.train import run_rcdl_experiment


@torch.no_grad()
def layer_inputs(model, x):
  """Queries q^(l) = RMSNorm(x^(l)) seen by each RCDL layer for inputs x (eval mode)."""
  qs, h = [], x
  for layer in model.layers:
    q = rms_norm(h)
    qs.append(q)
    h = h + model.act(layer(q))
  qs.append(rms_norm(h))
  return qs


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--config', default='experiments/configs/repro_1epoch.json')
  p.add_argument('--name', default='gws')
  p.add_argument('--seed', type=int, default=1)
  p.add_argument('--out', default='results/inspect_gws.json')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  args = p.parse_args()
  spec = json.load(open(args.config))
  cfg = RCDLConfig(**(spec['experiments'][args.name]['cfg'] | {'global_init_seed': args.seed}))
  data, _ = load_cifar10_tensors('data', 'parquet', args.device)
  res = run_rcdl_experiment(cfg, data, eval_every=10 ** 9, eval_subset=1000, keep_model=True)
  model = res.pop('model')
  model.eval()
  test_x = data[2][:1000]
  qs = layer_inputs(model, test_x)
  report = {'config': spec['experiments'][args.name]['cfg'], 'final_test_acc': res['final_test_acc'], 'layers': {}}
  for q, (name, layer) in zip(qs, model.rcdl_layers().items()):
    m, B = layer.size, cfg.batch_size
    w = layer.weights[:m, 0]
    z = q @ layer.keys[:m].T / (cfg.temperature * math.sqrt(q.shape[-1]))
    z = z - z.max(-1, keepdim=True).values
    contrib = torch.exp(z) * w[None, :]
    d0_share = (contrib[:, :B].sum(-1) / contrib.sum(-1)).cpu()
    new_w = w[B:]
    report['layers'][name] = {
        'rows': m, 'frac_w_zero': float((new_w == 0).float().mean()),
        'w_quantiles_nonzero': [float(v) for v in torch.quantile(new_w[new_w > 0].float().cpu(),
                                                                  torch.tensor([0.1, 0.5, 0.9, 0.99]))],
        'd0_denominator_share_median': float(d0_share.median()), 'd0_denominator_share_p90': float(d0_share.quantile(0.9))}
    print(name, report['layers'][name], flush=True)
  os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
  json.dump(report, open(args.out, 'w'), indent=1)


if __name__ == '__main__':
  main()
