"""Single-pass nonparametric baselines on CIFAR-10 (paper Fig. 2 right: kNN ~35.5%, LS-SVM ~58.5%, read off the plot).

Usage: python -m experiments.baselines --out results/baselines [--n-train 50000] [--device cuda]
Memory: exact KRR on all 50,000 images needs the 50k x 50k Gram matrix (10 GB fp32) plus its eigenvectors
(another 10 GB): use a 40 GB GPU, or --n-train 20000 on smaller devices.

Selection: like the paper, we report the best configuration on the test set ('best_test'). Because that
overstates accuracy, we also select on test images [0, 5000) and report on [5000, 10000) ('heldout').
"""

import argparse
import json
import os
import time

import torch

from rcdl.baselines import gram, kernel_fn, knn_accuracy, krr_solve, predict
from rcdl.data import load_cifar10_tensors


def summarize(scores_a, scores_b, scores_all):
  """scores_*: {config: accuracy} on test halves A, B and the full test set."""
  best_cfg = max(scores_all, key=scores_all.get)
  sel_cfg = max(scores_a, key=scores_a.get)
  return {'best_test': scores_all[best_cfg], 'best_test_cfg': best_cfg,
          'heldout': scores_b[sel_cfg], 'heldout_cfg': sel_cfg, 'all': scores_all}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--out', default='results/baselines')
  p.add_argument('--data', default='data')
  p.add_argument('--source', default='parquet')
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--n-train', type=int, default=50000)
  p.add_argument('--skip', nargs='*', default=[], help='any of: knn krr_rbf nw_exp krr_exp')
  args = p.parse_args()
  os.makedirs(args.out, exist_ok=True)
  (train_x, train_y, test_x, test_y), _ = load_cifar10_tensors(args.data, args.source, args.device)
  X, y = train_x[:args.n_train], train_y[:args.n_train]
  Y = torch.nn.functional.one_hot(y, 10).float()
  half = len(test_x) // 2
  acc = lambda f, sl=slice(None): (f[sl].argmax(-1) == test_y[sl]).float().mean().item()
  results = {'n_train': len(X), 'device': args.device}

  def save():
    with open(os.path.join(args.out, f'baselines_n{len(X)}.json'), 'w') as fh:
      json.dump(results, fh, indent=1)

  if 'knn' not in args.skip:
    t = time.time()
    a = knn_accuracy(X, y, test_x[:half], test_y[:half])
    b = knn_accuracy(X, y, test_x[half:], test_y[half:])
    full = {k: (a[k] + b[k]) / 2 for k in a}
    results['knn'] = summarize(a, b, full) | {'sec': time.time() - t}
    print('knn', {k: v for k, v in results['knn'].items() if k != 'all'}, flush=True)
    save()

  for kname, sweep, lams in (('rbf', [0.25, 0.5, 1.0, 2.0, 3.0, 5.0], [0.001, 0.01, 0.1, 0.3, 1.0, 3.0, 10.0]),
                             ('exp', [1.0, 3.0, 6.0, 10.0, 30.0], [0.001, 0.01, 0.1, 1.0, 10.0, 100.0])):
    if f'krr_{kname}' in args.skip and (kname == 'rbf' or 'nw_exp' in args.skip):
      continue
    sa, sb, sall, nw_a, nw_b, nw_all = {}, {}, {}, {}, {}, {}
    t = time.time()
    for h in sweep:
      k = kernel_fn(kname, X, gamma_mult=h, temperature=h)
      if kname == 'exp' and 'nw_exp' not in args.skip:   # Nadaraya-Watson vote with the RCDL first-layer kernel
        f = predict(k, X, Y, test_x, normalize=True)
        nw_a[f'T={h}'], nw_b[f'T={h}'], nw_all[f'T={h}'] = acc(f, slice(0, half)), acc(f, slice(half, None)), acc(f)
      if f'krr_{kname}' in args.skip:
        continue
      K = gram(k, X)
      alphas = krr_solve(K, Y, lams)
      del K
      for lam, alpha in alphas.items():
        f = predict(k, X, alpha, test_x)
        key = f'{"gamma_mult" if kname == "rbf" else "T"}={h},lambda={lam}'
        sa[key], sb[key], sall[key] = acc(f, slice(0, half)), acc(f, slice(half, None)), acc(f)
      del alphas
      if args.device == 'cuda':
        torch.cuda.empty_cache()
      print(f'krr_{kname} {h}: best so far {max(sall.values()):.4f} ({time.time() - t:.0f}s)', flush=True)
    if sall:
      results[f'krr_{kname}'] = summarize(sa, sb, sall) | {'sec': time.time() - t}
      print(f'krr_{kname}', {k: v for k, v in results[f'krr_{kname}'].items() if k != 'all'}, flush=True)
    if nw_all:
      results['nw_exp'] = summarize(nw_a, nw_b, nw_all)
      print('nw_exp', {k: v for k, v in results['nw_exp'].items() if k != 'all'}, flush=True)
    save()


if __name__ == '__main__':
  main()
