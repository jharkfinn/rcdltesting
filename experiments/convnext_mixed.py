"""Is the neuron basis privileged for WordNet coherence? Units against random directions, ConvNeXt and InceptionV1.

E0d's rule for a 'mixed' unit: among its k_img strongest validation images (spatial maximum of the response), the two
largest WordNet clusters (average linkage over the WordNet 3.0 class distances, merged while the mean distance is at most
`cut`) each hold at least `min_group` images. Here the rule is applied, per block of a ConvNeXt-SwiGLU checkpoint, to
  units     the SwiGLU products u_n (n = 1 .. U)
  dense     Q random unit-norm Gaussian directions w in the U-dimensional unit space, score max over positions of u . w
  pairs     Q random pairs (u_i + u_j) / sqrt 2 (two units packed into one direction)
and, for comparison, to the channels of InceptionV1 layers from E0's validation statistics (`--inception_stats`, per-image
spatial maxima of every channel). Superposition in the Toy Models sense predicts that the units are about as mixed as
random directions; a privileged, clean basis predicts units much less mixed than random directions. Also reported per
block: the median number of distinct classes among the k_img images, and the median WordNet distance between the two
largest clusters of the mixed ones.

Usage: python3 -m experiments.convnext_mixed --run <run dir> --val <validation parquet dir> [--step N] [--q 64]
       [--inception_stats val_stats.npz] [--out file.json]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.inception_mixed import groups_of
from rcdl.convnext import build_convnext
from rcdl.imagenet import val_batches


@torch.no_grad()
def val_stats(model, val_dir, dirs, batch=250, limit=None):
  """Per validation image: spatial max of every unit and of every random direction, per block. dirs: {block: (U, Q)}"""
  dev = next(model.parameters()).device
  named = model.blocks()
  feats = {}
  hooks = [b.norm.register_forward_hook(lambda m, i, o, n=n, b=b: feats.__setitem__(n, F.silu(b.pw_gate(o)) * b.pw_value(o)))
           for n, b in named]
  out = {n: {'units': [], **{k: [] for k in dirs[n]}} for n, _ in named}
  labels = []
  for x, y in val_batches(val_dir, batch, dev):
    feats.clear()
    model(x)
    labels.append(y.cpu())
    if limit and sum(len(t) for t in labels) >= limit:
      break
    for n, _ in named:
      u = feats[n].flatten(1, 2).float()                               # (B, positions, U)
      out[n]['units'].append(u.amax(1).half().cpu())
      for k, W in dirs[n].items():
        out[n][k].append((u @ W).amax(1).half().cpu())
  for h in hooks:
    h.remove()
  return torch.cat(labels).numpy(), {n: {k: torch.cat(v).float().numpy() for k, v in d.items()} for n, d in out.items()}


def mixed_stats(M, labels, dist, k_img, cut, min_group):
  """M (N images, V directions) -> share of mixed directions, median distinct classes, median cluster distance."""
  n_mixed, n_cls, dists = 0, [], []
  for v in range(M.shape[1]):
    idx = np.argsort(-M[:, v])[:k_img]
    n_cls.append(len(np.unique(labels[idx])))
    g = groups_of(idx, labels, dist, cut, min_group)
    if g is not None:
      n_mixed += 1
      dists.append(g[2])
  return {'n': int(M.shape[1]), 'mixed': n_mixed / M.shape[1], 'distinct_classes_median': float(np.median(n_cls)),
          'cluster_distance_median': float(np.median(dists)) if dists else None}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--val', required=True)
  p.add_argument('--step', type=int, default=None, help='checkpoint step (default: the last)')
  p.add_argument('--q', type=int, default=64, help='random directions of each kind per block')
  p.add_argument('--k', type=int, default=100)
  p.add_argument('--cut', type=float, default=7.0)
  p.add_argument('--min_group', type=int, default=15)
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--inception_stats', default=None)
  p.add_argument('--inception_layers', default='mixed3b,mixed4a,mixed4b,mixed4c,mixed4d,mixed4e,mixed5a,mixed5b')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--limit', type=int, default=None, help='first N validation images only (smoke tests)')
  p.add_argument('--out', default=None)
  a = p.parse_args()
  dist = np.load(a.wordnet)['dist']
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  steps = sorted(int(f[-8:-3]) for f in glob.glob(os.path.join(a.run, 'model_[0-9]*.pt')))
  step = a.step if a.step is not None else steps[-1]
  model = build_convnext(cfg['size']).to(dev).eval()
  model.load_state_dict(torch.load(os.path.join(a.run, f'model_{step:05d}.pt'), map_location=dev))
  g = torch.Generator().manual_seed(a.seed)
  dirs = {}
  for n, b in model.blocks():
    U = b.pw_gate.out_features
    Wd = torch.randn(U, a.q, generator=g)
    Wd /= Wd.norm(dim=0, keepdim=True)
    Wp = torch.zeros(U, a.q)
    for j in range(a.q):
      i1, i2 = torch.randperm(U, generator=g)[:2].tolist()
      Wp[i1, j] = Wp[i2, j] = 2 ** -0.5
    dirs[n] = {'dense': Wd.to(dev), 'pairs': Wp.to(dev)}
  labels, st = val_stats(model, a.val, dirs, limit=a.limit)
  print('validation images', len(labels), flush=True)
  res = {'run': a.run, 'step': step, 'k_img': a.k, 'cut': a.cut, 'min_group': a.min_group, 'q': a.q, 'convnext': {}}
  for n, d in st.items():
    res['convnext'][n] = {k: mixed_stats(M, labels, dist, a.k, a.cut, a.min_group) for k, M in d.items()}
    print(n, json.dumps({k: (v['mixed'], v['distinct_classes_median']) for k, v in res['convnext'][n].items()}), flush=True)
  if a.inception_stats:
    from experiments.inception_e0 import concat_stats
    ist = np.load(a.inception_stats)
    res['inception'] = {}
    for layer in a.inception_layers.split(','):
      try:
        M = concat_stats(ist, layer, 'max')
      except Exception as e:                           # a layer missing from the statistics file
        print('skip', layer, type(e).__name__, flush=True)
        continue
      res['inception'][layer] = {'units': mixed_stats(M, ist['labels'], dist, a.k, a.cut, a.min_group)}
      print(layer, json.dumps(res['inception'][layer]), flush=True)
  json.dump(res, open(a.out or os.path.join(a.run, f'mixed_basis_{step:05d}.json'), 'w'), indent=1)


if __name__ == '__main__':
  main()
