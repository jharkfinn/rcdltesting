"""E0d: across whole InceptionV1 layers, are the halves of a mixed unit driven by shared or by separate inputs?

E0b found that the essay's two "shared pose" units, 4d:241 and 4d:295, take their vehicle and their dog responses
from almost disjoint mixed4c channels (cosine 0.08 and 0.12 between the two input patterns, against 0.99 between
random halves of one group). This repeats that test on every unit of mixed4c, mixed4d, mixed4e and mixed5b.

Mixed units. A unit's 100 strongest validation images (spatial maximum, the E0 statistics) are labelled with their
classes; the classes are clustered by WordNet 3.0 tree distance (average linkage, cut at `cut`, default 7: dog breeds
or car types stay together, dogs and cars do not). A unit is mixed when its two largest clusters each hold at least
`min_group` (15) of the 100 images. The two clusters' relatedness is the image-weighted mean WordNet distance between
their classes.

Shares. Between two consecutive mixed layers the network is piecewise linear, so the unit's pre-ReLU response z at its
strongest position splits exactly over the C channels of the previous mixed layer, z = sum_c s_c + r, with
s_c = sum_{y,x} x[c,y,x] dz/dx[c,y,x] and r the bias part (experiments/inception_paths.shares). For the image groups A
and B with mean share vectors sA and sB (each of length C):

  overlap      = cos(sA, sB)
  within       = mean over 50 random half splits of A, and of B, of the cosine between the two halves' means
  ratio        = overlap / within (about 1: one input pattern for both groups; about 0: separate inputs)
  cross_top10  = share of B's mean z written by A's ten strongest channels (and the reverse, averaged)

Summary per layer: the number of mixed units, the distribution of the ratio, and its rank correlation with the
WordNet distance between the two groups (the essay predicts that related groups share inputs, "generalization").

Usage: python3 -m experiments.inception_mixed --stats val_stats.npz --val <validation parquet dir>
       --imagenet <dir with classes.py> --weights inception5h.pth --out results/e0/mixed.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.inception_e0 import channel_index, class_names, concat_stats, iter_val, load_model, preprocess
from experiments.inception_paths import Probe, clear_hooks, cos, shares

SOURCE = {'mixed4c': 'mixed4b', 'mixed4d': 'mixed4c', 'mixed4e': 'mixed4d', 'mixed5b': 'mixed5a'}


def average_linkage(D: np.ndarray, cut: float):
  """Cluster labels for a small distance matrix D (n, n): merge the two clusters with the smallest mean pairwise
  distance while it is at most `cut`."""
  clusters = [[i] for i in range(len(D))]
  while len(clusters) > 1:
    best, pair = None, None
    for a in range(len(clusters)):
      for b in range(a + 1, len(clusters)):
        d = D[np.ix_(clusters[a], clusters[b])].mean()
        if best is None or d < best:
          best, pair = d, (a, b)
    if best > cut:
      break
    a, b = pair
    clusters[a] += clusters[b]
    del clusters[b]
  lab = np.empty(len(D), dtype=int)
  for k, c in enumerate(clusters):
    lab[c] = k
  return lab


def image_cache(val_dir, device, batch=250):
  """All validation center crops, preprocessed as in the E0 statistics (x * 255 - 117), fp16 on the device."""
  xs = []
  for raw, _ in iter_val(val_dir, batch):
    xs.append(preprocess(raw, device).half())
  return torch.cat(xs)


def groups_of(idx, labels, dist, cut, min_group):
  cls = labels[idx]
  uniq = np.unique(cls)
  lab = average_linkage(dist[np.ix_(uniq, uniq)].astype(float), cut)
  cl_of = dict(zip(uniq.tolist(), lab.tolist()))
  img_cl = np.array([cl_of[c] for c in cls.tolist()])
  sizes = np.bincount(img_cl)
  order = np.argsort(-sizes)
  if len(order) < 2 or sizes[order[1]] < min_group:
    return None
  A, B = np.where(img_cl == order[0])[0], np.where(img_cl == order[1])[0]
  dAB = float(dist[np.ix_(cls[A], cls[B])].mean())
  return A, B, dAB


def half_cos(S, rng, n=50):
  out = []
  for _ in range(n):
    p = torch.from_numpy(rng.permutation(len(S)))
    out.append(cos(S[p[:len(S) // 2]].mean(0), S[p[len(S) // 2:]].mean(0)))
  return float(np.mean(out))


def top_cross(sa, sb, k=10):
  top = torch.argsort(-sa.mean(0))[:k]
  return float(sb[:, top].sum(1).mean() / sb.sum(1).mean())


def spearman(a, b):
  ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
  return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float('nan')


def run(stats_path, val_dir, imagenet_dir, weights, wordnet, out, layers, k_img=100, cut=7.0, min_group=15, seed=0):
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  st = np.load(stats_path)
  labels = st['labels']
  names = class_names(imagenet_dir)
  dist = np.load(wordnet)['dist']
  rng = np.random.default_rng(seed)
  model = load_model(weights, dev)
  for p in model.parameters():
    p.requires_grad_(False)
  t0 = time.time()
  cache = image_cache(val_dir, dev)
  assert len(cache) == len(labels)
  print(f'cached {len(cache)} images in {time.time() - t0:.0f}s', flush=True)
  res = {'k_img': k_img, 'cut': cut, 'min_group': min_group, 'layers': {}}
  for layer in layers:
    M = concat_stats(st, layer, 'max')
    units, n_mixed = [], 0
    for u in range(M.shape[1]):
      idx = np.argsort(-M[:, u])[:k_img]
      g = groups_of(idx, labels, dist, cut, min_group)
      if g is None:
        continue
      n_mixed += 1
      A, B, dAB = g
      branch, ch = channel_index(model, layer, u)
      probe = Probe(model, branch, source=SOURCE[layer])
      S, Z, R = shares(model, probe, cache[torch.from_numpy(idx).to(dev)].float(), ch)
      clear_hooks(model)
      S = S.double().cpu()
      sA, sB = S[A], S[B]
      within = 0.5 * (half_cos(sA, rng) + half_cos(sB, rng))
      overlap = cos(sA.mean(0), sB.mean(0))
      top = lambda G: [names[c] for c, _ in sorted(((c, int((labels[idx][G] == c).sum())) for c in set(labels[idx][G].tolist())),
                                                    key=lambda t: -t[1])[:4]]
      units.append({'unit': u, 'nA': len(A), 'nB': len(B), 'wordnet_dist': dAB, 'overlap': overlap, 'within': within,
                    'ratio': overlap / within if within > 0 else float('nan'),
                    'cross_top10': 0.5 * (top_cross(sA, sB) + top_cross(sB, sA)),
                    'z_mean_A': float(Z[torch.from_numpy(A)].mean()), 'z_mean_B': float(Z[torch.from_numpy(B)].mean()),
                    'bias_share': float((R / Z).mean()), 'classes_A': top(A), 'classes_B': top(B)})
    r = np.array([x['ratio'] for x in units])
    d = np.array([x['wordnet_dist'] for x in units])
    summ = {'units': int(M.shape[1]), 'mixed': n_mixed,
            'ratio_quartiles': np.percentile(r, [25, 50, 75]).round(3).tolist() if len(r) else [],
            'share_ratio_below_0.5': float((r < 0.5).mean()) if len(r) else float('nan'),
            'share_ratio_above_0.8': float((r > 0.8).mean()) if len(r) else float('nan'),
            'spearman_ratio_vs_wordnet_dist': spearman(r, d),
            'ratio_median_related (dist <= 10)': float(np.median(r[d <= 10])) if (d <= 10).any() else float('nan'),
            'ratio_median_unrelated (dist > 10)': float(np.median(r[d > 10])) if (d > 10).any() else float('nan')}
    res['layers'][layer] = {'summary': summ, 'units': units}
    print(json.dumps({layer: summ}), f'{time.time() - t0:.0f}s', flush=True)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  json.dump(res, open(out, 'w'), indent=1)
  print('saved', out)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--stats', required=True)
  p.add_argument('--val', required=True)
  p.add_argument('--imagenet', required=True)
  p.add_argument('--weights', required=True)
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--out', default='results/e0/mixed.json')
  p.add_argument('--layers', default='mixed4c,mixed4d,mixed4e,mixed5b')
  p.add_argument('--k', type=int, default=100)
  p.add_argument('--cut', type=float, default=7.0)
  p.add_argument('--min_group', type=int, default=15)
  a = p.parse_args()
  run(a.stats, a.val, a.imagenet, a.weights, a.wordnet, a.out, a.layers.split(','), a.k, a.cut, a.min_group)
