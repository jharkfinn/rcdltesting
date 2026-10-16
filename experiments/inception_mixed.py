"""E0d: across whole InceptionV1 layers, are the two halves of a mixed unit read through the same inputs?

E0b found that the essay's two "shared pose" units, 4d:241 and 4d:295, read their vehicle and their dog images through
almost orthogonal mixed4c input patterns (cosine 0.08 and 0.12; the top 10 vehicle channels write 14% and 29% of the
dog response). This repeats the test on every unit of mixed4c, mixed4d, mixed4e and mixed5b, with the baselines and
references a review of the first version showed to be necessary (2026-10-16).

Mixed units. A unit's 100 strongest validation images (spatial maximum, the E0 statistics) are labelled with their
classes; the classes are clustered by WordNet 3.0 tree distance (average linkage, cut at `cut`, default 7). A unit is
mixed when its two largest clusters A and B each hold at least `min_group` (15) of the 100 images ("distant" pair,
WordNet distance above the cut). Inside the largest cluster, the same rule at `sub_cut` (default 6) gives a "related"
pair when one exists (for example terriers and cats, which merge at 7).

Shares. Between two consecutive mixed layers the network is piecewise linear, so the pre-ReLU response z of a unit at
a position splits exactly over the C channels of the previous mixed layer, z = sum_c s_c + r, with
s_c = sum_{y,x} x[c,y,x] dz/dx[c,y,x] and r the bias part. With sA, sB the mean share vectors of the two groups:

  overlap       cos(sA, sB)
  within        mean cosine between the means of random halves of A (and of B): the sampling ceiling
  class_ref     mean cosine between two random halves of the classes of A (and of B): what different classes of one
                WordNet cluster give, the reference for "one feature on two classes"
  mass          sum_c min(pA_c, pB_c) with p the positive part of a mean share vector normalized to sum 1 (shared
                input mass; less sensitive than the cosine to one dominant channel), and mass_within its ceiling
  baseline      the same overlap (and mass) for 32 other units of the same branch, read at this unit's strongest
                positions on the same images: what any reader of this branch gets from these two image groups,
                because the groups' previous-layer activations are already similar or not
  excess        overlap - median baseline, with a 95% bootstrap interval (images of A and B resampled, unit and
                baseline together); rank = share of baseline readers below the unit

Verdict per unit and measure: 'separate' when the excess interval is below 0 and rank <= 0.05 (the unit reads the two
groups through more separate inputs than an arbitrary reader would), 'shared' when the interval is above 0 and
rank >= 0.95, else 'undetermined'. A unit counts as separate or shared only when the cosine and mass verdicts agree.

Visual axis. WordNet distance is not visual similarity (car wheel and beach wagon are 12 apart). When `--embed`
is given, every validation crop is embedded with an image model independent of InceptionV1 (DINOv2 ViT-S/14, CLS
token, centred over the 50,000 images), and vis_ratio = cos(mean embedding of A, of B) / its half-split ceiling.
Visually near pairs (both readings predict shared inputs) are a positive control; the visually far pairs decide.

Reading. A 'separate' verdict fits packing two features (Zoom In) and also one variable built in this layer from
group-specific channels (e.g. a pose made from wheel channels on cars and leg channels on dogs; E0b's OR variant),
so the separate fraction is an upper bound on packing. A 'shared' verdict supports a feature shared at the previous
layer. The rank correlation with WordNet distance is descriptive only: source-layer similarity falls with distance, so
it is negative under both readings.

Summaries: per layer, branch type and pair kind (distant / related), with 95% intervals from a bootstrap over the
pairs of global WordNet clusters (units that mix the same two clusters are not independent).

Usage: python3 -m experiments.inception_mixed --stats val_stats.npz --val <validation parquet dir>
       --imagenet <dir with classes.py> --weights inception5h.pth [--embed dinov2] --out results/e0/mixed.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.inception_e0 import channel_index, class_names, concat_stats, iter_val, load_model, preprocess
from experiments.inception_paths import Probe, clear_hooks, cos

SOURCE = {'mixed4c': 'mixed4b', 'mixed4d': 'mixed4c', 'mixed4e': 'mixed4d', 'mixed5b': 'mixed5a'}
ANIMAL_MAX = 397                                    # ImageNet classes 0-397 are animals


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


def global_clusters(D: np.ndarray, cut: float):
  """The same average-linkage rule over all classes (Lance-Williams update); used to key units for the cluster
  bootstrap."""
  D = D.astype(float).copy()
  n = len(D)
  np.fill_diagonal(D, np.inf)
  size, lab = np.ones(n), np.arange(n)
  while True:
    i, j = np.unravel_index(np.argmin(D), D.shape)
    if not np.isfinite(D[i, j]) or D[i, j] > cut:
      break
    D[i] = D[:, i] = (size[i] * D[i] + size[j] * D[j]) / (size[i] + size[j])
    D[i, i] = np.inf
    D[j] = D[:, j] = np.inf
    size[i] += size[j]
    lab[lab == j] = i
  return np.unique(lab, return_inverse=True)[1]


def image_cache(val_dir, device, batch=250):
  """All validation center crops, preprocessed as in the E0 statistics (x * 255 - 117), fp16 on the device."""
  out = None
  for raw, _ in iter_val(val_dir, batch):
    x = preprocess(raw, device).half()
    if out is None:
      out = torch.empty((50000,) + tuple(x.shape[1:]), dtype=torch.float16, device=device)
      n = 0
    out[n:n + len(x)] = x
    n += len(x)
  return out[:n]


@torch.no_grad()
def embed_dinov2(cache, device, batch=500):
  """Centred DINOv2 ViT-S/14 CLS embeddings (N, 384) of the cached crops (an image model independent of InceptionV1)."""
  from transformers import AutoModel
  m = AutoModel.from_pretrained('facebook/dinov2-small').to(device).eval()
  mean = torch.tensor([0.485, 0.456, 0.406], device=device)[:, None, None]
  std = torch.tensor([0.229, 0.224, 0.225], device=device)[:, None, None]
  out = []
  for i in range(0, len(cache), batch):
    x = ((cache[i:i + batch].float() + 117.0) / 255.0 - mean) / std
    out.append(m(pixel_values=x).last_hidden_state[:, 0].float().cpu())
  e = torch.cat(out)
  return e - e.mean(0)


def groups_of(idx, labels, dist, cut, min_group):
  """(A, B, mean WordNet distance) for the two largest clusters of the images idx, or None."""
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
  return A, B, float(dist[np.ix_(cls[A], cls[B])].mean())


def related_pair(idx, labels, dist, cut, sub_cut, min_group):
  """Two sub-clusters (at sub_cut) inside the largest cluster at `cut`, as positions into idx, or None."""
  cls = labels[idx]
  uniq = np.unique(cls)
  lab = average_linkage(dist[np.ix_(uniq, uniq)].astype(float), cut)
  cl_of = dict(zip(uniq.tolist(), lab.tolist()))
  img = np.array([cl_of[c] for c in cls.tolist()])
  sel = np.where(img == np.bincount(img).argmax())[0]
  g = groups_of(idx[sel], labels, dist, sub_cut, min_group)
  return None if g is None else (sel[g[0]], sel[g[1]], g[2])


def branch_shares(model, probe, x, chs, pos_from):
  """Exact shares (len(chs) tensors of shape (B, C_source)) of several channels of the probed branch, each at the
  strongest position of channel `pos_from` (one forward; one backward per channel)."""
  model(x)
  flat = probe.z.flatten(2)
  pos = flat[:, pos_from].argmax(1)
  out, zs = [], None
  for k, c in enumerate(chs):
    z = flat[:, c].gather(1, pos[:, None])[:, 0]
    if c == pos_from:
      zs = z.detach().cpu()
    (g,) = torch.autograd.grad(z.sum(), probe.x, retain_graph=k < len(chs) - 1)
    out.append((g * probe.x).sum((2, 3)).detach().double().cpu())
  return out, zs


def pmass(m):
  p = m.clamp(min=0)
  return p / p.sum().clamp(min=1e-12)


def mass(a, b):
  return float(torch.minimum(pmass(a), pmass(b)).sum())


def half_stat(S, stat, rng, n=50):
  out = []
  for _ in range(n):
    p = torch.from_numpy(rng.permutation(len(S)))
    out.append(stat(S[p[:len(S) // 2]].mean(0), S[p[len(S) // 2:]].mean(0)))
  return float(np.mean(out))


def class_split(S, cls, stat, rng, n=50):
  u = np.unique(cls)
  if len(u) < 2:
    return float('nan')
  out = []
  for _ in range(n):
    m = torch.from_numpy(np.isin(cls, rng.permutation(u)[:len(u) // 2]))
    out.append(stat(S[m].mean(0), S[~m].mean(0)))
  return float(np.mean(out))


def matched(sA, sB, stat, rng, n=50):
  """The between-group statistic at half-group sizes (comparable with within and class_ref)."""
  out = []
  for _ in range(n):
    a = torch.from_numpy(rng.permutation(len(sA))[:len(sA) // 2])
    b = torch.from_numpy(rng.permutation(len(sB))[:len(sB) // 2])
    out.append(stat(sA[a].mean(0), sB[b].mean(0)))
  return float(np.mean(out))


def rows_stat(name, X, Y):
  """cos or mass between matching rows of X and Y (R, C)."""
  if name == 'cos':
    return (X * Y).sum(1) / (X.norm(dim=1) * Y.norm(dim=1)).clamp(min=1e-12)
  px, py = X.clamp(min=0), Y.clamp(min=0)
  px, py = px / px.sum(1, keepdim=True).clamp(min=1e-12), py / py.sum(1, keepdim=True).clamp(min=1e-12)
  return torch.minimum(px, py).sum(1)


def measure(S, Sv, A, B, cls, rng, n_boot=200):
  """All per-pair quantities for the unit's shares S (N, C) and the baseline readers' shares Sv (list of (N, C))."""
  rec = {'nA': int(len(A)), 'nB': int(len(B))}
  T = torch.stack([S] + list(Sv))                              # (1 + readers, N, C); row 0 is the unit
  for name, stat in (('cos', cos), ('mass', mass)):
    vals = rows_stat(name, T[:, A].mean(1), T[:, B].mean(1))
    ov, base = float(vals[0]), vals[1:].numpy()
    boot = []
    for _ in range(n_boot):
      a, b = rng.choice(A, len(A)), rng.choice(B, len(B))
      v = rows_stat(name, T[:, a].mean(1), T[:, b].mean(1))
      boot.append(float(v[0]) - float(v[1:].median()))
    ci = np.percentile(boot, [2.5, 97.5])
    rank = float((base < ov).mean())
    within = 0.5 * (half_stat(S[A], stat, rng) + half_stat(S[B], stat, rng))
    class_ref = float(np.nanmean([class_split(S[A], cls[A], stat, rng), class_split(S[B], cls[B], stat, rng)]))
    rec[name] = {'overlap': ov, 'within': within, 'class_ref': class_ref,
                 'overlap_matched': matched(S[A], S[B], stat, rng),
                 'baseline': float(np.median(base)), 'baseline_q10_q90': np.percentile(base, [10, 90]).tolist(),
                 'excess': ov - float(np.median(base)), 'excess_ci': ci.tolist(), 'rank': rank,
                 'verdict': 'separate' if ci[1] < 0 and rank <= 0.05 else 'shared' if ci[0] > 0 and rank >= 0.95
                 else 'undetermined'}
  rec['verdict'] = rec['cos']['verdict'] if rec['cos']['verdict'] == rec['mass']['verdict'] else 'undetermined'
  pa, pb = pmass(S[A].mean(0)), pmass(S[B].mean(0))
  rec['top1_mass'] = [float(pa.max()), float(pb.max())]
  return rec


def cluster_ci(keys, stat, rng, n=2000):
  ks = sorted(set(keys))
  groups = [np.array([i for i, k in enumerate(keys) if k == kk]) for kk in ks]
  out = []
  for _ in range(n):
    pick = np.concatenate([groups[j] for j in rng.integers(0, len(ks), len(ks))])
    out.append(stat(pick))
  return np.nanpercentile(out, [2.5, 97.5]).round(3).tolist()


def spearman(a, b):
  ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
  return float(np.corrcoef(ra, rb)[0, 1]) if len(a) > 2 else float('nan')


def summarize(rows, rng):
  if not rows:
    return {'n': 0}
  keys = [tuple(r['cluster_pair']) for r in rows]
  v = np.array([r['verdict'] for r in rows])
  exc = np.array([r['cos']['excess'] for r in rows])
  out = {'n': len(rows), 'n_cluster_pairs': len(set(keys))}
  for verdict in ('separate', 'shared', 'undetermined'):
    out[f'frac_{verdict}'] = float((v == verdict).mean())
    out[f'frac_{verdict}_ci95'] = cluster_ci(keys, lambda p, verdict=verdict: float((v[p] == verdict).mean()), rng)
  out['cos_excess_quartiles'] = np.percentile(exc, [25, 50, 75]).round(3).tolist()
  out['cos_excess_median_ci95'] = cluster_ci(keys, lambda p: float(np.median(exc[p])), rng)
  om = np.array([r['cos']['overlap_matched'] for r in rows])
  cr = np.array([r['cos']['class_ref'] for r in rows])
  ok = np.isfinite(cr)
  out['frac_overlap_below_class_ref'] = float((om[ok] < cr[ok]).mean()) if ok.any() else float('nan')
  out['class_ref_quartiles'] = np.percentile(cr[ok], [25, 50, 75]).round(3).tolist() if ok.any() else []
  d = np.array([r['wordnet_dist'] for r in rows])
  out['spearman_cos_overlap_vs_wordnet_dist (descriptive)'] = spearman(np.array([r['cos']['overlap'] for r in rows]), d)
  if rows[0].get('vis_ratio') is not None:
    vr = np.array([r['vis_ratio'] for r in rows])
    far = vr <= np.median(vr)
    out['vis_ratio_median'] = float(np.median(vr))
    for name, m in (('visually_far', far), ('visually_near', ~far)):
      out[f'{name}_n'] = int(m.sum())
      for verdict in ('separate', 'shared'):
        out[f'{name}_frac_{verdict}'] = float((v[m] == verdict).mean()) if m.any() else float('nan')
  return out


def run(stats_path, val_dir, imagenet_dir, weights, wordnet, out, layers, k_img=100, cut=7.0, sub_cut=6.0,
        min_group=15, n_base=32, embed=None, seed=0, cache=None, stats=None, max_units=None):
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  st = np.load(stats_path) if stats is None else stats
  labels = st['labels']
  names = class_names(imagenet_dir) if imagenet_dir else [str(i) for i in range(1000)]
  wn = np.load(wordnet)
  dist, lca = wn['dist'], wn['lca_depth']
  gcl = global_clusters(dist, cut)
  rng = np.random.default_rng(seed)
  model = load_model(weights, dev)
  for p in model.parameters():
    p.requires_grad_(False)
  t0 = time.time()
  if cache is None:
    cache = image_cache(val_dir, dev)
  assert len(cache) == len(labels)
  print(f'cached {len(cache)} images in {time.time() - t0:.0f}s', flush=True)
  emb = None
  if embed == 'dinov2':
    try:
      emb = embed_dinov2(cache, dev)
      print('embedded with DINOv2', tuple(emb.shape), flush=True)
    except Exception as e:                            # no transformers or no network: run without the visual axis
      print('no visual axis:', type(e).__name__, str(e)[:200], flush=True)
      embed = None
  res = {'k_img': k_img, 'cut': cut, 'sub_cut': sub_cut, 'min_group': min_group, 'n_base': n_base, 'embed': embed,
         'layers': {}}
  for layer in layers:
    M = concat_stats(st, layer, 'max')
    branch_of = [channel_index(model, layer, v) for v in range(M.shape[1])]
    rows = []
    for u in range(M.shape[1] if max_units is None else min(max_units, M.shape[1])):
      idx = np.argsort(-M[:, u])[:k_img]
      pairs = []
      g = groups_of(idx, labels, dist, cut, min_group)
      if g is not None:
        pairs.append(('distant',) + g)
      g = related_pair(idx, labels, dist, cut, sub_cut, min_group)
      if g is not None:
        pairs.append(('related',) + g)
      if not pairs:
        continue
      branch, ch = branch_of[u]
      same = [v for v in range(M.shape[1]) if v != u and branch_of[v][0] == branch]
      readers = [branch_of[int(v)][1] for v in rng.choice(same, min(n_base, len(same)), replace=False)]
      probe = Probe(model, branch, source=SOURCE[layer])
      Ss, z = branch_shares(model, probe, cache[torch.from_numpy(idx).to(dev)].float(), [ch] + readers, ch)
      clear_hooks(model)
      S, Sv = Ss[0], Ss[1:]
      cls = labels[idx]
      top = lambda G: [names[c] for c, _ in sorted(((c, int((cls[G] == c).sum())) for c in set(cls[G].tolist())),
                                                    key=lambda t: -t[1])[:4]]
      for kind, A, B, dAB in pairs:
        rec = {'unit': u, 'branch': branch.split('_')[1], 'pair': kind, 'wordnet_dist': dAB,
               'lca_depth': float(lca[np.ix_(cls[A], cls[B])].mean()),
               'superclass': '-'.join(sorted('animal' if np.mean(cls[G] <= ANIMAL_MAX) > 0.5 else 'other' for G in (A, B))),
               'cluster_pair': sorted([int(gcl[np.bincount(cls[A]).argmax()]), int(gcl[np.bincount(cls[B]).argmax()])]),
               'classes_A': top(A), 'classes_B': top(B), 'z_mean_A': float(z[A].mean()), 'z_mean_B': float(z[B].mean())}
        rec.update(measure(S, Sv, A, B, cls, rng))
        if emb is not None:
          E = emb[torch.from_numpy(idx)]
          vo = cos(E[A].mean(0), E[B].mean(0))
          vw = 0.5 * (half_stat(E[A], cos, rng) + half_stat(E[B], cos, rng))
          rec['vis_overlap'], rec['vis_within'], rec['vis_ratio'] = vo, vw, vo / vw if vw > 0 else float('nan')
        rows.append(rec)
    summ = {}
    for kind in ('distant', 'related'):
      sub = [r for r in rows if r['pair'] == kind]
      summ[kind] = summarize(sub, rng)
      summ[kind]['by_branch'] = {b: summarize([r for r in sub if r['branch'] == b], rng)
                                 for b in sorted({r['branch'] for r in sub})}
    summ['units'] = int(M.shape[1])
    res['layers'][layer] = {'summary': summ, 'units': rows}
    brief = {k: {kk: summ[k].get(kk) for kk in ('n', 'n_cluster_pairs', 'frac_separate', 'frac_shared',
                                                  'cos_excess_quartiles', 'frac_overlap_below_class_ref')}
             for k in ('distant', 'related')}
    print(json.dumps({layer: brief}), f'{time.time() - t0:.0f}s', flush=True)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  json.dump(res, open(out, 'w'), indent=1)
  print('saved', out)
  return res


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
  p.add_argument('--sub_cut', type=float, default=6.0)
  p.add_argument('--min_group', type=int, default=15)
  p.add_argument('--n_base', type=int, default=32)
  p.add_argument('--embed', default=None, choices=[None, 'dinov2'])
  a = p.parse_args()
  run(a.stats, a.val, a.imagenet, a.weights, a.wordnet, a.out, a.layers.split(','), a.k, a.cut, a.sub_cut,
      a.min_group, a.n_base, a.embed)
