"""E2 / E3 on the E1 writer atlas: who wrote each unit's pole classes, in what order they entered, and whether they
entered through relatives (recursive generalization) or by their own writes.

Input: atlas_{step}.pt files of experiments/convnext_atlas.py (one per checkpoint, pole classes fixed at the final
checkpoint) and the WordNet 3.0 distances of rcdl/imagenet_hierarchy.py. For block b, unit n and pole class c, the
mean value pre-activation on class c at checkpoint t splits exactly as

  vbar_t[n, c] = I_t[n, c] + sum_k V_t[n, c, k],

with V_t[n, c, k] the part written by training class k (mixup: an example's part split between its two classes by
lam, 1 - lam) and I_t the decayed initial weights' part. With d(c, k) the WordNet tree distance:

  own share       V[n, c, c] / vbar
  relative share  sum over k != c with d(c, k) <= `near` of V[n, c, k] / vbar
  other share     the rest, without I

Growth: the positive pole of a unit is the set of its `poles` strongest classes at the final checkpoint whose mean
response R (the SwiGLU product at the strongest position) exceeds tau times the unit's largest |R|; a class enters at
the first checkpoint where its R exceeds tau times that checkpoint's largest |R|. For every class that is not the first
to enter (E3, "head start"): its own share at entry; and (E2, "spillover") whether more than half of vbar at entry was
written by the classes already in the pole, and, if so, whether the strongest of those writers is a relative of the
entering class, against the rate expected if it were drawn at random from the earlier members. Also the WordNet
distance from each entering class to the nearest earlier member, against the same for random pole-sized class sets.

Usage: python3 -m experiments.convnext_growth --run <run dir> [--tau 0.3] [--near 4]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def load_run(run):
  files = sorted(glob.glob(os.path.join(run, 'atlas_[0-9]*.pt')))
  steps = [int(f[-8:-3]) for f in files]
  return steps, [torch.load(f) for f in files]


def analyze(steps, atlases, dist, tau=0.3, near=4, seed=0):
  rng = np.random.default_rng(seed)
  n_classes = dist.shape[0]
  final = atlases[-1]
  out = {}
  for b in final:
    P = final[b]['poles'].shape[1] // 2
    rec = {'own': [], 'relative': [], 'other': [], 'init': [], 'entries': 0, 'head_start_own_at_entry': [],
           'spill': 0, 'spill_from_relative': 0, 'spill_from_relative_null': 0.0,
           'nearest_earlier_dist': [], 'nearest_earlier_dist_null': []}
    R_T = final[b]['R']                                           # (units, classes)
    for n in range(R_T.shape[0]):
      scale = R_T[n].abs().max()
      if scale <= 0:
        continue
      cls = final[b]['poles'][n, :P].tolist()                    # strongest classes, by final R
      pole = [(j, c) for j, c in enumerate(cls) if R_T[n, c] > tau * scale]
      if not pole:
        continue
      W = final[b]['value_writers'][n].float()                    # (2P, K)
      I = final[b]['value_init'][n]
      for j, c in pole:
        vbar = float(W[j].sum() + I[j])
        if vbar <= 0:                                             # the gate opened it with a negative value: skip
          continue
        rel = [k for k in range(n_classes) if k != c and dist[c, k] <= near]
        rec['own'].append(float(W[j, c]) / vbar)
        rec['relative'].append(float(W[j, rel].sum()) / vbar)
        rec['other'].append((float(W[j].sum()) - float(W[j, c]) - float(W[j, rel].sum())) / vbar)
        rec['init'].append(float(I[j]) / vbar)
      entry = {}
      for j, c in pole:
        for ti, at in enumerate(atlases):
          Rt = at[b]['R'][n]
          if Rt[c] > tau * Rt.abs().max().clamp(min=1e-12):
            entry[(j, c)] = ti
            break
      order = sorted(entry, key=lambda jc: entry[jc])
      for i, (j, c) in enumerate(order[1:], 1):
        te = entry[(j, c)]
        earlier = [cc for (jj, cc) in order[:i] if entry[(jj, cc)] < te]
        if not earlier:
          continue
        at = atlases[te][b]
        Wt, It = at['value_writers'][n, j].float(), float(at['value_init'][n, j])
        vbar = float(Wt.sum()) + It
        if vbar <= 0:
          continue
        rec['entries'] += 1
        rec['head_start_own_at_entry'].append(float(Wt[c]) / vbar)
        d_near = min(int(dist[c, e]) for e in earlier)
        rec['nearest_earlier_dist'].append(d_near)
        rand = rng.choice([k for k in range(n_classes) if k != c], size=len(earlier), replace=False)
        rec['nearest_earlier_dist_null'].append(int(dist[c, rand].min()))
        if float(Wt[earlier].sum()) > 0.5 * vbar:
          rec['spill'] += 1
          top = earlier[int(torch.argmax(Wt[earlier]))]
          rec['spill_from_relative'] += int(dist[c, top] <= near)
          rec['spill_from_relative_null'] += float(np.mean([dist[c, e] <= near for e in earlier]))
    out[b] = rec
  return out


def summarize(out):
  rows = {}
  for b, r in out.items():
    med = lambda x: float(np.median(x)) if len(x) else float('nan')
    rows[b] = {'pole_entries_measured': len(r['own']), 'own': med(r['own']), 'relative': med(r['relative']),
               'other': med(r['other']), 'init': med(r['init']), 'entries': r['entries'],
               'own_share_at_entry_median': med(r['head_start_own_at_entry']),
               'entered_with_own_share_below_half': float(np.mean(np.array(r['head_start_own_at_entry']) < 0.5))
               if r['head_start_own_at_entry'] else float('nan'),
               'spill': r['spill'], 'spill_from_relative': r['spill_from_relative'],
               'spill_from_relative_null': r['spill_from_relative_null'],
               'nearest_earlier_dist_median': med(r['nearest_earlier_dist']),
               'nearest_earlier_dist_null_median': med(r['nearest_earlier_dist_null'])}
  return rows


def coherence(R, dist, k=10, cut=7.0, rng=None):
  """Per unit: the share of its k strongest classes (by R) in their largest WordNet cluster (average linkage cut at
  `cut`, as in experiments/inception_mixed): 1 = all k classes are close relatives. With rng: the same for k random
  classes per unit (chance level)."""
  from experiments.inception_mixed import average_linkage
  out = []
  for n in range(R.shape[0]):
    cls = (rng.choice(R.shape[1], k, replace=False) if rng is not None
           else torch.argsort(R[n], descending=True)[:k].numpy())
    lab = average_linkage(dist[np.ix_(cls, cls)].astype(float), cut)
    out.append(np.bincount(lab).max() / k)
  return np.array(out)


def static_shares(rec, dist, tau=0.3, near=4):
  """One checkpoint's own poles (prelim atlas): median own / relative / other / init shares of the positive pole."""
  P = rec['poles'].shape[1] // 2
  R = rec['R']
  own, rel, oth, ini = [], [], [], []
  for n in range(R.shape[0]):
    scale = R[n].abs().max()
    for j, c in enumerate(rec['poles'][n, :P].tolist()):
      if scale <= 0 or R[n, c] <= tau * scale:
        continue
      W, I = rec['value_writers'][n, j].float(), float(rec['value_init'][n, j])
      vbar = float(W.sum()) + I
      if vbar <= 0:
        continue
      r = [k for k in np.where(dist[c] <= near)[0] if k != c]
      own.append(float(W[c]) / vbar)
      rel.append(float(W[r].sum()) / vbar)
      ini.append(I / vbar)
      oth.append(1 - own[-1] - rel[-1] - ini[-1])
  med = lambda x: float(np.median(x)) if x else float('nan')
  return {'n': len(own), 'own': med(own), 'relative': med(rel), 'other': med(oth), 'init': med(ini)}


def prelim(run, dist, tau, near):
  """Per preliminary checkpoint (each with its own poles): static shares and WordNet coherence, by stage."""
  rng = np.random.default_rng(0)
  rows = []
  for f in sorted(glob.glob(os.path.join(run, 'atlas_prelim_*.pt'))):
    step, at = int(f[-8:-3]), torch.load(f)
    for stage in sorted({b.split('.')[0] for b in at}):
      blocks = [b for b in at if b.startswith(stage + '.')]
      sh = [static_shares(at[b], dist, tau, near) for b in blocks]
      coh = np.concatenate([coherence(at[b]['R'], dist) for b in blocks])
      chance = np.concatenate([coherence(at[b]['R'], dist, rng=rng) for b in blocks])
      w = np.array([x['n'] for x in sh], float)
      avg = lambda k: float(np.nansum([x[k] * x['n'] for x in sh]) / max(w.sum(), 1))
      rows.append({'step': step, 'stage': stage, 'pole_entries': int(w.sum()), 'own': avg('own'),
                   'relative': avg('relative'), 'other': avg('other'), 'init': avg('init'),
                   'coherence_median': float(np.median(coh)), 'coherent_units_(>=0.8)': float((coh >= 0.8).mean()),
                   'coherent_units_chance': float((chance >= 0.8).mean())})
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--tau', type=float, default=0.3)
  p.add_argument('--near', type=int, default=4)
  p.add_argument('--out', default=None)
  p.add_argument('--prelim', action='store_true', help='per-checkpoint static view of atlas_prelim_*.pt')
  a = p.parse_args()
  dist = np.load(a.wordnet)['dist']
  if a.prelim:
    rows = prelim(a.run, dist, a.tau, a.near)
    print('step stage | pole entries: own / relative / other / init (block medians, weighted) | coherence median,'
          ' share of units >= 0.8 (chance)')
    for r in rows:
      print(f"{r['step']:5d} {r['stage']} | {r['pole_entries']:5d}: {r['own']:.3f} / {r['relative']:.3f} / {r['other']:.3f}"
            f" / {r['init']:.3f} | {r['coherence_median']:.2f}, {r['coherent_units_(>=0.8)']:.2f} ({r['coherent_units_chance']:.2f})")
    json.dump(rows, open(a.out or os.path.join(a.run, 'growth_prelim.json'), 'w'), indent=1)
    return
  steps, atlases = load_run(a.run)
  rows = summarize(analyze(steps, atlases, dist, a.tau, a.near))
  print('block | poles: own / relative / other / init (median shares) | entries: own share at entry, share below 0.5 |'
        ' spillover: n, from relative (null) | nearest earlier member, WordNet distance (null)')
  for b, r in rows.items():
    print(f"{b:14s} | {r['own']:.2f} / {r['relative']:.2f} / {r['other']:.2f} / {r['init']:.2f} | {r['entries']}:"
          f" {r['own_share_at_entry_median']:.2f}, {r['entered_with_own_share_below_half']:.2f} | {r['spill']},"
          f" {r['spill_from_relative']} ({r['spill_from_relative_null']:.1f}) | {r['nearest_earlier_dist_median']}"
          f" ({r['nearest_earlier_dist_null_median']})")
  json.dump({'steps': steps, 'tau': a.tau, 'near': a.near, 'blocks': rows},
            open(a.out or os.path.join(a.run, 'growth_summary.json'), 'w'), indent=1)


if __name__ == '__main__':
  main()
