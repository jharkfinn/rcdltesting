"""E8 growth analysis: who wrote each neuron's response to each feature, and how features entered a neuron's pole.

Input: the *_growth.npz files of experiments/toy_packing.py. For a tracked map k (ReLU 'neurons', or SwiGLU 'gate' and
'value') and checkpoint t, the exact decomposition gives the pre-activation of neuron h for a unit input of feature j as

  z_t[h, j] = I_t[j, h] + sum_w W_t[w, j, h],

where w runs over writer features (the training samples that contain w; a pair sample's part is split equally) and I is
the decayed initial weight's part. For each neuron's final positive pole (response R > tau * max over features):

  own share      W[j, j, h] / z[h, j]             written by samples of feature j itself
  family share   sum_{w in family(j), w != j} W   written by its relatives (spillover through similar inputs)
  other share    the rest                         written by unrelated features

and, for each pole member j other than the first to enter (entry = first checkpoint with R > tau * max), whether at its
entry its pre-activation was written mostly by the members that were already in the pole ("spillover entry"), and
whether those earlier members are its relatives. Recursive generalization predicts spillover entries from relatives;
packing of unrelated features predicts entries written by the new member itself.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def analyze_run(json_path: str, tau: float = 0.3):
  res = json.load(open(json_path))
  g = np.load(os.path.join(os.path.dirname(json_path), res['growth_file']))
  fam = np.array(res['family'])
  act = res['args'].get('act', 'relu')
  key = 'neurons' if act == 'relu' else 'value'          # SwiGLU: the value map carries the sign of the product
  R, W, I = g['R'], g[f'W_{key}'], g[f'I_{key}']        # (T, H, n), (T, n, n, H), (T, n, H)
  T, H, n = R.shape
  final = R[-1]
  out = {'own': [], 'family': [], 'other': [], 'entries': 0, 'spill_entries': 0, 'spill_from_family': 0,
         'spill_from_family_null': 0.0}
  for h in range(H):
    scale = np.abs(final[h]).max()
    if scale <= 0:
      continue
    pole = np.where(final[h] > tau * scale)[0]
    for j in pole:
      z = I[-1, j, h] + W[-1, :, j, h].sum()
      if z <= 0:
        continue
      own = W[-1, j, j, h]
      famw = W[-1, (fam == fam[j]) & (np.arange(n) != j), j, h].sum()
      out['own'].append(own / z)
      out['family'].append(famw / z)
      out['other'].append((z - own - famw - I[-1, j, h]) / z)
    entry = {}
    for j in pole:
      above = np.where(R[:, h, j] > tau * np.abs(R[:, h, :]).max(1).clip(min=1e-12))[0]
      if len(above):
        entry[j] = above[0]
    order = sorted(entry, key=lambda j: entry[j])
    for i, j in enumerate(order[1:], 1):
      t = entry[j]
      earlier = [e for e in order[:i] if entry[e] < t]
      if not earlier:
        continue
      z = I[t, j, h] + W[t, :, j, h].sum()
      if z <= 0:
        continue
      out['entries'] += 1
      by_earlier = W[t, earlier, j, h].sum()
      if by_earlier > 0.5 * z:
        out['spill_entries'] += 1
        top = earlier[int(np.argmax(W[t, earlier, j, h]))]
        out['spill_from_family'] += int(fam[top] == fam[j])
        out['spill_from_family_null'] += float(np.mean(fam[earlier] == fam[j]))
  return res['args'], out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--dir', default='results/toy_packing')
  p.add_argument('--tau', type=float, default=0.3)
  a = p.parse_args()
  cells = defaultdict(list)
  for f in sorted(glob.glob(os.path.join(a.dir, '*.json'))):
    if not os.path.exists(f[:-5] + '_growth.npz'):
      continue
    args, out = analyze_run(f, a.tau)
    cells[(args.get('act', 'relu'), args['rho'], args['m'], args['H'])].append(out)
  print('act rho m H runs | median shares of pole responses own / family / other | entries: spillover share,'
        ' of which from a relative (null)')
  for key in sorted(cells):
    rs = cells[key]
    own = np.concatenate([r['own'] for r in rs]) if rs else np.array([])
    famv = np.concatenate([r['family'] for r in rs])
    oth = np.concatenate([r['other'] for r in rs])
    ent = sum(r['entries'] for r in rs)
    sp = sum(r['spill_entries'] for r in rs)
    sf = sum(r['spill_from_family'] for r in rs)
    sfn = sum(r['spill_from_family_null'] for r in rs)
    print(f'{key[0]:6s} {key[1]:.1f} {key[2]:2d} {key[3]:2d} {len(rs):2d} | {np.median(own):.2f} / {np.median(famv):.2f}'
          f' / {np.median(oth):.2f} | {sp}/{ent} = {sp / max(ent, 1):.2f}, from relative {sf}/{sp}'
          f' ({sfn:.1f} expected)')


if __name__ == '__main__':
  main()
