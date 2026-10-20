"""Class split against random-group split of the same training (runs with --rand_groups; atlas_*.pt and atlasR_*.pt).

Every part is linear in the per-example gradients: for a unit n and a probe class c, the part of group g is
W_g = sum over examples i in g of u_i, where u_i is example i's (optimizer-scaled, accumulated) contribution to the mean
pre-activation of class c at the unit's strongest position. Write u_i = mu_{y_i} + e_i (class mean plus within-class
deviation). With K groups of equal size m:
  class split   E_class = sum_k W_k^2  ~ m^2 sum_k mu_k^2 + K m s2
  random split  E_rand  = sum_g Wr_g^2 ~ K m^2 mubar^2 + K m s2          (mubar: mean of mu over classes, s2: within-class
                                                                          variance of u)
so E_class - E_rand ~ m^2 sum_k (mu_k - mubar)^2, the between-class energy, and f = 1 - E_rand / E_class is the share of
the class split that is class structure (the rest is within-class example noise). Under mixup each example's gradient
is split between two groups with lam and 1 - lam in both splits, so the same holds with these weights.

Per block, for each unit and each final positive-pole class c (as in convnext_growth_tests: the `poles` strongest classes
with R_T > tau * max |R_T|), for the value and the gate map:
  f            1 - E_rand / E_class (median over unit-class pairs; per checkpoint)
  gross_net    sum_g |W_g| / |vbar| in each split (vbar: the class mean of the value)
  own_z        W_own / sd(Wr) - mean(Wr) / sd(Wr): the own part in units of the random-group spread
  rel_z        (sum over relatives of W_k - n_rel mean(Wr)) / (sqrt(n_rel) sd(Wr))
  top10        share of the positive mass in the 10 largest positive groups, class and random split

Usage: python3 -m experiments.convnext_split_control --run <run dir> [--out file.json]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def load(run, prefix):
  files = sorted(glob.glob(os.path.join(run, f'{prefix}_[0-9]*.pt')))
  return [int(f[-8:-3]) for f in files], files


def top_share(W, k=10):
  pos = np.sort(np.clip(W, 0, None))[::-1]
  return float(pos[:k].sum() / pos.sum()) if pos.sum() > 0 else float('nan')


def analyze(run, dist, tau=0.3, near=4):
  steps, fc = load(run, 'atlas')
  steps_r, fr = load(run, 'atlasR')
  assert steps == steps_r, (steps, steps_r)
  final, final_r = torch.load(fc[-1]), torch.load(fr[-1])
  out = {}
  for b in final:
    P = final[b]['poles'].shape[1] // 2
    assert torch.equal(final[b]['poles'], final_r[b]['poles']), 'both atlases must use the same pole classes'
    R = final[b]['R'].numpy()
    pairs = []
    for n in range(R.shape[0]):
      scale = float(np.abs(R[n]).max())
      cand = final[b]['poles'][n, :P].tolist()
      pairs += [(n, j, c) for j, c in enumerate(cand) if scale > 0 and R[n, c] > tau * scale]
    rec = {'pairs': len(pairs), 'per_step': []}
    for si, (s, f1, f2) in enumerate(zip(steps, fc, fr)):
      A = final if si == len(steps) - 1 else torch.load(f1)
      Ar = final_r if si == len(steps) - 1 else torch.load(f2)
      row = {'step': s}
      for m in ('value', 'gate'):
        key = f'{m}_writers'
        if key not in A[b]:
          continue
        Wc, Wr = A[b][key].float().numpy(), Ar[b][key].float().numpy()
        init = A[b][f'{m}_init'].numpy()
        fs, gn_c, gn_r, oz, rz, t_c, t_r, chk = [], [], [], [], [], [], [], []
        for n, j, c in pairs:
          w, wr = Wc[n, j], Wr[n, j]
          chk.append(abs(w.sum() - wr.sum()) / max(np.abs(w).sum(), 1e-12))
          ec, er = float((w ** 2).sum()), float((wr ** 2).sum())
          if ec > 0:
            fs.append(1 - er / ec)
          net = abs(w.sum() + init[n, j])
          if net > 1e-9:
            gn_c.append(np.abs(w).sum() / net); gn_r.append(np.abs(wr).sum() / net)
          mu, sd = wr.mean(), wr.std()
          if sd > 0:
            oz.append((w[c] - mu) / sd)
            rel = [k for k in np.where(dist[c] <= near)[0] if k != c]
            if rel:
              rz.append((w[rel].sum() - len(rel) * mu) / (np.sqrt(len(rel)) * sd))
          t_c.append(top_share(w)); t_r.append(top_share(wr))
        q = lambda x: [round(float(v), 4) for v in np.nanpercentile(x, [25, 50, 75])] if len(x) else None
        row[m] = {'f': q(fs), 'gross_net_class': q(gn_c), 'gross_net_random': q(gn_r), 'own_z': q(oz), 'rel_z': q(rz),
                  'top10_class': q(t_c), 'top10_random': q(t_r), 'sum_mismatch_max': float(np.max(chk)) if chk else None}
      rec['per_step'].append(row)
    out[b] = rec
    last = rec['per_step'][-1]
    print(b, json.dumps({'pairs': rec['pairs'], 'final': last}), flush=True)
  return steps, out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--tau', type=float, default=0.3)
  p.add_argument('--near', type=int, default=4)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  dist = np.load(a.wordnet)['dist']
  steps, res = analyze(a.run, dist, a.tau, a.near)
  json.dump({'steps': steps, 'tau': a.tau, 'near': a.near, 'blocks': res},
            open(a.out or os.path.join(a.run, 'split_control.json'), 'w'), indent=1)


if __name__ == '__main__':
  main()
