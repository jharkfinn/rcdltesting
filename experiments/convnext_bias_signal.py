"""The per-class teaching signal on each unit, read exactly from the bias parts.

A tracked map's bias has input 1, so group k's bias part of unit n is

  Cb_t[k, n] = - sum over steps s <= t of lr_s * mhat_s[k, n] / (sqrt(vb_s[n]) + eps)

where mhat_s[k, n] is the bias-corrected momentum of group k's bias gradients (the summed teaching signal dL/dz_n over
all positions of group k's rows) and vb_s[n] the shared second moment. The factor 1 / (sqrt(vb_s[n]) + eps) is positive
and the same for every group, so within a unit the sign of a group's bias part, and its size relative to the other
groups, is the sign and relative size of that group's summed teaching signal, reweighted over time by positive factors.
There is no bias decay, so the increment dCb_I = Cb_t2 - Cb_t1 is exactly the group's writes inside interval
I = (t1, t2].

Per block, map and interval (pooled over units; units with a flat response profile are skipped):
- gross_net: sum_k |dCb[k, n]| / |sum_k dCb[k, n]| (median over units). For K independent zero-mean pushes of equal
  size this is about 0.8 sqrt(K) (25 for K = 1,000); larger values mean the pushes cancel more than chance.
- pushes by class role, oriented so that positive raises the unit's response on its pole (value map: the value sign on
  the unit's top-20 classes at the final checkpoint; gate map: the same sign, since du/dg = SiLU'(g) v): the mean push
  of the unit's top-20 classes at t1, of their WordNet relatives outside the top 20, and of the rest, each in units of
  the mean |push| over all classes.
- rho: Spearman correlation over classes of the oriented push with the class's response z-score at t1 (median over
  units).
- leader_share: for each class, the share of its total |push| over the block's units taken by the unit where its
  z-score is largest (median over classes; uniform = 1 / units).
With --split random (attrR_*.pt of a run trained with --rand_groups), the same pushes of 1,000 random groups of class
size give the noise scale: their mean |push| and gross_net are reported next to the class split's, and every class's
|push| is ranked among the random groups' |push| on the same unit (median percentile per role).

  python3 -m experiments.convnext_bias_signal --run results/e1r_tiny_adamw_s1 --out results/e1r_tiny_adamw_s1/bias_signal.json
"""

import argparse
import glob
import json
import os

import numpy as np
import torch


def spearman(a: np.ndarray, b: np.ndarray) -> float:
  ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
  return float(np.corrcoef(ra, rb)[0, 1])


def load_cb(path: str) -> dict:
  """{map: (K, units) fp32 bias parts} from a compact attribution file."""
  A = torch.load(path, map_location='cpu', weights_only=False)
  return {k: (r['Cb'].float() / r['Cb_scale']).numpy() for k, r in A.items() if r.get('Cb') is not None}


def block_rows(dg, dv, R1, sign, rel, rand=None):
  """Measures for one block and interval. dg, dv: (K, U) gate and value pushes; R1: (U, C) responses at t1;
  sign: (U,) orientation; rel: (C, C) bool relatives; rand: optional {'gate', 'value'}: (K, U) random-group pushes."""
  U = R1.shape[0]
  z = ((R1 - R1.mean(1, keepdims=True)) / (R1.std(1, keepdims=True) + 1e-12)).T          # (C, U)
  live = np.ptp(R1, 1) > 1e-3 * np.abs(R1).max()
  out = {}
  for m, d in (('gate', dg), ('value', dv)):
    o = d * sign[None]
    scale = np.abs(o).mean() + 1e-30
    gn = np.abs(d).sum(0) / (np.abs(d.sum(0)) + 1e-30)
    top = np.argsort(-z, 0)[:20]
    roles = {'top20': [], 'relatives': [], 'rest': []}
    pct = {'top20': [], 'relatives': [], 'rest': []}
    rho = []
    for n in np.flatnonzero(live):
      T = top[:, n]
      inT = np.zeros(len(z), bool)
      inT[T] = True
      relm = rel[T].any(0) & ~inT
      rest = ~inT & ~relm
      for name, mask in (('top20', inT), ('relatives', relm), ('rest', rest)):
        if mask.any():
          roles[name].append(o[mask, n].mean() / scale)
          if rand is not None:
            rr = np.sort(np.abs(rand[m][:, n]))
            pct[name].append(np.median(np.searchsorted(rr, np.abs(d[mask, n])) / len(rr)))
      rho.append(spearman(o[:, n], z[:, n]))
    lead = z.argmax(1)
    share = np.abs(d)[np.arange(len(z)), lead] / (np.abs(d).sum(1) + 1e-30)
    rec = {'gross_net': float(np.median(gn[live])), 'rho': float(np.nanmedian(rho)),
           'leader_share': float(np.median(share)), 'uniform_share': 1.0 / U,
           **{k: float(np.mean(v)) for k, v in roles.items()}}
    if rand is not None:
      r = rand[m]
      rec['rand_gross_net'] = float(np.median((np.abs(r).sum(0) / (np.abs(r.sum(0)) + 1e-30))[live]))
      rec['class_over_rand_abs'] = float(np.abs(d).mean() / (np.abs(r).mean() + 1e-30))
      rec.update({f'pct_vs_rand_{k}': float(np.median(v)) for k, v in pct.items()})
    out[m] = rec
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', required=True)
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--out', default=None)
  a = p.parse_args()
  steps = sorted(int(f[-8:-3]) for f in glob.glob(os.path.join(a.run, 'atlas_[0-9]*.pt')))
  steps = [s for s in steps if os.path.exists(os.path.join(a.run, f'attr_{s:05d}.pt'))]
  has_rand = all(os.path.exists(os.path.join(a.run, f'attrR_{s:05d}.pt')) for s in steps)
  rel = np.load(a.wordnet)['dist'] <= 4
  np.fill_diagonal(rel, False)
  atl = {s: torch.load(os.path.join(a.run, f'atlas_{s:05d}.pt'), map_location='cpu', weights_only=False) for s in steps}
  last = atl[steps[-1]]
  cb = {s: load_cb(os.path.join(a.run, f'attr_{s:05d}.pt')) for s in steps}
  cbr = {s: load_cb(os.path.join(a.run, f'attrR_{s:05d}.pt')) for s in steps} if has_rand else None
  res = {'steps': steps, 'random_null': has_rand, 'blocks': {}}
  for b in sorted(last, key=lambda n: [int(t) for t in n.replace('stage', '').split('.block')]):
    RT, VT = last[b]['R'].numpy(), last[b]['Vm'].numpy()
    topT = np.argsort(-RT, 1)[:, :20]
    sign = np.sign(np.take_along_axis(VT, topT, 1).mean(1))
    sign[sign == 0] = 1
    rows = []
    for t1, t2 in zip(steps[:-1], steps[1:]):
      d = {m: cb[t2][f'{b}.pw_{m}'] - cb[t1][f'{b}.pw_{m}'] for m in ('gate', 'value')}
      rnd = {m: cbr[t2][f'{b}.pw_{m}'] - cbr[t1][f'{b}.pw_{m}'] for m in ('gate', 'value')} if has_rand else None
      rows.append({'t1': t1, 't2': t2, **block_rows(d['gate'], d['value'], atl[t1][b]['R'].numpy(), sign, rel, rnd)})
    res['blocks'][b] = rows
    print(b, {m: {k: round(v, 3) for k, v in rows[-1][m].items()} for m in ('gate', 'value')}, flush=True)
  json.dump(res, open(a.out or os.path.join(a.run, 'bias_signal.json'), 'w'), indent=1)
  print('BIAS_SIGNAL_DONE', flush=True)


if __name__ == '__main__':
  main()
