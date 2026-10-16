"""What each class's bias writes do to a unit's response, with the local derivative of the SwiGLU product per class.

convnext_bias_signal.py orients every push by one sign per unit (the value sign on the unit's top-20 classes at the end).
That sign is the sign of du/dv only where SiLU(g) > 0, and of du/dg only where SiLU'(g) > 0 (g > -1.278). In stage 1 about
half of the pole responses run through the negative lobe of the gate (g < 0, v < 0, u = SiLU(g) v > 0), so this script
replaces the one sign with the derivative at each class's own pre-activations.

For block b, interval I = (t1, t2], unit n and classes k, c (class means at the strongest position p*, at t1):
  dv[k, n], dg[k, n]   class k's value and gate bias writes in I (exact: no bias decay)
  a_v[n, c] = SiLU(Gm[n, c])                 du/dv at class c's mean pre-activations
  a_g[n, c] = SiLU'(Gm[n, c]) Vm[n, c]       du/dg at class c's mean pre-activations
  self[k, n] = dv[k, n] a_v[n, k] + dg[k, n] a_g[n, k]
      first-order change of class k's own response on unit n from class k's own bias writes
  pole[k, n] = dv[k, n] mean_{c in P_n} a_v[n, c] + dg[k, n] mean_{c in P_n} a_g[n, c]
      first-order change of the mean response of the unit's top-20 classes P_n (by z-score at t1) from class k's writes
Roles per unit: top20 (P_n), WordNet relatives of P_n outside it, rest. Per role: mean / mean |.| over the unit's 1,000
classes (a per-unit scale; convnext_bias_signal.py divides by the mean over the whole block, so its numbers are on a
different scale), and the share of positive entries. Units are typed by their top-20 classes at t1:
'DN' if most have Gm < 0 and Vm < 0 (the negative lobe), 'PP' if most have both positive; each type is also reported
alone. 'oriented' repeats convnext_bias_signal.py's value-map measure (one sign per unit, from the end) for comparison,
and 'flip' is the share of units whose value sign on P_n at t1 differs from that end sign.

Assumptions: the derivative at the class-mean pre-activations stands for the mean derivative over the class's probe
images (SiLU is not linear, so this is first order and approximate); the strongest position p* is where the response is
read, while a bias write moves every position.

  python3 -m experiments.convnext_bias_self --run results/e1r_tiny_adamw_s1
  python3 -m experiments.convnext_bias_self --npz e1_cb.npz --rdir e1R --out e1_bias_self.json   (extracted E1 parts)
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

G_MIN = -1.2785                                    # SiLU'(g) = 0: SiLU's minimum


def silu(g):
  return g / (1 + np.exp(-g))


def dsilu(g):
  s = 1 / (1 + np.exp(-g))
  return s * (1 + g * (1 - s))


def block_self(dg, dv, R1, G1, V1, rel, sign_T, top_n=20):
  """dg, dv: (K, U) bias writes; R1, G1, V1: (U, C) response, gate and value class means at t1; rel: (C, C) bool;
  sign_T: (U,) value sign on the top-20 classes at the end. Returns {group: {measure: value}}."""
  z = ((R1 - R1.mean(1, keepdims=True)) / (R1.std(1, keepdims=True) + 1e-12))
  live = np.ptp(R1, 1) > 1e-3 * np.abs(R1).max()
  av, ag = silu(G1), dsilu(G1) * V1                                          # (U, C)
  top = np.argsort(-z, 1)[:, :top_n]                                         # (U, 20)
  self_e = dv * av.T + dg * ag.T                                             # (K, U)
  pa_v = np.take_along_axis(av, top, 1).mean(1)                              # (U,)
  pa_g = np.take_along_axis(ag, top, 1).mean(1)
  pole_v, pole_g = dv * pa_v[None], dg * pa_g[None]
  pole = pole_v + pole_g
  gt, vt = np.take_along_axis(G1, top, 1), np.take_along_axis(V1, top, 1)
  dn = ((gt < 0) & (vt < 0)).mean(1)
  pp = ((gt > 0) & (vt > 0)).mean(1)
  kind = np.where(dn > 0.5, 'DN', np.where(pp > 0.5, 'PP', 'mixed'))
  sign1 = np.sign(vt.mean(1))
  sign1[sign1 == 0] = 1
  oriented = dv * sign_T[None]
  sc = {k: np.abs(m).mean(0) + 1e-30 for k, m in (('self', self_e), ('pole', pole), ('pole_v', pole_v),
                                                  ('pole_g', pole_g), ('oriented', oriented))}
  per_unit = []
  for n in np.flatnonzero(live):
    inT = np.zeros(R1.shape[1], bool)
    inT[top[n]] = True
    relm = rel[top[n]].any(0) & ~inT
    rest = ~inT & ~relm
    rec = {'kind': kind[n], 'dn': dn[n], 'g_below_min': float((gt[n] < G_MIN).mean()),
           'flip': float(sign1[n] * sign_T[n] < 0)}
    for name, mask in (('top20', inT), ('relatives', relm), ('rest', rest)):
      if not mask.any():
        continue
      for m, M in (('self', self_e), ('pole', pole), ('pole_v', pole_v), ('pole_g', pole_g), ('oriented', oriented)):
        rec[f'{m}_{name}'] = M[mask, n].mean() / sc[m][n]
      rec[f'self_pos_{name}'] = float((self_e[mask, n] > 0).mean())
      rec[f'pole_pos_{name}'] = float((pole[mask, n] > 0).mean())
    rec['net_pole'] = pole[:, n].sum() / sc['pole'][n]                       # all classes' writes on the pole, net
    per_unit.append(rec)
  out = {}
  for grp in ('all', 'DN', 'PP'):
    rows = [r for r in per_unit if grp == 'all' or r['kind'] == grp]
    if not rows:
      continue
    keys = sorted({k for r in rows for k in r if k != 'kind'})
    out[grp] = {'units': len(rows), **{k: float(np.nanmean([r.get(k, np.nan) for r in rows])) for k in keys}}
  return out


def load_npz(npz, rdir):
  z = np.load(npz)
  steps = [int(s) for s in z['steps']]
  arrays = {k.split('|')[0]: z[k] for k in z.files if k.endswith('|Cb')}       # decompressed once
  cb = {s: {k: v[i] for k, v in arrays.items()} for i, s in enumerate(steps)}
  atl = {s: torch.load(os.path.join(rdir, f'R_{s:05d}.pt'), map_location='cpu', weights_only=False) for s in steps}
  return steps, cb, atl


def load_run(run):
  from experiments.convnext_bias_signal import load_cb
  steps = sorted(int(f[-8:-3]) for f in glob.glob(os.path.join(run, 'atlas_[0-9]*.pt')))
  steps = [s for s in steps if os.path.exists(os.path.join(run, f'attr_{s:05d}.pt'))]
  cb = {s: load_cb(os.path.join(run, f'attr_{s:05d}.pt')) for s in steps}
  atl = {s: torch.load(os.path.join(run, f'atlas_{s:05d}.pt'), map_location='cpu', weights_only=False) for s in steps}
  return steps, cb, atl


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', default=None)
  p.add_argument('--npz', default=None, help='extracted bias parts (steps, <map>|Cb arrays of shape (T, K, U))')
  p.add_argument('--rdir', default=None, help='directory of R_{step}.pt (per block R, Vm, Gm) for --npz')
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--from_step', type=int, default=0)
  p.add_argument('--out', default=None, help='default: bias_self.json in --run, or next to --npz')
  a = p.parse_args()
  if not (a.run or a.npz):
    p.error('give --run, or --npz with --rdir')
  if a.npz and not a.rdir:
    p.error('--rdir is required with --npz')
  steps, cb, atl = load_npz(a.npz, a.rdir) if a.npz else load_run(a.run)
  rel = np.load(a.wordnet)['dist'] <= 4
  np.fill_diagonal(rel, False)
  last = atl[steps[-1]]
  res = {'steps': steps, 'blocks': {}}
  for b in sorted(last, key=lambda n: [int(t) for t in n.replace('stage', '').split('.block')]):
    RT, VT = last[b]['R'].numpy(), last[b]['Vm'].numpy()
    sign_T = np.sign(np.take_along_axis(VT, np.argsort(-RT, 1)[:, :20], 1).mean(1))
    sign_T[sign_T == 0] = 1
    rows = []
    for t1, t2 in zip(steps[:-1], steps[1:]):
      if t1 < a.from_step:
        continue
      d = {m: cb[t2][f'{b}.pw_{m}'] - cb[t1][f'{b}.pw_{m}'] for m in ('gate', 'value')}
      A1 = atl[t1][b]
      rows.append({'t1': t1, 't2': t2, **block_self(d['gate'], d['value'], A1['R'].numpy(), A1['Gm'].numpy(),
                                                     A1['Vm'].numpy(), rel, sign_T)})
    res['blocks'][b] = rows
    r = rows[-1]['all']
    print(b, {k: round(r[k], 3) for k in ('dn', 'flip', 'oriented_top20', 'pole_top20', 'self_top20', 'pole_rest')},
          flush=True)
  out = a.out or os.path.join(a.run or os.path.dirname(os.path.abspath(a.npz)), 'bias_self.json')
  json.dump(res, open(out, 'w'), indent=1)
  print('BIAS_SELF_DONE', flush=True)


if __name__ == '__main__':
  main()
