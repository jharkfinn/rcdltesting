"""E2 / E3 decisive tests on the E1 writer atlas (the tests a review of the first analysis asked for, 2026-10-16).

Input: atlas_{step}.pt of experiments/convnext_atlas.py for a complete run (pole classes fixed at the final checkpoint),
and the WordNet 3.0 distances (rcdl/imagenet_hierarchy.py). Per block, over its units, for the final positive-pole
classes c of unit n (the `poles` strongest classes with R_T[n, c] > tau * max |R_T[n]|):

T1 order of entry against a within-pole null. Stable entry t_c = first checkpoint from which c stays among the unit's
   `poles` strongest classes to the end. Statistic: mean over entering classes (those with an earlier member) of the
   WordNet distance to the nearest earlier member, and the share whose nearest earlier member is a relative (d <= near).
   Null: the entry checkpoints permuted among the unit's own pole classes (plain, and within terciles of final R), so
   pole coherence is held fixed and only the order is tested. Power: the same test on a planted branch order (the
   observed entry times reassigned along a greedy nearest-neighbour WordNet chain from the strongest class).
   Recursive generalization predicts observed < permuted. Caveat found by the power check: with tied entry checkpoints
   only the classes with an earlier member are counted, and a planted branch order gives the latest times to the WordNet
   outliers of the pole, so this statistic can move the wrong way under branch order. Hence also a Mantel statistic,
   robust to ties: per unit (3 or more pole classes) the Pearson correlation over class pairs of |t_c - t_c'| with the
   WordNet distance d(c, c'), averaged over units; same permutations, same planted order. Any growth by WordNet
   neighbourhood (branch order, or relatives entering together) predicts a positive correlation.
T2 half-rise against distance. t50_c = first checkpoint with R_c(t) >= R_c(T) / 2; d_c = WordNet distance from c to the
   unit's strongest final class. Within-unit regression of t50 on d with log R_c(T) as a covariate; null: d permuted
   within units. Recursive generalization predicts a positive slope (classes far from the core rise later).
T3 relatives lead or follow. For each pole class with positive own and relative parts at the end: the first checkpoint
   at which its own part, and the sum of its relatives' parts, reach 25% and 50% of their final values; lag =
   t_own - t_rel (positive: relatives lead). Median and bootstrap interval over units.
T4 positive-mass composition at the end: own, relative and other shares of the positive mass sum_k max(W_k, 0), with the
   chance shares by count (1 / K, n_rel / K; K classes); gross / net = sum_k |W_k| / |vbar| (vbar: the class mean of the
   value); mean_rel_over_mean_abs_other = mean of W_k over relatives / mean of |W_k| over the other classes.
T5 knockout at entry (head start; needs gate and value writers): with class means, R'_c = SiLU(G_c - G_own) (V_c - V_own);
   the class 'fires without its own writes' if R'_c still exceeds the unit's 20th strongest R among other classes.
   Rate for entering classes at their entry checkpoint against incumbents (already in the pole) at the same checkpoint.
   The essay's head start predicts entering > incumbents. Approximation: SiLU(G_c) V_c of class means stands for the class
   mean of SiLU(g) v, and the strongest position is held fixed; the rate with own writes kept (no knockout) is reported as
   a calibration of the approximation (it is 1 if the approximation is exact).

Usage: python3 -m experiments.convnext_growth_tests --run <run dir> [--perm 500] [--out file.json]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def load(run):
  files = sorted(glob.glob(os.path.join(run, 'atlas_[0-9]*.pt')))
  return [int(f[-8:-3]) for f in files], [torch.load(f) for f in files]


def stable_entry(Rt, n, cls, P):
  """Rt (T, units, classes) -> entry checkpoint index per class in cls."""
  T = Rt.shape[0]
  tops = [set(np.argsort(-Rt[t, n])[:P].tolist()) for t in range(T)]
  out = []
  for c in cls:
    inside = [c in tops[t] for t in range(T)]
    out.append(next(t for t in range(T) if all(inside[t:])))
  return np.array(out)


def nearest_earlier(t, Dc, near):
  E = t[None, :] < t[:, None]
  has = E.any(1)
  d = np.where(E, Dc, np.inf).min(1)[has]
  return d, (d <= near)


def chain_order(cls, Dc, Rfinal, t):
  """Planted branch order: the observed entry times, sorted, reassigned along a greedy nearest-neighbour WordNet chain
  that starts at the strongest class."""
  n = len(cls)
  start = int(np.argmax(Rfinal))
  seq, left = [start], set(range(n)) - {start}
  while left:
    cur = seq[-1]
    nxt = min(left, key=lambda k: (Dc[seq, k].min(), k))
    seq.append(nxt)
    left.remove(nxt)
  tt = np.empty(n, int)
  tt[seq] = np.sort(t)
  return tt


def mantel(t, x, iu):
  """Pearson correlation over class pairs (iu) of |t_c - t_c'| with the pair distances x; nan if either is constant."""
  y = np.abs(t[:, None] - t[None, :])[iu].astype(float)
  if y.std() == 0 or x.std() == 0:
    return np.nan
  return float(np.corrcoef(x, y)[0, 1])


def first_reach(x, frac):
  """x (T,) cumulative trajectory; first index with x >= frac * x[-1] (x[-1] > 0)."""
  idx = np.where(x >= frac * x[-1])[0]
  return int(idx[0]) if len(idx) else len(x) - 1


def boot_ci(per_unit, rng, n=1000, stat=np.median):
  """per_unit: list of arrays (one per unit); bootstrap over units."""
  per_unit = [np.asarray(u) for u in per_unit if len(u)]
  if not per_unit:
    return [float('nan')] * 2
  vals = []
  for _ in range(n):
    pick = rng.integers(0, len(per_unit), len(per_unit))
    vals.append(stat(np.concatenate([per_unit[i] for i in pick])))
  return np.percentile(vals, [2.5, 97.5]).round(3).tolist()


def analyze(steps, atlases, dist, tau=0.3, near=4, n_perm=500, seed=0):
  rng = np.random.default_rng(seed)
  final = atlases[-1]
  out = {}
  for b in final:
    P = final[b]['poles'].shape[1] // 2
    Rt = np.stack([at[b]['R'].numpy() for at in atlases])          # (T, units, classes)
    T, U, K = Rt.shape
    t1 = {'obs': 0.0, 'cnt': 0, 'rel_obs': 0, 'perm': np.zeros(n_perm), 'perm_terc': np.zeros(n_perm),
          'rel_perm': np.zeros(n_perm), 'chain_obs': 0.0, 'chain_perm': np.zeros(n_perm),
          'm_obs': 0.0, 'm_n': 0, 'm_perm': np.zeros(n_perm), 'm_chain': 0.0}
    t2_rows = []                                                  # (unit, t50, d, logR)
    t3 = {'own25': [], 'rel25': [], 'own50': [], 'rel50': []}
    lag25, lag50 = [], []
    t4 = {'own': [], 'rel': [], 'other': [], 'own_chance': [], 'rel_chance': [], 'gross_net': [], 'mean_rel_over_mean_abs_other': []}
    t5 = {'enter_fire': 0, 'enter_n': 0, 'inc_fire': 0, 'inc_n': 0, 'enter_keep': 0, 'inc_keep': 0}
    has_gate = 'gate_writers' in final[b] and 'Gm' in final[b]
    for n in range(U):
      scale = float(np.abs(Rt[-1, n]).max())
      if scale <= 0:
        continue
      cand = final[b]['poles'][n, :P].tolist()
      js = [j for j, c in enumerate(cand) if Rt[-1, n, c] > tau * scale]
      if len(js) < 2:
        continue
      cls = np.array([cand[j] for j in js])
      t = stable_entry(Rt, n, cls, P)
      Dc = dist[np.ix_(cls, cls)].astype(float)
      np.fill_diagonal(Dc, np.inf)
      d, r = nearest_earlier(t, Dc, near)
      tc = chain_order(cls, Dc, Rt[-1, n, cls], t)
      iu = np.triu_indices(len(cls), 1)
      mx = Dc[iu]
      m0 = mantel(t, mx, iu) if len(cls) >= 3 else np.nan
      if not np.isnan(m0):                      # t not constant, so every permutation is defined as well
        t1['m_obs'] += m0; t1['m_n'] += 1; t1['m_chain'] += mantel(tc, mx, iu)
        for k in range(n_perm):
          t1['m_perm'][k] += mantel(rng.permutation(t), mx, iu)
      if len(d):
        t1['obs'] += d.sum(); t1['cnt'] += len(d); t1['rel_obs'] += int(r.sum())
        terc = np.minimum(np.argsort(np.argsort(-Rt[-1, n, cls])) * 3 // len(cls), 2)
        dc, _ = nearest_earlier(tc, Dc, near)
        t1['chain_obs'] += dc.sum()
        for k in range(n_perm):
          dp, rp = nearest_earlier(rng.permutation(t), Dc, near)
          t1['perm'][k] += dp.sum(); t1['rel_perm'][k] += int(rp.sum())
          tp = t.copy()
          for g in range(3):
            ix = np.where(terc == g)[0]
            tp[ix] = rng.permutation(t[ix])
          t1['perm_terc'][k] += nearest_earlier(tp, Dc, near)[0].sum()
          t1['chain_perm'][k] += nearest_earlier(rng.permutation(tc), Dc, near)[0].sum()
      # T2: half-rise against distance to the strongest final class
      anchor = cls[int(np.argmax(Rt[-1, n, cls]))]
      for c in cls:
        if c == anchor:
          continue
        traj = Rt[:, n, c]
        if traj[-1] <= 0:
          continue
        t2_rows.append((n, first_reach(np.maximum(traj, 0), 0.5), dist[c, anchor], np.log(traj[-1])))
      # T3, T4 (value writers of the fixed final poles at every checkpoint)
      l25, l50 = [], []
      for j, c in zip(js, cls):
        rel = [k for k in np.where(dist[c] <= near)[0] if k != c]
        own_tr = np.array([float(at[b]['value_writers'][n, j, c]) for at in atlases])
        rel_tr = np.array([float(at[b]['value_writers'][n, j].float()[rel].sum()) if rel else 0.0 for at in atlases])
        if own_tr[-1] > 0 and rel_tr[-1] > 0:
          a25, b25 = first_reach(own_tr, 0.25), first_reach(rel_tr, 0.25)
          a50, b50 = first_reach(own_tr, 0.5), first_reach(rel_tr, 0.5)
          l25.append(a25 - b25); l50.append(a50 - b50)
          t3['own25'].append(a25); t3['rel25'].append(b25); t3['own50'].append(a50); t3['rel50'].append(b50)
        W = final[b]['value_writers'][n, j].float().numpy()
        pos = np.clip(W, 0, None)
        mass = pos.sum()
        if mass > 0:
          relm = np.zeros(K, bool)
          relm[rel] = True
          other = ~relm
          other[c] = False
          t4['own'].append(pos[c] / mass); t4['rel'].append(pos[relm].sum() / mass); t4['other'].append(pos[other].sum() / mass)
          t4['own_chance'].append(1 / K); t4['rel_chance'].append(len(rel) / K)
          vbar = W.sum() + float(final[b]['value_init'][n, j])
          if abs(vbar) > 1e-9:
            t4['gross_net'].append(np.abs(W).sum() / abs(vbar))
          if rel:
            t4['mean_rel_over_mean_abs_other'].append(W[relm].mean() / max(np.abs(W[other]).mean(), 1e-12))
      lag25.append(l25); lag50.append(l50)
      # T5 knockout at entry
      if has_gate:
        for ti in sorted(set(t.tolist())):
          if ti == 0:
            continue
          at = atlases[ti][b]
          Gm, Vm = at['Gm'][n].float(), at['Vm'][n].float()
          Rapprox = F.silu(Gm) * Vm
          for jj, (j, c) in enumerate(zip(js, cls)):
            if t[jj] > ti:
              continue
            gown, vown = float(at['gate_writers'][n, j, c]), float(at['value_writers'][n, j, c])
            rko = float(F.silu(torch.tensor(float(Gm[c]) - gown)) * (float(Vm[c]) - vown))
            others = Rapprox.clone()
            others[c] = -float('inf')
            thr = float(torch.sort(others, descending=True).values[P - 1])
            fire, keep = rko >= thr, float(Rapprox[c]) >= thr
            if t[jj] == ti:
              t5['enter_n'] += 1; t5['enter_fire'] += int(fire); t5['enter_keep'] += int(keep)
            else:
              t5['inc_n'] += 1; t5['inc_fire'] += int(fire); t5['inc_keep'] += int(keep)
    rec = {}
    if t1['cnt']:
      m = t1['cnt']
      rec['T1'] = {'entries': m, 'nearest_mean_obs': t1['obs'] / m,
                   'nearest_mean_perm': float(t1['perm'].mean() / m), 'nearest_perm_sd': float(t1['perm'].std() / m),
                   'nearest_mean_perm_tercile': float(t1['perm_terc'].mean() / m),
                   'z_tercile': float((t1['obs'] - t1['perm_terc'].mean()) / max(t1['perm_terc'].std(), 1e-9)),
                   'p_obs_le_perm_tercile': float((t1['perm_terc'] <= t1['obs']).mean()),
                   'z': float((t1['obs'] - t1['perm'].mean()) / max(t1['perm'].std(), 1e-9)),
                   'p_obs_le_perm': float((t1['perm'] <= t1['obs']).mean()),
                   'earlier_relative_obs': t1['rel_obs'] / m, 'earlier_relative_perm': float(t1['rel_perm'].mean() / m),
                   'p_rel_perm_ge_obs': float((t1['rel_perm'] >= t1['rel_obs']).mean()),
                   'power_chain_z': float((t1['chain_obs'] - t1['chain_perm'].mean()) / max(t1['chain_perm'].std(), 1e-9)),
                   'power_chain_mean_obs': t1['chain_obs'] / m,
                   '_raw': {'obs': float(t1['obs']), 'perm': t1['perm'].tolist(), 'perm_terc': t1['perm_terc'].tolist(),
                            'rel_obs': t1['rel_obs'], 'rel_perm': t1['rel_perm'].tolist()}}
    if t1['m_n']:
      k_ = t1['m_n']
      rec['T1m'] = {'units': k_, 'mantel_obs': t1['m_obs'] / k_, 'mantel_perm': float(t1['m_perm'].mean() / k_),
                    'mantel_perm_sd': float(t1['m_perm'].std() / k_),
                    'z': float((t1['m_obs'] - t1['m_perm'].mean()) / max(t1['m_perm'].std(), 1e-9)),
                    'p_perm_ge_obs': float((t1['m_perm'] >= t1['m_obs']).mean()),
                    'power_chain_obs': t1['m_chain'] / k_,
                    'power_chain_z': float((t1['m_chain'] - t1['m_perm'].mean()) / max(t1['m_perm'].std(), 1e-9)),
                    '_raw': {'obs': t1['m_obs'], 'perm': t1['m_perm'].tolist(), 'chain': t1['m_chain']}}
    if len(t2_rows) > 10:
      arr = np.array(t2_rows, float)
      def slope(a):
        y, x, z = a[:, 1].copy(), a[:, 2].copy(), a[:, 3].copy()
        for u in np.unique(a[:, 0]):
          s_ = a[:, 0] == u
          y[s_] -= y[s_].mean(); x[s_] -= x[s_].mean(); z[s_] -= z[s_].mean()
        X = np.stack([x, z], 1)
        return float(np.linalg.lstsq(X, y, rcond=None)[0][0])
      obs = slope(arr)
      perm = []
      for _ in range(min(n_perm, 300)):
        a2 = arr.copy()
        for u in np.unique(a2[:, 0]):
          s_ = np.where(a2[:, 0] == u)[0]
          a2[s_, 2] = rng.permutation(a2[s_, 2])
        perm.append(slope(a2))
      rec['T2'] = {'n': len(arr), 'slope_t50_checkpoint_index_per_wordnet_step': obs, 'perm_mean': float(np.mean(perm)),
                   'perm_sd': float(np.std(perm)), 'p_perm_ge_obs': float((np.array(perm) >= obs).mean())}
    l25 = [x for x in lag25 if x]
    l50 = [x for x in lag50 if x]
    if l50:
      rec['T3'] = {'n': int(sum(len(x) for x in l50)), 'lag25_median': float(np.median(np.concatenate(l25))),
                   'lag25_ci': boot_ci(l25, rng), 'lag50_median': float(np.median(np.concatenate(l50))),
                   'lag50_ci': boot_ci(l50, rng), 'lag50_mean': float(np.mean(np.concatenate(l50))),
                   'lag50_mean_ci': boot_ci(l50, rng, stat=np.mean),
                   'share_relatives_lead_50': float((np.concatenate(l50) > 0).mean()),
                   'share_own_leads_50': float((np.concatenate(l50) < 0).mean())}
    if t4['own']:
      med = lambda k: float(np.median(t4[k]))
      rec['T4'] = {k: med(k) for k in t4 if t4[k]}
      rec['T4']['n'] = len(t4['own'])
    if t5['enter_n']:
      rec['T5'] = {'entering_fire_without_own': t5['enter_fire'] / t5['enter_n'], 'entering_n': t5['enter_n'],
                   'incumbents_fire_without_own': t5['inc_fire'] / max(t5['inc_n'], 1), 'incumbents_n': t5['inc_n'],
                   'entering_fire_with_own': t5['enter_keep'] / t5['enter_n'],
                   'incumbents_fire_with_own': t5['inc_keep'] / max(t5['inc_n'], 1)}
    out[b] = rec
    print(b, json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != '_raw'}
                         for k, v in rec.items()}), flush=True)
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--tau', type=float, default=0.3)
  p.add_argument('--near', type=int, default=4)
  p.add_argument('--perm', type=int, default=500)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  dist = np.load(a.wordnet)['dist']
  steps, atlases = load(a.run)
  res = analyze(steps, atlases, dist, a.tau, a.near, a.perm)
  json.dump({'steps': steps, 'tau': a.tau, 'near': a.near, 'perm': a.perm, 'blocks': res},
            open(a.out or os.path.join(a.run, 'growth_tests.json'), 'w'), indent=1)


if __name__ == '__main__':
  main()
