"""E2 / E3 tests on a writer atlas (v3, after the adversarial check of 2026-10-16).

Input: atlas_{step}.pt of experiments/convnext_atlas.py for a complete run (pole classes fixed at the final checkpoint)
and a class distance (WordNet 3.0 by default; --dist: the DINOv2 neighbour rank of experiments/class_embed.py). Per
block, over its units n, for the final positive-pole classes c (the P = 20 strongest classes with R_T[n, c] > tau max
|R_T[n]|). Notation: t_c stable entry checkpoint index (first checkpoint from which c stays among the unit's P strongest
classes to the end); R_c(t) class mean of the SwiGLU product at the unit's strongest position; d(c, c') class distance;
'relatives' of c: d <= near; W_k(t) the part of training class k in the class mean of the value (gate) pre-activation of
c, oriented by s_c = sign(vbar_c(T)) (vbar: the class mean of the value; for vbar < 0 the unit's positive response runs
through a negative value, so the writes that build it are the negative ones). Checkpoints are indices on the run's
grid (0, 50, 100, 200, 350, 500, 1,000, 2,000, 2,500, 5,000, 7,500, ..., 25,020 steps), not equal steps.

T1  entry order within the pole, against permutations of the entry checkpoints among the unit's own pole classes
    (plain, and within terciles of final strength).
    nearest: mean distance from an entering class to its nearest strictly earlier pole member, and the share whose
      nearest earlier member is a relative. Power check: a planted order (the observed checkpoints, sorted, reassigned
      along a tree grown from the strongest class, each next class the one nearest to any class placed: Prim order).
      Because of tied checkpoints (above all the large tie at the last checkpoint, which a planted order fills with
      the pole's outliers), a planted order RAISES the WordNet statistic; it lowers the visual-rank one. So the observed
      shift is read against the planted shift: frac = (obs - null) / (planted - null).
    mantel: per unit, the correlation over class pairs of |t_c - t_c'| with d(c, c'), mean over units (needs 3 or more
      classes and a non-constant t). Symmetric in time: it measures co-timing of near classes, which a branch order and
      relatives entering together both produce. Controls: the tercile null (final strength), the Mantel of
      |log R_c - log R_c'| with d (one route of the strength confound), and a class clock: t_c minus the class's mean
      entry checkpoint over all other units of the network where it is a final pole class (relatives that become
      learnable at the same time everywhere give co-timing without any spread inside a unit).
T2  direction: within units, regression of the half-rise checkpoint of R_c on d(c, anchor) (anchor: the unit's strongest
    final class) with log R_c(T) as a covariate; first crossing (first checkpoint with R_c >= R_c(T)/2) and stable
    crossing (first checkpoint from which it stays there). Null: d permuted within units. Power: the observed half-rise
    checkpoints reassigned within each unit in order of d (planted outward order).
T3  relatives' writes against own writes over time: for each pole class with oriented own and summed relatives' parts
    positive at the end, the checkpoint at which each reaches 50% of its final value (first and stable crossing); lag =
    t_own - t_rel (positive: relatives first). Null: the same with the summed parts of as many random non-relative
    classes (a sum of several noisy parts crosses earlier than one part even with the same time course).
T4  composition at the end (oriented parts): shares of the positive mass for own, relatives, others; chance as the
    median share of one random non-relative class (own) and of as many random non-relatives as relatives (rel); own
    percentile among all other classes' parts, and the mean over relatives of each relative's percentile among the
    non-relatives (0.5 by chance); gross / net = sum_k |W_k| / |vbar|. All medians over (unit, class) pairs.
T5  knockout at entry (needs gate and value writers): approximate class response SiLU(G_c) V_c from class means of gate
    and value, strongest position held fixed. Removing the writes of a class set S changes the response of every
    stored pole class (both poles, 2P per unit), so the bar (the P-th strongest response among the other classes) is
    recomputed with them; classes not stored keep their response. 'Survives': the class stays at or above the bar.
    Sets: own, relatives, as many random non-relatives, one random non-relative. Rates for classes at their entry
    checkpoint ('entering') and for classes that entered earlier ('incumbents'), raw and conditional on surviving with
    no knockout (the approximation's calibration), and the entering - incumbent difference within each checkpoint,
    pooled with weights = entering count. A strict head start predicts that most entering classes survive without their
    own writes; the comparison with incumbents is a weak test (any account where own writes keep growing after entry
    predicts entering > incumbents).

Usage: python3 -m experiments.convnext_growth_tests --run <run dir> [--perm 500] [--dist file.npz --near 10]
       [--out file.json]
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


def prim_order(Dc, start):
  """Classes in the order of a tree grown from `start`, each next class the one nearest to any class placed."""
  n = len(Dc)
  seq, left = [start], set(range(n)) - {start}
  while left:
    nxt = min(left, key=lambda k: (Dc[seq, k].min(), k))
    seq.append(nxt)
    left.remove(nxt)
  return seq


def planted(seq, t):
  tt = np.empty(len(t), int)
  tt[seq] = np.sort(t)
  return tt


def mantel(t, x, iu):
  """Pearson correlation over class pairs (iu) of |t_c - t_c'| with the pair distances x; nan if either is constant."""
  y = np.abs(t[:, None] - t[None, :])[iu].astype(float)
  if y.std() == 0 or x.std() == 0:
    return np.nan
  return float(np.corrcoef(x, y)[0, 1])


def crossing(x, frac, stable=False):
  """x (T,) with x[-1] > 0: first index with x >= frac x[-1]; stable: first index from which it stays there."""
  ok = x >= frac * x[-1]
  if stable:
    bad = np.where(~ok)[0]
    return int(bad[-1] + 1) if len(bad) else 0
  return int(np.argmax(ok))


def boot_mean(per_unit, rng, n=1000):
  per_unit = [np.asarray(u, float) for u in per_unit if len(u)]
  if not per_unit:
    return [float('nan')] * 2
  vals = [np.concatenate([per_unit[i] for i in rng.integers(0, len(per_unit), len(per_unit))]).mean() for _ in range(n)]
  return np.percentile(vals, [2.5, 97.5]).round(4).tolist()


def wslope(arr, col):
  """Within-unit least-squares slope of t (col) on d with log R as covariate. arr rows: unit, t_first, t_stable, d, logR"""
  y, x, z = arr[:, col].copy(), arr[:, 3].copy(), arr[:, 4].copy()
  for u in np.unique(arr[:, 0]):
    s_ = arr[:, 0] == u
    y[s_] -= y[s_].mean(); x[s_] -= x[s_].mean(); z[s_] -= z[s_].mean()
  return float(np.linalg.lstsq(np.stack([x, z], 1), y, rcond=None)[0][0])


def pole_classes(final, b, n, P, tau, Rfin):
  """Final positive-pole classes of unit n (slot indices into the stored poles, and classes). Dead units (response
  nearly the same for every class: spread below 1e-3 of its largest magnitude, where the ranking is decided by ties)
  have none; classes outside this ranking's top P at the end (ties) are dropped."""
  scale = float(np.abs(Rfin[n]).max())
  if scale <= 0 or float(np.ptp(Rfin[n])) <= 1e-3 * scale:
    return [], np.array([], int)
  top = set(np.argsort(-Rfin[n])[:P].tolist())
  cand = final[b]['poles'][n, :P].tolist()
  js = [j for j, c in enumerate(cand) if Rfin[n, c] > tau * scale and c in top]
  return js, np.array([cand[j] for j in js], int)


def class_clock(atlases, tau):
  """Entry checkpoints of every final pole class in every unit of the network: {(block, unit): (cls, t)}, and per class
  the sum and count of its entry checkpoints over all units."""
  final = atlases[-1]
  K = final[next(iter(final))]['R'].shape[1]
  entries, s, cnt, dead = {}, np.zeros(K), np.zeros(K), []
  for b in final:
    P_b = final[b]['poles'].shape[1] // 2
    Rt = np.stack([at[b]['R'].numpy() for at in atlases])
    for n in range(Rt.shape[1]):
      js, cls = pole_classes(final, b, n, P_b, tau, Rt[-1])
      if float(np.ptp(Rt[-1, n])) <= 1e-3 * float(np.abs(Rt[-1, n]).max()):
        dead.append((b, n))
      if len(cls) < 2:
        continue
      t = stable_entry(Rt, n, cls, P_b)
      entries[(b, n)] = (cls, t)
      np.add.at(s, cls, t); np.add.at(cnt, cls, 1)
  return entries, s, cnt, dead


def analyze(steps, atlases, dist, tau=0.3, near=4, n_perm=500, seed=0):
  rng = np.random.default_rng(seed)
  final = atlases[-1]
  entries, clk_s, clk_n, dead = class_clock(atlases, tau)
  out = {}
  for b in final:
    P = final[b]['poles'].shape[1] // 2
    Rt = np.stack([at[b]['R'].numpy() for at in atlases])          # (T, units, classes)
    T, U, K = Rt.shape
    z = lambda: np.zeros(n_perm)
    t1 = {'obs': 0.0, 'cnt': 0, 'perm': z(), 'perm_terc': z(), 'chain': 0.0, 'rel_obs': 0, 'rel_perm': z(), 'rel_chain': 0,
          'm_obs': 0.0, 'm_n': 0, 'm_perm': z(), 'm_terc': z(), 'm_chain': 0.0, 'm_strength': 0.0,
          'c_obs': 0.0, 'c_n': 0, 'c_perm': z(), 'c_terc': z()}
    t2_rows, t2_plant = [], []                                    # (unit, t_first, t_stable, d, logR)
    t3 = {k: [] for k in ('rel_first', 'rel_stable', 'rnd_first', 'rnd_stable', 'diff_first', 'diff_stable')}
    t4 = {k: [] for k in ('own', 'rel', 'other', 'own_chance', 'rel_chance', 'own_pct', 'rel_pct', 'gross_net', 'neg_vbar')}
    t5 = {}                                                       # (group, checkpoint) -> counts
    has_gate = 'gate_writers' in final[b] and 'Gm' in final[b]
    for n in range(U):
      if (b, n) not in entries:
        continue
      cls, t = entries[(b, n)]
      js, _ = pole_classes(final, b, n, P, tau, Rt[-1])
      Dc = dist[np.ix_(cls, cls)].astype(float)
      np.fill_diagonal(Dc, np.inf)
      Rfin = Rt[-1, n, cls]
      seq = prim_order(Dc, int(np.argmax(Rfin)))
      tc = planted(seq, t)
      terc = np.minimum(np.argsort(np.argsort(-Rfin)) * 3 // len(cls), 2)

      def perm_terc(tt):
        tp = tt.copy()
        for g in range(3):
          ix = np.where(terc == g)[0]
          tp[ix] = rng.permutation(tt[ix])
        return tp
      perms = [rng.permutation(t) for _ in range(n_perm)]
      perms_t = [perm_terc(t) for _ in range(n_perm)]
      # T1 nearest
      d, r = nearest_earlier(t, Dc, near)
      if len(d):
        t1['obs'] += d.sum(); t1['cnt'] += len(d); t1['rel_obs'] += int(r.sum())
        dc, rc = nearest_earlier(tc, Dc, near)
        t1['chain'] += dc.sum(); t1['rel_chain'] += int(rc.sum())
        for k in range(n_perm):
          dp, rp = nearest_earlier(perms[k], Dc, near)
          t1['perm'][k] += dp.sum(); t1['rel_perm'][k] += int(rp.sum())
          t1['perm_terc'][k] += nearest_earlier(perms_t[k], Dc, near)[0].sum()
      # T1 Mantel, plain and against the class clock
      iu = np.triu_indices(len(cls), 1)
      mx = Dc[iu]
      m0 = mantel(t, mx, iu) if len(cls) >= 3 else np.nan
      if not np.isnan(m0):
        t1['m_obs'] += m0; t1['m_n'] += 1; t1['m_chain'] += mantel(tc, mx, iu)
        lr = np.log(Rfin)
        ms = np.corrcoef(mx, np.abs(lr[:, None] - lr[None, :])[iu])[0, 1]
        t1['m_strength'] += 0.0 if np.isnan(ms) else ms
        for k in range(n_perm):
          t1['m_perm'][k] += mantel(perms[k], mx, iu)
          mt = mantel(perms_t[k], mx, iu)
          t1['m_terc'][k] += 0.0 if np.isnan(mt) else mt
      others = clk_n[cls] - 1
      ok = others > 0
      if ok.sum() >= 3:
        res = (t[ok] - (clk_s[cls][ok] - t[ok]) / others[ok]).astype(float)
        iu2 = np.triu_indices(int(ok.sum()), 1)
        Dk = Dc[np.ix_(ok, ok)]
        m1 = mantel(res, Dk[iu2], iu2)
        if not np.isnan(m1):
          tk = terc[ok]
          t1['c_obs'] += m1; t1['c_n'] += 1
          for k in range(n_perm):
            t1['c_perm'][k] += mantel(rng.permutation(res), Dk[iu2], iu2)
            rp = res.copy()
            for g in range(3):
              ix = np.where(tk == g)[0]
              rp[ix] = rng.permutation(res[ix])
            mt = mantel(rp, Dk[iu2], iu2)
            t1['c_terc'][k] += 0.0 if np.isnan(mt) else mt
      # T2
      anchor = cls[int(np.argmax(Rfin))]
      rows = []
      for c in cls:
        traj = Rt[:, n, c]
        if c == anchor or traj[-1] <= 0:
          continue
        x = np.maximum(traj, 0)
        rows.append((n, crossing(x, 0.5), crossing(x, 0.5, True), dist[c, anchor], np.log(traj[-1])))
      if len(rows) >= 2:
        rows = np.array(rows, float)
        t2_rows.append(rows)
        pl = rows.copy()
        order = np.lexsort((rng.random(len(rows)), rows[:, 3]))   # by distance, ties at random
        for col in (1, 2):
          pl[order, col] = np.sort(rows[:, col])
        t2_plant.append(pl)
      # T3, T4 (oriented value parts)
      l_rel = {'first': [], 'stable': []}
      for j, c in zip(js, cls):
        rel = np.array([k for k in np.where(dist[c] <= near)[0] if k != c], int)
        nonrel = np.setdiff1d(np.arange(K), np.append(rel, c))
        rnd = rng.choice(nonrel, size=len(rel), replace=False) if len(rel) else rel
        W = final[b]['value_writers'][n, j].float().numpy()
        vbar = W.sum() + float(final[b]['value_init'][n, j])
        sgn = 1.0 if vbar >= 0 else -1.0
        own_tr = sgn * np.array([float(at[b]['value_writers'][n, j, c]) for at in atlases])
        if len(rel):
          VW = np.stack([at[b]['value_writers'][n, j].float().numpy() for at in atlases])   # (T, K)
          rel_tr, rnd_tr = sgn * VW[:, rel].sum(1), sgn * VW[:, rnd].sum(1)
          if own_tr[-1] > 0 and rel_tr[-1] > 0 and rnd_tr[-1] > 0:
            for mode in ('first', 'stable'):
              st = mode == 'stable'
              a_, r_, q_ = crossing(own_tr, 0.5, st), crossing(rel_tr, 0.5, st), crossing(rnd_tr, 0.5, st)
              t3[f'rel_{mode}'].append(a_ - r_); t3[f'rnd_{mode}'].append(a_ - q_); t3[f'diff_{mode}'].append(q_ - r_)
              l_rel[mode].append(q_ - r_)
        Wo = sgn * W
        pos = np.clip(Wo, 0, None)
        mass = pos.sum()
        t4['neg_vbar'].append(float(vbar < 0))
        if mass > 0:
          relm = np.zeros(K, bool); relm[rel] = True
          other = ~relm; other[c] = False
          k0 = rng.choice(nonrel)
          t4['own'].append(pos[c] / mass); t4['rel'].append(pos[relm].sum() / mass); t4['other'].append(pos[other].sum() / mass)
          t4['own_chance'].append(pos[k0] / mass); t4['rel_chance'].append(pos[rnd].sum() / mass if len(rnd) else 0.0)
          t4['own_pct'].append(float((Wo[np.arange(K) != c] < Wo[c]).mean()))
          if len(rel):
            t4['rel_pct'].append(float(np.mean([(Wo[other] < Wo[k]).mean() for k in rel])))
          if abs(vbar) > 1e-9:
            t4['gross_net'].append(np.abs(W).sum() / abs(vbar))
      t3.setdefault('_unit_diff_first', []).append(l_rel['first'])
      t3.setdefault('_unit_diff_stable', []).append(l_rel['stable'])
      # T5 knockout at entry, bar recomputed over the stored pole classes
      if has_gate:
        stored = final[b]['poles'][n].numpy()                     # 2P classes with writers
        for ti in sorted(set(t.tolist())):
          if ti == 0:
            continue
          at = atlases[ti][b]
          Gm, Vm = at['Gm'][n].float(), at['Vm'][n].float()
          base = F.silu(Gm) * Vm
          GW, VW_ = at['gate_writers'][n].float(), at['value_writers'][n].float()   # (2P, K)
          for jj, (j, c) in enumerate(zip(js, cls)):
            if t[jj] > ti:
              continue
            c = int(c)
            rel = [int(k) for k in np.where(dist[c] <= near)[0] if k != c]
            nonrel = np.setdiff1d(np.arange(K), np.array(rel + [c], int))
            sets = {'own': [int(c)], 'rel': rel, 'rand': rng.choice(nonrel, size=len(rel), replace=False).tolist() if rel else [],
                    'rand1': [int(rng.choice(nonrel))]}
            key = ('enter' if t[jj] == ti else 'inc', ti)
            r5 = t5.setdefault(key, {'n': 0, 'keep': 0, 'own_neg': 0, **{f'{s}_{m}': 0 for s in sets for m in ('fix', 'new', 'new_keep')}})
            thr0 = float(torch.sort(torch.cat([base[:c], base[c + 1:]]), descending=True).values[P - 1])
            keep = float(base[c]) >= thr0
            r5['n'] += 1; r5['keep'] += int(keep); r5['own_neg'] += int(float(VW_[j, c]) < 0)
            for s, S in sets.items():
              resp = base.clone()
              if S:
                resp[stored] = F.silu(Gm[stored] - GW[:, S].sum(1)) * (Vm[stored] - VW_[:, S].sum(1))
              rc = float(resp[c])
              fix = rc >= thr0
              new = rc >= float(torch.sort(torch.cat([resp[:c], resp[c + 1:]]), descending=True).values[P - 1])
              r5[f'{s}_fix'] += int(fix); r5[f'{s}_new'] += int(new); r5[f'{s}_new_keep'] += int(new and keep)
    # records
    rec = {'dead_units': sum(1 for d_ in dead if d_[0] == b)}
    if t1['cnt']:
      m, p, pt = t1['cnt'], t1['perm'], t1['perm_terc']
      rec['T1'] = {'entries': m, 'nearest_obs': t1['obs'] / m, 'nearest_perm': float(p.mean() / m),
                   'nearest_perm_tercile': float(pt.mean() / m), 'nearest_planted': t1['chain'] / m,
                   'z': float((t1['obs'] - p.mean()) / max(p.std(), 1e-9)),
                   'z_tercile': float((t1['obs'] - pt.mean()) / max(pt.std(), 1e-9)),
                   'z_planted': float((t1['chain'] - p.mean()) / max(p.std(), 1e-9)),
                   'frac_of_planted_tercile': float((t1['obs'] - pt.mean()) / (t1['chain'] - p.mean())) if t1['chain'] != p.mean() else None,
                   'earlier_relative_obs': t1['rel_obs'] / m, 'earlier_relative_perm': float(t1['rel_perm'].mean() / m),
                   'earlier_relative_planted': t1['rel_chain'] / m,
                   'z_earlier_relative': float((t1['rel_obs'] - t1['rel_perm'].mean()) / max(t1['rel_perm'].std(), 1e-9)),
                   '_raw': {'obs': float(t1['obs']), 'perm': p.tolist(), 'perm_terc': pt.tolist(), 'chain': float(t1['chain']),
                            'rel_obs': t1['rel_obs'], 'rel_perm': t1['rel_perm'].tolist(), 'rel_chain': t1['rel_chain']}}
    if t1['m_n']:
      k_, mp, mt = t1['m_n'], t1['m_perm'], t1['m_terc']
      rec['T1m'] = {'units': k_, 'mantel_obs': t1['m_obs'] / k_, 'mantel_perm': float(mp.mean() / k_),
                    'mantel_perm_tercile': float(mt.mean() / k_), 'mantel_planted': t1['m_chain'] / k_,
                    'z': float((t1['m_obs'] - mp.mean()) / max(mp.std(), 1e-9)),
                    'z_tercile': float((t1['m_obs'] - mt.mean()) / max(mt.std(), 1e-9)),
                    'mantel_strength_vs_dist': t1['m_strength'] / k_,
                    '_raw': {'obs': t1['m_obs'], 'perm': mp.tolist(), 'perm_terc': mt.tolist(), 'chain': t1['m_chain']}}
    if t1['c_n']:
      k_, cp, ct = t1['c_n'], t1['c_perm'], t1['c_terc']
      rec['T1clock'] = {'units': k_, 'mantel_obs': t1['c_obs'] / k_, 'mantel_perm': float(cp.mean() / k_),
                        'mantel_perm_tercile': float(ct.mean() / k_),
                        'z': float((t1['c_obs'] - cp.mean()) / max(cp.std(), 1e-9)),
                        'z_tercile': float((t1['c_obs'] - ct.mean()) / max(ct.std(), 1e-9)),
                        '_raw': {'obs': t1['c_obs'], 'perm': cp.tolist(), 'perm_terc': ct.tolist()}}
    if len(t2_rows) > 5:
      arr, pl = np.concatenate(t2_rows), np.concatenate(t2_plant)
      rec['T2'] = {'n': len(arr), 'units': len(t2_rows)}
      for col, mode in ((1, 'first'), (2, 'stable')):
        obs = wslope(arr, col)
        perm = []
        for _ in range(n_perm):
          a2 = arr.copy()
          for rows in t2_rows:
            s_ = np.where(a2[:, 0] == rows[0, 0])[0]
            a2[s_, 3] = rng.permutation(a2[s_, 3])
          perm.append(wslope(a2, col))
        perm = np.array(perm)
        rec['T2'][mode] = {'slope': obs, 'perm_mean': float(perm.mean()), 'perm_sd': float(perm.std()),
                           'p_perm_ge_obs': float((perm >= obs).mean()), 'planted_slope': wslope(pl, col)}
    if t3['rel_first']:
      rec['T3'] = {'n': len(t3['rel_first'])}
      for mode in ('first', 'stable'):
        rec['T3'][mode] = {'lag_rel_mean': float(np.mean(t3[f'rel_{mode}'])), 'lag_rand_mean': float(np.mean(t3[f'rnd_{mode}'])),
                           'rel_minus_rand': float(np.mean(t3[f'diff_{mode}'])),
                           'rel_minus_rand_ci': boot_mean(t3[f'_unit_diff_{mode}'], rng),
                           'share_rel_first': float((np.array(t3[f'rel_{mode}']) > 0).mean()),
                           'share_own_first': float((np.array(t3[f'rel_{mode}']) < 0).mean())}
    if t4['own']:
      rec['T4'] = {k: float(np.median(v)) for k, v in t4.items() if v and k != 'neg_vbar'}
      rec['T4']['share_neg_vbar'] = float(np.mean(t4['neg_vbar']))
      rec['T4']['n'] = len(t4['own'])
    if t5:
      sets = ('own', 'rel', 'rand', 'rand1')
      agg = {g: {} for g in ('enter', 'inc')}
      for (g, ti), r5 in t5.items():
        for k, v in r5.items():
          agg[g][k] = agg[g].get(k, 0) + v
      rec['T5'] = {}
      for g in agg:
        a = agg[g]
        rec['T5'][g] = {'n': a['n'], 'fire_no_knockout': a['keep'] / a['n'], 'own_value_negative': a['own_neg'] / a['n'],
                        **{f'survive_{s}_fixed_bar': a[f'{s}_fix'] / a['n'] for s in sets},
                        **{f'survive_{s}': a[f'{s}_new'] / a['n'] for s in sets},
                        **{f'survive_{s}_given_fire': a[f'{s}_new_keep'] / max(a['keep'], 1) for s in sets}}
      strat = {}
      for s in sets:
        num = den = 0.0
        for (g, ti), r5 in t5.items():
          if g != 'enter' or ('inc', ti) not in t5:
            continue
          ri = t5[('inc', ti)]
          num += r5['n'] * (r5[f'{s}_new'] / r5['n'] - ri[f'{s}_new'] / ri['n'])
          den += r5['n']
        strat[s] = num / den if den else None
      rec['T5']['enter_minus_inc_within_checkpoint'] = strat
    out[b] = rec
    print(b, json.dumps({k: ({kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != '_raw'}
                             if isinstance(v, dict) else v) for k, v in rec.items()}, default=str)[:3000], flush=True)
  return out


def pooled(blocks, names):
  """Pooled T1 statistics over a set of blocks (index-aligned sums of the per-block permutation sums)."""
  out = {}
  out['dead_units'] = sum(blocks[b].get('dead_units', 0) for b in names)
  have = [b for b in names if 'T1' in blocks[b]]
  if have:
    R = [blocks[b]['T1']['_raw'] for b in have]
    m = sum(blocks[b]['T1']['entries'] for b in have)
    o, p, pt, ch = sum(r['obs'] for r in R), np.sum([r['perm'] for r in R], 0), np.sum([r['perm_terc'] for r in R], 0), sum(r['chain'] for r in R)
    ro, rp, rc = sum(r['rel_obs'] for r in R), np.sum([r['rel_perm'] for r in R], 0), sum(r['rel_chain'] for r in R)
    out['nearest'] = {'obs': o / m, 'perm': p.mean() / m, 'planted': ch / m, 'z': (o - p.mean()) / p.std(),
                      'z_tercile': (o - pt.mean()) / pt.std(), 'frac_of_planted_tercile': (o - pt.mean()) / (ch - p.mean()),
                      'earlier_relative': [ro / m, rp.mean() / m, rc / m], 'z_earlier_relative': (ro - rp.mean()) / rp.std()}
  for key in ('T1m', 'T1clock'):
    have = [b for b in names if key in blocks[b]]
    if not have:
      continue
    R = [blocks[b][key]['_raw'] for b in have]
    u = sum(blocks[b][key]['units'] for b in have)
    o, p, pt = sum(r['obs'] for r in R), np.sum([r['perm'] for r in R], 0), np.sum([r['perm_terc'] for r in R], 0)
    rec = {'r': o / u, 'perm': p.mean() / u, 'perm_tercile': pt.mean() / u, 'z': (o - p.mean()) / p.std(),
           'z_tercile': (o - pt.mean()) / pt.std(), 'p_tercile': float(((pt >= o).sum() + 1) / (len(pt) + 1))}
    if key == 'T1m':
      ch = sum(r['chain'] for r in R)
      rec['planted'] = ch / u
      rec['frac_of_planted_tercile'] = (o - pt.mean()) / (ch - p.mean())
    out[key] = rec
  have = [b for b in names if 'T2' in blocks[b]]
  for mode in ('first', 'stable'):
    s = np.array([blocks[b]['T2'][mode]['slope'] - blocks[b]['T2'][mode]['perm_mean'] for b in have])
    sd = np.array([blocks[b]['T2'][mode]['perm_sd'] for b in have])
    pl = np.array([blocks[b]['T2'][mode]['planted_slope'] for b in have])
    if len(s):
      out[f'T2_{mode}'] = {'mean_slope': s.mean(), 'z_fixed': s.sum() / np.sqrt((sd ** 2).sum()),
                           't_over_blocks': s.mean() / (s.std(ddof=1) / np.sqrt(len(s))) if len(s) > 1 else None,
                           'planted_mean': pl.mean(), 'frac_of_planted': s.mean() / pl.mean() if pl.mean() else None}
  return {k: ({kk: (float(vv) if isinstance(vv, (float, np.floating)) else vv) for kk, vv in v.items()} if isinstance(v, dict) else v)
          for k, v in out.items()}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--dist', default=None, help='npz with rank_dist (visual neighbour ranks) in place of WordNet')
  p.add_argument('--tau', type=float, default=0.3)
  p.add_argument('--near', type=int, default=4)
  p.add_argument('--perm', type=int, default=500)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  dist = np.load(a.dist)['rank_dist'].astype(np.int64) if a.dist else np.load(a.wordnet)['dist']
  steps, atlases = load(a.run)
  res = analyze(steps, atlases, dist, a.tau, a.near, a.perm)
  blocks = list(res)
  groups = {'all': blocks, 'early': [b for b in blocks if b.startswith(('stage0', 'stage1'))],
            'late': [b for b in blocks if not b.startswith(('stage0', 'stage1'))]}
  pool = {g: pooled(res, bs) for g, bs in groups.items() if bs}
  print('pooled', json.dumps(pool, default=float), flush=True)
  json.dump({'version': 3, 'steps': steps, 'tau': a.tau, 'near': a.near, 'perm': a.perm, 'dist': a.dist or 'wordnet',
             'pooled': pool, 'blocks': res}, open(a.out or os.path.join(a.run, 'growth_tests_v3.json'), 'w'), indent=1,
            default=float)


if __name__ == '__main__':
  main()
