"""Rare-class experiment (protocol v2, docs/EXACT_HISTORY.md Sec. 4.3): the primary and secondary measures, controls,
floors and statistics. Class sets, probes, zeta_c and first-order effects E_S are defined in experiments/rare_common.py.

Units and responses. Unit (b, n) of block b: u = SiLU(g) v with gate and value pre-activations g = W_g[n] h + b_g[n],
v = W_v[n] h + b_v[n] at the block's LayerNorm output h. p*(x) = argmax over positions of u(x, p, n); g, v, u, h below
are taken there. On the probe Q: R[n, k] = mean over Q images of class k of u at p* (the atlas response), ubar_n = mean
of u over Q images and all positions, sigma_n = std over the classes k of R[n, k] (one response scale per unit, run and
checkpoint). top20(n) = the 20 classes with the largest R[n, k] at the pole step (the run's final checkpoint).
Parts of a group set S in unit n on image x: G_S(x) = sum_{g in S} (C^gate_g[n] h + Cb^gate_g[n]), V_S likewise (class
split for class sets, fine split for Own_m and N_own; whole class = the class split's part of c).

Carriers. Delta_n(x) = zeta_y(x) - zeta_y(x | u_n := ubar_n) (mean ablation of u_n at every position, label y).
DeltaQ[n, k] = mean of Delta_n over Q images of class k; DeltaV[n, c] = mean over V_sel(c) of Delta_n with label c.
Specific effect Delta~[n, c] = DeltaV[n, c] - median over k != c of DeltaQ[n, k]. K(c) = the 8 units with the largest
Delta~ > 0 (selected without attribution, on V_sel).

P1, route provenance (on V_est, n in K(c), s_n = sign of mean u on V_c):
  L_S(n)  = s_n mean_x [u - SiLU(g - G_S)(v - V_S)] / sigma_n        (exact local knockout at fixed block inputs)
  L^gate_S, L^value_S: the same with only G_S or only V_S removed
  F(c)    = median_n [L_Near10(n) - median over matched random10 draws of L(n)]
  O(c)    = median_n [L_Own(n) - median over N_own of L_Own(c')(n)]
  PI(c)   = (F - O) / (|F| + |O|), defined when |F| + |O| > the 95th percentile of the null |F_j| + |O_j| (F_j with
            Near10 replaced by matched draw j, O_j with Own_m replaced by the N_own class j mod |N_own| and its median
            over the other N_own classes); otherwise 'no route'.
  lin     the linear version at the pre-activations: lin_S(n) = mean_x [s^g_n G_S + s^v_n V_S] / sigma_n (s^g, s^v the
          signs of mean g, v on V_c); F_lin_pct = median_n percentile of lin_Near10 among the draws, O_lin_pct the same
          for Own_m among N_own.
  O_whole the same as O with own = the whole class (class split) and N_own's whole classes (the C0 calibration).
  Universality: signature(c) = O above the 99th percentile of its pseudo-own null (the O_j) and F at or below the
  median of the F_j; pseudo_signature(c) = the same with Own_m replaced by its first N_own class.
S3, own niche on K(c): own_rank = percentile of L_Own among the L_k of the 999 other single class parts; own_share =
  max(0, L_Own) / (max(0, L_Own) + sum_{k != c} max(0, L_k)); medians over K(c).

P2, depth of the own route (V_c, all images): X^s_S = mean zeta_c(theta) - mean zeta_c(theta minus S's parts in the
unit maps of stage s), exact; X^H_S the same for the head (exact and linear: the head reads fixed features).
  X^s(c) = X^s_Own - median over N_own of X^s_Own(c');  lambda_3 = X^3+ / sum_s X^s+  (X+ = max(0, X)), defined when
  sum_s X^s+ > the 95th percentile of the same sum for each N_own class as pseudo-own (null median over the others).
  head_share = X^H+ / (sum_s X^s+ + X^H+). lambda3_fo: the same from first-order stage effects. lambda3_whole: own = the
  whole class (class split), the reference of the C0 run. head_q(c) = mean over head row c (weights and bias) of
  1 / (sqrt(v_hat) + eps) from the checkpoint's snapshot (effective step size; compare rare against C0).

P3, own writes against gradient flow (V_est, all units):
  E_S(n)  = first-order effect of S's parts in rows n of the gate and value maps on mean_{V_est} zeta_c
  w_n     = max(0, E_Own(n) - median over N_own of E_Own(c')(n))
  rel(n, c) = |top20(n) & Near10(c)| / 20;  rel_mis(n, c) = mean of rel(n, c'') over 20 selected classes c'' of c's
  stratum unrelated to c (WordNet > 4 and not in each other's Near10)
  R_own = sum_n w_n rel / sum_n w_n;  R_mis = the same with rel_mis;  R_flow, R_flowD = the same with the flow weights
  f_n, f^D_n (subcommand flow); Delta_flow = R_own - R_flow, defined when sum_n w_n > the 95th percentile of the same
  sum for each N_own class as pseudo-own. q_n = mean over the unit's gate and value rows (weights and bias) of
  1 / (sqrt(v_hat) + eps) from the snapshot; spearman_wf_q = Spearman(w_n / f_n, q_n) over units with f_n > 0.
  check_units: exact excision of Own_m from rows n only, on the 16 units with the largest E_Own(n), against E_Own(n).
Flow weights (subcommand flow): f_n = sum over checkpoints t of l_t mean over T_c of ||lam grad_{rows n} L||, l_t the
summed learning rate of the interval starting at t, T_c the class's own training images (relabelled or noise images in
a control's rare run), 4 augmentations each mixed with a random kept partner at lam ~ Beta(0.8, 0.8), L the mixup loss;
lam grad is the own group's share of the update under the soft-group split. f^D: the gradient scaled by
D = 1 / (sqrt(v_hat) + eps) from snap_t (for t = 0, whose v is still zero, and for checkpoints without a snapshot,
the nearest snapshot with t > 0, also from the run's history/ folder; the steps used are saved as snap_steps).

S1, builders from cross-writes (atlas_{step}.pt of the class split): for unit n,
  beta^x_n(k) = sum over c' in top20(n), c' not in {c, k} of [max(0, s^v V[n, c', k]) + max(0, s^g G[n, c', k])]
  (V, G the atlas writer vectors, s the signs of the class-mean value and gate), beta(c) = 0. RBp^x(n) = percentile of
  the Near10 share of beta among the matched draws' shares; RPp(n) = the same for |top20(n) & S| (no attribution).
  Reported as the median over K(c) and the w-weighted mean over all units.
S2, memorization per image (level 10, needs the training set): on T_c (center crops of the 10 own images, one group
  each), A_s(i, x) = sum over units n of stage s of |P^g_i + P^v_i|(x) at p*(x); sigma^s_i = A_s(i, x_i) / mean_{j != i}
  A_s(j, x_i); sigma^H_i the same with |head row c part of group i| (C_i[c] f(x) + Cb_i[c]). Medians over i.
  head_row_share = ||C^head_i[c]||^2 / ||C^head_i||^2, the share of the group's head part in its own label row (the
  rest comes from mixup partners' labels).
S4, churn (subcommand churn, descriptive): for n in K(c) and consecutive checkpoints t < t', the exact new writes of
  the sets Near10, Far (class split) and Own_m (fine split) within the interval (rcdl.attribution.interval_buckets),
  evaluated on fixed inputs h_t'(x) at p*_t'(x) on V_est, and the drift S_t (h_t' - h_t) of the earlier parts:
  traffic_w = sum over intervals and units of mean_x |s^g write^g_w + s^v write^v_w| / sigma_n; tau = (traffic_Far /
  |Far|) / (traffic_Near10 / 10). With --split attrR (E1r) the same sets index random groups: the noise floor.
S7, consolidation (phase 0, descriptive, on Q): over the units of stages 2 and 3, Delta~Q = DeltaQ[:, c] - median over
  k != c; C1 = max Delta~Q+ / sum Delta~Q+, PR = (sum Delta~Q+)^2 / sum (Delta~Q+)^2, for the 548 stratum candidates,
  with Spearman correlations with cos10 and nW; with the atlas, N = the share of c's own writes in the builder row of
  c on its top carrier, raw[c, c] / sum_k raw[c, k] (raw as in S1 without the cross-write rule; c must be a pole).
M4 (phase 0): removing the 10 Near10 class parts from all 36 unit maps at once, X^U_Near10 on V_c against matched
  draws, with the probe accuracy after removal (documents the collapse).

Statistics (subcommand compare): unit of analysis the class; primary sample the selected classes at levels 10 and 100
in their rare run. Paired bootstrap over classes (10,000 resamples) of the median, two-sided p = 2 min(P(b <= 0),
P(b >= 0)); Holm over P1 to P3; contrasts rare - C0 against the same-seed floor D0 = other-set run - C0 (Wilcoxon
signed-rank of D - D0, normal approximation with tie correction); floors C0 - E1 (instrumentation) and E1r - E1
(seed); seed-1 replication: same sign, one-sided p < 0.05; universality: signatures against pseudo-own signatures,
one-sided binomial test at the pseudo rate (k + 1) / (n + 2). The P2 primary D(c) = lambda_3(rare, Own_m) -
lambda_3^ref(C0, whole class) has the floor D0(c) = lambda_3(other-set run, Own_m) - lambda_3^ref(C0). Decision rules
(section 7): a count against the essay needs Holm significance, the floor (P2) and seed-1 replication; without
seed-1 runs it is listed as pending.
Amendment 1 (docs/data/rare/protocol_amendment_1.md): classes sorted by id before every bootstrap; PI, lambda_3 and
R_own enter every statistic only where defined (PI: 'defined', lambda_3: P2.defined, R_own: P3.defined); a floor
passes only when its Wilcoxon z has the sign of the contrast's median; the noise control has three states (pass,
fail, uninformative); secondary reports: P1 over all classes, with 'no route' as 0 and with symmetric F_pct gates;
the protocol's other P1 versions; P2 with the first-order and Own_m references; McNemar tests for universality
(against the pseudo-own, and rare against common); S1 RBpx_w; S2 and the relabel control with their floors; and
flags when a count against the essay also holds in the common run (C0, C1) or when the P2 references disagree.

Subcommands:
  measure  --run R --step S [--flow flow.npz] [--train /dev/shm/imagenet256] -> rare_measure_{S}.json (+ .npz arrays)
  flow     --run R --train /dev/shm/imagenet256 -> rare_flow.npz
  churn    --run R [--split attr|attrR] -> rare_churn.json
  compare  --rare A0.json B0.json --c0 C0.json [--rare1 ... --c1 ...] [--e1 ... --e1r ...] -> rare_compare.json
Phase 0: measure on E1 / E1r (no fine split: own = the whole class) with --phase0 adds M4 and S7.
"""

import argparse
import json
import math
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from experiments.rare_common import (Selection, auroc, by_region, class_gradients, clean_json, excise, group_effects,
                                     load_npz, load_parts, load_probe_images, load_spec, logits_of, nearest_snapshot,
                                     noise_images, normalize, own_groups, pct_rank, probe_kind, probes, rankdata,
                                     snapshot_files, spearman, tracked_modules, zeta)

N_DRAWS = 1000
K_UNITS = 8
EPS, BETA2 = 1e-8, 0.999


# ------------------------------------------------------------------------------------------------- model plumbing

def layer_seq(model) -> list:
  """[(name, module)]: 'down{i}' and 'stage{i}.block{j}' in forward order (ConvNeXt.features without the pooling)."""
  seq = []
  for i, (down, stage) in enumerate(zip(model.downsample, model.stages)):
    seq.append((f'down{i}', down))
    for j, blk in enumerate(stage):
      seq.append((f'stage{i}.block{j}', blk))
  return seq


def run_from(model, seq, start: int, x):
  """Logits from the input x of seq[start] (block caching)."""
  for _, m in seq[start:]:
    x = m(x)
  return model.head(model.norm(x.mean((2, 3))))


def unit_list(model) -> List[tuple]:
  return [(b, n) for b, blk in model.blocks() for n in range(blk.pw_gate.out_features)]


def block_inputs(model, x, names):
  """{block: (B, H, W, d) LayerNorm output h} for one batch."""
  out, hooks = {}, []
  for n, b in model.blocks():
    if n in names:
      hooks.append(b.norm.register_forward_hook(lambda m, i, o, n=n: out.__setitem__(n, o)))
  try:
    with torch.no_grad():
      z = model(x)
  finally:
    for h in hooks:
      h.remove()
  return out, z


@torch.no_grad()
def unit_star(model, x, blocks=None, batch: int = 100) -> Dict[str, dict]:
  """{block: {'h': (N, U, d), 'g', 'v', 'u', 'p': (N, U)}} at p*(x) for normalized images x."""
  named = dict(model.blocks())
  blocks = list(named) if blocks is None else list(blocks)
  out = {b: {k: [] for k in ('h', 'g', 'v', 'u', 'p')} for b in blocks}
  for i in range(0, len(x), batch):
    hs, _ = block_inputs(model, x[i:i + batch], set(blocks))
    for b in blocks:
      blk = named[b]
      h = hs[b].flatten(1, 2)
      g, v = blk.pw_gate(h), blk.pw_value(h)
      u = F.silu(g) * v
      p = u.argmax(1)
      take = lambda t: torch.gather(t, 1, p[:, None, :])[:, 0]
      out[b]['h'].append(torch.gather(h, 1, p[:, :, None].expand(-1, -1, h.shape[-1])))
      out[b]['g'].append(take(g))
      out[b]['v'].append(take(v))
      out[b]['u'].append(take(u))
      out[b]['p'].append(p)
  return {b: {k: torch.cat(v) for k, v in d.items()} for b, d in out.items()}


@torch.no_grad()
def probe_stats(model, xQ_u8, yQ, n_classes: int, device, batch: int = 200, dtype=torch.float32) -> Dict[str, dict]:
  """{block: {'R': (U, n_classes) class-mean u*, 'ubar': (U,) mean u over images and positions, 'sigma': (U,)}}."""
  named = dict(model.blocks())
  yQ = torch.as_tensor(np.asarray(yQ))
  acc = {b: {'R': 0, 'ubar': 0, 'n': 0, 'cnt': torch.zeros(n_classes, dtype=dtype)} for b in named}
  for i in range(0, len(yQ), batch):
    x = normalize(xQ_u8[i:i + batch], device).to(dtype) if xQ_u8.dtype == torch.uint8 else xQ_u8[i:i + batch]
    hs, _ = block_inputs(model, x, set(named))
    yb = yQ[i:i + batch]
    for b, blk in named.items():
      h = hs[b].flatten(1, 2)
      u = F.silu(blk.pw_gate(h)) * blk.pw_value(h)                 # (B, P, U)
      us = u.max(1).values.cpu().to(dtype)                         # u at p*
      R = torch.zeros(n_classes, us.shape[1], dtype=dtype).index_add_(0, yb, us)
      acc[b]['R'] = acc[b]['R'] + R
      acc[b]['ubar'] = acc[b]['ubar'] + u.sum((0, 1)).cpu().to(dtype)
      acc[b]['n'] += u.shape[0] * u.shape[1]
      acc[b]['cnt'] += torch.bincount(yb, minlength=n_classes).to(dtype)
  out = {}
  for b, a in acc.items():
    R = (a['R'] / a['cnt'].clamp_min(1)[:, None]).T
    out[b] = {'R': R, 'ubar': a['ubar'] / a['n'], 'sigma': R.std(1).clamp_min(1e-12)}
  return out


@torch.no_grad()
def unit_ablation(model, x, y, ubar: Dict[str, torch.Tensor], units_per_pass: int = 8, batch: int = 128):
  """Delta (n_units, N): zeta_{y_i}(x_i) - zeta_{y_i}(x_i | u_n := ubar_n), units in unit_list order. Block caching:
  the prefix up to a block runs once per batch, and units_per_pass units are ablated in one pass of the suffix.
  x: normalized images, or uint8 (normalized per batch on the model's device)."""
  p0 = next(model.parameters())
  seq = layer_seq(model)
  named = dict(model.blocks())
  offs, o = {}, 0
  for b, blk in model.blocks():
    offs[b] = o
    o += blk.pw_gate.out_features
  y = torch.as_tensor(y, device=p0.device)
  D = torch.zeros(o, len(x), dtype=torch.float64)
  for i0 in range(0, len(x), batch):
    xb, yb = x[i0:i0 + batch], y[i0:i0 + batch]
    xb = normalize(xb, p0.device).to(p0.dtype) if xb.dtype == torch.uint8 else xb.to(p0.device)
    B = len(yb)
    act = xb
    base = None
    for li, (name, mod) in enumerate(seq):
      if name in named:
        if base is None:
          base = zeta(run_from(model, seq, li, act), yb)
        blk = named[name]
        U = blk.pw_gate.out_features
        for n0 in range(0, U, units_per_pass):
          ns = list(range(n0, min(n0 + units_per_pass, U)))

          def pre(mod_, inp, ns=ns, val=ubar[name]):
            u = inp[0].clone()
            for r, n in enumerate(ns):
              u[r * B:(r + 1) * B, ..., n] = val[n].to(u.dtype)
            return (u,)
          hk = blk.pw_down.register_forward_pre_hook(pre)
          try:
            z = run_from(model, seq, li, act.repeat(len(ns), 1, 1, 1))
          finally:
            hk.remove()
          za = zeta(z, yb.repeat(len(ns))).view(len(ns), B)
          D[offs[name] + n0:offs[name] + n0 + len(ns), i0:i0 + B] = (base[None] - za).double().cpu()
      act = mod(act)
  return D


def specific_effect(DeltaV_c: torch.Tensor, DeltaQ: torch.Tensor, c: int) -> torch.Tensor:
  """Delta~ (n_units,) = DeltaV_c - median over the other classes of DeltaQ."""
  others = torch.cat([DeltaQ[:, :c], DeltaQ[:, c + 1:]], 1)
  return DeltaV_c - others.median(1).values


def carriers(spec_eff: torch.Tensor, k: int = K_UNITS) -> List[int]:
  """The k units with the largest positive specific effect (fewer if fewer are positive)."""
  order = torch.argsort(spec_eff, descending=True)[:k]
  return [int(j) for j in order if spec_eff[j] > 0]


# ------------------------------------------------------------------------------------------------- parts, knockouts

def unit_map_parts(parts: Dict[str, dict], device, dtype=torch.float32) -> Dict[str, dict]:
  """Unit maps of a compact state unscaled on the device: {key: {'C': (G, U, d), 'Cb': (G, U)}}."""
  out = {}
  for k, r in parts.items():
    if k == 'head':
      continue
    out[k] = {'C': r['C'].to(device).to(dtype) / r.get('C_scale', 1.0),
              'Cb': r['Cb'].to(device).to(dtype) / r.get('Cb_scale', 1.0)}
  return out


def unit_parts(up: Dict[str, dict], block: str, n: int, h: torch.Tensor):
  """(P^g, P^v): (G, N) gate and value parts of every group in unit n of block on inputs h (N, d)."""
  g, v = up[f'{block}.pw_gate'], up[f'{block}.pw_value']
  h = h.to(g['C'].dtype)
  return g['C'][:, n] @ h.T + g['Cb'][:, n, None], v['C'][:, n] @ h.T + v['Cb'][:, n, None]


def set_sums(P: torch.Tensor, sets: Sequence[Sequence[int]]) -> torch.Tensor:
  """(S, N): the parts summed over each group set."""
  return torch.stack([P[torch.as_tensor(list(s), dtype=torch.long, device=P.device)].sum(0) if len(s)
                      else torch.zeros_like(P[0]) for s in sets])


def knockout(g, v, Gs, Vs, s: float, sigma: float, mode: str = 'both') -> torch.Tensor:
  """L (S,): s mean_x [u - u_S] / sigma with u = SiLU(g) v and u_S the unit with the sets' parts removed from the
  gate and value ('both'), the gate only, or the value only, at fixed block inputs."""
  u = F.silu(g) * v
  if mode == 'both':
    uS = F.silu(g - Gs) * (v - Vs)
  elif mode == 'gate':
    uS = F.silu(g - Gs) * v
  else:
    uS = F.silu(g) * (v - Vs)
  return s * (u - uS).mean(-1) / sigma


# ------------------------------------------------------------------------------------------------ matched nulls

def deciles(x) -> np.ndarray:
  """Decile 0..9 of every entry by average rank (ties share a decile)."""
  x = np.asarray(x, np.float64)
  return np.minimum((10 * (rankdata(x) - 0.5) / len(x)).astype(int), 9)


def matched_draws(near: Sequence[int], pool: Sequence[int], count_dec: np.ndarray, norm_dec: np.ndarray,
                  n_draws: int, rng) -> np.ndarray:
  """(n_draws, len(near)): for each Near10 class k, a class of the pool in the same decile of image count and of part
  norm (fallbacks: the same norm decile, then the nearest norm deciles), no class twice within a draw."""
  pool = np.asarray(pool)
  cands = []
  for k in near:
    c = pool[(count_dec[pool] == count_dec[k]) & (norm_dec[pool] == norm_dec[k])]
    if len(c) < 2:
      c = pool[norm_dec[pool] == norm_dec[k]]
    if len(c) < 2:
      c = pool[np.argsort(np.abs(norm_dec[pool] - norm_dec[k]), kind='stable')[:20]]
    cands.append(c)
  out = np.empty((n_draws, len(near)), np.int64)
  for d in range(n_draws):
    used = set()
    for i, c in enumerate(cands):
      pick = int(rng.choice(c))
      for _ in range(20):
        if pick not in used:
          break
        pick = int(rng.choice(c))
      used.add(pick)
      out[d, i] = pick
  return out


def mis_classes(sel: Selection, c: int, n: int = 20, seed: int = 0) -> List[int]:
  """20 selected classes of c's stratum (any stratum for controls) unrelated to c (R_mis)."""
  st = sel.rows[c]['stratum']
  near_c = set(sel.near10(c))
  cand = [k for k in sel.selected if k != c and (sel.rows[k]['stratum'] == st or st not in ('dense', 'iso'))
          and sel.D[c, k] > 4 and k not in near_c and c not in sel.near10(k)]
  rng = np.random.default_rng([seed, 4242, c])
  return sorted(int(k) for k in rng.choice(cand, min(n, len(cand)), replace=False))


def part_norms(parts: Dict[str, dict], keys) -> np.ndarray:
  """(G,) Frobenius norm of each group's parts over the given maps (weights and biases)."""
  tot = 0
  for k in keys:
    r = parts[k]
    tot = tot + (r['C'].float() / r.get('C_scale', 1.0)).pow(2).sum((1, 2)).double()
    tot = tot + (r['Cb'].float() / r.get('Cb_scale', 1.0)).pow(2).sum(1).double()
  return np.sqrt(np.asarray(tot))


# --------------------------------------------------------------------------------------------------- P1, S3

def p1_unit(g, v, s, sig, P_cls, P_own, near, draws, own_set, nown_sets, c, sg, sv):
  """All knockout numbers of one unit. P_cls = (P^g, P^v) (G_cls, N) class parts; P_own = (P^g, P^v) of the own split
  (fine, or the class split for own = whole class). Returns a dict of arrays (see summarize_p1)."""
  Pg, Pv = P_cls
  Og, Ov = P_own
  idx = torch.as_tensor(np.concatenate([np.asarray(near)[None], np.asarray(draws)]), device=Pg.device)
  Gc, Vc = Pg[idx].sum(1), Pv[idx].sum(1)                               # (1 + draws, N): Near10, then the draws
  out = {}
  for mode in ('both', 'gate', 'value'):
    Lc = knockout(g, v, Gc, Vc, s, sig, mode)
    out[f'near_{mode}'], out[f'rand_{mode}'] = float(Lc[0]), Lc[1:].cpu().numpy()
  lin = (sg * Gc.mean(1) + sv * Vc.mean(1)) / sig
  out['lin_near'], out['lin_rand'] = float(lin[0]), lin[1:].cpu().numpy()
  if own_set is not None:
    so = [own_set] + list(nown_sets)
    Go, Vo = set_sums(Og, so), set_sums(Ov, so)
    for mode in ('both', 'gate', 'value'):
      Lo = knockout(g, v, Go, Vo, s, sig, mode)
      out[f'own_{mode}'], out[f'nown_{mode}'] = float(Lo[0]), Lo[1:].cpu().numpy()
    lo = (sg * Go.mean(1) + sv * Vo.mean(1)) / sig
    out['lin_own'], out['lin_nown'] = float(lo[0]), lo[1:].cpu().numpy()
  # single class parts (S3) and the whole own class (O_whole)
  Lk = knockout(g, v, Pg, Pv, s, sig, 'both').cpu().numpy()
  out['L_class'] = Lk
  return out


def summarize_p1(units: List[dict], c: int, nown_cls: Sequence[int], prefix='') -> dict:
  """F, O, PI, the null, the linear version, universality and S3 from per-unit knockouts (see the module docstring)."""
  if not units:
    return {'n_units': 0}
  res = {'n_units': len(units)}
  for mode in ('both', 'gate', 'value'):
    tag = '' if mode == 'both' else f'_{mode}'
    Fn = np.array([u[f'near_{mode}'] - np.median(u[f'rand_{mode}']) for u in units])
    res[f'F{tag}'] = float(np.median(Fn))
    has_own = f'own_{mode}' in units[0]
    if has_own and len(units[0][f'nown_{mode}']):
      On = np.array([u[f'own_{mode}'] - np.median(u[f'nown_{mode}']) for u in units])
      res[f'O{tag}'] = float(np.median(On))
  F_, O_ = res['F'], res.get('O')
  # nulls: draw j as Near10, N_own class j mod m as own (median over the other N_own classes)
  rand = np.stack([u['rand_both'] for u in units])                       # (units, draws)
  Fj = np.median(rand - np.median(rand, 1, keepdims=True), 0)            # (draws,)
  res['F_null_median'], res['F_pct'] = float(np.median(Fj)), pct_rank(F_, Fj)
  if O_ is not None:
    nown = np.stack([u['nown_both'] for u in units])                     # (units, m)
    m = nown.shape[1]
    Oj = np.array([np.median(nown[:, j] - np.median(np.delete(nown, j, 1), 1)) if m > 1 else 0.0 for j in range(m)])
    res['O_pct'] = pct_rank(O_, Oj)
    null = np.abs(Fj) + np.abs(Oj[np.arange(len(Fj)) % m])
    res['def_threshold'] = float(np.percentile(null, 95))
    res['FO_size'] = abs(F_) + abs(O_)
    res['defined'] = bool(res['FO_size'] > res['def_threshold'])
    res['PI'] = float((F_ - O_) / (abs(F_) + abs(O_))) if abs(F_) + abs(O_) > 0 else 0.0
    res['signature'] = bool(O_ > np.percentile(Oj, 99) and F_ <= np.median(Fj))
    if m > 1:                                                             # pseudo-own: the first N_own class
      O_ps = Oj[0]
      Oj_ps = np.array([np.median(nown[:, j] - np.median(np.delete(nown, [0, j] if j else [0], 1), 1))
                        for j in range(1, m)]) if m > 2 else np.array([0.0])
      res['pseudo_signature'] = bool(O_ps > np.percentile(Oj_ps, 99) and F_ <= np.median(Fj))
    res['F_lin_pct'] = float(np.median([pct_rank(u['lin_near'], u['lin_rand']) for u in units]))
    res['O_lin_pct'] = float(np.median([pct_rank(u['lin_own'], u['lin_nown']) for u in units]))
    # S3: own niche among the single class parts
    ranks, shares = [], []
    for u in units:
      Lk = np.delete(u['L_class'], c)
      ranks.append(pct_rank(u['own_both'], Lk))
      pos = max(0.0, u['own_both'])
      shares.append(pos / (pos + np.maximum(Lk, 0).sum()) if pos + np.maximum(Lk, 0).sum() > 0 else 0.0)
    res['own_rank'], res['own_share'] = float(np.median(ranks)), float(np.median(shares))
  else:
    res['F_lin_pct'] = float(np.median([pct_rank(u['lin_near'], u['lin_rand']) for u in units]))
  # whole own class (class split): calibration
  Lw = np.array([u['L_class'][c] - np.median(u['L_class'][list(nown_cls)]) if len(nown_cls) else np.nan
                 for u in units])
  res['O_whole'] = float(np.median(Lw))
  return {prefix + k: v for k, v in res.items()}


# --------------------------------------------------------------------------------------------------------- P2

@torch.no_grad()
def stage_excision(model, mods, parts, sets: Sequence[Sequence[int]], x, c: int, stages=None) -> np.ndarray:
  """X (len(sets), n_stages + 1): exact effects on mean zeta_c over x of removing each set's parts from the unit maps
  of one stage (block caching), and from the head (last column; linear in the head parts at fixed features)."""
  stages = range(len(model.stages)) if stages is None else stages
  seq = layer_seq(model)
  names = [n for n, _ in seq]
  base_z = model(x)
  base = float(zeta(base_z, c).mean())
  feats = model.features(x)
  cache, act = {}, x
  starts = {s: names.index(f'down{s}') + 1 for s in stages}
  for li, (name, mod) in enumerate(seq):
    for s, st in starts.items():
      if li == st:
        cache[s] = act
    act = mod(act)
  out = np.zeros((len(sets), len(starts) + 1))
  for i, S in enumerate(sets):
    if len(S) == 0:
      continue
    for j, s in enumerate(stages):
      keys = [k for k in parts if k.startswith(f'stage{s}.')]
      undo = excise(mods, parts, S, keys)
      try:
        out[i, j] = base - float(zeta(run_from(model, seq, starts[s], cache[s]), c).mean())
      finally:
        undo()
    if 'head' in parts:
      r = parts['head']
      idx = torch.as_tensor(np.asarray(S), dtype=torch.long)
      Ch = (r['C'][idx].double().sum(0) / r.get('C_scale', 1.0)).to(feats.device, feats.dtype)
      Cbh = (r['Cb'][idx].double().sum(0) / r.get('Cb_scale', 1.0)).to(feats.device, feats.dtype)
      dz = feats @ Ch.T + Cbh
      out[i, -1] = float(zeta(dz, c).mean())
  return out


def lambda3(X: np.ndarray) -> dict:
  """X (1 + m, stages + head): own row first, then the N_own rows. Null-subtracted shares (see the docstring);
  'lambda3' is the share of the last stage (stage 3 of the ConvNeXt)."""
  own, nown = X[0], X[1:]
  m, ns = len(nown), X.shape[1] - 1
  if m == 0:
    return {}
  Xs = own - np.median(nown, 0)
  pos = np.maximum(Xs[:ns], 0)
  tot = pos.sum()
  null = np.array([np.maximum(nown[j, :ns] - np.median(np.delete(nown, j, 0), 0)[:ns], 0).sum() if m > 1 else 0.0
                   for j in range(m)])
  hp = max(0.0, Xs[ns])
  return {'X_stage': Xs[:ns].tolist(), 'X_head': float(Xs[ns]), 'lambda3': float(pos[-1] / tot) if tot > 0 else None,
          'sumXpos': float(tot), 'sumXpos_p95': float(np.percentile(null, 95)),
          'defined': bool(tot > np.percentile(null, 95)),
          'head_share': float(hp / (tot + hp)) if tot + hp > 0 else None, 'X_own_raw': own.tolist()}


def head_q(snap: dict, c: int) -> Optional[float]:
  """Mean over head row c (weights and bias) of 1 / (sqrt(v_hat) + eps) (snapshot of a run with --track_head)."""
  st = snap.get('attr', {}).get('head')
  if st is None:
    return None
  c2 = 1 - BETA2 ** st['t']
  q = 1.0 / ((st['v'][c].double() / c2).sqrt() + EPS)
  qb = 1.0 / ((st['vb'][c].double() / c2).sqrt() + EPS)
  return float(torch.cat([q, qb[None]]).mean())


def unit_quietness(snap: dict, model) -> Optional[np.ndarray]:
  """q (n_units,): mean over the unit's gate and value rows (weights and bias) of 1 / (sqrt(v_hat) + eps)."""
  A = snap.get('attr', {})
  out = []
  for b, blk in model.blocks():
    rows = []
    for m in ('pw_gate', 'pw_value'):
      st = A.get(f'{b}.{m}')
      if st is None:
        return None
      c2 = 1 - BETA2 ** st['t']
      q = 1.0 / ((st['v'].double() / c2).sqrt() + EPS)
      qb = 1.0 / ((st['vb'].double() / c2).sqrt() + EPS)
      rows.append(torch.cat([q, qb[:, None]], 1))
    out.append(torch.cat(rows, 1).mean(1))
  return torch.cat(out).numpy()


# --------------------------------------------------------------------------------------------------------- P3

def unit_effects(parts: Dict[str, dict], grads: Dict[str, torch.Tensor], sets: Sequence[Sequence[int]], model,
                 dtype=torch.float64) -> torch.Tensor:
  """E (S, n_cls, n_units): first-order effect of each set's parts in rows n of the gate and value maps (weights and
  bias) on the gradients' objectives, units in unit_list order."""
  out = []
  for b, blk in model.blocks():
    acc = 0
    for m in ('pw_gate', 'pw_value'):
      k = f'{b}.{m}'
      r, gr = parts[k], grads[k].to(dtype)
      U, d = r['C'].shape[1], r['C'].shape[2]
      PW = torch.stack([(r['C'][torch.as_tensor(list(s), dtype=torch.long)].to(dtype).sum(0) if len(s)
                         else torch.zeros(U, d, dtype=dtype)) for s in sets]) / r.get('C_scale', 1.0)
      PB = torch.stack([(r['Cb'][torch.as_tensor(list(s), dtype=torch.long)].to(dtype).sum(0) if len(s)
                         else torch.zeros(U, dtype=dtype)) for s in sets]) / r.get('Cb_scale', 1.0)
      GW, GB = gr[:, :U * d].reshape(-1, U, d).cpu(), gr[:, U * d:].cpu()
      acc = acc + torch.einsum('sud,cud->scu', PW, GW) + PB[:, None, :] * GB[None, :, :]
    out.append(acc)
  return torch.cat(out, 2)


def own_weights(E_own: np.ndarray, E_nown: np.ndarray) -> np.ndarray:
  """w_n = max(0, E_own(n) - median over N_own of E(n)); E_nown (m, n_units)."""
  return np.maximum(E_own - np.median(E_nown, 0), 0) if len(E_nown) else np.maximum(E_own, 0)


def weighted(w, x):
  w = np.asarray(w, np.float64)
  return float((w * x).sum() / w.sum()) if w.sum() > 0 else None


def p3_class(E_own, E_nown, rel, rel_mis, flow=None, flowD=None, q=None) -> dict:
  w = own_weights(E_own, E_nown)
  m = len(E_nown)
  null = np.array([own_weights(E_nown[j], np.delete(E_nown, j, 0)).sum() for j in range(m)]) if m > 1 else np.zeros(1)
  res = {'sum_w': float(w.sum()), 'sum_w_p95': float(np.percentile(null, 95)),
         'defined': bool(w.sum() > np.percentile(null, 95)), 'R_own': weighted(w, rel),
         'R_mis': weighted(w, rel_mis) if rel_mis is not None else None}
  if flow is not None:
    res['R_flow'] = weighted(flow, rel)
    res['R_flow_mis'] = weighted(flow, rel_mis) if rel_mis is not None else None
    if res['R_own'] is not None and res['R_flow'] is not None:
      res['Delta_flow'] = res['R_own'] - res['R_flow']
    if flowD is not None:
      res['R_flowD'] = weighted(flowD, rel)
    if q is not None:
      ok = flow > 0
      res['spearman_wf_q'] = spearman(w[ok] / flow[ok], q[ok]) if ok.sum() > 2 else None
  if q is not None:
    res['spearman_w_q'] = spearman(w, q)
  return res


# --------------------------------------------------------------------------------------------------------- S1

def builder_matrices(atlas: dict, model) -> List[dict]:
  """Per unit (unit_list order): top20 classes, the (20, K) contribution matrix raw[j, k] = max(0, s^v V[n, c'_j, k])
  + max(0, s^g G[n, c'_j, k]) and its cross-write version M (M[j, c'_j] = 0)."""
  out = []
  for b, blk in model.blocks():
    rec = atlas[b]
    for n in range(blk.pw_gate.out_features):
      top = [int(c) for c in rec['poles'][n][:20]]
      sv = torch.sign(rec['Vm'][n, top]).float()[:, None]
      sg = torch.sign(rec['Gm'][n, top]).float()[:, None]
      M = (sv * rec['value_writers'][n, :20].float()).clamp_min(0) \
        + (sg * rec['gate_writers'][n, :20].float()).clamp_min(0)
      raw = M.double().numpy()
      M[torch.arange(20), torch.as_tensor(top)] = 0
      out.append({'top': top, 'M': M.double().numpy(), 'raw': raw})
  return out


def builders(bm: dict, c: int) -> np.ndarray:
  """beta^x (K,) of one unit for class c: cross-writes, c' not in {c, k}, beta(c) = 0."""
  rows = [j for j, k in enumerate(bm['top']) if k != c]
  beta = bm['M'][rows].sum(0)
  beta[c] = 0.0
  return beta


def s1_class(bms: List[dict], units: Sequence[int], w: Optional[np.ndarray], c, near, draws) -> dict:
  def rbp(j):
    beta = builders(bms[j], c)
    tot = beta.sum()
    if tot <= 0:
      return None, None
    share = beta[near].sum() / tot
    null = beta[draws].sum(1) / tot
    top = set(bms[j]['top'])
    pm = sum(k in top for k in near)
    pnull = np.array([sum(int(k) in top for k in d) for d in draws])
    return pct_rank(share, null), pct_rank(pm, pnull)
  res = {}
  vals = [rbp(j) for j in units]
  vals = [v for v in vals if v[0] is not None]
  if vals:
    res['RBpx_K'], res['RPp_K'] = float(np.median([v[0] for v in vals])), float(np.median([v[1] for v in vals]))
  if w is not None and w.sum() > 0:
    js = np.flatnonzero(w > 0)
    vals = [(j, rbp(j)) for j in js]
    vals = [(j, v) for j, v in vals if v[0] is not None]
    if vals:
      ww = np.array([w[j] for j, _ in vals])
      res['RBpx_w'] = float((ww * np.array([v[0] for _, v in vals])).sum() / ww.sum())
      res['RPp_w'] = float((ww * np.array([v[1] for _, v in vals])).sum() / ww.sum())
  return res


# --------------------------------------------------------------------------------------------------------- S2

@torch.no_grad()
def s2_sigma(model, parts_f: Dict[str, dict], groups: Sequence[int], x, c: int) -> dict:
  """Per-image memorization on T_c (x: one image per own group, same order): see the module docstring."""
  st = unit_star(model, x)
  feats = model.features(x)
  m = len(groups)
  idx = torch.as_tensor(np.asarray(groups), dtype=torch.long)
  A = {s: torch.zeros(m, m, dtype=torch.float64) for s in range(len(model.stages))}
  for b, d in st.items():
    s = int(b.split('.')[0][5:])
    h = d['h']                                                        # (N, U, d)
    rg, rv = parts_f[f'{b}.pw_gate'], parts_f[f'{b}.pw_value']
    Cg = rg['C'][idx].to(h.device, h.dtype) / rg.get('C_scale', 1.0)     # (m, U, d)
    Cv = rv['C'][idx].to(h.device, h.dtype) / rv.get('C_scale', 1.0)
    bg = rg['Cb'][idx].to(h.device, h.dtype) / rg.get('Cb_scale', 1.0)
    bv = rv['Cb'][idx].to(h.device, h.dtype) / rv.get('Cb_scale', 1.0)
    P = torch.einsum('iud,xud->ixu', Cg + Cv, h) + (bg + bv)[:, None, :]     # (groups, images, U)
    A[s] += P.abs().sum(2).double().cpu()
  out = {}
  off = ~torch.eye(m, dtype=torch.bool)
  for s, a in A.items():
    num = a.diagonal()                                                # A(i, x_i)
    den = torch.stack([a[off[:, i], i].mean() for i in range(m)])     # mean over j != i of A(j, x_i)
    out[f'sigma_stage{s}'] = float((num / den.clamp_min(1e-30)).median())
  if 'head' in parts_f:
    r = parts_f['head']
    Ch = r['C'][idx].to(feats.device, feats.dtype) / r.get('C_scale', 1.0)    # (m, K, d)
    Cb = r['Cb'][idx].to(feats.device, feats.dtype) / r.get('Cb_scale', 1.0)
    PH = (torch.einsum('id,xd->ix', Ch[:, c], feats) + Cb[:, c, None]).abs().double().cpu()
    num = PH.diagonal()
    den = torch.stack([PH[off[:, i], i].mean() for i in range(m)])
    out['sigma_head'] = float((num / den.clamp_min(1e-30)).median())
    full = Ch.pow(2).sum((1, 2)) + Cb.pow(2).sum(1)
    row = Ch[:, c].pow(2).sum(1) + Cb[:, c].pow(2)
    out['head_row_share'] = float((row / full.clamp_min(1e-30)).median())
  return out


# ------------------------------------------------------------------------------------------------------- probes

# ------------------------------------------------------------------------------------------------------- measure

def class_counts(spec, n_classes, labels_path=''):
  if spec is not None:
    return np.bincount(spec['labels'][spec['keep']], minlength=n_classes)
  if labels_path and os.path.exists(labels_path):
    return np.bincount(np.load(labels_path), minlength=n_classes)
  return np.ones(n_classes, int)


def membership(top20: np.ndarray, n_classes: int) -> np.ndarray:
  """(n_units, n_classes) bool: k in top20(n)."""
  M = np.zeros((len(top20), n_classes), bool)
  M[np.arange(len(top20))[:, None], top20] = True
  return M


def measure(model, sel: Selection, spec, parts_c, parts_f, V, Vpos, Q, Qy, device, *, run_set=None, snap=None,
            pole_R=None, atlas=None, flow=None, train_x=None, noise_x=None, counts=None, classes=None,
            n_draws=N_DRAWS, seed=0, dtype=None, log=print, check_units=16, p2=True) -> dict:
  """Per-class records of P1, P2, P3, S1 to S3 and the accuracy numbers for one run and checkpoint (see the module
  docstring). parts_f None: own = the whole class (class split), as in phase 0. Also returns '_DeltaQ' (n_units,
  n_classes) for S7 (not for JSON)."""
  t0 = time.time()
  dtype = dtype or next(model.parameters()).dtype
  n_classes = int(model.head.weight.shape[0])
  classes = sorted(sel.rows) if classes is None else list(classes)
  units = unit_list(model)
  blocks = [b for b, _ in model.blocks()]
  n_stages = len(model.stages)
  fine = parts_f is not None
  own_split = parts_f if fine else parts_c
  own_of = (lambda k: [int(g) for g in own_groups(spec, k)]) if fine else (lambda k: [int(k)])
  # probe statistics (sigma, ubar, R) and the pole classes
  ps = probe_stats(model, Q, Qy, n_classes, device, dtype=dtype)
  sigma = torch.cat([ps[b]['sigma'] for b in blocks]).double().numpy()
  R_pole = pole_R if pole_R is not None else torch.cat([ps[b]['R'] for b in blocks]).numpy()
  top20 = np.argsort(-R_pole, 1, kind='stable')[:, :20]
  member = membership(top20, n_classes).astype(np.float64)
  log(f'probe stats {time.time() - t0:.0f}s')
  # carriers: mean ablation on Q (every class) and on V_sel (the analysed classes, one pass for all)
  ubar = {b: ps[b]['ubar'].to(device, dtype) for b in blocks}
  Qy_t = torch.as_tensor(np.asarray(Qy))
  zQ = logits_of(model, Q, device, dtype=dtype)
  DQ = unit_ablation(model, Q, Qy_t, ubar)                     # uint8 images stay on the CPU, normalized per batch
  DeltaQ = torch.zeros(len(units), n_classes, dtype=torch.float64).index_add_(1, Qy_t, DQ)
  DeltaQ /= torch.bincount(Qy_t, minlength=n_classes).clamp_min(1).double()[None]
  xs = {c: probes(sel, c, V, Vpos, noise_x) for c in classes}
  xsel = torch.cat([xs[c][0::2] for c in classes])
  ysel = torch.cat([torch.full((len(xs[c][0::2]),), c) for c in classes])
  DV = unit_ablation(model, xsel, ysel, ubar)
  log(f'unit ablation {time.time() - t0:.0f}s')
  # class tables for the matched nulls
  ukeys = [k for k in parts_c if k != 'head']
  norms = part_norms(parts_c, ukeys)[:n_classes]
  cdec, ndec = deciles(counts if counts is not None else np.ones(n_classes)), deciles(norms)
  # first-order effects per unit (P3, on V_est) and per region (P2, on all probe images)
  mods = tracked_modules(model, list(own_split))
  grads_est = class_gradients(model, mods, [xs[c][1::2] for c in classes], classes, device, dtype)
  grads_all = class_gradients(model, mods, [xs[c] for c in classes], classes, device, dtype)
  set_classes = sorted(set(sel.rows) | set(classes))   # every special class's own set (the nulls), even with --only
  set_id = {k: i for i, k in enumerate(set_classes)}
  E_unit = unit_effects(own_split, grads_est, [own_of(k) for k in set_classes], model).numpy()  # (S, n_cls, n_units)
  E_reg = by_region(group_effects(own_split, grads_all))
  E_reg_w = by_region(group_effects(parts_c, {k: grads_all[k] for k in parts_c if k in grads_all}))
  log(f'first-order effects {time.time() - t0:.0f}s')
  up_c = unit_map_parts(parts_c, device, dtype)
  up_o = unit_map_parts(parts_f, device, dtype) if fine else up_c
  q_units = unit_quietness(snap, model) if snap is not None else None
  recs, pos, nulls = [], 0, {}
  for ci, c in enumerate(classes):
    row = sel.rows[c]
    x = normalize(xs[c], device).to(dtype)
    near, pool = sel.near10(c), sel.pi(c)
    nown = sel.n_own(c)
    draws = matched_draws(near, pool, cdec, ndec, n_draws, np.random.default_rng([seed, c]))
    rec = {'cls': c, 'name': row['name'], 'kind': row['kind'], 'set': row['set'], 'stratum': row['stratum'],
           'level': row['level'], 'cell': sel.cell(c), 'condition': sel.condition(c, run_set), 'n_nown': len(nown),
           'probe': probe_kind(sel, c, noise_x)}
    has_own = len(own_of(c)) > 0
    # accuracy on the probes (zeta_c against the Q images of Far(c) and Near10(c))
    z = logits_of(model, x, device, dtype=dtype)
    sv_, sq = zeta(z, c).numpy(), zeta(zQ, c).numpy()
    rec.update(top1=float((z.argmax(1) == c).float().mean()),
               top5=float((torch.topk(z, 5, 1).indices == c).any(1).float().mean()),
               auroc_far=auroc(sv_, sq[np.isin(Qy, sel.far(c))]), auroc_near=auroc(sv_, sq[np.isin(Qy, near)]))
    # carriers K(c) on V_sel
    nsel = len(x[0::2])
    spec_eff = specific_effect(DV[:, pos:pos + nsel].mean(1), DeltaQ, c)
    pos += nsel
    K = carriers(spec_eff)
    rec['K'] = [list(units[j]) for j in K]
    rec['K_effect'] = [float(spec_eff[j]) for j in K]
    # P1 and S3 on V_est at p*
    kb = sorted({units[j][0] for j in K})
    st_est, st_all = unit_star(model, x[1::2], blocks=kb), unit_star(model, x, blocks=kb)
    per_unit = []
    for j in K:
      b, n = units[j]
      d = st_est[b]
      s = float(torch.sign(st_all[b]['u'][:, n].mean())) or 1.0
      sg = float(torch.sign(st_all[b]['g'][:, n].mean())) or 1.0
      sv = float(torch.sign(st_all[b]['v'][:, n].mean())) or 1.0
      per_unit.append(p1_unit(d['g'][:, n], d['v'][:, n], s, sigma[j], unit_parts(up_c, b, n, d['h'][:, n]),
                              unit_parts(up_o, b, n, d['h'][:, n]), near, draws, own_of(c) if has_own else None,
                              [own_of(k) for k in nown], c, sg, sv))
    rec.update(summarize_p1(per_unit, c, nown))
    if per_unit:                                     # per-unit knockouts and their nulls (saved, amendment 1 C3)
      nulls[c] = {k: np.array([u[k] for u in per_unit]) for k in ('near_both', 'rand_both', 'own_both', 'nown_both')
                  if k in per_unit[0]}
    # P2: stage excision of Own_m and N_own (and of the whole classes), exact and first order
    if p2 and has_own and nown:
      X = stage_excision(model, mods, own_split, [own_of(c)] + [own_of(k) for k in nown], x, c)
      rec['P2'] = lambda3(X)
      fo = np.array([[float(E_reg[f'stage{s}'][own_of(k), ci].sum()) for s in range(n_stages)]
                     + [float(E_reg['head'][own_of(k), ci].sum()) if 'head' in E_reg else 0.0] for k in [c] + nown])
      rec['P2_fo'] = lambda3(fo)
    if p2 and nown:
      Xw = stage_excision(model, tracked_modules(model, list(parts_c)), parts_c, [[c]] + [[k] for k in nown], x, c)
      rec['P2_whole'] = lambda3(Xw)
      fo = np.array([[float(E_reg_w[f'stage{s}'][k, ci]) for s in range(n_stages)]
                     + [float(E_reg_w['head'][k, ci]) if 'head' in E_reg_w else 0.0] for k in [c] + nown])
      rec['P2_whole_fo'] = lambda3(fo)
    if snap is not None:
      rec['head_q'] = head_q(snap, c)
    # P3: own weights over all units
    w = None
    if has_own and nown:
      Eo = E_unit[set_id[c], ci]
      En = np.stack([E_unit[set_id[k], ci] for k in nown])
      rel = member[:, near].sum(1) / 20
      mis = mis_classes(sel, c)
      rel_mis = np.mean([member[:, sel.near10(k)].sum(1) / 20 for k in mis], 0) if mis else None
      fw = fD = None
      if flow is not None and c in flow['classes']:
        fi = flow['classes'].index(c)
        fw, fD = flow['f'][fi], flow['fD'][fi]
        if not np.any(fD):
          fD = None
      rec['P3'] = p3_class(Eo, En, rel, rel_mis, fw, fD, q_units)
      w = own_weights(Eo, En)
      if check_units:                                  # exact excision of Own_m from rows n, top units by E_own(n)
        top = np.argsort(-Eo)[:check_units]
        rec['P3']['check'] = unit_row_check(model, mods, own_split, own_of(c), [units[j] for j in top], x[1::2], c,
                                            Eo[top])
    if atlas is not None:
      rec['S1'] = s1_class(atlas, K, w, c, near, draws)
    if train_x is not None and fine and c in train_x:
      xt, groups = train_x[c]
      rec['S2'] = s2_sigma(model, parts_f, groups, normalize(xt, device).to(dtype), c)
    recs.append(rec)
    if ci % 10 == 0:
      log(f'class {ci + 1}/{len(classes)} {time.time() - t0:.0f}s')
  return {'classes': recs, 'sigma': sigma.tolist(), 'units': [list(u) for u in units], '_DeltaQ': DeltaQ,
          '_nulls': nulls}


@torch.no_grad()
def unit_row_check(model, mods, parts, groups, units, x, c, E) -> dict:
  """Exact effect of removing the groups' parts from rows n of the gate and value maps of one unit at a time, against
  the first-order E(n): correlation and median ratio."""
  base = float(zeta(model(x), c).mean())
  idx = torch.as_tensor(np.asarray(groups), dtype=torch.long)
  X = []
  for b, n in units:
    saved = []
    for m in ('pw_gate', 'pw_value'):
      k = f'{b}.{m}'
      lin, r = mods[k], parts[k]
      dW = (r['C'][idx][:, n].double().sum(0) / r.get('C_scale', 1.0)).to(lin.weight.device, lin.weight.dtype)
      db = (r['Cb'][idx][:, n].double().sum(0) / r.get('Cb_scale', 1.0)).to(lin.bias.device, lin.bias.dtype)
      saved.append((lin, lin.weight[n].clone(), lin.bias[n].clone()))
      lin.weight[n] -= dW
      lin.bias[n] -= db
    try:
      X.append(base - float(zeta(model(x), c).mean()))
    finally:
      for lin, w, bb in saved:
        lin.weight[n] = w
        lin.bias[n] = bb
  X, E = np.array(X), np.asarray(E)
  ok = np.abs(X) > 0
  return {'corr': float(np.corrcoef(E, X)[0, 1]) if len(X) > 2 and X.std() > 0 and E.std() > 0 else None,
          'median_ratio': float(np.median(E[ok] / X[ok])) if ok.any() else None, 'E': E.tolist(), 'X': X.tolist()}


# ---------------------------------------------------------------------------------------------- flow weights

def lr_schedule(cfg: dict):
  lr, warm, steps = cfg.get('lr', 2e-3), cfg.get('warmup', 1250), cfg.get('steps', 25020)
  return lambda s: lr * ((s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm))))


def interval_lr(cfg: dict, ckpts: Sequence[int]) -> Dict[int, float]:
  """l_t: summed learning rate over [t, next checkpoint) for each checkpoint but the last."""
  sched = lr_schedule(cfg)
  ck = sorted(ckpts)
  return {t: float(sum(sched(s) for s in range(t, t2))) for t, t2 in zip(ck[:-1], ck[1:])}


def per_example_row_norms(model, x, y, y2, lam, D: Optional[Dict[str, tuple]] = None) -> torch.Tensor:
  """(B, n_units): ||lam_i grad_{rows n of gate and value} L_i|| per example, L_i = -(lam_i log p_y + (1 - lam_i)
  log p_y2) of the mixed image; with D, every gradient coordinate is first scaled by D (AdamW's 1 / (sqrt(v_hat) +
  eps), (weight, bias) per map)."""
  return row_norms(model, x, y, y2, lam, [D])[0]


def row_norms(model, x, y, y2, lam, Ds: Sequence[Optional[Dict[str, tuple]]]) -> List[torch.Tensor]:
  """per_example_row_norms for several scalings Ds (None: unscaled) from one forward and backward pass."""
  caps, hooks = {}, []
  for b, blk in model.blocks():
    for m in ('pw_gate', 'pw_value'):
      lin = getattr(blk, m)
      key = f'{b}.{m}'

      def fwd(mod, inp, out, key=key):
        caps[key] = {'h': inp[0].detach()}
        out.register_hook(lambda g, key=key: caps[key].__setitem__('d', g.detach()))
      hooks.append(lin.register_forward_hook(fwd))
  try:
    logp = F.log_softmax(model(x), -1)
    L = -(lam * logp.gather(1, y[:, None])[:, 0] + (1 - lam) * logp.gather(1, y2[:, None])[:, 0])
    L.sum().backward()
  finally:
    for h in hooks:
      h.remove()
  outs = [[] for _ in Ds]
  for b, blk in model.blocks():
    sqs = [0] * len(Ds)
    for m in ('pw_gate', 'pw_value'):
      cp = caps[f'{b}.{m}']
      B = cp['h'].shape[0]
      h = cp['h'].reshape(B, -1, cp['h'].shape[-1])
      d = cp['d'].reshape(B, -1, cp['d'].shape[-1])
      G = torch.bmm(d.transpose(1, 2), h)                 # (B, U, d)
      Gb = d.sum(1)                                       # (B, U)
      for i, D in enumerate(Ds):
        Gs, Gbs = (G, Gb) if D is None else (G * D[f'{b}.{m}'][0], Gb * D[f'{b}.{m}'][1])
        sqs[i] = sqs[i] + Gs.pow(2).sum(2) + Gbs.pow(2)
    for i in range(len(Ds)):
      outs[i].append(sqs[i].sqrt() * lam[:, None])
  return [torch.cat(o, 1).detach() for o in outs]


def step_scales(snap_attr: Dict[str, dict], device=None) -> Dict[str, tuple]:
  """{unit map: (D_weight, D_bias)}, D = 1 / (sqrt(v / (1 - beta2^t)) + eps), from a snapshot's 'attr' (t > 0)."""
  D = {}
  for k, st in snap_attr.items():
    if k == 'head' or st.get('v') is None:
      continue
    assert st['t'] > 0, f'{k}: no second moment at t = 0'
    c2 = 1 - BETA2 ** st['t']
    D[k] = tuple((1.0 / ((v.to(device).double() / c2).sqrt() + EPS)).to(v.dtype) for v in (st['v'], st['vb']))
  return D


def flow_snapshot(snaps: Dict[int, str], t: int) -> Optional[int]:
  """The snapshot whose second moment scales the interval starting at checkpoint t: snap_t itself when t > 0, else
  (t = 0: v is still zero, D would be 1 / eps) or when snap_t is missing (E1: history snapshots at other steps) the
  nearest snapshot with t > 0."""
  return t if t > 0 and t in snaps else nearest_snapshot(snaps, t)


def cmd_flow(a):
  """f and f^D for every special class (T_c: own training images; see the module docstring)."""
  from rcdl.convnext import build_convnext
  from rcdl.imagenet import GPUTrainLoader
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  sel = Selection(a.classes, a.wordnet, a.visual)
  spec = load_spec(a.run)
  ck = sorted(int(f[6:11]) for f in os.listdir(a.run) if f.startswith('model_') and f.endswith('.pt'))
  lt = interval_lr(cfg, ck)
  loader = GPUTrainLoader(a.train, 64, dev, keep=spec['keep'] if spec is not None else None,
                          labels=spec['labels'] if spec is not None else None,
                          noise=spec.get('noise') if spec is not None else None)
  classes = [int(c) for c in a.only.split(',')] if a.only else sorted(sel.rows)
  T = {}
  for c in classes:
    if spec is not None:
      idx = np.flatnonzero(np.isin(spec['group'], own_groups(spec, c)))
    else:                                   # phase 0: 100 images of the whole class
      idx = np.flatnonzero(loader.labels == c)
      idx = idx[np.random.default_rng([a.seed, c]).permutation(len(idx))[:100]]
    T[c] = idx
  keep = loader.index
  model = build_convnext(cfg['size']).to(dev).eval()
  f = np.zeros((len(classes), sum(b.pw_gate.out_features for _, b in model.blocks())))
  fD = np.zeros_like(f)
  snaps, used = snapshot_files(a.run), {}
  for t, l in lt.items():
    model.load_state_dict(torch.load(os.path.join(a.run, f'model_{t:05d}.pt'), map_location=dev))
    ss = flow_snapshot(snaps, t)
    used[t] = ss
    D = step_scales(torch.load(snaps[ss], map_location='cpu', weights_only=False)['attr'], dev) if ss is not None \
      else None
    for ci, c in enumerate(classes):
      idx = np.repeat(T[c], a.n_aug)
      if not len(idx):
        continue
      rng = np.random.default_rng([a.seed, t, c])    # partners, crops and lam: a function of (seed, t, c)
      part = keep[rng.integers(len(keep), size=len(idx))]
      acc, accD = 0, 0
      for i in range(0, len(idx), 64):
        x1 = loader.decode_crop(idx[i:i + 64], rng)
        x2 = loader.decode_crop(part[i:i + 64], rng)
        lam = torch.as_tensor(rng.beta(0.8, 0.8, size=len(x1)), dtype=x1.dtype, device=dev)
        x = lam[:, None, None, None] * x1 + (1 - lam[:, None, None, None]) * x2
        y = loader.labels_gpu[torch.as_tensor(idx[i:i + 64], device=dev)]
        y2 = loader.labels_gpu[torch.as_tensor(part[i:i + 64], device=dev)]
        nr = row_norms(model, x, y, y2, lam, [None] + ([D] if D is not None else []))
        acc = acc + nr[0].sum(0).cpu().numpy()
        if D is not None:
          accD = accD + nr[1].sum(0).cpu().numpy()
        model.zero_grad(set_to_none=True)
      f[ci] += l * acc / len(idx)
      if D is not None:
        fD[ci] += l * accD / len(idx)
    print(json.dumps({'checkpoint': t, 'l_t': l, 'snapshot': ss}), flush=True)
  np.savez(a.out or os.path.join(a.run, 'rare_flow.npz'), classes=np.array(classes), f=f, fD=fD,
           ckpts=np.array(sorted(lt)), l=np.array([lt[t] for t in sorted(lt)]),
           snap_steps=np.array([-1 if used[t] is None else used[t] for t in sorted(lt)]))


def load_flow(path):
  if not path or not os.path.exists(path):
    return None
  d = np.load(path)
  return {'classes': [int(c) for c in d['classes']], 'f': d['f'], 'fD': d['fD']}


# ---------------------------------------------------------------------------------------------- churn (S4)

def compact_sums(parts: Dict[str, dict], sets: Dict[str, Sequence[int]], keys) -> Dict[str, dict]:
  """group_sums form (rcdl.attribution.GroupAttribution.group_sums) from a compact state, for interval_buckets."""
  out = {}
  for k in keys:
    r = parts[k]
    rec = {'a': r['a'], 'ab': r['ab'], 't': r['t'], 'sets': {}, 'R': r['R'].float(), 'Rb': r['Rb'].float()}
    for name, ids in sets.items():
      idx = torch.as_tensor(list(ids), dtype=torch.long)
      rec['sets'][name] = {'C': (r['C'][idx].double().sum(0) / r.get('C_scale', 1.0)).float(),
                           'Cb': (r['Cb'][idx].double().sum(0) / r.get('Cb_scale', 1.0)).float()}
    out[k] = rec
  return out


def churn_interval(s1: dict, s2: dict, init: dict, unit_ids, units, s_g, s_v, sigma, h2: dict, h1: dict) -> dict:
  """Traffic of each set over one interval on the units: new writes on fixed inputs h_t' and drift of the earlier
  parts, S_t (h_t' - h_t), both at p*_t'. s1, s2: compact_sums at t, t' (sets partition the groups); h2, h1: {block:
  (N, U, d)} inputs at p*_t' under the models of t' and t. Returns {set: (new, drift)} (sum over units of mean_x
  |oriented gate + value term| / sigma_n). interval_buckets asserts that the buckets add up to W_t' - W_t."""
  from rcdl.attribution import interval_buckets
  keys = sorted({f'{units[j][0]}.{m}' for j in unit_ids for m in ('pw_gate', 'pw_value')})
  bk = interval_buckets({k: s1[k] for k in keys}, {k: s2[k] for k in keys}, {k: init[k] for k in keys})
  out = {}
  for w in s1[keys[0]]['sets']:
    new = drift = 0.0
    for j in unit_ids:
      b, n = units[j]
      x2, x1 = h2[b][:, n].double().cpu(), h1[b][:, n].double().cpu()
      tn = td = 0.0
      for m, sgn in (('pw_gate', s_g[j]), ('pw_value', s_v[j])):
        k = f'{b}.{m}'
        tn = tn + sgn * (x2 @ bk[k]['write'][w][n].double() + bk[k]['write_b'][w][n].double())
        td = td + sgn * ((x2 - x1) @ s1[k]['sets'][w]['C'][n].double())
      new += float(tn.abs().mean()) / sigma[j]
      drift += float(td.abs().mean()) / sigma[j]
    out[w] = (new, drift)
  return out


def inputs_at(model, x, blocks, p) -> Dict[str, torch.Tensor]:
  """{block: (N, U, d)} the block inputs h of this model at given positions p[block] (N, U)."""
  hs, _ = block_inputs(model, x, set(blocks))
  out = {}
  for b in blocks:
    h = hs[b].flatten(1, 2)
    out[b] = torch.gather(h, 1, p[b][:, :, None].expand(-1, -1, h.shape[-1]))
  return out


def cmd_churn(a):
  """S4 (descriptive): set traffic over all checkpoint intervals on K(c) (from a measure file)."""
  from rcdl.convnext import build_convnext
  dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False   # p* and h as in measure
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  sel = Selection(a.classes, a.wordnet, a.visual)
  spec = load_spec(a.run)
  meas = json.load(open(a.measure))
  init = torch.load(os.path.join(a.run, 'attr_init.pt'), map_location='cpu', weights_only=False)
  pre = a.split + '_'
  ck = sorted(int(f[len(pre):len(pre) + 5]) for f in os.listdir(a.run)
              if f.startswith(pre) and f[len(pre):len(pre) + 5].isdigit())
  V, Vpos, _, _ = load_images(a, sel)
  noise_x = noise_images([spec] + [load_npz(p) for p in a.noise_specs.split(',') if p], sel)
  models = {}

  def model_at(t):
    if t not in models:
      if len(models) > 2:
        models.pop(min(models))
      m = build_convnext(cfg['size']).to(dev).eval()
      m.load_state_dict(torch.load(os.path.join(a.run, f'model_{t:05d}.pt'), map_location=dev))
      models[t] = m
    return models[t]
  units = unit_list(model_at(ck[-1]))
  uidx = {tuple(u): j for j, u in enumerate(units)}
  sigma = np.array(meas['sigma'])
  recs = [r for r in meas['classes'] if r['K']]
  res = {}
  for t1, t2 in zip(ck[:-1], ck[1:]):
    p1, p2 = load_parts(os.path.join(a.run, f'{pre}{t1:05d}.pt')), load_parts(os.path.join(a.run, f'{pre}{t2:05d}.pt'))
    f1 = f2 = None
    if spec is not None and a.split == 'attr' and os.path.exists(os.path.join(a.run, f'attrF_{t2:05d}.pt')):
      f1 = load_parts(os.path.join(a.run, f'attrF_{t1:05d}.pt'))
      f2 = load_parts(os.path.join(a.run, f'attrF_{t2:05d}.pt'))
    m1, m2 = model_at(t1), model_at(t2)
    for rec in recs:
      c = rec['cls']
      K = [uidx[tuple(u)] for u in rec['K']]
      near, far = sel.near10(c), sel.far(c)
      rest = [k for k in range(len(sel.order)) if k not in set(near) | set(far)]
      keys = sorted({f'{units[j][0]}.{m}' for j in K for m in ('pw_gate', 'pw_value')})
      x = normalize(probes(sel, c, V, Vpos, noise_x), dev)[1::2]
      blocks = sorted({units[j][0] for j in K})
      st2 = unit_star(m2, x, blocks)
      h2 = {b: st2[b]['h'] for b in blocks}
      h1 = inputs_at(m1, x, blocks, {b: st2[b]['p'] for b in blocks})
      s_g = {j: float(torch.sign(st2[units[j][0]]['g'][:, units[j][1]].mean())) or 1.0 for j in K}
      s_v = {j: float(torch.sign(st2[units[j][0]]['v'][:, units[j][1]].mean())) or 1.0 for j in K}
      sets = {'near': near, 'far': far, 'rest': rest}
      out = churn_interval(compact_sums(p1, sets, keys), compact_sums(p2, sets, keys), init, K, units, s_g, s_v,
                           sigma, h2, h1)
      own = [int(g) for g in own_groups(spec, c)] if f1 is not None else []
      if own:
        fs = {'own': own, 'rest_f': [g for g in range(len(spec['group_class'])) if g not in set(own)]}
        out.update(churn_interval(compact_sums(f1, fs, keys), compact_sums(f2, fs, keys), init, K, units, s_g, s_v,
                                  sigma, h2, h1))
      res.setdefault(c, {'series': []})['series'].append(
        {'t1': t1, 't2': t2, **{w: list(v) for w, v in out.items() if not w.startswith('rest')}})
    print(json.dumps({'interval': [t1, t2]}), flush=True)
  for c, r in res.items():
    tot = {w: sum(s[w][0] for s in r['series'] if w in s) for w in ('near', 'far', 'own')}
    nfar = len(sel.far(c))
    r['traffic'] = tot
    r['tau'] = (tot['far'] / nfar) / (tot['near'] / 10) if tot['near'] > 0 and nfar else None
  json.dump(clean_json({str(c): r for c, r in res.items()}),
            open(a.out or os.path.join(a.run, f'rare_churn_{a.split}.json'), 'w'), indent=1)


# --------------------------------------------------------------------------------------------------- phase 0

@torch.no_grad()
def m4_collapse(model, mods, parts_c, sel: Selection, classes, V, Vpos, Q, Qy, counts, device, n_null=20, seed=0,
                dtype=None) -> list:
  """Old M4 on a few classes: removing the 10 Near10 class parts from all unit maps at once (X^U on V_c) against
  matched draws, with the probe top-1 after removal."""
  dtype = dtype or next(model.parameters()).dtype
  ukeys = [k for k in parts_c if k != 'head']
  norms = part_norms(parts_c, ukeys)
  cdec, ndec = deciles(counts), deciles(norms)
  yQ = torch.as_tensor(np.asarray(Qy))
  out = []
  for c in classes:
    x = normalize(V[Vpos[c]], device).to(dtype)
    base = float(zeta(model(x), c).mean())
    draws = matched_draws(sel.near10(c), sel.pi(c), cdec, ndec, n_null, np.random.default_rng([seed, c]))
    res = {'cls': c}
    for name, S in [('near', sel.near10(c))] + [(f'rand{i}', list(d)) for i, d in enumerate(draws)]:
      undo = excise(mods, parts_c, S, ukeys)
      try:
        X = base - float(zeta(model(x), c).mean())
        acc = None
        if name in ('near', 'rand0'):
          acc = float((logits_of(model, Q, device, dtype=dtype).argmax(1) == yQ).float().mean())
      finally:
        undo()
      res[name] = {'X': X, 'probe_top1': acc}
    nulls = [res[f'rand{i}']['X'] for i in range(n_null)]
    res['X_near'], res['X_null_median'] = res['near']['X'], float(np.median(nulls))
    res['X_null_iqr'] = float(np.subtract(*np.percentile(nulls, [75, 25])))
    out.append(res)
  return out


def s7_consolidation(DeltaQ: torch.Tensor, units, sel: Selection, atlas: Optional[List[dict]] = None) -> dict:
  """S7 for the stratum candidates of experiments/rare_select.py (dense: nW >= 8 and cos10 <= its 60th percentile;
  iso: nW <= 4 and cos10 >= its median): C1, PR over the units of stages 2 and 3 on Q, the own niche share N on the
  top carrier (with the atlas: N = raw[c, c] / sum_k raw[c, k], the share of c's own writes among the writes that
  build c's response on that unit, defined when c is one of its top-20 poles, as the atlas stores writer vectors of
  the poles only), and Spearman correlations with cos10 and nW."""
  nW, cos10 = sel.nW, sel.cos10
  cand = {'dense': np.flatnonzero((nW >= 8) & (cos10 <= np.percentile(cos10, 60))),
          'iso': np.flatnonzero((nW <= 4) & (cos10 >= np.percentile(cos10, 50)))}
  late = [j for j, (b, n) in enumerate(units) if b.startswith('stage2') or b.startswith('stage3')]
  Dq = DeltaQ[late].numpy()
  rows = []
  for st, cs in cand.items():
    for c in cs:
      t = Dq[:, c] - np.median(np.delete(Dq, c, 1), 1)
      tp = np.maximum(t, 0)
      tot = tp.sum()
      r = {'cls': int(c), 'stratum': st, 'cos10': float(cos10[c]), 'nW': int(nW[c]),
           'C1': float(tp.max() / tot) if tot > 0 else None,
           'PR': float(tot ** 2 / (tp ** 2).sum()) if tot > 0 else None}
      if atlas is not None:                         # c's own row of the top carrier's builder matrix, if c is a pole
        bm = atlas[late[int(np.argmax(t))]]
        if int(c) in bm['top']:
          beta = bm['raw'][bm['top'].index(int(c))]
          r['N'] = float(beta[c] / beta.sum()) if beta.sum() > 0 else None
        else:
          r['N'] = None
      rows.append(r)
  out = {'classes': rows}
  for k in ('C1', 'PR', 'N'):
    ok = [r for r in rows if r.get(k) is not None]
    if len(ok) > 2:
      out[f'spearman_{k}_cos10'] = spearman([r[k] for r in ok], [r['cos10'] for r in ok])
      out[f'spearman_{k}_nW'] = spearman([r[k] for r in ok], [r['nW'] for r in ok])
  return out


# --------------------------------------------------------------------------------------------------- statistics

def bootstrap_median(x, n: int = 10000, seed: int = 0) -> dict:
  """Median, its 95% bootstrap interval over classes, two-sided p = 2 min(P(b <= 0), P(b >= 0)) and the one-sided
  p values P(b <= 0) (for a positive effect) and P(b >= 0)."""
  x = np.asarray([v for v in x if v is not None and np.isfinite(v)], np.float64)
  if len(x) == 0:
    return {'n': 0}
  rng = np.random.default_rng(seed)
  b = np.median(x[rng.integers(len(x), size=(n, len(x)))], 1)
  lo, hi = np.percentile(b, [2.5, 97.5])
  p_le, p_ge = float((b <= 0).mean()), float((b >= 0).mean())
  return {'n': len(x), 'median': float(np.median(x)), 'ci95': [float(lo), float(hi)],
          'p_two_sided': min(1.0, 2 * min(p_le, p_ge)), 'p_pos': p_le, 'p_neg': p_ge}


def holm(p: Dict[str, float]) -> Dict[str, float]:
  """Holm-adjusted p values."""
  items = sorted((v, k) for k, v in p.items() if v is not None)
  m, out, run = len(items), {}, 0.0
  for i, (v, k) in enumerate(items):
    run = max(run, min(1.0, (m - i) * v))
    out[k] = run
  return out


def wilcoxon(d) -> dict:
  """Wilcoxon signed-rank test of paired differences d (zeros dropped), normal approximation with tie correction."""
  d = np.asarray([v for v in d if v is not None and np.isfinite(v) and v != 0], np.float64)
  n = len(d)
  if n == 0:
    return {'n': 0, 'p': None}
  r = rankdata(np.abs(d))
  Wp = float(r[d > 0].sum())
  mu = n * (n + 1) / 4
  _, cnt = np.unique(np.abs(d), return_counts=True)
  var = n * (n + 1) * (2 * n + 1) / 24 - (cnt ** 3 - cnt).sum() / 48
  z = (Wp - mu) / math.sqrt(var) if var > 0 else 0.0
  p = math.erfc(abs(z) / math.sqrt(2))
  return {'n': n, 'W_plus': Wp, 'z': z, 'p': p}


def binom_sf(k: int, n: int, p: float) -> float:
  """P(X >= k) for X ~ Binomial(n, p)."""
  return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def mcnemar_exact(x, y) -> dict:
  """Exact McNemar test of paired booleans (the same classes; pairs with a None dropped): b = x and not y, c = y and
  not x; p_one_sided = P(B >= b) for B ~ Binomial(b + c, 1/2) (x true more often than y)."""
  pairs = [(bool(u), bool(v)) for u, v in zip(x, y) if u is not None and v is not None]
  b = sum(u and not v for u, v in pairs)
  c = sum(v and not u for u, v in pairs)
  n = b + c
  p_ge = binom_sf(b, n, 0.5) if n else 1.0
  p_le = 1.0 - binom_sf(b + 1, n, 0.5) if n else 1.0
  return {'n': len(pairs), 'x_only': b, 'y_only': c, 'both': sum(u and v for u, v in pairs),
          'p_one_sided': p_ge, 'p_two_sided': min(1.0, 2 * min(p_ge, p_le))}


def floor_agrees(res: dict) -> bool:
  """The same-seed floor (Sec. 6, amendment 1 A2): the Wilcoxon test of D - D0 has p < 0.05 and its z has the sign of
  the contrast's median, so D exceeds D0 in the direction of the effect."""
  w, m = res.get('floor_wilcoxon'), res.get('median')
  return bool(w and w.get('p') is not None and w['p'] < 0.05 and m is not None and m != 0
              and np.sign(w['z']) == np.sign(m))


def noise_state(fit, noise_pi, real_pi, min_fit: float = 0.5) -> str:
  """The noise control (Sec. 4.3 and 7, amendment 1 A1/B1): 'uninformative' when the fit check fails (median top-1
  on T below min_fit) or no noise or real PI is defined; otherwise 'pass' when the median noise PI is below the median
  real PI (defined PIs), else 'fail'."""
  if not len(fit) or np.median(fit) < min_fit or not len(noise_pi) or not len(real_pi):
    return 'uninformative'
  return 'pass' if np.median(noise_pi) < np.median(real_pi) else 'fail'


def controls_state(noise: str, relabel: str) -> str:
  """'fail' if either positive control fails, else 'uninformative' if either is (no relabel control: uninformative),
  else 'pass'."""
  if 'fail' in (noise, relabel):
    return 'fail'
  return 'uninformative' if 'uninformative' in (noise, relabel) else 'pass'


CONTROL_NOTES = {'pass': None, 'fail': 'positive controls failed: primaries are unvalidated',
                 'uninformative': 'noise control uninformative (fit check failed or no defined noise PI), or no '
                                  'relabel control: the primaries are not validated by it'}


def get(rec: dict, path: str):
  """rec['a']['b'] for path 'a.b' (None if missing)."""
  for p in path.split('.'):
    if not isinstance(rec, dict) or p not in rec:
      return None
    rec = rec[p]
  return rec


def cmd_compare(a):
  """Pairing over runs, floors, bootstrap, Holm, the seed rule, universality, controls and the decision rules
  (protocol v2 with amendment 1)."""
  load = lambda fs: [json.load(open(f)) for f in (fs or [])]
  rare, c0, rare1, c1 = load(a.rare), load(a.c0), load(a.rare1), load(a.c1)
  e1, e1r = load(a.e1), load(a.e1r)

  def by_cls(run, cond=None):                      # sorted by class id: results do not depend on the file order
    return {r['cls']: r for r in sorted(run['classes'], key=lambda r: r['cls'])
            if cond is None or r['condition'] == cond}

  def rare_recs(runs):
    out = {}
    for o in runs:
      out.update(by_cls(o, 'rare'))
    return dict(sorted(out.items()))

  def common_recs(runs):
    out = {}
    for o in runs:
      out.update(by_cls(o, 'common'))
    return dict(sorted(out.items()))

  # values that exist only where their definedness gate holds (amendment 1, A5)
  pi_d = lambda r: r.get('PI') if r.get('defined') else None
  l3_d = lambda r: get(r, 'P2.lambda3') if get(r, 'P2.defined') else None
  rown_d = lambda r: get(r, 'P3.R_own') if get(r, 'P3.defined') else None
  med = lambda v: float(np.median(v)) if len(v) else None

  def stats(rr, cc, other):
    prim = sorted(c for c, r in rr.items() if r['kind'] == 'selected' and r['level'] in (10, 100))
    res = {}
    # P1 (primary): median PI over the primary classes where PI is defined; 'no route' classes are counted
    pi = [rr[c]['PI'] for c in prim if rr[c].get('defined')]
    res['P1'] = bootstrap_median(pi)
    res['P1']['n_no_route'] = len(prim) - len(pi)
    # P1 secondaries (amendment 1, A4/B2/B7): all classes, 'no route' as 0, symmetric F_pct gates, tail counts, and
    # the protocol's other versions (gate-only, value-only, linear, gate opening, absent classes)
    fpct = [rr[c].get('F_pct') for c in prim if rr[c].get('F_pct') is not None]
    lo, hi = sum(v <= 0.05 for v in fpct), sum(v >= 0.95 for v in fpct)
    sym = [rr[c]['PI'] for c in prim if rr[c].get('PI') is not None and rr[c].get('F_pct') is not None
           and (rr[c]['F_pct'] <= 0.025 or rr[c]['F_pct'] >= 0.975)]
    res['P1_secondary'] = {
      'raw_all': bootstrap_median([rr[c].get('PI') for c in prim]),
      'no_route_as_zero': bootstrap_median([rr[c]['PI'] if rr[c].get('defined') else 0.0 for c in prim]),
      'symmetric_gate_2.5_97.5': bootstrap_median(sym),
      'F_pct_tails': {'n': len(fpct), 'le_0.05': lo, 'ge_0.95': hi,
                      'p_low': binom_sf(lo, len(fpct), 0.05) if fpct else None,
                      'p_high': binom_sf(hi, len(fpct), 0.05) if fpct else None}}
    versions = (('F_gate', lambda r: r.get('F_gate')), ('F_value', lambda r: r.get('F_value')),
                ('F_lin_pct_minus_half', lambda r: r['F_lin_pct'] - 0.5 if r.get('F_lin_pct') is not None else None),
                ('O_lin_pct_minus_half', lambda r: r['O_lin_pct'] - 0.5 if r.get('O_lin_pct') is not None else None),
                ('F_gate_minus_O_gate', lambda r: r['F_gate'] - r['O_gate'] if r.get('F_gate') is not None
                 and r.get('O_gate') is not None else None))
    res['P1_versions'] = {k: {'all': bootstrap_median([fn(rr[c]) for c in prim]),
                              'defined': bootstrap_median([fn(rr[c]) for c in prim if rr[c].get('defined')])}
                          for k, fn in versions}
    absent = [c for c, r in rr.items() if r['kind'] == 'selected' and r['level'] == 0]
    res['P1_versions']['F_absent'] = bootstrap_median([rr[c].get('F') for c in absent])

    # P2 (primary: the exact whole-class reference in C0) and the first-order and Own_m references (secondary, A3)
    def p2(ref_of):
      d2, f2 = [], []
      for c in prim:
        l3, ref = l3_d(rr[c]), ref_of(cc.get(c, {}))
        if l3 is not None and ref is not None:
          d2.append(l3 - ref)
          o = l3_d(other.get(c, {}))                  # floor D0: the same contrast where c is common (other set)
          if o is not None:
            f2.append((l3 - ref) - (o - ref))
      r = bootstrap_median(d2)
      r['floor_wilcoxon'] = wilcoxon(f2) if f2 else None
      return r
    res['P2'] = p2(lambda r: get(r, 'P2_whole.lambda3'))
    res['P2_ref_first_order'] = p2(lambda r: get(r, 'P2_whole_fo.lambda3'))
    res['P2_ref_own_m'] = p2(l3_d)
    l3r = [l3_d(rr[c]) for c in prim if l3_d(rr[c]) is not None]
    res['P2']['median_lambda3_rare'] = med(l3r)
    res['P2']['frac_lambda3_gt_half'] = float(np.mean([v > 0.5 for v in l3r])) if l3r else None

    # P3 (primary), and the same in the common run
    def p3(recs):
      df = [get(recs.get(c, {}), 'P3.Delta_flow') for c in prim if get(recs.get(c, {}), 'P3.defined')]
      df = [v for v in df if v is not None]            # None: no flow weights for the class
      r = bootstrap_median(df)
      gap = [get(recs[c], 'P3.R_flow') - get(recs[c], 'P3.R_mis') for c in prim if c in recs
             and get(recs[c], 'P3.R_flow') is not None and get(recs[c], 'P3.R_mis') is not None]
      r['delta'] = 0.2 * float(np.median(gap)) if gap else None
      r['n_excluded'] = len(prim) - len(df)
      return r
    res['P3'] = p3(rr)
    om = [rown_d(rr[c]) - get(rr[c], 'P3.R_mis') for c in prim
          if rown_d(rr[c]) is not None and get(rr[c], 'P3.R_mis') is not None]
    res['P3_own_minus_mis'] = bootstrap_median(om)
    # the common run (C0, or C1 at seed 1): P1 and P3 by the same statistics (secondary, A10)
    res['common'] = {'P1': bootstrap_median([pi_d(cc[c]) for c in prim if c in cc]), 'P3': p3(cc)}
    # rare - C0 contrasts, with the same-seed floor D0 = other-set run - C0
    for name, fn in (('PI', pi_d), ('F', lambda r: r.get('F')), ('O', lambda r: r.get('O')), ('lambda3', l3_d),
                     ('R_own', rown_d), ('auroc_far', lambda r: r.get('auroc_far')),
                     ('auroc_near', lambda r: r.get('auroc_near')), ('own_share', lambda r: r.get('own_share')),
                     ('fit_T', lambda r: r.get('fit_T'))):
      D, D0 = [], []
      for c in prim:
        r, q, o = fn(rr[c]), fn(cc.get(c, {})), fn(other.get(c, {}))
        if r is not None and q is not None:
          D.append(r - q)
          D0.append((o - q) if o is not None else None)
      pairs = [(x, y) for x, y in zip(D, D0) if y is not None]          # every key is written (n = 0: not measured)
      res[f'rare_minus_C0_{name}'] = bootstrap_median(D)
      res[f'rare_minus_C0_{name}']['floor_wilcoxon'] = wilcoxon([x - y for x, y in pairs]) if pairs else None
    # S5 crossover: rare run against the other set's run (where c is common)
    for name, fn in (('PI', pi_d), ('F', lambda r: r.get('F')), ('lambda3', l3_d),
                     ('auroc_far', lambda r: r.get('auroc_far'))):
      d = [fn(rr[c]) - fn(other[c]) for c in prim if c in other and fn(rr[c]) is not None
           and fn(other[c]) is not None]
      res[f'S5_rare_minus_other_{name}'] = bootstrap_median(d)

    # S1: builders beyond pole membership, intercept of RBp^x on (RPp - 0.5) over classes, bootstrap interval; on K(c)
    # and weighted by own effect (RBpx_w on RPp_w, A8). xn, yn: output names (the K version keeps its original keys)
    def s1(xk, yk, xn, yn):
      pts = [(get(rr[c], f'S1.{xk}'), get(rr[c], f'S1.{yk}')) for c in prim]
      pts = np.array([p for p in pts if None not in p], np.float64).reshape(-1, 2)
      if len(pts) <= 3:
        return None
      def icept(P):
        A = np.stack([np.ones(len(P)), P[:, 0] - 0.5], 1)
        return float(np.linalg.lstsq(A, P[:, 1], rcond=None)[0][0])
      rng = np.random.default_rng(0)
      boot = [icept(pts[rng.integers(len(pts), size=len(pts))]) for _ in range(2000)]
      return {'n': len(pts), 'intercept': icept(pts), 'ci95': np.percentile(boot, [2.5, 97.5]).tolist(),
              f'median_{yn}': float(np.median(pts[:, 1])), f'median_{xn}': float(np.median(pts[:, 0])),
              f'median_{yn}_C0': med([get(cc[c], f'S1.{yk}') for c in prim if c in cc
                                      and get(cc[c], f'S1.{yk}') is not None])}
    for key, xk, yk, xn, yn in (('S1_RBpx_at_RPp_half', 'RPp_K', 'RBpx_K', 'RPp', 'RBpx'),
                                ('S1_RBpx_w_at_RPp_w_half', 'RPp_w', 'RBpx_w', 'RPp_w', 'RBpx_w')):
      r = s1(xk, yk, xn, yn)
      if r is not None:
        res[key] = r
    # S2: per-image memorization, rare - C0 on the same images, per stage and head, with the floor (A7)
    s2, s2f = {}, {}
    for c in prim:
      a_, b_, o_ = get(rr[c], 'S2'), get(cc.get(c, {}), 'S2'), get(other.get(c, {}), 'S2')
      if a_ and b_:
        for k in a_:
          if k.startswith('sigma') and a_[k] is not None and b_.get(k) is not None:
            s2.setdefault(k, []).append(a_[k] - b_[k])
            if o_ and o_.get(k) is not None:
              s2f.setdefault(k, []).append(a_[k] - o_[k])     # D - D0 = (rare - C0) - (other - C0)
    res['S2_rare_minus_C0'] = {k: {**bootstrap_median(v), 'floor_wilcoxon': wilcoxon(s2f[k]) if s2f.get(k) else None}
                               for k, v in s2.items()}

    # universality (primary test: binomial at the pseudo rate) and the McNemar tests (secondary, A6/A10)
    def univ(recs):
      sig = [recs[c].get('signature') for c in prim if c in recs and recs[c].get('signature') is not None]
      ps = [recs[c].get('pseudo_signature') for c in prim if c in recs and recs[c].get('pseudo_signature') is not None]
      k, n, kp = int(sum(sig)), len(sig), int(sum(ps))
      return {'n': n, 'signatures': k, 'pseudo_signatures': kp,
              'p_excess': binom_sf(k, n, (kp + 1) / (n + 2)) if n else None}
    res['universality'] = univ(rr)
    sig_r = [rr[c].get('signature') for c in prim]
    res['universality']['mcnemar_vs_pseudo'] = mcnemar_exact(sig_r, [rr[c].get('pseudo_signature') for c in prim])
    res['universality']['common'] = univ(cc)
    res['universality']['mcnemar_rare_vs_common'] = mcnemar_exact(sig_r, [cc.get(c, {}).get('signature')
                                                                          for c in prim])
    # cells (descriptive; PI where defined)
    cells = {}
    for c in rr:
      r = rr[c]
      key = f"{r['kind']}/{r['stratum']}/{r['level']}"
      cells.setdefault(key, []).append(r)
    res['cells'] = {k: {'n': len(v), 'n_defined': sum(bool(x.get('defined')) for x in v),
                        'median_PI': med([pi_d(x) for x in v if pi_d(x) is not None]),
                        'median_F': med([x['F'] for x in v if x.get('F') is not None])}
                    for k, v in sorted(cells.items())}

    # controls
    noise = [r for r in rr.values() if r['kind'] == 'noise']
    real = [pi_d(rr[c]) for c in prim if pi_d(rr[c]) is not None]
    npi = [pi_d(r) for r in noise if pi_d(r) is not None]
    fit = [r['top1'] for r in noise if r.get('top1') is not None]          # noise probes are its training images
    nF = [r['F'] for r in noise if r.get('F') is not None]
    state = noise_state(fit, npi, real)
    res['noise_control'] = {'n': len(noise), 'fit_T': fit, 'informative': bool(fit and np.median(fit) >= 0.5),
                            'cls': [r['cls'] for r in noise if r.get('top1') is not None],     # the order of fit_T
                            'n_defined': len(npi), 'median_PI': med(npi), 'median_PI_real': med(real),
                            'median_PI_raw': med([r['PI'] for r in noise if r.get('PI') is not None]),
                            'median_PI_real_raw': med([rr[c]['PI'] for c in prim if rr[c].get('PI') is not None]),
                            'auroc_noise_below_real': auroc(real, npi) if npi and real else None,
                            'median_F': med(nF), 'n_F_negative': sum(v < 0 for v in nF),
                            'R_own_minus_mis': [get(r, 'P3.R_own') - get(r, 'P3.R_mis') for r in noise
                                                if get(r, 'P3.R_own') is not None and get(r, 'P3.R_mis') is not None],
                            'state': state, 'passes': state == 'pass'}
    stat = lambda r: r['F_src'] - r['F'] if r.get('F_src') is not None and r.get('F') is not None else None
    rel = [c for c, r in rr.items() if r['kind'] == 'relabel' and stat(r) is not None]
    diff = [stat(rr[c]) for c in rel]
    D = [(stat(rr[c]) - stat(cc[c]), stat(other[c]) - stat(cc[c]) if c in other and stat(other[c]) is not None
          else None) for c in rel if c in cc and stat(cc[c]) is not None]
    res['relabel_control'] = {'n': len(diff), 'F_src_minus_F_label': diff, 'cls': rel,
                              'passes': bool(diff and np.median(diff) > 0),
                              'state': ('pass' if np.median(diff) > 0 else 'fail') if diff else 'uninformative',
                              'sign_test_p': binom_sf(sum(v > 0 for v in diff), sum(v != 0 for v in diff), 0.5)
                              if diff else None,
                              'rare_minus_C0': [x for x, _ in D],
                              'floor_wilcoxon': wilcoxon([x - y for x, y in D if y is not None]) if D else None}
    cal = [get(cc.get(c, {}), 'O_whole') for c in prim]
    cal = [v for v in cal if v is not None]
    thr = [(get(cc[c], 'O_whole'), get(cc[c], 'def_threshold')) for c in prim if c in cc]
    thr = [(o, t) for o, t in thr if o is not None and t is not None]
    orare = [rr[c]['O'] for c in prim if rr[c].get('O') is not None]
    res['C0_calibration'] = {'median_O_whole_C0': med(cal), 'median_O_rare': med(orare),
                             'frac_O_whole_above_C0_threshold': float(np.mean([abs(o) > t for o, t in thr]))
                             if thr else None}
    return res

  out = {}
  rr0 = rare_recs(rare)
  cc0 = by_cls(c0[0]) if c0 else {}
  out['seed0'] = s0 = stats(rr0, cc0, common_recs(rare))
  pv = {k: s0[k].get('p_two_sided') for k in ('P1', 'P2', 'P3')}
  out['holm_seed0'] = holm(pv)
  s1 = None
  if rare1:
    rr1 = rare_recs(rare1)
    # without C1 the protocol uses E1r for class-split measures (the P2 reference P2_whole, the C0 calibration)
    cc1 = by_cls(c1[0]) if c1 else (by_cls(e1r[0]) if e1r else {})
    out['seed1_reference'] = 'C1' if c1 else ('E1r' if e1r else None)
    out['seed1'] = s1 = stats(rr1, cc1, common_recs(rare1))
    rep = {}
    for k in ('P1', 'P2', 'P3'):
      m0, m1 = s0[k].get('median'), s1[k].get('median')
      if m0 is None or m1 is None:
        rep[k] = None
        continue
      p1s = s1[k]['p_pos'] if m0 > 0 else s1[k]['p_neg']
      rep[k] = {'same_sign': bool(np.sign(m0) == np.sign(m1)), 'p_one_sided': p1s,
                'replicated': bool(np.sign(m0) == np.sign(m1) and p1s < 0.05)}
    out['replication'] = rep
  # floors: C0 - E1 (instrumentation) and E1r - E1 (seed) on measures both runs have
  def floor(A, B, paths=('auroc_far', 'top1', 'F', 'O_whole', 'P2_whole.lambda3')):
    if not A or not B:
      return None
    a_, b_ = by_cls(A[0]), by_cls(B[0])
    res = {}
    for p in paths:
      d = [get(a_[c], p) - get(b_[c], p) for c in a_ if c in b_ and get(a_[c], p) is not None
           and get(b_[c], p) is not None]
      res[p] = {'median_abs': float(np.median(np.abs(d))) if d else None, 'median': float(np.median(d)) if d else None}
    return res
  out['floor_instrumentation_C0_minus_E1'] = floor(c0, e1)
  out['floor_seed_E1r_minus_E1'] = floor(e1r, e1)
  out['floor_seed_C1_minus_C0'] = floor(c1, c0, ('auroc_far', 'top1', 'F', 'O', 'PI', 'O_whole', 'P2_whole.lambda3'))
  # decision rules (section 7): a count against the essay needs Holm significance (P1, P2), the same-seed floor (P2,
  # the only rare - C0 primary contrast, in the effect's direction) and seed-1 replication; without seed-1 runs it is
  # only 'pending'. The flags below are secondary (amendment 1) and change no count.
  hp = out['holm_seed0']
  sig = lambda k: hp.get(k) is not None and hp[k] < 0.05
  rep = out.get('replication', {})
  replicated = lambda k: (rep.get(k) or {}).get('replicated') if rep else None
  d = s0['P3'].get('delta')
  below = lambda r, dd: bool(r.get('n') and dd is not None and r['ci95'][1] < -dd)

  def in_common(s, claim, sign=-1):
    """The claim holds in that seed's common run by the same statistic (P1, P3) or test (universality)."""
    if s is None:
      return None
    cm = s['common']
    if claim == 'P1':
      return bool(cm['P1'].get('n') and np.sign(cm['P1']['median']) == sign and cm['P1']['p_two_sided'] < 0.05)
    if claim == 'P3':
      return below(cm['P3'], cm['P3'].get('delta'))
    pe = s['universality']['common'].get('p_excess')
    return bool(pe is not None and pe < 0.05)

  def refs(s, sign):
    """P2 medians with the three references, and whether all have the given sign."""
    if s is None:
      return None
    m = {k: s[key].get('median') for k, key in (('exact', 'P2'), ('first_order', 'P2_ref_first_order'),
                                                ('own_m', 'P2_ref_own_m'))}
    return {**m, 'all_same_sign': all(v is not None and np.sign(v) == sign for v in m.values())}
  items = []
  if sig('P1') and s0['P1']['median'] < 0:
    items.append({'claim': 'median PI < 0', 'replicated': replicated('P1'), 'floor_ok': True,
                  'also_in_common': {'seed0': in_common(s0, 'P1'), 'seed1': in_common(s1, 'P1')}})
  if sig('P2') and s0['P2']['median'] < 0:
    items.append({'claim': 'median lambda_3 difference < 0', 'replicated': replicated('P2'),
                  'floor_ok': floor_agrees(s0['P2']), 'references': {'seed0': refs(s0, -1), 'seed1': refs(s1, -1)}})
  if below(s0['P3'], d):
    items.append({'claim': 'Delta_flow interval below -delta', 'floor_ok': True,
                  'replicated': below(s1['P3'], s1['P3'].get('delta')) if s1 else None,
                  'also_in_common': {'seed0': in_common(s0, 'P3'), 'seed1': in_common(s1, 'P3')}})
  u = s0['universality']
  if u.get('p_excess') is not None and u['p_excess'] < 0.05:
    items.append({'claim': 'excess of packing signatures', 'floor_ok': True,
                  'replicated': bool(s1['universality'].get('p_excess') is not None
                                     and s1['universality']['p_excess'] < 0.05) if s1 else None,
                  'also_in_common': {'seed0': in_common(s0, 'U'), 'seed1': in_common(s1, 'U')},
                  'mcnemar_vs_pseudo_p': {'seed0': u['mcnemar_vs_pseudo']['p_one_sided'],
                                          'seed1': s1['universality']['mcnemar_vs_pseudo']['p_one_sided']
                                          if s1 else None}})
  counts = [i['claim'] for i in items if i['floor_ok'] and i['replicated']]
  pending = [i['claim'] for i in items if i['floor_ok'] and i['replicated'] is None]
  flags = []
  for i in items:
    if i['claim'] not in counts + pending:
      continue
    for seed, v in (i.get('also_in_common') or {}).items():
      if v:
        flags.append(f"'{i['claim']}' also holds in the common run ({seed})")
    for seed, v in (i.get('references') or {}).items():
      if v and not v['all_same_sign']:
        flags.append(f"'{i['claim']}': the P2 references are not all negative ({seed})")
  ok_p3 = s0['P3'].get('n') and d is not None and s0['P3']['ci95'][0] >= -d and \
    (s0['P3_own_minus_mis'].get('median') or -1) > 0
  support = []
  if sig('P1') and s0['P1']['median'] > 0 and sig('P2') and s0['P2']['median'] > 0 and floor_agrees(s0['P2']) and \
     (s0['P2'].get('median_lambda3_rare') or 0) > 0.5 and ok_p3:
    support.append('all of P1 > 0, P2 > 0 (beyond the floor) with lambda_3 > 0.5, P3 within -delta with R_own > R_mis')
    for seed, s in (('seed0', s0), ('seed1', s1)):
      if in_common(s, 'P1', 1):
        flags.append(f"'P1 > 0' also holds in the common run ({seed})")
      r = refs(s, 1)
      if r and not r['all_same_sign']:
        flags.append(f"'P2 > 0': the P2 references are not all positive ({seed})")
  states = {'seed0': controls_state(s0['noise_control']['state'], s0['relabel_control']['state'])}
  if s1:
    states['seed1'] = controls_state(s1['noise_control']['state'], s1['relabel_control']['state'])
  out['decision'] = {'against_essay': counts, 'against_pending_seed1': pending, 'against_candidates': items,
                     'support_for_essay': support, 'secondary_flags': flags,
                     'controls_pass': states['seed0'] == 'pass', 'controls_state': states,
                     'replicated': {k: replicated(k) for k in ('P1', 'P2', 'P3')} if rep else 'no seed-1 runs',
                     'rule': 'a count against the essay needs Holm significance, the same-seed floor (P2, in the '
                             "effect's direction) and seed-1 replication",
                     'note': CONTROL_NOTES[states['seed0']],
                     'note_seed1': CONTROL_NOTES[states['seed1']] if s1 else None}
  json.dump(clean_json(out), open(a.out, 'w'), indent=1)
  print(json.dumps(clean_json(out['decision']), indent=1))


# ------------------------------------------------------------------------------------------------------- CLI

def load_images(a, sel: Selection):
  """V (validation images of the special classes and the relabel sources), Vpos, Q, Qy."""
  return load_probe_images(a.val, sel, a.val_cache, a.probe_cache, a.images)


def cmd_measure(a):
  from rcdl.convnext import build_convnext
  dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  log = lambda s: print(s, flush=True)
  cfg = json.load(open(os.path.join(a.run, 'config.json')))

  def load_model(step):
    st = torch.load(os.path.join(a.run, f'model_{step:05d}.pt'), map_location='cpu')
    m = build_convnext(cfg['size'], n_classes=st['head.weight'].shape[0])
    m.load_state_dict(st)
    return m.to(dev).eval()
  model = load_model(a.step)
  sel = Selection(a.classes, a.wordnet, a.visual)
  spec = load_spec(a.run)
  run_set = spec['meta']['run_set'] if spec is not None else None
  parts_c = load_parts(os.path.join(a.run, f'attr_{a.step:05d}.pt'))
  pf = os.path.join(a.run, f'attrF_{a.step:05d}.pt')
  parts_f = load_parts(pf) if os.path.exists(pf) and spec is not None and not a.phase0 else None
  V, Vpos, Q, Qy = load_images(a, sel)
  noise_x = noise_images([spec] + [load_npz(p) for p in a.noise_specs.split(',') if p], sel)
  steps = sorted(int(f[6:11]) for f in os.listdir(a.run) if f.startswith('model_') and f.endswith('.pt'))
  pole_step = a.pole_step or max(steps)
  pole_R = None
  if pole_step != a.step:                          # the final pole classes, fixed per run
    pm = load_model(pole_step)
    ps = probe_stats(pm, Q, Qy, int(pm.head.weight.shape[0]), dev)
    pole_R = torch.cat([ps[b]['R'] for b, _ in pm.blocks()]).numpy()
    del pm
  snaps = snapshot_files(a.run)                    # the checkpoint's own snapshot, else the nearest (E1: history/)
  snap_step = a.step if a.step in snaps and a.step > 0 else nearest_snapshot(snaps, a.step)
  snap = torch.load(snaps[snap_step], map_location='cpu', weights_only=False) if snap_step is not None else None
  atlas = None
  for f in (f'atlas_{a.step:05d}.pt', f'atlas_prelim_{a.step:05d}.pt'):
    if os.path.exists(os.path.join(a.run, f)):
      atlas = builder_matrices(torch.load(os.path.join(a.run, f), map_location='cpu', weights_only=False), model)
      break
  train_x = None
  if a.train and spec is not None and parts_f is not None and os.path.exists(os.path.join(a.train, 'data.bin')):
    from experiments.rare_common import train_center_u8
    train_x = {}
    for r in sel.sel['classes']:
      if r['level'] == 10:
        gids = own_groups(spec, r['cls'])
        idx = np.array([int(np.flatnonzero(spec['group'] == g)[0]) for g in gids])
        train_x[r['cls']] = (train_center_u8(a.train, idx, spec.get('noise', ())), [int(g) for g in gids])
  counts = class_counts(spec, int(model.head.weight.shape[0]), a.labels)
  out = measure(model, sel, spec, parts_c, parts_f, V, Vpos, Q, Qy, dev, run_set=run_set, snap=snap, pole_R=pole_R,
                atlas=atlas, flow=load_flow(a.flow), train_x=train_x, noise_x=noise_x, counts=counts,
                classes=[int(c) for c in a.only.split(',')] if a.only else None, n_draws=a.n_draws, seed=a.seed,
                log=log)
  DeltaQ, nulls = out.pop('_DeltaQ'), out.pop('_nulls')
  out.update({'run': a.run, 'step': a.step, 'run_set': run_set, 'own': 'Own_m (fine split)' if parts_f is not None
              else 'whole class (class split)', 'pole_step': pole_step, 'atlas': atlas is not None,
              'snapshot': snap is not None, 'snapshot_step': snap_step, 'flow': bool(a.flow)})
  for rec in out['classes']:                       # relabel controls: F with the source class's Near10
    if rec['kind'] == 'relabel' and rec.get('K'):
      rec['F_src'] = relabel_source_F(model, sel, rec, parts_c, V, Vpos, counts, dev, out['sigma'], a.n_draws, a.seed)
  if a.train and spec is not None and os.path.exists(os.path.join(a.train, 'data.bin')):
    from experiments.rare_common import train_center_u8
    for rec in out['classes']:                     # S6: train fit on T_c (center crops of the own training images)
      idx = np.flatnonzero(np.isin(spec['group'], own_groups(spec, rec['cls'])))
      if len(idx):
        xt = train_center_u8(a.train, idx, spec.get('noise', ()))
        hit = logits_of(model, xt, dev).argmax(1) == torch.as_tensor(spec['labels'][idx])
        rec['fit_T'], rec['n_T'] = float(hit.float().mean()), int(len(idx))
  if a.phase0:
    mods = tracked_modules(model, list(parts_c))
    m4cls = sorted(int(c) for c in np.random.default_rng(7).choice(sel.selected, a.m4, replace=False))
    out['M4'] = m4_collapse(model, mods, parts_c, sel, m4cls, V, Vpos, Q, Qy, counts, dev, seed=a.seed)
    out['S7'] = s7_consolidation(DeltaQ, unit_list(model), sel, atlas)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump(clean_json(out), open(a.out, 'w'), indent=1)
  np.savez_compressed(a.out.replace('.json', '') + '_arrays.npz', DeltaQ=DeltaQ.numpy(), sigma=np.array(out['sigma']),
                      **{f'{k}_{c}': v for c, d in nulls.items() for k, v in d.items()})
  log(f'wrote {a.out}')


def relabel_source_F(model, sel, rec, parts_c, V, Vpos, counts, dev, sigma, n_draws, seed=0) -> Optional[float]:
  """F of a relabel control computed with the source class's Near10 (c excluded) in place of its label's Near10, on
  the same carriers and probes (the source class's validation images)."""
  c = rec['cls']
  src = sel.rows[c]['source']
  near = [k for k in sel.order[src] if k != c][:10]
  units = unit_list(model)
  uidx = {tuple(u): j for j, u in enumerate(units)}
  K = [uidx[tuple(u)] for u in rec['K']]
  dtype = next(model.parameters()).dtype
  x = normalize(probes(sel, c, V, Vpos), dev).to(dtype)
  blocks = sorted({units[j][0] for j in K})
  st_est, st_all = unit_star(model, x[1::2], blocks), unit_star(model, x, blocks)
  up_c = unit_map_parts(parts_c, dev, dtype)
  ukeys = [k for k in parts_c if k != 'head']
  cdec, ndec = deciles(counts), deciles(part_norms(parts_c, ukeys)[:len(counts)])
  pool = [k for k in sel.pi(c) if k not in near and k != src]
  draws = matched_draws(near, pool, cdec, ndec, n_draws, np.random.default_rng([seed, 1, c]))
  Fn = []
  for j in K:
    b, n = units[j]
    d = st_est[b]
    s = float(torch.sign(st_all[b]['u'][:, n].mean())) or 1.0
    Pg, Pv = unit_parts(up_c, b, n, d['h'][:, n])
    sets = [[int(k) for k in near]] + [list(dd) for dd in draws]
    L = knockout(d['g'][:, n], d['v'][:, n], set_sums(Pg, sets), set_sums(Pv, sets), s, sigma[j])
    Fn.append(float(L[0]) - float(np.median(L[1:].cpu().numpy())))    # as summarize_p1 (np.median, not torch's lower)
  return float(np.median(Fn)) if Fn else None


def main():
  p = argparse.ArgumentParser()
  sub = p.add_subparsers(dest='cmd', required=True)
  for name in ('measure', 'flow', 'churn'):
    q = sub.add_parser(name)
    q.add_argument('--run', required=True)
    q.add_argument('--step', type=int, default=25020)
    q.add_argument('--val', default='/marimo/work/data/imagenet/data')
    q.add_argument('--classes', default='docs/data/rare/rare_classes_v2.json')
    q.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
    q.add_argument('--visual', default='docs/data/imagenet_dinov2_classdist.npz')
    q.add_argument('--probe_cache', default='/dev/shm/probe_8.pt')
    q.add_argument('--val_cache', default='/marimo/work/rare_val_v2_src.pt')
    q.add_argument('--images', default='', help='pre-built images {V, Vlabels, Q, Qy} (tests)')
    q.add_argument('--train', default='')
    q.add_argument('--labels', default='/dev/shm/imagenet256/labels.npy', help='class counts when there is no spec')
    q.add_argument('--out', default='')
    q.add_argument('--device', default=None)
    q.add_argument('--seed', type=int, default=0)
    q.add_argument('--noise_specs', default='/marimo/work/specs/spec_A.npz,/marimo/work/specs/spec_B.npz',
                   help='specs that flag the noise controls\' images (probes of noise controls in every run)')
  m = sub.choices['measure']
  m.add_argument('--flow', default='', help='rare_flow.npz (subcommand flow)')
  m.add_argument('--pole_step', type=int, default=0)
  m.add_argument('--n_draws', type=int, default=N_DRAWS)
  m.add_argument('--only', default='', help='comma-separated classes (tests, partial runs)')
  m.add_argument('--phase0', action='store_true', help='own = the whole class (class split), adds M4')
  m.add_argument('--m4', type=int, default=20)
  sub.choices['flow'].add_argument('--n_aug', type=int, default=4)
  sub.choices['flow'].add_argument('--only', default='', help='comma-separated classes (partial runs)')
  c = sub.choices['churn']
  c.add_argument('--measure', required=True)
  c.add_argument('--split', default='attr', choices=['attr', 'attrR'])
  q = sub.add_parser('compare')
  for k in ('rare', 'c0', 'rare1', 'c1', 'e1', 'e1r'):
    q.add_argument(f'--{k}', nargs='*', default=[])
  q.add_argument('--out', required=True)
  a = p.parse_args()
  if a.cmd == 'measure':
    a.out = a.out or os.path.join(a.run, f"rare_measure{'_phase0' if a.phase0 else ''}_{a.step:05d}.json")
    cmd_measure(a)
  elif a.cmd == 'flow':
    cmd_flow(a)
  elif a.cmd == 'churn':
    cmd_churn(a)
  else:
    cmd_compare(a)


if __name__ == '__main__':
  main()
