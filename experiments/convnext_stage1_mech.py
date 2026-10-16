"""Stage-1 reversal: the residual stream along a unit's write direction where the unit is active (X1), and its pole push
with its same-direction partners turned down (X2).

convnext_bias_paths.py finds that from about step 1,000 on, a stage-1 unit's 20 strongest classes push its response down
(late blocks: up), and convnext_unit_roles.py that most stage-1 units have a partner whose write direction is nearly
parallel (|cos| > 0.9; the 8-dimensional write space is nearly collapsed). Candidate mechanisms (review of 2026-10-16):
  M1 overshoot: same-direction partners already fill the stream along the unit's direction, so more of the unit hurts
  M4 cancellation: the unit's write cancels a useful incoming component of the stream along its direction
  M5 class-specific negative readout of the direction, independent of the stream's level
  M3 SwiGLU curvature: the push depends on the unit's own level, not on the stream's
This script measures what separates them. Notation as in convnext_bias_paths.py: for block b, unit n, image i of class
k and position p, u = SiLU(g) v, delta = dCE_i / du, the unit's top-20 classes P_n (by class-mean u* = max_p u), and the
pole coefficients pa_v = mean_{c in P_n} SiLU(Gm[n, c]), pa_g = mean_{c in P_n} SiLU'(Gm[n, c]) Vm[n, c].

X1, the stream state. The block writes w_n = gamma * W_down[:, n] per unit of u_n; e = w_n / |w_n|. At every position:
  s_in   = x_in . e                     the block's input along e
  s_own  = u_n |w_n|                    the unit's own write along e
  s_part = (omega - u_n w_n) . e        the other units' writes along e (with the pw_down bias write, also reported
                                        alone as s_bias), omega = x_out - x_in
  s_out  = x_out . e = s_in + s_own + s_part
  kappa  = SiLU(g) pa_v + SiLU'(g) v pa_g   (the derivative of the pole response through this position's bias gradient)
  push   = -kappa delta                 its sum over positions, class-averaged, is convnext_bias_paths' pole[k, n]
Roles per unit: top20 (P_n), relatives (WordNet relatives of P_n outside it, with --wordnet), rest (all other classes).
Each quantity is a class value, then a mean over the role's classes per unit, then a mean over live units:
  pole_{role}               class-mean push summed over positions, in units of the unit's mean |pole| over all classes
                            (convnext_bias_paths' ce split, with the relatives left out of 'rest')
  s_{in,own,part,bias,out}_{role}   |kappa|-weighted means at the unit's active positions (u >= 0.5 u*, u* > 0; p* is
                            among them), in stream units; pstar_s_*_{role}: the same at p* alone (unweighted)
  frac_sout_pos_{role}      share of (unit, class) pairs with s_out > 0 at the active positions
  frac_m1_{role}            share with s_out > 0 and s_part > s_out / 2 (the partners supply most of the stream)
  part_share_{role}         mean of s_part / (|s_in| + |s_own| + |s_part|)
  frac_sin_opposes_own_{role}   share with s_in s_own < 0 (s_own > 0 at active positions, so s_in < 0)
  frac_sout_below_sin_{role}    share with |s_out| < |s_in| (the block shrinks the stream along e)
  frac_m4_{role}            share with both (the unit's write cancels an incoming component)
  push_by_{v}_bin_{role}    for v in sout, sin, u (u_n): the class-mean push summed within each of 10 bins of v, in
                            units of the unit's mean |pole| (the 10 entries add up to pole_{role})
  push_per_pos_by_{v}_bin_{role}  push per position in each bin (pooled over the role's images) times the positions per
                            image, same units: the pole push an image would get if all its positions were in that state
  occ_by_{v}_bin_{role}     share of the role's positions in each bin
  slope_by_{v}_{role}       Spearman correlation of push_per_pos with the bin index (over bins with positions)
  role_offset_by_{v}        per unit, push_per_pos top20 minus rest at the same bin, weighted by the top20 occupancy,
                            then averaged over units
  rms_sin_all, rms_sout_all  RMS of s_in, s_out over all positions (the scale of the s_* values)
The bins of v are per unit and shared by all roles: deciles of v over all positions of all probe images, each position
weighted by |kappa| (so each bin holds a tenth of the unit's derivative mass; --edges positions: unweighted), estimated
from --n_sample random positions per image.
Readings: M1: s_out > 0 at the active pole positions, mostly from s_part, and push_per_pos falls with s_out on one
curve for both roles (role_offset_by_sout near 0); M4: s_in < 0 against s_own > 0 with |s_out| < |s_in| (frac_m4
high); M5: a role offset at fixed bins (role_offset_by_* < 0 for the reversal); M3: push_per_pos changes along the u
bins at a fixed stream state (slope_by_u) and not along s_out.

X2, partner zeroing (stage-1 blocks by default). For each live unit n, with signed cos between write directions:
S+ = {m != n: cos(w_m, w_n) > 0.9}, S- = {cos < -0.9}, rand = a random set of the size of S+ among {|cos| < 0.5}
(seeded per unit; smaller if there are too few). For each set and alpha in --alphas (0, 0.5), the units of the set are
scaled by alpha at every position of block b (n itself is not), and on the images of P_n and of 40 other classes
(random, not WordNet relatives of P_n when --wordnet is given) dCE / du is recomputed by torch.autograd.grad; n's pole
push is rebuilt with the BASELINE pa_v, pa_g (its own u, g, v do not change: the set only changes the block's output).
Per unit: the class-mean push of the top-20 and of the 40 classes, divided by the baseline mean |push| over the 60.
Summary per block:
  top20_base, rest_base       baseline, all live units
  top20_{set}_a{alpha}, rest_{set}_a{alpha}   over the units with a non-empty set (an empty set is the baseline)
  top20_base_{set}            the baseline of the same units; units_{set}: their number; n_empty_{set}: the others
  share_up_{set}_a{alpha}     share of those units whose top-20 push rises (less reversed) against their baseline
  n_short_rand                units whose random set is smaller than S+
Readings: M1: the reversal (top20 < 0) shrinks or vanishes for S+, deepens for S-, and is unchanged for rand; M5: all
unchanged.

Both start from each block's input, computed once for all probe images (as convnext_unit_roles.py does). Raw arrays
(class-level X1 accumulators, bin edges and sums, X2 per-unit rows) go to stage1_mech_raw_{step}.pt next to the JSON.

  python3 -m experiments.convnext_stage1_mech --run results/e1_tiny_adamw --steps 1000,2000,12500
"""

import argparse
import json
import os
import warnings
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

from experiments.convnext_atlas import probe_set
from experiments.convnext_bias_paths import dsilu, pole_coefs, spearman
from experiments.convnext_unit_roles import block_inputs, block_out, write_dir
from rcdl.convnext import build_convnext

DEFAULT_BLOCKS = 'stage0.block2,stage1.block0,stage1.block1,stage1.block2,stage2.block0'
X2_BLOCKS = 'stage1.block0,stage1.block1,stage1.block2'
N_BINS = 10
BIN_VARS = ('sout', 'sin', 'u')
PARTS = ('s_in', 's_own', 's_part', 's_bias', 's_out')


def parse_block(name):
  return tuple(int(t) for t in name.replace('stage', '').split('.block'))


@contextmanager
def no_param_grads(model):
  """Parameters without requires_grad while inside (only the gradient with respect to u is formed)."""
  flags = [p.requires_grad for p in model.parameters()]
  model.requires_grad_(False)
  try:
    yield
  finally:
    for p, f in zip(model.parameters(), flags):
      p.requires_grad_(f)


def block_guv(blk, x):
  """Gate, value and SwiGLU product (B, H, W, U) of a block for its input x (B, C, H, W)."""
  h = blk.norm(blk.dwconv(x).permute(0, 2, 3, 1))
  g, v = blk.pw_gate(h), blk.pw_value(h)
  return g, v, F.silu(g) * v


def tail(model, i, j, x, u):
  """Logits from block (i, j)'s input x and its units u (B, H, W, U)."""
  x = block_out(model.stages[i][j], x, u)
  for blk in model.stages[i][j + 1:]:
    x = blk(x)
  for down, stage in zip(model.downsample[i + 1:], model.stages[i + 1:]):
    x = stage(down(x))
  return model.head(model.norm(x.mean((2, 3))))


@torch.no_grad()
def write_geometry(blk):
  """Write directions of a block: W (U, d) with rows w_n = gamma * W_down[:, n], norms |w_n|, unit directions E,
  M[m, n] = w_m . e_n, and bias_e[n] = (gamma * b_down) . e_n."""
  assert blk.skip, 'X1 needs a residual block'
  U = blk.pw_down.in_features
  W = torch.stack([write_dir(blk, n) for n in range(U)]).float()
  norms = W.norm(dim=1)
  E = W / (norms[:, None] + 1e-30)
  b = blk.pw_down.bias * blk.gamma if blk.gamma is not None else blk.pw_down.bias
  return {'W': W, 'norms': norms, 'E': E, 'M': W @ E.T, 'bias_e': E @ b.float()}


@torch.no_grad()
def stream_parts(blk, x, u, geo):
  """s_in, s_own, s_part, s_bias, s_out (B, P, U) along each unit's write direction (see the module docstring)."""
  E = geo['E']
  uf = u.flatten(1, 2)
  s_in = x.permute(0, 2, 3, 1).flatten(1, 2) @ E.T
  s_own = uf * geo['norms']
  s_bias = geo['bias_e'].expand_as(uf)
  s_part = uf @ geo['M'] - s_own + s_bias
  s_out = block_out(blk, x, u).permute(0, 2, 3, 1).flatten(1, 2) @ E.T
  return {'s_in': s_in, 's_own': s_own, 's_part': s_part, 's_bias': s_bias, 's_out': s_out}


@torch.no_grad()
def class_stats(model, name, X, y, n_classes, batch):
  """R, G, V (U, C): class means of u* = max_p u and of g, v at p* (as convnext_bias_paths.probe_paths)."""
  i, j = parse_block(name)
  blk = model.stages[i][j]
  U = blk.pw_gate.out_features
  acc = torch.zeros(3, n_classes, U, dtype=torch.float64, device=X.device)
  for s in range(0, len(X), batch):
    g, v, u = (t.flatten(1, 2) for t in block_guv(blk, X[s:s + batch]))
    ustar, pstar = u.max(1)
    yb = y[s:s + batch]
    for k, t in enumerate((ustar, g.gather(1, pstar[:, None])[:, 0], v.gather(1, pstar[:, None])[:, 0])):
      acc[k].index_add_(0, yb, t.double())
  cnt = torch.bincount(y, minlength=n_classes).double()[:, None]
  return tuple((acc[k] / cnt).T.cpu().numpy() for k in range(3))


def weighted_edges(vals, w, n_bins=N_BINS):
  """vals, w: (N, U). Per unit the n_bins - 1 inner edges (U, n_bins - 1) of the w-weighted quantiles of vals
  (unweighted for a unit whose weights are all zero)."""
  v, order = vals.T.sort(1)
  cw = torch.gather(w.T.double(), 1, order).cumsum(1)
  N = v.shape[1]
  flat = cw[:, -1] <= 0
  if flat.any():
    cw[flat] = torch.arange(1, N + 1, dtype=torch.float64, device=cw.device)
  q = torch.arange(1, n_bins, dtype=torch.float64, device=cw.device) / n_bins
  k = torch.searchsorted(cw.contiguous(), (q[None] * cw[:, -1:]).contiguous()).clamp(max=N - 1)
  return torch.gather(v, 1, k)


def _kappa(g, v, pa_v, pa_g):
  return F.silu(g) * pa_v + dsilu(g) * v * pa_g


def _nanmean(a, axis=None):
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    return np.nanmean(a, axis)


def roles_of(top, n_classes, rel=None):
  """{role: (C, U) bool}: top20, relatives (only with rel) and rest."""
  U = top.shape[0]
  inT = np.zeros((n_classes, U), bool)
  np.put_along_axis(inT, top.T, True, 0)
  relm = (rel[top].any(1).T & ~inT) if rel is not None else np.zeros_like(inT)
  out = {'top20': inT, 'rest': ~inT & ~relm}
  if rel is not None:
    out['relatives'] = relm
  return out


def x1_block(model, name, X, y, rel=None, stats=None, batch=100, n_sample=16, edges='kappa', seed=0, top_n=20):
  """X1 for one block. X: inputs of the block for all probe images (on the device), y: labels (same device).
  Returns (summary, raw)."""
  i, j = parse_block(name)
  blk = model.stages[i][j]
  dev = X.device
  C = int(y.max()) + 1
  U = blk.pw_gate.out_features
  R, G, V = stats if stats is not None else class_stats(model, name, X, y, C, batch)
  top, pa_v, pa_g, live = pole_coefs(R, G, V, top_n)
  pv, pg = (torch.from_numpy(a).float().to(dev) for a in (pa_v, pa_g))
  geo = write_geometry(blk)
  gen = torch.Generator().manual_seed(seed)
  # pass 1: bin edges from a sample of positions, and the RMS of the stream along e
  samp = {k: [] for k in BIN_VARS + ('w',)}
  sq = torch.zeros(2, U, dtype=torch.float64, device=dev)
  n_pos = 0
  with torch.no_grad():
    for s in range(0, len(X), batch):
      xb = X[s:s + batch]
      g, v, u = block_guv(blk, xb)
      sp = stream_parts(blk, xb, u, geo)
      B, P = sp['s_in'].shape[:2]
      kap = _kappa(g, v, pv, pg).flatten(1, 2)
      idx = torch.randint(P, (B, min(n_sample, P)), generator=gen).to(dev)[..., None].expand(-1, -1, U)
      for k, t in (('sout', sp['s_out']), ('sin', sp['s_in']), ('u', u.flatten(1, 2)), ('w', kap.abs())):
        samp[k].append(t.gather(1, idx).flatten(0, 1))
      sq[0] += (sp['s_in'].double() ** 2).sum((0, 1))
      sq[1] += (sp['s_out'].double() ** 2).sum((0, 1))
      n_pos += B * P
  w = torch.cat(samp['w']) if edges == 'kappa' else torch.ones_like(torch.cat(samp['w']))
  E = {k: weighted_edges(torch.cat(samp[k]), w).float() for k in BIN_VARS}                # (U, N_BINS - 1)
  del samp
  # pass 2: dCE/du and the accumulators
  f64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
  acc = {'push': f64(C, U), 'push_act': f64(C, U), 'n_act': f64(C, U), 'den': f64(C, U),
         **{f'num_{k}': f64(C, U) for k in PARTS}, **{f'pstar_{k}': f64(C, U) for k in PARTS},
         **{f'bin_{t}_{k}': f64(C * U * N_BINS) for k in BIN_VARS for t in ('push', 'cnt')}}
  ar = torch.arange(U, device=dev)
  with no_param_grads(model):
    for s in range(0, len(X), batch):
      xb, yb = X[s:s + batch], y[s:s + batch]
      with torch.no_grad():
        g, v, u = block_guv(blk, xb)
        sp = stream_parts(blk, xb, u, geo)
      ul = u.detach().requires_grad_(True)
      ce = F.cross_entropy(tail(model, i, j, xb, ul).float(), yb, reduction='sum')
      delta, = torch.autograd.grad(ce, [ul])
      with torch.no_grad():
        uf = u.flatten(1, 2)
        kap = _kappa(g, v, pv, pg).flatten(1, 2)
        push = -kap * delta.flatten(1, 2)                                                  # (B, P, U)
        ustar, pstar = uf.max(1)
        act = (uf >= 0.5 * ustar[:, None]) & (ustar[:, None] > 0)
        wk = kap.abs() * act
        add = lambda key, t: acc[key].index_add_(0, yb, t.double())
        add('push', push.sum(1))
        add('push_act', (push * act).sum(1))
        add('n_act', act.sum(1))
        add('den', wk.sum(1))
        for k in PARTS:
          add(f'num_{k}', (wk * sp[k]).sum(1))
          add(f'pstar_{k}', sp[k].gather(1, pstar[:, None])[:, 0])
        base = (yb[:, None, None] * U + ar) * N_BINS                                     # (B, 1, U)
        for k, t in (('sout', sp['s_out']), ('sin', sp['s_in']), ('u', uf)):
          bins = torch.searchsorted(E[k], t.permute(2, 0, 1).reshape(U, -1).contiguous())  # (U, B * P)
          flat = (base + bins.reshape(U, *t.shape[:2]).permute(1, 2, 0)).flatten()
          acc[f'bin_push_{k}'].index_add_(0, flat, push.flatten().double())
          acc[f'bin_cnt_{k}'].index_add_(0, flat, torch.ones_like(flat, dtype=torch.float64))
  P = n_pos // len(X)
  n_img = torch.bincount(y, minlength=C).double().cpu().numpy()                          # (C,)
  A = {k: (t.view(C, U, N_BINS) if k.startswith('bin_') else t).cpu().numpy() for k, t in acc.items()}
  raw = {'top': top, 'pa_v': pa_v, 'pa_g': pa_g, 'live': live, 'R': R, 'G': G, 'V': V, 'n_img': n_img, 'P': P,
         'edges': {k: e.cpu().numpy() for k, e in E.items()}, 'edges_weight': edges, **A}
  pole = A['push'] / n_img[:, None]
  raw['pole'] = pole
  scale = np.abs(pole).mean(0) + 1e-30
  with np.errstate(invalid='ignore', divide='ignore'):
    S = {k: A[f'num_{k}'] / A['den'] for k in PARTS}                                       # (C, U), NaN: never active
  Sp = {k: A[f'pstar_{k}'] / n_img[:, None] for k in PARTS}
  roles = roles_of(top, C, rel)
  rm = lambda M, mask: float(_nanmean(_nanmean(np.where(mask, M, np.nan), 0)[live]))
  out = {'units': int(live.sum()), 'edges_weight': edges, 'positions': P,
         'rms_sin_all': float(np.sqrt(sq[0].cpu().numpy() / n_pos)[live].mean()),
         'rms_sout_all': float(np.sqrt(sq[1].cpu().numpy() / n_pos)[live].mean())}
  si, so, sw, sx = S['s_in'], S['s_own'], S['s_part'], S['s_out']
  fin = np.isfinite(sx)
  frac = lambda cond: np.where(fin, cond.astype(float), np.nan)
  ppp, occ = {}, {}
  for role, mask in roles.items():
    out[f'pole_{role}'] = rm(pole / scale[None], mask)
    out[f'push_act_{role}'] = rm(A['push_act'] / n_img[:, None] / scale[None], mask)
    out[f'n_act_{role}'] = rm(A['n_act'] / n_img[:, None], mask)
    for k in PARTS:
      out[f'{k}_{role}'] = rm(S[k], mask)
      out[f'pstar_{k}_{role}'] = rm(Sp[k], mask)
    with np.errstate(invalid='ignore'):
      out[f'frac_sout_pos_{role}'] = rm(frac(sx > 0), mask)
      out[f'frac_m1_{role}'] = rm(frac((sx > 0) & (sw > 0.5 * sx)), mask)
      out[f'part_share_{role}'] = rm(sw / (np.abs(si) + np.abs(so) + np.abs(sw) + 1e-30), mask)
      out[f'frac_sin_opposes_own_{role}'] = rm(frac(si * so < 0), mask)
      out[f'frac_sout_below_sin_{role}'] = rm(frac(np.abs(sx) < np.abs(si)), mask)
      out[f'frac_m4_{role}'] = rm(frac((si * so < 0) & (np.abs(sx) < np.abs(si))), mask)
    m3 = mask[..., None]
    for k in BIN_VARS:
      bp_, bc = A[f'bin_push_{k}'], A[f'bin_cnt_{k}']                                        # (C, U, N_BINS)
      per_cls = bp_ / n_img[:, None, None] / scale[None, :, None]
      out[f'push_by_{k}_bin_{role}'] = [float(x) for x in _nanmean(_nanmean(np.where(m3, per_cls, np.nan), 0)[live], 0)]
      ps, cs = np.where(m3, bp_, 0).sum(0), np.where(m3, bc, 0).sum(0)                      # (U, N_BINS)
      with np.errstate(invalid='ignore', divide='ignore'):
        ppp[role, k] = np.where(cs > 0, ps / cs, np.nan) * P / scale[:, None]
        occ[role, k] = cs / (np.where(mask, n_img[:, None], 0).sum(0)[:, None] * P)
      curve = _nanmean(ppp[role, k][live], 0)
      out[f'push_per_pos_by_{k}_bin_{role}'] = [float(x) for x in curve]
      out[f'occ_by_{k}_bin_{role}'] = [float(x) for x in _nanmean(occ[role, k][live], 0)]
      ok = np.isfinite(curve)
      out[f'slope_by_{k}_{role}'] = spearman(np.arange(N_BINS)[ok], curve[ok]) if ok.sum() > 2 else float('nan')
  for k in BIN_VARS:                            # top20 minus rest at the same bin, weighted by the top20 occupancy
    d = ppp['top20', k] - ppp['rest', k]                                                     # (U, N_BINS)
    wgt = np.where(np.isfinite(d), occ['top20', k], 0)
    with np.errstate(invalid='ignore', divide='ignore'):
      off = (wgt * np.nan_to_num(d)).sum(1) / wgt.sum(1)
    out[f'role_offset_by_{k}'] = float(_nanmean(off[live]))
  return out, raw


def x2_push(model, name, X, y, by_class, cls, n, configs, pa_v, pa_g, batch=100):
  """Class-mean pole push of unit n on the classes cls under each config {key: None (baseline) or (units, alpha)}.
  Returns {key: (len(cls),) float64}."""
  i, j = parse_block(name)
  blk = model.stages[i][j]
  dev = X.device
  U = blk.pw_gate.out_features
  idx = np.concatenate([by_class[c] for c in cls])
  pos = torch.from_numpy(np.repeat(np.arange(len(cls)), [len(by_class[c]) for c in cls])).to(dev)
  scales = {}
  for key, cfg in configs.items():
    sc = torch.ones(U, device=dev)
    if cfg is not None:
      sc[torch.as_tensor(np.asarray(cfg[0], dtype=np.int64), device=dev)] = float(cfg[1])
    scales[key] = sc
  acc = {k: torch.zeros(len(cls), dtype=torch.float64, device=dev) for k in configs}
  with no_param_grads(model):
    for s in range(0, len(idx), batch):
      ib = torch.from_numpy(idx[s:s + batch]).to(dev)
      xb, yb, pb = X[ib], y[ib], pos[s:s + batch]
      with torch.no_grad():
        g, v, u = block_guv(blk, xb)
        kap = _kappa(g[..., n], v[..., n], float(pa_v), float(pa_g))                      # (B, H, W)
      for key, sc in scales.items():
        ul = u.detach().requires_grad_(True)
        ce = F.cross_entropy(tail(model, i, j, xb, ul * sc).float(), yb, reduction='sum')
        d, = torch.autograd.grad(ce, [ul])
        acc[key].index_add_(0, pb, (-(kap * d[..., n]).flatten(1).sum(1)).double())
  cnt = torch.bincount(pos, minlength=len(cls)).double()
  return {k: (a / cnt).cpu().numpy() for k, a in acc.items()}


def x2_block(model, name, X, y, rel=None, stats=None, alphas=(0.0, 0.5), n_top=20, n_other=40, batch=100, seed=0,
             cos_hi=0.9, cos_lo=0.5, units=None):
  """X2 for one block (see the module docstring). units: optional subset of unit indices. Returns (summary, rows)."""
  i, j = parse_block(name)
  blk = model.stages[i][j]
  y_np = y.cpu().numpy()
  C = int(y_np.max()) + 1
  R, G, V = stats if stats is not None else class_stats(model, name, X, y, C, batch)
  top, pa_v, pa_g, live = pole_coefs(R, G, V, n_top)
  E = write_geometry(blk)['E']
  cos = (E @ E.T).cpu().numpy()
  by_class = [np.flatnonzero(y_np == c) for c in range(C)]
  rows = []
  for n in (np.flatnonzero(live) if units is None else units):
    rng = np.random.default_rng([seed, int(n)])           # per unit: its classes do not depend on the other units
    others = np.setdiff1d(np.arange(C), top[n])
    if rel is not None:
      far = others[~rel[top[n]].any(0)[others]]
      others = far if len(far) >= n_other else others
    oth = rng.choice(others, min(n_other, len(others)), replace=False)
    cls = np.concatenate([top[n], oth])
    not_n = np.arange(len(cos)) != n
    sets = {'splus': np.flatnonzero(not_n & (cos[n] > cos_hi)), 'sminus': np.flatnonzero(not_n & (cos[n] < -cos_hi))}
    cand = np.flatnonzero(not_n & (np.abs(cos[n]) < cos_lo))
    rng_s = np.random.default_rng([seed, int(n), 1])      # nor on the sizes of its sets
    sets['rand'] = np.sort(rng_s.choice(cand, min(len(sets['splus']), len(cand)), replace=False))
    configs = {'base': None}
    configs.update({f'{k}_a{al:g}': (S, al) for k, S in sets.items() if len(S) for al in alphas})
    push = x2_push(model, name, X, y, by_class, cls, n, configs, pa_v[n], pa_g[n], batch)
    norm = np.abs(push['base']).mean() + 1e-30
    rec = {'unit': int(n), **{f'n_{k}': int(len(S)) for k, S in sets.items()},
           'top20_base': float(push['base'][:n_top].mean() / norm),
           'rest_base': float(push['base'][n_top:].mean() / norm)}
    for k, S in sets.items():
      for al in alphas:
        p = push.get(f'{k}_a{al:g}', push['base'])                                          # empty set: the baseline
        rec[f'top20_{k}_a{al:g}'] = float(p[:n_top].mean() / norm)
        rec[f'rest_{k}_a{al:g}'] = float(p[n_top:].mean() / norm)
    rows.append(rec)
  out = {'units': len(rows), 'top20_base': float(np.mean([r['top20_base'] for r in rows])),
         'rest_base': float(np.mean([r['rest_base'] for r in rows])),
         'n_short_rand': int(sum(r['n_rand'] < r['n_splus'] for r in rows))}
  for k in ('splus', 'sminus', 'rand'):
    sel = [r for r in rows if r[f'n_{k}'] > 0]
    out[f'units_{k}'] = len(sel)
    out[f'n_empty_{k}'] = len(rows) - len(sel)
    out[f'top20_base_{k}'] = float(np.mean([r['top20_base'] for r in sel])) if sel else float('nan')
    out[f'rest_base_{k}'] = float(np.mean([r['rest_base'] for r in sel])) if sel else float('nan')
    for al in alphas:
      key = f'{k}_a{al:g}'
      for role in ('top20', 'rest'):
        out[f'{role}_{key}'] = float(np.mean([r[f'{role}_{key}'] for r in sel])) if sel else float('nan')
      up = [r[f'top20_{key}'] > r['top20_base'] for r in sel]
      out[f'share_up_{key}'] = float(np.mean(up)) if sel else float('nan')
  return out, rows


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', required=True, help='run directory with config.json and model_{step}.pt')
  p.add_argument('--steps', required=True)
  p.add_argument('--blocks', default=DEFAULT_BLOCKS, help="X1 blocks ('' to skip X1)")
  p.add_argument('--x2_blocks', default=X2_BLOCKS, help="X2 blocks ('' to skip X2)")
  p.add_argument('--alphas', default='0,0.5', help='X2 output scales of the partner sets')
  p.add_argument('--val', default='/marimo/work/data/imagenet/data')
  p.add_argument('--probe_cache', default='/dev/shm/probe_8.pt')
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--edges', default='kappa', choices=['kappa', 'positions'],
                 help='X1 bin edges: |kappa|-weighted deciles of all positions, or unweighted')
  p.add_argument('--n_sample', type=int, default=16, help='X1: positions per image for the bin edges')
  p.add_argument('--batch', type=int, default=50)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default=None, help='JSON (default: stage1_mech.json in --run); raw arrays next to it')
  a = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  model = build_convnext(cfg['size']).to(dev).eval()
  model.requires_grad_(False)
  xs, y = probe_set(a.val, 8, a.probe_cache)
  y = y.to(dev)
  C = int(y.max()) + 1
  rel = None
  if a.wordnet and os.path.exists(a.wordnet):
    rel = np.load(a.wordnet)['dist'][:C, :C] <= 4
    np.fill_diagonal(rel, False)
  else:
    print(f"no WordNet file at {a.wordnet}: 'rest' includes the relatives", flush=True)
  alphas = tuple(float(t) for t in a.alphas.split(','))
  b1 = [b for b in a.blocks.split(',') if b]
  b2 = [b for b in a.x2_blocks.split(',') if b]
  out = a.out or os.path.join(a.run, 'stage1_mech.json')
  odir = os.path.dirname(os.path.abspath(out))
  os.makedirs(odir, exist_ok=True)
  res = {}
  for s in [int(t) for t in a.steps.split(',')]:
    model.load_state_dict(torch.load(os.path.join(a.run, f'model_{s:05d}.pt'), map_location=dev))
    res[s], raw = {'x1': {}, 'x2': {}}, {'x1': {}, 'x2': {}}
    for b in sorted(set(b1) | set(b2), key=parse_block):
      i, j = parse_block(b)
      X = block_inputs(model, xs, i, j, a.batch, dev)
      stats = class_stats(model, b, X, y, C, a.batch)
      if b in b1:
        res[s]['x1'][b], raw['x1'][b] = x1_block(model, b, X, y, rel, stats, a.batch, a.n_sample, a.edges, a.seed)
        r = res[s]['x1'][b]
        print(s, b, 'X1', {k: round(r[k], 3) for k in ('pole_top20', 's_in_top20', 's_own_top20', 's_part_top20',
                                                         's_out_top20', 'frac_m1_top20', 'frac_m4_top20',
                                                         'role_offset_by_sout', 'role_offset_by_u')}, flush=True)
      if b in b2:
        res[s]['x2'][b], raw['x2'][b] = x2_block(model, b, X, y, rel, stats, alphas, batch=a.batch, seed=a.seed)
        r = res[s]['x2'][b]
        print(s, b, 'X2', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
      del X
    torch.save(raw, os.path.join(odir, f'stage1_mech_raw_{s:05d}.pt'))
    json.dump(res, open(out, 'w'), indent=1)
  print('STAGE1_MECH_DONE', flush=True)


if __name__ == '__main__':
  main()
