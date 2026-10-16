"""E6 umbrella = bee, analysis (protocol v2 = the critique's revised protocol; docs/EXACT_HISTORY.md Sec. 4.2).

Input: run directories of experiments/convnext_umbrella.py (model_{s:04d}.pt, sums_{s:04d}.pt = group_sums over the
named sets, shadow_{s:04d}.pt = 8 shadow compact states, attr_{s:04d}.pt, attr_init.pt, tau.json, config.json,
log.jsonl), E1's final files (model_25020.pt, attr_25020.pt, attr_init.pt, atlas_25020.pt) and the validation set.

Notation (per run):
  theta_s   weights after fine-tuning step s (model_{s}); theta_0 = theta_w, the branch state; theta_E1 = E1 step 25,020.
  tau       the flip step (tau.json; censored: the cap was hit); end = tau + phase2 (taught) or tau (solo).
  I1 = [0, tau], I2 = [tau, end], I_1 = [0, 1].
  x_P, t, s the taught image, its target class and its source class (the selection arm's source; for a control, the
            top-1 class of x_P at theta_w, or the second one if that is t).
  m_j       z_t(x_j) - z_s(x_j), the logit margin of image j; zt, zs the two logits.
  maps      tracked nn.Linear maps: pw_gate, pw_value, pw_down of the 18 blocks ('blocks') and the head.
  buckets   an exact split of theta_s2 - theta_s1 (saved weights) over an interval (s1, s2] (load_buckets): the writes
            S_s2^w - (a_s2 / a_s1) S_s1^w of the named sets w = P (the photo), BEE (target class), UMB (source
            class), A, B (unrelated oversampled classes), RELb, RELu (relatives), OTH (all other classes), hist (the
            momentum from before the branch), with S^w the set's summed parts and a the weight-decay factor; dec_P
            (decay of the photo's earlier writes), dec_rest (of every other set's), dec_0 ((a_s2 - a_s1) W_0), R (the
            rest of the saved weights' change: the fp32 storage remainder of the parts, so the buckets add up to the
            saved change exactly) and unt (untracked parameters; standard optimizer only). Extra columns, not part of
            the partition: the pseudo-groups 'sh{r}.g{g}' (writes of slot group g = 1..8 of shadow split r = 0..7:
            31 or 32 random replay slots per step, the same slots in the taught and untaught runs), and 'k{c}' (class
            c's writes, from the full states).
  IG        IG_j^w(f; path) = sum over segments [theta_a, theta_b] of the piecewise-linear path through saved states,
            sum over Gauss-Legendre nodes (s_q, w_q) on [0, 1], of w_q <grad f_j(theta_a + s_q (theta_b - theta_a)),
            dtheta_seg^w>. The buckets of each segment sum to theta_b - theta_a, so sum_w IG_j^w = f_j(end) -
            f_j(start) up to quadrature (completeness eps_j = |sum_w IG_j^w - df_j| / max(|df_j|, 0.1); nodes are
            doubled, up to 64, while a segment's error exceeds its share tol * 0.1 / n_segments). Tracked maps: the
            per-example gradient sum_p delta_p h_p^T from (input h, output gradient delta) hooks; unt: a forward-mode
            JVP. Per-row restriction: rows of gate, value and head (bias with its row); columns of pw_down (its bias
            stays per output channel, 'down_bias'). F^w(I) = IG_P^w(m; I) restricted to a group of maps.
  paths     fine: every saved step (phase 1: every step); common: steps saved with shadow states in both the taught
            and the untaught run (0, multiples of 25, tau, tau + 500), so that real and pseudo-group buckets share one
            path; V sets: each interval as one segment.

M1 feed and shelter (C1), on the common path, F restricted to the block maps unless stated:
  DiD^w(I) = F^w(I; taught) - F^w(I; untaught); the 64 pseudo-groups give DiD^{r,g}; z^w = (DiD^w - mean) / sd over
  them (sd with ddof 1). Feed phi_BEE(I) = DiD^BEE(I) / dm_P(I1), dm_P(I1) = m_P(tau) - m_P(0) in the taught run.
  Shelter E_OTH = -F^OTH(I2; taught) / dm_P(I1); E_dec = -F^{dec_P}(I2; taught, all maps) / dm_P(I1). Umbrella
  coupling: DiD^UMB and z over I1 and I2 per map group (blocks, stage0-3, gate+value, down, head, total). Decline
  composition: F^w(I2; taught, all maps) / (m_P(end) - m_P(tau)). Retained fraction rho = (m_P(end) - m_P(0)) /
  (m_P(tau) - m_P(0)). Triple difference (descriptive): DiD^w on x_P minus the mean DiD^w over M_P (the 10 images of
  V_umb whose margin at theta_w is closest to the photo's). F_fine: F^w(I1) on the taught run's every-step path.
M2 footprint (C2), V sets in their own one-segment passes (vdtype): r_j(I) = IG_j^P(m_j; I) / IG_P^P(m_P; I);
  G^P(V; I) = mean_{j in V} r_j(I) for V in {V_near, V_umb, V_bee, V_ref} and I in {I_1, I1, [0, end]}; latching index
  L = 1 - G^P(V_near; I1) / G^P(V_near; I_1); net footprint N(V) = mean_j dm_j([0, end]) / dm_P([0, end]); carving
  K = 1 - N(V_near) / G^P(V_near; [0, end]) with n_w = mean_{j in V_near} IG_j^w / dm_P per bucket; per-step
  footprint G_t = G^P(V_near; [t-1, t]) for t in I1; p_umb_ref: one-sided permutation p (images relabelled between
  V_umb and V_ref) of G^P(V_umb; [0, end]) - G^P(V_ref; [0, end]); beta: OLS slope of r_j(I1) on the cosine of
  theta_w's pooled features of x_j and x_P over V_ref and V_umb, with its permutation p.
M3 carriers (C3): u_n(x; theta) the SwiGLU product of unit n (block b, index i); its strongest position p* = argmax_p
  u_n at theta_tau. c_n = m_P(theta_tau) - m_P(theta_tau with u_n on x_P at every position set to its theta_w value);
  C50 = the fewest units holding >= 50% of sum_n max(c_n, 0). Pole sets (E1 atlas R[n, c], class mean u* at theta_E1):
  B_pole = units with t among the 20 largest or 20 smallest R[n, .]; U_pole the same for s. Exact split at p*, with
  g, v the gate and value pre-activations and h the LayerNorm output: g_tau - g_w = sum_w (dW^{gate,w}[n] h_tau +
  db^w[n]) (own row, by bucket over I1) + W_w^gate[n] (h_tau - h_w) (upstream), the same for v, and
  u_tau - u_w = A (g_tau - g_w) + Bv (v_tau - v_w) with A = (SiLU(g_tau) - SiLU(g_w)) / (g_tau - g_w) (v_tau + v_w) / 2,
  Bv = (SiLU(g_tau) + SiLU(g_w)) / 2 (exact); own_u, up_u the two parts in u; own-row share = |own_u| / (|own_u| +
  |up_u|); P share = own_u^P / own_u. Two-level dual form (gate, value): W_w[n] = a W_init[n] + sum_k C_pre^k[n] +
  R_pre[n] + dW_warm[n] (E1's parts at 25,020; dW_warm = W_w - W_E1), so up_u = sum_k T_k + init + R + warm with
  T_k = A C_pre^{gate,k}[n] dh + Bv C_pre^{value,k}[n] dh (dh = h_tau - h_w); q_n^c = #{k : o T_k < o T_c} / 999 with
  o = sign(u_tau - u_w). Matched comparison: each carrier in B_pole (U_pole, fading: u_tau < u_w) against the mean q of
  the 3 non-carriers of its block and pole set nearest on |R[n, c]|; one-sided signed-rank test by random sign flips.
  Selectivity sel_n = sigma_n u*_n(x_P) / Q_0.999{sigma_n u*_n(x_j) : j in V_ref} at the same theta, sigma_n =
  sign(u*_n(x_P)), u* = u at the strongest position. Drift: carriers at theta_end (patched to theta_w), carrier mass
  in pole units and in photo-selective (sel >= 1.5) non-pole units, at tau and end. Provenance (optional, file of
  'provenance'): pi_n^c = #{k != c : s_c V_pre[n,c,c] > s_c V_pre[n,c,k]} / 999 (s_c = sign(Vm[n, c]); E1 atlas
  writers of the value and gate), carriers against matched non-carriers. Context: r^bee_n = u*_n(x_P) / R[n, t],
  r^umb_n = u*_n(x_P) / R[n, s] at theta_w and theta_tau; route share = carrier mass in B_pole u U_pole.
M4 localization (C4): S_b = IG_P^{P, maps of block b}(m; I1, fine path) / sum over all maps of IG_P^P(m; I1);
  S_late = sum over stage 3, S_head. Swaps into theta_w for a set S of maps: theta^P(S) = theta_w + C_tau^{P} on S,
  theta^tot(S) = theta_w with S (and, standard optimizer, the untracked parameters of S's blocks) from theta_tau,
  theta^-P(S) = theta_tau - C_tau^{P} on S; readouts on x_P: argmax, p_t, p_s, m_P.

Decision rules (decide; B = primary frozen-v runs, F = H-leaning controls, Bp = standard optimizer, I = leave-group-
out, E = relabel triples, J = E1r replicate, C = solo):
  P-C1a O: DiD^BEE(I2) > 0, z >= 3 and DiD^BEE(I2) > DiD^A(I2), DiD^B(I2) in >= 4 of 5 photos in >= 2 of 3 seeds, and
        DiD^BEE(I2) > 0 in >= 4 of 5 Bp photos; against: |z| < 2 in >= 3 of 5 photos in >= 2 seeds; arm I sign
        agreement of dLGO_BEE with DiD^BEE(I2) reported.
  P-C1b O: E_OTH(photo) < E_OTH of every seed-matched control (controls with z^BEE(I2) >= 3 reported, not counted) in
        >= 4 of 5 photos in >= 2 of 3 seeds; against (our reading of H): E_OTH(photo) >= the controls' median in >= 3
        of 5 photos in >= 2 seeds.
  P-C1c reported: |z^UMB| >= 3 in I1 or I2 (O-like), |z^UMB| < 2 in both (H-like), with signs.
  P-C2  calibrated if >= 2 of the 4 seed-0 controls have L >= 0.5; then O: L(photo) < min L(controls) in >= 4 of 5
        photos at seed 0 and in >= 3 of 5 at seeds 1 and 2, and p_umb_ref < 0.01 in >= 4 of 5 photos at seed 0; H:
        L(photo) within the controls' range and p_umb_ref >= 0.01 in >= 3 of 5; L(replay) > L(solo) reported as
        replay-induced latching.
  P-C3  per run, O: median own-row share over C50 <= 0.5, median q^t over C50 in B_pole >= 0.9 with signed-rank p <
        0.01 against matched non-carriers, the same for q^s over fading C50 in U_pole (when there are any), median sel
        <= 1.0, pole carrier mass at end >= 2/3 of that at tau; H': own-row share >= 0.8, or sel >= 1.5, or the
        carrier-mass share of selective non-pole units rises from tau to end and reaches 0.5; H: fewer than half of
        C50 in a pole set and median q^t within 0.1 of 0.5.
  P-C4  O: S_late >= 0.8, argmax theta^P(stage3) = t, argmax theta^P(head) != t; S_head >= 0.5 is a deviation.
  Verdict: C1 supported if P-C1a and P-C1b are O, contradicted if both are 'against'; C2, C3, C4 separately.

Subcommands: vsets (evaluation sets of one selection arm), provenance (E1 writers for an arm's target and source),
run (one taught or solo run: M1-M4 and checks -> analysis.json, analysis.pt), decide (all analyses under a root).
  python3 -m experiments.convnext_umbrella_analysis run --run <dir> --untaught <dir> --tag B --e1 <E1 dir> \
    --val <val dir> --vsets <file> [--provenance <file>]
"""

import argparse
import glob
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.attribution import interval_buckets
from rcdl.convnext import build_convnext
from rcdl.imagenet import MEAN, STD

SWAP_SETS = ('stage3', 'stage3+head', 'head', 'stage2', 'stages0-1', 'stage3.block0', 'stage3.block1',
             'stage3.block2', 'stage3.gate+value', 'stage3.down')


# ------------------------------------------------------------------------------------------------- run files

def _ld(run, name):
  return torch.load(os.path.join(run, name), map_location='cpu', weights_only=False)


def steps_of(run: str, prefix: str) -> list:
  """Saved steps of files <prefix>_{s}.pt (numeric suffix only)."""
  out = []
  for f in glob.glob(os.path.join(run, f'{prefix}_*.pt')):
    tail = os.path.basename(f)[len(prefix) + 1:-3]
    if tail.isdigit():
      out.append(int(tail))
  return sorted(out)


def tracked_maps(model) -> dict:
  """{'stage{i}.block{j}.pw_gate' (pw_value, pw_down): nn.Linear, 'head': model.head}, as the fine-tuning script."""
  maps = {f'{n}.{k}': getattr(b, k) for n, b in model.blocks() for k in ('pw_gate', 'pw_value', 'pw_down')}
  maps['head'] = model.head
  return maps


def map_keys(model, maps) -> dict:
  """{map: module prefix in the model's state_dict}."""
  names = {id(m): n for n, m in model.named_modules()}
  return {k: names[id(lin)] for k, lin in maps.items()}


def run_info(run: str) -> dict:
  cfg = json.load(open(os.path.join(run, 'config.json')))
  tj = os.path.join(run, 'tau.json')
  tau = json.load(open(tj)) if os.path.exists(tj) else {'tau': None, 'censored': None}
  log = []
  if os.path.exists(os.path.join(run, 'log.jsonl')):
    log = [json.loads(l) for l in open(os.path.join(run, 'log.jsonl')) if l.strip()]
  return {'cfg': cfg, 'tau': tau['tau'], 'censored': tau['censored'], 'log': log}


def path_vertices(run: str, s1: int, s2: int, shadows: bool) -> list:
  ok = set(steps_of(run, 'sums')) & set(steps_of(run, 'model'))
  if shadows:
    ok &= set(steps_of(run, 'shadow'))
  return sorted(s for s in ok if s1 <= s <= s2)


def common_path(runs, s1: int, s2: int, shadows: bool = True) -> list:
  """Steps in [s1, s2] saved (model, sums and, with shadows, shadow states) in every run; must contain s1 and s2."""
  v = set.intersection(*[set(path_vertices(r, s1, s2, shadows)) for r in runs])
  if s1 not in v or s2 not in v:
    raise ValueError(f'steps {s1} and {s2} must be saved (with shadows={shadows}) in {list(runs)}: run the untaught '
                     f'run with --full_saves listing them')
  return sorted(v)


def segments_of(vertices) -> list:
  return list(zip(vertices[:-1], vertices[1:]))


# ------------------------------------------------------------------------------------------------- buckets

def _cv(t, dtype, device):
  return t.to(device=device, dtype=dtype)


@torch.no_grad()
def load_buckets(run: str, s1: int, s2: int, keys: dict, shadows: bool = False, per_class: bool = False,
                 dtype=torch.float64, device='cpu', th=None) -> dict:
  """Exact split of theta_s2 - theta_s1 (the saved weights) over the buckets of the module docstring.
  keys: {map: module prefix}; th: optional (state_dict s1, state_dict s2) already loaded.
  Returns {'names': partition buckets, 'extra': pseudo-group (and per-class) columns, 'W': {map: (n, out, in)},
  'b': {map: (n, out)} over names + extra, 'unt': {param: change} of the untracked parameters that moved,
  'check': {'rem_vs_sums': max |R - (R_s2 - R_s1)| / max |W|, 'shadow_sum': max over shadows of the relative error of
  each shadow's own split, 'shadow_vs_replay': max relative difference of the 8 slot groups' writes from the class
  writes (fp16 storage of the shadows)}}."""
  init = _ld(run, 'attr_init.pt')
  S1, S2 = _ld(run, f'sums_{s1:04d}.pt'), _ld(run, f'sums_{s2:04d}.pt')
  th1, th2 = th if th is not None else (_ld(run, f'model_{s1:04d}.pt'), _ld(run, f'model_{s2:04d}.pt'))
  ib = interval_buckets(S1, S2, init)              # asserts that its buckets add up to the change of the parts
  cv = lambda t: _cv(t, dtype, device)
  sets = list(next(iter(S1.values()))['sets'])
  names = sets + ['dec_P', 'dec_rest', 'dec_0', 'R']
  replay = [w for w in sets if w not in ('P', 'hist')]
  out = {'names': names, 'extra': [], 'W': {}, 'b': {}, 'unt': {}, 'check': {}}
  rem_err = 0.0
  for k, pre in keys.items():
    r, W1, W2 = ib[k], cv(th1[pre + '.weight']), cv(th2[pre + '.weight'])
    rows = [cv(r['write'][w]) for w in sets]
    zw = torch.zeros_like(W1)
    rows.append(cv(r['dec']['P']) if 'P' in sets else zw)
    rows.append(sum((cv(r['dec'][w]) for w in sets if w != 'P'), zw))
    rows.append(cv(r['dec_0']))
    rows.append((W2 - W1) - sum(rows, zw))
    rem_err = max(rem_err, float((rows[-1] - cv(r['rem'])).abs().max() / W2.abs().max().clamp(min=1e-30)))
    out['W'][k] = torch.stack(rows)
    b1, b2 = cv(th1[pre + '.bias']), cv(th2[pre + '.bias'])
    zb = torch.zeros_like(b1)
    wb = r.get('write_b', {})
    rows_b = [cv(wb[w]) if w in wb else zb for w in sets]
    db = r.get('dec_b', {})
    rows_b.append(cv(db['P']) if 'P' in db else zb)
    rows_b.append(sum((cv(db[w]) for w in sets if w != 'P' and w in db), zb))
    rows_b.append(cv(r['dec_0b']) if 'dec_0b' in r else zb)
    rows_b.append((b2 - b1) - sum(rows_b, zb))
    out['b'][k] = torch.stack(rows_b)
  out['check']['rem_vs_sums'] = rem_err
  extra_W = {k: [] for k in keys}
  extra_b = {k: [] for k in keys}
  if shadows:
    sh1, sh2 = _ld(run, f'shadow_{s1:04d}.pt'), _ld(run, f'shadow_{s2:04d}.pt')
    err_sum = err_rep = 0.0
    for r_, (c1, c2) in enumerate(zip(sh1, sh2)):
      for k, pre in keys.items():
        e1, e2 = c1[k], c2[k]
        C1, C2 = cv(e1['C']) / e1['C_scale'], cv(e2['C']) / e2['C_scale']
        rho = e2['a'] / e1['a']
        wr = C2 - rho * C1                               # (10, out, in): photo, 8 slot groups, hist
        Cb1, Cb2 = cv(e1['Cb']) / e1['Cb_scale'], cv(e2['Cb']) / e2['Cb_scale']
        wrb = Cb2 - (e2['ab'] / e1['ab']) * Cb1
        for g in range(1, wr.shape[0] - 1):
          extra_W[k].append(wr[g])
          extra_b[k].append(wrb[g])
        W1, W2 = cv(th1[pre + '.weight']), cv(th2[pre + '.weight'])
        own = wr.sum(0) + (rho - 1) * C1.sum(0) + (e2['a'] - e1['a']) * cv(init[k]['W0']) + cv(e2['R'] - e1['R'])
        scale = float(W2.abs().max().clamp(min=1e-30))
        err_sum = max(err_sum, float((own - (W2 - W1)).abs().max()) / scale)
        rep = out['W'][k][[names.index(w) for w in replay]].sum(0)
        err_rep = max(err_rep, float((wr[1:-1].sum(0) - rep).abs().max()) / max(float(rep.abs().max()), 1e-30))
    out['extra'] += [f'sh{r_}.g{g}' for r_ in range(len(sh1)) for g in range(1, 9)]
    out['check'].update(shadow_sum=err_sum, shadow_vs_replay=err_rep)
  if per_class:
    A1, A2 = _ld(run, f'attr_{s1:04d}.pt'), _ld(run, f'attr_{s2:04d}.pt')
    n_cls = None
    for k in keys:
      e1, e2 = A1[k], A2[k]
      n_cls = e1['C'].shape[0] - 2                       # the last two groups are the photo and hist
      rho = e2['a'] / e1['a']
      wr = cv(e2['C'][:n_cls]) / e2['C_scale'] - rho * cv(e1['C'][:n_cls]) / e1['C_scale']
      wrb = cv(e2['Cb'][:n_cls]) / e2['Cb_scale'] - (e2['ab'] / e1['ab']) * cv(e1['Cb'][:n_cls]) / e1['Cb_scale']
      extra_W[k] += list(wr)
      extra_b[k] += list(wrb)
    out['extra'] += [f'k{c}' for c in range(n_cls)]
  for k in keys:
    if extra_W[k]:
      out['W'][k] = torch.cat([out['W'][k], torch.stack(extra_W[k])])
      out['b'][k] = torch.cat([out['b'][k], torch.stack(extra_b[k])])
  tracked = {pre + sfx for pre in keys.values() for sfx in ('.weight', '.bias')}
  for n in th1:
    if n not in tracked:
      d = cv(th2[n]) - cv(th1[n])
      if bool((d != 0).any()):
        out['unt'][n] = d
  if out['unt']:
    out['names'] = out['names'] + ['unt']               # integrated by forward-mode JVP, not by the hooks
  return out


# ---------------------------------------------------------------------------------------- integrated gradients

def gauss_legendre(n: int):
  x, w = np.polynomial.legendre.leggauss(n)
  return (x + 1) / 2, w / 2


class GradCapture:
  """Forward hooks on the tracked maps: each map's input h and the gradient delta of a backward through its output,
  giving per-example weight gradients sum_p delta_p h_p^T and bias gradients sum_p delta_p (a map whose output needs
  no gradient gets a leaf output that does, as in rcdl.attribution)."""

  def __init__(self, maps: dict):
    self.maps, self.h, self.d, self.active = maps, {}, {}, True
    self.handles = [lin.register_forward_hook(self._hook(n)) for n, lin in maps.items()]

  def _hook(self, name):
    def hook(mod, inp, out):
      if not self.active or not torch.is_grad_enabled():
        return None
      if not out.requires_grad:
        out = out.detach().requires_grad_(True)
      self.h[name] = inp[0].detach()
      out.register_hook(lambda g: self.d.__setitem__(name, g.detach()))
      return out
    return hook

  def remove(self):
    for h in self.handles:
      h.remove()

  def per_example(self, name):
    h = self.h[name]
    B = h.shape[0]
    h = h.reshape(B, -1, h.shape[-1])
    d = self.d.get(name)
    if d is None:                                        # the function does not depend on this map
      return h.new_zeros(B, self.maps[name].out_features, h.shape[-1]), h.new_zeros(B, self.maps[name].out_features)
    d = d.reshape(B, -1, d.shape[-1])
    return torch.bmm(d.transpose(1, 2), h), d.sum(1)


class Images:
  """A sequence of images from parts: normalized float tensors, or uint8 crops normalized with the ImageNet mean and
  std on access; get(i0, i1) returns that range on the device in the given dtype."""

  def __init__(self, parts, device, dtype):
    self.parts, self.dev, self.dt = [p for p in parts if len(p)], device, dtype
    self.bounds = np.cumsum([0] + [len(p) for p in self.parts])

  def __len__(self):
    return int(self.bounds[-1])

  def get(self, i0, i1):
    out = []
    for p, lo, hi in zip(self.parts, self.bounds[:-1], self.bounds[1:]):
      a, b = max(i0, lo), min(i1, hi)
      if a < b:
        x = p[a - lo:b - lo].to(self.dev)
        if x.dtype == torch.uint8:
          x = (x.float() - MEAN.to(self.dev)[:, None, None]) / STD.to(self.dev)[:, None, None]
        out.append(x.to(self.dt))
    return torch.cat(out)


@torch.no_grad()
def _set_params(model, th1, th2, s):
  for n, p in model.named_parameters():
    p.copy_(th2[n] if s == 1.0 else th1[n] if s == 0.0 else th1[n] + s * (th2[n] - th1[n]))


def _forward_grads(model, cap, x, fns):
  with torch.enable_grad():
    z = model(x)
    vals, grads = {}, {}
    for i, (k, fn) in enumerate(fns.items()):
      cap.d.clear()
      f = fn(z)
      f.sum().backward(retain_graph=i < len(fns) - 1)
      vals[k] = f.detach()
      grads[k] = {m: cap.per_example(m) for m in cap.maps}
  return vals, grads


def _jvp(model, cap, x, fns, dirs):
  """Per-image directional derivative of every fn along dirs (untracked parameters), forward mode."""
  from torch.func import functional_call, jvp
  names = list(dirs)
  params = dict(model.named_parameters())
  prim = tuple(params[n].detach() for n in names)
  tang = tuple(dirs[n] for n in names)

  def f(*ps):
    z = functional_call(model, dict(zip(names, ps)), (x,))
    return tuple(fn(z) for fn in fns.values())
  cap.active = False
  try:
    _, t = jvp(f, prim, tang)
  finally:
    cap.active = True
  return {k: v.detach() for k, v in zip(fns, t)}


def _segment(model, cap, maps, th1, th2, bk, x, fns, nodes, rsel, batch):
  """One straight segment th1 -> th2, every image of x: per fn {'ig': {map: (N, n)}, 'unt': (N,), 'fa', 'fb': (N,),
  'rows': {map: (N, len(rsel), units)}, 'down_bias': {map: (N, len(rsel))}} (CPU float64)."""
  sq, wq = gauss_legendre(nodes)
  N = len(x)
  dirs = bk['unt']
  res = {k: {'ig': {m: [] for m in maps}, 'unt': [], 'fa': [], 'fb': [], 'rows': {m: [] for m in maps},
             'down_bias': {m: [] for m in maps}} for k in fns}
  for i0 in range(0, N, batch):
    xb = x.get(i0, min(N, i0 + batch)) if isinstance(x, Images) else x[i0:i0 + batch]
    G = {k: {m: None for m in maps} for k in fns}
    U = {k: torch.zeros(len(xb), dtype=xb.dtype, device=xb.device) for k in fns}
    for s, w in zip(sq, wq):
      _set_params(model, th1, th2, float(s))
      _, grads = _forward_grads(model, cap, xb, fns)
      for k in fns:
        for m in maps:
          g, gb = grads[k][m]
          if G[k][m] is None:
            G[k][m] = [w * g, w * gb]
          else:
            G[k][m][0].add_(g, alpha=float(w))
            G[k][m][1].add_(gb, alpha=float(w))
      if dirs:
        j = _jvp(model, cap, xb, fns, dirs)
        for k in fns:
          U[k].add_(j[k], alpha=float(w))
    with torch.no_grad():
      _set_params(model, th1, th2, 0.0)
      za = model(xb)
      _set_params(model, th1, th2, 1.0)
      zb = model(xb)
    for k, fn in fns.items():
      r = res[k]
      r['fa'].append(fn(za).double().cpu())
      r['fb'].append(fn(zb).double().cpu())
      r['unt'].append(U[k].double().cpu())
      for m in maps:
        g, gb = G[k][m]
        W, b = bk['W'][m], bk['b'][m]
        r['ig'][m].append((torch.einsum('boi,noi->bn', g, W) + torch.einsum('bo,no->bn', gb, b)).double().cpu())
        if rsel:
          Ws, bs = W[rsel], b[rsel]
          if m.endswith('pw_down'):
            r['rows'][m].append(torch.einsum('boi,noi->bni', g, Ws).double().cpu())
            r['down_bias'][m].append(torch.einsum('bo,no->bn', gb, bs).double().cpu())
          else:
            r['rows'][m].append((torch.einsum('boi,noi->bno', g, Ws) + gb[:, None, :] * bs[None]).double().cpu())
  for k in fns:
    r = res[k]
    for key in ('unt', 'fa', 'fb'):
      r[key] = torch.cat(r[key])
    for key in ('ig', 'rows', 'down_bias'):
      r[key] = {m: torch.cat(v) for m, v in r[key].items() if v}
  return res


def param_ig(model, maps: dict, segments, loader, x, fns: dict, nodes: int = 8, rows=(), tol: float = 1e-3,
             max_nodes: int = 64, batch: int = 64, check=(0,), keep_segments: bool = False) -> dict:
  """Integrated gradients of every fn (logits (N, classes) -> (N,)) along the piecewise-linear path through
  segments [(s1, s2)], split over the buckets loader(s1, s2) returns ({'th1', 'th2': state dicts in the model's
  dtype and device, 'bk': load_buckets output}). x: (N, ...) tensor or Images. rows: bucket names whose IG is also
  kept per row of gate, value and head and per column of pw_down. check: images whose completeness drives the node
  refinement. Returns {'names', 'main' (number of partition buckets, unt excluded), 'segments': [{'s1', 's2', 'nodes',
  'err'}], and per fn: 'ig' {map: (N, n)} (numpy), 'unt' (N,), 'f0', 'f1' (N,) at the path's ends, 'eps' (N,),
  'rows' {map: (N, len(rows), units)}, 'down_bias' {map: (N, len(rows))}, and with keep_segments 'seg' [(N, n) summed
  over maps]}."""
  for p in model.parameters():
    p.requires_grad_(False)
  cap = GradCapture(maps)
  out = {'segments': [], 'names': None}
  budget = tol * 0.1 / max(len(segments), 1)
  try:
    for s1, s2 in segments:
      L = loader(s1, s2)
      bk = L['bk']
      names = [n for n in bk['names'] if n != 'unt'] + bk['extra']
      main = len([n for n in bk['names'] if n != 'unt'])
      if out['names'] is None:
        out['names'], out['main'] = names + (['unt'] if bk['unt'] else []), main
      rsel = [names.index(r) for r in rows if r in names]
      n_used = nodes
      while True:
        res = _segment(model, cap, maps, L['th1'], L['th2'], bk, x, fns, n_used, rsel, batch)
        err = 0.0
        for k in fns:
          tot = sum(v[:, :main].sum(1) for v in res[k]['ig'].values()) + res[k]['unt']
          e = (tot - (res[k]['fb'] - res[k]['fa'])).abs()[list(check)]
          err = max(err, float(e.max()))
        if err <= budget or n_used >= max_nodes:
          break
        n_used *= 2
      out['segments'].append({'s1': s1, 's2': s2, 'nodes': n_used, 'err': err, 'check': bk['check']})
      for k in fns:
        r = res[k]
        acc = out.setdefault(k, {'ig': {}, 'unt': 0.0, 'rows': {}, 'down_bias': {}, 'seg': []})
        if 'f0' not in acc:
          acc['f0'] = r['fa']
        acc['f1'] = r['fb']
        acc['unt'] = acc['unt'] + r['unt']
        for key in ('ig', 'rows', 'down_bias'):
          for m, v in r[key].items():
            acc[key][m] = acc[key][m] + v if m in acc[key] else v
        if keep_segments:
          acc['seg'].append(sum(r['ig'].values()).numpy())
      del L, bk
  finally:
    cap.remove()
  main = out['main']
  for k in fns:
    acc = out[k]
    tot = sum(v[:, :main].sum(1) for v in acc['ig'].values()) + acc['unt']
    df = acc['f1'] - acc['f0']
    acc['eps'] = ((tot - df).abs() / df.abs().clamp(min=0.1)).numpy()
    for key in ('ig', 'rows', 'down_bias'):
      acc[key] = {m: v.numpy() for m, v in acc[key].items()}
    acc['unt'] = np.asarray(acc['unt'])
    acc['f0'], acc['f1'] = acc['f0'].numpy(), acc['f1'].numpy()
  return out


def make_loader(run: str, model, keys: dict, dtype, device, shadows=False, per_class=False):
  """loader(s1, s2) for param_ig: the saved states in (dtype, device) and the buckets of (s1, s2]."""
  cache = {}

  def state(s):
    if s not in cache:
      if len(cache) > 2:
        cache.pop(next(iter(cache)))
      cache[s] = _ld(run, f'model_{s:04d}.pt')
    return cache[s]
  conv = lambda sd: {n: _cv(sd[n], dtype, device) for n, _ in model.named_parameters()}

  def load(s1, s2):
    a, b = state(s1), state(s2)
    bk = load_buckets(run, s1, s2, keys, shadows, per_class, dtype, device, th=(a, b))
    return {'th1': conv(a), 'th2': conv(b), 'bk': bk}
  return load


def bucket_checks(res: dict) -> dict:
  """Largest value over the path's segments of every load_buckets check."""
  out = {}
  for sg in res['segments']:
    for k, v in sg['check'].items():
      out[k] = max(out.get(k, 0.0), v)
  return out


def map_groups(maps) -> dict:
  g = {'total': list(maps), 'blocks': [m for m in maps if m.startswith('stage')], 'head': [m for m in maps if m == 'head']}
  for i in range(4):
    g[f'stage{i}'] = [m for m in maps if m.startswith(f'stage{i}.')]
  g['gate+value'] = [m for m in g['blocks'] if m.endswith(('pw_gate', 'pw_value'))]
  g['down'] = [m for m in g['blocks'] if m.endswith('pw_down')]
  return g


def ig_table(res_fn: dict, names, j: int, groups: dict) -> dict:
  """{bucket: {map group: IG of image j}} (unt under 'unt' in 'total' only)."""
  out = {}
  for i, n in enumerate(names):
    if n == 'unt':
      out[n] = {g: (float(res_fn['unt'][j]) if g == 'total' else 0.0) for g in groups}
      continue
    out[n] = {g: float(sum(res_fn['ig'][m][j, i] for m in ms)) if ms else 0.0 for g, ms in groups.items()}
  return out


def ig_images(res_fn: dict, names, name: str, maps_sel=None) -> np.ndarray:
  """(N,) IG of bucket `name` per image, summed over maps_sel (all maps if None)."""
  if name == 'unt':
    return np.asarray(res_fn['unt'])
  i = names.index(name)
  ms = maps_sel if maps_sel is not None else list(res_fn['ig'])
  return sum(res_fn['ig'][m][:, i] for m in ms)


# ------------------------------------------------------------------------------------------------- statistics

def did_z(obs: float, pseudo) -> dict:
  """z of an observed difference against size-matched pseudo-group differences, centred on their mean."""
  p = np.asarray(pseudo, dtype=np.float64)
  mu = float(p.mean()) if len(p) else float('nan')
  sd = float(p.std(ddof=1)) if len(p) > 1 else float('nan')
  z = (obs - mu) / sd if sd > 0 else float('nan')
  return {'did': float(obs), 'mean_pseudo': mu, 'sd_pseudo': sd, 'z': z, 'n_pseudo': int(len(p))}


def perm_mean_diff(a, b, n: int = 10000, seed: int = 0) -> dict:
  """mean(a) - mean(b) and its one-sided permutation p (labels permuted over a u b)."""
  a, b = np.asarray(a, float), np.asarray(b, float)
  obs = a.mean() - b.mean()
  pool, rng = np.concatenate([a, b]), np.random.default_rng(seed)
  hits = 0
  for _ in range(n):
    q = rng.permutation(pool)
    hits += (q[:len(a)].mean() - q[len(a):].mean()) >= obs
  return {'diff': float(obs), 'p': (hits + 1) / (n + 1)}


def slope_perm(x, y, n: int = 10000, seed: int = 0) -> dict:
  """OLS slope of y on x and its two-sided permutation p (x permuted)."""
  x, y = np.asarray(x, float), np.asarray(y, float)
  xc = x - x.mean()
  den = (xc ** 2).sum()
  if den == 0:
    return {'beta': float('nan'), 'p': float('nan')}
  beta = (xc * (y - y.mean())).sum() / den
  rng = np.random.default_rng(seed)
  hits = sum(abs((rng.permutation(xc) * (y - y.mean())).sum() / den) >= abs(beta) for _ in range(n))
  return {'beta': float(beta), 'p': (hits + 1) / (n + 1)}


def signed_rank_test(d, n: int = 10000, seed: int = 0) -> dict:
  """One-sided Wilcoxon signed-rank test that the differences d are above 0: W+ (sum of the mid-ranks of |d| over
  d > 0, zeros dropped) and its p from random sign flips."""
  d = np.asarray(d, float)
  d = d[d != 0]
  if len(d) == 0:
    return {'n': 0, 'W': float('nan'), 'p': float('nan')}
  a = np.abs(d)
  order = np.argsort(a, kind='stable')
  ranks = np.empty(len(a))
  ranks[order] = np.arange(1, len(a) + 1)
  for v in np.unique(a):                                 # mid-ranks for ties
    ranks[a == v] = ranks[a == v].mean()
  W = ranks[d > 0].sum()
  rng = np.random.default_rng(seed)
  flips = rng.random((n, len(d))) < 0.5
  null = (flips * ranks).sum(1)
  return {'n': int(len(d)), 'W': float(W), 'p': float(((null >= W).sum() + 1) / (n + 1))}


# ------------------------------------------------------------------------------------------------ unit states

def block_states(model, x) -> tuple:
  """(logits, {block: {'h': (B, P, d) LayerNorm output, 'g', 'v', 'u': (B, P, units)}}) at the current weights."""
  hs = {}
  hooks = [b.norm.register_forward_hook(lambda m, i, o, n=n: hs.__setitem__(n, o)) for n, b in model.blocks()]
  try:
    with torch.no_grad():
      z = model(x)
  finally:
    for h in hooks:
      h.remove()
  out = {}
  with torch.no_grad():
    for n, b in model.blocks():
      h = hs[n].flatten(1, 2)
      g, v = F.linear(h, b.pw_gate.weight, b.pw_gate.bias), F.linear(h, b.pw_value.weight, b.pw_value.bias)
      out[n] = {'h': h, 'g': g, 'v': v, 'u': F.silu(g) * v}
  return z, out


@torch.no_grad()
def unit_star(model, images, batch: int = 128) -> np.ndarray:
  """(N, blocks x units) u* = u at the unit's strongest position (argmax over positions), block order."""
  out = []
  for i0 in range(0, len(images), batch):
    xb = images.get(i0, min(len(images), i0 + batch)) if isinstance(images, Images) else images[i0:i0 + batch]
    _, st = block_states(model, xb)
    out.append(torch.cat([s['u'].max(1).values for s in st.values()], 1).double().cpu())
  return torch.cat(out).numpy()


@torch.no_grad()
def patch_carriers(model, xP, u_ref: dict, fn, chunk: int = 64) -> dict:
  """{block: (units,)} c_n = fn(x_P) - fn(x_P with unit n's SwiGLU product set to u_ref[block][..., n] at every
  position), at the model's current weights. u_ref: {block: (1, H, W, units)} (the pw_down input)."""
  base = float(fn(model(xP))[0])
  out = {}
  for name, b in model.blocks():
    U = u_ref[name]
    units = U.shape[-1]
    vals = []
    for i0 in range(0, units, chunk):
      idx = list(range(i0, min(units, i0 + chunk)))

      def pre(mod, args, idx=idx):
        u = args[0].clone()
        for r, n in enumerate(idx):
          u[r, ..., n] = U[0, ..., n]
        return (u,)
      hk = b.pw_down.register_forward_pre_hook(pre)
      try:
        vals.append(fn(model(xP.expand(len(idx), *xP.shape[1:]))).double().cpu())
      finally:
        hk.remove()
    out[name] = (base - torch.cat(vals)).numpy()
  return out


@torch.no_grad()
def pw_down_inputs(model, xP) -> dict:
  """{block: (1, H, W, units)} the SwiGLU products (pw_down inputs) on x_P at the current weights."""
  cap = {}
  hooks = [b.pw_down.register_forward_pre_hook(lambda m, a, n=n: cap.__setitem__(n, a[0].clone()))
           for n, b in model.blocks()]
  try:
    model(xP)
  finally:
    for h in hooks:
      h.remove()
  return cap


def c50(c: np.ndarray) -> np.ndarray:
  """Indices of the fewest units holding >= 50% of sum max(c, 0)."""
  cp = np.maximum(c, 0)
  if cp.sum() <= 0:
    return np.zeros(0, int)
  order = np.argsort(-cp, kind='stable')
  k = int(np.searchsorted(np.cumsum(cp[order]), 0.5 * cp.sum()) + 1)
  return order[:k]


def pole_sets(atlas: dict, blocks, t: int, s: int, poles: int = 20) -> tuple:
  """(B_pole, U_pole) boolean (blocks x units) from atlas R (units, classes): class among the `poles` largest or
  smallest R[n, .]."""
  B, U = [], []
  for b in blocks:
    R = atlas[b]['R'].double()
    top = torch.argsort(R, 1, descending=True)[:, :poles]
    bot = torch.argsort(R, 1)[:, :poles]
    pol = torch.cat([top, bot], 1)
    B.append((pol == t).any(1).numpy())
    U.append((pol == s).any(1).numpy())
  return np.concatenate(B), np.concatenate(U)


def match_noncarriers(carriers, pool_mask, block_of, key, k: int = 3) -> dict:
  """{carrier: [up to k non-carriers of the same block in pool_mask nearest on key]}."""
  cs = set(int(c) for c in carriers)
  out = {}
  for c in carriers:
    cand = [n for n in np.flatnonzero(pool_mask) if n not in cs and block_of[n] == block_of[c]]
    cand.sort(key=lambda n: (abs(key[n] - key[c]), n))
    if cand:
      out[int(c)] = [int(n) for n in cand[:k]]
  return out


def own_upstream_split(st_tau, st_w, model_tau, model_w, bk_I1, keys_names, blocks, e1=None) -> dict:
  """Exact split of every unit's change at its strongest position on x_P (module docstring, M3). st_*: block_states
  of x_P (batch 1) at theta_tau and theta_w; bk_I1: buckets of [0, tau]; e1: optional {'attr', 'init', 'state'}
  (E1 at 25,020) for the two-level dual form. Returns arrays over blocks x units: du, own_u, up_u, own_P, own by bucket
  (n, U), and with e1 the class terms T (classes, U), the other terms and the residuals of both identities."""
  part = [n for n in bk_I1['names'] if n != 'unt']
  rec = {k: [] for k in ('du', 'own', 'up', 'own_by', 'T', 'init', 'R', 'warm', 'resid', 'resid_pre', 'A', 'Bv')}
  W_tau = {n: b for n, b in model_tau.blocks()}
  W_w = {n: b for n, b in model_w.blocks()}
  for blk in blocks:
    a, w = st_tau[blk], st_w[blk]
    p = a['u'][0].argmax(0)                              # (units,) strongest position at theta_tau
    take = lambda t_, p=p: t_[0][p, torch.arange(len(p))]
    ht = a['h'][0][p]                                    # (units, d)
    hw = w['h'][0][p]
    gt, gw, vt, vw = take(a['g']), take(w['g']), take(a['v']), take(w['v'])
    ut, uw = F.silu(gt) * vt, F.silu(gw) * vw
    dg, dv = gt - gw, vt - vw
    sec = torch.where(dg.abs() > 1e-12, (F.silu(gt) - F.silu(gw)) / torch.where(dg.abs() > 1e-12, dg, torch.ones_like(dg)),
                      torch.sigmoid(gt) * (1 + gt * (1 - torch.sigmoid(gt))))
    A = sec * (vt + vw) / 2
    Bv = (F.silu(gt) + F.silu(gw)) / 2
    parts = {}
    for mp in ('pw_gate', 'pw_value'):
      key = f'{blk}.{mp}'
      Wb, bb = bk_I1['W'][key][:len(part)], bk_I1['b'][key][:len(part)]     # (n, units, d), (n, units)
      own_by = torch.einsum('nud,ud->nu', Wb.to(ht.dtype), ht) + bb.to(ht.dtype)
      Ww = getattr(W_w[blk], mp).weight.to(ht.dtype)
      up = (Ww * (ht - hw)).sum(1)
      parts[mp] = (own_by, up, Ww)
    own_by = A * parts['pw_gate'][0] + Bv * parts['pw_value'][0]
    up = A * parts['pw_gate'][1] + Bv * parts['pw_value'][1]
    du = ut - uw
    rec['du'].append(du)
    rec['own_by'].append(own_by)
    rec['own'].append(own_by.sum(0))
    rec['up'].append(up)
    rec['resid'].append((own_by.sum(0) + up - du).abs())
    rec['A'].append(A)
    rec['Bv'].append(Bv)
    if e1 is not None:
      dh = ht - hw
      T = 0.0
      other = {'init': 0.0, 'R': 0.0, 'warm': 0.0}
      for mp, coef in (('pw_gate', A), ('pw_value', Bv)):
        key = f'{blk}.{mp}'
        ek = e1['attr'][key]
        C = ek['C'].to(dh.device, dh.dtype) / ek['C_scale']           # (classes, units, d)
        W0 = e1['init'][key]['W0'].to(dh.device, dh.dtype)
        R = ek['R'].to(dh.device, dh.dtype)
        pre = keys_names[key]
        Wm = e1['state'][pre + '.weight'].to(dh.device, dh.dtype)
        warm = parts[mp][2] - Wm
        T = T + coef * torch.einsum('kud,ud->ku', C, dh)
        other['init'] = other['init'] + coef * (ek['a'] * W0 * dh).sum(1)
        other['R'] = other['R'] + coef * (R * dh).sum(1)
        other['warm'] = other['warm'] + coef * (warm * dh).sum(1)
      rec['T'].append(T)
      for k_, v in other.items():
        rec[k_].append(v)
      rec['resid_pre'].append((T.sum(0) + other['init'] + other['R'] + other['warm'] - up).abs())
  out = {}
  for k, v in rec.items():
    if v:
      out[k] = torch.cat(v, -1).double().cpu().numpy()
  return out


def class_percentile(T: np.ndarray, du: np.ndarray, c: int) -> np.ndarray:
  """(units,) q_n^c = #{k : o T[k, n] < o T[c, n]} / (classes - 1), o = sign(du_n)."""
  o = np.sign(du)
  o[o == 0] = 1
  To = T * o[None]
  return (To < To[c][None]).sum(0) / (T.shape[0] - 1)


# ----------------------------------------------------------------------------------------------------- swaps

def swap_states(th_w: dict, th_tau: dict, photo: dict, keys: dict, S: list, untracked_blocks=()) -> dict:
  """theta^P(S), theta^tot(S), theta^-P(S) as state dicts (module docstring, M4). photo: {map: (dW, db)} the photo's
  parts at tau (all its writes since the branch); S: maps; untracked_blocks: module prefixes whose untracked
  parameters theta^tot also takes from theta_tau (standard optimizer)."""
  P, tot, mP = dict(th_w), dict(th_w), dict(th_tau)
  for m in S:
    pre = keys[m]
    dW, db = photo[m]
    P[pre + '.weight'] = th_w[pre + '.weight'] + dW
    P[pre + '.bias'] = th_w[pre + '.bias'] + db
    mP[pre + '.weight'] = th_tau[pre + '.weight'] - dW
    mP[pre + '.bias'] = th_tau[pre + '.bias'] - db
    for sfx in ('.weight', '.bias'):
      tot[pre + sfx] = th_tau[pre + sfx]
  for pre in untracked_blocks:
    for n in th_w:
      if n.startswith(pre + '.'):
        tot[n] = th_tau[n]
  return {'P': P, 'tot': tot, '-P': mP}


def swap_set_maps(name: str, maps) -> list:
  st = lambda i: [m for m in maps if m.startswith(f'stage{i}.')]
  if name == 'stage3':
    return st(3)
  if name == 'stage3+head':
    return st(3) + ['head']
  if name == 'head':
    return ['head']
  if name == 'stage2':
    return st(2)
  if name == 'stages0-1':
    return st(0) + st(1)
  if name.startswith('stage3.block'):
    return [m for m in maps if m.startswith(name + '.')]
  if name == 'stage3.gate+value':
    return [m for m in st(3) if m.endswith(('pw_gate', 'pw_value'))]
  if name == 'stage3.down':
    return [m for m in st(3) if m.endswith('pw_down')]
  raise ValueError(name)


@torch.no_grad()
def readout(model, theta: dict, xP, t: int, s: int) -> dict:
  model.load_state_dict({k: v.to(next(model.parameters())) for k, v in theta.items()})
  z = model(xP)[0].double()
  p = F.softmax(z, -1)
  return {'argmax': int(z.argmax()), 'p_t': float(p[t]), 'p_s': float(p[s]), 'm': float(z[t] - z[s])}


# -------------------------------------------------------------------------------------------- evaluation sets

def candidate_exclusions(sels) -> set:
  """Global validation indices never used for evaluation: every arm's excluded_from_eval (the source and target
  probe_8 images, every passing validation candidate) and every photo and reserve image, over all selection files."""
  ex = set()
  for sel in sels:
    for arm in sel['arms'].values():
      ex |= {int(i) for i in arm.get('excluded_from_eval', [])}
      for r in arm.get('photos', []) + arm.get('reserve', []):
        kind, idx = r['photo'].split(':')
        if kind == 'val':
          ex.add(int(idx))
  return ex


def eval_set_indices(vlabels: np.ndarray, arm: dict, excluded, first_n: int = 10, n_classes: int = 1000) -> dict:
  """V_ref: the first first_n validation images (parquet order) of every class except the target, the source and
  their relatives; V_umb: the source's images; V_bee: the target's; all without the excluded indices."""
  t, s = arm['target'], arm['source']
  drop = {t, s} | set(arm['rel_target']) | set(arm['rel_source'])
  ex = set(int(i) for i in excluded)
  ref = []
  for c in range(n_classes):
    if c not in drop:
      ref += [int(i) for i in np.flatnonzero(vlabels == c)[:first_n] if int(i) not in ex]
  return {'V_ref': ref, 'V_umb': [int(i) for i in np.flatnonzero(vlabels == s) if int(i) not in ex],
          'V_bee': [int(i) for i in np.flatnonzero(vlabels == t) if int(i) not in ex]}


def find_arm(sels, arm: str) -> dict:
  for sel in sels:
    if arm in sel['arms']:
      return sel['arms'][arm]
  raise KeyError(f'arm {arm} not in the selection files')


def cmd_vsets(a):
  """Index sets of one arm, the uint8 images of V_umb and V_bee, and V_ref as positions into the first-10 cache
  (convnext_atlas.probe_set with 10 per class, shared by all arms)."""
  from experiments.convnext_atlas import probe_set
  from experiments.convnext_umbrella import val_images, val_index
  sels = [json.load(open(f)) for f in a.sel]
  arm = find_arm(sels, a.arm)
  _, vlabels = val_index(a.val)
  idx = eval_set_indices(vlabels, arm, candidate_exclusions(sels))
  probe_set(a.val, 10, a.first10)                         # builds the cache if missing
  first10 = np.concatenate([np.flatnonzero(vlabels == c)[:10] for c in range(1000)])
  pos = {int(g): i for i, g in enumerate(first10)}
  out = {'arm': a.arm, 'target': arm['target'], 'source': arm['source'], 'idx': idx, 'first10': a.first10,
         'V_ref_pos': [pos[g] for g in idx['V_ref']],
         'labels': {k: [int(vlabels[i]) for i in v] for k, v in idx.items()},
         'x': {k: val_images(a.val, idx[k]) for k in ('V_umb', 'V_bee')}}
  torch.save(out, a.out + '.tmp')
  os.replace(a.out + '.tmp', a.out)
  print({k: len(v) for k, v in idx.items()}, flush=True)


def load_vsets(path: str) -> dict:
  """{'V_ref', 'V_umb', 'V_bee': {'x': uint8 (N, 3, 224, 224), 'idx', 'y'}}."""
  d = torch.load(path, weights_only=False)
  f10 = torch.load(d['first10'], weights_only=False)
  out = {'V_ref': {'x': f10['x'][torch.tensor(d['V_ref_pos'], dtype=torch.long)], 'idx': d['idx']['V_ref'],
                   'y': d['labels']['V_ref']}}
  for k in ('V_umb', 'V_bee'):
    out[k] = {'x': d['x'][k], 'idx': d['idx'][k], 'y': d['labels'][k]}
  return out


# ------------------------------------------------------------------------------------------------ provenance

def provenance_pi(res: dict, t: int, s: int) -> dict:
  """From atlas_step output with pole classes (t, s) per unit: {'target', 'source', 'blocks': {block: {'pi_value_t',
  'pi_value_s', 'pi_gate_t', 'pi_gate_s': (units,), 'resid'}}}, pi_n^c = #{k != c : s_c W[n, c, c] > s_c W[n, c, k]} /
  (classes - 1) with W the writers of the map and s_c = sign(class mean of the map's pre-activation on c)."""
  out = {'target': t, 'source': s, 'blocks': {}}
  for b, r in res.items():
    rec = {'resid': r['resid']}
    for mp, key in (('value', 'Vm'), ('gate', 'Gm')):
      Wr = r[f'{mp}_writers'].double()                 # (units, 2, classes)
      M = r[key].double()
      mask = torch.ones(Wr.shape[-1], dtype=torch.bool)
      for j, c in enumerate((t, s)):
        sg = torch.sign(M[:, c])
        sg[sg == 0] = 1
        own = (sg * Wr[:, j, c])[:, None]
        oth = sg[:, None] * Wr[:, j]
        mask[:] = True
        mask[c] = False
        rec[f'pi_{mp}_{"t" if j == 0 else "s"}'] = ((own > oth)[:, mask].sum(1) / (Wr.shape[-1] - 1)).numpy()
    out['blocks'][b] = rec
  return out


def cmd_provenance(a):
  """pi_n^c for the arm's target and source from E1's atlas writers (convnext_atlas.atlas_step with the pole
  classes fixed to (t, s))."""
  from experiments.convnext_atlas import atlas_step, probe_set
  dev = torch.device(a.device)
  sel = find_arm([json.load(open(f)) for f in a.sel], a.arm)
  t, s = sel['target'], sel['source']
  model = build_convnext('tiny').to(dev).eval()
  model.load_state_dict(torch.load(os.path.join(a.e1, 'model_25020.pt'), map_location=dev))
  attr = torch.load(os.path.join(a.e1, 'attr_25020.pt'), map_location='cpu', weights_only=False)
  init = torch.load(os.path.join(a.e1, 'attr_init.pt'), map_location='cpu', weights_only=False)
  x, y = probe_set(a.val, 8, a.probe_cache)
  units = {n: b.pw_gate.out_features for n, b in model.blocks()}
  pidx = {n: torch.tensor([[t, s]] * u) for n, u in units.items()}
  res = atlas_step(model, attr, init, x, y, 1000, poles_idx=pidx, device=dev)
  out = provenance_pi(res, t, s)
  torch.save(out, a.out + '.tmp')
  os.replace(a.out + '.tmp', a.out)
  print({b: round(float(r['resid']), 6) for b, r in res.items()}, flush=True)


# ------------------------------------------------------------------------------------------------- analysis

def _jsonable(o):
  if isinstance(o, dict):
    return {str(k): _jsonable(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [_jsonable(v) for v in o]
  if isinstance(o, (np.floating, float)):
    return None if not math.isfinite(float(o)) else float(o)
  if isinstance(o, (np.integer,)):
    return int(o)
  if isinstance(o, np.bool_):
    return bool(o)
  if isinstance(o, np.ndarray):
    return _jsonable(o.tolist())
  if torch.is_tensor(o):
    return _jsonable(o.tolist())
  return o


def _margin(t, s):
  return lambda z: z[:, t] - z[:, s]


def _fns(t, s):
  return {'m': _margin(t, s), 'zt': lambda z: z[:, t], 'zs': lambda z: z[:, s]}


def analyze(run: str, ctx: dict, untaught: str = None, skip=()) -> tuple:
  """M1-M4 of one taught (or solo) run; returns (summary dict, arrays dict). ctx: 'make_model' (callable -> fp32
  module on CPU), 'device', 'dtype' (photo passes), 'vdtype' (V-set passes), 'x_P' (1, 3, H, W) normalized, 'target',
  'source' (None: top-1 of x_P at theta_w), 'kind', 'vsets' (load_vsets form or None), 'e1' ({'atlas', 'attr',
  'init', 'state'} or None), 'prov' (cmd_provenance output or None), and the numeric options 'nodes', 'vnodes',
  'step_nodes', 'tol', 'max_nodes', 'batch', 'vbatch', 'perm', 'n_near', 'n_mp', 'q_sel', 'rows'."""
  dev, dt, vdt = ctx['device'], ctx['dtype'], ctx['vdtype']
  info = run_info(run)
  cfg, tau = info['cfg'], info['tau']
  mode = cfg['mode']
  assert mode in ('taught', 'solo'), 'analyze: taught and solo runs (untaught runs are the paired baseline)'
  end = tau + cfg['phase2'] if mode == 'taught' else tau
  model = ctx['make_model']().to(dev, dt).eval()
  vmodel = ctx['make_model']().to(dev, vdt).eval()
  maps, vmaps = tracked_maps(model), tracked_maps(vmodel)
  keys = map_keys(model, maps)
  groups = map_groups(maps)
  xP = ctx['x_P'].to(dev, dt)
  th = {s: _ld(run, f'model_{s:04d}.pt') for s in sorted({0, 1, tau, end})}
  conv = lambda sd, d: {k: _cv(v, d, dev) for k, v in sd.items()}
  model.load_state_dict(conv(th[0], dt))
  with torch.no_grad():
    zw = model(xP)[0]
  t = int(ctx['target'])
  s = ctx['source']
  if s is None:                                          # controls: the top-1 class at theta_w (not the target)
    order = torch.argsort(zw, descending=True).tolist()
    s = order[0] if order[0] != t else order[1]
  s = int(s)
  fns = _fns(t, s)
  m_of = lambda step: float(readout(model, conv(th[step], dt), xP, t, s)['m'])
  summ = {'run': run, 'mode': mode, 'opt': cfg['opt'], 'seed': cfg['seed'], 'lr': cfg['lr'], 'arm': cfg['arm'],
          'photo': cfg['photo'], 'kind': ctx['kind'], 'target': t, 'source': s, 'tau': tau,
          'censored': info['censored'], 'end': end, 'm_P': {'0': m_of(0), 'tau': m_of(tau), 'end': m_of(end)},
          'top1_path': [r.get('top1') for r in info['log'] if r.get('phase') == 1], 'checks': {}}
  arrays = {}
  dm1 = summ['m_P']['tau'] - summ['m_P']['0']
  opt = dict(nodes=ctx['nodes'], tol=ctx['tol'], max_nodes=ctx['max_nodes'])

  # vsets and features at theta_w
  vs = ctx.get('vsets')
  near = mp_idx = None
  if vs is not None:
    assert len(vs['V_ref']['x']) and len(vs['V_umb']['x']), 'V_ref and V_umb must not be empty'
    vmodel.load_state_dict(conv(th[0], vdt))
    ref_umb = Images([vs['V_ref']['x'], vs['V_umb']['x']], dev, vdt)
    with torch.no_grad():
      feats = torch.cat([vmodel.features(ref_umb.get(i, min(len(ref_umb), i + ctx['vbatch'])))
                         for i in range(0, len(ref_umb), ctx['vbatch'])]).double()
      fP = vmodel.features(xP.to(vdt)).double()
      cos = F.cosine_similarity(feats, fP, dim=1).cpu().numpy()
      near = np.argsort(-cos, kind='stable')[:ctx['n_near']]          # positions in V_ref + V_umb
      zu = vmodel(Images([vs['V_umb']['x']], dev, vdt).get(0, len(vs['V_umb']['x']))).double()
      mu = (zu[:, t] - zu[:, s]).cpu().numpy()
    mp_idx = np.argsort(np.abs(mu - float(zw[t] - zw[s])), kind='stable')[:ctx['n_mp']]
    nr = len(vs['V_ref']['x'])
    near_x = torch.cat([vs['V_ref']['x'][torch.as_tensor(near[near < nr])],
                        vs['V_umb']['x'][torch.as_tensor(near[near >= nr] - nr)]])
    near = np.concatenate([near[near < nr], near[near >= nr]])        # the order of near_x
    arrays['cos'] = cos
    arrays['near'] = near

  # M1: feed and shelter on the common path (taught runs with an untaught run)
  if mode == 'taught' and untaught is not None and 'M1' not in skip:
    xm = [xP]
    if mp_idx is not None:
      xm.append(Images([vs['V_umb']['x'][torch.as_tensor(mp_idx)]], dev, dt).get(0, len(mp_idx)))
    xm = torch.cat(xm)
    ut_info = run_info(untaught)
    for k in ('seed', 'lr', 'opt', 'replay', 'rest', 'rest_exclude', 'arm'):
      assert ut_info['cfg'].get(k) == cfg.get(k), f'untaught run: different {k} (not the same replay stream)'
    paths = {'I1': common_path([run, untaught], 0, tau), 'I2': common_path([run, untaught], tau, end)}
    Fr = {}
    for who, rd in (('taught', run), ('untaught', untaught)):
      ld = make_loader(rd, model, keys, dt, dev, shadows=True)
      for I, verts in paths.items():
        Fr[(who, I)] = param_ig(model, maps, segments_of(verts), ld, xm, fns, batch=ctx['batch'],
                                check=(0,), **opt)
    names = Fr[('taught', 'I1')]['names']
    pseudo = [n for n in names if n.startswith('sh')]
    real = [n for n in names if not n.startswith('sh')]
    M1 = {'paths': paths, 'dm_P': {'I1': dm1, 'I2': summ['m_P']['end'] - summ['m_P']['tau']}, 'F': {}, 'DiD': {},
          'z': {}, 'completeness': {}}
    full = {}
    for (who, I), r in Fr.items():
      full[(who, I)] = {fk: ig_table(r[fk], names, 0, groups) for fk in fns}
      M1['F'].setdefault(who, {})[I] = {fk: {n: v for n, v in tb.items() if n in real} for fk, tb in full[(who, I)].items()}
      M1['completeness'][f'{who}_{I}'] = {'eps_P': float(r['m']['eps'][0]), 'segments': len(r['segments']),
                                          'max_nodes': max(sg['nodes'] for sg in r['segments'])}
      summ['checks'][f'buckets_{who}_{I}'] = bucket_checks(r)
    arrays['M1_pseudo_DiD'] = {}
    for I in paths:
      T, U = full[('taught', I)]['m'], full[('untaught', I)]['m']
      M1['DiD'][I] = {n: {g: T[n][g] - U[n][g] for g in groups} for n in real}
      arrays['M1_pseudo_DiD'][I] = {g: np.array([T[q][g] - U[q][g] for q in pseudo]) for g in groups}
      M1['z'][I] = {}
      for n in ('BEE', 'UMB', 'A', 'B', 'RELb', 'RELu', 'OTH', 'hist', 'P'):
        if n in real:
          M1['z'][I][n] = {g: did_z(T[n][g] - U[n][g], [T[q][g] - U[q][g] for q in pseudo]) for g in groups}
      M1.setdefault('z_zt', {})[I] = {g: did_z(full[('taught', I)]['zt']['BEE'][g] - full[('untaught', I)]['zt']['BEE'][g],
                                               [full[('taught', I)]['zt'][q][g] - full[('untaught', I)]['zt'][q][g]
                                                for q in pseudo]) for g in ('blocks', 'total')}
      if mp_idx is not None and len(mp_idx):
        trip = {}
        for n in real:
          for g in ('blocks', 'total'):
            ms = groups[g]
            dP = ig_images(Fr[('taught', I)]['m'], names, n, ms) - ig_images(Fr[('untaught', I)]['m'], names, n, ms)
            trip.setdefault(n, {})[g] = float(dP[0] - dP[1:].mean())
        M1.setdefault('triple', {})[I] = trip
    zb = M1['z']['I2'].get('BEE', {}).get('blocks', {})
    d2 = M1['DiD']['I2']
    safe = lambda v, d: v / d if d != 0 else float('nan')
    M1['feed'] = {'DiD_BEE_I2': d2['BEE']['blocks'], 'z_BEE_I2': zb.get('z'), 'DiD_A_I2': d2['A']['blocks'],
                  'DiD_B_I2': d2['B']['blocks'], 'phi_BEE_I2': safe(d2['BEE']['blocks'], dm1),
                  'phi_BEE_I1': safe(M1['DiD']['I1']['BEE']['blocks'], dm1),
                  'z_BEE_I1': M1['z']['I1']['BEE']['blocks']['z']}
    Ft2 = M1['F']['taught']['I2']['m']
    M1['shelter'] = {'E_OTH': safe(-Ft2['OTH']['blocks'], dm1), 'E_dec': safe(-Ft2['dec_P']['total'], dm1),
                     'E_dec_blocks': safe(-Ft2['dec_P']['blocks'], dm1)}
    M1['umbrella'] = {I: {g: M1['z'][I]['UMB'][g] for g in groups} for I in paths}
    dec = summ['m_P']['end'] - summ['m_P']['tau']
    M1['decline'] = {n: safe(Ft2[n]['total'], dec) for n in real}
    M1['rho_retained'] = safe(summ['m_P']['end'] - summ['m_P']['0'], dm1)
    summ['M1'] = M1

  # fine path of the taught (or solo) run over I1 (every step): F_fine, M4 shares, per-row IG
  ld_fine = make_loader(run, model, keys, dt, dev)
  verts = path_vertices(run, 0, tau, shadows=False)
  fine = param_ig(model, maps, segments_of(verts), ld_fine, xP, fns, rows=ctx['rows'], batch=1, **opt)
  fnames = fine['names']
  summ['F_fine_I1'] = ig_table(fine['m'], fnames, 0, groups)
  summ['checks']['fine_I1'] = {'eps_P': float(fine['m']['eps'][0]), 'n_segments': len(verts) - 1,
                               'max_nodes': max(sg['nodes'] for sg in fine['segments'])} | bucket_checks(fine)

  # per-class writes (descriptive): F^k over I1 and I2 on one-segment paths
  if ctx.get('per_class'):
    ldk = make_loader(run, model, keys, dt, dev, per_class=True)
    pc = {}
    for I, (a_, b_) in (('I1', (0, tau)), ('I2', (tau, end))):
      if b_ > a_:
        r = param_ig(model, maps, [(a_, b_)], ldk, xP, {'m': fns['m']}, batch=1, **opt)
        cls = [n for n in r['names'] if n.startswith('k')]
        vals = np.array([ig_images(r['m'], r['names'], n, groups['blocks'])[0] for n in cls])
        pc[I] = vals
        top = np.argsort(-vals)[:10]
        summ.setdefault('per_class', {})[I] = {'top_feed': [(int(k), float(vals[k])) for k in top],
                                               'top_erode': [(int(k), float(vals[k])) for k in np.argsort(vals)[:10]],
                                               'eps_P': float(r['m']['eps'][0])}
    arrays['per_class_blocks'] = pc
  arrays['rows_I1'] = {'names': [r for r in ctx['rows'] if r in fnames], 'rows': fine['m']['rows'],
                       'down_bias': fine['m']['down_bias']}
  rows_sum = {m: (fine['m']['rows'][m][0].sum(-1) + (fine['m']['down_bias'][m][0] if m in fine['m']['down_bias']
                                                     else 0.0)) for m in fine['m']['rows']}
  rsel = [fnames.index(r) for r in ctx['rows'] if r in fnames]
  summ['checks']['rows_vs_map'] = max((float(np.abs(rows_sum[m] - fine['m']['ig'][m][0, rsel]).max())
                                       for m in rows_sum), default=0.0)

  # M4 localization
  if 'M4' not in skip:
    iP = fnames.index('P')
    totP = sum(fine['m']['ig'][m][0, iP] for m in maps)
    blocks_ = [n for n, _ in model.blocks()]
    S = {b: float(sum(fine['m']['ig'][m][0, iP] for m in maps if m.startswith(b + '.')) / totP) for b in blocks_}
    S['head'] = float(fine['m']['ig']['head'][0, iP] / totP)
    M4 = {'IG_P_total': float(totP), 'S': S, 'S_late': sum(v for k, v in S.items() if k.startswith('stage3')),
          'S_head': S['head'], 'swaps': {}}
    bkP = load_buckets(run, 0, tau, keys, dtype=dt, device=dev, th=(th[0], th[tau]))
    ip = bkP['names'].index('P')
    photo = {m: (bkP['W'][m][ip], bkP['b'][m][ip]) for m in maps}
    thw, tht = conv(th[0], dt), conv(th[tau], dt)
    unt_blocks = []
    for name in SWAP_SETS:
      Sm = swap_set_maps(name, maps)
      if bkP['unt']:
        unt_blocks = sorted({keys[m].rsplit('.', 1)[0] for m in Sm if m != 'head'})
      sw = swap_states(thw, tht, photo, keys, Sm, unt_blocks)
      M4['swaps'][name] = {k: readout(model, v, xP, t, s) for k, v in sw.items()}
    M4['readout_w'] = readout(model, thw, xP, t, s)
    M4['readout_tau'] = readout(model, tht, xP, t, s)
    sw_all = swap_states(thw, tht, photo, keys, list(maps))
    summ['checks']['swap_all_identity'] = max(float((sw_all['P'][k] + sw_all['-P'][k] - thw[k] - tht[k]).abs().max())
                                              for k in thw)
    summ['M4'] = M4

  # M2 footprint
  if vs is not None and 'M2' not in skip:
    nr, nu = len(vs['V_ref']['x']), len(vs['V_umb']['x'])
    allx = Images([xP.to(vdt), vs['V_ref']['x'], vs['V_umb']['x'], vs['V_bee']['x']], dev, vdt)
    ofs = {'V_ref': 1, 'V_umb': 1 + nr, 'V_bee': 1 + nr + nu}
    near_pos = 1 + near                                  # V_ref then V_umb, right after x_P
    vopt = dict(nodes=ctx['vnodes'], tol=ctx['tol'], max_nodes=ctx['max_nodes'], batch=ctx['vbatch'])
    ldv = make_loader(run, vmodel, map_keys(vmodel, vmaps), vdt, dev)
    m_only = {'m': fns['m']}
    rv = {'I1': param_ig(vmodel, vmaps, [(0, tau)], ldv, allx, m_only, **vopt)}
    if end != tau:
      rv['end'] = param_ig(vmodel, vmaps, [(0, end)], ldv, allx, m_only, **vopt)
    else:
      rv['end'] = rv['I1']
    xn = Images([xP.to(vdt), near_x], dev, vdt)
    rv['I_1'] = param_ig(vmodel, vmaps, [(0, 1)], ldv, xn, m_only, **vopt)
    step = param_ig(vmodel, vmaps, segments_of(verts), ldv, xn, m_only, keep_segments=True,
                    **(vopt | {'nodes': ctx['step_nodes']}))
    vn = rv['I1']['names']
    iP = vn.index('P')
    ratio = {}
    for I, r in rv.items():
      ig = ig_images(r['m'], vn, 'P')
      ratio[I] = ig / ig[0]
    G = {}
    for I in ('I1', 'end'):
      for V, o in ofs.items():
        n = len(vs[V]['x'])
        G.setdefault(V, {})[I] = float(ratio[I][o:o + n].mean()) if n else float('nan')
      G.setdefault('V_near', {})[I] = float(ratio[I][near_pos].mean())
    G['V_near']['I_1'] = float(ratio['I_1'][1:].mean())
    L = 1 - G['V_near']['I1'] / G['V_near']['I_1'] if G['V_near']['I_1'] != 0 else float('nan')
    de = rv['end']['m']['f1'] - rv['end']['m']['f0']
    Nn = {V: float(de[o:o + len(vs[V]['x'])].mean() / de[0]) for V, o in ofs.items() if len(vs[V]['x'])}
    Nn['V_near'] = float(de[near_pos].mean() / de[0])
    K = 1 - Nn['V_near'] / G['V_near']['end'] if G['V_near']['end'] != 0 else float('nan')
    n_split = {n: float(ig_images(rv['end']['m'], vn, n)[near_pos].mean() / de[0]) for n in vn}
    seg = np.stack(step['m']['seg'])                    # (steps, 1 + n_near, n)
    Gt = (seg[:, 1:, iP].mean(1) / seg[:, 0, iP]).tolist()
    umb = ratio['end'][ofs['V_umb']:ofs['V_umb'] + nu]
    ref = ratio['end'][ofs['V_ref']:ofs['V_ref'] + nr]
    pu = perm_mean_diff(umb, ref, ctx['perm']) if nu and nr else {'diff': float('nan'), 'p': float('nan')}
    beta = slope_perm(cos, ratio['I1'][1:1 + nr + nu], ctx['perm'])
    summ['M2'] = {'G': G, 'L': L, 'N': Nn, 'K': K, 'K_split': n_split, 'G_t': Gt, 'p_umb_ref': pu['p'],
                  'umb_minus_ref': pu['diff'], 'beta': beta, 'n': {V: len(vs[V]['x']) for V in ofs}}
    summ['checks']['M2_eps_P'] = {I: float(r['m']['eps'][0]) for I, r in rv.items()}
    summ['checks']['M2_eps_V_median'] = {I: float(np.median(r['m']['eps'])) for I, r in rv.items()}
    summ['checks']['M2_eps_step_P'] = float(step['m']['eps'][0])
    arrays['M2_ratio'] = ratio
    arrays['M2_dm_end'] = de

  # M3 carriers and provenance
  if mode == 'taught' and 'M3' not in skip:
    blocks_ = [n for n, _ in model.blocks()]
    units = model.blocks()[0][1].pw_gate.out_features
    block_of = np.repeat(np.arange(len(blocks_)), units)
    m_w = ctx['make_model']().to(dev, dt).eval()
    m_w.load_state_dict(conv(th[0], dt))
    u_w = pw_down_inputs(m_w, xP)
    _, st_w = block_states(m_w, xP)
    fn = fns['m']
    res3 = {}
    for when in ('tau', 'end'):
      mt = ctx['make_model']().to(dev, dt).eval()
      mt.load_state_dict(conv(th[tau if when == 'tau' else end], dt))
      cdict = patch_carriers(mt, xP, u_w, fn)
      c = np.concatenate([cdict[b] for b in blocks_])
      _, st_t = block_states(mt, xP)
      ustar_P = np.concatenate([st_t[b]['u'][0].max(0).values.double().cpu().numpy() for b in blocks_])
      sel = np.full(len(c), np.nan)
      if vs is not None:
        us = unit_star(mt, Images([vs['V_ref']['x']], dev, dt), ctx['vbatch'])
        sig = np.sign(ustar_P)
        sig[sig == 0] = 1
        Q = np.quantile(sig[None] * us, ctx['q_sel'], axis=0)
        sel = np.where(Q > 0, sig * ustar_P / np.where(Q > 0, Q, 1), np.inf)
      res3[when] = {'c': c, 'C50': c50(c), 'ustar_P': ustar_P, 'sel': sel, 'st': st_t, 'model': mt}
    rt = res3['tau']
    C, c = rt['C50'], rt['c']
    bkI1 = load_buckets(run, 0, tau, keys, dtype=dt, device=dev, th=(th[0], th[tau]))
    sp = own_upstream_split(rt['st'], st_w, rt['model'], m_w, bkI1, keys, blocks_, ctx.get('e1'))
    share = np.abs(sp['own']) / (np.abs(sp['own']) + np.abs(sp['up']) + 1e-30)
    iP3 = bkI1['names'].index('P')
    pshare = sp['own_by'][iP3] / np.where(sp['own'] != 0, sp['own'], np.nan)
    M3 = {'n_units': len(c), 'C50': C.tolist(), 'n_C50': len(C), 'c_pos_total': float(np.maximum(c, 0).sum()),
          'own_share': {'median_C50': float(np.median(share[C])) if len(C) else float('nan')},
          'P_share_own': {'median_C50': float(np.nanmedian(pshare[C])) if len(C) else float('nan')},
          'sel': {'median_C50': float(np.median(rt['sel'][C])) if len(C) else float('nan')}}
    summ['checks']['own_upstream_resid'] = float(sp['resid'].max())
    if 'resid_pre' in sp:
      summ['checks']['two_level_resid'] = float(sp['resid_pre'].max())
    e1 = ctx.get('e1')
    if e1 is not None:
      Bp, Up = pole_sets(e1['atlas'], blocks_, t, s, ctx.get('poles', 20))
      Rt = np.concatenate([e1['atlas'][b]['R'][:, t].double().numpy() for b in blocks_])
      Rs = np.concatenate([e1['atlas'][b]['R'][:, s].double().numpy() for b in blocks_])
      pole = Bp | Up
      isC = np.zeros(len(c), bool)
      isC[C] = True
      q_t = class_percentile(sp['T'], sp['du'], t) if 'T' in sp else None
      q_s = class_percentile(sp['T'], sp['du'], s) if 'T' in sp else None
      M3['poles'] = {'n_B_pole': int(Bp.sum()), 'n_U_pole': int(Up.sum()), 'unit_fraction': float(pole.mean()),
                     'C50_in_pole': float(pole[C].mean()) if len(C) else float('nan'),
                     'route_share': float(np.maximum(c, 0)[pole].sum() / max(np.maximum(c, 0).sum(), 1e-30))}

      def matched(mask, q, key, cs):
        pairs = match_noncarriers(cs, mask, block_of, np.abs(key))
        d = [q[k] - np.mean(q[v]) for k, v in pairs.items()]
        return {'n': len(cs), 'median_q': float(np.median(q[cs])) if len(cs) else float('nan'),
                'median_q_matched': float(np.median([np.mean(q[v]) for v in pairs.values()])) if pairs else float('nan'),
                'test': signed_rank_test(d, ctx['perm'])}
      if q_t is not None:
        cB = C[Bp[C]]
        cU = C[Up[C] & (sp['du'][C] < 0)]
        rel_t = [k for k in ctx.get('rel_target', []) if k < sp['T'].shape[0]]
        M3['q'] = {'B_pole': matched(Bp, q_t, Rt, cB), 'U_pole_fading': matched(Up, q_s, Rs, cU)}
        if rel_t and len(cB):
          qrel = np.mean([class_percentile(sp['T'], sp['du'], k) for k in rel_t], 0)
          M3['q']['B_pole']['median_q_relatives'] = float(np.median(qrel[cB]))
        arrays['q_t'], arrays['q_s'] = q_t, q_s
      prov = ctx.get('prov')
      if prov is not None:
        pt = np.concatenate([prov['blocks'][b]['pi_value_t'] for b in blocks_])
        ps = np.concatenate([prov['blocks'][b]['pi_value_s'] for b in blocks_])
        M3['provenance'] = {'B_pole': matched(Bp, pt, Rt, C[Bp[C]]),
                            'U_pole_fading': matched(Up, ps, Rs, C[Up[C] & (sp['du'][C] < 0)])}
      re = res3['end']
      mass = lambda r, m: float(np.maximum(r['c'], 0)[m].sum())
      tot = lambda r: max(float(np.maximum(r['c'], 0).sum()), 1e-30)
      selm = lambda r: (r['sel'] >= 1.5) & ~pole
      M3['drift'] = {'pole_mass_tau': mass(rt, pole), 'pole_mass_end': mass(re, pole),
                     'pole_share_tau': mass(rt, pole) / tot(rt), 'pole_share_end': mass(re, pole) / tot(re),
                     'sel_nonpole_share_tau': mass(rt, selm(rt)) / tot(rt),
                     'sel_nonpole_share_end': mass(re, selm(re)) / tot(re), 'n_C50_end': len(re['C50'])}
      ratio_ctx = {}
      for when, ust in (('w', np.concatenate([st_w[b]['u'][0].max(0).values.double().cpu().numpy() for b in blocks_])),
                        ('tau', rt['ustar_P'])):
        cb, cu = C[Bp[C]], C[Up[C]]
        ratio_ctx[f'r_bee_{when}'] = float(np.median(ust[cb] / Rt[cb])) if len(cb) else float('nan')
        ratio_ctx[f'r_umb_{when}'] = float(np.median(ust[cu] / Rs[cu])) if len(cu) else float('nan')
      M3['figure_ratios'] = ratio_ctx
    arrays['M3'] = {'c_tau': rt['c'], 'c_end': res3['end']['c'], 'sel_tau': rt['sel'], 'sel_end': res3['end']['sel'],
                    'du': sp['du'], 'own': sp['own'], 'up': sp['up'], 'own_by': sp['own_by'], 'names': bkI1['names']}
    summ['M3'] = M3
  return summ, arrays


def cmd_run(a):
  from experiments.convnext_umbrella import photo_tensor
  dev = torch.device(a.device)
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  info = run_info(a.run)
  cfg = info['cfg']
  sel_path = a.sel or cfg['sel']
  sels = [json.load(open(sel_path))] + [json.load(open(f)) for f in a.sel_extra]
  arm = find_arm(sels, cfg['arm'])
  photo = cfg['photo']
  kind = 'control' if photo.split(':')[0] in ('noise', 'shuffle', 'shuffle_train') else 'photo'
  e1 = None
  if a.e1 and os.path.exists(os.path.join(a.e1, 'atlas_25020.pt')):
    lde = lambda f: torch.load(os.path.join(a.e1, f), map_location='cpu', weights_only=False)
    e1 = {'atlas': lde('atlas_25020.pt'), 'attr': lde('attr_25020.pt'), 'init': lde('attr_init.pt'),
          'state': lde('model_25020.pt')}
  ctx = {'make_model': lambda: build_convnext('tiny'), 'device': dev, 'dtype': getattr(torch, a.dtype),
         'vdtype': getattr(torch, a.vdtype), 'x_P': photo_tensor(photo, a.val, dev, a.train).cpu(),
         'target': cfg['target'] if cfg['target'] >= 0 else arm['target'],
         'source': arm['source'] if kind == 'photo' else None, 'kind': kind,
         'rel_target': arm['rel_target'], 'vsets': load_vsets(a.vsets) if a.vsets else None, 'e1': e1,
         'prov': torch.load(a.provenance, weights_only=False) if a.provenance else None,
         'nodes': a.nodes, 'vnodes': a.vnodes, 'step_nodes': a.step_nodes, 'tol': a.tol, 'max_nodes': a.max_nodes,
         'batch': a.batch, 'vbatch': a.vbatch, 'perm': a.perm, 'n_near': 100, 'n_mp': 10, 'q_sel': 0.999,
         'rows': ('P', 'BEE', 'UMB', 'OTH'), 'per_class': a.per_class}
  summ, arrays = analyze(a.run, ctx, a.untaught, skip=set(filter(None, a.skip.split(','))))
  summ['tag'] = a.tag
  summ['untaught'] = a.untaught
  out = a.out or os.path.join(a.run, 'analysis.json')
  torch.save(arrays, out[:-5] + '.pt')
  json.dump(_jsonable(summ), open(out + '.tmp', 'w'), indent=1)
  os.replace(out + '.tmp', out)
  print(json.dumps(_jsonable({k: summ[k] for k in ('tau', 'm_P', 'checks')})), flush=True)


# --------------------------------------------------------------------------------------------- decision rules

def _get(d, *path, default=float('nan')):
  for p in path:
    if not isinstance(d, dict) or p not in d or d[p] is None:
      return default
    d = d[p]
  return d


def rule_c1a(B: list, Bp: list, lgo: dict, n_photos: int = 5) -> dict:
  """P-C1a over primary runs B (dicts with seed, photo, M1) and standard-optimizer runs Bp; lgo: {(photo, seed):
  {'dLGO_BEE', ...}}."""
  per = []
  for r in B:
    did, z = _get(r, 'M1', 'feed', 'DiD_BEE_I2'), _get(r, 'M1', 'feed', 'z_BEE_I2')
    dA, dB = _get(r, 'M1', 'feed', 'DiD_A_I2'), _get(r, 'M1', 'feed', 'DiD_B_I2')
    ok = bool(did > 0 and z >= 3 and did > dA and did > dB)
    l = lgo.get((r['photo'], r['seed']), {})
    per.append({'photo': r['photo'], 'seed': r['seed'], 'DiD_BEE': did, 'z': z, 'DiD_A': dA, 'DiD_B': dB, 'O': ok,
                'null': bool(abs(z) < 2), 'dLGO_BEE': l.get('dLGO_BEE'),
                'lgo_sign_agrees': None if l.get('dLGO_BEE') is None else bool(np.sign(l['dLGO_BEE']) == np.sign(did))})
  seeds = sorted({p['seed'] for p in per})
  need = math.ceil(0.8 * n_photos)                       # 4 of 5
  O_seeds = [sd for sd in seeds if sum(p['O'] for p in per if p['seed'] == sd) >= need]
  null_seeds = [sd for sd in seeds if sum(p['null'] for p in per if p['seed'] == sd) >= math.ceil(0.6 * n_photos)]
  bp_pos = sum(_get(r, 'M1', 'feed', 'DiD_BEE_I2') > 0 for r in Bp)
  bp_ok = bool(len(Bp) and bp_pos >= need)
  verdict = 'O' if len(O_seeds) >= 2 and bp_ok else 'against' if len(null_seeds) >= 2 else 'mixed'
  return {'per_run': per, 'O_seeds': O_seeds, 'null_seeds': null_seeds, 'Bp_positive': int(bp_pos),
          'Bp_n': len(Bp), 'Bp_ok': bp_ok, 'verdict': verdict,
          'lgo_agreement': [p['lgo_sign_agrees'] for p in per if p['lgo_sign_agrees'] is not None]}


def rule_c1b(B: list, Fc: list, n_photos: int = 5) -> dict:
  """P-C1b: E_OTH of each photo against the seed-matched controls (those with z^BEE(I2) >= 3 excluded)."""
  per = []
  for r in B:
    ctl = [c for c in Fc if c['seed'] == r['seed']]
    ok_ctl = [c for c in ctl if not (_get(c, 'M1', 'feed', 'z_BEE_I2') >= 3)]
    e = _get(r, 'M1', 'shelter', 'E_OTH')
    ce = [_get(c, 'M1', 'shelter', 'E_OTH') for c in ok_ctl]
    per.append({'photo': r['photo'], 'seed': r['seed'], 'E_OTH': e, 'controls': ce,
                'rode_bee_route': [c['photo'] for c in ctl if c not in ok_ctl],
                'O': bool(ce and e < min(ce)), 'H': bool(ce and e >= float(np.median(ce)))})
  seeds = sorted({p['seed'] for p in per})
  O_seeds = [sd for sd in seeds if sum(p['O'] for p in per if p['seed'] == sd) >= math.ceil(0.8 * n_photos)]
  H_seeds = [sd for sd in seeds if sum(p['H'] for p in per if p['seed'] == sd) >= math.ceil(0.6 * n_photos)]
  verdict = 'O' if len(O_seeds) >= 2 else 'against' if len(H_seeds) >= 2 else 'mixed'
  return {'per_run': per, 'O_seeds': O_seeds, 'H_seeds': H_seeds, 'verdict': verdict}


def rule_c1c(B: list) -> dict:
  per = []
  for r in B:
    z = {I: _get(r, 'M1', 'umbrella', I, 'blocks', 'z') for I in ('I1', 'I2')}
    d = {I: _get(r, 'M1', 'umbrella', I, 'blocks', 'did') for I in ('I1', 'I2')}
    per.append({'photo': r['photo'], 'seed': r['seed'], 'z': z, 'DiD': d,
                'O_like': bool(max(abs(v) for v in z.values()) >= 3),
                'H_like': bool(max(abs(v) for v in z.values()) < 2),
                'sign_I2': float(np.sign(d['I2']))})
  return {'per_run': per, 'n_O_like': sum(p['O_like'] for p in per), 'n_H_like': sum(p['H_like'] for p in per),
          'n': len(per)}


def rule_c2(B: list, Fc: list, solo: list, n_photos: int = 5) -> dict:
  ctl0 = [c for c in Fc if c['seed'] == 0]
  Lc0 = [_get(c, 'M2', 'L') for c in ctl0]
  calibrated = sum(l >= 0.5 for l in Lc0) >= 2
  per = []
  for r in B:
    Lc = [_get(c, 'M2', 'L') for c in Fc if c['seed'] == r['seed']]
    L = _get(r, 'M2', 'L')
    so = [x for x in solo if x['photo'] == r['photo']]
    per.append({'photo': r['photo'], 'seed': r['seed'], 'L': L, 'L_controls': Lc,
                'below_controls': bool(Lc and L < min(Lc)),
                'within_controls': bool(Lc and min(Lc) <= L <= max(Lc)),
                'p_umb_ref': _get(r, 'M2', 'p_umb_ref'), 'L_solo': _get(so[0], 'M2', 'L') if so else None})
  s0 = [p for p in per if p['seed'] == 0]
  other = {sd: [p for p in per if p['seed'] == sd] for sd in sorted({p['seed'] for p in per}) if sd != 0}
  O = (calibrated and sum(p['below_controls'] for p in s0) >= math.ceil(0.8 * n_photos)
       and all(sum(p['below_controls'] for p in v) >= math.ceil(0.6 * n_photos) for v in other.values())
       and sum(p['p_umb_ref'] < 0.01 for p in s0) >= math.ceil(0.8 * n_photos))
  H = (calibrated and sum(p['within_controls'] for p in s0) >= math.ceil(0.6 * n_photos)
       and sum(not (p['p_umb_ref'] < 0.01) for p in s0) >= math.ceil(0.6 * n_photos))
  latch = [p['photo'] for p in s0 if p['L_solo'] is not None and p['L'] > p['L_solo']]
  verdict = 'not testable in tiny' if not calibrated else 'O' if O else 'H' if H else 'mixed'
  return {'calibrated': bool(calibrated), 'L_controls_seed0': Lc0, 'per_run': per,
          'replay_induced_latching': latch, 'verdict': verdict}


def rule_c3_run(r: dict) -> dict:
  M3 = r.get('M3', {})
  own = _get(M3, 'own_share', 'median_C50')
  sel = _get(M3, 'sel', 'median_C50')
  qB, qU = _get(M3, 'q', 'B_pole'), _get(M3, 'q', 'U_pole_fading')
  dr = M3.get('drift', {})
  okB = isinstance(qB, dict) and qB.get('median_q', float('nan')) >= 0.9 and _get(qB, 'test', 'p') < 0.01
  okU = (not isinstance(qU, dict)) or qU.get('n', 0) == 0 or (qU.get('median_q', 0) >= 0.9 and _get(qU, 'test', 'p') < 0.01)
  mass_ok = dr.get('pole_mass_end', 0) >= (2 / 3) * dr.get('pole_mass_tau', float('inf'))
  O = bool(own <= 0.5 and okB and okU and sel <= 1.0 and mass_ok)
  drift_sel = dr.get('sel_nonpole_share_end', 0) > dr.get('sel_nonpole_share_tau', 0) and dr.get('sel_nonpole_share_end', 0) >= 0.5
  Hp = bool(own >= 0.8 or sel >= 1.5 or drift_sel)
  H = bool(_get(M3, 'poles', 'C50_in_pole') < 0.5 and isinstance(qB, dict) and abs(qB.get('median_q', 0) - 0.5) < 0.1)
  return {'photo': r['photo'], 'seed': r['seed'], 'own_share': own, 'sel': sel, 'q_B': _get(qB, 'median_q'),
          'p_B': _get(qB, 'test', 'p'), 'pole_mass_ok': bool(mass_ok), 'O': O, "H'": Hp, 'H': H,
          'label': 'O' if O else "H'" if Hp else 'H' if H else 'mixed'}


def rule_c4_run(r: dict) -> dict:
  M4 = r.get('M4', {})
  t = r['target']
  S_late, S_head = M4.get('S_late', float('nan')), M4.get('S_head', float('nan'))
  st3 = _get(M4, 'swaps', 'stage3', 'P', 'argmax', default=-1) == t
  head = _get(M4, 'swaps', 'head', 'P', 'argmax', default=-1) == t
  return {'photo': r['photo'], 'seed': r['seed'], 'S_late': S_late, 'S_head': S_head, 'stage3_P_says_target': st3,
          'head_P_says_target': head, 'O': bool(S_late >= 0.8 and st3 and not head), 'head_deviation': bool(S_head >= 0.5)}


def decide(recs: list, lgo: list) -> dict:
  by = lambda tag: [r for r in recs if r.get('tag') == tag]
  B, Bp, Fc, C, E, J = by('B'), by('Bp'), by('F'), by('C'), by('E'), by('J')
  B = [r for r in B if r.get('kind') == 'photo']
  lg = {}
  for x in lgo:
    lg.setdefault((x['photo'], x['seed']), {})[x['variant']] = x['m_P_end']
  lgd = {k: {'dLGO_BEE': v['target'] - v['base'], 'dLGO_UMB': v['source'] - v['base'], 'dLGO_A': v['A'] - v['base']}
         for k, v in lg.items() if all(q in v for q in ('base', 'target', 'source', 'A'))}
  c1a, c1b = rule_c1a(B, Bp, lgd), rule_c1b(B, Fc)
  umb_lgo = [bool(np.sign(lgd[(r['photo'], r['seed'])]['dLGO_UMB']) == np.sign(_get(r, 'M1', 'DiD', 'I2', 'UMB', 'blocks')))
             for r in B if (r['photo'], r['seed']) in lgd]
  c1 = 'supported' if c1a['verdict'] == c1b['verdict'] == 'O' else \
       'contradicted' if c1a['verdict'] == c1b['verdict'] == 'against' else 'mixed'
  solo = [r for r in C if r.get('opt') == 'frozen_v']
  c3 = [rule_c3_run(r) for r in B]
  c4 = [rule_c4_run(r) for r in B]
  out = {'P-C1a': c1a, 'P-C1b': c1b, 'P-C1c': rule_c1c(B), 'lgo_UMB_sign_agreement': umb_lgo,
         'P-C2': rule_c2(B, Fc, solo), 'P-C3': {'per_run': c3, 'counts': {k: sum(x['label'] == k for x in c3)
                                                                       for k in ('O', "H'", 'H', 'mixed')}},
         'P-C4': {'per_run': c4, 'n_O': sum(x['O'] for x in c4), 'n_head_deviation': sum(x['head_deviation'] for x in c4)},
         'C1': c1, 'n': {'B': len(B), 'Bp': len(Bp), 'F': len(Fc), 'C': len(C), 'E': len(E), 'J': len(J),
                         'lgo': len(lgo)}}
  if E:                                                  # arm E: per triple against the seed-0 controls of arm B
    per = []
    for r in E:
      z = _get(r, 'M1', 'feed', 'z_BEE_I2')
      did = _get(r, 'M1', 'feed', 'DiD_BEE_I2')
      ce = [_get(c, 'M1', 'shelter', 'E_OTH') for c in Fc if c['seed'] == 0]
      Lc = [_get(c, 'M2', 'L') for c in Fc if c['seed'] == 0]
      per.append({'arm': r['arm'], 'C1a_O': bool(did > 0 and z >= 3 and did > _get(r, 'M1', 'feed', 'DiD_A_I2')
                                                and did > _get(r, 'M1', 'feed', 'DiD_B_I2')),
                  'C1b_O_cross_stream': bool(ce and _get(r, 'M1', 'shelter', 'E_OTH') < min(ce)),
                  'C2_O_cross_stream': bool(Lc and _get(r, 'M2', 'L') < min(Lc)), 'z_BEE_I2': z})
    out['E'] = {'per_triple': per, 'n_C1a_O': sum(p['C1a_O'] for p in per),
                'n_C1b_O': sum(p['C1b_O_cross_stream'] for p in per), 'n_C2_O': sum(p['C2_O_cross_stream'] for p in per)}
  if J:
    out['J'] = {'DiD_BEE_I2_positive': sum(_get(r, 'M1', 'feed', 'DiD_BEE_I2') > 0 for r in J if r.get('kind') == 'photo'),
                'n': sum(r.get('kind') == 'photo' for r in J)}
  for tag in ('S2',):                                    # secondary weak-route arm, seed 0: descriptive
    S2 = [r for r in by(tag) if r.get('kind') == 'photo']
    if S2:
      out[tag] = {'P-C1a': rule_c1a(S2, [], {}), 'P-C1b': rule_c1b(S2, by(tag + 'F')),
                  'P-C2': rule_c2(S2, by(tag + 'F'), [])}
  return out


def cmd_decide(a):
  recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.root, '**', 'analysis.json'), recursive=True))]
  lgo = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(a.root, '**', 'lgo_result.json'), recursive=True))]
  out = decide(recs, lgo)
  json.dump(_jsonable(out), open(a.out + '.tmp', 'w'), indent=1)
  os.replace(a.out + '.tmp', a.out)
  print(json.dumps(_jsonable({k: out[k] for k in ('C1', 'n')}) | {k: out[k]['verdict'] for k in ('P-C1a', 'P-C1b', 'P-C2')}),
        flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  q = sub.add_parser('vsets', help='evaluation sets of one selection arm')
  q.add_argument('--sel', action='append', required=True, help='selection files (all candidates are excluded)')
  q.add_argument('--arm', required=True)
  q.add_argument('--val', required=True)
  q.add_argument('--first10', default='/dev/shm/probe_10.pt')
  q.add_argument('--out', required=True)
  q = sub.add_parser('provenance', help="E1 writers' provenance pi for an arm's target and source")
  q.add_argument('--sel', action='append', required=True)
  q.add_argument('--arm', required=True)
  q.add_argument('--e1', required=True)
  q.add_argument('--val', required=True)
  q.add_argument('--probe_cache', default='/dev/shm/probe_8.pt')
  q.add_argument('--device', default=dev)
  q.add_argument('--out', required=True)
  q = sub.add_parser('run', help='M1-M4 of one taught or solo run')
  q.add_argument('--run', required=True)
  q.add_argument('--untaught', default=None, help='the paired untaught run (taught runs; M1)')
  q.add_argument('--tag', required=True, help='arm set: B, Bp, C, F, Fsolo, E, J, S2, S2F')
  q.add_argument('--val', required=True)
  q.add_argument('--train', default='/dev/shm/imagenet256')
  q.add_argument('--sel', default=None, help='selection file (default: the run config)')
  q.add_argument('--sel_extra', action='append', default=[])
  q.add_argument('--vsets', default=None)
  q.add_argument('--e1', default=None, help='E1 run directory (atlas_25020, attr_25020, attr_init, model_25020)')
  q.add_argument('--provenance', default=None)
  q.add_argument('--device', default=dev)
  q.add_argument('--dtype', default='float64')
  q.add_argument('--vdtype', default='float32')
  q.add_argument('--nodes', type=int, default=8)
  q.add_argument('--vnodes', type=int, default=32)
  q.add_argument('--step_nodes', type=int, default=4)
  q.add_argument('--max_nodes', type=int, default=64)
  q.add_argument('--tol', type=float, default=1e-3)
  q.add_argument('--batch', type=int, default=16)
  q.add_argument('--vbatch', type=int, default=128)
  q.add_argument('--perm', type=int, default=10000)
  q.add_argument('--skip', default='', help='comma-separated: M1, M2, M3, M4')
  q.add_argument('--per_class', action='store_true', help='also the per-class writes over I1 and I2 (descriptive)')
  q.add_argument('--out', default=None)
  q = sub.add_parser('decide', help='pre-registered decision rules over all analyses under a root')
  q.add_argument('--root', required=True)
  q.add_argument('--out', required=True)
  a = p.parse_args()
  {'vsets': cmd_vsets, 'provenance': cmd_provenance, 'run': cmd_run, 'decide': cmd_decide}[a.cmd](a)


if __name__ == '__main__':
  main()
