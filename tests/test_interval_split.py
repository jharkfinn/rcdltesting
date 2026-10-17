"""rcdl.interval_split: the interval split adds up exactly (random fp64 data, and a toy SwiGLU network trained with
GroupAttribution, where it equals the actual change of every unit's response); the class-mean route equals the
per-image route; fp16 storage bounds hold; the class split and a shadow split have the same total; remainders of full
states are computed, not assumed 0; the momentum carry-over estimate is exact when it should be."""

import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.attribution import GroupAttribution
from rcdl.interval_split import (MAPS, ClassSums, as_unit_major, class_weighted_inputs, column_terms, dsilu,
                                 gl_nodes, half_ulp, image_split, increments, load_parts, lr_schedule,
                                 momentum_carryover, own_terms, path_weights, set_terms, split_completeness)

DT = torch.float64


def test_path_weights_integrate_the_response_change():
  """u(1) - u(0) = l_g Q0_g + 2 q_g Q1_g + l_v Q0_v + 2 q_v Q1_v along quadratic pre-activation paths."""
  g = torch.Generator().manual_seed(0)
  g1, gl, gq, v1, vl, vq = (torch.randn(50, generator=g, dtype=DT) * s for s in (2, 3, 1, 1, 2, 1))
  Q0g, Q1g, Q0v, Q1v = path_weights(g1, gl, gq, v1, vl, vq, nodes=32)
  u = lambda gg, vv: F.silu(gg) * vv
  du = u(g1 + gl + gq, v1 + vl + vq) - u(g1, v1)
  assert (gl * Q0g + 2 * gq * Q1g + vl * Q0v + 2 * vq * Q1v - du).abs().max() < 1e-12
  x, w = gl_nodes(8)
  assert abs(float((w * x ** 15).sum()) - 1 / 16) < 1e-15 and abs(float(w.sum()) - 1) < 1e-15
  t = torch.linspace(-6, 6, 101, dtype=DT, requires_grad=True)
  (grad,) = torch.autograd.grad(F.silu(t).sum(), t)
  assert (grad - dsilu(t.detach())).abs().max() < 1e-14


@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16, torch.float32])
def test_half_ulp_bounds_round_to_nearest(dtype):
  g = torch.Generator().manual_seed(1)
  x = torch.cat([torch.randn(4000, generator=g, dtype=DT) * 10.0 ** torch.randint(-9, 4, (4000,), generator=g),
                 torch.tensor([0.0, 2.0 ** -14, 2.0 ** -15, 2.0 ** -24, 1.0, 1.0 - 1e-12, 2.0 ** 10], dtype=DT)])
  if dtype == torch.float16:
    x = x.clamp(-6e4, 6e4)
  r = x.to(dtype)
  hu = half_ulp(r)
  err = (x - r.double()).abs()
  assert (err <= hu).all()
  frac, emin = {torch.float16: (10, -24), torch.bfloat16: (7, -133), torch.float32: (23, -149)}[dtype]
  assert (hu <= 2.0 ** -(frac + 1) * r.double().abs() + 2.0 ** (emin - 1)).all()    # and the bound is not loose


def random_parts(K, U, d, a, ab, scale, seed, remainder=1e-4):
  g = torch.Generator().manual_seed(seed)
  r = lambda *s: torch.randn(*s, generator=g, dtype=DT)
  return dict(a=a, ab=ab, C=r(K, U, d) * scale, Cb=r(K, U) * scale, R=r(U, d) * remainder, Rb=r(U) * remainder)


def make_parts(rec, W0, b0):
  from rcdl.interval_split import Parts
  return Parts(a=rec['a'], ab=rec['ab'], W0=W0, b0=b0, C=rec['C'], Cb=rec['Cb'], R=rec['R'], Rb=rec['Rb'])


def per_image_group_terms(inc, h1, h2, rec):
  """T_k(x) and F_k(x) for every group, image and unit (K, N, U), straight from the definitions."""
  Dh = h2 - h1
  T = F_ = 0
  for M in MAPS:
    I = inc[M]
    Q0, Q1 = rec[f'Q0_{M}'], rec[f'Q1_{M}']
    T = T + (torch.einsum('kud,nud->knu', I.nu, h1) + I.nub[:, None]) * Q0 + torch.einsum('kud,nud->knu', I.nu, Dh) * Q1
    dec = torch.einsum('kud,nud->knu', I.C1, h1) * Q0 + torch.einsum('kud,nud->knu', I.C1, Dh) * Q1
    F_ = F_ + (I.rho - 1) * dec + (I.rhob - 1) * I.Cb1[:, None] * Q0
  return T, F_


def test_completeness_on_random_fp64_data():
  """Random parts at two grid points (weight and bias decay, remainders), inputs at the reference and at other
  'strongest' positions: per image and per class, sum_k T_k + sum_k F_k + F_0 + E + Phi + Pi = dR, and the class-mean
  route (Hbar, qbar) gives the class means of the per-image terms."""
  K, U, d, N, C = 12, 5, 7, 60, 6
  g = torch.Generator().manual_seed(2)
  inc = {}
  for i, M in enumerate(MAPS):
    W0, b0 = torch.randn(U, d, generator=g, dtype=DT) * 0.3, torch.randn(U, generator=g, dtype=DT) * 0.1
    p1 = make_parts(random_parts(K, U, d, 0.9, 0.97, 0.05, 10 + i), W0, b0)
    p2r = random_parts(K, U, d, 0.8, 0.95, 0.02, 20 + i)
    p2r['C'] = (0.8 / 0.9) * p1.C + p2r['C']
    p2r['Cb'] = (0.95 / 0.97) * p1.Cb + p2r['Cb']
    inc[M] = increments(p1, make_parts(p2r, W0, b0))
    assert inc[M].resid < 1e-14
  h1 = torch.randn(N, U, d, generator=g, dtype=DT)
  h2 = h1 + 0.5 * torch.randn(N, U, d, generator=g, dtype=DT)
  y = torch.arange(N) % C
  # the 'strongest' responses come from other positions: Pi carries the switch
  hb1, hb2 = torch.randn(N, U, d, generator=g, dtype=DT), torch.randn(N, U, d, generator=g, dtype=DT)
  resp = lambda h, W, b: (F.silu((h * W['gate']).sum(-1) + b['gate']) * ((h * W['value']).sum(-1) + b['value']))
  u1b = resp(hb1, {M: inc[M].W1 for M in MAPS}, {M: inc[M].b1 for M in MAPS})
  u2b = resp(hb2, {M: inc[M].W2 for M in MAPS}, {M: inc[M].b2 for M in MAPS})
  rec = image_split(h1, h2, inc, u1b, u2b, nodes=32)
  T, Fk = per_image_group_terms(inc, h1, h2, rec)
  Dh = h2 - h1
  F0 = E = 0
  for M in MAPS:
    I, Q0, Q1 = inc[M], rec[f'Q0_{M}'], rec[f'Q1_{M}']
    F0 = F0 + ((h1 * I.F0w).sum(-1) + I.F0b) * Q0 + (Dh * I.F0w).sum(-1) * Q1
    E = E + ((h1 * I.eps).sum(-1) + I.epsb) * Q0 + (Dh * I.eps).sum(-1) * Q1
  total = T.sum(0) + Fk.sum(0) + F0 + E + rec['Phi'] + rec['Pi']
  assert (total - (u2b - u1b)).abs().max() < 1e-12
  assert rec['resid'].abs().max() < 1e-12
  cm = as_unit_major(class_weighted_inputs(h1, h2, inc, y, C, u1b, u2b))
  sc = split_completeness(inc, cm)
  assert sc['max_abs'] < 1e-12 and sc['weights_identity'] < 1e-12
  for n in range(U):
    ct = column_terms(inc, {M: cm[f'Hbar_{M}'][n] for M in MAPS}, {M: cm[f'qbar_{M}'][n] for M in MAPS}, n)
    Tc = torch.stack([T[:, y == c, n].mean(1) for c in range(C)], 1)
    Fc = torch.stack([Fk[:, y == c, n].mean(1) for c in range(C)], 1)
    assert (ct['T'] - Tc).abs().max() < 1e-13 and (ct['F'] - Fc).abs().max() < 1e-13
    tot = ct['T'].sum(0) + ct['F'].sum(0) + ct['F0'] + ct['E'] + cm['Phi'][n] + cm['Pi'][n]
    assert (tot - cm['dR'][n]).abs().max() < 1e-12
    assert (ct['W'] - (ct['T'].sum(0) + ct['F'].sum(0) + ct['F0'] + ct['E'])).abs().max() < 1e-12
    # the cross term moves wholly to the weights or the input: still exact
    cx = column_terms(inc, {M: cm[f'Hcross_{M}'][n] for M in MAPS}, {}, n)
    for s in (1, -1):
      tot_s = (ct['W'] + s * cx['W']) + (cm['Phi'][n] - s * cm['cross'][n]) + cm['Pi'][n]
      assert (tot_s - cm['dR'][n]).abs().max() < 1e-12
  assert (cx['W'] - cm['cross'][n]).abs().max() < 1e-13
  # own and set terms (K = C groups for this check)
  inc_c = {M: copy.copy(inc[M]) for M in MAPS}
  for M in MAPS:
    inc_c[M].nu, inc_c[M].nub = inc[M].nu[:C], inc[M].nub[:C]
    inc_c[M].C1, inc_c[M].Cb1 = inc[M].C1[:C], inc[M].Cb1[:C]
  own = own_terms(inc_c, cm)
  mem = (torch.rand(C, C, generator=g) < 0.4).to(DT)
  st, sf = set_terms(inc_c, cm, mem, 'T'), set_terms(inc_c, cm, mem, 'F')
  for n in range(U):
    ct = column_terms(inc_c, {M: cm[f'Hbar_{M}'][n] for M in MAPS}, {M: cm[f'qbar_{M}'][n] for M in MAPS}, n)
    assert (own[n] - ct['T'].diagonal()).abs().max() < 1e-13
    assert (st[n] - (mem * ct['T'].T).sum(1)).abs().max() < 1e-13
    assert (sf[n] - (mem * ct['F'].T).sum(1)).abs().max() < 1e-13


# ---------------------------------------------------------------------------------- toy SwiGLU trained with RCDL

class ToyBlock(nn.Module):
  def __init__(self, d, U):
    super().__init__()
    self.norm = nn.LayerNorm(d)
    self.pw_gate, self.pw_value, self.pw_down = nn.Linear(d, U), nn.Linear(d, U), nn.Linear(U, d)

  def forward(self, x):
    h = self.norm(x)
    return x + self.pw_down(F.silu(self.pw_gate(h)) * self.pw_value(h))


class Toy(nn.Module):
  """Two SwiGLU blocks on sequences (B, P, d_in): positions play the role of a ConvNeXt unit's spatial positions."""

  def __init__(self, d_in=6, d=5, U=4, K=5):
    super().__init__()
    self.emb = nn.Linear(d_in, d)
    self.blocks = nn.ModuleList([ToyBlock(d, U), ToyBlock(d, U)])
    self.head = nn.Linear(d, K)

  def forward(self, x, keep=None):
    x = self.emb(x)
    for i, b in enumerate(self.blocks):
      if keep is not None:
        keep[i] = b.norm(x)
      x = b(x)
    return self.head(x.mean(1))


def train_toy(steps=(3, 7, 10), K=5, Kr=4, wd=0.1, bias_wd=0.05, seed=0):
  """Trains the toy for max(steps) AdamW steps with the class split and a random shadow split; at each saved step
  returns the model weights, both splits' full states (fp64) and the class split's compact (fp16) state."""
  torch.manual_seed(seed)
  model = Toy(K=K).double()
  layers = {f'b{i}.{m}': getattr(b, f'pw_{m}') for i, b in enumerate(model.blocks) for m in MAPS}
  kw = dict(optimizer='adamw', lr=3e-2, betas=(0.9, 0.99), weight_decay=wd, bias_weight_decay=bias_wd, dtype=DT)
  attr = GroupAttribution(layers, n_groups=K, **kw)
  shadow = GroupAttribution(layers, n_groups=Kr, shadow=True, **kw)
  init = {k: {'W0': tr.W0.clone(), 'b0': tr.b0.clone()} for k, tr in attr.tracked.items()}
  rest = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.AdamW(rest, lr=3e-2, weight_decay=wd)
  g = torch.Generator().manual_seed(seed + 1)
  saved = {}
  for t in range(1, max(steps) + 1):
    x = torch.randn(16, 6, 6, generator=g, dtype=DT)
    y = torch.randint(0, K, (16,), generator=g)
    r = torch.randint(0, Kr, (16,), generator=g)
    opt.zero_grad()
    F.cross_entropy(model(x), y).backward()
    attr.step(y)
    shadow.step(r)
    opt.step()
    if t in steps:
      saved[t] = {'model': copy.deepcopy(model.state_dict()), 'full': copy.deepcopy(attr.full_state()),
                  'shadow': copy.deepcopy(shadow.full_state()), 'compact': copy.deepcopy(attr.compact_state())}
  return model, init, saved


@torch.no_grad()
def probe_inputs(model, sd, x, b):
  """Block b's input h (N, P, d) and responses u (N, P, U) under the weights sd."""
  m = copy.deepcopy(model)
  m.load_state_dict(sd)
  keep = {}
  m(x, keep)
  blk = m.blocks[b]
  h = keep[b]
  return h, F.silu(blk.pw_gate(h)) * blk.pw_value(h)


def toy_interval(model, init, saved, t1, t2, b=1, which='full', N=30, C=5, seed=3):
  g = torch.Generator().manual_seed(seed)
  x = torch.randn(N, 6, 6, generator=g, dtype=DT)
  y = torch.arange(N) % C
  h1a, u1a = probe_inputs(model, saved[t1]['model'], x, b)
  h2a, u2a = probe_inputs(model, saved[t2]['model'], x, b)
  p1, p2 = u1a.argmax(1), u2a.argmax(1)                         # (N, U) strongest positions
  take = lambda h, p: torch.gather(h, 1, p[:, :, None].expand(-1, -1, h.shape[-1]))
  u1best, u2best = torch.gather(u1a, 1, p1[:, None])[:, 0], torch.gather(u2a, 1, p2[:, None])[:, 0]
  inc = {}
  for M in MAPS:
    key = f'b{b}.{M}'
    parts = []
    for t in (t1, t2):
      sd = saved[t]['model']
      W, bb = sd[f'blocks.{b}.pw_{M}.weight'], sd[f'blocks.{b}.pw_{M}.bias']
      rec = saved[t][which][key]
      parts.append(load_parts(rec, init[key], W=W, b=bb))
    inc[M] = increments(*parts)
  return dict(x=x, y=y, h1=take(h1a, p1), h2=take(h2a, p1), u1best=u1best, u2best=u2best, inc=inc,
              actual=u2best - u1best)


@pytest.mark.parametrize('which', ['full', 'compact'])
def test_toy_split_equals_actual_change_of_every_unit(which):
  """The interval split of a GroupAttribution-trained toy (AdamW with weight and bias decay) equals the actual change
  of each unit's response at its strongest position, per image and per class, from fp64 parts and from fp16 storage
  (whose rounding the fp64 remainder absorbs)."""
  model, init, saved = train_toy()
  d = toy_interval(model, init, saved, 3, 10, which=which)
  rec = image_split(d['h1'], d['h2'], d['inc'], d['u1best'], d['u2best'])
  T, Fk = per_image_group_terms(d['inc'], d['h1'], d['h2'], rec)
  Dh = d['h2'] - d['h1']
  rest = 0
  for M in MAPS:
    I, Q0, Q1 = d['inc'][M], rec[f'Q0_{M}'], rec[f'Q1_{M}']
    rest = rest + ((d['h1'] * (I.F0w + I.eps)).sum(-1) + I.F0b + I.epsb) * Q0 + (Dh * (I.F0w + I.eps)).sum(-1) * Q1
  total = T.sum(0) + Fk.sum(0) + rest + rec['Phi'] + rec['Pi']
  assert (total - d['actual']).abs().max() < 1e-12 * max(1.0, float(d['actual'].abs().max()))
  assert (rec['Pi'] >= -1e-14).all()                             # reference p*_t1: the switch only adds
  assert float(rec['Pi'].max()) > 1e-3 and float(rec['Phi'].abs().max()) > 1e-3    # both are exercised
  assert (rec['u1'] - d['u1best']).abs().max() < 1e-13           # the reference response at t1 is the strongest one
  cm = as_unit_major(class_weighted_inputs(d['h1'], d['h2'], d['inc'], d['y'], 5, d['u1best'], d['u2best']))
  sc = split_completeness(d['inc'], cm)
  assert sc['max_abs'] < 1e-12
  dR = torch.stack([d['actual'][d['y'] == c].mean(0) for c in range(5)], 1)    # (U, C)
  assert (cm['dR'] - dR).abs().max() < 1e-14
  for M in MAPS:
    I = d['inc'][M]
    assert I.rho < 1 and I.rhob < 1                              # weight and bias decay both exercised
    if which == 'compact':
      assert float(I.eps.abs().max()) > 0                        # fp16 rounding sits in the remainder


def test_fp16_bounds_hold_and_shadow_total_equals_class_total():
  """Group terms from the fp16 (compact) parts are within the propagated half-ulp bounds of the fp64 parts' terms; the
  random shadow split adds up to the same total as the class split, per class and unit."""
  model, init, saved = train_toy()
  dfull = toy_interval(model, init, saved, 3, 10, which='full')
  dcomp = toy_interval(model, init, saved, 3, 10, which='compact')
  cm = as_unit_major(class_weighted_inputs(dfull['h1'], dfull['h2'], dfull['inc'], dfull['y'], 5, dfull['u1best'],
                                           dfull['u2best']))
  # the shadow split of the same interval
  inc_r = {}
  for M in MAPS:
    key = f'b1.{M}'
    parts = [load_parts(saved[t]['shadow'][key], W=saved[t]['model'][f'blocks.1.pw_{M}.weight'],
                        b=saved[t]['model'][f'blocks.1.pw_{M}.bias']) for t in (3, 10)]
    inc_r[M] = increments(*parts)
  for n in range(cm['dR'].shape[0]):
    H = {M: cm[f'Hbar_{M}'][n] for M in MAPS}
    q = {M: cm[f'qbar_{M}'][n] for M in MAPS}
    a = column_terms(dfull['inc'], H, q, n)
    b = column_terms(dcomp['inc'], H, q, n, bounds=True)
    r = column_terms(inc_r, H, q, n)
    assert ((b['T'] - a['T']).abs() <= b['T_bound'] + 1e-15).all()
    assert ((b['F'] - a['F']).abs() <= b['F_bound'] + 1e-15).all()
    assert float(b['T_bound'].max()) > 0
    tot = lambda s: s['T'].sum(0) + s['F'].sum(0) + s['F0'] + s['E']
    assert (tot(a) - tot(r)).abs().max() < 1e-12
    assert (tot(a) - tot(b)).abs().max() < 1e-12
    assert (a['T'].sum(0) - r['T'].sum(0)).abs().max() > 0       # the splits differ, only their totals agree


def test_sub_intervals_telescope_and_path_dependence_is_reported():
  """Splitting (t1, t2] at an inner grid point gives the same total change (dR telescopes, every split is complete);
  the per-group allocations may differ (path dependence), which is what the pipeline's G+ check reports."""
  model, init, saved = train_toy()
  direct = toy_interval(model, init, saved, 3, 10)
  first, second = toy_interval(model, init, saved, 3, 7), toy_interval(model, init, saved, 7, 10)
  totals = []
  for d in (direct, first, second):
    cm = as_unit_major(class_weighted_inputs(d['h1'], d['h2'], d['inc'], d['y'], 5, d['u1best'], d['u2best']))
    assert split_completeness(d['inc'], cm)['max_abs'] < 1e-12
    totals.append(cm['dR'])
  assert (totals[1] + totals[2] - totals[0]).abs().max() < 1e-13


def test_full_state_remainder_is_computed_not_assumed():
  """A full fp32-style state whose parts do not add up to the weights (as E1's snapshots, up to 8e-5): R is computed
  from W in fp64 and the split stays exact; without W there is no remainder to use."""
  model, init, saved = train_toy()
  st = copy.deepcopy(saved)
  for t in (3, 10):
    for M in MAPS:
      rec = st[t]['full'][f'b1.{M}']
      rec['C'] = rec['C'] + 1e-5 * torch.randn_like(rec['C'])      # parts off by rounding-sized amounts
  d = toy_interval(model, init, st, 3, 10, which='full')
  for M in MAPS:
    assert float(d['inc'][M].eps.abs().max()) > 1e-6
  rec = image_split(d['h1'], d['h2'], d['inc'], d['u1best'], d['u2best'])
  cm = as_unit_major(class_weighted_inputs(d['h1'], d['h2'], d['inc'], d['y'], 5, d['u1best'], d['u2best']))
  assert split_completeness(d['inc'], cm)['max_abs'] < 1e-12
  assert (rec['resid']).abs().max() < 1e-12
  with pytest.raises(ValueError):
    load_parts(st[3]['full']['b1.gate'])


def test_momentum_carryover_is_the_whole_write_of_a_silent_group():
  """With the second moment frozen, no decay and a constant learning rate, a group that gets no examples after t1
  writes exactly what its momentum at t1 carries: nu_k = momentum_carryover(m_t1^(k), ...)."""
  torch.manual_seed(0)
  lin = nn.Linear(4, 3).double()
  attr = GroupAttribution({'l': lin}, n_groups=3, optimizer='adamw', lr=1e-2, betas=(0.9, 0.999), dtype=DT)
  g = torch.Generator().manual_seed(1)
  for t in range(6):
    x = torch.randn(8, 4, generator=g, dtype=DT)
    (lin(x) ** 2).sum().backward()
    attr.step(torch.randint(0, 3, (8,), generator=g))
  s1 = copy.deepcopy(attr.full_state()['l'])
  attr.freeze_second_moment()
  for t in range(20):
    x = torch.randn(8, 4, generator=g, dtype=DT)
    (lin(x) ** 2).sum().backward()
    attr.step(torch.randint(1, 3, (8,), generator=g))              # group 0 is silent after t1
  s2 = attr.full_state()['l']
  p1, p2 = load_parts(s1, W=s1['C'].sum(0) + s1['W0'], b=s1['Cb'].sum(0) + s1['b0']), \
           load_parts(s2, W=s2['C'].sum(0) + s2['W0'], b=s2['Cb'].sum(0) + s2['b0'])
  nu = increments(p1, p2).nu[0]
  est = momentum_carryover(s1['m'][0], s1['v'], s1['t'], s1['t'] + 20, lambda s: 1e-2, horizon=1000)
  assert (nu - est).abs().max() < 1e-14 * max(1.0, float(nu.abs().max())) + 1e-16
  assert float(nu.abs().max()) > 0


def test_lr_schedule_is_the_trainers():
  import math
  cfg = {'lr': 2e-3, 'warmup': 1250, 'steps': 25020, 'opt': 'adamw'}
  f = lr_schedule(cfg)
  assert f(0) == 2e-3 / 1250 and abs(f(1250) - 2e-3) < 1e-15
  assert abs(f(13135) - 2e-3 * 0.5 * (1 + math.cos(math.pi * 11885 / 23770))) < 1e-15


def test_class_sums_accumulate_like_one_batch():
  K, U, d, N, C = 6, 3, 4, 40, 5
  g = torch.Generator().manual_seed(4)
  inc = {}
  for i, M in enumerate(MAPS):
    W0, b0 = torch.randn(U, d, generator=g, dtype=DT), torch.randn(U, generator=g, dtype=DT)
    p1 = make_parts(random_parts(K, U, d, 0.9, 1.0, 0.05, 30 + i), W0, b0)
    p2 = make_parts(random_parts(K, U, d, 0.85, 1.0, 0.05, 40 + i), W0, b0)
    inc[M] = increments(p1, p2)
  h1 = torch.randn(N, U, d, generator=g, dtype=DT)
  h2 = h1 + torch.randn(N, U, d, generator=g, dtype=DT)
  y = torch.randint(0, C, (N,), generator=g)
  one = class_weighted_inputs(h1, h2, inc, y, C)
  acc = ClassSums(C, U, d)
  for i in range(0, N, 7):
    rec = image_split(h1[i:i + 7], h2[i:i + 7], inc)
    acc.add(y[i:i + 7], rec, rec['u1'], rec['u2'])
  two = acc.means()
  for k in ('Hbar_gate', 'Hcross_value', 'qbar_value', 'Phi', 'dR', 'R1'):
    assert (one[k] - two[k]).abs().max() < 1e-13
  assert float(one['Pi'].abs().max()) == 0.0


# ------------------------------------------------------------------------- the exits pipeline on a toy ConvNeXt run

TOY_GRID = (0, 5, 10, 20, 30, 40, 50, 60)
TOY_SNAPS = (10, 25, 40, 55)                    # 10 and 40 are also checkpoints (integrity check I2)
TOY_BLOCKS = ('stage0.block2', 'stage2.block4', 'stage3.block2')
TOY_PARAMS = dict(P=3, best=2, gone=5, est_step=10, anchor_k=2, anchor_min=1, n_boot=40, n_perm=10, n_rand=40,
                  x5_min_len=5, d_window=15, d_min_step=10, visual_near=2, unrel_wn=6, unrel_vis=3)


def build_toy_run(root, C=10, per_train=16, per_probe=4, size=32, steps=60, seed=0):
  """A run directory in the layout of experiments/convnext_train.py and convnext_snapshots --full: tiny ConvNeXt on
  32 px class-template images, AdamW with weight decay, the class split of three blocks' gate and value maps and a
  random shadow split, compact checkpoints, history snapshots with fp32 full states (nonzero remainders)."""
  import json
  import numpy as np
  from rcdl.convnext import build_convnext
  hist, full = root / 'history', root / 'full'
  hist.mkdir()
  full.mkdir()
  torch.manual_seed(seed)
  g = torch.Generator().manual_seed(seed + 1)
  templ = torch.rand(C, 3, size, size, generator=g)
  def images(n_per, gen):
    y = torch.arange(C).repeat_interleave(n_per)
    x = 0.6 * templ[y] + 0.4 * torch.rand(len(y), 3, size, size, generator=gen)
    return (x * 255).round().clamp(0, 255).to(torch.uint8), y
  xtr, ytr = images(per_train, g)
  xpr, ypr = images(per_probe, torch.Generator().manual_seed(seed + 2))
  torch.save({'x': xpr, 'y': ypr}, root / 'probe.pt')
  rand = ytr[torch.randperm(len(ytr), generator=g)]
  np.save(root / 'rand_groups.npy', rand.numpy().astype(np.int16))
  model = build_convnext('tiny', n_classes=C)
  named = dict(model.blocks())
  layers = {f'{n}.{k}': getattr(named[n], k) for n in TOY_BLOCKS for k in ('pw_gate', 'pw_value')}
  kw = dict(n_groups=C, optimizer='adamw', lr=4e-3, weight_decay=0.05, bias_weight_decay=0.0)
  attr = GroupAttribution(layers, **kw)
  attr_r = GroupAttribution(layers, shadow=True, **kw)
  torch.save({k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in attr.tracked.items()}, root / 'attr_init.pt')
  opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=4e-3, weight_decay=0.05)
  json.dump({'size': 'tiny', 'opt': 'adamw', 'lr': 4e-3, 'muon_lr': 2e-2, 'wd': 0.05, 'steps': steps, 'warmup': 5,
             'ckpt_steps': list(TOY_GRID)}, open(root / 'config.json', 'w'))
  mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None] * 255
  std = torch.tensor([0.229, 0.224, 0.225])[:, None, None] * 255
  log = []
  def save(step):
    if step in TOY_GRID:
      torch.save(model.state_dict(), root / f'model_{step:05d}.pt')
      torch.save(attr.compact_state(), root / f'attr_{step:05d}.pt')
      torch.save(attr_r.compact_state(), root / f'attrR_{step:05d}.pt')
      log.append({'step': step, 'val_acc': step / 1000, 'n_val': 0, 'sec': float(step)})
      json.dump(log, open(root / 'log.json', 'w'))
    if step in TOY_SNAPS:
      torch.save({'step': step, 'model': copy.deepcopy(model.state_dict()), 'log': list(log)},
                 hist / f'snap_{step:05d}.pt')
      torch.save({'step': step, 'attr': copy.deepcopy(attr.full_state())}, full / f'full_{step:05d}.pt')
  save(0)
  for step in range(steps):
    idx = torch.randint(0, len(ytr), (24,), generator=g)
    lr = 4e-3 * ((step + 1) / 5 if step < 5 else 0.5 * (1 + np.cos(np.pi * (step - 5) / (steps - 5))))
    for pg in opt.param_groups:
      pg['lr'] = lr
    opt.zero_grad()
    F.cross_entropy(model((xtr[idx].float() - mean) / std), ytr[idx]).backward()
    attr.step(ytr[idx], lr)
    attr_r.step(rand[idx], lr)
    opt.step()
    save(step + 1)
  rng = np.random.default_rng(seed)
  M = rng.integers(1, 200, (C, C))
  vis = np.argsort(np.argsort(M + M.T, 1), 1)
  vis = np.minimum(vis, vis.T)
  np.fill_diagonal(vis, 0)
  wn = rng.integers(2, 14, (C, C))
  wn = np.minimum(wn, wn.T)
  np.fill_diagonal(wn, 0)
  np.savez(root / 'visual.npz', rank_dist=vis.astype(np.int16))
  np.savez(root / 'wordnet.npz', dist=wn.astype(np.int16))


@pytest.fixture(scope='module')
def toy_exits(tmp_path_factory):
  """The toy run with every stage of experiments.convnext_exits run through its command line (one CPU thread: the
  test machine oversubscribes threads badly)."""
  import json
  import sys
  from experiments import convnext_exits as X
  threads = torch.get_num_threads()
  torch.set_num_threads(1)
  root = tmp_path_factory.mktemp('toyrun')
  try:
    build_toy_run(root)
    common = ['--run', str(root), '--out', str(root / 'exits'), '--history', str(root / 'history'), '--full',
              str(root / 'full'), '--probe', str(root / 'probe.pt'), '--visual', str(root / 'visual.npz'), '--wordnet',
              str(root / 'wordnet.npz'), '--batch', '16', '--device', 'cpu', '--grid', ','.join(map(str, TOY_GRID)),
              '--params', json.dumps(TOY_PARAMS)]
    argv = sys.argv
    for stage in ('integrity', 'stock', 'events', 'intervals', 'gradstate', 'tests'):
      sys.argv = ['convnext_exits', stage] + common
      X.main()
    sys.argv = argv
    yield root
  finally:
    torch.set_num_threads(threads)


def test_exits_pipeline_checks_and_outputs(toy_exits):
  """Every stage runs; snapshots pass the integrity checks; every interval file adds up exactly (fp64), after fp32
  storage, with the other position reference, and against the random split; the X1 excess shares sum to 1."""
  import json
  root = toy_exits / 'exits'
  integ = json.load(open(root / 'integrity.json'))
  assert integ['usable'] == list(TOY_SNAPS)
  assert integ['snapshots']['10']['I2']['ok'] and integ['snapshots']['40']['I2']['ok']
  assert all(r['I3_remainder_rel'] > 0 for r in integ['snapshots'].values())        # fp32 parts do not add up exactly
  ev = json.load(open(root / 'events.json'))
  assert ev['counts']['s0m1']['exits'] > 50 and ev['counts']['s0m1']['with_C1'] > 10
  res = json.load(open(root / 'exits_tests.json'))
  m = res['interval_checks']['max']
  assert m['complete_rel'] < 1e-12 and m['complete_rel_p2'] < 1e-12 and m['image_resid'] < 1e-12
  assert m['complete_rel_stored'] < 1e-6 and m['class_vs_random_total_rel'] < 1e-12 and m['zcheck'] < 1e-5
  ck = res['checks']
  assert ck['complete_rel_max'] < 1e-6 and ck['view_check'] < 1e-6 and ck['prov_check'] < 1e-5
  assert ck['event_check'] < 1e-6
  for gname in ('all', 'late'):
    sh = res['pooled'][gname]['s0m1']['X1']['anchored_replicated']['excess_vs_C1']
    if sh.get('shares'):
      assert abs(sum(v['est'] for v in sh['shares'].values()) - 1) < 1e-9
      assert abs(sum(v['est'] for v in sh['fine'].values()) - 1) < 1e-9
  assert set(res['primary']['p']) == {'X1_own_excess_share', 'X2_own_rise_dropped_minus_persistent',
                                      'X3_S1_withdrawn_minus_C1', 'X4_F2_relatives', 'X5_loop_own'}
  assert res['X5']['chains']['snap'] and res['path_dependence'] and res['carryover'] and res['rho']
  assert res['units'] and res['X4']['s0m1']['stage']['all']['F2']['n'] > 0


def test_exits_components_equal_brute_force_and_stock(toy_exits):
  """For exits with controls and for entries: the accumulated components equal a brute-force recomputation from the
  definitions (every group, every class), and they add up to the margin change (rise) measured by the independent
  stock passes at the window's ends."""
  import argparse
  import numpy as np
  from experiments import convnext_exits as X
  T = str(toy_exits)
  a = argparse.Namespace(out=f'{T}/exits', visual=f'{T}/visual.npz', wordnet=f'{T}/wordnet.npz', class_counts='',
                         prm=X.defaults() | TOY_PARAMS, skip_integrity=False)
  run = X.Run(T, f'{T}/history', f'{T}/full', grid=list(TOY_GRID),
              snaps=X.usable_snaps(a, T, f'{T}/history', f'{T}/full'))
  ctx = X.Ctx(a, run)
  b = 'stage2.block4'
  rec = X.analyze_block(ctx, b)
  g = run.g16
  exits = [ac for ac in rec['accs'] if ac.kind == 'exit' and len(ac.xs) > 1][:6]
  entries = [ac for ac in rec['accs'] if ac.kind == 'entry' and len(ac.xs) > 1][:6]
  assert exits and entries
  for ac in exits + entries:
    e, (s_, m_) = ac.e, X.DIRS[ac.d]
    n = e['n']
    lo, hi = (e['pk'], e['out']) if ac.kind == 'exit' else (e['pre'], e['first'])
    tot = np.zeros((len(ac.xs), len(X.COMP)))
    for j in range(lo + 1, hi + 1):
      xb = ctx.xhw(g[j - 1], g[j])['blocks'][b]
      ct = column_terms(run.inc(g[j - 1], g[j], b), {M: xb[f'Hbar_{M}'][m_, n].double() for M in MAPS},
                        {M: xb[f'qbar_{M}'][m_, n].double() for M in MAPS}, n)
      Tm, Fm, F0, E = ct['T'].numpy(), ct['F'].numpy(), ct['F0'].numpy(), ct['E'].numpy()
      Phi, Pi = xb['Phi'][m_, n].numpy(), xb['Pi'][m_, n].numpy()
      for i, (x, Q, cat) in enumerate(zip(ac.xs.tolist(), ac.Qs, ac.cat.numpy())):
        rest = lambda s: Fm[:, s].sum() - Fm[x, s] + F0[s]
        on_c = [Tm[cat == k, x].sum() for k in range(5)] + [Fm[x, x], rest(x), Phi[x], Pi[x], E[x]]
        on_q = [[Tm[cat == k, s].sum() for k in range(5)] + [Fm[x, s], rest(s), Phi[s], Pi[s], E[s]] for s in Q]
        tot[i, 0::2] += on_c
        if Q:
          tot[i, 1::2] -= np.mean(on_q, 0)
    got = ac.comp['nom'].numpy()
    assert np.abs(got - tot).max() <= 1e-12 * np.abs(tot).max()
    R0, R1 = ctx.R[g[lo]][b][m_][n], ctx.R[g[hi]][b][m_][n]
    for i, (x, Q) in enumerate(zip(ac.xs.tolist(), ac.Qs)):
      dM = (R1[x] - R0[x]) - (np.mean([R1[s] - R0[s] for s in Q]) if Q else 0.0)
      assert abs(got[i].sum() - dM) <= 1e-6 * np.abs(R0).max()


# ------------------------------------------------------------------------------------------- review fixes (exits)

def test_classify_needs_a_band_of_at_least_three_controls():
  """Without a random split the own band is the C1 own terms' 25-75%: one or two controls give no band ('few_controls'
  instead of an 'active' label decided by a single control); three controls, or a random band, classify."""
  import numpy as np
  from experiments import convnext_exits as X
  ex = np.zeros(len(X.COMP))
  ex[0] = -1.0                                                     # the excess fall is all the exit's own write
  thr = {'P_q05': None, 'rand_band': None, 'rand_q05': None, 'band_min': 3}
  assert X.classify(ex, np.zeros((1, len(X.COMP))), thr) == 'few_controls'
  assert X.classify(ex, np.zeros((2, len(X.COMP))), thr) == 'few_controls'
  ctrl3 = np.zeros((3, len(X.COMP)))
  ctrl3[:, 0] = [-0.1, 0.0, 0.1]
  assert X.classify(ex, ctrl3, thr) == 'active'
  assert X.classify(ex, np.zeros((1, len(X.COMP))), dict(thr, rand_band=(-0.2, 0.2), rand_q05=-0.2)) == 'active'
  assert X.classify(np.zeros(len(X.COMP)), ctrl3, thr) == 'no_excess_fall'
  assert 'few_controls' in X.UNLABELLED


def test_s2_permutation_null_keeps_exits_that_share_t_out():
  """S2's within-unit null reassigns the t_out of the unit's other exits: an exit with the same t_out must stay in it
  (the old rule dropped every exit whose t_out equals this one's, so the null could never say 'same')."""
  from types import SimpleNamespace
  import numpy as np
  from experiments import convnext_exits as X
  b, g = 'stage2.block0', [0, 10, 20, 30, 40]
  cap = {1: (1.0, 1.0), 2: (1.0, 1.0), 3: (0.1, 1.0), 4: (0.1, 1.0)}   # own capture of classes 0, 1: unit 0, unit 1
  own = {b: {(g[j - 1], g[j], h): torch.tensor([[cap[j][0]] * 2 + [1.0], [cap[j][1]] * 2 + [1.0]])
             for j in cap for h in (0, 1)}}
  ex = [dict(b=b, n=0, c=c, pk=1, out=3, anchored=True, rep=True, C1=[]) for c in (0, 1)]
  ctx = SimpleNamespace(run=SimpleNamespace(g16=g, blocks=[b], U={b: 2}), sd=lambda s, bb: np.ones((2, 2)),
                        floor=lambda s, h: 1e-3, events={'directions': {d: {'exits': ex} for d in X.DIRS}},
                        prm=dict(carriers=1, n_boot=20, seed=0))
  res = X.x3_s2(ctx, own)
  for d in X.DIRS:                                                 # unit 0's share halves in the t_out interval
    assert res[d]['n'] == 2 and res[d]['same']['exits'] == 1.0 and res[d]['same']['perm_null'] == 1.0


def test_bias_write_rate_parts_give_the_one_step_bias_write():
  """bias_rate_parts (the S3 crowding split): -lr_of(t - 1) mb_t / c1_t Db_t is the bias write of update t, i.e.
  Cb_t - Cb_(t-1), under a varying learning rate (the learning-rate factor of the split's dlr term)."""
  import numpy as np
  from experiments import convnext_exits as X
  torch.manual_seed(0)
  lin = nn.Linear(4, 3).double()
  sched = lambda s: 1e-2 * (s + 1) / 5 if s < 5 else 1e-2 * 0.5 ** (s - 4)
  attr = GroupAttribution({'l': lin}, n_groups=3, optimizer='adamw', lr=1e-2, betas=(0.9, 0.999), dtype=DT)
  g = torch.Generator().manual_seed(1)
  states = {}
  for s in range(8):
    x = torch.randn(8, 4, generator=g, dtype=DT)
    (lin(x) ** 2).sum().backward()
    attr.step(torch.randint(0, 3, (8,), generator=g), lr=sched(s))
    states[s + 1] = copy.deepcopy(attr.full_state()['l'])
  for t in (3, 7):
    for n in range(3):
      p = X.bias_rate_parts(states[t], n, [0, 1, 2], sched)
      write = -p['lr'] * p['mb'] / p['c1'] * p['Db']
      actual = (states[t]['Cb'][:, n] - states[t - 1]['Cb'][:, n]).numpy()
      assert np.abs(write - actual).max() <= 1e-12 * np.abs(actual).max() and np.abs(actual).max() > 0


def toy_ctx(root):
  import argparse
  from experiments import convnext_exits as X
  T = str(root)
  a = argparse.Namespace(out=f'{T}/exits', visual=f'{T}/visual.npz', wordnet=f'{T}/wordnet.npz', class_counts='',
                         prm=X.defaults() | TOY_PARAMS, skip_integrity=False)
  run = X.Run(T, f'{T}/history', f'{T}/full', grid=list(TOY_GRID),
              snaps=X.usable_snaps(a, T, f'{T}/history', f'{T}/full'))
  return X.Ctx(a, run), run


def test_crowding_split_adds_up_to_the_bias_write_rate_change(toy_exits):
  """S3: dlr + dD + dm + dc1 is the log change of each class's bias write rate between two snapshots (the learning
  rate term was missing, so the 'exact' split did not add up under the cosine schedule)."""
  import numpy as np
  from experiments import convnext_exits as X
  ctx, run = toy_ctx(toy_exits)
  b, n, xs, sa, sb = 'stage2.block4', 3, [0, 1, 2, 5], 25, 55
  res = X.crowding(ctx, b, n, xs, sa, sb, 1)
  lr_of = lr_schedule(run.cfg)
  lograte = lambda p: np.log(np.abs(p['lr'] * p['mb'] / p['c1'] * p['Db']))
  for M in MAPS:
    pa, pb = (X.bias_rate_parts(run.full(s)['attr'][f'{b}.pw_{M}'], n, xs, lr_of) for s in (sa, sb))
    want = lograte(pb) - lograte(pa)
    got = np.array(res[f'bias_{M}']['total'])
    ok = np.isfinite(want)
    assert ok.any() and np.abs(got[ok] - want[ok]).max() < 1e-12
    assert abs(res[f'bias_{M}']['dlr']) > 0.1                       # the schedule moved between the snapshots


def test_remainder_checks_flag_a_model_file_from_another_trajectory(toy_exits):
  """I3 at checkpoints: the stored fp32 remainder agrees with R computed from the model file (about 1e-7 of |W|); a
  model file from another point of training is flagged instead of being absorbed silently by R and E."""
  import json
  from experiments import convnext_exits as X
  res = json.load(open(toy_exits / 'exits' / 'exits_tests.json'))
  m = res['interval_checks']['max']
  assert m['R_stored_rel'] < 1e-6 and m['R_rel'] > 0 and m['R_stored_rel_random'] < 1e-6
  assert 'anchored_all' in res['pooled']['all']['s0m1']['X1']
  _, run = toy_ctx(toy_exits)
  b = 'stage2.block4'
  assert X.remainder_checks([run.parts(20, b)])['R_stored_rel'] < 1e-6
  key, other = f'{b}.pw_gate', run.model_sd(30)
  P = {'gate': load_parts(run.attr(20)[key], run.init[key], W=other[X.sd_key(b, 'gate')],
                          b=other[X.sd_key(b, 'gate', 'bias')])}
  assert X.remainder_checks([P])['R_stored_rel'] > 1e-3


def test_batch_components_refuses_a_class_without_a_column():
  """A class missing from a view's columns (interval files made for another events.json) raises instead of reading
  the last column through index -1."""
  from experiments import convnext_exits as X
  z = lambda *s: torch.zeros(*s, dtype=DT)
  view = X.make_view({'T': z(4, 2), 'F': z(4, 2), 'F0': z(2), 'E': z(2)}, z(2), z(2), z(2), torch.tensor([0, 1]), 4)
  xs = torch.tensor([0])
  cat = X.category_index(0, {}, 4)[None]
  Qp, Qm = X.padded([[1]], xs, 'cpu')
  assert X.batch_components(view, xs, Qp, Qm, cat).shape == (1, len(X.COMP))
  Qp, Qm = X.padded([[3]], xs, 'cpu')
  with pytest.raises(ValueError):
    X.batch_components(view, xs, Qp, Qm, cat)
