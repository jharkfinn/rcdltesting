"""Rare-class analysis (experiments/rare_common.py, rare_gonogo.py, convnext_rare.py) on a toy ConvNeXt-SwiGLU in fp64,
trained a few steps with a class split and a fine shadow split (head tracked): the decompositions add up, the local
knockout and the cached forward passes equal direct recomputation, first-order effects equal finite differences,
excision undo is bitwise, the fine split folds to the class split, and the statistics helpers are right."""

import itertools
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from experiments import convnext_rare as cr
from experiments import rare_common as rc
from experiments import rare_gonogo as gg
from rcdl.attribution import GroupAttribution, interval_buckets
from rcdl.convnext import ConvNeXt

NC, PER = 40, 12          # toy classes, images per class


def exact_parts(attr: GroupAttribution) -> dict:
  """Compact-state form with fp64 parts (no fp16 storage): W = a W_0 + sum_k C_k + R exactly."""
  out = {}
  for k, tr in attr.tracked.items():
    C, Cb = tr.C.clone(), tr.Cb.clone()
    out[k] = {'C': C, 'Cb': Cb, 'a': tr.a, 'ab': tr.ab, 't': tr.t,
              'R': tr.lin.weight.detach() - tr.a * tr.W0 - C.sum(0),
              'Rb': tr.lin.bias.detach() - tr.ab * tr.b0 - Cb.sum(0)}
  return out


def toy_selection():
  """8 selected classes (sets A and B, levels 10 and 100, stratum dense) of NC toy classes, random class tables."""
  rows = []
  for c, (s, lvl) in enumerate([('A', 10)] * 3 + [('A', 100)] * 3 + [('B', 10)] * 2):
    rows.append(dict(cls=c, name=f'c{c}', set=s, stratum='dense', level=lvl))
  rng = np.random.default_rng(3)
  e = rng.normal(size=(NC, 6))
  e /= np.linalg.norm(e, axis=1, keepdims=True)
  cs = np.clip(1 - e @ e.T, 0, None)
  np.fill_diagonal(cs, 0)
  D = rng.integers(2, 12, (NC, NC))
  D = np.minimum(D, D.T)
  np.fill_diagonal(D, 0)
  return rc.Selection({'classes': rows, 'relabel': [], 'noise': [], 'summary': {}}, D, cs)


def toy_spec():
  """Every special class: 3 single-image own groups (kind 1); class 0 also a decoy group of 2 images of class 20."""
  labels = np.repeat(np.arange(NC), PER)
  group = labels.astype(np.int32).copy()
  gclass, gkind, gowner = list(range(NC)), [0] * NC, [-1] * NC
  for c in range(8):
    for i in np.flatnonzero(labels == c)[:3]:
      group[i] = len(gclass)
      gclass.append(c), gkind.append(1), gowner.append(c)
  dec = np.flatnonzero(labels == 20)[:2]
  group[dec] = len(gclass)
  gclass.append(20), gkind.append(4), gowner.append(0)
  return dict(keep=np.arange(len(labels)), labels=labels, group=group, group_class=np.array(gclass, np.int16),
              group_kind=np.array(gkind, np.int8), group_owner=np.array(gowner, np.int16),
              noise=np.zeros(0, np.int64), meta={'run_set': 'A'})


@pytest.fixture(scope='module')
def toy():
  old, threads = torch.get_default_dtype(), torch.get_num_threads()
  torch.set_default_dtype(torch.float64)
  torch.set_num_threads(1)                        # tiny tensors: one thread is far faster on a shared CPU
  try:
    torch.manual_seed(0)
    model = ConvNeXt(dims=(4, 6), units=(5, 7), depths=(1, 2), n_classes=NC, stem=2, layer_scale=0.5)
    spec = toy_spec()
    layers = {f'{n}.{k}': getattr(b, k) for n, b in model.blocks() for k in ('pw_gate', 'pw_value')}
    layers['head'] = model.head
    akw = dict(optimizer='adamw', lr=3e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, bias_weight_decay=0.0,
               dtype=torch.float64)
    attr = GroupAttribution(layers, n_groups=NC, **akw)
    attr_f = GroupAttribution(layers, n_groups=len(spec['group_class']), shadow=True, **akw)
    tracked = [t for l in layers.values() for t in (l.weight, l.bias)]
    rest = [p for p in model.parameters() if p.requires_grad and not any(p is t for t in tracked)]
    opt = torch.optim.AdamW(rest, lr=3e-2)
    init = {k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in attr.tracked.items()}
    g = torch.Generator().manual_seed(1)
    X = torch.randn(NC * PER, 3, 8, 8, generator=g) + torch.as_tensor(spec['labels'])[:, None, None, None] / NC
    rng = np.random.default_rng(0)
    states = {}
    for t in range(6):
      idx = rng.choice(NC * PER, 32, replace=False)
      lam = float(rng.beta(0.8, 0.8))
      x = lam * X[idx] + (1 - lam) * X[idx].flip(0)
      y = torch.as_tensor(spec['labels'][idx])
      f = torch.as_tensor(spec['group'][idx].astype(np.int64))
      logp = F.log_softmax(model(x), -1)
      loss = -(lam * logp.gather(1, y[:, None]) + (1 - lam) * logp.gather(1, y.flip(0)[:, None])).mean()
      opt.zero_grad()
      loss.backward()
      attr.step(torch.stack([y, y.flip(0)], 1), 3e-2, weights=(lam, 1 - lam))
      attr_f.step(torch.stack([f, f.flip(0)], 1), 3e-2, weights=(lam, 1 - lam))
      opt.step()
      if t in (3, 5):
        states[t] = {'c': exact_parts(attr), 'f': exact_parts(attr_f), 'model': {k: v.clone() for k, v in
                                                                                  model.state_dict().items()}}
    model.eval()
    for a in (attr, attr_f):                       # analysis forwards with gradients must not reach the trackers
      for tr in a.tracked.values():
        tr.handle.remove()
    yield dict(model=model, spec=spec, attr=attr, attr_f=attr_f, pc=exact_parts(attr), pf=exact_parts(attr_f),
               init=init, X=X, states=states, sel=toy_selection())
  finally:
    torch.set_default_dtype(old)
    torch.set_num_threads(threads)


def images(toy, n=6, seed=5):
  g = torch.Generator().manual_seed(seed)
  return torch.randn(n, 3, 8, 8, generator=g, dtype=torch.float64)


@torch.no_grad()
def zeta_mean(model, x, c):
  return float(rc.zeta(model(x), c).mean())


# ------------------------------------------------------------------------------------------------ decompositions

def test_preactivation_parts_add_up_and_local_knockout_is_exact(toy):
  model, pc, pf = toy['model'], toy['pc'], toy['pf']
  x = images(toy)
  st = cr.unit_star(model, x)
  named = dict(model.blocks())
  mods = rc.tracked_modules(model, list(pc))
  for parts, S in ((pc, [1, 3, 5]), (pf, list(rc.own_groups(toy['spec'], 0)) + [41])):
    up = cr.unit_map_parts(parts, 'cpu', torch.float64)
    for b, blk in model.blocks():
      d = st[b]
      for n in (0, blk.pw_gate.out_features - 1):
        h = d['h'][:, n]
        Pg, Pv = cr.unit_parts(up, b, n, h)
        for m, P, val in (('pw_gate', Pg, d['g'][:, n]), ('pw_value', Pv, d['v'][:, n])):
          r = parts[f'{b}.{m}']
          init = r['a'] * toy['init'][f'{b}.{m}']['W0'][n] @ h.T + r['ab'] * toy['init'][f'{b}.{m}']['b0'][n] \
            + r['R'][n] @ h.T + r['Rb'][n]
          assert torch.allclose(P.sum(0) + init, val, atol=1e-12, rtol=0)
        Gs, Vs = cr.set_sums(Pg, [S]), cr.set_sums(Pv, [S])
        # direct: remove S's parts from the block's maps and recompute u at the same positions and block inputs
        xin = {}
        hk = blk.register_forward_pre_hook(lambda mod, inp: xin.__setitem__('x', inp[0]))
        with torch.no_grad():
          model(x)
        hk.remove()
        for mode, keys in (('both', [f'{b}.pw_gate', f'{b}.pw_value']), ('gate', [f'{b}.pw_gate']),
                           ('value', [f'{b}.pw_value'])):
          L = cr.knockout(d['g'][:, n], d['v'][:, n], Gs, Vs, 1.0, 1.0, mode)
          undo = rc.excise(mods, parts, S, keys)
          with torch.no_grad():
            u2 = blk.units(xin['x']).flatten(1, 2)[:, :, n]
          undo()
          u2 = torch.gather(u2, 1, d['p'][:, n:n + 1])[:, 0]
          assert torch.allclose(L[0], (d['u'][:, n] - u2).mean(), atol=1e-12, rtol=0), (b, n, mode)


def test_first_order_effects_equal_finite_differences(toy):
  model, pf, spec = toy['model'], toy['pf'], toy['spec']
  mods = rc.tracked_modules(model, list(pf))
  classes = [0, 3]
  xs = [images(toy, 4, 10 + c) for c in classes]
  grads = rc.class_gradients(model, mods, xs, classes, 'cpu')
  E = rc.group_effects(pf, grads)
  reg = rc.by_region(E)
  S = list(rc.own_groups(spec, 0))
  eps = 1e-5

  def shifted(keys, sign):
    with torch.no_grad():
      for k in keys:
        lin, r = mods[k], pf[k]
        lin.weight += sign * eps * r['C'][S].sum(0)
        lin.bias += sign * eps * r['Cb'][S].sum(0)
  for j, c in enumerate(classes):
    for name, keys in (('all', list(pf)), ('stage1', [k for k in pf if k.startswith('stage1.')]), ('head', ['head'])):
      shifted(keys, 1)
      zp = zeta_mean(model, xs[j], c)
      shifted(keys, -2)
      zm = zeta_mean(model, xs[j], c)
      shifted(keys, 1)
      fd = (zp - zm) / (2 * eps)
      e = float(reg[name][S, j].sum())
      assert abs(fd - e) <= 1e-7 * max(1.0, abs(e)), (name, c, fd, e)
  # per-unit effects (rows n of the gate and value maps) against central differences on those rows only
  Eu = cr.unit_effects(pf, grads, [S], model)[0, 0].numpy()
  units = cr.unit_list(model)
  for j in np.argsort(-np.abs(Eu))[:3]:
    b, n = units[j]

    def rows(sign):
      with torch.no_grad():
        for m in ('pw_gate', 'pw_value'):
          lin, r = mods[f'{b}.{m}'], pf[f'{b}.{m}']
          lin.weight[n] += sign * eps * r['C'][S][:, n].sum(0)
          lin.bias[n] += sign * eps * r['Cb'][S][:, n].sum(0)
    rows(1)
    zp = zeta_mean(model, xs[0], classes[0])
    rows(-2)
    zm = zeta_mean(model, xs[0], classes[0])
    rows(1)
    assert abs((zp - zm) / (2 * eps) - Eu[j]) <= 1e-7 * max(1e-3, abs(Eu[j]))
  # additivity over units: the per-unit effects of the gate and value rows sum to the unit-map total
  Eu = cr.unit_effects(pf, grads, [S, [S[0]], [S[1], S[2]]], model)
  assert torch.allclose(Eu[0].sum(1), reg['units'][S].sum(0), atol=1e-12, rtol=0)
  assert torch.allclose(Eu[0], Eu[1] + Eu[2], atol=1e-12, rtol=0)
  # regions add up to the total, and the total is the inner product with the summed parts
  assert torch.allclose(sum(reg[r] for r in ('stage0', 'stage1', 'head')), reg['all'], atol=1e-12, rtol=0)
  tot = sum(rc.flat_parts(pf[k], S, dtype=torch.float64).sum(0) @ grads[k].T for k in pf)
  assert torch.allclose(tot, reg['all'][S].sum(0), atol=1e-12, rtol=0)


def test_excision_undo_restores_weights_bitwise(toy):
  model, pf, pc = toy['model'], toy['pf'], toy['pc']
  mods = rc.tracked_modules(model, list(pf))
  before = {k: v.clone() for k, v in model.state_dict().items()}
  undo = rc.excise(mods, pf, list(rc.own_groups(toy['spec'], 3)), list(pf))
  assert not torch.equal(model.head.weight, before['head.weight'])
  undo()
  for k, v in model.state_dict().items():
    assert torch.equal(v, before[k]), k
  x = images(toy)
  cr.stage_excision(model, mods, pc, [[1, 2], [7]], x, 1)
  for k, v in model.state_dict().items():
    assert torch.equal(v, before[k]), k


def test_stage_excision_and_unit_ablation_equal_full_forward(toy):
  model, pf = toy['model'], toy['pf']
  mods = rc.tracked_modules(model, list(pf))
  x, c = images(toy), 3
  S = list(rc.own_groups(toy['spec'], c))
  X = cr.stage_excision(model, mods, pf, [S], x, c)
  base = zeta_mean(model, x, c)
  for j, keys in enumerate([[k for k in pf if k.startswith(f'stage{s}.')] for s in range(2)] + [['head']]):
    undo = rc.excise(mods, pf, S, keys)
    assert abs(X[0, j] - (base - zeta_mean(model, x, c))) < 1e-12
    undo()
  # unit ablation with block caching and several units per pass against one full forward per unit
  ubar = {b: torch.linspace(-0.3, 0.4, blk.pw_gate.out_features) for b, blk in model.blocks()}
  y = torch.arange(len(x)) % NC
  D = cr.unit_ablation(model, x, y, ubar, units_per_pass=3, batch=4)
  for j, (b, n) in enumerate(cr.unit_list(model)):
    blk = dict(model.blocks())[b]

    def pre(mod, inp):
      u = inp[0].clone()
      u[..., n] = ubar[b][n]
      return (u,)
    hk = blk.pw_down.register_forward_pre_hook(pre)
    with torch.no_grad():
      za = rc.zeta(model(x), y)
    hk.remove()
    with torch.no_grad():
      z0 = rc.zeta(model(x), y)
    assert torch.allclose(D[j], z0 - za, atol=1e-12, rtol=0)


def test_fold_of_fine_parts_equals_class_parts(toy):
  pf, pc, spec = toy['pf'], toy['pc'], toy['spec']
  folded = rc.fold({k: {'C': r['C'], 'Cb': r['Cb']} for k, r in pf.items()}, spec['group_class'], NC)
  for k in pc:
    assert torch.allclose(folded[k]['C'].double(), pc[k]['C'], atol=1e-6 * float(pc[k]['C'].abs().max()), rtol=0), k
    assert torch.allclose(folded[k]['Cb'].double(), pc[k]['Cb'], atol=1e-6 * float(pc[k]['Cb'].abs().max()), rtol=0)
  # fp64 fold (no fp32 accumulation): exact to rounding
  for k in pc:
    exact = torch.zeros_like(pc[k]['C']).index_add_(0, torch.as_tensor(spec['group_class'], dtype=torch.long),
                                                     pf[k]['C'])
    assert torch.allclose(exact, pc[k]['C'], atol=1e-14, rtol=0)
  # the stored fp16 form (with its fp32 remainder) folds within fp16 tolerance, and W = a W_0 + sum C + R
  cf, cc = toy['attr_f'].compact_state(), toy['attr'].compact_state()
  chk = gg.fold_check(cf, cc, spec['group_class'], NC)
  assert chk['max_rel'] < 2e-3
  for k, r in cc.items():
    W = r['a'] * toy['init'][k]['W0'].float() + r['C'].float().sum(0) / r['C_scale'] + r['R']
    lin = toy['attr'].tracked[k].lin
    assert torch.allclose(W.double(), lin.weight.detach(), atol=1e-6, rtol=0)


def test_interval_buckets_on_checkpoint_sums_add_up(toy):
  st, init, model = toy['states'], toy['init'], toy['model']
  sets = {'near': [1, 2, 3], 'far': [10, 11], 'rest': [k for k in range(NC) if k not in (1, 2, 3, 10, 11)]}
  keys = [k for k in st[3]['c'] if k.startswith('stage1.block1')]
  s1, s2 = cr.compact_sums(st[3]['c'], sets, keys), cr.compact_sums(st[5]['c'], sets, keys)
  bk = interval_buckets(s1, s2, {k: init[k] for k in keys})
  for k in keys:
    tot = sum(bk[k]['write'].values()) + sum(bk[k]['dec'].values()) + bk[k]['dec_0'] + bk[k]['rem']
    name = k.replace('stage', 'stages.').replace('.block', '.') + '.weight'
    W2, W1 = st[5]['model'][name].float(), st[3]['model'][name].float()
    assert torch.allclose(tot, W2 - W1, atol=1e-6, rtol=0)
  # the churn traffic on two units is finite and non-negative
  units = cr.unit_list(model)
  ids = [j for j, (b, n) in enumerate(units) if b == 'stage1.block1'][:2]
  x = images(toy)
  sb = cr.unit_star(model, x, ['stage1.block1'])
  h = {'stage1.block1': sb['stage1.block1']['h']}
  out = cr.churn_interval(s1, s2, init, ids, units, {j: 1.0 for j in ids}, {j: -1.0 for j in ids},
                          np.ones(len(units)), h, h)
  assert all(np.isfinite(v).all() and min(v) >= 0 for v in out.values())
  assert all(v[1] == 0 for v in out.values())          # the same inputs at t and t': no drift


# ---------------------------------------------------------------------------------------------- nulls, statistics

def test_auroc_and_ranks():
  rng = np.random.default_rng(0)
  pos, neg = rng.integers(0, 5, 30).astype(float), rng.integers(0, 5, 40).astype(float)
  brute = np.mean([(p > q) + 0.5 * (p == q) for p, q in itertools.product(pos, neg)])
  assert abs(rc.auroc(pos, neg) - brute) < 1e-12
  assert rc.auroc([3, 4], [1, 2]) == 1.0 and rc.auroc([1, 2], [3, 4]) == 0.0 and rc.auroc([1], [1]) == 0.5
  assert np.array_equal(rc.rankdata([3, 1, 3, 2]), [3.5, 1, 3.5, 2])
  assert rc.pct_rank(2, [1, 2, 3]) == 0.5


def test_matched_draws_reproducible_and_matched():
  rng = np.random.default_rng(1)
  counts = rng.choice([732, 900, 1300, 1300, 1300], 200)
  norms = rng.lognormal(size=200)
  cd, nd = cr.deciles(counts), cr.deciles(norms)
  assert cd.min() >= 0 and cd.max() <= 9 and len(set(cd[counts == 1300])) == 1    # ties share a decile
  near, pool = list(range(10)), list(range(20, 200))
  a = cr.matched_draws(near, pool, cd, nd, 300, np.random.default_rng([0, 7]))
  b = cr.matched_draws(near, pool, cd, nd, 300, np.random.default_rng([0, 7]))
  c = cr.matched_draws(near, pool, cd, nd, 300, np.random.default_rng([0, 8]))
  assert np.array_equal(a, b) and not np.array_equal(a, c)
  assert np.isin(a, pool).all()
  for i, k in enumerate(near):
    joint = [p for p in pool if cd[p] == cd[k] and nd[p] == nd[k]]
    if len(joint) >= 2:
      assert (cd[a[:, i]] == cd[k]).all() and (nd[a[:, i]] == nd[k]).all()
    else:
      assert (nd[a[:, i]] == nd[k]).all() or len([p for p in pool if nd[p] == nd[k]]) < 2
  assert all(len(set(r)) == len(r) for r in a)


def test_statistics_helpers():
  h = cr.holm({'a': 0.01, 'b': 0.04, 'c': 0.03})
  assert abs(h['a'] - 0.03) < 1e-12 and abs(h['c'] - 0.06) < 1e-12 and abs(h['b'] - 0.06) < 1e-12
  assert cr.binom_sf(0, 10, 0.3) == pytest.approx(1.0)
  assert cr.binom_sf(10, 10, 0.5) == pytest.approx(0.5 ** 10)
  w = cr.wilcoxon([1, -1, 2, -2, 3, -3])
  assert w['p'] == pytest.approx(1.0)
  w = cr.wilcoxon(np.arange(1, 21))
  assert w['p'] < 1e-3 and w['W_plus'] == 210
  b1, b2 = cr.bootstrap_median([0.5, 0.2, 0.9, 0.4], 2000, 3), cr.bootstrap_median([0.5, 0.2, 0.9, 0.4], 2000, 3)
  assert b1 == b2 and b1['median'] == 0.45 and b1['p_two_sided'] < 0.01


def test_pi_lambda3_and_p3_formulas():
  units = []
  for k in range(3):
    units.append({'near_both': 1.0 + k, 'rand_both': np.array([0.0, 0.1, -0.1, 0.2]), 'near_gate': 0.5,
                  'rand_gate': np.zeros(4), 'near_value': 0.5, 'rand_value': np.zeros(4), 'own_both': 0.2,
                  'nown_both': np.array([0.0, 0.1, -0.1]), 'own_gate': 0.1, 'nown_gate': np.zeros(3),
                  'own_value': 0.1, 'nown_value': np.zeros(3), 'lin_near': 1.0, 'lin_rand': np.zeros(4),
                  'lin_own': 0.1, 'lin_nown': np.zeros(3), 'L_class': np.linspace(-1, 1, 10)})
  r = cr.summarize_p1(units, 2, [3, 4])
  F_, O_ = np.median([1.0 + k - 0.05 for k in range(3)]), 0.2
  assert r['F'] == pytest.approx(F_) and r['O'] == pytest.approx(O_)
  assert r['PI'] == pytest.approx((F_ - O_) / (abs(F_) + abs(O_)))
  assert r['defined'] and r['F_lin_pct'] == 1.0
  X = np.array([[0.1, 0.2, 0.3, 0.6, 0.5], [0, 0, 0, 0.1, 0], [0, 0.1, 0, 0, 0], [0, 0, 0.1, 0, 0]])
  l3 = cr.lambda3(X)
  Xs = X[0] - np.median(X[1:], 0)
  assert l3['lambda3'] == pytest.approx(max(Xs[3], 0) / np.maximum(Xs[:4], 0).sum())
  assert l3['head_share'] == pytest.approx(Xs[4] / (np.maximum(Xs[:4], 0).sum() + Xs[4]))
  E_own, E_nown = np.array([1.0, 0.0, 2.0]), np.array([[0.0, 0.5, 0.0], [0.2, 0.0, 0.0]])
  rel, mis = np.array([1.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.0])
  p3 = cr.p3_class(E_own, E_nown, rel, mis, flow=np.array([1.0, 1.0, 1.0]))
  w = np.maximum(E_own - np.median(E_nown, 0), 0)
  assert p3['R_own'] == pytest.approx((w * rel).sum() / w.sum()) and p3['R_flow'] == pytest.approx(0.5)
  assert p3['Delta_flow'] == pytest.approx(p3['R_own'] - 0.5) and p3['R_mis'] == 0.0


# --------------------------------------------------------------------------------------------- pipelines (smoke)

def toy_images(seed=2, per=10):
  g = torch.Generator().manual_seed(seed)
  Vl = np.repeat(np.arange(8), per)
  V = torch.randint(0, 256, (len(Vl), 3, 8, 8), generator=g, dtype=torch.uint8)
  Qy = np.repeat(np.arange(NC), 4)
  Q = torch.randint(0, 256, (len(Qy), 3, 8, 8), generator=g, dtype=torch.uint8)
  return V, {c: np.flatnonzero(Vl == c) for c in range(8)}, Q, Qy


def test_measure_pipeline_on_toy(toy):
  V, Vpos, Q, Qy = toy_images()
  out = cr.measure(toy['model'], toy['sel'], toy['spec'], toy['pc'], toy['pf'], V, Vpos, Q, Qy, 'cpu', run_set='A',
                   counts=np.full(NC, PER), n_draws=50, log=lambda s: None, check_units=4, classes=[0, 1, 2, 3, 4, 6])
  recs = {r['cls']: r for r in out['classes']}
  assert set(recs) == {0, 1, 2, 3, 4, 6}
  r = recs[0]
  assert r['condition'] == 'rare' and recs[6]['condition'] == 'common'
  for key in ('F', 'top1', 'auroc_near', 'P2', 'P2_whole', 'P3'):
    assert key in r, key
  if r['K']:
    assert np.isfinite(r['F']) and 'PI' in r and -1 <= r['PI'] <= 1
    nl = out['_nulls'][0]                                  # saved per-unit knockouts (amendment 1, C3)
    Fj = np.median(nl['rand_both'] - np.median(nl['rand_both'], 1, keepdims=True), 0)
    assert nl['rand_both'].shape == (len(r['K']), 50) and r['F_pct'] == pytest.approx(cr.pct_rank(r['F'], Fj))
  assert len(r['P2']['X_stage']) == 2 and r['P3']['check']['X'] and np.isfinite(r['P3']['sum_w'])
  E, X = np.array(r['P3']['check']['E']), np.array(r['P3']['check']['X'])   # full removal: first order is approximate
  assert np.isfinite(E).all() and np.isfinite(X).all()
  # phase 0: own = the whole class
  out0 = cr.measure(toy['model'], toy['sel'], None, toy['pc'], None, V, Vpos, Q, Qy, 'cpu', n_draws=20,
                    log=lambda s: None, check_units=0, classes=[0, 3])
  assert all('O_whole' in r or not r['K'] for r in out0['classes'])
  s7 = cr.s7_consolidation(out0['_DeltaQ'], cr.unit_list(toy['model']), toy['sel'])
  assert 'classes' in s7


def test_gonogo_on_toy(toy):
  V, Vpos, Q, Qy = toy_images(per=6)
  recs, ex = gg.analyse(toy['model'], toy['sel'], toy['spec'], toy['pc'], toy['pf'], V, Vpos, Q, Qy, 'A', 'cpu',
                        log=lambda s: None)
  by = {r['cls']: r for r in recs}
  assert all(np.isclose(sum(v for k, v in r['E_own_region'].items() if k not in ('all', 'units')), r['E_own'])
             for r in recs)
  assert ex['n'] == 3 and ex['sign_agreement'] >= 0 and 'X_own' in by[3]
  assert by[3]['E_pass'] == (by[3]['E_own'] > np.percentile(by[3]['E_null'], 95))
  v = gg.verdict(recs)
  assert v['n_level100'] == 3 and v['decision'] in ('GO', 'NO-GO')
  fake = [dict(r, auroc_far=0.7, E_pass=True) for r in recs]
  assert gg.verdict(fake)['decision'] == 'GO'
  fake = [dict(r, auroc_far=0.55, E_pass=True) for r in recs]
  assert gg.verdict(fake)['decision'] == 'NO-GO'


def test_per_example_row_norms_equal_single_example_gradients(toy):
  model = toy['model']
  x = images(toy, 3, 21)
  y, y2 = torch.tensor([0, 3, 5]), torch.tensor([7, 1, 2])
  lam = torch.tensor([0.3, 0.8, 0.5], dtype=torch.float64)
  params = [p for b, blk in model.blocks() for m in ('pw_gate', 'pw_value') for p in getattr(blk, m).parameters()]
  for p in params:
    p.requires_grad_(True)
  D = {f'{b}.{m}': (torch.rand_like(getattr(blk, m).weight) + 0.5, torch.rand_like(getattr(blk, m).bias) + 0.5)
       for b, blk in model.blocks() for m in ('pw_gate', 'pw_value')}
  try:
    for scale in (None, D):
      model.zero_grad(set_to_none=True)
      got = cr.per_example_row_norms(model, x, y, y2, lam, scale)
      for i in range(3):
        model.zero_grad(set_to_none=True)
        logp = F.log_softmax(model(x[i:i + 1]), -1)[0]
        (-(lam[i] * logp[y[i]] + (1 - lam[i]) * logp[y2[i]])).backward()
        ref = []
        for b, blk in model.blocks():
          sq = 0
          for m in ('pw_gate', 'pw_value'):
            lin = getattr(blk, m)
            gw, gb = lin.weight.grad, lin.bias.grad
            if scale is not None:
              gw, gb = gw * D[f'{b}.{m}'][0], gb * D[f'{b}.{m}'][1]
            sq = sq + gw.pow(2).sum(1) + gb.pow(2)
          ref.append(sq.sqrt() * lam[i])
        assert torch.allclose(got[i], torch.cat(ref), atol=1e-12, rtol=1e-10)
  finally:
    for p in params:
      p.requires_grad_(False)
    model.zero_grad(set_to_none=True)


def test_interval_learning_rates_sum_the_schedule():
  cfg = {'lr': 2e-3, 'warmup': 10, 'steps': 100}
  lt = cr.interval_lr(cfg, [0, 5, 50, 100])
  sched = cr.lr_schedule(cfg)
  assert lt[0] == pytest.approx(sum(sched(s) for s in range(5)))
  assert sum(lt.values()) == pytest.approx(sum(sched(s) for s in range(100)))


def test_compare_on_synthetic_runs(tmp_path):
  import argparse
  import json
  rng = np.random.default_rng(0)

  def rec(c, cond, kind='selected', level=10, pi=0.3, l3=0.7, df=0.0, top1=0.3):
    return {'cls': c, 'kind': kind, 'stratum': 'dense', 'level': level, 'condition': cond, 'set': 'A',
            'PI': pi + 0.1 * rng.normal(), 'defined': True, 'F': 0.5, 'O': 0.1, 'O_whole': 2.0, 'auroc_far': 0.7,
            'top1': top1, 'own_share': 0.1, 'signature': False, 'pseudo_signature': False,
            'P2': {'lambda3': l3, 'defined': True}, 'P2_whole': {'lambda3': 0.5},
            'P3': {'defined': True, 'Delta_flow': df + 0.01 * rng.normal(), 'R_own': 0.3, 'R_mis': 0.1,
                   'R_flow': 0.3}}
  rare = {'run': 'A0', 'run_set': 'A', 'classes': [rec(c, 'rare') for c in range(30)]
          + [rec(100 + c, 'rare', 'noise', None, pi=-0.5, top1=0.8) for c in range(4)]}
  c0 = {'run': 'C0', 'run_set': 'C', 'classes': [rec(c, 'common', l3=0.4) for c in range(30)]}
  other = {'run': 'B0', 'run_set': 'B', 'classes': [rec(c, 'common', l3=0.45) for c in range(30)]}
  files = {}
  for name, o in (('rare', rare), ('c0', c0), ('other', other)):
    files[name] = str(tmp_path / f'{name}.json')
    json.dump(o, open(files[name], 'w'))
  out = str(tmp_path / 'cmp.json')
  a = argparse.Namespace(rare=[files['rare'], files['other']], c0=[files['c0']], rare1=[], c1=[], e1=[], e1r=[],
                         out=out)
  cr.cmd_compare(a)
  res = json.load(open(out))
  s0 = res['seed0']
  assert s0['P1']['n'] == 30 and s0['P1']['median'] > 0 and s0['P1']['p_two_sided'] < 0.01
  assert s0['P2']['median'] == pytest.approx(0.2) and set(res['holm_seed0']) == {'P1', 'P2', 'P3'}
  assert s0['noise_control']['passes'] and res['decision']['against_essay'] == []
  assert s0['noise_control']['state'] == 'pass' and s0['relabel_control']['state'] == 'uninformative'
  assert res['decision']['controls_state'] == {'seed0': 'uninformative'}       # no relabel control in this toy


# ---------------------------------------------------------------------------------------- review fixes (adversarial)

def control_selection():
  """toy_selection plus a relabel control (class 8, source 30) and a noise control (class 9, source 31), set A."""
  base = toy_selection()
  sel = dict(base.sel)
  sel['relabel'] = [dict(cls=8, name='c8', set='A', source=30, source_name='c30', n=10)]
  sel['noise'] = [dict(cls=9, name='c9', set='A', source=31, source_name='c31', n=10)]
  D = base.D.copy()
  np.fill_diagonal(D, 0)
  CS = np.zeros((NC, NC))
  CS[np.arange(NC)[:, None], base.order] = np.arange(1, NC)[None, :] / NC     # the same visual order as base
  return rc.Selection(sel, D, CS)


def test_builder_scores_count_cross_writes_only():
  """beta^x(k) sums k's writes into the top-20 poles other than c and k (no pole leak), beta(c) = 0."""
  rng = np.random.default_rng(4)
  model = ConvNeXt(dims=(4,), units=(3,), depths=(1,), n_classes=NC, stem=2)
  atlas = {}
  for b, blk in model.blocks():
    U = blk.pw_gate.out_features
    poles = torch.as_tensor(np.stack([rng.permutation(NC) for _ in range(U)]))
    atlas[b] = {'poles': poles, 'Vm': torch.randn(U, NC, dtype=torch.float64),
                'Gm': torch.randn(U, NC, dtype=torch.float64),
                'value_writers': torch.randn(U, NC, NC, dtype=torch.float64).half(),
                'gate_writers': torch.randn(U, NC, NC, dtype=torch.float64).half()}
  bms = cr.builder_matrices(atlas, model)
  b = model.blocks()[0][0]
  for n, bm in enumerate(bms):
    top = [int(k) for k in atlas[b]['poles'][n][:20]]
    raw = np.zeros((20, NC))
    for j, cp in enumerate(top):
      sv, sg = np.sign(float(atlas[b]['Vm'][n, cp])), np.sign(float(atlas[b]['Gm'][n, cp]))
      raw[j] = np.maximum(sv * atlas[b]['value_writers'][n, j].double().numpy(), 0) + \
        np.maximum(sg * atlas[b]['gate_writers'][n, j].double().numpy(), 0)
    assert np.allclose(bm['raw'], raw, atol=1e-6)
    for c in (top[0], top[5], int(atlas[b]['poles'][n][30])):            # c a pole, and c not a pole
      beta = cr.builders(bm, c)
      brute = np.array([sum(raw[j, k] for j, cp in enumerate(top) if cp not in (c, k)) for k in range(NC)])
      brute[c] = 0.0
      assert np.allclose(beta, brute, atol=1e-6), (n, c)
      k = top[3] if top[3] != c else top[4]                             # a pole: its own response is not counted
      assert np.isclose(beta[k], sum(raw[j, k] for j, cp in enumerate(top) if cp not in (c, k)), atol=1e-6)
  # S7's own niche share: c's own row of the top carrier when c is a pole, else undefined
  units = cr.unit_list(model)
  DeltaQ = torch.zeros(len(units), NC, dtype=torch.float64)
  sel = toy_selection()
  sel.nW[:] = 10
  sel.cos10[:] = 0.0
  units = [('stage2.block0', n) for _, n in units]                    # S7 reads the late stages only
  top0 = [int(k) for k in atlas[b]['poles'][0][:20]]
  c_in, c_out = top0[2], int(atlas[b]['poles'][0][25])
  DeltaQ[0, c_in] = DeltaQ[0, c_out] = 1.0                              # unit 0 is the top carrier of both
  s7 = cr.s7_consolidation(DeltaQ, units, sel, bms)
  rows = {r['cls']: r for r in s7['classes']}
  raw0 = bms[0]['raw']
  assert np.isclose(rows[c_in]['N'], raw0[2, c_in] / raw0[2].sum())
  assert rows[c_out]['N'] is None


def test_control_sets_probes_and_snapshots(tmp_path):
  sel = control_selection()
  assert 30 not in sel.pi(8) and 30 not in sel.far(8) and 31 not in sel.pi(9)       # the source is not a random class
  assert 30 in sel.pi(0) or 30 in sel.near10(0)                                    # other classes keep it
  assert sel.n_own(8) == [] and sel.cell(8) == 'A/relabel'
  V = torch.arange(40 * 2 * 3 * 4 * 4, dtype=torch.int64).remainder(256).to(torch.uint8).view(80, 3, 4, 4)
  Vpos = {c: np.arange(2 * i, 2 * i + 2) for i, c in enumerate(range(40))}
  assert torch.equal(rc.probes(sel, 8, V, Vpos), V[Vpos[30]])                     # relabel: the source's images
  nx = {9: torch.zeros(10, 3, 4, 4, dtype=torch.uint8)}
  assert torch.equal(rc.probes(sel, 9, V, Vpos, nx), nx[9])
  assert torch.equal(rc.probes(sel, 9, V, Vpos, {}), V[Vpos[9]])                  # no noise images: no KeyError
  assert rc.probe_kind(sel, 9, {}) != 'noise' and rc.probe_kind(sel, 8) == 'source_val'
  # noise images: the run's own spec first, then the other specs
  labels = np.repeat(np.arange(NC), 3)
  labels[[93, 94]] = 9
  spec = {'noise': np.array([93, 94]), 'labels': labels}
  ni = rc.noise_images([None, {'noise': np.zeros(0, np.int64)}, spec], sel)
  assert list(ni) == [9] and ni[9].shape == (2, 3, 224, 224)
  # snapshots: step 0 (v = 0) is never used for step sizes; history/ snapshots are found
  os.makedirs(tmp_path / 'history')
  for s in (0, 50, 2500):
    open(tmp_path / f'snap_{s:05d}.pt', 'w').close()
  open(tmp_path / 'history' / 'snap_12000.pt', 'w').close()
  snaps = rc.snapshot_files(str(tmp_path))
  assert sorted(snaps) == [0, 50, 2500, 12000] and snaps[12000].endswith(os.path.join('history', 'snap_12000.pt'))
  assert cr.flow_snapshot(snaps, 0) == 50 and cr.flow_snapshot(snaps, 2500) == 2500
  assert cr.flow_snapshot(snaps, 12500) == 12000 and rc.nearest_snapshot({0: 'x'}, 0) is None
  st = {'m': {'v': torch.full((2, 3), 4e-6), 'vb': torch.full((2,), 1e-6), 't': 10}}
  D = cr.step_scales({'stage0.block0.pw_gate': st['m'], 'head': st['m']})
  c2 = 1 - 0.999 ** 10
  assert list(D) == ['stage0.block0.pw_gate']
  assert torch.allclose(D['stage0.block0.pw_gate'][0], torch.full((2, 3), 1 / ((4e-6 / c2) ** 0.5 + 1e-8)))
  with pytest.raises(AssertionError):
    cr.step_scales({'k': dict(st['m'], t=0)})


def test_row_norms_for_several_scalings_equal_single_calls(toy):
  model = toy['model']
  x = images(toy, 3, 22)
  y, y2 = torch.tensor([0, 3, 5]), torch.tensor([7, 1, 2])
  lam = torch.tensor([0.3, 0.8, 0.5], dtype=torch.float64)
  params = [p for b, blk in model.blocks() for m in ('pw_gate', 'pw_value') for p in getattr(blk, m).parameters()]
  for p in params:
    p.requires_grad_(True)
  D = {f'{b}.{m}': (torch.rand_like(getattr(blk, m).weight) + 0.5, torch.rand_like(getattr(blk, m).bias) + 0.5)
       for b, blk in model.blocks() for m in ('pw_gate', 'pw_value')}
  try:
    both = cr.row_norms(model, x, y, y2, lam, [None, D])
    model.zero_grad(set_to_none=True)
    one = [cr.per_example_row_norms(model, x, y, y2, lam, s) for s in (None, D)]
    for a_, b_ in zip(both, one):
      assert torch.allclose(a_, b_, atol=1e-13, rtol=0)
  finally:
    for p in params:
      p.requires_grad_(False)
    model.zero_grad(set_to_none=True)


def test_measure_nulls_do_not_depend_on_the_analysed_classes(toy):
  """--only: the N_own nulls (O, PI, P2, P3) use every special class's own groups, so a partial run gives the same
  numbers as a run over more classes."""
  V, Vpos, Q, Qy = toy_images()
  kw = dict(run_set='A', counts=np.full(NC, PER), n_draws=30, log=lambda s: None, check_units=2)
  one = cr.measure(toy['model'], toy['sel'], toy['spec'], toy['pc'], toy['pf'], V, Vpos, Q, Qy, 'cpu', classes=[3],
                   **kw)['classes'][0]
  more = {r['cls']: r for r in cr.measure(toy['model'], toy['sel'], toy['spec'], toy['pc'], toy['pf'], V, Vpos, Q,
                                          Qy, 'cpu', classes=[3, 4, 5], **kw)['classes']}[3]
  assert one['n_nown'] == 2 and 'P2' in one and 'P3' in one
  if one['K']:
    assert one['O'] == pytest.approx(more['O']) and one['PI'] == pytest.approx(more['PI'])
  assert one['P2']['X_stage'] == pytest.approx(more['P2']['X_stage'])
  assert one['P3']['sum_w'] == pytest.approx(more['P3']['sum_w'])


def test_gonogo_probes_controls_on_their_images(toy):
  """Relabel controls are probed on the source class's images (Far without the source), noise controls on their
  noise images; the units-only pass is reported."""
  sel = control_selection()
  g = torch.Generator().manual_seed(9)
  cls = list(range(10)) + [30, 31]
  Vl = np.repeat(cls, 6)
  V = torch.randint(0, 256, (len(Vl), 3, 8, 8), generator=g, dtype=torch.uint8)
  Vpos = {c: np.flatnonzero(Vl == c) for c in cls}
  Qy = np.repeat(np.arange(NC), 4)
  Q = torch.randint(0, 256, (len(Qy), 3, 8, 8), generator=g, dtype=torch.uint8)
  nx = {9: torch.randint(0, 256, (4, 3, 8, 8), generator=g, dtype=torch.uint8)}
  recs, _ = gg.analyse(toy['model'], sel, toy['spec'], toy['pc'], toy['pf'], V, Vpos, Q, Qy, 'A', 'cpu',
                       log=lambda s: None, noise_x=nx, excision=False)
  by = {r['cls']: r for r in recs}
  assert by[8]['probe'] == 'source_val' and by[8]['n_probe'] == 6 and by[9]['probe'] == 'noise'
  assert by[9]['n_probe'] == 4 and by[8]['condition'] == 'rare'
  zs = rc.logits_of(toy['model'], V[Vpos[30]], 'cpu')
  assert by[8]['top1'] == pytest.approx(float((zs.argmax(1) == 8).float().mean()))
  sq = rc.zeta(rc.logits_of(toy['model'], Q, 'cpu'), 8).numpy()
  assert by[8]['auroc_near'] == pytest.approx(rc.auroc(rc.zeta(zs, 8).numpy(), sq[np.isin(Qy, sel.near10(8))]))
  assert by[3]['E_units_pass'] == (by[3]['E_own_region']['units'] > np.percentile(by[3]['E_units_null'], 95))
  assert 'frac_E_units_pass' in gg.verdict(recs)
  recs2, _ = gg.analyse(toy['model'], sel, toy['spec'], toy['pc'], toy['pf'], V, Vpos, Q, Qy, 'A', 'cpu',
                        log=lambda s: None, noise_x={}, excision=False)   # spec_A missing: real images, no crash
  assert {r['cls']: r for r in recs2}[9]['probe'].startswith('val')


def test_streamed_validation_crops_equal_val_images(tmp_path):
  pytest.importorskip('pyarrow')
  import io
  import pyarrow as pa
  import pyarrow.parquet as pq
  from PIL import Image
  from experiments.convnext_umbrella import val_images
  d = tmp_path / 'val' / 'data'
  d.mkdir(parents=True)
  for f in range(2):
    rows = []
    for i in range(5):
      buf = io.BytesIO()
      Image.new('RGB', (60 + 7 * i, 50 + f), (40 * i, 9 * f, 7)).save(buf, format='JPEG')
      rows.append({'image': {'bytes': buf.getvalue(), 'path': None}, 'label': i})
    pq.write_table(pa.Table.from_pylist(rows), d / f'validation-{f:05d}-of-00002.parquet')
  gidx = [7, 1, 9, 0, 5]
  cache = str(tmp_path / 'cache.pt')
  x = rc.load_val_images(str(tmp_path / 'val'), gidx, cache)
  assert torch.equal(x, val_images(str(tmp_path / 'val'), gidx))
  assert torch.equal(rc.load_val_images(str(tmp_path / 'val'), gidx, cache), x)          # from the cache
  assert np.array_equal(rc.val_class_index(str(tmp_path / 'val')), [0, 1, 2, 3, 4] * 2)


def test_compare_counts_against_only_beyond_floor_and_replicated(tmp_path):
  import argparse
  import json

  def rec(c, cond, l3, ref=0.5):
    return {'cls': c, 'kind': 'selected', 'stratum': 'dense', 'level': 10, 'condition': cond, 'set': 'A', 'PI': 0.1,
            'defined': True, 'F': 0.5, 'O': 0.1, 'O_whole': 2.0, 'auroc_far': 0.7, 'top1': 0.3,
            'P2': {'lambda3': l3, 'defined': True}, 'P2_whole': {'lambda3': ref}}

  def write(name, o):
    p = str(tmp_path / f'{name}.json')
    json.dump(o, open(p, 'w'))
    return p
  ids = range(30)
  rare = write('rare', {'run': 'A0', 'run_set': 'A', 'classes': [rec(c, 'rare', 0.2) for c in ids]})
  other = write('other', {'run': 'B0', 'run_set': 'B', 'classes': [rec(c, 'common', 0.5) for c in ids]})
  same = write('same', {'run': 'B0', 'run_set': 'B', 'classes': [rec(c, 'common', 0.2) for c in ids]})
  c0 = write('c0', {'run': 'C0', 'run_set': 'C', 'classes': [rec(c, 'common', 0.5) for c in ids]})
  out = str(tmp_path / 'cmp.json')
  args = dict(c0=[c0], e1=[], out=out)
  cr.cmd_compare(argparse.Namespace(rare=[rare, other], rare1=[], c1=[], e1r=[], **args))
  dec = json.load(open(out))['decision']
  assert dec['against_essay'] == [] and dec['against_pending_seed1'] == ['median lambda_3 difference < 0']
  # seed 1 without C1: the class-split reference (P2_whole) comes from E1r
  cr.cmd_compare(argparse.Namespace(rare=[rare, other], rare1=[rare, other], c1=[], e1r=[c0], **args))
  res = json.load(open(out))
  assert res['seed0']['P2']['floor_wilcoxon']['p'] < 1e-4 and res['seed1_reference'] == 'E1r'
  assert res['decision']['against_essay'] == ['median lambda_3 difference < 0']
  cr.cmd_compare(argparse.Namespace(rare=[rare, same], rare1=[rare, same], c1=[c0], e1r=[], **args))  # D = D0
  dec = json.load(open(out))['decision']
  assert dec['against_essay'] == [] and dec['against_candidates'][0]['floor_ok'] is False


def test_zeta_accepts_numpy_classes_and_fold_check_flags():
  z = torch.tensor([[1.0, 2.0, 6.0], [0.0, 3.0, 0.0]])
  for c in (2, np.int64(2)):
    assert torch.allclose(rc.zeta(z, c), torch.tensor([3.0, -1.0]))
  assert torch.allclose(rc.zeta(z, torch.tensor([2, 1])), torch.tensor([3.0, 2.0]))
  rec = {'C': torch.ones(3, 2, 2, dtype=torch.float16) * 4, 'C_scale': 4.0, 'Cb': torch.ones(3, 2, dtype=torch.float16),
         'Cb_scale': 1.0}
  cls = {'C': torch.tensor([[[2.0, 2.0], [2.0, 2.0]], [[1.0, 1.0], [1.0, 1.0]]]).half(), 'C_scale': 1.0,
         'Cb': torch.tensor([[2.0, 2.0], [1.0, 1.0]]).half(), 'Cb_scale': 1.0}
  chk = gg.fold_check({'m': rec}, {'m': cls}, np.array([0, 0, 1]), 2)
  assert chk['max_rel'] == 0.0 and chk['ok']
  cls['Cb'] = torch.tensor([[2.0, 2.0], [1.5, 1.0]]).half()
  assert not gg.fold_check({'m': rec}, {'m': cls}, np.array([0, 0, 1]), 2)['ok']


# ------------------------------------------------------------------------------------- protocol amendment 1 (compare)

def test_amendment_helpers():
  m = cr.mcnemar_exact([True] * 18 + [False] * 94, [False] * 18 + [True] + [False] * 93)
  assert (m['x_only'], m['y_only'], m['both']) == (18, 1, 0) and m['p_one_sided'] == pytest.approx(20 / 2 ** 19)
  assert cr.mcnemar_exact([True, None], [False, True])['n'] == 1
  wil = lambda z: {'p': 1e-4, 'z': z}
  assert cr.floor_agrees({'median': -0.1, 'floor_wilcoxon': wil(-3.0)})
  assert not cr.floor_agrees({'median': -0.1, 'floor_wilcoxon': wil(3.6)})      # the floor points the other way
  assert not cr.floor_agrees({'median': -0.1, 'floor_wilcoxon': {'p': 0.2, 'z': -1.0}})
  assert not cr.floor_agrees({'median': -0.1, 'floor_wilcoxon': None})
  assert cr.noise_state([1, 0, 0, 0], [-0.9], [0.5]) == 'uninformative'          # fit check fails: not a failure
  assert cr.noise_state([1, 1, 0.5], [], [0.5]) == 'uninformative'
  assert cr.noise_state([1, 1, 0.5], [-0.9], [0.5]) == 'pass'
  assert cr.noise_state([1, 1, 0.5], [0.9], [0.5]) == 'fail'
  assert cr.controls_state('uninformative', 'pass') == 'uninformative'
  assert cr.controls_state('uninformative', 'fail') == 'fail' and cr.controls_state('pass', 'pass') == 'pass'
  assert cr.controls_state('pass', 'uninformative') == 'uninformative'


def test_compare_amendment_1(tmp_path):
  import argparse
  import json
  rng = np.random.default_rng(1)
  ids = range(40)

  def rec(c, cond, l3, pi=-0.99, defined=None, sig=False, fo=0.5, top1=0.0, kind='selected', level=10):
    defined = c % 2 == 0 if defined is None else defined
    return {'cls': c, 'kind': kind, 'stratum': 'dense', 'level': level, 'condition': cond, 'set': 'A',
            'PI': pi + 0.001 * rng.normal() if defined else 0.9, 'defined': defined, 'F_pct': 0.01 if defined else 0.6,
            'F': -20.0 if defined else 1.0, 'O': 0.01, 'O_whole': 2.0, 'def_threshold': 15.0, 'auroc_far': 0.7,
            'top1': top1, 'signature': sig, 'pseudo_signature': False,
            'S1': {'RBpx_K': 0.6, 'RPp_K': 0.5, 'RBpx_w': 0.45 + 0.01 * rng.normal(), 'RPp_w': 0.5},
            'S2': {'sigma_stage0': 1.0 + 0.1 * rng.normal()},
            'P2': {'lambda3': l3 + 0.01 * rng.normal(), 'defined': True}, 'P2_whole': {'lambda3': 0.5},
            'P2_whole_fo': {'lambda3': fo}}

  def write(name, recs):
    path = str(tmp_path / f'{name}.json')
    json.dump({'run': name, 'classes': recs}, open(path, 'w'))
    return path
  noise = [rec(200 + k, 'rare', 0.4, top1=float(k == 0), kind='noise', level=None) for k in range(4)]
  rare = write('rare', [rec(c, 'rare', 0.4, sig=c < 10) for c in ids] + noise)
  c0 = write('c0', [rec(c, 'common', 0.5, sig=c < 3) for c in ids])
  shallow = write('shallow', [rec(c, 'common', 0.3) for c in ids])            # D0 below D: the floor points up
  deep = write('deep', [rec(c, 'common', 0.5) for c in ids])
  out = str(tmp_path / 'cmp.json')
  args = dict(c0=[c0], rare1=[], c1=[], e1=[], e1r=[], out=out)
  cr.cmd_compare(argparse.Namespace(rare=[rare, shallow], **args))
  res = json.load(open(out))
  s0, dec = res['seed0'], res['decision']
  # P1: primary on the defined classes; secondaries over all 40 and with 'no route' as 0
  assert s0['P1']['n'] == 20 and s0['P1_secondary']['raw_all']['n'] == 40
  assert s0['P1_secondary']['no_route_as_zero']['n'] == 40 and s0['P1_secondary']['F_pct_tails']['le_0.05'] == 20
  assert s0['P1_secondary']['symmetric_gate_2.5_97.5']['n'] == 20
  # P2: a negative median whose floor points the other way is not a count; the references are all reported
  assert s0['P2']['median'] < 0 and s0['P2']['floor_wilcoxon']['z'] > 0
  p2 = [i for i in dec['against_candidates'] if i['claim'] == 'median lambda_3 difference < 0'][0]
  assert p2['floor_ok'] is False and 'median lambda_3 difference < 0' not in dec['against_pending_seed1']
  assert {'P2_ref_first_order', 'P2_ref_own_m'} <= set(s0)
  # P1 and the signature excess also hold in C0: flagged, still pending
  assert dec['against_pending_seed1'] == ['median PI < 0', 'excess of packing signatures']
  assert any('median PI < 0' in f and 'common run' in f for f in dec['secondary_flags'])
  u = s0['universality']
  assert u['mcnemar_vs_pseudo']['x_only'] == 10 and u['mcnemar_vs_pseudo']['p_one_sided'] == pytest.approx(0.5 ** 10)
  assert u['common']['signatures'] == 3 and u['mcnemar_rare_vs_common']['x_only'] == 7
  # noise control: fit check fails -> uninformative, not failed
  assert s0['noise_control']['state'] == 'uninformative' and dec['controls_state']['seed0'] == 'uninformative'
  assert dec['note'].startswith('noise control uninformative') and dec['note'] != cr.CONTROL_NOTES['fail']
  # S1 weighted version and S2 with its floor
  assert s0['S1_RBpx_w_at_RPp_w_half']['n'] == 40 and s0['S2_rare_minus_C0']['sigma_stage0']['floor_wilcoxon']
  # the floor agrees, but the first-order reference is positive: pending with a secondary flag
  fo = write('rare_fo', [rec(c, 'rare', 0.4, fo=0.3) for c in ids])
  c0fo = write('c0_fo', [rec(c, 'common', 0.5, fo=0.3) for c in ids])
  cr.cmd_compare(argparse.Namespace(rare=[fo, deep], **{**args, 'c0': [c0fo]}))
  res = json.load(open(out))
  assert res['seed0']['P2_ref_first_order']['median'] > 0
  assert 'median lambda_3 difference < 0' in res['decision']['against_pending_seed1']
  assert any('references are not all negative' in f for f in res['decision']['secondary_flags'])
  # the result does not depend on the order of the files
  cr.cmd_compare(argparse.Namespace(rare=[deep, fo], **{**args, 'c0': [c0fo]}))
  assert json.load(open(out))['seed0'] == res['seed0']
