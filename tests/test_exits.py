"""experiments.convnext_exits, fixes of the adversarial review (protocol v2): fp16 corners that keep the split exact,
fp16 bounds of the own capture and the bias pushes, the unbiased S1 C1 band, the X1 primary population, the sd floor
at selection and in S2, class counts, left-censored entries, F2 at step 0, bootstrap p-values, missing input files,
Muon's D, atomic writes and the compare stage. CPU, small and fast; one toy pipeline run (as in
test_interval_split.py)."""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from experiments import convnext_exits as X
from rcdl.attribution import to_fp16_scaled
from rcdl.interval_split import MAPS, column_terms, increments, load_parts, own_bounds, own_terms

DT = torch.float64


# ------------------------------------------------------------------------------------- fp16 parts and their bounds

def fp16_pair(K=7, U=3, d=5, seed=0):
  """Increments of one interval from the 'true' fp64 parts and from their fp16 storage (compact records), the same
  model weights at both ends (so R absorbs the rounding), with storage bounds on the stored side."""
  g = torch.Generator().manual_seed(seed)
  r = lambda *s: torch.randn(*s, generator=g, dtype=DT)
  inc_true, inc_st = {}, {}
  for M in MAPS:
    W0, b0 = r(U, d) * 0.3, r(U) * 0.1
    init = {'W0': W0, 'b0': b0}
    pts = []
    for a, ab, sc in ((0.9, 0.97, 0.05), (0.8, 0.95, 0.07)):
      C, Cb = (r(K, U, d) * sc).float().double(), (r(K, U) * sc).float().double()    # fp32 values, as the trainer's
      W, bb = a * W0 + C.sum(0) + r(U, d) * 1e-5, ab * b0 + Cb.sum(0) + r(U) * 1e-5
      true = load_parts({'C': C, 'Cb': Cb, 'a': a, 'ab': ab, 'W0': W0, 'b0': b0}, W=W, b=bb, bounds=False)
      q, s = to_fp16_scaled(C.float())
      qb, sb = to_fp16_scaled(Cb.float())
      st = load_parts({'C': q, 'C_scale': s, 'Cb': qb, 'Cb_scale': sb, 'a': a, 'ab': ab, 't': 1}, init, W=W, b=bb)
      pts.append((true, st))
    inc_true[M] = increments(pts[0][0], pts[1][0])
    inc_st[M] = increments(pts[0][1], pts[1][1])
  return inc_true, inc_st


def test_corners_keep_the_split_and_the_bounds_hold_for_the_true_parts():
  """Swapping the stored fp16 parts for the true ones moves the X1 components within the bounds batch_components
  gives (F_rest's bound is F_own's, E's the writes' total) and leaves their sum, F_own + F_rest and writes + E
  unchanged; corner() moves components the same way, keeping the total (the old corners moved it by about the summed
  bounds of every group)."""
  K, U, d, S = 7, 3, 5, 6
  inc_true, inc_st = fp16_pair(K, U, d)
  g = torch.Generator().manual_seed(1)
  n = 1
  H = {M: torch.randn(S, d, generator=g, dtype=DT) for M in MAPS}
  q = {M: torch.rand(S, generator=g, dtype=DT) for M in MAPS}
  cols = torch.arange(S)
  z = torch.zeros(S, dtype=DT)
  ct_t, ct_s = column_terms(inc_true, H, q, n), column_terms(inc_st, H, q, n, bounds=True)
  vt, vs = X.make_view(ct_t, z, z, z, cols, S), X.make_view(ct_s, z, z, z, cols, S)
  xs = torch.tensor([0, 2])
  Qp, Qm = X.padded([[1, 3], [4]], xs, 'cpu')
  cat = torch.stack([X.category_index(0, {'P': [1, 3], 'rel': [5]}, K), X.category_index(2, {'P': [4]}, K)])
  ct, cs = X.batch_components(vt, xs, Qp, Qm, cat), X.batch_components(vs, xs, Qp, Qm, cat)
  bnd = X.batch_components(vs, xs, Qp, Qm, cat, bounds=True).numpy()
  diff = (ct - cs).numpy()
  assert np.abs(diff).max() > 0                                    # the storage error is real
  assert np.abs(diff.sum(1)).max() < 1e-12                         # the total is fixed
  iFo, iFr, iE = X.COMP.index('Fown_c'), X.COMP.index('Frest_c'), X.COMP.index('E_c')
  for o in (0, 1):
    assert np.abs(diff[:, iFo + o] + diff[:, iFr + o]).max() < 1e-13
    assert np.abs(diff[:, o:X.NWRITE:2].sum(1) + diff[:, iE + o]).max() < 1e-13
  assert (np.abs(diff) <= bnd + 1e-15).all()
  assert np.allclose(bnd[:, iFr:iFr + 2], bnd[:, iFo:iFo + 2])
  assert np.allclose(bnd[:, iE:iE + 2].sum(1), bnd[:, :X.NWRITE].sum(1))
  rest = (ct_s['F_bound'][:, 0].sum() - ct_s['F_bound'][0, 0]).item()   # the old F_rest bound: the others' sum
  assert rest > 2 * bnd[0, iFr]
  cs_np = cs.numpy()
  for pat in X.CORNERS:
    sg = X.corner_sign(pat)
    for sgn in (1, -1):
      y = X.corner(cs_np, bnd, sgn * sg)
      assert np.abs(y.sum(1) - cs_np.sum(1)).max() < 1e-12
      dy = y - cs_np
      for o in (0, 1):
        assert np.abs(dy[:, iFo + o] + dy[:, iFr + o]).max() < 1e-14
        assert np.abs(dy[:, o:X.NWRITE:2].sum(1) + dy[:, iE + o]).max() < 1e-14
      assert (dy[:, X.COMP.index('Phi_c'):X.COMP.index('Pi_P') + 1] == 0).all()
  assert np.allclose(X.corner(cs_np, bnd, X.corner_sign('all+')).mean(0),
                     X.corner(cs_np.mean(0), bnd.mean(0), X.corner_sign('all+')))     # linear: mean of corners


def test_own_bounds_is_the_diagonal_of_T_bound_and_holds():
  K = C = 5
  U, d = 3, 4
  inc_true, inc_st = fp16_pair(K, U, d, seed=3)
  g = torch.Generator().manual_seed(4)
  cm = {f'Hbar_{M}': torch.randn(U, C, d, generator=g, dtype=DT) for M in MAPS}
  cm |= {f'qbar_{M}': torch.rand(U, C, generator=g, dtype=DT) for M in MAPS}
  ob = own_bounds(inc_st, cm)
  for n in range(U):
    ct = column_terms(inc_st, {M: cm[f'Hbar_{M}'][n] for M in MAPS}, {M: cm[f'qbar_{M}'][n] for M in MAPS}, n,
                      bounds=True)
    assert (ob[n] - ct['T_bound'].diagonal()).abs().max() < 1e-15
  err = (own_terms(inc_st, cm) - own_terms(inc_true, cm)).abs()
  assert (err <= ob + 1e-16).all() and float(err.max()) > 0
  with pytest.raises(ValueError):
    own_bounds(inc_true, cm)


def test_bias_push_bounds_contain_the_fp32_push():
  """bias_pushes: the push from the stored fp16 bias parts is within pushb of the push from the unrounded parts."""
  K, U, C = 6, 4, 6
  g = torch.Generator().manual_seed(5)
  b = 'stage2.block0'
  true, recs = {}, {}
  for s, ab in ((0, 1.0), (10, 0.97), (20, 0.95)):
    true[s], recs[s] = {}, {}
    for M in MAPS:
      Cb = (torch.randn(K, U, generator=g, dtype=DT) * 0.03).float().double()
      q, sc = to_fp16_scaled(Cb.float())
      true[s][M] = Cb
      recs[s][f'{b}.pw_{M}'] = {'Cb': q, 'Cb_scale': sc, 'ab': ab}
  st = {'Abar': torch.randn(2, U, C, generator=g), 'Bbar': torch.randn(2, U, C, generator=g)}
  ctx = SimpleNamespace(run=SimpleNamespace(g16=[0, 10, 20], has_random=False, attr=lambda s, split: recs[s]),
                        stock={0: {b: st}, 10: {b: st}}, C=C)
  bp = X.bias_pushes(ctx, b)
  for j, (s1, s2) in ((1, (0, 10)), (2, (10, 20))):
    rho = recs[s2][f'{b}.pw_gate']['ab'] / recs[s1][f'{b}.pw_gate']['ab']
    for M in MAPS:
      exact = (true[s2][M] - rho * true[s1][M]).numpy()
      err = np.abs(bp['push'][(j, M)] - exact)
      assert (err <= bp['pushb'][(j, M)] * (1 + 1e-9)).all() and err.max() > 0


# --------------------------------------------------------------------------------------------- X3 S1 C1 band

def test_s1_c1_band_is_unbiased_when_the_exit_is_exchangeable_with_c1():
  """Without a random split, exit and C1 pushes drawn from one distribution (m = 3..6): the exit's leave-one-out label
  fractions minus the C1 labels average 0; the old rule (the exit against all m pushes, each C1 member against m - 1)
  is biased towards 'withdrawn'."""
  rng = np.random.default_rng(0)
  new, old = [], []
  for i in range(8000):
    m = 3 + i % 4
    v = rng.standard_t(3, m + 1)
    pe, pc = v[0], list(v[1:])
    lb = X.s1_c1_band(pe, pc)
    c1 = np.mean([x == 'withdrawn' for x in lb['labels_C1']])
    new.append(lb['label_frac']['withdrawn'] - c1)
    old.append(float(X.s1_label(pe, *np.percentile(pc, [25, 75])) == 'withdrawn') - c1)
  new, old = np.array(new), np.array(old)
  se = new.std() / np.sqrt(len(new))
  assert abs(new.mean()) < 4 * se
  assert old.mean() > 8 * old.std() / np.sqrt(len(old))
  assert X.s1_c1_band(0.0, [1.0, 2.0]) is None                     # m >= 3
  lb = X.s1_c1_band(0.0, [-1.0, 0.0, 1.0], be=2.0, bc=[0, 0, 0])
  assert lb['flip_fp16'] and not X.s1_c1_band(0.0, [-1.0, 0.0, 1.0], be=1e-9, bc=[1e-9] * 3)['flip_fp16']


def test_x3_population_uses_the_leave_one_out_fractions_and_reports_the_random_band_apart():
  recs = []
  for i in range(30):
    lb = X.s1_c1_band(float(i % 3) - 1, [-1.0, 0.0, 1.0, 2.0])
    recs.append(dict(lb, d='s0m1', b='stage2.block0', n=i, c=i, nC1=4, label_rand='withdrawn',
                     labels_C1_rand=['reversed'] * 4))
  recs.append({'d': 's0m1', 'b': 'stage2.block0', 'n': 99, 'c': 99, 'nC1': 4, 'skip': 'low_sd'})
  out = X.x3_population(SimpleNamespace(prm={'seed': 0, 'n_boot': 50}), recs)
  want = np.mean([r['label_frac']['withdrawn'] for r in recs[:30]])
  assert out['band'] == 'C1' and abs(out['withdrawn']['exits'] - want) < 1e-12
  assert out['withdrawn_randband']['exits'] == 1.0 and out['withdrawn_randband']['C1'] == 0.0
  assert out['n_low_sd'] == 1 and out['nC1'] == {4: 30} and out['flips']['n'] == 30


def test_x3_block_uses_the_c1_band_in_every_run_and_skips_low_sd_units():
  """x3_block: the S1 labels come from the C1 band (s1_c1_band) with and without a random split (the random band is an
  extra result), need m >= 3 C1 members, and a unit below the sd floor on the selection half is skipped."""
  b, g, U, K = 'stage2.block0', [0, 10, 20, 30], 2, 6
  rng = np.random.default_rng(4)
  push = {(j, M): rng.standard_normal((K, U)) for j in (1, 2, 3) for M in MAPS}
  pushR = {k: rng.standard_normal((K, U)) for k in push}
  bp = {'push': push, 'pushb': {k: np.full((K, U), 1e-6) for k in push}, 'med': {k: np.ones(U) for k in push},
        'sign': {j: {M: np.ones((2, U, K)) for M in MAPS} for j in (1, 2, 3)}}
  e = dict(b=b, n=1, c=0, pk=1, out=3, anchored=True, rep=False, C1=[1, 2, 3, 4])
  ev = [e, dict(e, c=5, C1=[1, 2]), dict(e, n=0)]
  ctx = SimpleNamespace(run=SimpleNamespace(g16=g, snaps=[]), events={'directions': {d: {'exits': ev} for d in X.DIRS}},
                        sd=lambda s, bb: np.array([[1e-12, 1.0]] * 2), floor=lambda s, h: 1e-3,
                        prm={'band_min': 3}, n_c=np.ones(K), n_r=np.ones(K))
  for pr in (None, pushR):
    out = X.x3_block(ctx, b, dict(bp, pushR=pr))
    r, few, low = out[:3]
    want = sum(0.5 * (push[(j, 'gate')][0, 1] + push[(j, 'value')][0, 1]) for j in (1, 2, 3))
    assert r['push'] == pytest.approx(want)
    lb = X.s1_c1_band(r['push'], r['push_C1'])
    assert r['band'] == 'C1' and r['label_frac'] == lb['label_frac'] and r['labels_C1'] == lb['labels_C1']
    assert r['push_bound'] == pytest.approx(3e-6) and not r['flip_fp16']
    assert 'label_frac' not in few and low['skip'] == 'low_sd'
    assert ('label_rand' in r) == (pr is not None) and ('label_rand' in few) == (pr is not None)


# ----------------------------------------------------------------------------------- events: floor and censoring

def test_find_events_drops_low_sd_units_and_counts_left_censored_entries():
  """A unit with the same ranks as another but a near-zero sd (dead) gives the same events without the floor and none
  with it (counted); entries in Top_n from the first checkpoint on are counted, not dropped silently."""
  rng = np.random.default_rng(1)
  T, U, C = 8, 6, 40
  steps = list(range(0, 80, 10))
  prm = X.defaults() | dict(P=3, best=2, gone=5, est_step=10, anchor_k=2, anchor_min=1)
  base = rng.standard_normal((T, U, C)).cumsum(0)
  base[:, :, 0] += 50                                               # class 0 on top from step 0 to T
  Rs = np.concatenate([base, 1e-9 * base], 1)                        # units U..2U-1: the same ranks, dead
  ex0, pe0, info0 = X.find_events(Rs, Rs, steps, prm)
  e = lambda L, ns: [(x['n'] % U, x['c'], x.get('pk', x.get('first'))) for x in L if x['n'] in ns]
  live, dead = range(U), range(U, 2 * U)
  assert e(ex0, live) and e(ex0, live) == e(ex0, dead) and e(pe0, live) == e(pe0, dead)
  low = X.low_sd({'b': Rs}, prm['sd_floor'])['b']
  assert low[:, U:].all() and not low[:, :U].any()
  ex, pe, info = X.find_events(Rs, Rs, steps, prm, low)
  assert not e(ex, dead) and e(ex, live) == e(ex0, live) and not e(pe, dead)
  assert info['low_sd_exits'] == len(e(ex0, dead)) and info0['low_sd_exits'] == 0
  top = X.rank_desc(Rs[:, :U]) <= prm['P']
  from0 = int(top.all(0).sum())
  assert from0 >= U and info['entries_from_step0']['persistent'] == 2 * from0        # both halves, before the floor
  assert info['low_sd_entries']['persistent'] == len(e(pe0, dead))
  assert info['entries_from_step0']['dropped'] == sum(x['entry'] is None for x in ex)
  assert X.sum_info([info, info])['low_sd_exits'] == 2 * info['low_sd_exits']


def test_s2_masks_a_dead_unit_instead_of_letting_it_dominate():
  """A near-dead unit (tiny sd, so O / sd is huge) would own the stage total and change every share; below the floor
  it is set to 0, and S2 equals the result without it."""
  b, g = 'stage2.block0', [0, 10, 20, 30, 40]
  cap = {1: (1.0, 1.0), 2: (1.0, 1.0), 3: (0.1, 1.0), 4: (0.1, 1.0)}
  ex = [dict(b=b, n=0, c=c, pk=1, out=3, anchored=True, rep=True, C1=[]) for c in (0, 1)]
  def ctx_of(U, sd):
    own = {b: {(g[j - 1], g[j], h): torch.tensor([[cap[j][0]] * 2 + [1.0], [cap[j][1]] * 2 + [1.0],
                                                  [1e-9] * 3][:U]) for j in cap for h in (0, 1)}}
    floor = lambda s, h: 1e-3 * float(np.median(sd[h]))
    return own, SimpleNamespace(run=SimpleNamespace(g16=g, blocks=[b], U={b: U}), sd=lambda s, bb: sd, floor=floor,
                                events={'directions': {d: {'exits': ex} for d in X.DIRS}},
                                prm=dict(carriers=1, n_boot=20, seed=0))
  ref = X.x3_s2(*reversed(ctx_of(2, np.ones((2, 2)))))
  got = X.x3_s2(*reversed(ctx_of(3, np.array([[1.0, 1.0, 1e-12]] * 2))))
  for d in X.DIRS:
    assert got[d]['n'] == ref[d]['n'] == 2
    for k in ('before', 'same', 'after'):
      assert got[d][k]['exits'] == ref[d][k]['exits']


# -------------------------------------------------------------------------------------------------- class counts

def test_class_counts_reads_int_per_class_counts_and_refuses_silent_fallbacks(tmp_path):
  C = 5
  run = SimpleNamespace(dir=str(tmp_path))
  a = lambda src, **p: argparse.Namespace(class_counts=src, prm=X.defaults() | p)
  np.save(tmp_path / 'counts.npy', np.array([732, 1300, 900, 1000, 1100], np.int64))
  n_c, n_r, info = X.class_counts(a(str(tmp_path / 'counts.npy')), run, C)
  assert n_c.tolist() == [732, 1300, 900, 1000, 1100] and n_r is None and info['kind'] == 'per_class'
  np.save(tmp_path / 'labels.npy', np.array([0, 0, 1, 2, 3, 4, 4, 4], np.int16))
  n_c, _, info = X.class_counts(a(str(tmp_path / 'labels.npy')), run, C)
  assert n_c.tolist() == [2, 1, 1, 1, 3] and info['kind'] == 'labels' and info['sum'] == 8
  np.save(tmp_path / 'bad.npy', np.array([0, 1, 7], np.int64))
  with pytest.raises(SystemExit):
    X.class_counts(a(str(tmp_path / 'bad.npy')), run, C)
  if os.path.exists('/dev/shm/imagenet256/labels.npy'):
    pytest.skip('a training label file is present on this machine')
  with pytest.raises(SystemExit):
    X.class_counts(a(''), run, C)
  n_c, _, info = X.class_counts(a('', equal_counts=True), run, C)
  assert (n_c == 1).all() and info['equal'] and info['kind'] == 'equal'


def test_class_counts_never_come_from_random_groups(tmp_path):
  """A random-split run (E1r) has rand_groups.npy (group ids per example): it gives n_r only; n_c comes from the class
  labels even when the group sizes differ from the class sizes."""
  C = 4
  run = SimpleNamespace(dir=str(tmp_path))
  np.save(tmp_path / 'rand_groups.npy', np.array([0, 0, 0, 0, 1, 2, 3, 3], np.int16))
  np.save(tmp_path / 'labels.npy', np.array([0, 1, 1, 2, 2, 2, 3, 3], np.int16))
  a = argparse.Namespace(class_counts=str(tmp_path / 'labels.npy'), prm=X.defaults())
  n_c, n_r, info = X.class_counts(a, run, C)
  assert n_c.tolist() == [1, 2, 3, 2] and n_r.tolist() == [4, 1, 1, 2] and info['source'].endswith('labels.npy')
  if not os.path.exists('/dev/shm/imagenet256/labels.npy'):
    with pytest.raises(SystemExit):                                  # no silent use of the group ids as classes
      X.class_counts(argparse.Namespace(class_counts='', prm=X.defaults()), run, C)


# ---------------------------------------------------------------------------------------- statistics helpers

def test_rank_ols_leaves_columns_without_spread_missing():
  rng = np.random.default_rng(2)
  B, n = 4, 30
  Xa = rng.standard_normal((B, n, 3))
  Xa[..., 1] = 0.0                                                  # e.g. the class parts' provenance at step 0
  Y = X.rank_cols(rng.standard_normal((B, n, 1)))[..., 0]
  coef = X.rank_ols(Y, Xa)
  assert np.isnan(coef[:, 1]).all() and np.isfinite(coef[:, [0, 2]]).all()
  ref = X.batched_ols(Y, X.rank_cols(Xa[..., [0, 2]]))
  assert np.allclose(coef[:, [0, 2]], ref)
  assert X.boot_median(np.r_[coef[:, 1], 1.0, 2.0, 3.0], np.arange(B + 3), 50, rng)['n'] == 3


def test_bootstrap_p_values_are_never_0_and_ignore_empty_draws():
  p, B = X.boot_p0(np.ones(1999))
  assert B == 1999 and p == pytest.approx(2 / 2000)
  d = np.r_[np.full(1000, np.nan), -np.ones(500), np.ones(500)]
  p, B = X.boot_p0(d)
  assert B == 1000 and p == pytest.approx(1.0)                      # nan draws do not deflate p
  p, B = X.boot_p0(np.full(10, np.nan))
  assert B == 0 and np.isnan(p)
  p, B = X.boot_p0(np.stack([np.ones(9), -np.ones(9)], 1))
  assert p.tolist() == pytest.approx([0.2, 0.2]) and B.tolist() == [9, 9]
  rng = np.random.default_rng(0)
  r = X.boot_ratio(np.ones(20), np.ones(20), np.arange(20), np.arange(20), 99, rng)
  assert r['p0'] == pytest.approx(2 / 100) and r['B'] == 99
  holm = X.holm({'a': r['p0'], 'b': 0.5})
  assert holm['a'] > 0


def test_x1_population_counts_missing_nulls():
  rng = np.random.default_rng(3)
  recs = []
  for i in range(12):
    miss = {'own': 'few_classes'} if i < 4 else ({'P': 'few_sets'} if i < 6 else {})
    z = {k: float(rng.standard_normal()) for k in ('own', 'P', 'Ent') if k not in miss}
    r = {'comp': rng.standard_normal(len(X.COMP)), 'ctrl_mean': rng.standard_normal(len(X.COMP)), 'c2_mean': None,
         'dR_c': -1.0, 'b': 'stage2.block0', 'n': i, 'c': i, 'nC1': 3, 'nC2': 0, 'label': 'unclassified', 'flips': {},
         'z': z, 'pct': {}, 'null_missing': miss, 'null_classes': {k: 5 if i == 7 else 50 for k in z}, 'check': 0.0}
    if not miss:
      r['most_negative'] = min(z, key=z.get)
    recs.append(r)
  out = X.x1_population(SimpleNamespace(prm={'seed': 0, 'n_boot': 20}), recs)
  assert out['null_missing'] == {'own': 4, 'P': 2, 'Ent': 0, 'rel': 0}
  assert out['labels_P_null_missing'] == 2 and out['most_negative_excluded'] == 6
  assert out['n_with_z'] == {'own': 8, 'P': 10, 'Ent': 12, 'rel': 0}
  assert sum(out['most_negative_standardized'].values()) == 6
  assert out['null_few_classes']['Ent'] == 1 and out['null_missing_reasons']['few_classes'] == 4


# --------------------------------------------------------------------------------- inputs, writes and compare

def fake_run(root, steps=100, rand_groups=False):
  root.mkdir(exist_ok=True)
  json.dump({'size': 'tiny', 'steps': steps, 'rand_groups': rand_groups}, open(root / 'config.json', 'w'))
  init = {f'stage2.block0.pw_{M}': {'W0': torch.zeros(2, 4), 'b0': torch.zeros(2)} for M in MAPS}
  torch.save(init, root / 'attr_init.pt')
  sd = {'head.weight': torch.zeros(3, 4), 'stages.2.0.pw_gate.weight': torch.zeros(2, 4)}
  for s in X.expected_grid({'steps': steps}):
    torch.save(sd, root / f'model_{s:05d}.pt')
    torch.save({}, root / f'attr_{s:05d}.pt')
  return root


def test_run_refuses_missing_checkpoints_attrR_and_full_states(tmp_path):
  root = fake_run(tmp_path / 'run')
  assert X.expected_grid({'steps': 100}) == [0, 50, 100]
  assert X.expected_grid({'steps': 25020, 'attr_steps': '0,2500:10000:2500,25020'}) == [0, 2500, 5000, 7500, 10000,
                                                                                         25020]
  run = X.Run(str(root))
  assert run.g16 == [0, 50, 100] and not run.has_random and run.blocks == ['stage2.block0']
  os.remove(root / 'attr_00050.pt')
  with pytest.raises(SystemExit, match='attr_00050.pt'):
    X.Run(str(root))
  assert X.Run(str(root), grid=[0, 100]).g16 == [0, 100]
  torch.save({}, root / 'attr_00050.pt')
  torch.save({}, root / 'attrR_00000.pt')
  with pytest.raises(SystemExit, match='attrR_00050.pt'):
    X.Run(str(root))
  assert not X.Run(str(root), no_random=True).has_random
  root2 = fake_run(tmp_path / 'run2', rand_groups=True)
  with pytest.raises(SystemExit, match='attrR'):
    X.Run(str(root2))
  hist, full = tmp_path / 'hist', tmp_path / 'full'
  hist.mkdir()
  full.mkdir()
  for s in (40, 60):
    torch.save({}, hist / f'snap_{s:05d}.pt')
  torch.save({}, full / 'full_00040.pt')
  run = X.Run(str(root), str(hist), str(full), no_random=True)
  assert run.snaps == [40] and run.snaps_no_full == [60]
  with pytest.raises(SystemExit, match='full_00060.pt'):
    X.require_full(argparse.Namespace(no_full=False), run)
  X.require_full(argparse.Namespace(no_full=True), run)
  assert X.Run(str(root), str(hist), None, no_random=True).snaps_no_full == [40, 60]
  a = argparse.Namespace(out=str(tmp_path), skip_integrity=False)
  with pytest.raises(SystemExit, match='1 snapshot'):
    X.usable_snaps(a, str(root), str(hist), str(full), run.snaps_avail)
  json.dump({'usable': [], 'checked': []}, open(tmp_path / 'integrity.json', 'w'))
  with pytest.raises(SystemExit, match='did not check'):
    X.usable_snaps(a, str(root), str(hist), str(full), run.snaps_avail)


def test_precond_under_muon_is_one_for_weights_and_biases():
  U, d = 2, 3
  v, vb = torch.full((U, d), 1e-6, dtype=DT), torch.full((U,), 1e-10, dtype=DT)
  attr = {f'b.pw_{M}': {'t': 2000, 'v': v, 'vb': vb} for M in MAPS}
  prm = {'d_window': 1500, 'd_min_step': 1000}
  for opt in ('muon', 'adamw'):
    run = SimpleNamespace(snaps=[2000], U={'b': U}, d={'b': d}, cfg={'opt': opt}, dev=torch.device('cpu'),
                          full=lambda x: {'attr': attr})
    D, src = X.precond(run, 2000, 'b', prm)
    assert src == 2000 and D.shape == (U, 2 * d + 2)
    if opt == 'muon':
      assert (D == 1).all()
    else:
      c2 = 1 - 0.999 ** 2000
      assert torch.allclose(D[:, d], 1 / ((vb / c2).sqrt() + 1e-8)) and float(D[:, d].min()) > 1e4
  assert 'muon' in X.D_RULE


def test_writes_are_atomic(tmp_path, monkeypatch):
  p = str(tmp_path / 'stock_00010.pt')
  def broken(obj, path):
    open(path, 'wb').write(b'PK\x03')                               # a kill in the middle of the write
    raise OSError('killed')
  monkeypatch.setattr(torch, 'save', broken)
  with pytest.raises(OSError):
    X.save_atomic({'x': 1}, p)
  assert not os.path.exists(p)
  monkeypatch.undo()
  with pytest.raises(TypeError):
    X.dump_json({'a': object()}, str(tmp_path / 'events.json'))
  assert not os.path.exists(tmp_path / 'events.json')
  X.save_atomic({'x': 1}, p)
  assert torch.load(p)['x'] == 1 and not os.path.exists(p + '.tmp')


def test_compare_writes_a_default_output_and_refuses_mismatched_runs(tmp_path, monkeypatch):
  def res(est, band='C1', sha='abc', steps=(0, 50)):
    keys = ('X1_own_excess_share', 'X5_loop_own')
    return {'steps': list(steps), 'band_source': {'X3_S1_primary': band}, 'class_counts': {'n_c_sha': sha},
            'primary': {'estimates': dict.fromkeys(keys, est), 'null': dict.fromkeys(keys, 0.0),
                        'ci': {'X1_own_excess_share': [0.1, 0.5], 'X5_loop_own': None},
                        'p': {'X1_own_excess_share': 0.01, 'X5_loop_own': 0.01}}}
  ref, oth = tmp_path / 'E1.json', tmp_path / 'sub' / 'E1r.json'
  oth.parent.mkdir()
  json.dump(res(0.3), open(ref, 'w'))
  json.dump(res(0.2), open(oth, 'w'))
  monkeypatch.setattr(sys, 'argv', ['convnext_exits', 'compare', '--ref', str(ref), '--other', str(oth)])
  X.main()
  out = json.load(open(tmp_path / 'sub' / 'exits_compare.json'))
  assert out['mismatch'] == [] and out['primaries']['X1_own_excess_share']['replicated'] is True
  json.dump(res(0.2, band='random', sha='def'), open(oth, 'w'))
  X.main()
  out = json.load(open(tmp_path / 'sub' / 'exits_compare.json'))
  assert out['mismatch'] == ['X3_S1_band', 'class_counts']
  assert out['primaries']['X1_own_excess_share']['replicated'] is None
  monkeypatch.setattr(sys, 'argv', sys.argv + ['--allow_mismatch'])
  X.main()
  assert json.load(open(tmp_path / 'sub' / 'exits_compare.json'))['primaries']['X1_own_excess_share']['replicated']
  monkeypatch.setattr(sys, 'argv', ['convnext_exits', 'compare', '--ref', str(ref)])
  with pytest.raises(SystemExit):
    X.main()


# ------------------------------------------------------------------------------------------ the toy pipeline

@pytest.fixture(scope='module')
def toy(tmp_path_factory):
  """The toy run of test_interval_split.py through every stage of the command line (one CPU thread)."""
  from test_interval_split import TOY_GRID, TOY_PARAMS, build_toy_run
  threads = torch.get_num_threads()
  torch.set_num_threads(1)
  root = tmp_path_factory.mktemp('toyrun_exits')
  argv = sys.argv
  try:
    build_toy_run(root)
    common = ['--run', str(root), '--out', str(root / 'exits'), '--history', str(root / 'history'), '--full',
              str(root / 'full'), '--probe', str(root / 'probe.pt'), '--visual', str(root / 'visual.npz'), '--wordnet',
              str(root / 'wordnet.npz'), '--batch', '16', '--device', 'cpu', '--grid', ','.join(map(str, TOY_GRID)),
              '--class_counts', str(root / 'labels.npy'), '--params', json.dumps(TOY_PARAMS)]
    for stage in ('integrity', 'stock', 'events', 'intervals', 'gradstate', 'tests'):
      sys.argv = ['convnext_exits', stage] + common
      X.main()
    yield root, common
  finally:
    sys.argv = argv
    torch.set_num_threads(threads)


def test_toy_outputs_record_the_fixes(toy):
  root, _ = toy
  ev = json.load(open(root / 'exits' / 'events.json'))
  assert ev['version'] == 3 and ev['has_random']
  for d in X.DIRS:
    c = ev['counts'][d]
    assert {'low_sd_exits', 'low_sd_entries', 'entries_from_step0'} <= set(c)
    assert c['entries_from_step0']['persistent'] > 0                 # the toy grid starts at step 0
  integ = json.load(open(root / 'exits' / 'integrity.json'))
  assert integ['checked'] == [10, 25, 40, 55] and integ['dropped_no_full'] == []
  res = json.load(open(root / 'exits' / 'exits_tests.json'))
  assert res['has_random'] and res['band_source']['X3_S1_primary'] == 'C1' and res['snapshots_dropped_no_full'] == []
  assert res['class_counts']['source'].endswith('labels.npy') and res['class_counts']['kind'] == 'labels'
  assert res['checks']['n_c_equal'] is True                        # the toy trains 16 images per class
  late = res['pooled']['late']['s0m1']
  sh = late['X1']['anchored_all']['excess_vs_C1'].get('shares')
  if sh:                                                           # the primary: selection-half population
    assert res['primary']['estimates']['X1_own_excess_share'] == sh['own']['est']
  assert {'anchored_replicated', 'turnover', 'turnover_replicated', 'low_sd_exits'} <= set(late['X1'])
  x1 = res['pooled']['all']['s0m1']['X1']['anchored_all']
  assert {'null_missing', 'n_with_z', 'most_negative_excluded', 'labels_P_null_missing'} <= set(x1)
  x2 = res['pooled']['all']['s0m1']['X2']
  assert x2['entries_from_step0']['persistent'] > 0 and 'zero_within_bound_own' in x2
  if 'own_diff' in x2:
    assert set(x2['own_diff']['fp16_corners']) == set(X.CORNERS) and x2['own_diff']['p0'] > 0
  s1 = res['pooled']['all']['s0m1']['X3_S1']
  assert s1['band'] == 'C1' and 'nC1' in s1 and 'n_labelled_randband' in s1
  x4 = res['X4']['s0m1']
  n_iv = len(res['steps']) - 1                                    # F2 has no rows at t1 = 0 (the first interval)
  assert x4['stage']['all']['F2']['n'] * n_iv == x4['stage']['all']['F1']['n'] * (n_iv - 1)
  assert x4['own_within_bound'] and 'F1_fp16' in x4['stage']['all']
  assert 'own_within_bound' in res['X5'] and 'n_low_sd' in res['X3_S2']['s0m1']
  for p in res['primary']['p'].values():
    assert p is None or p > 0


def test_toy_f2_columns_are_zero_at_step_0(toy):
  """The premise of the F2 fix on the toy run: at t1 = 0 the relatives', own and others' provenance columns are all 0
  (no class has written), so their coefficients are missing (nan), not an estimated 0."""
  root, _ = toy
  T = str(root)
  from test_interval_split import TOY_GRID, TOY_PARAMS
  a = argparse.Namespace(out=f'{T}/exits', visual=f'{T}/visual.npz', wordnet=f'{T}/wordnet.npz',
                         class_counts=f'{T}/labels.npy',
                         prm=X.defaults() | TOY_PARAMS, skip_integrity=False)
  run = X.Run(T, f'{T}/history', f'{T}/full', grid=list(TOY_GRID), snaps=X.usable_snaps(a, T, f'{T}/history',
                                                                                         f'{T}/full'))
  ctx = X.Ctx(a, run)
  b = 'stage2.block4'
  pv, _ = X.provenance(run, 0, b, ctx.stockh(0)[b], ctx.stock[0][b], torch.from_numpy(ctx.rel))
  assert float(pv[..., :3].abs().max()) == 0 and float(pv[..., 3].abs().max()) > 0
  zs = np.random.default_rng(0).standard_normal((run.U[b], ctx.C))
  Xr = np.stack([zs, zs ** 2] + [pv[0][:, :, i].numpy() for i in range(4)], -1)
  coef = X.rank_ols(X.rank_cols(np.random.default_rng(1).standard_normal((ctx.C, run.U[b], 1)))[..., 0],
                    np.moveaxis(Xr, 1, 0))
  assert np.isnan(coef[:, 2:5]).all() and np.isfinite(coef[:, [0, 1, 5]]).all()


def test_toy_integrity_refuses_history_without_full_states(toy, tmp_path, monkeypatch):
  root, common = toy
  argv = [x if x != str(root / 'full') else str(tmp_path) for x in common]
  argv[argv.index('--out') + 1] = str(tmp_path / 'out')
  monkeypatch.setattr(sys, 'argv', ['convnext_exits', 'integrity'] + argv)
  with pytest.raises(SystemExit, match='full_00010.pt'):
    X.main()
  monkeypatch.setattr(sys, 'argv', sys.argv + ['--no_full'])
  X.main()
  integ = json.load(open(tmp_path / 'out' / 'integrity.json'))
  assert integ['usable'] == [] and integ['dropped_no_full'] == [10, 25, 40, 55]
