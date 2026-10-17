"""experiments.convnext_stage1_mech: the stream decomposition adds up, the X1 push is convnext_bias_paths' pole push and
its bins, quadrants and readout r add up to it, every X2 config reduces to the baseline at scale 1 and matches an
independent computation otherwise, and X4's readers add up to the readout r, each matches a finite difference that only
it sees, and X4's total is X1's readout."""

import numpy as np
import torch
import torch.nn.functional as F

from experiments import convnext_bias_paths as bp
from experiments import convnext_stage1_mech as sm
from experiments import convnext_unit_roles as ur
from rcdl.convnext import ConvNeXt

NAME = 'stage1.block1'
NAME0 = 'stage1.block0'


def _model():
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 2, 1, 1), n_classes=4, layer_scale=0.5)
  for p in m.parameters():
    p.data.normal_(0, 0.3)
  return m.eval()


def _data():
  g = torch.Generator().manual_seed(1)
  xs = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8, generator=g)
  return xs, torch.arange(4).repeat_interleave(2)


def _rest_logits(m, i, j, x_out):
  """Logits from block (i, j)'s output."""
  x = sm.stage_rest(m, i, j, x_out)
  for down, st in zip(m.downsample[i + 1:], m.stages[i + 1:]):
    x = st(down(x))
  return m.head(m.norm(x.mean((2, 3))))


def test_stream_decomposition():
  m = _model()
  xs, y = _data()
  for i, j in ((1, 0), (1, 1)):
    X = ur.block_inputs(m, xs, i, j, 8, 'cpu')
    blk = m.stages[i][j]
    geo = sm.write_geometry(blk)
    with torch.no_grad():
      g, v, u = sm.block_guv(blk, X)
      x_out = ur.block_out(blk, X, u)
      x_end = sm.stage_rest(m, i, j, x_out)
      sp = sm.stream_parts(blk, X, u, geo, x_end)
      torch.testing.assert_close(sp['s_in'] + sp['s_own'] + sp['s_part'] + sp['s_bias'], sp['s_out'], rtol=1e-5,
                                 atol=1e-5)
      omega = x_out - X                                                       # all writes and the bias
      bias = (blk.gamma * blk.pw_down.bias).view(1, -1, 1, 1)
      for n in (0, 4):
        w = ur.write_dir(blk, n)
        e = (w / w.norm()).view(1, -1, 1, 1)
        s_part = ((omega - u[..., n][:, None] * w.view(1, -1, 1, 1) - bias) * e).sum(1).flatten(1)
        torch.testing.assert_close(sp['s_part'][..., n], s_part, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sp['s_bias'][..., n], (bias * e).sum(1).flatten(1).expand_as(s_part))
        torch.testing.assert_close(sp['s_in'][..., n], (X * e).sum(1).flatten(1), rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(sp['s_own'][..., n], u[..., n].flatten(1) * w.norm(), rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(sp['s_later'][..., n], ((x_end - x_out) * e).sum(1).flatten(1), rtol=1e-5,
                                   atol=1e-5)
      if j == len(m.stages[i]) - 1:
        assert sp['s_later'].abs().max() == 0


def test_x1_push_is_the_pole_push():
  xs, y = _data()
  P = bp.probe_paths(_model(), xs, y, n_classes=4, batch=8, device='cpu')[NAME]   # its own model: probe_paths hooks it
  m = _model()
  X = ur.block_inputs(m, xs, 1, 1, 8, 'cpu')
  rel = np.zeros((4, 4), bool)
  rel[0, 1] = rel[1, 0] = rel[2, 3] = rel[3, 2] = True
  summ, raw = sm.x1_block(m, NAME, X, y, rel=rel, batch=8, n_sample=8, top_n=1)
  for k, ref in (('R', 'u'), ('G', 'g'), ('V', 'v')):
    np.testing.assert_allclose(raw[k], P[ref].T, rtol=1e-4, atol=1e-5)
  pole = -(P['Bv_ce'] * raw['pa_v'][None] + P['Bg_ce'] * raw['pa_g'][None])
  np.testing.assert_allclose(raw['pole'], pole, rtol=1e-4, atol=1e-6)
  np.testing.assert_allclose(sum(raw[f'push_{q}'] for q in sm.QUADS), raw['push'], rtol=1e-6, atol=1e-9)
  roles = sm.roles_of(raw['top'], 4, rel)
  for ri, role in enumerate(sm.ROLES):                                       # the role bins split the push exactly
    role_sum = np.where(roles[role], raw['pole'], 0).sum(0)                  # (U,): sum of the role's class means
    for k in sm.BIN_VARS:
      np.testing.assert_allclose(raw[f'bin_push_{k}'][ri].sum(-1), role_sum, rtol=1e-6, atol=1e-9)
      np.testing.assert_allclose(raw[f'bin_cnt_{k}'][ri].sum(-1), raw['n_classes_role'][ri] * raw['P'], rtol=1e-9)
    for t in sm.BIN_T:
      np.testing.assert_allclose(raw[f'bin2_{t}_sout_sin'].sum(-1), raw[f'bin_{t}_sout'], rtol=1e-6, atol=1e-9)
      np.testing.assert_allclose(raw[f'bin2_{t}_sout_u'].sum(-2), raw[f'bin_{t}_u'], rtol=1e-6, atol=1e-9)
  ref = bp.summarize(P, top_n=1, rel=rel)['ce']
  np.testing.assert_allclose([summ[f'pole_{r}'] for r in sm.ROLES], [ref[r] for r in sm.ROLES], rtol=1e-4, atol=1e-6)
  for role in sm.ROLES:
    for k in sm.BIN_VARS:
      np.testing.assert_allclose(np.nansum(summ[f'push_by_{k}_bin_{role}']), summ[f'pole_{role}'], rtol=1e-5,
                                 atol=1e-7)
    np.testing.assert_allclose(sum(summ[f'pole_{q}_{role}'] for q in sm.QUADS), summ[f'pole_{role}'], rtol=1e-5,
                               atol=1e-7)
    np.testing.assert_allclose(summ[f's_in_{role}'] + summ[f's_own_{role}'] + summ[f's_part_{role}']
                               + summ[f's_bias_{role}'], summ[f's_out_{role}'], rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(summ[f's_out_{role}'] + summ[f's_later_{role}'], summ[f's_end_{role}'], rtol=1e-4,
                               atol=1e-5)
  # the readout r = e . dCE/dx_out, from autograd on the block's output (another path than dCE/du / |w_n|)
  blk = m.stages[1][1]
  geo = sm.write_geometry(blk)
  with torch.no_grad():
    g, v, u = sm.block_guv(blk, X)
  x_out = ur.block_out(blk, X, u).detach().requires_grad_(True)
  go, = torch.autograd.grad(F.cross_entropy(_rest_logits(m, 1, 1, x_out), y, reduction='sum'), [x_out])
  wimg = 1 / np.bincount(y.numpy())[y.numpy()]
  ridx = np.select([roles['top20'], roles['relatives']], [0, 1], 2)                      # (C, U)
  for n in range(6):
    r = (go * geo['E'][n].view(1, -1, 1, 1)).sum(1).flatten(1).numpy()                 # (B, P)
    ka = sm._kappa(g[..., n], v[..., n], float(raw['pa_v'][n]), float(raw['pa_g'][n])).abs().flatten(1).numpy()
    for ri in range(3):
      sel = ridx[y.numpy(), n] == ri
      np.testing.assert_allclose(raw['bin_rw_sout'][ri, n].sum(), (wimg[sel, None] * ka[sel] * r[sel]).sum(),
                                 rtol=1e-4, atol=1e-8)
  for k in ('frac_m1_top20', 'frac_m4_top20', 'role_offset_by_sout', 'role_offset_r_by_sout', 'rms_sin_all',
            'write_eff_rank', 'r0_top20', 'b_excl_top20', 'b_own_top20', 'slope_r_by_u_within_sout_rest'):
    assert np.isfinite(summ[k]), k
  assert 1 <= summ['write_eff_rank'] <= 8


def test_x2_configs():
  m = _model()
  xs, y = _data()
  X = ur.block_inputs(m, xs, 1, 1, 8, 'cpu')
  stats = sm.class_stats(m, NAME, X, y, 4, 8)
  _, raw = sm.x1_block(m, NAME, X, y, stats=stats, batch=8, n_sample=8, top_n=1)
  by_class = [np.flatnonzero(y.numpy() == c) for c in range(4)]
  blk = m.stages[1][1]
  geo = sm.write_geometry(blk)
  with torch.no_grad():
    g, v, u = sm.block_guv(blk, X)
  S = np.array([0, 2, 3])
  for n in (1, 5):
    pa = (raw['pa_v'][n], raw['pa_g'][n])
    cfgs = {'base': None, 'u1': ('units', {1: S}, 1.0), 'e1': ('e_only', S, 1.0), 'p1': ('perp_only', S, 1.0),
            'b1': ('beta', 1.0), 'e0': ('e_only', S, 0.0), 'p0': ('perp_only', S, 0.0), 'b13': ('beta', 1.3),
            'u0': ('units', {1: S}, 0.0)}
    push, ce = sm.x2_push(m, NAME, X, y, by_class, np.arange(4), n, cfgs, *pa, 8)
    np.testing.assert_allclose(push['base'], raw['pole'][:, n], rtol=1e-5, atol=1e-8)   # X2's baseline is X1's
    for k in ('u1', 'e1', 'p1', 'b1'):                                       # scale 1: the baseline
      np.testing.assert_allclose(push[k], push['base'], rtol=1e-5, atol=1e-8)
      np.testing.assert_allclose(ce[k], ce['base'], rtol=1e-6)
    kap = sm._kappa(g[..., n], v[..., n], *(float(t) for t in pa)).flatten(1)
    e = geo['E'][n].view(1, -1, 1, 1)
    with torch.no_grad():
      out = ur.block_out(blk, X, u)
      part = (u[..., S] @ geo['M'][S, n])[:, None] * e
      whole = (u[..., S] @ geo['W'][S]).permute(0, 3, 1, 2)                   # S's writes
    for k, xo in (('b13', out + 0.3 * (out * e).sum(1, keepdim=True) * e), ('e0', out - part),
                  ('p0', out - (whole - part)), ('u0', out - whole)):
      xo = xo.detach().requires_grad_(True)                                  # direct readout at the changed state
      cei = F.cross_entropy(_rest_logits(m, 1, 1, xo), y, reduction='none')
      go, = torch.autograd.grad(cei.sum(), [xo])
      r = (go * e).sum(1).flatten(1)
      ref = -(kap * r * geo['norms'][n]).sum(1).view(4, 2).mean(1)
      np.testing.assert_allclose(push[k], ref.numpy(), rtol=1e-4, atol=1e-7)
      np.testing.assert_allclose(ce[k], cei.detach().view(4, 2).mean(1).numpy(), rtol=1e-5)
    assert np.abs(push['u0'] - push['base']).max() > 1e-3 * np.abs(push['base']).max()
  kw = dict(stats=stats, n_top=1, n_other=2, batch=8)
  s, rows = sm.x2_block(m, NAME, X, y, cos_hi=2.0, betas=(1.0,), **kw)         # every set empty: base and beta only
  assert s['units_splus'] == 0 and s['n_empty_splus'] == s['units'] and np.isnan(s['d_splus_vs_ctrl_a0'])
  for r in rows:
    assert r['n_splus'] == r['n_sminus'] == r['n_later'] == 0 and 'top20_splus_a0' not in r
    np.testing.assert_allclose(r['top20_beta1'], r['top20_base'], rtol=1e-5, atol=1e-7)
  s1, rows1 = sm.x2_block(m, NAME, X, y, cos_hi=-1.0, alphas=(1.0, 0.0), betas=(), **kw)   # S+, S-: every other unit
  assert sum(r['n_act'] > 0 for r in rows1) >= 4
  for r in rows1:
    assert r['n_splus'] + r['n_sminus'] == (5 if r['n_act'] else 0) and r['n_later'] == 0   # the stage's last block
    for k in ('splus', 'sminus'):
      if r[f'n_{k}']:
        np.testing.assert_allclose(r[f'top20_{k}_a1'], r['top20_base'], rtol=1e-5, atol=1e-7)
        np.testing.assert_allclose(r[f'rest_{k}_a1'], r['rest_base'], rtol=1e-5, atol=1e-7)
  assert any(abs(r.get('top20_splus_a0', r['top20_base']) - r['top20_base']) > 1e-3 for r in rows1)
  np.testing.assert_allclose([r['top20_base'] for r in rows], [r['top20_base'] for r in rows1], rtol=1e-6)


def test_x2_later_partners():
  """Later-block partners: their mean write along e_n matches a forward from that block's own input, and scaling them
  matches the full model with that block's units scaled (convnext_bias_paths.unit_hooks)."""
  m = _model()
  xs, y = _data()
  X = ur.block_inputs(m, xs, 1, 0, 8, 'cpu')
  X1 = ur.block_inputs(m, xs, 1, 1, 8, 'cpu')
  stats = sm.class_stats(m, NAME0, X, y, 4, 8)
  top, pa_v, pa_g, live = bp.pole_coefs(*stats, 1)
  geos = {j2: sm.write_geometry(m.stages[1][j2]) for j2 in (0, 1)}
  n = 2
  ib = torch.from_numpy(np.flatnonzero(y.numpy() == top[n, 0]))
  c, size, _, n_act = sm.partner_writes(m, 1, 0, X[ib], n, geos, 8)
  with torch.no_grad():
    un = sm.block_guv(m.stages[1][0], X[ib])[2].flatten(1, 2)[..., n]
    act = (un >= 0.5 * un.amax(1, keepdim=True)) & (un.amax(1, keepdim=True) > 0)
    u1 = ur.block_units(m.stages[1][1], X1[ib]).flatten(1, 2)
    ref = u1[act].mean(0) * (geos[1]['W'] @ geos[0]['E'][n])
  assert n_act == int(act.sum()) > 0
  np.testing.assert_allclose(c[1], ref.numpy(), rtol=1e-4, atol=1e-6)
  S = np.array([1, 4])
  by_class = [np.flatnonzero(y.numpy() == k) for k in range(4)]
  push, _ = sm.x2_push(m, NAME0, X, y, by_class, np.arange(4), n, {'x': ('units', {1: S}, 0.5)}, pa_v[n], pa_g[n], 8)
  m2, store = _model(), {}
  bp.unit_hooks(m2, store)
  a = torch.ones(6)
  a[S] = 0.5
  m2.stages[1][1]._alpha = a
  xb = (xs.float() - bp.MEAN[:, None, None]) / bp.STD[:, None, None]
  F.cross_entropy(m2(xb), y, reduction='sum').backward()
  u_s, g_s, v_s = store[NAME0]
  k_ = sm._kappa(g_s[..., n], v_s[..., n], float(pa_v[n]), float(pa_g[n]))
  ref = -(k_ * u_s.grad[..., n]).flatten(1).sum(1).view(4, 2).mean(1)
  np.testing.assert_allclose(push['x'], ref.detach().numpy(), rtol=1e-4, atol=1e-7)
  s, rows = sm.x2_block(m, NAME0, X, y, stats=stats, n_top=1, n_other=2, batch=8, cos_hi=-1.0, alphas=(1.0, 0.0),
                        betas=())
  assert s['units_later'] == sum(r['n_act'] > 0 for r in rows) > 0
  for r in rows:
    if r['n_later']:
      np.testing.assert_allclose(r['top20_later_a1'], r['top20_base'], rtol=1e-5, atol=1e-7)
  assert any(abs(r.get('top20_later_a0', r['top20_base']) - r['top20_base']) > 1e-3 * abs(r['top20_base'])
             for r in rows)


def test_partner_sets_and_size_matched_controls():
  cos = {0: np.array([1.0, 0.95, -0.95, 0.1, 0.2, -0.3, 0.0, 0.6, 0.98]), 1: np.array([0.99, 0.3, -0.92])}
  c = {0: np.array([1.0, 0.5, 0.4, 0.1, 0.1, 0.1, 0.1, 0.1, -0.2]), 1: np.array([0.3, 0.1, 0.2])}
  size = {0: np.array([1.0, 2.0, 0.3, 1.9, 0.31, 5.0, 0.0, 2.0, 0.6]), 1: np.array([1.0, 1.1, 1.0])}
  live = {0: np.ones(9, bool), 1: np.ones(3, bool)}
  sets = sm.x2_sets(0, 0, cos, c, size, live, np.random.default_rng(0))
  assert list(sets['splus'][0]) == [1, 2]            # unit 2: cos < 0 but a negative u there, so it raises the level
  assert list(sets['sminus'][0]) == [8]               # cos > 0 but lowers the level
  assert list(sets['ctrl_splus'][0]) == [3, 4]        # |cos| < 0.5, size > 0, closest log size: 2.0 -> 1.9, 0.3 -> 0.31
  assert list(sets['ctrl_sminus'][0]) == [4]          # 0.6 -> 0.31 (log distance 0.66) rather than 1.9 (1.15)
  assert list(sets['later'][1]) == [0, 2] and list(sets['ctrl_later'][1]) == [1]
  live[0][3] = False
  assert list(sm.x2_sets(0, 0, cos, c, size, live, np.random.default_rng(0))['ctrl_splus'][0]) == [4, 5]


def test_geometry_stats():
  m = _model()
  blk = m.stages[1][0]
  r = sm.geometry_stats(blk)
  W = torch.stack([ur.write_dir(blk, n) for n in range(6)]).detach()
  sv = torch.linalg.svdvals(W)
  np.testing.assert_allclose(r['write_eff_rank'], float(sv.sum() ** 2 / (sv ** 2).sum()), rtol=1e-5)
  with torch.no_grad():
    blk.pw_down.weight[:] = blk.pw_down.weight[:, :1]                         # every unit writes the same direction
  r1 = sm.geometry_stats(blk)
  np.testing.assert_allclose(r1['write_eff_rank'], 1.0, rtol=1e-5)
  assert r1['share_cos_hi'] == 1.0


def _model4(dtype=torch.float32, last_skip=True):
  """Three stage-1 blocks, two blocks in each later stage, widths that change at the downsamples; last_skip=False: the
  last block without a residual connection (as tiny_crush13)."""
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 12, 10), units=(6, 6, 6, 6), depths=(1, 3, 2, 2), n_classes=4, layer_scale=0.5,
               last_skip=last_skip)
  for p in m.parameters():
    p.data.normal_(0, 0.3)
  return m.to(dtype).eval()


def _stage1_out(m, xs, b):
  """Stage-1 block b's output for the images xs (in the model's dtype)."""
  dt = m.head.weight.dtype
  x = (xs.to(dt) - bp.MEAN[:, None, None].to(dt)) / bp.STD[:, None, None].to(dt)
  with torch.no_grad():
    return m.stages[1][:b + 1](m.downsample[1](m.stages[0](m.downsample[0](x))))


def _dirs(m, b):
  blk = m.stages[1][b]
  W = torch.stack([ur.write_dir(blk, n) for n in range(blk.pw_down.in_features)]).detach()
  return W / W.norm(dim=1, keepdim=True)


def _proj(t, E):
  return t.permute(0, 2, 3, 1).flatten(1, 2) @ E.T                                 # (B, P, U)


def _fd_loss(m, b, x_b, y, d, reader, loss):
  """L with the perturbation d of x_b (stage-1 block b's output) seen by one reader only: pure skips carry it to that
  reader's input (through a downsample D as D(x + d) - D(x)), the reader's block gets it in its input and its skip
  does not pass it on; head: the pool reads it (nothing if a block without skip is in the way)."""
  x, dd = x_b, d
  for t in range(1, len(m.stages)):
    if t > 1:
      xd = m.downsample[t](x)
      dd, x = (None if dd is None else m.downsample[t](x + dd) - xd), xd
    for l, blk in enumerate(m.stages[t]):
      if t == 1 and l <= b:
        continue
      if dd is not None and reader == f'stage{t}.block{l}':
        x, dd = (blk(x + dd) - dd if blk.skip else blk(x + dd)), None    # x + f(x + dd), or f(x + dd)
      else:
        x, dd = blk(x), (dd if blk.skip else None)
  if reader == 'head':
    x = x if dd is None else x + dd
  else:
    assert dd is None, reader
  return sm.X4_LOSS[loss](m.head(m.norm(x.mean((2, 3)))), y)


def test_x4_readers_add_up_to_the_readout():
  """For every stage-1 block and unit, the readers' parts add up to r = e . dCE/dx_out from autograd (another graph),
  and L_true's parts plus logsumexp's are CE's, reader by reader."""
  m = _model4(torch.float64)
  xs, y = _data()
  x0 = _stage1_out(m, xs, 0)
  terms = sm.x4_terms(m, 1, 0, x0, y, ('ce', 'true', 'lse'))
  expect = ['stage2.block0', 'stage2.block1', 'stage3.block0', 'stage3.block1', 'head']
  for b in range(3):
    x_out = _stage1_out(m, xs, b).requires_grad_(True)
    go, = torch.autograd.grad(F.cross_entropy(_rest_logits(m, 1, b, x_out), y, reduction='sum'), [x_out])
    g, T = terms['ce']
    torch.testing.assert_close(g[b], go, rtol=1e-10, atol=1e-12)
    rd = sm.x4_readers(1, b, 0, g, T)
    assert list(rd) == [f'stage1.block{j}' for j in range(b + 1, 3)] + expect
    E = _dirs(m, b)
    r = _proj(go, E)
    parts = {R: _proj(t, E) for R, t in rd.items()}
    torch.testing.assert_close(sum(parts.values()), r, rtol=1e-9, atol=1e-11)
    assert all(p.abs().max() > 1e-6 for p in parts.values())                    # every reader reads something
    lt, ll = (sm.x4_readers(1, b, 0, *terms[k]) for k in ('true', 'lse'))      # true + lse = ce, reader by reader
    for R in rd:
      torch.testing.assert_close(lt[R] + ll[R], rd[R], rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(terms['true'][0][b] + terms['lse'][0][b], g[b], rtol=1e-9, atol=1e-12)
  tb = sm.x4_terms(m, 1, 1, _stage1_out(m, xs, 1), y, ('ce',))['ce']           # starting later: the same terms
  for R, t in sm.x4_readers(1, 1, 1, *tb).items():
    torch.testing.assert_close(t, sm.x4_readers(1, 1, 0, *terms['ce'])[R], rtol=1e-10, atol=1e-12)


def test_x4_single_readers_by_finite_differences():
  """Perturb x_b by eps e_n at one position of one image and let only reader R see it: (L(+eps) - L(-eps)) / 2 eps
  is r_R there (float64), for CE and L_true; also with a last block without skip (its reader is entered by every path
  that reaches it, and head reads nothing)."""
  xs, y = _data()
  eps = 1e-5
  for skip, b, n, img, (h, w) in ((True, 0, 1, 3, (2, 5)), (True, 1, 4, 6, (7, 0)), (True, 2, 0, 1, (4, 4)),
                                  (False, 0, 2, 5, (3, 3))):
    m = _model4(torch.float64, skip)
    x_b = _stage1_out(m, xs, b)
    E = _dirs(m, b)
    d = torch.zeros_like(x_b)
    d[img, :, h, w] = eps * E[n]
    for loss in ('ce', 'true'):
      rd = sm.x4_readers(1, b, b, *sm.x4_terms(m, 1, b, x_b, y, (loss,))[loss])
      tot = 0.0
      with torch.no_grad():
        for R, t in rd.items():
          fd = float(_fd_loss(m, b, x_b, y, d, R, loss) - _fd_loss(m, b, x_b, y, -d, R, loss)) / (2 * eps)
          ref = float(t[img, :, h, w] @ E[n])
          np.testing.assert_allclose(fd, ref, rtol=1e-6, atol=1e-9, err_msg=f'{b} {loss} {R}')
          tot += fd
        if not skip:
          assert rd['head'].abs().max() == 0 and rd['stage3.block1'].abs().max() > 1e-6
        x_p = x_b.clone()
        x_p[img, :, h, w] += eps * E[n]
        x_m = x_b.clone()
        x_m[img, :, h, w] -= eps * E[n]
        L = lambda x_: sm.X4_LOSS[loss](_rest_logits(m, 1, b, x_), y)
        np.testing.assert_allclose(tot, float(L(x_p) - L(x_m)) / (2 * eps), rtol=1e-6, atol=1e-9)


def test_x4_fixed_sigma():
  """Under fixed sigma the readers add up to the fixed-sigma readout (autograd under bias_paths.fixed_sigma), which
  differs from the full one (the scale path carries gradient)."""
  m = _model4(torch.float64)
  xs, y = _data()
  terms = sm.x4_terms(m, 1, 0, _stage1_out(m, xs, 0), y, ('ce', 'true'), sigma=True)
  for b in range(3):
    x_out = _stage1_out(m, xs, b).requires_grad_(True)
    with bp.fixed_sigma():
      z = _rest_logits(m, 1, b, x_out)
    gf, = torch.autograd.grad(F.cross_entropy(z, y, reduction='sum'), [x_out])
    g0, = torch.autograd.grad(F.cross_entropy(_rest_logits(m, 1, b, x_out), y, reduction='sum'), [x_out])
    assert (gf - g0).abs().max() > 1e-3 * g0.abs().max()
    g, T = terms['ce']
    torch.testing.assert_close(g[b], gf, rtol=1e-9, atol=1e-12)
    torch.testing.assert_close(sum(sm.x4_readers(1, b, 0, g, T).values()), gf, rtol=1e-9, atol=1e-11)
    gt, Tt = terms['true']
    torch.testing.assert_close(sum(sm.x4_readers(1, b, 0, gt, Tt).values()), gt[b], rtol=1e-9, atol=1e-11)


def test_x4_block_matches_x1():
  """x4_block's total is X1's readout (same edges, weights, roles, live units); the readers, the groups and the
  fixed-sigma pass add up; the control direction is a unit vector orthogonal to e."""
  m = _model4()
  xs, y = _data()
  rel = np.zeros((4, 4), bool)
  rel[0, 1] = rel[1, 0] = rel[2, 3] = rel[3, 2] = True
  X0 = ur.block_inputs(m, xs, 1, 0, 3, 'cpu')
  names = ['stage1.block2', 'stage1.block0', 'stage1.block1']
  kw = dict(rel=rel, batch=3, n_sample=8, top_n=1, seed=2)
  r4 = sm.x4_block(m, names, X0, y, sigma=True, **kw)
  assert list(r4) == sorted(names)
  for j, name in enumerate(sorted(names)):
    X = ur.block_inputs(m, xs, 1, j, 3, 'cpu')
    s1, raw1 = sm.x1_block(m, name, X, y, **kw)
    s4, raw4 = r4[name]
    np.testing.assert_array_equal(raw4['edges_sout'], raw1['edges']['sout'])
    np.testing.assert_array_equal(raw4['live'], raw1['live'])
    np.testing.assert_allclose(raw4['bin_w'], raw1['bin_w_sout'], rtol=1e-6)
    np.testing.assert_allclose(raw4['bin_rw_ce']['total'], raw1['bin_rw_sout'], rtol=1e-4, atol=1e-7)
    ce = s4['ce']
    np.testing.assert_allclose(ce['role_offset_r_total_by_sout'], s1['role_offset_r_by_sout'], rtol=1e-4, atol=1e-6)
    for role in ('top20', 'relatives', 'rest'):
      np.testing.assert_allclose(ce[f'r_total_by_sout_bin_{role}'], s1[f'r_by_sout_bin_{role}'], rtol=1e-4,
                                 atol=1e-6)
    assert s4['readers'][:2 - j] == [f'stage1.block{j2}' for j2 in range(j + 1, 3)]
    assert s4['groups']['stage1_later'] == s4['readers'][:2 - j] and s4['groups']['head'] == ['head']
    for k in ('ce', 'true', 'ce_fixed', 'true_fixed'):
      d = s4[k]
      assert s4[f'sum_err_{k}'] < 1e-5
      for f in ('role_offset_r_{}_by_sout', 'r0_{}_top20', 'r0_{}_rest'):
        np.testing.assert_allclose(sum(d[f.format(R)] for R in s4['readers']), d[f.format('total')], rtol=1e-5,
                                   atol=1e-7)
        np.testing.assert_allclose(sum(d[f.format(R)] for R in s4['groups']), d[f.format('total')], rtol=1e-5,
                                   atol=1e-7)
      for key in ('role_offset_r_perp_by_sout', 'role_offset_abs_r_perp_by_sout', 'role_offset_abs_r_total_by_sout',
                  'r0_head_top20', 'r0_stage2_rest'):
        assert np.isfinite(d[key]), (k, key)
    Ep, E = raw4['e_perp'], sm.write_geometry(m.stages[1][j])['E'].numpy()
    np.testing.assert_allclose(np.abs((Ep * E).sum(1)).max(), 0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(Ep, axis=1), 1, rtol=1e-6)
  r4b = sm.x4_block(m, ['stage1.block1'], X0, y, **kw)                          # alone: the same numbers
  np.testing.assert_allclose(r4b['stage1.block1'][0]['ce']['role_offset_r_stage2_by_sout'],
                             r4['stage1.block1'][0]['ce']['role_offset_r_stage2_by_sout'], rtol=1e-6)
  assert 'ce_fixed' not in r4b['stage1.block1'][0]
