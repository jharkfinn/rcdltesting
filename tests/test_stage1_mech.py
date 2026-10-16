"""experiments.convnext_stage1_mech: the stream decomposition adds up, the X1 push is convnext_bias_paths' pole push, and
X2 reduces to the baseline for an empty set and for alpha = 1."""

import numpy as np
import torch

from experiments import convnext_bias_paths as bp
from experiments import convnext_stage1_mech as sm
from experiments import convnext_unit_roles as ur
from rcdl.convnext import ConvNeXt

NAME = 'stage1.block1'


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


def test_stream_decomposition():
  m = _model()
  xs, y = _data()
  X = ur.block_inputs(m, xs, 1, 1, 8, 'cpu')
  blk = m.stages[1][1]
  geo = sm.write_geometry(blk)
  with torch.no_grad():
    g, v, u = sm.block_guv(blk, X)
    sp = sm.stream_parts(blk, X, u, geo)
    torch.testing.assert_close(sp['s_in'] + sp['s_own'] + sp['s_part'], sp['s_out'], rtol=1e-5, atol=1e-5)
    omega = ur.block_out(blk, X, u) - X                                       # (B, d, H, W): all writes and the bias
    for n in (0, 4):
      w = ur.write_dir(blk, n)
      e = (w / w.norm()).view(1, -1, 1, 1)
      s_part = ((omega - u[..., n][:, None] * w.view(1, -1, 1, 1)) * e).sum(1).flatten(1)
      torch.testing.assert_close(sp['s_part'][..., n], s_part, rtol=1e-5, atol=1e-5)
      torch.testing.assert_close(sp['s_in'][..., n], (X * e).sum(1).flatten(1), rtol=1e-5, atol=1e-5)
      torch.testing.assert_close(sp['s_own'][..., n], u[..., n].flatten(1) * w.norm(), rtol=1e-5, atol=1e-6)


def test_x1_push_is_the_pole_push():
  xs, y = _data()
  P = bp.probe_paths(_model(), xs, y, n_classes=4, batch=8, device='cpu')[NAME]   # its own model: probe_paths hooks it
  m = _model()
  X = ur.block_inputs(m, xs, 1, 1, 8, 'cpu')
  summ, raw = sm.x1_block(m, NAME, X, y, batch=8, n_sample=8, top_n=1)
  for k, ref in (('R', 'u'), ('G', 'g'), ('V', 'v')):
    np.testing.assert_allclose(raw[k], P[ref].T, rtol=1e-4, atol=1e-5)
  pole = -(P['Bv_ce'] * raw['pa_v'][None] + P['Bg_ce'] * raw['pa_g'][None])
  np.testing.assert_allclose(raw['pole'], pole, rtol=1e-4, atol=1e-6)
  for k in sm.BIN_VARS:                                                       # the bins split the push exactly
    np.testing.assert_allclose(raw[f'bin_push_{k}'].sum(-1) / raw['n_img'][:, None], raw['pole'], rtol=1e-6, atol=1e-9)
    assert (raw[f'bin_cnt_{k}'].sum(-1) == raw['n_img'][:, None] * raw['P']).all()
  ref = bp.summarize(P, top_n=1)['ce']
  np.testing.assert_allclose([summ['pole_top20'], summ['pole_rest']], [ref['top20'], ref['rest']], rtol=1e-4, atol=1e-6)
  for role in ('top20', 'rest'):
    np.testing.assert_allclose(sum(summ[f'push_by_sout_bin_{role}']), summ[f'pole_{role}'], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(summ[f's_in_{role}'] + summ[f's_own_{role}'] + summ[f's_part_{role}'],
                               summ[f's_out_{role}'], rtol=1e-4, atol=1e-5)
  assert all(np.isfinite(summ[k]) for k in ('frac_m1_top20', 'frac_m4_top20', 'role_offset_by_sout', 'rms_sin_all'))


def test_x2_baseline_cases():
  m = _model()
  xs, y = _data()
  X = ur.block_inputs(m, xs, 1, 1, 8, 'cpu')
  stats = sm.class_stats(m, NAME, X, y, 4, 8)
  _, raw = sm.x1_block(m, NAME, X, y, stats=stats, batch=8, n_sample=8, top_n=1)
  by_class = [np.flatnonzero(y.numpy() == c) for c in range(4)]
  for n in (1, 5):                                                           # X2's baseline push is X1's
    push = sm.x2_push(m, NAME, X, y, by_class, np.arange(4), n, {'base': None}, raw['pa_v'][n], raw['pa_g'][n], 8)
    np.testing.assert_allclose(push['base'], raw['pole'][:, n], rtol=1e-5, atol=1e-8)
  kw = dict(stats=stats, n_top=1, n_other=2, batch=8)
  s, rows = sm.x2_block(m, NAME, X, y, cos_hi=2.0, **kw)                  # every set empty: the baseline
  assert s['n_empty_splus'] == s['n_empty_sminus'] == s['units'] and s['units_splus'] == 0
  for r in rows:
    for k in ('splus', 'sminus', 'rand'):
      for al in (0, 0.5):
        assert r[f'top20_{k}_a{al:g}'] == r['top20_base'] and r[f'rest_{k}_a{al:g}'] == r['rest_base']
  s1, rows1 = sm.x2_block(m, NAME, X, y, cos_hi=-1.0, alphas=(1.0, 0.0), **kw)   # S+ and S-: every other unit
  assert s1['n_empty_splus'] == 0
  for r in rows1:
    assert r['n_splus'] == r['n_sminus'] == 5
    for k in ('splus', 'sminus', 'rand'):                                    # alpha = 1: the baseline
      np.testing.assert_allclose(r[f'top20_{k}_a1'], r['top20_base'], rtol=1e-5, atol=1e-7)
      np.testing.assert_allclose(r[f'rest_{k}_a1'], r['rest_base'], rtol=1e-5, atol=1e-7)
  assert any(abs(r['top20_splus_a0'] - r['top20_base']) > 1e-3 for r in rows1)   # zeroing the others changes it
  np.testing.assert_allclose([r['top20_base'] for r in rows], [r['top20_base'] for r in rows1], rtol=1e-6)
