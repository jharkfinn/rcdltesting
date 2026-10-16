"""experiments.convnext_bias_paths: the probe bias gradients equal autograd's bias gradients, and the splits add up."""

import numpy as np
import torch
import torch.nn.functional as F

from experiments import convnext_bias_paths as bp
from rcdl.convnext import ConvNeXt


def _model(n_classes=3):
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 1, 1, 1), n_classes=n_classes, layer_scale=0.5)
  for p in m.parameters():                     # non-trivial weights so that every path carries gradient
    p.data.normal_(0, 0.3)
  return m.eval()


def test_bias_gradients_and_splits():
  m = _model()
  x = torch.randint(0, 256, (6, 3, 64, 64), dtype=torch.uint8)
  y = torch.tensor([0, 0, 1, 1, 2, 2])
  P = bp.probe_paths(m, x, y, n_classes=3, batch=6, device='cpu')
  ref = _model()                               # plain model: autograd's bias gradients of the summed per-class loss
  mean, std = bp.MEAN[:, None, None], bp.STD[:, None, None]
  xb = (x.float() - mean) / std
  for c in range(3):
    ref.zero_grad()
    F.cross_entropy(ref(xb[y == c]), y[y == c], reduction='sum').backward()
    for name, blk in ref.blocks():
      np.testing.assert_allclose(P[name]['Bv_ce'][c] * 2, blk.pw_value.bias.grad.numpy(), rtol=1e-4, atol=1e-6)
      np.testing.assert_allclose(P[name]['Bg_ce'][c] * 2, blk.pw_gate.bias.grad.numpy(), rtol=1e-4, atol=1e-6)
  for name, a in P.items():
    for k in ('Bv', 'Bg'):
      np.testing.assert_allclose(a[f'{k}_pstar'] + a[f'{k}_on'] + a[f'{k}_off'], a[f'{k}_ce'], rtol=1e-4, atol=1e-6)
      np.testing.assert_allclose(a[f'{k}_lse'] + a[f'{k}_true'], a[f'{k}_ce'], rtol=1e-4, atol=1e-6)
      np.testing.assert_allclose(a[f'{k}_fixed'] + a[f'{k}_scale'], a[f'{k}_ce'], rtol=1e-4, atol=1e-6)
    assert np.abs(P[name]['Bv_scale']).max() > 1e-6  # the scale path carries gradient


def test_fixed_sigma_same_forward_other_gradient():
  """Inside fixed_sigma(), nn.LayerNorm and rcdl's LayerNorm2d both run the fixed-sigma form (same values, other
  gradient); outside, F.layer_norm is torch's again; keyword calls work and a module reload does not recurse."""
  import importlib
  from rcdl.convnext import LayerNorm2d
  orig = F.layer_norm
  torch.manual_seed(0)
  for mod, x in ((torch.nn.LayerNorm(8, eps=1e-6), torch.randn(4, 5, 8)),
                 (LayerNorm2d(8, eps=1e-6), torch.randn(2, 8, 3, 4))):
    with torch.no_grad():
      mod.weight.normal_()
      mod.bias.normal_()
    x.requires_grad_(True)
    y0 = mod(x)
    with bp.fixed_sigma():
      assert F.layer_norm is bp.layer_norm_fixed_sigma
      y1 = mod(x)
    assert F.layer_norm is orig
    torch.testing.assert_close(y0, y1, rtol=1e-5, atol=1e-5)
    g0, = torch.autograd.grad((y0 ** 3).sum(), x)
    g1, = torch.autograd.grad((y1 ** 3).sum(), x)    # backward after leaving the context keeps the fixed scale
    assert (g0 - g1).abs().max() > 1e-3
  x = torch.randn(3, 8)
  torch.testing.assert_close(bp.layer_norm_fixed_sigma(input=x, normalized_shape=(8,), eps=1e-6),
                             F.layer_norm(input=x, normalized_shape=(8,), eps=1e-6), rtol=1e-5, atol=1e-5)
  importlib.reload(bp)
  assert F.layer_norm is orig
  torch.nn.LayerNorm(8)(x)


def test_summarize_roles_and_maps():
  m = _model(6)
  x = torch.randint(0, 256, (12, 3, 64, 64), dtype=torch.uint8)
  y = torch.arange(6).repeat_interleave(2)
  P = bp.probe_paths(m, x, y, n_classes=6, batch=12, device='cpu')
  rel = np.zeros((6, 6), bool)
  rel[0, 1] = rel[1, 0] = True
  for name, a in P.items():
    s0, s1 = bp.summarize(a, top_n=2), bp.summarize(a, top_n=2, rel=rel)
    np.testing.assert_allclose(s0['ce_v_top20'] + s0['ce_g_top20'], s0['ce']['top20'], rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(s0['ce_v_rest'] + s0['ce_g_rest'], s0['ce']['rest'], rtol=1e-5, atol=1e-7)
    assert s1['rest_excludes_relatives'] and not s0['rest_excludes_relatives']
    assert s0['ce']['top20'] == s1['ce']['top20'] and 'relatives' in s1['ce']
    for k in ('lse', 'true', 'fixed', 'scale', 'pstar', 'on', 'off'):
      assert np.isfinite(s1[k]['top20'])
    np.testing.assert_allclose(s0['lse']['top20'] + s0['true']['top20'], s0['ce']['top20'], rtol=1e-4, atol=1e-6)


def test_scale_derivative_and_ablation():
  m = _model()
  x = torch.randint(0, 256, (6, 3, 64, 64), dtype=torch.uint8)
  y = torch.tensor([0, 0, 1, 1, 2, 2])
  P = bp.probe_paths(m, x, y, n_classes=3, batch=6, device='cpu')
  mean, std = bp.MEAN[:, None, None], bp.STD[:, None, None]
  xb = (x.float() - mean) / std
  name, blk = m.blocks()[1]
  eps = 1e-3
  with torch.no_grad():
    for n in (0, 3):
      ce = []
      for al in (1 - eps, 1 + eps):
        blk._alpha = torch.ones(6)
        blk._alpha[n] = al
        ce.append(F.cross_entropy(m(xb), y, reduction='none').view(3, 2).mean(1))
      blk._alpha = None
      np.testing.assert_allclose(((ce[1] - ce[0]) / (2 * eps)).numpy(), P[name]['S_ce'][:, n], rtol=2e-2, atol=1e-4)
  A = bp.ablate(m, x, y, P, n_top=1, n_rest=1, device='cpu')
  assert set(A) == {b for b, _ in m.blocks()}
  assert all(np.isfinite(v) for a in A.values() for v in a.values())


def test_rest_classes_leave_out_relatives():
  rel = np.zeros((10, 10), bool)
  rel[0, 1:5] = rel[1:5, 0] = True
  top = np.array([0, 9])
  for seed in range(5):
    r = bp.rest_classes(np.random.default_rng(seed), 10, top, rel, 3)
    assert len(set(r)) == 3 and not set(r) & {0, 9, 1, 2, 3, 4}
  r = bp.rest_classes(np.random.default_rng(0), 10, top, rel, 6)      # too few far classes: any non-top class
  assert len(set(r)) == 6 and not set(r) & {0, 9}
  r0 = np.random.default_rng(3).choice(np.setdiff1d(np.arange(10), top), 4, replace=False)
  assert (bp.rest_classes(np.random.default_rng(3), 10, top, None, 4) == r0).all()   # without rel: as before


def test_replay_history_group_is_left_out(tmp_path):
  """A replay's attribution has one group more than there are classes (history): summarize and convnext_bias_self
  use the class groups only, and bias_self reads atlas_prelim files."""
  from experiments import convnext_bias_self as bs
  m = _model(4)
  x = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8)
  y = torch.arange(4).repeat_interleave(2)
  P = bp.probe_paths(m, x, y, n_classes=4, batch=8, device='cpu')
  name = 'stage1.block0'
  rng = np.random.default_rng(0)
  w = {k: rng.normal(size=(5, 6)) for k in ('gate', 'value')}
  s5, s4 = bp.summarize(P[name], w, top_n=2), bp.summarize(P[name], {k: v[:4] for k, v in w.items()}, top_n=2)
  assert s5['rho_writes'] == s4['rho_writes'] and s5['writes_top20'] == s4['writes_top20']
  R, G, V = P[name]['u'].T, P[name]['g'].T, P[name]['v'].T
  rel = np.zeros((4, 4), bool)
  b5 = bs.block_self(w['gate'], w['value'], R, G, V, rel, np.ones(6), top_n=2)
  b4 = bs.block_self(w['gate'][:4], w['value'][:4], R, G, V, rel, np.ones(6), top_n=2)
  assert b5 == b4
  for s in (1000, 1500):
    torch.save({f'{name}.pw_{k}': {'Cb': torch.from_numpy(w[k] * s).half(), 'Cb_scale': 1.0} for k in w},
               tmp_path / f'attr_{s:05d}.pt')
    torch.save({name: {'R': torch.from_numpy(R), 'Gm': torch.from_numpy(G), 'Vm': torch.from_numpy(V)}},
               tmp_path / f'atlas_prelim_{s:05d}.pt')
  steps, cb, atl = bs.load_run(str(tmp_path))
  assert steps == [1000, 1500] and cb[1500][f'{name}.pw_gate'].shape == (5, 6)
