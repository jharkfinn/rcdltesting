"""experiments.convnext_bias_paths: the probe bias gradients equal autograd's bias gradients, and the splits add up."""

import numpy as np
import torch
import torch.nn.functional as F

from experiments import convnext_bias_paths as bp
from rcdl.convnext import ConvNeXt


def _model():
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 1, 1, 1), n_classes=3, layer_scale=0.5)
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
      np.testing.assert_allclose(a[f'{k}_lse'] - a[f'{k}_own'], a[f'{k}_ce'], rtol=1e-4, atol=1e-6)
      np.testing.assert_allclose(a[f'{k}_fixed'] + a[f'{k}_scale'], a[f'{k}_ce'], rtol=1e-4, atol=1e-6)
    assert np.abs(P[name]['Bv_scale']).max() > 1e-6  # the scale path carries gradient


def test_fixed_sigma_same_forward_other_gradient():
  x = torch.randn(4, 5, 8, requires_grad=True)
  w, b = torch.randn(8), torch.randn(8)
  y0 = F.layer_norm(x, (8,), w, b, 1e-6)
  bp._FIX_SIGMA[0] = True
  try:
    y1 = F.layer_norm(x, (8,), w, b, 1e-6)
  finally:
    bp._FIX_SIGMA[0] = False
  torch.testing.assert_close(y0, y1, rtol=1e-5, atol=1e-5)
  g0, = torch.autograd.grad((y0 ** 3).sum(), x)
  g1, = torch.autograd.grad((y1 ** 3).sum(), x)
  assert (g0 - g1).abs().max() > 1e-3


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
