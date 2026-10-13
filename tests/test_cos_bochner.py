"""The Cos-Bochner layer (rcdl.layers.CosBochnerLayer) is the random-feature form of the growing RBF layer:
its features reproduce the RBF kernel, one SGD step at rate η B equals the RBF write rule in feature space, and a frozen
Cos-Bochner model follows the growing RBF model."""

import math

import torch
import torch.nn.functional as F

from rcdl.layers import CosBochnerLayer, RCDLConfig, rms_norm
from rcdl.models import CifarRCDLClassifier
from rcdl.train import create_rcdl_optimizer, rcdl_train_step


def test_features_approximate_rbf_kernel():
  d, T, D = 16, 1.0, 1 << 17
  layer = CosBochnerLayer(RCDLConfig(temperature=T, cos_width=D), d, 1, torch.Generator().manual_seed(0))
  g = torch.Generator().manual_seed(1)
  q, k = rms_norm(torch.randn(6, d, generator=g)), rms_norm(torch.randn(6, d, generator=g))
  approx = layer.features(q) @ layer.features(k).T
  exact = torch.exp(-torch.cdist(q, k).pow(2) / (2 * T * math.sqrt(d)))
  assert (approx - exact).abs().max().item() < 0.02         # random-feature error ~ 1 / sqrt(2D) = 0.002 per entry


def test_sgd_step_equals_rbf_write_rule():
  """With lr η B, SGD on A adds Σ_b phi(q_b) v_b^T, v_b = -η B ḡ_b (ḡ_b: gradient of the batch-mean loss w.r.t. s_b)."""
  eta, B, d = 0.3, 4, 24
  cfg = RCDLConfig(kernel_type='cos', hidden_dim=d, num_layers=0, num_classes=5, batch_size=B, temperature=1.0,
                   cos_width=512, optimizer='sgd', learning_rate=eta * B, global_init_seed=2)
  m = CifarRCDLClassifier(cfg)
  opt = create_rcdl_optimizer(m, cfg)
  g = torch.Generator().manual_seed(3)
  for _ in range(3):
    x, y = torch.randn(B, d, generator=g) * 2, torch.randint(0, 5, (B,), generator=g)
    layer = m.output_layer
    A0 = layer.A.detach().clone()
    q = rms_norm(x)
    s = (layer.features(q) @ A0).requires_grad_(True)
    gbar, = torch.autograd.grad(F.cross_entropy(s, y), s)
    expect = A0 + layer.features(q).T @ (-eta * B * gbar)
    rcdl_train_step(m, opt, x, y)
    assert torch.allclose(layer.A.detach(), expect, atol=1e-6, rtol=1e-5)
  assert [n for n, p in m.named_parameters()] == ['output_layer.A']   # frozen features: only A is trained


def test_frozen_cos_bochner_follows_growing_rbf():
  """L = 0 on the same stream: wide frozen random features + SGD (lr η B) track the growing RBF layer's outputs.

  The remaining difference is the random-feature error alone (same-rows check, measured: 0.8% at D = 2^18, 0.5% at
  2^20 for T = 4; 8.3% at T = 1, where the kernel between unrelated points is only ~0.0075, so the per-entry error
  ~1/sqrt(2D) is large relative to it)."""
  eta, B, d, steps = 0.3, 4, 24, 10
  base = dict(hidden_dim=d, num_layers=0, num_classes=5, batch_size=B, temperature=4.0, eta=eta, global_init_seed=4)
  rbf = CifarRCDLClassifier(RCDLConfig(kernel_type='rbf', db_size=B * steps, **base))
  cos_cfg = RCDLConfig(kernel_type='cos', cos_width=1 << 18, optimizer='sgd', learning_rate=eta * B, **base)
  cos = CifarRCDLClassifier(cos_cfg)
  opt = create_rcdl_optimizer(cos, cos_cfg)
  g = torch.Generator().manual_seed(5)
  for _ in range(steps):
    x, y = torch.randn(B, d, generator=g) * 2, torch.randint(0, 5, (B,), generator=g)
    rcdl_train_step(rbf, None, x.clone(), y)
    rcdl_train_step(cos, opt, x.clone(), y)
  tx = torch.randn(16, d, generator=torch.Generator().manual_seed(6)) * 2
  rbf.eval(), cos.eval()
  with torch.no_grad():
    a, b = rbf(tx), cos(tx)
  rel = ((a - b).norm() / a.norm()).item()
  assert rel < 0.02, rel


def test_trained_features_receive_gradients():
  cfg = RCDLConfig(kernel_type='cos', hidden_dim=24, num_layers=1, num_classes=5, batch_size=4, cos_width=64,
                   cos_train_features=True, optimizer='adam', learning_rate=1e-2)
  m = CifarRCDLClassifier(cfg)
  opt = create_rcdl_optimizer(m, cfg)
  omega0 = m.layers[0].omega.detach().clone()
  g = torch.Generator().manual_seed(7)
  for _ in range(3):
    rcdl_train_step(m, opt, torch.randn(4, 24, generator=g), torch.randint(0, 5, (4,), generator=g))
  assert not torch.equal(m.layers[0].omega.detach(), omega0)
