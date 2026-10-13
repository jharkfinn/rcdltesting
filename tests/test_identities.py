"""Numerical checks of the paper's mathematical claims (float64, tiny dimensions).

Each test names the paper section it checks. Notation: q query, k key, v value, w scalar weight, ṽ = w v weighted
value, s prediction, g = ∇_ŷ L output gradient, η / β learning rates, σ_k^2 / σ_v^2 Gaussian variances, T temperature.
"""

import math

import numpy as np
import pytest
import torch

from rcdl.layers import age_coefficients, rbf_attention, rms_norm

torch.set_default_dtype(torch.float64)


def wsm(q, keys, wvals, weights):
  """Discrete weighted softmax with weighted values (Eq. 6, T sqrt(d) absorbed): Σ e^<k,q> ṽ / Σ w e^<k,q>."""
  e = torch.exp(q @ keys.T)
  return (e @ wvals) / (e @ weights)


def test_linear_layer_dual_form():
  """Sec. 2, Eq. 1: a linear layer trained by SGD inside a deep net equals W_0 q + Σ_t v_t <x_t, q>."""
  g = torch.Generator().manual_seed(0)
  w1, w2 = torch.randn(5, 4, generator=g), torch.randn(3, 5, generator=g)
  w1_0, lr, xs, vs = w1.clone(), 0.05, [], []
  for _ in range(20):
    x, y = torch.randn(4, generator=g), torch.randn(3, generator=g)
    w1_ = w1.clone().requires_grad_(True)
    h = w1_ @ x
    h.retain_grad()
    loss = ((w2 @ torch.tanh(h) - y) ** 2).sum()
    loss.backward()
    xs.append(x)
    vs.append(-lr * h.grad)
    w1 = w1 - lr * w1_.grad
  q = torch.randn(4, generator=g)
  dual = w1_0 @ q + sum(v * (x @ q) for x, v in zip(xs, vs))
  assert torch.allclose(w1 @ q, dual, atol=1e-12)


def test_pdk_layer_dual_form_explicit_features():
  """Sec. 3.1 / App. A: SGD on an explicit feature-space weight equals growing (k_t = x_t, v_t = -η g_t)."""
  g = torch.Generator().manual_seed(1)
  feat = lambda x: torch.cat([torch.ones(1), math.sqrt(2) * x, (x[:, None] * x[None, :]).flatten()])  # (1+x.y)^2
  kern = lambda a, b: (1 + a @ b) ** 2
  W, lr, keys, vals = torch.zeros(2, 1 + 3 + 9), 0.1, [], []
  for _ in range(15):
    x, y = torch.randn(3, generator=g), torch.randn(2, generator=g)
    out_primal = W @ feat(x)
    out_dual = sum((v * kern(k, x) for k, v in zip(keys, vals)), torch.zeros(2))
    assert torch.allclose(out_primal, out_dual, atol=1e-10)
    grad_out = 2 * (out_primal - y)
    W = W - lr * grad_out[:, None] * feat(x)[None, :]
    keys.append(x)
    vals.append(-lr * grad_out)


def test_rbf_on_sphere_is_scaled_exponential_kernel():
  """Code-level identity (not stated in the paper): with RMS-normalized q and k (||q||^2 = ||k||^2 = d), the RBF
  kernel exp(-||q-k||^2 / (2 T sqrt d)) = exp(-sqrt(d)/T) exp(<q,k> / (T sqrt d)). The 'RBF' layer is therefore an
  unnormalized softmax-kernel layer with a constant scale exp(-sqrt(d)/T)."""
  g = torch.Generator().manual_seed(2)
  d, T = 64, 3.0
  q = rms_norm(torch.randn(5, d, generator=g), eps=0.0)
  k = rms_norm(torch.randn(7, d, generator=g), eps=0.0)
  v = torch.randn(7, 4, generator=g)
  lhs = rbf_attention(q, k, v, T)
  rhs = math.exp(-math.sqrt(d) / T) * torch.exp(q @ k.T / (T * math.sqrt(d))) @ v
  assert torch.allclose(lhs, rhs, rtol=1e-10)


def test_gaussian_mixture_induces_weighted_softmax_1d():
  """Sec. 3.2: the continuous softmax functional of a Gaussian-mixture weight equals the discrete weighted softmax,
  independent of σ_k, σ_v (d_in = d_out = 1, numerical quadrature)."""
  wi, ki, vi = np.array([0.3, 1.2, 0.5]), np.array([-0.4, 0.2, 0.9]), np.array([1.0, -2.0, 0.5])
  kk, vv = np.meshgrid(np.linspace(-12, 12, 1601), np.linspace(-12, 12, 1601), indexing='ij')
  for sk, sv in ((0.3, 0.5), (0.8, 1.5)):
    dens = sum(w * np.exp(-(kk - k) ** 2 / (2 * sk ** 2) - (vv - v) ** 2 / (2 * sv ** 2)) for w, k, v in zip(wi, ki, vi))
    for q in (-1.0, 0.3, 1.7):
      cont = (dens * np.exp(kk * q) * vv).sum() / (dens * np.exp(kk * q)).sum()
      disc = (wi * np.exp(ki * q) * vi).sum() / (wi * np.exp(ki * q)).sum()
      assert abs(cont - disc) < 1e-6


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_functional_gradient_tuple_is_descent_with_exact_rate(seed):
  """App. C.3: functional GD w <- w - η ∇_w J equals appending the tuple
     k_new = σ_k^2 q_t,  w_new = η <s_t, g_t> / D_t,  ṽ_new = -η σ_v^2 g_t / D_t,  D_t = Σ_i w_i e^<k_i, q_t>,
  and the loss decreases at rate ||∇_w J||^2_H = e^{σ_k^2 ||q_t||^2} (σ_v^2 ||g_t||^2 + <g_t, s_t>^2) / D_t^2.
  The 1/D_t factor is the one the paper's experiments drop (App. E.3)."""
  g = torch.Generator().manual_seed(seed)
  n, d_in, d_out, sk2, sv2 = 6, 3, 2, 0.7, 1.3
  keys, weights = torch.randn(n, d_in, generator=g), torch.rand(n, 1, generator=g) + 0.2
  wvals = weights * torch.randn(n, d_out, generator=g)
  q, y = 0.5 * torch.randn(d_in, generator=g), torch.randn(d_out, generator=g)
  loss = lambda s: 0.5 * ((s - y) ** 2).sum()
  s0 = wsm(q, keys, wvals, weights)
  g_t = s0 - y
  D_t = (torch.exp(q @ keys.T) @ weights).item()
  rate = math.exp(sk2 * (q @ q).item()) * (sv2 * (g_t @ g_t).item() + (g_t @ s0).item() ** 2) / D_t ** 2
  for eta in (1e-4, 1e-5):
    k_new = (sk2 * q)[None]
    w_new = torch.tensor([[eta * (s0 @ g_t).item() / D_t]])
    v_new = (-eta * sv2 * g_t / D_t)[None]
    s1 = wsm(q, torch.cat([keys, k_new]), torch.cat([wvals, v_new]), torch.cat([weights, w_new]))
    slope = (loss(s1) - loss(s0)).item() / eta
    assert abs(slope + rate) < 50 * eta * rate + 1e-9


def test_gauge_invariance_and_locality():
  """App. C.6: shifting values by ξ shifts the output by ξ (Prop. C.2), and the gauge-fixed update only appends
  one tuple (Thm. C.3) with w_new = η α_t ||g||^2 > 0."""
  g = torch.Generator().manual_seed(3)
  n, d_in, d_out = 8, 4, 3
  keys, weights = torch.randn(n, d_in, generator=g), torch.rand(n, 1, generator=g) + 0.1
  wvals = weights * torch.randn(n, d_out, generator=g)
  qs = torch.randn(5, d_in, generator=g)
  xi = torch.randn(d_out, generator=g)
  assert torch.allclose(wsm(qs, keys, wvals + weights * xi, weights), wsm(qs, keys, wvals, weights) + xi)

  q_t, g_t = torch.randn(d_in, generator=g), torch.randn(d_out, generator=g)
  eta, beta, alpha = 0.3, 0.7, 0.4
  s_t = wsm(q_t[None], keys, wvals, weights)[0]
  b_t = (-(s_t @ g_t) / (g_t @ g_t) + alpha) * g_t
  w_new = eta * alpha * (g_t @ g_t)
  assert w_new > 0
  # Route 1: to the virtual gauge, append (w_new, q_t, -β g), back to the canonical gauge.
  v_virtual = torch.cat([wvals + weights * b_t, (-beta * g_t)[None]])
  w_all = torch.cat([weights, torch.tensor([[w_new]])])
  v_route1 = v_virtual - w_all * b_t
  # Route 2: Theorem C.3, a single local append.
  v_final = (-w_new * alpha - beta + eta * alpha * (s_t @ g_t)) * g_t
  v_route2 = torch.cat([wvals, v_final[None]])
  assert torch.allclose(v_route1, v_route2, atol=1e-12)


@pytest.mark.parametrize('mu,gamma', [(0.9, 0.0), (0.0, 0.05), (0.8, 0.05)])
def test_momentum_and_decay_dual_exact_rule(mu, gamma):
  """App. D: SGD with momentum μ and decoupled decay γ on a linear layer equals the growing layer whose stored value
  written at step j is scaled by (1-γ)^a Σ_{m=0}^{a} (μ/(1-γ))^m, a = t - j (age_rule='exact')."""
  g = torch.Generator().manual_seed(4)
  d_in, d_out, lr, steps = 3, 2, 0.1, 25
  W, vel = torch.zeros(d_out, d_in), torch.zeros(d_out, d_in)
  keys, vals, births = [], [], []
  for t in range(steps):
    x, y = torch.randn(d_in, generator=g), torch.randn(d_out, generator=g)
    if keys:
      coef = age_coefficients(torch.tensor(births), t, mu, gamma, 'exact').to(torch.float64)
      dual = ((torch.stack(vals) * coef) * (torch.stack(keys) @ x)[:, None]).sum(0)
    else:
      dual = torch.zeros(d_out)
    assert torch.allclose(W @ x, dual, atol=1e-6)
    grad_out = W @ x - y
    vel = mu * vel + grad_out[:, None] * x[None, :]
    W = (1 - gamma) * W - lr * vel
    keys.append(x)
    vals.append(-lr * grad_out)
    births.append(t + 1)


def test_mesa_layer_dual_is_foof_recursion():
  """Sec. 4 / App. E.2 'Algorithm A' (γ = 1, Λ = λ): the mesa layer W_T = argmin_W Σ_t ||v_t - W x_t||^2 + λ ||W||_F^2,
  with gradient-corrected targets v_t = W_{t-1} x_t - η g_t, equals the recursion
  Φ_T = Φ_{T-1} + x_T x_T^T (Φ_0 = λ I),  W_T = W_{T-1} - η g_T x_T^T Φ_T^{-1}  (W_0 = 0), i.e. the FOOF update."""
  g = torch.Generator().manual_seed(5)
  d_in, d_out, eta, lam = 4, 3, 0.2, 0.5
  W, Phi, xs, vs = torch.zeros(d_out, d_in), lam * torch.eye(d_in), [], []
  for _ in range(30):
    x, y = torch.randn(d_in, generator=g), torch.randn(d_out, generator=g)
    grad_out = W @ x - y                       # g_t = ∇_y L for L = ||W x - y||^2 / 2
    xs.append(x)
    vs.append(W @ x - eta * grad_out)          # v_t, computed with W_{t-1}
    Phi = Phi + torch.outer(x, x)
    W = W - eta * torch.outer(grad_out, x) @ torch.linalg.inv(Phi)
  X, V = torch.stack(xs, 1), torch.stack(vs, 1)
  W_closed = V @ X.T @ torch.linalg.inv(X @ X.T + lam * torch.eye(d_in))
  assert torch.allclose(W, W_closed, atol=1e-10)


def test_mesa_and_delta_optimizer_steps():
  """One step of rcdl.train.MesaOptimizer (base SGD) and DeltaOptimizer against their defining formulas."""
  from rcdl.train import DeltaOptimizer, MesaOptimizer
  g = torch.Generator().manual_seed(6)
  for kind in ('mesa', 'delta'):
    lin = torch.nn.Linear(5, 3, bias=False)
    W0 = lin.weight.detach().clone()
    X = torch.randn(4, 5, generator=g)
    opt = MesaOptimizer(lin, lr=0.1, gamma=0.9, lam=0.3, base='sgd') if kind == 'mesa' else DeltaOptimizer(lin, 0.1, 0.05)
    lin.train()
    loss = (lin(X) ** 2).sum()
    opt.zero_grad()
    loss.backward()
    G = lin.weight.grad.clone()
    opt.step()
    if kind == 'mesa':                        # Γ_1 = 1, Ψ_1 = X^T X / (B d)
      expect = W0 - 0.1 * G @ torch.linalg.inv(X.T @ X / (4 * 5) + 0.3 * torch.eye(5))
    else:
      expect = W0 @ (torch.eye(5) - 0.05 * 0.1 * X.T @ X) - 0.1 * G
    assert torch.allclose(lin.weight.detach(), expect, atol=1e-8), kind


def test_notebook_age_rule_differs_from_exact_when_both_active():
  """The notebook multiplies separate decay and momentum factors; this is exact only if μ = 0 or γ = 0."""
  b = torch.arange(10)
  a = age_coefficients(b, 10, 0.9, 0.05, 'notebook')
  e = age_coefficients(b, 10, 0.9, 0.05, 'exact')
  assert not torch.allclose(a, e, rtol=1e-3)
  assert torch.allclose(age_coefficients(b, 10, 0.9, 0.0, 'notebook'), age_coefficients(b, 10, 0.9, 0.0, 'exact'),
                        rtol=1e-4)


def test_mesa_low_rank_solve_equals_full_solve():
  """MesaOptimizer: G (Psi + lam I)^{-1} via B right-hand sides (G = D^T X) equals the full d_out-column solve."""
  import copy
  from rcdl.layers import RCDLConfig
  from rcdl.models import build_model
  from rcdl.train import MesaOptimizer
  torch.manual_seed(0)
  cfg = RCDLConfig(kernel_type='standard', hidden_dim=64, num_layers=2, nonlinearity='gelu', num_classes=10,
                   batch_size=16, optimizer='mesa', learning_rate=1e-3)
  m_low = build_model(cfg).double()
  m_full = copy.deepcopy(m_low)
  o_low, o_full = MesaOptimizer(m_low, 1e-3, 0.99, 0.01), MesaOptimizer(m_full, 1e-3, 0.99, 0.01)
  for _ in range(3):
    x, y = torch.randn(16, 64, dtype=torch.float64), torch.randint(0, 10, (16,))
    for model, opt, full in ((m_low, o_low, False), (m_full, o_full, True)):
      opt.zero_grad()
      torch.nn.functional.cross_entropy(model(x), y).backward()
      if full:
        for lin in opt.linears:
          lin._last_output_grad = None          # forces the d_out-column solve
      else:
        assert all(getattr(lin, '_last_output_grad', None) is not None for lin in opt.linears[:-1])
      opt.step()
  for p_low, p_full in zip(m_low.parameters(), m_full.parameters()):
    assert torch.allclose(p_low, p_full, atol=1e-10, rtol=1e-8)
