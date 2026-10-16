"""rcdl.attribution: the tracked updates equal torch's optimizers, and the per-group parts add up exactly."""

import copy

import pytest
import torch
import torch.nn as nn

from rcdl.attribution import GroupAttribution, newton_schulz_factor, muon_adjusted_lr



@pytest.fixture(autouse=True)
def float64_default():
  """float64 inside these tests only (a module-level set_default_dtype would leak into other test files)."""
  old = torch.get_default_dtype()
  torch.set_default_dtype(torch.float64)
  yield
  torch.set_default_dtype(old)


def tiny(seed=0, d_in=5, d_hid=7, d_out=3):
  torch.manual_seed(seed)
  return nn.Sequential(nn.Linear(d_in, d_hid), nn.Tanh(), nn.Linear(d_hid, d_out))


def data(T, B, d_in=5, d_out=3, K=4, seed=1, positions=None):
  g = torch.Generator().manual_seed(seed)
  shape = (T, B, d_in) if positions is None else (T, B, positions, d_in)
  x = torch.randn(*shape, generator=g)
  y = torch.randn(T, B, d_out, generator=g)
  groups = torch.randint(0, K, (T, B), generator=g)
  return x, y, groups


def loss_fn(out, y):
  if out.dim() == 3:                                   # (B, positions, d): mean over positions
    out = out.mean(1)
  return ((out - y) ** 2).sum(1).mean()


@pytest.mark.parametrize('wd', [0.0, 0.1])
def test_adamw_equals_torch_and_parts_add_up(wd):
  model = tiny()
  ref = copy.deepcopy(model)
  attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer='adamw', lr=1e-2, betas=(0.9, 0.95), eps=1e-8,
                          weight_decay=wd, bias_weight_decay=wd, dtype=torch.float64)
  other = torch.optim.AdamW(model[2].parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
  ref_opt = torch.optim.AdamW(ref.parameters(), lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=wd)
  x, y, groups = data(30, 8)
  for t in range(30):
    other.zero_grad()
    loss_fn(model(x[t]), y[t]).backward()
    attr.step(groups[t])
    other.step()
    ref_opt.zero_grad()
    loss_fn(ref(x[t]), y[t]).backward()
    ref_opt.step()
  for p, q in zip(model.parameters(), ref.parameters()):
    assert torch.allclose(p, q, atol=1e-11, rtol=0)
  res = attr.residual('l0')
  assert res['weight'] < 1e-12 and res['bias'] < 1e-12


def test_sgd_nesterov_momentum_equals_torch():
  model = tiny(1)
  ref = copy.deepcopy(model)
  attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer='sgd', lr=5e-2, momentum=0.9, nesterov=True,
                          dtype=torch.float64)
  other = torch.optim.SGD(model[2].parameters(), lr=5e-2, momentum=0.9, nesterov=True)
  ref_opt = torch.optim.SGD(ref.parameters(), lr=5e-2, momentum=0.9, nesterov=True)
  x, y, groups = data(25, 6, seed=3)
  for t in range(25):
    other.zero_grad()
    loss_fn(model(x[t]), y[t]).backward()
    attr.step(groups[t])
    other.step()
    ref_opt.zero_grad()
    loss_fn(ref(x[t]), y[t]).backward()
    ref_opt.step()
  for p, q in zip(model.parameters(), ref.parameters()):
    assert torch.allclose(p, q, atol=1e-12, rtol=0)
  assert attr.residual('l0')['weight'] < 1e-12


@pytest.mark.parametrize('shape', [(7, 5), (5, 7)])
def test_newton_schulz_factor_is_torch_iteration(shape):
  """O = P U / ||U|| (wide) or U P^T / ||U|| (tall), and O equals torch's bf16 iteration up to bf16 rounding."""
  from torch.optim._muon import _zeropower_via_newtonschulz
  U = torch.randn(*shape, generator=torch.Generator().manual_seed(4))
  O, P, transposed, norm = newton_schulz_factor(U)
  assert transposed == (shape[0] > shape[1])
  O2 = (U @ P.T if transposed else P @ U) / norm
  assert torch.allclose(O, O2, atol=1e-12)
  O_torch = _zeropower_via_newtonschulz(U.float(), (3.4445, -4.7750, 2.0315), 5, 1e-7).double()
  assert (O - O_torch).abs().max() < 0.05 * O.abs().max()


@pytest.mark.parametrize('d_hid', [7, 3])
def test_muon_parts_add_up_and_follow_the_muon_rule(d_hid):
  """Muon on a wide (7 x 5) and a tall-after-transpose (3 x 5 -> d_in > d_out) matrix: the weights follow the Muon rule
  computed independently from the gradients, and the per-group parts add up."""
  model = tiny(2, d_hid=d_hid)
  ref = copy.deepcopy(model)                         # before GroupAttribution, which adds a forward hook
  attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer='muon', lr=2e-2, momentum=0.95, nesterov=True,
                          weight_decay=0.05, bias_weight_decay=0.0, betas=(0.9, 0.95), dtype=torch.float64)
  buf = torch.zeros_like(ref[0].weight)
  x, y, groups = data(20, 6, seed=5)
  for t in range(20):
    loss_fn(model(x[t]), y[t]).backward()
    attr.step(groups[t])
    model[2].zero_grad()
    ref.zero_grad()
    loss_fn(ref(x[t]), y[t]).backward()
    with torch.no_grad():
      g = ref[0].weight.grad
      buf.lerp_(g, 1 - 0.95)
      O, _, _, _ = newton_schulz_factor(g.lerp(buf, 0.95))
      ref[0].weight.mul_(1 - 2e-2 * 0.05).add_(O, alpha=-muon_adjusted_lr(2e-2, ref[0].weight.shape))
      ref[0].bias.copy_(model[0].bias)                # bias rule checked by the AdamW test; layer 2 is not trained
  assert torch.allclose(model[0].weight, ref[0].weight, atol=1e-11, rtol=0)
  res = attr.residual('l0')
  assert res['weight'] < 1e-12 and res['bias'] < 1e-12


def test_one_group_per_example_is_linear_attention_over_training():
  """Plain SGD, one group per training example: the part of the pre-activation written by example k on a probe h is
  -lr <h_k, h> delta_k (RCDL Sec. 2, Eq. 1), and all parts plus the initial weights give the pre-activation."""
  model = tiny(3)
  T, B = 6, 4
  attr = GroupAttribution({'l0': model[0]}, n_groups=T * B, optimizer='sgd', lr=0.1, momentum=0.0, nesterov=False,
                          dtype=torch.float64)
  other = torch.optim.SGD(model[2].parameters(), lr=0.1)
  x, y, _ = data(T, B, seed=6)
  stored = []
  for t in range(T):
    other.zero_grad()
    loss_fn(model(x[t]), y[t]).backward()
    tr = attr.tracked['l0']
    stored.append((tr.h.clone(), tr.delta.clone()))
    attr.step(torch.arange(t * B, (t + 1) * B))
    other.step()
  probe = torch.randn(3, 5, generator=torch.Generator().manual_seed(7))
  parts = attr.parts('l0', probe)
  for t, (h, d) in enumerate(stored):
    for i in range(B):
      expect = -0.1 * (probe @ h[i])[:, None] * d[i][None, :] - 0.1 * d[i][None, :]
      assert torch.allclose(parts['groups'][t * B + i], expect, atol=1e-12)
  assert torch.allclose(parts['groups'].sum(0) + parts['init'], model[0](probe), atol=1e-12)


def test_channels_last_positions_sum_into_the_example_gradient():
  """Inputs (B, positions, d_in), as for the pointwise layers of a ConvNeXt block: the per-example gradients summed
  over the batch equal autograd's weight gradient, and the parts give the pre-activation at every position."""
  model = tiny(4)
  ref = copy.deepcopy(model)
  attr = GroupAttribution({'l0': model[0]}, n_groups=3, optimizer='adamw', lr=1e-2, dtype=torch.float64)
  x, y, groups = data(1, 5, K=3, seed=8, positions=9)
  loss_fn(model(x[0]), y[0]).backward()
  G, Gb = attr.tracked['l0'].per_example()
  loss_fn(ref(x[0]), y[0]).backward()
  assert torch.allclose(G.sum(0), ref[0].weight.grad, atol=1e-13)
  assert torch.allclose(Gb.sum(0), ref[0].bias.grad, atol=1e-13)
  loss_fn(model(x[0]), y[0]).backward()
  attr.step(groups[0])
  probe = x[0].reshape(-1, 5)
  parts = attr.parts('l0', probe)
  assert torch.allclose(parts['groups'].sum(0) + parts['init'], model[0](probe), atol=1e-12)


@pytest.mark.parametrize('opt', ['adamw', 'sgd', 'muon'])
def test_soft_groups_add_up_and_equal_hard_groups_when_one_hot(opt):
  """Soft groups (mixup: two source classes with weights lam, 1 - lam): the parts add up exactly; with one-hot weights
  they equal the hard-group parts."""
  torch.manual_seed(5)
  base = tiny(5)
  m1, m2 = copy.deepcopy(base), copy.deepcopy(base)
  kw = dict(n_groups=4, optimizer=opt, lr=1e-2, weight_decay=0.01, dtype=torch.float64)
  a1, a2 = GroupAttribution({'l0': m1[0]}, **kw), GroupAttribution({'l0': m2[0]}, **kw)
  x, y, groups = data(10, 6, seed=9)
  g = torch.Generator().manual_seed(10)
  for t in range(10):
    for m in (m1, m2):
      loss_fn(m(x[t]), y[t]).backward()
    other = torch.randint(0, 4, (6,), generator=g)
    a1.step(groups[t])
    a2.step(torch.stack([groups[t], other], 1), weights=torch.tensor([[1.0, 0.0]] * 6))
  assert torch.allclose(a1.tracked['l0'].C, a2.tracked['l0'].C, atol=1e-12)
  m3 = copy.deepcopy(base)
  a3 = GroupAttribution({'l0': m3[0]}, **kw)
  for t in range(10):
    loss_fn(m3(x[t]), y[t]).backward()
    lam = torch.rand(6, generator=g, dtype=torch.float64)
    a3.step(torch.stack([groups[t], torch.randint(0, 4, (6,), generator=g)], 1),
            weights=torch.stack([lam, 1 - lam], 1))
  r = a3.residual('l0')
  assert r['weight'] < 1e-12 and r['bias'] < 1e-12
