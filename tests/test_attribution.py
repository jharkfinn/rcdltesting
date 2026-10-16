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


def test_muon_bias_uses_adamw_at_the_scaled_learning_rate():
  """Under muon, a tracked bias follows torch AdamW at lr * bias_lr_scale (the caller's AdamW rate for 1-D params)."""
  model = tiny(3)
  ref = copy.deepcopy(model)
  attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer='muon', lr=2e-2, momentum=0.95, nesterov=True,
                          weight_decay=0.05, bias_weight_decay=0.0, betas=(0.9, 0.95), bias_lr_scale=0.1,
                          dtype=torch.float64)
  ref_b = torch.optim.AdamW([ref[0].bias], lr=2e-3, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
  x, y, groups = data(15, 6, seed=7)
  for t in range(15):
    loss_fn(model(x[t]), y[t]).backward()
    attr.step(groups[t])
    model[2].zero_grad()
    ref_b.zero_grad()
    ref[0].weight.grad = None
    loss_fn(ref(x[t]), y[t]).backward()
    ref_b.step()
    with torch.no_grad():
      ref[0].weight.copy_(model[0].weight)            # weight rule checked by the Muon test
  assert torch.allclose(model[0].bias, ref[0].bias, atol=1e-11, rtol=0)
  assert attr.residual('l0')['bias'] < 1e-12


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


@pytest.mark.parametrize('opt', ['adamw', 'muon'])
def test_scalar_soft_weights_equal_per_row_weights(opt):
  """weights=(lam, 1 - lam) as floats gives the same parts as the same weights given per row."""
  outs = []
  for scalar in (False, True):
    model = tiny(4)
    attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer=opt, lr=1e-2, weight_decay=0.05,
                            dtype=torch.float64)
    x, y, groups = data(10, 6, seed=9)
    for t in range(10):
      lam = 0.3 + 0.04 * t
      loss_fn(model(x[t]), y[t]).backward()
      model[2].zero_grad()
      g2 = torch.stack([groups[t], groups[t].flip(0)], 1)
      w = (lam, 1 - lam) if scalar else torch.tensor([[lam, 1 - lam]]).expand(6, 2)
      attr.step(g2, weights=w)
    outs.append(attr.tracked['l0'].C.clone())
  assert torch.allclose(outs[0], outs[1], atol=1e-14, rtol=0)


def test_compact_state_is_exact_with_its_remainder():
  """Scaled fp16 parts plus the fp32 remainder give back the weights to fp32 rounding, and tiny parts stay nonzero."""
  model = tiny(5)
  attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer='adamw', lr=1e-6, weight_decay=0.05,
                          dtype=torch.float32)
  x, y, groups = data(3, 6, seed=11)
  for t in range(3):
    loss_fn(model(x[t]), y[t]).backward()
    model[2].zero_grad()
    attr.step(groups[t])
  st = attr.compact_state()['l0']
  tr = attr.tracked['l0']
  C = st['C'].float() / st['C_scale']
  assert (C != 0).sum() == (tr.C != 0).sum()                      # lr 1e-6: plain fp16 would be subnormal or zero
  assert torch.allclose(C, tr.C, rtol=1e-3, atol=0)
  W = st['a'] * tr.W0.float() + C.sum(0) + st['R']
  assert torch.allclose(W, model[0].weight.float(), atol=1e-7, rtol=0)
  b = st['ab'] * tr.b0.float() + st['Cb'].float().sum(0) / st['Cb_scale'] + st['Rb']
  assert torch.allclose(b, model[0].bias.float(), atol=1e-7, rtol=0)


@pytest.mark.parametrize('opt', ['adamw', 'muon'])
def test_full_state_resumes_exactly(opt):
  """Saving full_state (with the model) mid-run and loading it into fresh objects continues bit for bit."""
  def make():
    model = tiny(6)
    attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer=opt, lr=1e-2, weight_decay=0.05,
                            dtype=torch.float64)
    return model, attr
  x, y, groups = data(10, 6, seed=13)
  def run(model, attr, ts):
    for t in ts:
      loss_fn(model(x[t]), y[t]).backward()
      model[2].zero_grad()
      attr.step(groups[t])
  model, attr = make()
  run(model, attr, range(5))
  saved = copy.deepcopy((model.state_dict(), attr.full_state()))
  run(model, attr, range(5, 10))
  model2, attr2 = make()
  model2.load_state_dict(saved[0])
  attr2.load_full_state(saved[1])
  run(model2, attr2, range(5, 10))
  assert torch.equal(model[0].weight, model2[0].weight)
  assert torch.equal(attr.tracked['l0'].C, attr2.tracked['l0'].C)
  assert attr.tracked['l0'].t == attr2.tracked['l0'].t == 10


@pytest.mark.parametrize('opt', ['adamw', 'sgd', 'muon'])
def test_lazy_decay_survives_strong_decay_and_flushes(opt):
  """Weight decay of the parts is deferred to a scale s; with lr * wd = 0.5 per step s falls below 1e-6 after 20
  steps and is flushed. The parts still add up, and AdamW still equals torch."""
  model = tiny(8)
  ref = copy.deepcopy(model)
  kw = dict(momentum=0.9, nesterov=True) if opt != 'adamw' else {}
  attr = GroupAttribution({'l0': model[0]}, n_groups=4, optimizer=opt, lr=0.1, weight_decay=5.0,
                          bias_weight_decay=5.0, dtype=torch.float64, **kw)
  ref_opt = torch.optim.AdamW(ref[0].parameters(), lr=0.1, weight_decay=5.0)
  x, y, groups = data(30, 6, seed=15)
  scales = []
  for t in range(30):
    loss_fn(model(x[t]), y[t]).backward()
    model[2].zero_grad()
    attr.step(groups[t])
    scales.append(attr.tracked['l0'].s)
    if opt == 'adamw':
      ref_opt.zero_grad()
      loss_fn(ref(x[t]), y[t]).backward()
      ref_opt.step()
      ref[2].zero_grad()
  assert any(b > a for a, b in zip(scales, scales[1:])) and min(scales) >= 0.5e-6   # flushed, so s stays bounded
  res = attr.residual('l0')
  assert res['weight'] < 1e-12 * max(1.0, res['weight_scale']) and res['bias'] < 1e-12
  if opt == 'adamw':
    assert torch.allclose(model[0].weight, ref[0].weight, atol=1e-11, rtol=0)


@pytest.mark.parametrize('opt', ['adamw', 'sgd', 'muon'])
def test_shadow_split_leaves_training_unchanged_and_adds_up(opt):
  """A shadow split over other groups (e.g. random groups as a control) does not change training: the weights equal
  those of the owner alone, bit for bit; both splits add up to the same weights."""
  base = tiny(6)
  m1, m2 = copy.deepcopy(base), copy.deepcopy(base)
  kw = dict(optimizer=opt, lr=1e-2, weight_decay=0.05, dtype=torch.float64)
  a1 = GroupAttribution({'l0': m1[0], 'l2': m1[2]}, n_groups=4, **kw)
  a2 = GroupAttribution({'l0': m2[0], 'l2': m2[2]}, n_groups=4, **kw)
  sh = GroupAttribution({'l0': m2[0], 'l2': m2[2]}, n_groups=5, shadow=True, **kw)
  x, y, groups = data(12, 6, seed=3)
  g = torch.Generator().manual_seed(4)
  for t in range(12):
    for m in (m1, m2):
      loss_fn(m(x[t]), y[t]).backward()
    lam = 0.2 + 0.05 * t
    a1.step(torch.stack([groups[t], groups[t].flip(0)], 1), weights=(lam, 1 - lam))
    a2.step(torch.stack([groups[t], groups[t].flip(0)], 1), weights=(lam, 1 - lam))
    rg = torch.randint(0, 5, (6,), generator=g)
    sh.step(torch.stack([rg, rg.flip(0)], 1), weights=(lam, 1 - lam))
  for p1, p2 in zip(m1.parameters(), m2.parameters()):
    assert torch.equal(p1, p2)
  for name in ('l0', 'l2'):
    for at in (a2, sh):
      r = at.residual(name)
      assert r['weight'] < 1e-12 and r['bias'] < 1e-12
    assert torch.allclose(a2.tracked[name].C.sum(0), sh.tracked[name].C.sum(0), atol=1e-12)
    assert not torch.allclose(a2.tracked[name].C[:4], sh.tracked[name].C[:4])


@pytest.mark.parametrize('opt', ['adamw', 'sgd', 'muon'])
def test_snapshot_replay_continues_training_and_adds_up(opt):
  """A new attribution started from a snapshot (old momentum in a 'history' group) continues training as the
  original run does, and its parts add up to the change since the snapshot."""
  base = tiny(8)
  kw = dict(optimizer=opt, lr=1e-2, weight_decay=0.05, dtype=torch.float64)
  x, y, groups = data(14, 6, seed=5)
  m1 = copy.deepcopy(base)
  a1 = GroupAttribution({'l0': m1[0], 'l2': m1[2]}, n_groups=4, **kw)
  m2 = copy.deepcopy(base)
  a2 = GroupAttribution({'l0': m2[0], 'l2': m2[2]}, n_groups=4, **kw)
  for t in range(14):
    loss_fn(m1(x[t]), y[t]).backward()
    a1.step(groups[t])
    if t < 7:
      loss_fn(m2(x[t]), y[t]).backward()
      a2.step(groups[t])
  snap = a2.snapshot()
  m3 = copy.deepcopy(base)                               # a clean copy: deepcopy(m2) would carry a2's hooks
  m3.load_state_dict(m2.state_dict())
  a3 = GroupAttribution({'l0': m3[0], 'l2': m3[2]}, n_groups=5, **kw)
  a3.load_snapshot(snap, history_group=4)
  for t in range(7, 14):
    loss_fn(m3(x[t]), y[t]).backward()
    a3.step(groups[t])
  for p1, p3 in zip(m1.parameters(), m3.parameters()):
    assert torch.allclose(p1, p3, atol=1e-12, rtol=0)
  for name in ('l0', 'l2'):
    r = a3.residual(name)
    assert r['weight'] < 1e-12 and r['bias'] < 1e-12
    assert a3.tracked[name].C[4].abs().max() > 0          # the history group still writes (momentum, Nesterov)


@pytest.mark.parametrize('opt', ['adamw', 'sgd'])
def test_fine_group_shadow_folds_to_class_parts(opt):
  """A shadow split over fine groups, each inside one class (own images of a rare class, subsets, decoy sets), with
  mixup's soft groups: summing the fine parts of each class gives that class's part, the head included."""
  m = tiny(9)
  kw = dict(optimizer=opt, lr=1e-2, weight_decay=0.05, dtype=torch.float64)
  layers = {'l0': m[0], 'head': m[2]}
  cls = GroupAttribution(layers, n_groups=4, **kw)
  group_class = torch.tensor([0, 0, 1, 2, 2, 2, 3, 1])           # 8 fine groups inside 4 classes
  fine = GroupAttribution(layers, n_groups=len(group_class), shadow=True, **kw)
  x, y, _ = data(10, 6, seed=7)
  g = torch.Generator().manual_seed(8)
  for t in range(10):
    f = torch.randint(0, len(group_class), (6,), generator=g)    # each example's fine group
    c = group_class[f]                                           # and its class
    loss_fn(m(x[t]), y[t]).backward()
    lam = 0.3 + 0.04 * t
    cls.step(torch.stack([c, c.flip(0)], 1), weights=(lam, 1 - lam))
    fine.step(torch.stack([f, f.flip(0)], 1), weights=(lam, 1 - lam))
  for name in layers:
    for part in ('C', 'Cb'):
      Cf = getattr(fine.tracked[name], part)
      folded = torch.zeros_like(getattr(cls.tracked[name], part)).index_add_(0, group_class, Cf)
      assert torch.allclose(folded, getattr(cls.tracked[name], part), atol=1e-12)
    assert cls.residual(name)['weight'] < 1e-12 and fine.residual(name)['weight'] < 1e-12


def _manual_adamw(params, grads, state, lr, wd, b1=0.9, b2=0.999, eps=1e-8, frozen=False):
  """Reference AdamW (decoupled weight decay on 2-D tensors only); frozen: v and its bias correction are kept."""
  state['t'] += 1
  t = state['t']
  for i, (p, g) in enumerate(zip(params, grads)):
    m, v = state['m'][i], state['v'][i]
    m.mul_(b1).add_(g, alpha=1 - b1)
    if not frozen:
      v.mul_(b2).addcmul_(g, g, value=1 - b2)
      state['c2'] = 1 - b2 ** t
    with torch.no_grad():
      p.mul_(1 - lr * (wd if p.dim() == 2 else 0.0))
      p.add_(-lr * (m / (1 - b1 ** t)) / ((v / state['c2']).sqrt() + eps))


def test_frozen_second_moment_matches_reference_and_adds_up():
  """After freeze_second_moment, the tracked update is lr * mhat / (sqrt(v_frozen / c2_frozen) + eps): equal to a
  reference AdamW with the same frozen denominator; the parts still add up."""
  base = tiny(11)
  m1, ref = copy.deepcopy(base), copy.deepcopy(base)
  layers = {'l0': m1[0], 'l2': m1[2]}
  at = GroupAttribution(layers, n_groups=3, optimizer='adamw', lr=1e-2, weight_decay=0.05, dtype=torch.float64)
  x, y, groups = data(9, 6, seed=12)
  groups = groups % 3
  rp = [ref[0].weight, ref[0].bias, ref[2].weight, ref[2].bias]
  st = {'t': 0, 'm': [torch.zeros_like(p, dtype=torch.float64) for p in rp],
        'v': [torch.zeros_like(p, dtype=torch.float64) for p in rp]}
  for p in rp:
    p.data = p.data.double()
  for t in range(9):
    if t == 4:
      at.freeze_second_moment()
    loss_fn(m1(x[t]), y[t]).backward()
    at.step(groups[t])
    for p in rp:
      p.grad = None
    loss_fn(ref(x[t].double()), y[t].double()).backward()
    _manual_adamw(rp, [p.grad.double() for p in rp], st, 1e-2, 0.05, frozen=t >= 4)
  for got, want in ((m1[0].weight, rp[0]), (m1[0].bias, rp[1]), (m1[2].weight, rp[2]), (m1[2].bias, rp[3])):
    assert torch.allclose(got.double(), want, atol=1e-10)
  for name in layers:
    assert at.residual(name)['weight'] < 1e-12


def test_interval_buckets_sum_to_the_change():
  from rcdl.attribution import interval_buckets
  m = tiny(13)
  layers = {'l0': m[0], 'l2': m[2]}
  at = GroupAttribution(layers, n_groups=4, optimizer='adamw', lr=1e-2, weight_decay=0.1, dtype=torch.float64)
  init = {k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in at.tracked.items()}
  sets = {'a': [0, 1], 'b': [2], 'c': [3]}
  x, y, groups = data(10, 6, seed=14)
  for t in range(10):
    loss_fn(m(x[t]), y[t]).backward()
    at.step(groups[t])
    if t == 2:
      s1 = at.group_sums(sets)
      W1 = {k: l.weight.detach().double().clone() for k, l in layers.items()}
  s2 = at.group_sums(sets)
  bk = interval_buckets(s1, s2, init)
  for k, l in layers.items():
    total = sum(bk[k]['write'].values()) + sum(bk[k]['dec'].values()) + bk[k]['dec_0'] + bk[k]['rem']
    assert torch.allclose(total.double(), l.weight.detach().double() - W1[k], atol=1e-6)
    assert not torch.allclose(bk[k]['dec']['a'], torch.zeros_like(bk[k]['dec']['a']))     # weight decay acted


def test_snapshot_from_torch_adamw_continues_torch_training():
  """Layers trained by torch AdamW, then tracked from a snapshot of torch's state: training continues as torch's
  would, and the parts add up to the change since the snapshot."""
  from rcdl.attribution import snapshot_from_torch_adamw
  base = tiny(15)
  ref = copy.deepcopy(base).double()
  def groups_of(model):
    return [{'params': [model[0].weight, model[2].weight], 'weight_decay': 0.05},
            {'params': [model[0].bias, model[2].bias], 'weight_decay': 0.0}]
  opt = torch.optim.AdamW(groups_of(ref), lr=1e-2)
  x, y, groups = data(10, 6, seed=16)
  for t in range(4):
    opt.zero_grad()
    loss_fn(ref(x[t].double()), y[t].double()).backward()
    opt.step()
  m2 = copy.deepcopy(ref)
  snap = snapshot_from_torch_adamw(opt, {'l0': ref[0], 'l2': ref[2]})
  at = GroupAttribution({'l0': m2[0], 'l2': m2[2]}, n_groups=5, optimizer='adamw', lr=1e-2, weight_decay=0.05,
                        dtype=torch.float64)
  at.load_snapshot(snap, history_group=4)
  for t in range(4, 10):
    opt.zero_grad()
    loss_fn(ref(x[t].double()), y[t].double()).backward()
    opt.step()
    loss_fn(m2(x[t].double()), y[t].double()).backward()
    at.step(groups[t] % 4)
  for a_, b_ in zip(ref.parameters(), m2.parameters()):
    assert torch.allclose(a_, b_, atol=1e-10)
  for name in ('l0', 'l2'):
    assert at.residual(name)['weight'] < 1e-12
