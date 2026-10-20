"""experiments.convnext_train --freeze_gamma: frozen layer scales stay fixed, everything else trains, and saved
optimizer states (snapshots, resume states) load into optimizers built with a different frozen set."""

import pytest
import torch
import torch.nn.functional as F

from experiments import convnext_train as ct
from rcdl.attribution import GroupAttribution
from rcdl.convnext import ConvNeXt


def _setup(opt, stages):
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 2, 1, 1), n_classes=4, layer_scale=0.5)
  layers = {f'{n}.{k}': getattr(b, k) for n, b in m.blocks() for k in ('pw_gate', 'pw_value')}
  akw = dict(n_groups=4, optimizer=opt, lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.05, bias_weight_decay=0.0)
  if opt == 'muon':
    akw |= dict(lr=2e-2, momentum=0.95, nesterov=True, bias_lr_scale=0.5)
  attr = GroupAttribution(layers, **akw)
  tracked = [t for l in layers.values() for t in (l.weight, l.bias)]
  frozen = ct.freeze_layer_scales(m, ct.parse_stages(stages))
  opts = ct.build_optimizers(m, tracked, opt, 1e-2, 2e-2, 0.05)
  return m, attr, tracked, frozen, opts


def _train(m, attr, opts, steps=3, seed=1):
  g = torch.Generator().manual_seed(seed)
  for _ in range(steps):
    x = torch.randn(4, 3, 32, 32, generator=g)
    y = torch.randint(0, 4, (4,), generator=g)
    loss = F.cross_entropy(m(x), y)
    for o in opts:
      o.zero_grad(set_to_none=True)
    loss.backward()
    attr.step(torch.stack([y, y.flip(0)], 1), 1e-2, weights=(0.7, 0.3))
    for o in opts:
      o.step()


@pytest.mark.parametrize('opt', ['adamw', 'muon'])
def test_frozen_stage_keeps_its_layer_scale(opt):
  m, attr, tracked, frozen, opts = _setup(opt, '1')
  assert frozen == ['stages.1.0.gamma', 'stages.1.1.gamma']
  in_opt = {id(p) for o in opts for g in o.param_groups for p in g['params']}
  assert not any(id(p) in in_opt for n, p in m.named_parameters() if n in frozen)
  before = {n: p.detach().clone() for n, p in m.named_parameters()}
  _train(m, attr, opts)
  for n, p in m.named_parameters():
    if n in frozen:
      assert torch.equal(p, before[n]), n
      assert p.grad is None
    else:
      assert not torch.equal(p, before[n]), f'{n} did not change'      # incl. the other gammas and the tracked maps


def test_default_builds_the_same_groups():
  m, attr, tracked, frozen, opts = _setup('adamw', '')
  assert frozen == []
  ref = ct.param_groups(m, tracked, 0.05)
  assert [[id(p) for p in g['params']] for g in opts[0].param_groups] == [[id(p) for p in g['params']] for g in ref]


@pytest.mark.parametrize('opt', ['adamw', 'muon'])
def test_snapshot_state_loads_with_a_frozen_stage(opt):
  m0, attr0, tracked0, _, opts0 = _setup(opt, '')
  _train(m0, attr0, opts0, steps=2)
  saved = [o.state_dict() for o in opts0]
  names0 = {id(p): n for n, p in m0.named_parameters()}
  state0 = {names0[id(p)]: o.state[p] for o in opts0 for g in o.param_groups for p in g['params']}
  m, attr, tracked, frozen, opts = _setup(opt, '1')
  m.load_state_dict(m0.state_dict())
  ct.load_optimizer_states(opts, saved, m, tracked, opt, 0.05, '', '1')
  names = {id(p): n for n, p in m.named_parameters()}
  for o in opts:
    for g in o.param_groups:
      for p in g['params']:
        n = names[id(p)]
        for k, v in state0[n].items():
          torch.testing.assert_close(o.state[p][k], v, rtol=0, atol=0)
  before = {n: p.detach().clone() for n, p in m.named_parameters()}
  _train(m, attr, opts, steps=2, seed=2)
  for n, p in m.named_parameters():
    assert torch.equal(p, before[n]) == (n in frozen), n
  # and back: a state saved with stage 1 frozen loads into optimizers without freezing (fresh moments for the gammas)
  m2, attr2, tracked2, _, opts2 = _setup(opt, '')
  m2.load_state_dict(m.state_dict())
  ct.load_optimizer_states(opts2, [o.state_dict() for o in opts], m2, tracked2, opt, 0.05, '1', '')
  _train(m2, attr2, opts2, steps=1, seed=3)
  assert not torch.equal(m2.stages[1][0].gamma, m.stages[1][0].gamma)


def test_stage_without_layer_scales_is_an_error():
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 2, 1, 1), n_classes=4, layer_scale=0.5)
  with pytest.raises(ValueError):
    ct.freeze_layer_scales(m, ct.parse_stages('4'))
  assert all(p.requires_grad for p in m.parameters())
  assert ct.freeze_layer_scales(m, ct.parse_stages('')) == []
