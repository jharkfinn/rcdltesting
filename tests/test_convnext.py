"""rcdl.convnext: shapes, the published parameter counts it is built to match, and attribution of SwiGLU gate/value."""

import torch

from rcdl.attribution import GroupAttribution
from rcdl.convnext import build_convnext, n_params


def test_sizes_match_published_counts():
  assert n_params(build_convnext('medium'), head=False) == 7_057_776      # essay: 7,057,377 (classifier left out)
  assert n_params(build_convnext('wider'), head=False) == 345_576         # essay: 345,552
  assert n_params(build_convnext('large')) == 28_597_960                  # essay: about 28.6M


def test_forward_and_swiglu_units():
  m = build_convnext('tiny', stem=2)
  x = torch.randn(2, 3, 64, 64)
  assert m(x).shape == (2, 1000)
  name, blk = m.blocks()[-1]
  assert name == 'stage3.block2' and blk.pw_gate.out_features == 64


def test_attribution_tracks_gate_and_value_of_a_block():
  torch.manual_seed(0)
  m = build_convnext('tiny', stem=2, n_classes=10)
  blk = dict(m.blocks())['stage3.block2']
  attr = GroupAttribution({'gate': blk.pw_gate, 'value': blk.pw_value}, n_groups=10, optimizer='adamw', lr=1e-3)
  opt = torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-3)
  x, y = torch.randn(8, 3, 64, 64), torch.randint(0, 10, (8,))
  for _ in range(3):
    opt.zero_grad()
    torch.nn.functional.cross_entropy(m(x), y).backward()
    attr.step(y)
    opt.step()
  for name in ('gate', 'value'):
    r = attr.residual(name)
    assert r['weight'] < 1e-6 * max(1.0, r['weight_scale']) and r['bias'] < 1e-6
