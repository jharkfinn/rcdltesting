"""rcdl.convnext: shapes, the published parameter counts it is built to match, and attribution of SwiGLU gate/value."""

import torch

from rcdl.attribution import GroupAttribution
from rcdl.convnext import build_convnext, n_params


def test_sizes_match_published_counts():
  assert n_params(build_convnext('medium'), head=False) == 7_057_776      # essay: 7,057,377 (classifier left out)
  assert n_params(build_convnext('wider'), head=False) == 345_576         # essay: 345,552
  assert n_params(build_convnext('large')) == 28_597_960                  # essay: about 28.6M


def test_medium_variant_counts():
  """The medium crush and narrow-stream sizes (without, with the classifier)."""
  want = {'medium': (7_057_776, 7_442_776), 'medium_crush13': (5_890_698, 6_275_698),
          'medium_crush13_skip': (5_891_082, 6_276_082), 'medium_crush64': (5_949_552, 6_334_552),
          'medium_w4': (1_705_116, 1_802_116), 'medium_w8': (855_342, 904_342)}
  models = {s: build_convnext(s) for s in want}
  for size, (body, full) in want.items():
    assert (n_params(models[size], head=False), n_params(models[size])) == (body, full), size
  last = {s: dict(m.blocks())['stage3.block2'] for s, m in models.items()}
  assert [last[s].pw_down.in_features for s in want] == [1024, 13, 13, 64, 1024, 1024]
  assert [last[s].skip for s in want] == [True, False, True, False, True, True]
  assert last['medium_crush13'].gamma is None and last['medium_crush13_skip'].gamma is not None
  assert dict(models['medium_crush13'].blocks())['stage3.block1'].pw_down.in_features == 1024


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
