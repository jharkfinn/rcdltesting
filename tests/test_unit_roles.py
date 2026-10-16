"""experiments.convnext_unit_roles: the forward from a block's input equals the model, and the ablations act as stated."""

import numpy as np
import torch

from experiments import convnext_unit_roles as ur
from rcdl.convnext import ConvNeXt


def _model():
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 2, 1, 1), n_classes=3, layer_scale=0.5)
  for p in m.parameters():
    p.data.normal_(0, 0.3)
  return m.eval()


def test_forward_from_and_ablations():
  m = _model()
  xs = torch.randint(0, 256, (6, 3, 64, 64), dtype=torch.uint8)
  y = torch.tensor([0, 0, 1, 1, 2, 2])
  xb = (xs.float() - ur.MEAN[:, None, None]) / ur.STD[:, None, None]
  X = ur.block_inputs(m, xs, 1, 1, 6, 'cpu')
  with torch.no_grad():
    torch.testing.assert_close(ur.forward_from(m, 1, 1, X), m(xb), rtol=1e-4, atol=1e-5)
    blk = m.stages[1][1]
    u = ur.block_units(blk, X)
    sc = torch.ones(6)
    sc[2] = 0.0
    out0 = ur.block_out(blk, X, u * sc)
    out1 = ur.block_out(blk, X, u)
    w = ur.write_dir(blk, 2)
    torch.testing.assert_close(out1 - out0, u[..., 2][:, None] * w.view(1, -1, 1, 1), rtol=1e-4, atol=1e-5)
  roles, raw = ur.unit_roles(m, xs, y, 'stage1.block1', batch=6, device='cpu', n_top=1)
  assert raw['dce'].shape == (6, 3) and raw['R'].shape == (6, 3)
  s = roles['summary']
  assert all(np.isfinite(v) for v in s.values())
  assert all(-1 <= r['rho_R_dce'] <= 1 for r in roles['units'])
