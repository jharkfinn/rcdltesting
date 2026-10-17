"""experiments.convnext_microscope: per-image extremes, bucket ranks, crop geometry and the written sheet / data."""

import json

import numpy as np
import torch

from experiments import convnext_microscope as cm
from rcdl.convnext import ConvNeXt


def _model():
  torch.manual_seed(0)
  m = ConvNeXt(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 1, 1, 1), n_classes=5, layer_scale=0.5)
  for p in m.parameters():
    p.data.normal_(0, 0.3)
  return m.eval()


def _batches(n=40, b=10, seed=0):
  g = torch.Generator().manual_seed(seed)
  x = torch.randn(n, 3, 64, 64, generator=g)
  y = torch.arange(n) % 5
  return [(x[i:i + b], y[i:i + b]) for i in range(0, n, b)]


def test_collect_matches_brute_force_and_dirs():
  m = _model()
  dirs = cm.random_dirs(6, 3, seed=1)
  S = cm.collect(m, 'stage3.block0', _batches(), dirs)
  got = {}
  h = m.stages[3][0].pw_down.register_forward_pre_hook(lambda mod, inp: got.__setitem__('u', inp[0]))
  with torch.no_grad():
    x = torch.cat([b[0] for b in _batches()])
    m(x)
  h.remove()
  u = got['u']
  u = torch.cat([u, u @ torch.as_tensor(dirs, dtype=torch.float32).T], -1).reshape(len(x), -1, 9)
  np.testing.assert_allclose(S['maxv'], u.max(1).values.numpy(), rtol=1e-6)
  np.testing.assert_allclose(S['minv'], u.min(1).values.numpy(), rtol=1e-6)
  assert (S['pmax'] == u.argmax(1).numpy()).all() and S['grid'] == (2, 2)
  with torch.no_grad():
    acc = (m(x).argmax(1) == torch.cat([b[1] for b in _batches()])).float().mean().item()
  assert abs(S['top1'] - acc) < 1e-6
  np.testing.assert_allclose(np.linalg.norm(dirs, axis=1), 1)


def test_select_ranks():
  N, C = 600, 2
  rng = np.random.default_rng(0)
  S = {'maxv': rng.standard_normal((N, C)), 'minv': rng.standard_normal((N, C)), 'pmax': np.zeros((N, C), int),
       'pmin': np.zeros((N, C), int)}
  sel = cm.select(S)
  for c in range(C):
    order = np.argsort(-S['maxv'][:, c])
    assert [x[0] for x in sel[c]['top16']] == list(order[:16]) and [x[1] for x in sel[c]['top16']] == list(range(1, 17))
    assert [x[1] for x in sel[c]['p9999']] == list(range(5, 13)) and [x[1] for x in sel[c]['p999']] == list(range(47, 55))
    assert [x[1] for x in sel[c]['p99']] == list(range(497, 505))
    assert [x[0] for x in sel[c]['bottom4']] == list(np.argsort(S['minv'][:, c])[:4])


def test_crop_box_centre_and_clamp():
  assert cm.crop_box(24, (7, 7), 3) == (16, 16, 192)            # centre position: centred crop
  assert cm.crop_box(0, (7, 7), 3) == (0, 0, 192)               # corner: moved inside the image
  assert cm.crop_box(48, (7, 7), 3) == (32, 32, 192)
  t, l, s = cm.crop_box(9, (14, 14), 2)                          # (0, 9): centre (8, 152), side 144
  assert (t, l, s) == (0, 80, 144)


def test_write_sheet_and_data(tmp_path):
  m = _model()
  b = _batches(n=600, b=100)
  S = cm.collect(m, 'stage3.block0', b, None)
  sel = cm.select(S)
  crops = cm.gather_crops(b, sel, S['grid'], 3)
  assert all(v.shape == (96, 96, 3) and v.dtype == np.uint8 for v in crops.values())
  names = [f'class{i}' for i in range(5)]
  cm.write(str(tmp_path), 'stage3_block0', sel, crops, S['labels'], names, 6, S)
  d = json.load(open(tmp_path / 'data_stage3_block0.json'))
  assert d['channels'] == 6 and [e['bucketId'] for e in d['neurons'][0]['examples']] == ['top16', 'p9999', 'p999', 'p99', 'bottom4']
  c0 = d['neurons'][2]['examples'][0]['cells'][0]
  assert c0['rank'] == 1 and c0['datasetLabel'] == names[int(S['labels'][c0['imageIndex']])]
  from PIL import Image
  assert Image.open(tmp_path / 'examples' / 'stage3_block0' / '0002.jpg').size == (1536, 288)


def test_inception_collect_units_and_directions():
  from experiments import inception_microscope as im
  from rcdl.third_party.lucent_inceptionv1 import InceptionV1
  torch.manual_seed(0)
  m = InceptionV1(pretrained=False, redirected_ReLU=False).eval()
  g = torch.Generator().manual_seed(1)
  x = torch.randn(3, 3, 224, 224, generator=g) * 50
  units = [5, 300, 1000]
  dirs = cm.random_dirs(1024, 2, seed=2)
  S = im.collect(m, 'mixed5b', [(x[:2], [0, 1]), (x[2:], [2])], units, dirs)
  cur, hs = im.hook_layer(m, 'mixed5b')
  with torch.no_grad():
    m(x)
  for h in hs:
    h.remove()
  a = torch.cat([cur[b] for b in im.BRANCHES], 1).flatten(2)
  d = torch.as_tensor(dirs, dtype=torch.float32)
  ref = torch.cat([a[:, units], torch.einsum('dc,bcp->bdp', d, a), torch.einsum('dc,bcp->bdp', d, a.clamp(min=0))], 1)
  assert S['grid'] == (7, 7) and S['maxv'].shape == (3, 7)
  np.testing.assert_allclose(S['maxv'], ref.max(-1).values.numpy(), rtol=1e-4, atol=1e-4)
  np.testing.assert_allclose(S['minv'], ref.min(-1).values.numpy(), rtol=1e-4, atol=1e-4)
  assert (S['pmax'] == ref.argmax(-1).numpy()).mean() > 0.95


def test_inception_crop_box():
  from experiments import inception_microscope as im
  assert im.crop_box(0, (7, 7), 224) == (0, 0, 224)
  assert im.crop_box(0, (14, 14), 144) == (0, 0, 144)
  top, left, side = im.crop_box(14 * 7 + 7, (14, 14), 144)                  # row 7, col 7: centre 120 -> 48
  assert (top, left, side) == (48, 48, 144)
  assert im.crop_box(14 * 14 - 1, (14, 14), 144) == (80, 80, 144)
