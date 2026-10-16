"""rcdl.imagenet.crop_params: torchvision's RandomResizedCrop.get_params, vectorized over a batch."""

import math

import numpy as np

from rcdl.imagenet import crop_params


def tv_get_params(rng, h, w, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3)):
  """torchvision.transforms.RandomResizedCrop.get_params (v0.20 source), with numpy draws."""
  area = h * w
  log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
  for _ in range(10):
    target_area = area * rng.uniform(scale[0], scale[1])
    aspect_ratio = math.exp(rng.uniform(log_ratio[0], log_ratio[1]))
    cw = int(round(math.sqrt(target_area * aspect_ratio)))
    ch = int(round(math.sqrt(target_area / aspect_ratio)))
    if 0 < cw <= w and 0 < ch <= h:
      return int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1)), ch, cw
  in_ratio = w / h
  if in_ratio < min(ratio):
    cw, ch = w, int(round(w / min(ratio)))
  elif in_ratio > max(ratio):
    ch, cw = h, int(round(h * max(ratio)))
  else:
    cw, ch = w, h
  return (h - ch) // 2, (w - cw) // 2, ch, cw


def test_fallback_geometry_matches_torchvision():
  hw = np.array([[256, 1024], [1024, 256], [256, 341], [300, 300], [256, 2000]])
  rng = np.random.default_rng(0)
  top, left, ch, cw, _ = crop_params(rng, hw, scale=(5.0, 5.0))       # no draw fits: always the fallback
  for k, (h, w) in enumerate(hw):
    assert (top[k], left[k], ch[k], cw[k]) == tv_get_params(np.random.default_rng(1), h, w, scale=(5.0, 5.0))


def test_crops_are_valid_and_distributed_like_torchvision():
  hw = np.array([[256, 341]] * 20000 + [[256, 1024]] * 20000)
  top, left, ch, cw, flip = crop_params(np.random.default_rng(2), hw)
  assert (top >= 0).all() and (left >= 0).all() and (top + ch <= hw[:, 0]).all() and (left + cw <= hw[:, 1]).all()
  assert abs(flip.mean() - 0.5) < 0.01
  rng = np.random.default_rng(3)
  for sl, (h, w) in ((slice(0, 20000), (256, 341)), (slice(20000, 40000), (256, 1024))):
    ref = np.array([tv_get_params(rng, h, w) for _ in range(20000)])
    for col, ours in ((2, ch[sl]), (3, cw[sl])):
      assert abs(ours.mean() - ref[:, col].mean()) < 0.02 * ref[:, col].mean()
      assert abs(ours.std() - ref[:, col].std()) < 0.04 * ref[:, col].std()
    assert abs((ch[sl] * cw[sl]).mean() - (ref[:, 2] * ref[:, 3]).mean()) < 0.02 * (ref[:, 2] * ref[:, 3]).mean()


def test_prepare_train_keeps_shard_and_row_order(tmp_path, monkeypatch):
  """Chunked parallel packing writes images and labels in shard order, then row order."""
  import pytest
  pytest.importorskip('huggingface_hub')
  pytest.importorskip('pyarrow')
  import io
  import shutil
  import pyarrow as pa
  import pyarrow.parquet as pq
  from PIL import Image
  import huggingface_hub
  from rcdl import imagenet
  src = tmp_path / 'src'
  (src / 'data').mkdir(parents=True)
  names, label = [], 0
  for s in range(4):
    rows = []
    for r in range(7):
      buf = io.BytesIO()
      Image.new('RGB', (40 + label, 30), (label % 256, 0, 0)).save(buf, format='JPEG')
      rows.append({'image': {'bytes': buf.getvalue(), 'path': None}, 'label': label})
      label += 1
    name = f'data/train-{s:05d}-of-00004.parquet'
    pq.write_table(pa.Table.from_pylist(rows), src / name)
    names.append(name)
  class FakeApi:
    def __init__(self, token=None):
      pass
    def list_repo_files(self, repo, repo_type=None):
      return list(reversed(names)) + ['classes.py']
  def fake_download(repo, f, repo_type=None, token=None, local_dir=None):
    dst = tmp_path / 'dl' / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src / f, dst)
    return str(dst)
  monkeypatch.setattr(huggingface_hub, 'HfApi', FakeApi)
  monkeypatch.setattr(huggingface_hub, 'hf_hub_download', fake_download)
  out = tmp_path / 'packed'
  imagenet.prepare_train(str(out), workers=4, short=20, tmp=str(tmp_path / 'tmp'), window=2)
  labels = np.load(out / 'labels.npy')
  offsets = np.load(out / 'offsets.npy')
  assert labels.tolist() == list(range(28))
  data = np.fromfile(out / 'data.bin', dtype=np.uint8)
  for i in (0, 9, 27):
    im = Image.open(io.BytesIO(data[offsets[i]:offsets[i + 1]].tobytes()))
    assert im.size[1] == 20 and abs(im.size[0] - round((40 + i) * 20 / 30)) <= 1
  assert not list((tmp_path / 'dl' / 'data').glob('*.parquet'))       # shards deleted after use


def _pack_tiny(root, n=12, seed=0):
  """A packed training set in the GPUTrainLoader format: n random JPEGs of varied sizes."""
  import torch
  from torchvision.io import encode_jpeg
  g = torch.Generator().manual_seed(seed)
  offsets, pos = [0], 0
  with open(root / 'data.bin', 'wb') as f:
    for i in range(n):
      h, w = 40 + 7 * i, 64 - 3 * i
      b = encode_jpeg(torch.randint(0, 256, (3, h, w), generator=g, dtype=torch.uint8)).numpy().tobytes()
      f.write(b)
      pos += len(b)
      offsets.append(pos)
  np.save(root / 'offsets.npy', np.array(offsets, dtype=np.int64))
  np.save(root / 'labels.npy', np.arange(n, dtype=np.int64) % 5)


def test_batch_seeded_crops_repeat_from_any_start(tmp_path):
  """With batch_seeded, an epoch begun at batch `start` (a resume or a replay) gets the same images, labels and crops
  as the uninterrupted epoch; without it, the order repeats but the crops do not."""
  import torch
  from rcdl.imagenet import GPUTrainLoader
  _pack_tiny(tmp_path)
  full = list(GPUTrainLoader(str(tmp_path), 2, 'cpu', size=16, seed=3, batch_seeded=True).epoch(1))
  part = list(GPUTrainLoader(str(tmp_path), 2, 'cpu', size=16, seed=3, batch_seeded=True).epoch(1, start=3))
  assert len(full) == 6 and len(part) == 3
  for (x1, y1, i1), (x2, y2, i2) in zip(full[3:], part):
    assert torch.equal(i1, i2) and torch.equal(y1, y2) and torch.equal(x1, x2)
  old_full = list(GPUTrainLoader(str(tmp_path), 2, 'cpu', size=16, seed=3).epoch(1))
  old_part = list(GPUTrainLoader(str(tmp_path), 2, 'cpu', size=16, seed=3).epoch(1, start=3))
  assert all(torch.equal(a[2], b[2]) for a, b in zip(old_full[3:], old_part))           # same order
  assert not all(torch.equal(a[0], b[0]) for a, b in zip(old_full[3:], old_part))       # different crops
  assert all(torch.equal(a[2], b[2]) for a, b in zip(full, old_full))                   # the order does not change
