"""CIFAR-10 loading, standardization and augmentation (paper App. E.1.1, notebook Section 2)."""

import io
import os
import tarfile
import urllib.request
from typing import Tuple

import numpy as np
import torch

CIFAR_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'
HF_URL = 'https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/{split}-00000-of-00001.parquet'
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)
CIFAR10_CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def _read_binary(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Reads the official binary release (same files and order as the torch notebook)."""
  data_dir = os.path.join(root, 'cifar-10-batches-bin')
  archive = os.path.join(root, 'cifar-10-binary.tar.gz')
  if not os.path.exists(os.path.join(data_dir, 'test_batch.bin')):
    if not os.path.exists(archive):
      os.makedirs(root, exist_ok=True)
      urllib.request.urlretrieve(CIFAR_URL, archive)
    with tarfile.open(archive) as tar:
      tar.extractall(root, filter='data')

  def read(files):
    raw = np.concatenate([np.fromfile(f, dtype=np.uint8) for f in files]).reshape(-1, 3073)
    return raw[:, 1:].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).copy(), raw[:, 0].astype(np.int64)

  train_u8, train_y = read([os.path.join(data_dir, f'data_batch_{i}.bin') for i in range(1, 6)])
  test_u8, test_y = read([os.path.join(data_dir, 'test_batch.bin')])
  return train_u8, train_y, test_u8, test_y


def _read_parquet(root: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Reads the Hugging Face mirror (uoft-cs/cifar10, PNG-encoded rows)."""
  import pyarrow.parquet as pq
  from PIL import Image

  out = []
  for split in ('train', 'test'):
    path = os.path.join(root, 'hf', f'{split}.parquet')
    if not os.path.exists(path):
      os.makedirs(os.path.dirname(path), exist_ok=True)
      urllib.request.urlretrieve(HF_URL.format(split=split), path)
    table = pq.read_table(path).to_pydict()
    imgs = np.stack([np.asarray(Image.open(io.BytesIO(r['bytes'])).convert('RGB')) for r in table['img']])
    out += [imgs.astype(np.uint8), np.asarray(table['label'], dtype=np.int64)]
  return tuple(out)


def load_cifar10_uint8(root: str = 'data', source: str = 'auto') -> Tuple[np.ndarray, ...]:
  """Returns (train_u8 [50000,32,32,3], train_y, test_u8 [10000,32,32,3], test_y), cached as .npz in `root`.

  source: 'binary' (official tarball), 'parquet' (HF mirror) or 'auto' (binary if extracted or complete, else parquet).
  The cache file records the source, since the image order of the two sources is checked, not assumed (tests).
  """
  cache = os.path.join(root, f'cifar10_u8_{source}.npz')
  if os.path.exists(cache):
    z = np.load(cache)
    return z['train_x'], z['train_y'], z['test_x'], z['test_y']
  if source == 'auto':
    has_bin = os.path.exists(os.path.join(root, 'cifar-10-batches-bin', 'test_batch.bin'))
    source = 'binary' if has_bin else 'parquet'
  arrays = _read_binary(root) if source == 'binary' else _read_parquet(root)
  np.savez(cache, train_x=arrays[0], train_y=arrays[1], test_x=arrays[2], test_y=arrays[3])
  return arrays


def standardize(u8: np.ndarray) -> np.ndarray:
  """Per-channel standardization, flattened HWC order to 3072 features (as in the notebooks)."""
  x = u8.astype(np.float32) / 255.0
  return ((x - CIFAR_MEAN) / CIFAR_STD).reshape(len(u8), -1)


def unnormalize_cifar(flat_imgs: np.ndarray) -> np.ndarray:
  """Inverts the standardization back to [0, 1] RGB (..., 32, 32, 3)."""
  return np.clip(flat_imgs.reshape((-1, 32, 32, 3)) * CIFAR_STD + CIFAR_MEAN, 0.0, 1.0)


def augment_epoch(u8: np.ndarray, gen: np.random.Generator) -> np.ndarray:
  """Paper App. E.1.1 augmentation for multi-epoch runs: reflect-pad 4, random 32x32 crop, random horizontal flip.

  Assumption: the paper's "epoch-wise horizontal flip" is read as one random flip per image per epoch.
  """
  n = len(u8)
  padded = np.pad(u8, ((0, 0), (4, 4), (4, 4), (0, 0)), mode='reflect')
  ox, oy = gen.integers(0, 9, size=n), gen.integers(0, 9, size=n)
  rows = oy[:, None] + np.arange(32)[None, :]
  cols = ox[:, None] + np.arange(32)[None, :]
  out = padded[np.arange(n)[:, None, None], rows[:, :, None], cols[:, None, :]]
  flip = gen.random(n) < 0.5
  out[flip] = out[flip, :, ::-1]
  return out


def load_cifar10_tensors(root: str = 'data', source: str = 'auto', device: str = 'cpu'):
  """Standardized, flattened float32 tensors (train_x, train_y, test_x, test_y) and the raw uint8 train images."""
  train_u8, train_y, test_u8, test_y = load_cifar10_uint8(root, source)
  tensors = [torch.as_tensor(standardize(train_u8), device=device), torch.as_tensor(train_y, device=device),
             torch.as_tensor(standardize(test_u8), device=device), torch.as_tensor(test_y, device=device)]
  return tensors, train_u8
