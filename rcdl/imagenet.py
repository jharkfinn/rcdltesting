"""ImageNet-1k (ILSVRC/imagenet-1k on Hugging Face, gated) for the ConvNeXt-SwiGLU runs, kept in RAM.

The molab sandboxes have no documented disk quota, and one ended while 146 GB of training parquet were being written to
local disk, so the training set is streamed shard by shard from the Hub and stored, shorter side resized to 256 and
re-encoded as JPEG (quality 95), in one packed file in RAM-backed /dev/shm (about 45 GB). Each shard's parquet is deleted
after use. This is a documented deviation from the essay, which crops from the original images; the essay's own
validation geometry is 256 -> 224 center crops.

Training batches are decoded on the GPU (nvjpeg), cropped with RandomResizedCrop (scale 0.08-1, ratio 3/4-4/3), flipped,
resized to 224 and normalized with the ImageNet mean and std; mixup (images and labels with the same weight, the
"corrected" recipe of the essay's fresh runs) is applied by the trainer.
"""

import io
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255
STD = torch.tensor([0.229, 0.224, 0.225]) * 255
REPO = 'ILSVRC/imagenet-1k'


def _shrink(jpeg: bytes, short: int = 256, quality: int = 95) -> bytes:
  from PIL import Image
  im = Image.open(io.BytesIO(jpeg))
  im.draft('RGB', (short, short))                     # JPEG DCT scaling: decode at >= the target size, much faster
  im = im.convert('RGB')
  w, h = im.size
  s = short / min(w, h)
  if s < 1:
    im = im.resize((max(short, round(w * s)), max(short, round(h * s))), Image.BICUBIC, reducing_gap=None)
  buf = io.BytesIO()
  im.save(buf, format='JPEG', quality=quality)
  return buf.getvalue()


def _process_shard(args) -> Tuple[List[bytes], List[int]]:
  path, short, quality = args
  import pyarrow.parquet as pq
  t = pq.read_table(path)
  imgs, labels = t.column('image').to_pylist(), t.column('label').to_pylist()
  return [_shrink(d['bytes'], short, quality) for d in imgs], labels


def prepare_train(out_dir: str = '/dev/shm/imagenet256', token: Optional[str] = None, workers: int = 18,
                  short: int = 256, quality: int = 95, tmp: str = '/marimo/work/tmp_shards') -> None:
  """Streams every train shard, shrinks its images and appends them to out_dir/{data.bin, offsets.npy, labels.npy}."""
  from huggingface_hub import HfApi, hf_hub_download
  os.makedirs(out_dir, exist_ok=True)
  os.makedirs(tmp, exist_ok=True)
  files = sorted(f for f in HfApi(token=token).list_repo_files(REPO, repo_type='dataset') if f.startswith('data/train-'))
  offsets, labels, pos, t0 = [0], [], 0, time.time()
  with open(os.path.join(out_dir, 'data.bin'), 'wb') as out, ProcessPoolExecutor(workers) as pool:
    pending = []
    for i, f in enumerate(files):
      path = hf_hub_download(REPO, f, repo_type='dataset', token=token, local_dir=tmp)
      pending.append((path, pool.submit(_process_shard, (path, short, quality))))
      while pending and (len(pending) >= 3 or i == len(files) - 1):
        p, fut = pending.pop(0)
        blobs, labs = fut.result()
        for b in blobs:
          out.write(b)
          pos += len(b)
          offsets.append(pos)
        labels += labs
        os.remove(p)
      print(f'{i + 1}/{len(files)} shards, {len(labels)} images, {pos / 1e9:.1f} GB, {time.time() - t0:.0f}s', flush=True)
  np.save(os.path.join(out_dir, 'offsets.npy'), np.array(offsets, dtype=np.int64))
  np.save(os.path.join(out_dir, 'labels.npy'), np.array(labels, dtype=np.int64))
  open(os.path.join(out_dir, 'DONE'), 'w').write(f'{len(labels)} images')


class GPUTrainLoader:
  """Epochs over the packed set: GPU JPEG decode, RandomResizedCrop + flip to `size`, normalization."""

  def __init__(self, root: str, batch: int, device, size: int = 224, seed: int = 0, drop_last: bool = True):
    self.data = np.memmap(os.path.join(root, 'data.bin'), dtype=np.uint8, mode='r')
    self.offsets = np.load(os.path.join(root, 'offsets.npy'))
    self.labels = np.load(os.path.join(root, 'labels.npy'))
    self.n, self.batch, self.dev, self.size = len(self.labels), batch, device, size
    self.rng = np.random.default_rng(seed)
    self.drop_last = drop_last
    self.mean, self.std = MEAN.to(device)[:, None, None], STD.to(device)[:, None, None]

  def __len__(self):
    return self.n // self.batch if self.drop_last else math.ceil(self.n / self.batch)

  def _crop(self, im: torch.Tensor) -> torch.Tensor:
    _, h, w = im.shape
    area = h * w
    for _ in range(10):
      a = area * self.rng.uniform(0.08, 1.0)
      r = math.exp(self.rng.uniform(math.log(3 / 4), math.log(4 / 3)))
      cw, ch = int(round(math.sqrt(a * r))), int(round(math.sqrt(a / r)))
      if 0 < cw <= w and 0 < ch <= h:
        top, left = self.rng.integers(0, h - ch + 1), self.rng.integers(0, w - cw + 1)
        break
    else:                                               # fallback: center crop of the whole image
      ch, cw = min(h, w), min(h, w)
      top, left = (h - ch) // 2, (w - cw) // 2
    c = im[:, top:top + ch, left:left + cw][None].float()
    c = F.interpolate(c, size=(self.size, self.size), mode='bilinear', antialias=True, align_corners=False)[0]
    return c.flip(-1) if self.rng.random() < 0.5 else c

  def epoch(self):
    from torchvision.io import ImageReadMode, decode_jpeg
    perm = self.rng.permutation(self.n)
    for i in range(len(self)):
      idx = perm[i * self.batch:(i + 1) * self.batch]
      raw = [torch.from_numpy(np.array(self.data[self.offsets[j]:self.offsets[j + 1]])) for j in idx]
      imgs = decode_jpeg(raw, mode=ImageReadMode.RGB, device=self.dev)
      x = torch.stack([self._crop(im) for im in imgs])
      x = (x - self.mean) / self.std
      yield x, torch.from_numpy(self.labels[idx]).to(self.dev), torch.from_numpy(idx)


def val_batches(val_dir: str, batch: int, device, size: int = 256, crop: int = 224):
  """ImageNet validation: shorter side to 256 (antialiased bilinear), center crop 224, normalized; GPU decode."""
  import glob
  import pyarrow.parquet as pq
  from torchvision.io import ImageReadMode, decode_image, decode_jpeg
  mean, std = MEAN.to(device)[:, None, None], STD.to(device)[:, None, None]
  for f in sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True)):
    t = pq.read_table(f)
    imgs, labels = t.column('image').to_pylist(), t.column('label').to_pylist()
    for i in range(0, len(labels), batch):
      datas = [torch.frombuffer(bytearray(d['bytes']), dtype=torch.uint8) for d in imgs[i:i + batch]]
      out = []
      for d in datas:
        try:
          im = decode_jpeg(d, mode=ImageReadMode.RGB, device=device)
        except RuntimeError:
          im = decode_image(d, mode=ImageReadMode.RGB).to(device)
        h, w = im.shape[-2:]
        s = size / min(h, w)
        im = F.interpolate(im[None].float(), size=(max(size, round(h * s)), max(size, round(w * s))), mode='bilinear',
                           antialias=True, align_corners=False)[0]
        H, W = im.shape[-2:]
        top, left = (H - crop) // 2, (W - crop) // 2
        out.append(im[:, top:top + crop, left:left + crop])
      x = (torch.stack(out) - mean) / std
      yield x, torch.tensor(labels[i:i + batch], device=device)
