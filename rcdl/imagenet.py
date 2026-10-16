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


def _process_rows(args) -> Tuple[List[bytes], List[int]]:
  path, lo, hi, short, quality = args
  import pyarrow.parquet as pq
  t = pq.read_table(path).slice(lo, hi - lo)
  imgs, labels = t.column('image').to_pylist(), t.column('label').to_pylist()
  return [_shrink(d['bytes'], short, quality) for d in imgs], labels


def prepare_train(out_dir: str = '/dev/shm/imagenet256', token: Optional[str] = None, workers: int = 18,
                  short: int = 256, quality: int = 95, tmp: str = '/marimo/work/tmp_shards', window: int = 3) -> None:
  """Streams every train shard, shrinks its images and appends them to out_dir/{data.bin, offsets.npy, labels.npy}.
  At most `window` shards are on disk at once; each is split into row chunks so that all workers are busy, and the
  chunks are written back in order (offsets and labels stay aligned with the shard order)."""
  import pyarrow.parquet as pq
  from huggingface_hub import HfApi, hf_hub_download
  os.makedirs(out_dir, exist_ok=True)
  os.makedirs(tmp, exist_ok=True)
  files = sorted(f for f in HfApi(token=token).list_repo_files(REPO, repo_type='dataset') if f.startswith('data/train-'))
  per_shard = max(1, math.ceil(workers / window))
  offsets, labels, pos, t0 = [0], [], 0, time.time()
  with open(os.path.join(out_dir, 'data.bin'), 'wb') as out, ProcessPoolExecutor(workers) as pool:
    pending = []
    for i, f in enumerate(files):
      path = hf_hub_download(REPO, f, repo_type='dataset', token=token, local_dir=tmp)
      n = pq.ParquetFile(path).metadata.num_rows
      step = math.ceil(n / per_shard)
      pending.append((path, [pool.submit(_process_rows, (path, lo, min(lo + step, n), short, quality))
                             for lo in range(0, n, step)]))
      while pending and (len(pending) >= window or i == len(files) - 1):
        p, futs = pending.pop(0)
        for fut in futs:
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


def crop_params(rng, hw: np.ndarray, scale=(0.08, 1.0), ratio=(3 / 4, 4 / 3), attempts: int = 10):
  """torchvision RandomResizedCrop.get_params for a batch of (height, width): up to 10 draws per image, the first
  valid one is used; otherwise torchvision's fallback (the whole image if its aspect ratio is in range, else a center
  crop clamped to the range). Returns top, left, height, width, flip (B,) arrays; flip is the random horizontal flip."""
  B = len(hw)
  h, w = hw[:, 0].astype(np.int64), hw[:, 1].astype(np.int64)
  area = (h * w).astype(np.float64)
  a = area[:, None] * rng.uniform(scale[0], scale[1], (B, attempts))
  r = np.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1]), (B, attempts)))
  cw, ch = np.round(np.sqrt(a * r)).astype(np.int64), np.round(np.sqrt(a / r)).astype(np.int64)
  ok = (cw > 0) & (cw <= w[:, None]) & (ch > 0) & (ch <= h[:, None])
  found, first = ok.any(1), ok.argmax(1)
  rows = np.arange(B)
  cw, ch = cw[rows, first], ch[rows, first]
  ar = w / h
  fw = np.where(ar < ratio[0], w, np.where(ar > ratio[1], np.round(h * ratio[1]).astype(np.int64), w))
  fh = np.where(ar < ratio[0], np.round(w / ratio[0]).astype(np.int64), h)
  cw, ch = np.where(found, cw, fw), np.where(found, ch, fh)
  u, v = rng.random(B), rng.random(B)
  top = np.where(found, np.floor(u * (h - ch + 1)).astype(np.int64), (h - ch) // 2)
  left = np.where(found, np.floor(v * (w - cw + 1)).astype(np.int64), (w - cw) // 2)
  return top, left, ch, cw, rng.random(B) < 0.5


class GPUTrainLoader:
  """Epochs over the packed set: GPU JPEG decode (in its own thread, one batch ahead), RandomResizedCrop + flip to
  `size`, normalization. Labels are indexed on the GPU, so no step waits on a host-to-device copy."""

  def __init__(self, root: str, batch: int, device, size: int = 224, seed: int = 0, drop_last: bool = True,
               batch_seeded: bool = False):
    self.data = np.memmap(os.path.join(root, 'data.bin'), dtype=np.uint8, mode='r')
    self.offsets = np.load(os.path.join(root, 'offsets.npy'))
    self.labels = np.load(os.path.join(root, 'labels.npy'))
    self.labels_gpu = torch.from_numpy(self.labels).to(device)
    self.n, self.batch, self.dev, self.size = len(self.labels), batch, device, size
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self.drop_last = drop_last
    self.batch_seeded = batch_seeded
    self.mean, self.std = MEAN.to(device)[:, None, None], STD.to(device)[:, None, None]

  def __len__(self):
    return self.n // self.batch if self.drop_last else math.ceil(self.n / self.batch)

  def epoch(self, index: int = None, start: int = 0):
    """One epoch of batches. With `index`, the order and the augmentation draws come from a generator seeded by
    (seed, index), so a resumed run can rebuild epoch `index` and begin at batch `start` (the crops of the resumed
    batches then differ from those of an uninterrupted run; the order does not). With batch_seeded, the crop and flip
    draws of batch i come from a generator seeded by (seed, index, i) instead, so a run resumed or replayed from any
    batch gets the same crops as the uninterrupted run."""
    import queue
    import threading
    from torchvision.io import ImageReadMode, decode_jpeg
    if index is not None:
      self.rng = np.random.default_rng([self.seed, index])
    perm = self.rng.permutation(self.n)
    perm_gpu = torch.from_numpy(perm).to(self.dev)
    q, stop = queue.Queue(2), object()

    def decode():                                       # the random draws stay in the consumer: same order always
      try:
        for i in range(start, len(self)):
          idx = perm[i * self.batch:(i + 1) * self.batch]
          raw = [torch.from_numpy(np.array(self.data[self.offsets[j]:self.offsets[j + 1]])) for j in idx]
          q.put((i, idx, decode_jpeg(raw, mode=ImageReadMode.RGB, device=self.dev)))
      except BaseException as e:
        q.put(e)
        return
      q.put(stop)

    threading.Thread(target=decode, daemon=True).start()
    while True:
      item = q.get()
      if item is stop:
        return
      if isinstance(item, BaseException):
        raise item
      i, idx, imgs = item
      rng = np.random.default_rng([self.seed, index, i]) if self.batch_seeded and index is not None else self.rng
      top, left, ch, cw, flip = crop_params(rng, np.array([im.shape[-2:] for im in imgs]))
      x = torch.empty(len(imgs), 3, self.size, self.size, device=self.dev)
      for k, im in enumerate(imgs):
        c = im[:, top[k]:top[k] + ch[k], left[k]:left[k] + cw[k]][None].float()
        c = F.interpolate(c, size=(self.size, self.size), mode='bilinear', antialias=True, align_corners=False)[0]
        x[k] = c.flip(-1) if flip[k] else c
      x = (x - self.mean) / self.std
      yield x, self.labels_gpu[perm_gpu[i * self.batch:(i + 1) * self.batch]], torch.from_numpy(idx)


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
      # Decode the whole batch first and keep every decoded image referenced, then synchronize: a GPU decode writes on
      # its own stream, and freeing its output early let the allocator hand that memory to a resize while the decode
      # was still writing (about 120 of 50,000 images got non-finite pixels when the GPU was busy).
      decoded = []
      for d in datas:
        try:
          decoded.append(decode_jpeg(d, mode=ImageReadMode.RGB, device=device))
        except RuntimeError:
          decoded.append(decode_image(d, mode=ImageReadMode.RGB).to(device))
      if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)
      out = []
      for im in decoded:
        h, w = im.shape[-2:]
        s = size / min(h, w)
        im = F.interpolate(im[None].float(), size=(max(size, round(h * s)), max(size, round(w * s))), mode='bilinear',
                           antialias=True, align_corners=False)[0]
        H, W = im.shape[-2:]
        top, left = (H - crop) // 2, (W - crop) // 2
        out.append(im[:, top:top + crop, left:left + crop])
      x = (torch.stack(out) - mean) / std
      yield x, torch.tensor(labels[i:i + batch], device=device)
