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
from typing import Dict, List, Optional, Tuple

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


def smooth_noise(index: int, device='cpu', size: int = 256) -> torch.Tensor:
  """The fixed noise image of a noise control (rare-class experiment): 16 x 16 Gaussian noise per channel, seeded by
  the image's global index, bilinearly upsampled to size x size, as uint8 128 + 64 z (clipped). (3, size, size)."""
  g = torch.Generator().manual_seed(int(index))
  z = torch.randn(1, 3, 16, 16, generator=g)
  z = F.interpolate(z, size=(size, size), mode='bilinear', align_corners=False)[0]
  return (128 + 64 * z).round().clamp(0, 255).to(torch.uint8).to(device)


class GPUTrainLoader:
  """Epochs over the packed set: GPU JPEG decode (in its own thread, one batch ahead), RandomResizedCrop + flip to
  `size`, normalization. Labels are indexed on the GPU, so no step waits on a host-to-device copy."""

  def __init__(self, root: str, batch: int, device, size: int = 224, seed: int = 0, drop_last: bool = True,
               batch_seeded: bool = False, keep: Optional[np.ndarray] = None, labels: Optional[np.ndarray] = None,
               noise: Optional[np.ndarray] = None):
    """keep: the global indices of the images to train on (None: all). labels: a full-length label table that
    replaces labels.npy (relabelled images). Batches still yield global indices, so per-example tables index them
    directly. With keep None the epochs are the same as before. noise: global indices whose decoded pixels are replaced
    by smooth_noise(index) before augmentation (noise controls)."""
    self.data = np.memmap(os.path.join(root, 'data.bin'), dtype=np.uint8, mode='r')
    self.offsets = np.load(os.path.join(root, 'offsets.npy'))
    self.labels = np.load(os.path.join(root, 'labels.npy')) if labels is None else np.asarray(labels, np.int64)
    assert len(self.labels) == len(self.offsets) - 1, 'label table: one label per packed image'
    self.labels_gpu = torch.from_numpy(self.labels).to(device)
    self.index = np.arange(len(self.labels)) if keep is None else np.asarray(keep, np.int64)
    self.n, self.batch, self.dev, self.size = len(self.index), batch, device, size
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self.drop_last = drop_last
    self.batch_seeded = batch_seeded
    self.noise = set(int(i) for i in noise) if noise is not None and len(noise) else None
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
    perm = self.index[self.rng.permutation(self.n)]     # global indices (keep None: a permutation of all images)
    perm_gpu = torch.from_numpy(perm).to(self.dev)
    q, stop = queue.Queue(2), object()
    cuda = torch.device(self.dev).type == 'cuda'
    dstream = torch.cuda.Stream(self.dev) if cuda else None

    def decode():                                       # the random draws stay in the consumer: same order always
      try:
        for i in range(start, len(self)):
          idx = perm[i * self.batch:(i + 1) * self.batch]
          raw = [torch.from_numpy(np.array(self.data[self.offsets[j]:self.offsets[j + 1]])) for j in idx]
          if cuda:
            # The GPU decode writes on its own stream. Its outputs are allocated from a separate stream's pool, so it
            # never receives a block that training kernels still queued on the default stream are using; the
            # synchronize makes the images complete before the consumer reads them.
            with torch.cuda.stream(dstream):
              imgs = decode_jpeg(raw, mode=ImageReadMode.RGB, device=self.dev)
            torch.cuda.synchronize(self.dev)
          else:
            imgs = decode_jpeg(raw, mode=ImageReadMode.RGB, device=self.dev)
          q.put((i, idx, imgs))
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
      if cuda:
        cur = torch.cuda.current_stream(self.dev)
        for im in imgs:                                 # read on this stream: the decode pool waits before reusing
          im.record_stream(cur)
      rng = np.random.default_rng([self.seed, index, i]) if self.batch_seeded and index is not None else self.rng
      x = self._crop(self._noise(imgs, idx), rng)
      yield x, self.labels_gpu[perm_gpu[i * self.batch:(i + 1) * self.batch]], torch.from_numpy(idx)

  def _noise(self, imgs, idx):
    if self.noise is None:
      return imgs
    return [smooth_noise(int(j), self.dev) if int(j) in self.noise else im for im, j in zip(imgs, idx)]

  def _crop(self, imgs, rng) -> torch.Tensor:
    """RandomResizedCrop + flip to `size` with draws from rng, then normalization: (B, 3, size, size)."""
    top, left, ch, cw, flip = crop_params(rng, np.array([im.shape[-2:] for im in imgs]))
    x = torch.empty(len(imgs), 3, self.size, self.size, device=self.dev)
    for k, im in enumerate(imgs):
      c = im[:, top[k]:top[k] + ch[k], left[k]:left[k] + cw[k]][None].float()
      c = F.interpolate(c, size=(self.size, self.size), mode='bilinear', antialias=True, align_corners=False)[0]
      x[k] = c.flip(-1) if flip[k] else c
    return (x - self.mean) / self.std

  def decode_crop(self, idx: np.ndarray, rng) -> torch.Tensor:
    """The training augmentation of the images with global indices idx, with draws from rng (synchronous)."""
    from torchvision.io import ImageReadMode, decode_jpeg
    raw = [torch.from_numpy(np.array(self.data[self.offsets[j]:self.offsets[j + 1]])) for j in idx]
    cuda = torch.device(self.dev).type == 'cuda'
    if cuda:      # the decode writes on its own stream: no queued kernel may still use the memory it receives
      torch.cuda.synchronize(self.dev)
    imgs = decode_jpeg(raw, mode=ImageReadMode.RGB, device=self.dev)
    if cuda:      # and the images must be complete before the crops read them
      torch.cuda.synchronize(self.dev)
    return self._crop(self._noise(imgs, idx), rng)


class StratifiedReplay:
  """A replay stream for fine-tuning: per step, `spec[c]` images of each oversampled class c plus `rest` images drawn
  class-first (uniform over rest_classes) then image-uniform, with the training augmentation. Everything is a function
  of (seed, step), so two runs (taught and untaught) see the same stream, and a resumed run repeats it.
  Oversampled classes walk a per-class permutation, a new one each cycle (default_rng([seed, c, cycle])); crops come
  from default_rng([seed, 1_000_000 + step]) and the rest draws from default_rng([seed, 1_000_001, step])."""

  def __init__(self, loader: GPUTrainLoader, spec: Dict[int, int], rest: int, rest_classes, seed: int):
    self.loader, self.spec, self.rest, self.seed = loader, dict(spec), rest, seed
    self.rest_classes = np.asarray(rest_classes, np.int64)
    order = np.argsort(loader.labels, kind='stable')
    bounds = np.searchsorted(loader.labels[order], np.arange(loader.labels.max() + 2))
    self.by_class = [order[bounds[c]:bounds[c + 1]] for c in range(len(bounds) - 1)]
    self.size = sum(self.spec.values()) + rest

  def indices(self, step: int) -> np.ndarray:
    out = []
    for c, n in self.spec.items():
      imgs = self.by_class[c]
      for p in range(step * n, (step + 1) * n):
        cycle, pos = divmod(p, len(imgs))
        out.append(imgs[np.random.default_rng([self.seed, c, cycle]).permutation(len(imgs))[pos]])
    rng = np.random.default_rng([self.seed, 1_000_001, step])
    for c in rng.choice(self.rest_classes, self.rest):
      out.append(self.by_class[c][rng.integers(len(self.by_class[c]))])
    return np.array(out, np.int64)

  def batch(self, step: int):
    """(x, y, idx) of step `step`: images (size, 3, 224, 224) normalized, labels on the device, global indices."""
    idx = self.indices(step)
    x = self.loader.decode_crop(idx, np.random.default_rng([self.seed, 1_000_000 + step]))
    return x, self.loader.labels_gpu[torch.from_numpy(idx).to(self.loader.dev)], idx

  def slot_groups(self, step: int, r: int, n_groups: int = 8) -> np.ndarray:
    """Pseudo-groups for shadow r at this step: the replay slots in a random order (default_rng([seed, 2_000_000 +
    8 step + r])) dealt into n_groups groups of near-equal size, numbered 1..n_groups. The same in every run with
    this seed."""
    perm = np.random.default_rng([self.seed, 2_000_000 + 8 * step + r]).permutation(self.size)
    g = np.empty(self.size, np.int64)
    g[perm] = 1 + np.arange(self.size) * n_groups // self.size
    return g


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
