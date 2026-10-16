"""E0 helper: montage of a channel's strongest (or most negative) validation images, center crops 224, with labels."""

import argparse
import glob
import io
import os

import numpy as np


def montage(stats: str, val_dir: str, imagenet_dir: str, layer: str, channel: int, kind: str, k: int, out: str,
            weights: str = None, only: str = None):
  from PIL import Image, ImageDraw
  import pyarrow.parquet as pq
  from experiments.inception_e0 import class_names, concat_stats
  st = np.load(stats)
  a = concat_stats(st, layer, kind)[:, channel]
  names = class_names(imagenet_dir)
  order = np.argsort(-a if kind == 'max' else a)
  if only:                                             # keep one class group: 'vehicles', 'dogs' or a class name
    from experiments.inception_paths import DOGS, VEHICLES
    keep = {'vehicles': {i for i, n in enumerate(names) if n in VEHICLES}, 'dogs': DOGS}.get(
      only, {i for i, n in enumerate(names) if n == only})
    order = order[np.isin(st['labels'][order], list(keep))]
  idx = order[:k]
  files = sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True))
  sizes = [pq.ParquetFile(f).metadata.num_rows for f in files]
  starts = np.cumsum([0] + sizes)
  cols, cell = 8, 160
  canvas = Image.new('RGB', (cols * cell, ((k + cols - 1) // cols) * (cell + 14)), 'white')
  draw = ImageDraw.Draw(canvas)
  tables = {}
  model = None
  if weights:                                          # heat overlay: where in the crop the channel responds
    import torch
    import torch.nn.functional as F
    from experiments.inception_e0 import channel_index, load_model
    model = load_model(weights, 'cpu')
    branch, ch = channel_index(model, layer, channel)
    grab = {}
    getattr(model, branch).register_forward_hook(lambda m, i_, o: grab.__setitem__('a', o[0, ch]))
  for n, i in enumerate(idx):
    fi = int(np.searchsorted(starts, i, side='right') - 1)
    if fi not in tables:
      tables[fi] = pq.read_table(files[fi], columns=['image', 'label'])
    row = tables[fi].slice(int(i - starts[fi]), 1).to_pylist()[0]
    im = Image.open(io.BytesIO(row['image']['bytes'])).convert('RGB')
    w, h = im.size
    s = 256 / min(w, h)
    im = im.resize((max(256, round(w * s)), max(256, round(h * s))), Image.BILINEAR)
    W, H = im.size
    im = im.crop(((W - 224) // 2, (H - 224) // 2, (W - 224) // 2 + 224, (H - 224) // 2 + 224))
    if model is not None:
      x = torch.from_numpy(np.asarray(im, dtype=np.float32)).permute(2, 0, 1)[None] - 117.0
      with torch.no_grad():
        model(x)
      m = grab['a'][None, None]
      m = (m if kind == 'max' else -m).clamp(min=0)
      m = F.interpolate(m, size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
      m = (m / m.max().clamp(min=1e-6)).numpy()
      arr = np.asarray(im, dtype=np.float32)
      arr = arr * (0.35 + 0.65 * m[..., None])                       # dim where the channel is quiet
      arr[..., 0] = np.maximum(arr[..., 0], 255 * m * 0.6)           # and tint the response red
      im = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
    im = im.resize((cell, cell))
    x, y = (n % cols) * cell, (n // cols) * (cell + 14)
    canvas.paste(im, (x, y))
    draw.text((x + 2, y + cell), f'{names[row["label"]][:18]} {a[i]:.0f}', fill='black')
  canvas.save(out, quality=85)
  print('saved', out)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--stats', required=True)
  p.add_argument('--val', required=True)
  p.add_argument('--imagenet', required=True)
  p.add_argument('--layer', required=True)
  p.add_argument('--channel', type=int, required=True)
  p.add_argument('--kind', default='max', choices=['max', 'min'])
  p.add_argument('--k', type=int, default=32)
  p.add_argument('--out', required=True)
  p.add_argument('--weights', default=None, help='inception5h.pth: overlay the channel\'s spatial response')
  p.add_argument('--only', default=None, help="'vehicles', 'dogs' or one class name")
  a = p.parse_args()
  montage(a.stats, a.val, a.imagenet, a.layer, a.channel, a.kind, a.k, a.out, a.weights, a.only)
