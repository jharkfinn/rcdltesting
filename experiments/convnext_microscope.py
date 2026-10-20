"""Example sheets of one block of a ConvNeXt-SwiGLU run, in the format of the essay's ConvNeXt microscopes, for the blind
audit (experiments/microscope_audit.py).

For block b, every unit n and, with --dirs D, D random dense directions d over the block's U units (seeded Gaussian, unit
norm; their response is u_d = sum_n d_n u_n), over the 50,000 validation images (shorter side 256, center crop 224, the
order of rcdl.imagenet.val_batches; image index i is the position in that order):
  max_i, pmax_i = the largest post-SwiGLU product u over the block's positions on image i, and its position
  min_i, pmin_i = the smallest, and its position
Buckets as in the essay's ConvNeXt microscope meta.json: top16 = ranks 1-16 by max_i (descending), p9999 = ranks 5-12,
p999 = ranks 47-54, p99 = ranks 497-504, bottom4 = ranks 1-4 by min_i (ascending). Positions are row-major on the block's
grid (stride s = 224 / grid). Crop: the stage's focus size (64 / 96 / 144 / 192 px for stages 0-3), centered on the
position's receptive-field center ((r + 0.5) s, (c + 0.5) s) and moved inside the 224-px image; resized to 96 px.
Sheet: 16 x 3 cells of 96 px, JPEG quality 90: row 0 the top 16; row 1 p9999 then p999; row 2 p99 then bottom4 (the last
4 cells blank). Data: data_{layer}.json as the essay's (neurons[c].examples[bucket].cells[imageIndex, datasetLabel, rank,
value, position]); channels U..U+D-1 are the random directions (their vectors in dirs_{layer}.npy).

  python3 -m experiments.convnext_microscope --run results/e1_tiny_adamw --step 25020 --block stage3.block2 --dirs 32 \
      --val /marimo/work/data/imagenet/data --imagenet /marimo/work/data/imagenet --out results/e1_tiny_adamw/microscope
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.convnext import build_convnext
from rcdl.imagenet import MEAN, STD

BUCKETS = (('top16', 'max', 0, 16), ('p9999', 'max', 4, 12), ('p999', 'max', 46, 54), ('p99', 'max', 496, 504),
           ('bottom4', 'min', 0, 4))
SHEET_POS = {'top16': [(0, i) for i in range(16)], 'p9999': [(1, i) for i in range(8)],
             'p999': [(1, 8 + i) for i in range(8)], 'p99': [(2, i) for i in range(8)], 'bottom4': [(2, 8 + i) for i in range(4)]}
FOCUS = (64, 96, 144, 192)
CELL = 96


def random_dirs(U, D, seed=0):
  d = np.random.default_rng(seed).standard_normal((D, U))
  return d / np.linalg.norm(d, axis=1, keepdims=True)


@torch.no_grad()
def collect(model, block, batches, dirs=None):
  """Per image and channel (units, then directions): max and min of u over positions and their row-major positions.
  batches yields (x, y). Returns dict of numpy arrays maxv, minv (N, C) float32, pmax, pmin (N, C) int32, labels (N,),
  and grid (H, W)."""
  name_i, name_j = (int(t) for t in block.replace('stage', '').split('.block'))
  blk = model.stages[name_i][name_j]
  got = {}
  h = blk.pw_down.register_forward_pre_hook(lambda m, inp: got.__setitem__('u', inp[0]))
  D = None if dirs is None else torch.as_tensor(dirs, dtype=torch.float32)
  out = {k: [] for k in ('maxv', 'minv', 'pmax', 'pmin', 'labels')}
  correct = 0
  try:
    for x, y in batches:
      correct += int((model(x).argmax(1) == y).sum())
      u = got['u'].float()                                         # (B, H, W, U)
      B, H, W, U = u.shape
      if D is not None:
        u = torch.cat([u, u @ D.to(u.device).T], -1)
      flat = u.reshape(B, H * W, -1)
      mx, pm = flat.max(1)
      mn, pn = flat.min(1)
      for k, v in (('maxv', mx), ('minv', mn), ('pmax', pm), ('pmin', pn), ('labels', y)):
        out[k].append(v.cpu())
      grid = (H, W)
  finally:
    h.remove()
  res = {k: torch.cat(v).numpy() for k, v in out.items()}
  for k in ('pmax', 'pmin'):
    res[k] = res[k].astype(np.int32)
  res['grid'] = grid
  res['top1'] = correct / max(len(res['labels']), 1)                  # the model's top-1 on these images
  return res


def select(S):
  """{channel: {bucket: [(image, rank, value, position)]}} by the microscope's bucket rules."""
  N, C = S['maxv'].shape
  sel = {}
  for c in range(C):
    order_max = np.argsort(-S['maxv'][:, c], kind='stable')
    order_min = np.argsort(S['minv'][:, c], kind='stable')
    rec = {}
    for b, src, lo, hi in BUCKETS:
      order, vals, pos = (order_max, S['maxv'], S['pmax']) if src == 'max' else (order_min, S['minv'], S['pmin'])
      rec[b] = [(int(i), r + 1, float(vals[i, c]), int(pos[i, c])) for r, i in zip(range(lo, hi), order[lo:hi])]
    sel[c] = rec
  return sel


def crop_box(position, grid, stage, size=224):
  """(top, left, side) of the focus crop centred on a grid position's receptive-field centre, inside the image."""
  H, W = grid
  r, c = divmod(position, W)
  s = size / H
  side = min(FOCUS[stage], size)
  cy, cx = (r + 0.5) * s, (c + 0.5) * s
  top = int(round(min(max(cy - side / 2, 0), size - side)))
  left = int(round(min(max(cx - side / 2, 0), size - side)))
  return top, left, side


@torch.no_grad()
def gather_crops(batches, sel, grid, stage):
  """{(channel, bucket, k): (CELL, CELL, 3) uint8} from a second pass over the same images."""
  want = {}
  for c, rec in sel.items():
    for b, cells in rec.items():
      for k, (i, _, _, pos) in enumerate(cells):
        want.setdefault(i, []).append((c, b, k, pos))
  crops, i0 = {}, 0
  mean, std = MEAN[:, None, None], STD[:, None, None]
  for x, _ in batches:
    for j in range(x.shape[0]):
      i = i0 + j
      if i not in want:
        continue
      img = (x[j].cpu() * std + mean).clamp(0, 255)                 # (3, 224, 224), pixel values
      for c, b, k, pos in want[i]:
        top, left, side = crop_box(pos, grid, stage, img.shape[-1])
        cr = F.interpolate(img[None, :, top:top + side, left:left + side], size=(CELL, CELL), mode='bilinear',
                           antialias=True, align_corners=False)[0]
        crops[c, b, k] = cr.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    i0 += x.shape[0]
  return crops


def write(out, layer, sel, crops, labels, names, n_units, S):
  from PIL import Image
  os.makedirs(os.path.join(out, 'examples', layer), exist_ok=True)
  neurons = []
  for c, rec in sel.items():
    sheet = Image.new('RGB', (16 * CELL, 3 * CELL), (240, 236, 226))
    examples = []
    for b, _, _, _ in BUCKETS:
      cells = []
      for k, (i, rank, val, pos) in enumerate(rec[b]):
        r, col = SHEET_POS[b][k]
        if (c, b, k) in crops:
          sheet.paste(Image.fromarray(crops[c, b, k]), (col * CELL, r * CELL))
        cells.append({'imageIndex': i, 'datasetLabel': names[int(labels[i])], 'rank': rank, 'value': val, 'position': pos})
      examples.append({'bucketId': b, 'cells': cells})
    sheet.save(os.path.join(out, 'examples', layer, f'{c:04d}.jpg'), quality=90)
    mx, mn = S['maxv'][:, c], S['minv'][:, c]
    neurons.append({'channel': c, 'kind': 'unit' if c < n_units else 'random_direction',
                    'stats': {'max': float(mx.max()), 'min': float(mn.min()), 'p999': float(np.quantile(mx, 0.999)),
                              'p99': float(np.quantile(mx, 0.99)), 'mean_max': float(mx.mean())},
                    'examples': examples})
  json.dump({'layer': layer, 'channels': len(neurons), 'units': n_units,
             'imageCount': int(len(labels)), 'top1': S.get('top1'), 'neurons': neurons}, open(os.path.join(out, f'data_{layer}.json'), 'w'))


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', required=True)
  p.add_argument('--step', type=int, required=True)
  p.add_argument('--block', default='stage3.block2')
  p.add_argument('--dirs', type=int, default=0, help='random dense directions over the block\'s units')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--val', required=True, help='directory with validation-*.parquet')
  p.add_argument('--imagenet', required=True, help='directory with classes.py (label names)')
  p.add_argument('--batch', type=int, default=250)
  p.add_argument('--out', required=True)
  a = p.parse_args()
  from experiments.inception_e0 import class_names
  from rcdl.imagenet import val_batches
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  model = build_convnext(cfg['size']).to(dev).eval()
  model.load_state_dict(torch.load(os.path.join(a.run, f'model_{a.step:05d}.pt'), map_location=dev))
  stage, j = (int(t) for t in a.block.replace('stage', '').split('.block'))
  U = model.stages[stage][j].pw_down.in_features
  dirs = random_dirs(U, a.dirs, a.seed) if a.dirs else None
  S = collect(model, a.block, val_batches(a.val, a.batch, dev), dirs)
  print('collected', S['maxv'].shape, 'grid', S['grid'], 'top1', round(S['top1'], 5), flush=True)
  sel = select(S)
  crops = gather_crops(val_batches(a.val, a.batch, dev), sel, S['grid'], stage)
  layer = a.block.replace('.', '_')
  os.makedirs(a.out, exist_ok=True)
  if dirs is not None:
    np.save(os.path.join(a.out, f'dirs_{layer}.npy'), dirs)
  np.savez_compressed(os.path.join(a.out, f'stats_{layer}.npz'), **{k: v for k, v in S.items() if k not in ('grid', 'top1')})
  write(a.out, layer, sel, crops, S['labels'], class_names(a.imagenet), U, S)
  print('MICROSCOPE_DONE', len(sel), 'channels', flush=True)


if __name__ == '__main__':
  main()
