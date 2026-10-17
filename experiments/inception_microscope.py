"""InceptionV1 example sheets for a subset of units and for random directions, rendered by one renderer, in the format of
experiments/convnext_microscope.py, for the blind audit's positive control (experiments/microscope_audit.py --local).

Question: can the audit tell a model's units from random directions in the same layer? For layer L (mixed4c, mixed5b)
of the public InceptionV1 (inception5h, experiments/inception_e0), a(x) is the concatenation of the four branch
convolutions' pre-ReLU outputs in Lucid's channel order (1x1, 3x3, 5x5, pool_reduce), shape (C, H, W), the response the
essay's microscope shows. Channels of the output, in this order (sheet id = position):
  units:                       a_n for the channels n given by --units (the audit's earlier samples)
  random_direction (pre-ReLU):  v_k = sum_n d_kn a_n, d_k seeded Gaussian over the C channels, unit norm, k < D
  random_direction_post:        w_k = sum_n d_kn relu(a_n), the same d_k, read in the space the next layer reads
Per image: the largest and smallest response over the H x W positions and their row-major positions. Buckets and the
sheet layout are those of convnext_microscope (top16, p9999, p999, p99, bottom4; 16 x 3 cells of 96 px). Crop: a
square window of side --focus (px of the 224-px input) centred on the position's receptive-field centre
((r + 0.5) * 224 / H, (c + 0.5) * 224 / W) and moved inside the image; 224 shows the whole image (mixed5b's
microscope), 144 the window used for stage 2 of the ConvNeXt microscopes. Images: all 50,000 validation images,
preprocessed as in inception_e0 (shorter side 256, centre crop 224); image index = position in the parquet order.

  python3 -m experiments.inception_microscope --val /marimo/work/data/imagenet/data --imagenet /marimo/work/data/imagenet \
      --weights inception5h.pth --layer mixed5b --units 48,59,... --dirs 40 --focus 224 --out results/iv1_micro
Output in --out/<layer>/: examples/<layer>/<id>.jpg, data_<layer>.json (neurons[id].kind and .source: the unit's channel or
the direction's k), dirs_<layer>.npy, stats_<layer>.npz.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.convnext_microscope import CELL, random_dirs, select, write
from experiments.inception_e0 import BRANCHES

KINDS = ('unit', 'random_direction', 'random_direction_post')


def hook_layer(model, layer):
  """Forward hooks on the layer's four branch convolutions; returns (dict that receives them, hook handles)."""
  cur, hs = {}, []
  for b in BRANCHES:
    hs.append(getattr(model, f'{layer}_{b}_pre_relu_conv').register_forward_hook(
      lambda m, i, o, b=b: cur.__setitem__(b, o)))
  return cur, hs


@torch.no_grad()
def collect(model, layer, batches, units, dirs):
  """batches yields (x, labels), x already preprocessed. Returns the arrays of convnext_microscope.collect for the
  channels [units, pre-ReLU directions, post-ReLU directions]: maxv, minv (N, K) float32, pmax, pmin (N, K) int32,
  labels (N,), grid (H, W)."""
  cur, hs = hook_layer(model, layer)
  out = {k: [] for k in ('maxv', 'minv', 'pmax', 'pmin', 'labels')}
  try:
    for x, y in batches:
      model(x)
      a = torch.cat([cur[b] for b in BRANCHES], 1).float()                    # (B, C, H, W)
      B, C, H, W = a.shape
      flat = a.flatten(2)                                                      # (B, C, P)
      d = torch.as_tensor(dirs, dtype=torch.float32, device=a.device)          # (D, C)
      ch = torch.cat([flat[:, list(units)], torch.einsum('dc,bcp->bdp', d, flat),
                      torch.einsum('dc,bcp->bdp', d, flat.clamp(min=0))], 1)   # (B, K, P)
      mx, pm = ch.max(-1)
      mn, pn = ch.min(-1)
      for k, v in (('maxv', mx), ('minv', mn), ('pmax', pm), ('pmin', pn), ('labels', torch.as_tensor(y))):
        out[k].append(v.cpu())
      grid = (H, W)
  finally:
    for h in hs:
      h.remove()
  res = {k: torch.cat(v).numpy() for k, v in out.items()}
  for k in ('pmax', 'pmin'):
    res[k] = res[k].astype(np.int32)
  res['grid'] = grid
  return res


def crop_box(position, grid, focus, size=224):
  """(top, left, side) of the window of side `focus` centred on a grid position's receptive-field centre, inside the
  image."""
  H, W = grid
  r, c = divmod(position, W)
  side = min(focus, size)
  cy, cx = (r + 0.5) * size / H, (c + 0.5) * size / W
  top = int(round(min(max(cy - side / 2, 0), size - side)))
  left = int(round(min(max(cx - side / 2, 0), size - side)))
  return top, left, side


@torch.no_grad()
def gather_crops(batches, sel, grid, focus, offset=117.0):
  """{(id, bucket, k): (CELL, CELL, 3) uint8} from a second pass over the same images (x + offset = pixel values)."""
  want = {}
  for c, rec in sel.items():
    for b, cells in rec.items():
      for k, (i, _, _, pos) in enumerate(cells):
        want.setdefault(i, []).append((c, b, k, pos))
  crops, i0 = {}, 0
  for x, _ in batches:
    for j in range(x.shape[0]):
      i = i0 + j
      if i not in want:
        continue
      img = (x[j].float().cpu() + offset).clamp(0, 255)
      for c, b, k, pos in want[i]:
        top, left, side = crop_box(pos, grid, focus, img.shape[-1])
        cr = F.interpolate(img[None, :, top:top + side, left:left + side], size=(CELL, CELL), mode='bilinear',
                           antialias=True, align_corners=False)[0]
        crops[c, b, k] = cr.round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
    i0 += x.shape[0]
  return crops


def label_kinds(path, units, D):
  """Rewrite data_<layer>.json's kind of every record (write() knows only units and directions) and add its source."""
  data = json.load(open(path))
  for i, n in enumerate(data['neurons']):
    if i < len(units):
      n['kind'], n['source'] = 'unit', int(units[i])
    else:
      k = i - len(units)
      n['kind'], n['source'] = KINDS[1 + k // D], k % D
  data['units'] = len(units)
  data['note'] = 'pre-ReLU concatenation of the branch convolutions; random_direction_post reads relu(a)'
  json.dump(data, open(path, 'w'))


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--val', required=True, help='directory with validation-*.parquet')
  p.add_argument('--imagenet', required=True, help='directory with classes.py (label names)')
  p.add_argument('--weights', required=True, help='inception5h.pth')
  p.add_argument('--layer', required=True)
  p.add_argument('--units', required=True, help='comma-separated channels (Lucid order)')
  p.add_argument('--dirs', type=int, default=40)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--focus', type=int, required=True, help='crop side in px of the 224-px input')
  p.add_argument('--batch', type=int, default=250)
  p.add_argument('--out', required=True)
  a = p.parse_args()
  from experiments.inception_e0 import class_names, iter_val, load_model, preprocess
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_model(a.weights, dev)
  units = [int(c) for c in a.units.split(',')]
  C = sum(getattr(model, f'{a.layer}_{b}_pre_relu_conv').out_channels for b in BRANCHES)
  dirs = random_dirs(C, a.dirs, a.seed)
  batches = lambda: ((preprocess(raw, dev), lab) for raw, lab in iter_val(a.val, a.batch))
  S = collect(model, a.layer, batches(), units, dirs)
  print('collected', S['maxv'].shape, 'grid', S['grid'], flush=True)
  sel = select(S)
  crops = gather_crops(batches(), sel, S['grid'], a.focus)
  out = os.path.join(a.out, a.layer)
  os.makedirs(out, exist_ok=True)
  np.save(os.path.join(out, f'dirs_{a.layer}.npy'), dirs)
  np.savez_compressed(os.path.join(out, f'stats_{a.layer}.npz'), **{k: v for k, v in S.items() if k != 'grid'})
  write(out, a.layer, sel, crops, S['labels'], class_names(a.imagenet), len(units), S)
  label_kinds(os.path.join(out, f'data_{a.layer}.json'), units, a.dirs)
  print('MICROSCOPE_DONE', len(sel), 'channels', flush=True)


if __name__ == '__main__':
  main()
