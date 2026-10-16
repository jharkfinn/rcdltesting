"""Car selectivity of every SwiGLU unit through training, computed as in the essay's training map, for comparison.

The essay's training map (already-clean.goodfire.com/training-map.html; assets/training-map/<run>/timeline.bin, public)
gives, for its fresh tiny runs (seed 0 and 1, 1,152 units, the same 16 checkpoint steps as experiments/convnext_train),
an AUROC per checkpoint, pooling mode and unit for four targets (float32 [checkpoint, mode, unit, target]):
  car         the ten car classes (ambulance, beach wagon, cab, convertible, jeep, limousine, minivan, Model T, racer,
              sports car) against all other sampled classes
  ambulance   ambulance against the other nine car classes;  cab: the same for cab
  road        car family against eleven other road-vehicle classes (not recomputed here: the eleven are not listed)
with modes mean+ (spatial mean of the SwiGLU product), mean- (its negative), max+ (spatial maximum), min- (minus the
spatial minimum). Its probe: 25 images of each car class and 4 of every other class (4,210). The essay's image indices
match neither the Hugging Face parquet order nor the file-name order of the validation set, so the same images cannot
be used; here the probe is all 50 validation images of each car class and the first 4 (parquet order) of every other
class. Population statistics (how many units pass an AUROC level, the best unit per block) are compared, not units.

AUROC = (sum of the ranks of the positives - n_pos (n_pos + 1) / 2) / (n_pos n_neg), ties ranked by order.

Usage: python3 -m experiments.convnext_carmap --run <E1 run dir> --val <validation parquet dir>
       --essay <dir with fresh-tiny-seed0/timeline.bin ...> [--probe_cache file]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.convnext import build_convnext

CAR = (407, 436, 468, 511, 609, 627, 656, 661, 751, 817)
AMBULANCE, CAB = 407, 468
MODES = ('mean+', 'mean-', 'max+', 'min-')
TARGETS = ('car', 'ambulance', 'cab')
MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255
STD = torch.tensor([0.229, 0.224, 0.225]) * 255


def car_probe(val_dir, cache=None, per_other=4):
  if cache and os.path.exists(cache):
    d = torch.load(cache)
    return d['x'], d['y']
  import pyarrow.parquet as pq
  from torchvision.io import ImageReadMode, decode_image
  want = {c: (50 if c in CAR else per_other) for c in range(1000)}
  xs, ys = [], []
  for f in sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True)):
    t = pq.read_table(f)
    for d, lab in zip(t.column('image').to_pylist(), t.column('label').to_pylist()):
      if want[lab] == 0:
        continue
      want[lab] -= 1
      im = decode_image(torch.frombuffer(bytearray(d['bytes']), dtype=torch.uint8), mode=ImageReadMode.RGB)
      h, w = im.shape[-2:]
      s = 256 / min(h, w)
      im = F.interpolate(im[None].float(), size=(max(256, round(h * s)), max(256, round(w * s))), mode='bilinear',
                         antialias=True, align_corners=False)[0]
      H, W = im.shape[-2:]
      top, left = (H - 224) // 2, (W - 224) // 2
      xs.append(im[:, top:top + 224, left:left + 224].round().clamp(0, 255).to(torch.uint8))
      ys.append(lab)
  x, y = torch.stack(xs), torch.tensor(ys)
  if cache:
    torch.save({'x': x, 'y': y}, cache)
  return x, y


@torch.no_grad()
def pooled_units(model, x_u8, batch=100):
  """(N, units) spatial mean, max and min of every block's SwiGLU product, blocks in model order (on the model's
  device; returned on the CPU)."""
  dev = next(model.parameters()).device
  mean, std = MEAN.to(dev)[:, None, None], STD.to(dev)[:, None, None]
  outs = {'mean': [], 'max': [], 'min': []}
  feats = []
  hooks = [b.norm.register_forward_hook(lambda m, i, o, b=b: feats.append(F.silu(b.pw_gate(o)) * b.pw_value(o)))
           for _, b in model.blocks()]
  for i in range(0, len(x_u8), batch):
    feats.clear()
    model((x_u8[i:i + batch].to(dev).float() - mean) / std)
    u = [f.flatten(1, 2) for f in feats]                      # (B, positions, units) per block
    outs['mean'].append(torch.cat([t.mean(1) for t in u], 1).cpu())
    outs['max'].append(torch.cat([t.amax(1) for t in u], 1).cpu())
    outs['min'].append(torch.cat([t.amin(1) for t in u], 1).cpu())
  for h in hooks:
    h.remove()
  return {k: torch.cat(v) for k, v in outs.items()}


def auroc(score, pos):
  """score (N, U), pos bool (N,) -> (U,) AUROC of each column for the positives."""
  ranks = torch.empty_like(score)
  order = torch.argsort(score, 0)
  ranks.scatter_(0, order, torch.arange(1, len(score) + 1, dtype=score.dtype)[:, None].expand_as(score))
  n_pos, n_neg = int(pos.sum()), int((~pos).sum())
  return (ranks[pos].sum(0) - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def timeline(run, x, y, size='tiny'):
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = build_convnext(size).to(dev).eval()
  steps = sorted(int(f[-8:-3]) for f in glob.glob(os.path.join(run, 'model_[0-9]*.pt')))
  is_car = torch.isin(y, torch.tensor(CAR))
  out = np.zeros((len(steps), len(MODES), sum(64 for _ in model.blocks()), len(TARGETS)), np.float32)
  for si, s in enumerate(steps):
    model.load_state_dict(torch.load(os.path.join(run, f'model_{s:05d}.pt'), map_location=dev))
    pu = pooled_units(model, x)
    for mi, mode in enumerate(MODES):
      sc = {'mean+': pu['mean'], 'mean-': -pu['mean'], 'max+': pu['max'], 'min-': -pu['min']}[mode].double()
      out[si, mi, :, 0] = auroc(sc, is_car).numpy()
      out[si, mi, :, 1] = auroc(sc[is_car], (y[is_car] == AMBULANCE)).numpy()
      out[si, mi, :, 2] = auroc(sc[is_car], (y[is_car] == CAB)).numpy()
    print(json.dumps({'step': s, 'car_auroc_max_by_mode': out[si, :, :, 0].max(1).round(3).tolist()}), flush=True)
  return steps, out


def summarize(T, steps, levels=(0.9, 0.95)):
  """Per checkpoint and mode: share of units at or above each car AUROC level, best car / ambulance / cab AUROC."""
  rows = []
  for si, s in enumerate(steps):
    for mi, mode in enumerate(MODES):
      r = {'step': s, 'mode': mode}
      for lv in levels:
        r[f'car>={lv}'] = float((T[si, mi, :, 0] >= lv).mean())
      for ti, t in enumerate(TARGETS):
        r[f'best_{t}'] = float(T[si, mi, :, ti].max())
      rows.append(r)
  return rows


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--val', required=True)
  p.add_argument('--essay', required=True)
  p.add_argument('--probe_cache', default=None)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  x, y = car_probe(a.val, a.probe_cache)
  print('probe', tuple(x.shape), 'car images', int(torch.isin(y, torch.tensor(CAR)).sum()), flush=True)
  steps, T = timeline(a.run, x, y)
  res = {'ours': summarize(T, steps)}
  for run in ('fresh-tiny-seed0', 'fresh-tiny-seed1'):
    f = os.path.join(a.essay, run, 'timeline.bin')
    if os.path.exists(f):
      E = np.fromfile(f, dtype='<f4').reshape(16, 4, 1152, 4)[..., :3]
      res[run] = summarize(E, [0, 50, 100, 200, 350, 500, 1000, 2000, 2500, 5000, 7500, 10000, 12500, 17500, 20000, 25020])
  out = a.out or os.path.join(a.run, 'carmap.json')
  json.dump(res, open(out, 'w'), indent=1)
  np.save(out.replace('.json', '_timeline.npy'), T)
  print('step  mode  | car>=0.9: ours / essay seed0 / seed1 | best car: ours / seed0 / seed1 | best cab: ours / seed0 / seed1')
  ess = {k: {(r['step'], r['mode']): r for r in v} for k, v in res.items() if k != 'ours'}
  for r in res['ours']:
    if r['mode'] not in ('mean+', 'max+'):
      continue
    e = [ess[k].get((r['step'], r['mode'])) for k in ('fresh-tiny-seed0', 'fresh-tiny-seed1') if k in ess]
    f = lambda key: ' / '.join([f"{r[key]:.3f}"] + [f"{x[key]:.3f}" if x else '-' for x in e])
    print(f"{r['step']:5d} {r['mode']:5s} | {f('car>=0.9')} | {f('best_car')} | {f('best_cab')}")


if __name__ == '__main__':
  main()
