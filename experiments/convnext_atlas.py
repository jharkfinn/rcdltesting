"""E1 writer atlas: for every tracked SwiGLU unit and checkpoint, which classes wrote its response to which classes.

Input: a run directory of experiments/convnext_train.py (model_{step}.pt, attr_{step}.pt with the per-class parts C, Cb
of pw_gate and pw_value in fp16, attr_init.pt with W_0, b_0) and a probe set of validation images (the first
`per_class` images of every class in the validation parquet order, center crops 224 of the 256 resize, as in
rcdl.imagenet.val_batches).

For block b, unit n and probe image i, the unit's response is the SwiGLU product at its strongest position
p* = argmax_p u[i, p, n] (the essay's microscopes use the spatial maximum): u* = SiLU(g) * v with the gate and value
pre-activations g = W_g[n] h + b_g[n], v = W_v[n] h + b_v[n] at h = h[i, p*] (the block's LayerNorm output). Each splits
exactly over the training classes k that wrote the map:

  v = a W_0[n] h + ab b_0[n] + sum_k (C^(k)[n] h + Cb^(k)[n]),

where a and ab are the decay factors of the initial weight and bias, and C^(k), Cb^(k) the weight and bias parts written
by the examples of class k (mixup splits an example's gradient between its two classes with lam, 1 - lam). Averaged over
the probe images of class c this gives V[n, c, k], the part of unit n's value on class c written by class k (and G for
the gate).

Saved per checkpoint (atlas_{step}.pt), per block: R (units, classes) mean u*; Vm, Gm (units, classes) mean v and g at
p*; and, for each unit's `poles` strongest and `poles` weakest classes at the final checkpoint (the pole classes, fixed
across checkpoints so that growth can be followed), the full writer vectors V, G (units, 2 * poles, K) and the initial
parts (units, 2 * poles). `resid` is max |v - (init + sum_k parts)| over the probe set; the fp16 storage remainder R of
the trainer's checkpoints is counted in the initial part, so `resid` is at fp32 rounding. While the run is incomplete, each checkpoint picks its own pole classes and is saved as
atlas_prelim_{step}.pt.

Usage: python3 -m experiments.convnext_atlas --run <run dir> --val <validation parquet dir> [--per_class 8]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.convnext import build_convnext


def probe_set(val_dir: str, per_class: int, cache: str, n_classes: int = 1000):
  """uint8 center crops (N, 3, 224, 224) and labels of the first `per_class` validation images of every class."""
  if cache and os.path.exists(cache):
    d = torch.load(cache)
    return d['x'], d['y']
  import pyarrow.parquet as pq
  from torchvision.io import ImageReadMode, decode_image
  counts, xs, ys = np.zeros(n_classes, int), [], []
  for f in sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True)):
    t = pq.read_table(f)
    for d, lab in zip(t.column('image').to_pylist(), t.column('label').to_pylist()):
      if counts[lab] >= per_class:
        continue
      counts[lab] += 1
      im = decode_image(torch.frombuffer(bytearray(d['bytes']), dtype=torch.uint8), mode=ImageReadMode.RGB)
      h, w = im.shape[-2:]
      s = 256 / min(h, w)
      im = F.interpolate(im[None].float(), size=(max(256, round(h * s)), max(256, round(w * s))), mode='bilinear',
                         antialias=True, align_corners=False)[0]
      H, W = im.shape[-2:]
      top, left = (H - 224) // 2, (W - 224) // 2
      xs.append(im[:, top:top + 224, left:left + 224].round().clamp(0, 255).to(torch.uint8))
      ys.append(lab)
    if counts.min() >= per_class:
      break
  x, y = torch.stack(xs), torch.tensor(ys)
  order = torch.argsort(y, stable=True)
  x, y = x[order], y[order]
  if cache:
    torch.save({'x': x, 'y': y}, cache)
  return x, y


MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255
STD = torch.tensor([0.229, 0.224, 0.225]) * 255


def block_inputs(model, x, names):
  """{block name: (B, H, W, d) LayerNorm output h} for one batch."""
  out, hooks = {}, []
  for n, b in model.blocks():
    if n in names:
      hooks.append(b.norm.register_forward_hook(lambda m, i, o, n=n: out.__setitem__(n, o)))
  with torch.no_grad():
    model(x)
  for h in hooks:
    h.remove()
  return out


def atlas_step(model, attr, init, x_u8, y, n_classes, poles_idx=None, poles=20, batch=100, device='cuda',
               chunk=None):
  """One checkpoint. poles_idx: {block: (units, 2 * poles) class indices} (None: choose from this checkpoint's R).
  The probe set must be sorted by class with the same number of images per class. Blocks are processed `chunk` at a
  time (None: all at once), re-running the probe forward for each chunk, to bound the memory of h* for wide models."""
  all_blocks = sorted({k.rsplit('.', 1)[0] for k in attr}, key=lambda n: [int(t) for t in n.replace('stage', '').split('.block')])
  per = int((y == y[0]).sum())
  assert torch.equal(y, torch.arange(n_classes).repeat_interleave(per)), 'probe set: sorted, equal count per class'
  mean, std = MEAN.to(device)[:, None, None], STD.to(device)[:, None, None]
  named = dict(model.blocks())
  res = {}
  chunk = chunk or len(all_blocks)
  for c0 in range(0, len(all_blocks), chunk):
    blocks = all_blocks[c0:c0 + chunk]
    star = {b: {'u': [], 'v': [], 'g': [], 'h': []} for b in blocks}
    for i in range(0, len(y), batch):
      xb = (x_u8[i:i + batch].to(device).float() - mean) / std
      hs = block_inputs(model, xb, set(blocks))
      for b in blocks:
        blk = named[b]
        h = hs[b].flatten(1, 2).float()                         # (B, P, d)
        g = h @ blk.pw_gate.weight.float().T + blk.pw_gate.bias.float()
        v = h @ blk.pw_value.weight.float().T + blk.pw_value.bias.float()
        u = F.silu(g) * v
        p = u.argmax(1)                                         # (B, units): the strongest position per unit
        take = lambda t: torch.gather(t, 1, p[:, None, :])[:, 0]
        star[b]['h'].append(torch.gather(h, 1, p[:, :, None].expand(-1, -1, h.shape[-1])))   # (B, units, d)
        for k, t in (('u', u), ('v', v), ('g', g)):
          star[b][k].append(take(t))
    for b in blocks:
      st = {k: torch.cat(v) for k, v in star[b].items()}
      cls = lambda t: t.view(n_classes, per, *t.shape[1:]).mean(1)                   # per-class means
      R, Vm, Gm = cls(st['u']).T, cls(st['v']).T, cls(st['g']).T                     # (units, classes)
      if poles_idx is None:
        pidx = torch.cat([torch.argsort(R, 1, descending=True)[:, :poles], torch.argsort(R, 1)[:, :poles]], 1).cpu()
      else:
        pidx = poles_idx[b]
      rec = {'R': R.cpu(), 'Vm': Vm.cpu(), 'Gm': Gm.cpu(), 'poles': pidx, 'resid': 0.0}
      for mname, key_star in (('value', 'v'), ('gate', 'g')):
        key = f'{b}.pw_{mname}'
        ak = attr[key]
        C = ak['C'].to(device).float() / ak.get('C_scale', 1.0)                       # (K, units, d)
        Cb = ak['Cb'].to(device).float() / ak.get('Cb_scale', 1.0)                    # (K, units)
        a, ab = float(ak['a']), float(ak['ab'])
        W0, b0 = init[key]['W0'].to(device).float(), init[key]['b0'].to(device).float()
        if 'R' in ak:                       # remainder of the fp16 storage: counted with the initial part, so exact
          W0 = W0 + ak['R'].to(device).float() / a
          b0 = b0 + ak['Rb'].to(device).float() / ab
        units, K = C.shape[1], C.shape[0]
        out = torch.zeros(units, pidx.shape[1], K, device=device)
        ini = torch.zeros(units, pidx.shape[1], device=device)
        for n in range(units):
          hn = st['h'][:, n]                                    # (N, d)
          parts = hn @ C[:, n].T + Cb[:, n]                     # (N, K)
          i0 = a * (hn @ W0[n]) + ab * b0[n]                    # (N,)
          rec['resid'] = max(rec['resid'], float((parts.sum(1) + i0 - st[key_star][:, n]).abs().max()))
          sel = pidx[n].to(device)
          out[n] = cls(parts)[sel]
          ini[n] = cls(i0)[sel]
        rec[f'{mname}_writers'] = out.half().cpu()
        rec[f'{mname}_init'] = ini.cpu()
      res[b] = rec
  return res


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--val', required=True)
  p.add_argument('--per_class', type=int, default=8)
  p.add_argument('--poles', type=int, default=20)
  p.add_argument('--probe_cache', default='/dev/shm/probe_{per_class}.pt')
  p.add_argument('--steps', default='', help='comma-separated checkpoint steps (default: all, final first)')
  p.add_argument('--chunk', type=int, default=None, help='blocks per probe pass (wide models)')
  a = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  model = build_convnext(cfg['size']).to(dev).eval()
  init = torch.load(os.path.join(a.run, 'attr_init.pt'), map_location='cpu')
  x, y = probe_set(a.val, a.per_class, a.probe_cache.format(per_class=a.per_class))
  done = sorted(int(f[-8:-3]) for f in glob.glob(os.path.join(a.run, 'attr_[0-9]*.pt')))
  steps = [int(t) for t in a.steps.split(',')] if a.steps else done
  final = max(done)
  complete = final == cfg['steps']
  steps = [final] + [t for t in steps if t != final]
  pole_file = os.path.join(a.run, 'atlas_poles.pt')
  poles = torch.load(pole_file) if os.path.exists(pole_file) else None
  for s in steps:
    out = os.path.join(a.run, f'atlas_{s:05d}.pt' if complete else f'atlas_prelim_{s:05d}.pt')
    if os.path.exists(out):
      continue
    model.load_state_dict(torch.load(os.path.join(a.run, f'model_{s:05d}.pt'), map_location=dev))
    attr = torch.load(os.path.join(a.run, f'attr_{s:05d}.pt'), map_location='cpu')
    res = atlas_step(model, attr, init, x, y, 1000, poles if complete else None, a.poles, device=dev, chunk=a.chunk)
    if complete and poles is None:                             # pole classes of the final checkpoint, kept for all
      poles = {b: r['poles'] for b, r in res.items()}
      torch.save(poles, pole_file)
    torch.save(res, out)
    print(json.dumps({'step': s, 'resid': {b: r['resid'] for b, r in res.items()}}), flush=True)


if __name__ == '__main__':
  main()
