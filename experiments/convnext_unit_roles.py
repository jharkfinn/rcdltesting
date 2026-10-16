"""Whom a unit helps: per-class ablation profiles, and the unit against its write direction in the residual stream.

convnext_bias_paths.py --ablate finds that removing a stage-1 unit lowers the cross-entropy of its own 20 strongest
classes and raises that of other classes (late blocks: the reverse). This script asks two follow-up questions.

1. Whom does each unit help? For block b and unit n, the unit's output is set to zero at every position (u_n = 0) and the
   cross-entropy change dCE[n, c] (class mean over the probe images of class c) is measured for all 1,000 classes.
   Per unit: the Spearman correlation over classes of dCE[n, :] with the unit's response R[n, :] (class mean of
   u* = max_p u, as in convnext_atlas.py; positive: the unit helps the classes that drive it), the mean dCE over its
   top-20 classes and over the rest, and the WordNet / visual relatedness of the 20 classes it helps most
   (largest dCE) to its top-20 classes, against random classes.

2. Is it the unit or its direction? The block writes into the residual stream x (width d: 8 in stages 0 and 1)
   through w_n = gamma * W_down[:, n], so u_n adds u_n(p) w_n at position p. With e = w_n / |w_n| and out the block's
   output (residual input plus all units' writes), on the images of the unit's top-20 classes and of 20 other classes:
     unit:      u_n -> 0                                   (as in 1)
     direction: out -> out - (out . e) e                   (the whole stream component along e removed)
     others:    out -> out - ((out - u_n w_n) . e) e        (every other contribution along e removed, u_n kept)
     scaled:    out -> out + (beta - 1)(out . e) e, beta in {0.5, 1.5}
   If removing the unit helps its top classes but removing the direction hurts them, the direction is useful and the
   unit adds too much of it (overshoot of the direction); if removing the direction helps too, the direction itself is
   read against those classes.
   Also per block: |cos| between write directions (the share of units with another unit at |cos| > 0.9).

The forward runs from the block's input (cached once per block), so stage 0 is not recomputed for each ablation.

  python3 -m experiments.convnext_unit_roles --run results/e1_tiny_adamw --steps 1000,20000
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.convnext_atlas import MEAN, STD, probe_set
from rcdl.convnext import build_convnext

DEFAULT_BLOCKS = 'stage1.block0,stage1.block1,stage1.block2,stage2.block5,stage3.block1'


def spearman(a, b):
  ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
  return float(np.corrcoef(ra, rb)[0, 1])


def block_units(blk, x):
  """u (B, H, W, U) of a block for its input x (B, C, H, W)."""
  h = blk.norm(blk.dwconv(x).permute(0, 2, 3, 1))
  return F.silu(blk.pw_gate(h)) * blk.pw_value(h)


def block_out(blk, x, u):
  """The block's output for input x and units u (eval mode: no drop path)."""
  h = blk.pw_down(u)
  if blk.gamma is not None:
    h = blk.gamma * h
  h = h.permute(0, 3, 1, 2)
  return x + h if blk.skip else h


def write_dir(blk, n):
  w = blk.pw_down.weight[:, n]
  return w * blk.gamma if blk.gamma is not None else w


def forward_from(model, i, j, x, unit_scale=None, out_fn=None):
  """Logits from the input x of block j of stage i. unit_scale: (U,) factor on that block's units; out_fn(out, u):
  transform of that block's output."""
  blk = model.stages[i][j]
  u = block_units(blk, x)
  if unit_scale is not None:
    u = u * unit_scale
  out = block_out(blk, x, u)
  if out_fn is not None:
    out = out_fn(out, u)
  x = out
  for blk2 in model.stages[i][j + 1:]:
    x = blk2(x)
  for down, stage in zip(model.downsample[i + 1:], model.stages[i + 1:]):
    x = stage(down(x))
  return model.head(model.norm(x.mean((2, 3))))


@torch.no_grad()
def block_inputs(model, xs, i, j, batch, device):
  """Inputs of block (i, j) for all probe images, kept on the device (fp32)."""
  mean, std = MEAN.to(device)[:, None, None], STD.to(device)[:, None, None]
  out = []
  for s in range(0, len(xs), batch):
    x = (xs[s:s + batch].to(device).float() - mean) / std
    for k in range(i + 1):
      x = model.downsample[k](x)
      blocks = model.stages[k] if k < i else model.stages[k][:j]
      for b in blocks:
        x = b(x)
    out.append(x)
  return torch.cat(out)


@torch.no_grad()
def ce_from(model, i, j, X, y, batch, **kw):
  out = []
  for s in range(0, len(X), batch):
    out.append(F.cross_entropy(forward_from(model, i, j, X[s:s + batch], **kw).float(), y[s:s + batch], reduction='none'))
  return torch.cat(out)


@torch.no_grad()
def unit_roles(model, xs, y, name, rel=None, vis=None, batch=500, device='cuda', seed=0, n_top=20):
  i, j = (int(t) for t in name.replace('stage', '').split('.block'))
  blk = model.stages[i][j]
  y = y.to(device)
  C = int(y.max()) + 1
  per = int((y == 0).sum())
  X = block_inputs(model, xs, i, j, batch, device)
  U = blk.pw_gate.out_features
  ustar = torch.cat([block_units(blk, X[s:s + batch]).flatten(1, 2).max(1).values for s in range(0, len(X), batch)])
  R = ustar.view(C, per, U).mean(1).T.cpu().numpy()                                    # (U, C)
  base = ce_from(model, i, j, X, y, batch)
  cls = lambda t: t.view(C, per).mean(1)
  dce = np.zeros((U, C), np.float32)
  for n in range(U):
    sc = torch.ones(U, device=device)
    sc[n] = 0.0
    dce[n] = cls(ce_from(model, i, j, X, y, batch, unit_scale=sc) - base).cpu().numpy()
  rng = np.random.default_rng(seed)
  W = torch.stack([write_dir(blk, n) for n in range(U)])                               # (U, d)
  Wn = W / (W.norm(dim=1, keepdim=True) + 1e-30)
  cos = (Wn @ Wn.T).abs().fill_diagonal_(0).cpu().numpy()
  units = []
  for n in range(U):
    if np.ptp(R[n]) <= 1e-3 * np.abs(R).max():
      continue
    top = np.argsort(-R[n])[:n_top]
    rest = rng.choice(np.setdiff1d(np.arange(C), top), n_top, replace=False)
    rec = {'unit': n, 'rho_R_dce': spearman(R[n], dce[n]), 'dce_top20': float(dce[n, top].mean()),
           'dce_rest': float(np.delete(dce[n], top).mean()), 'max_cos': float(cos[n].max())}
    helped = np.argsort(-dce[n])[:n_top]
    for tag, D in (('wordnet', rel), ('visual', vis)):
      if D is not None:
        rec[f'helped_rel_{tag}'] = float(D[np.ix_(helped, top)].any(1).mean())
        rand = rng.choice(np.setdiff1d(np.arange(C), top), n_top, replace=False)
        rec[f'random_rel_{tag}'] = float(D[np.ix_(rand, top)].any(1).mean())
        rec[f'helped_in_top20'] = float(np.isin(helped, top).mean())
    idx = torch.from_numpy((np.concatenate([top, rest])[:, None] * per + np.arange(per)[None]).ravel()).to(device)
    Xs, ys, b0 = X[idx], y[idx], base[idx]
    e = Wn[n].view(1, -1, 1, 1)
    w = W[n].view(1, -1, 1, 1)
    fns = {
      'direction': lambda out, u: out - (out * e).sum(1, keepdim=True) * e,
      'others': lambda out, u: out - ((out - u[..., n][:, None] * w) * e).sum(1, keepdim=True) * e,
      'beta0.5': lambda out, u: out - 0.5 * (out * e).sum(1, keepdim=True) * e,
      'beta1.5': lambda out, u: out + 0.5 * (out * e).sum(1, keepdim=True) * e,
    }
    half = n_top * per
    for k, fn in fns.items():
      d = ce_from(model, i, j, Xs, ys, batch, out_fn=fn) - b0
      rec[f'{k}_top20'] = float(d[:half].mean())
      rec[f'{k}_rest20'] = float(d[half:].mean())
    units.append(rec)
  keys = [k for k in units[0] if k != 'unit']
  summary = {k: float(np.nanmean([r[k] for r in units])) for k in keys}
  summary['share_helps_top_more'] = float(np.mean([r['dce_top20'] > r['dce_rest'] for r in units]))
  summary['share_rho_pos'] = float(np.mean([r['rho_R_dce'] > 0 for r in units]))
  summary['share_cos_0.9'] = float(np.mean(cos.max(1) > 0.9))
  summary['units'] = len(units)
  return {'summary': summary, 'units': units}, {'dce': dce, 'R': R, 'cos': cos}


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', required=True)
  p.add_argument('--steps', required=True)
  p.add_argument('--blocks', default=DEFAULT_BLOCKS)
  p.add_argument('--val', default='/marimo/work/data/imagenet/data')
  p.add_argument('--probe_cache', default='/dev/shm/probe_8.pt')
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--visual', default='docs/data/imagenet_dinov2_classdist.npz')
  p.add_argument('--batch', type=int, default=500)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  model = build_convnext(cfg['size']).to(dev).eval()
  xs, y = probe_set(a.val, 8, a.probe_cache)
  rel = np.load(a.wordnet)['dist'] <= 4
  np.fill_diagonal(rel, False)
  vis = None
  if os.path.exists(a.visual):
    vz = np.load(a.visual)
    vis = vz['rank_dist'] <= 10                                       # visual relatives, as in the growth tests
    np.fill_diagonal(vis, False)
  res = {}
  for s in [int(t) for t in a.steps.split(',')]:
    model.load_state_dict(torch.load(os.path.join(a.run, f'model_{s:05d}.pt'), map_location=dev))
    res[s], raw = {}, {}
    for b in a.blocks.split(','):
      res[s][b], raw[b] = unit_roles(model, xs, y, b, rel, vis, a.batch, dev)
      print(s, b, {k: round(v, 4) for k, v in res[s][b]['summary'].items()}, flush=True)
    torch.save(raw, os.path.join(a.run, f'unit_roles_raw_{s:05d}.pt'))
  json.dump(res, open(a.out or os.path.join(a.run, 'unit_roles.json'), 'w'), indent=1)
  print('UNIT_ROLES_DONE', flush=True)


if __name__ == '__main__':
  main()
