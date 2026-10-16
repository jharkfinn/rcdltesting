"""Where a class's bias teaching signal on a unit comes from: positions, logit terms and the LayerNorm scale path.

convnext_bias_self.py finds that in stage 1 (from about step 1,000 on) a unit's 20 strongest classes write its bias so
that its response on them falls, while the other classes raise it. The bias writes are exact but summed over every
position of every training image. This script recomputes the same signal on the probe set (clean validation images,
8 per class, hard labels) at a checkpoint and splits it.

For image i of class k, block b, unit n and position p, with u = SiLU(g) v the SwiGLU product:
  delta[i, p, n] = dL_i / du[i, p, n],   L_i = logsumexp(z_i) - z_i[k]   (cross-entropy)
  B_v[k, n] = mean_i sum_p delta SiLU(g),   B_g[k, n] = mean_i sum_p delta SiLU'(g) v      (bias gradients)
The bias write of class k is minus a positive multiple of its gradient, so the first-order change of the mean response
of the unit's top-20 classes P_n (class-mean u, g, v at the strongest position p*, as in convnext_atlas.py) is
  pole[k, n] = -(B_v[k, n] mean_{c in P_n} SiLU(Gm[n, c]) + B_g[k, n] mean_{c in P_n} SiLU'(Gm[n, c]) Vm[n, c])
(convnext_bias_self.py computes the same quantity from the exact training writes). Splits of delta, each summing to the
whole:
  positions: p* alone; 'on' (u >= 0.5 u* other than p*, u* = max_p u > 0); 'off' (the rest)
  logit terms: delta = delta_lse - delta_own, from logsumexp(z) and from the true class's logit z[k]
  LayerNorm scale: delta_fixed with every LayerNorm's 1 / sigma held fixed (forward values unchanged), and the rest,
    delta - delta_fixed, the path through the normalisers' scale
Per block and split: the mean of pole over the top-20 classes and over the rest, in units of the mean |pole| of the whole
signal (so the parts of a split add up to the whole), the share of positive top-20 entries, the active positions
(u > 0.5 u*) per image, and the Spearman correlation over classes, per unit, between the probe signal and the exact
training writes of the next checkpoint interval (if --writes is given).

Assumptions: clean images without mixup or augmentation stand for the training batches; the derivative at class-mean
pre-activations stands for the class mean of the derivative; first order.

  python3 -m experiments.convnext_bias_paths --run results/e1_tiny_adamw --steps 350,1000,2500,12500,25020
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from experiments.convnext_atlas import MEAN, STD, probe_set
from rcdl.convnext import build_convnext

SPLITS = ('ce', 'pstar', 'on', 'off', 'lse', 'own', 'fixed', 'scale')
_FIX_SIGMA = [False]
_layer_norm = F.layer_norm


def _ln(x, shape, weight=None, bias=None, eps=1e-5):
  """F.layer_norm; with _FIX_SIGMA set, 1 / sigma is detached (same forward value, no gradient through the scale)."""
  if not _FIX_SIGMA[0]:
    return _layer_norm(x, shape, weight, bias, eps)
  dims = tuple(range(-len(shape), 0))
  mu = x.mean(dims, keepdim=True)
  var = x.var(dims, unbiased=False, keepdim=True)
  y = (x - mu) * torch.rsqrt(var + eps).detach()
  if weight is not None:
    y = y * weight
  if bias is not None:
    y = y + bias
  return y


F.layer_norm = _ln          # rcdl.convnext calls F.layer_norm (LayerNorm2d) and nn.LayerNorm (which calls F.layer_norm)


def unit_hooks(model, store):
  """Replace each block's SwiGLU so that u keeps its gradient and g, v are stored (forward values unchanged)."""
  for name, blk in model.blocks():
    def units(x, blk=blk, name=name):
      h = blk.norm(blk.dwconv(x).permute(0, 2, 3, 1))
      g, v = blk.pw_gate(h), blk.pw_value(h)
      u = F.silu(g) * v
      u.retain_grad()
      store[name] = (u, g.detach(), v.detach())
      return u
    blk.units = units


def dsilu(g):
  s = torch.sigmoid(g)
  return s * (1 + g * (1 - s))


@torch.no_grad()
def _accumulate(acc, store, deltas, y, n_classes):
  """Add per-class sums of B_v, B_g for every split, and the class sums of u*, g*, v* at p* and active counts."""
  for name, (u, g, v) in store.items():
    B, H, W, U = u.shape
    uf, gf, vf = u.detach().flatten(1, 2), g.flatten(1, 2), v.flatten(1, 2)           # (B, P, U)
    av, ag = F.silu(gf), dsilu(gf) * vf
    ustar, pstar = uf.max(1)                                                          # (B, U)
    onehot = torch.zeros_like(uf).scatter_(1, pstar[:, None], 1.0)
    on = ((uf >= 0.5 * ustar[:, None]) & (ustar[:, None] > 0)).float()
    masks = {'pstar': onehot, 'on': on * (1 - onehot), 'off': 1 - torch.maximum(on, onehot)}
    a = acc.setdefault(name, {})
    def add(key, val):                                                                # val (B, U) -> class sums
      t = a.setdefault(key, torch.zeros(n_classes, U, device=val.device, dtype=torch.float64))
      t.index_add_(0, y, val.double())
    for key, d in deltas.items():
      df = d[name].flatten(1, 2)
      if key == 'ce':
        for mk, m in masks.items():
          add(f'Bv_{mk}', (df * av * m).sum(1))
          add(f'Bg_{mk}', (df * ag * m).sum(1))
      add(f'Bv_{key}', (df * av).sum(1))
      add(f'Bg_{key}', (df * ag).sum(1))
    take = lambda t: torch.gather(t, 1, pstar[:, None])[:, 0]
    add('u', ustar)
    add('g', take(gf))
    add('v', take(vf))
    add('n_on', on.sum(1))


def probe_paths(model, x_u8, y, n_classes=1000, batch=50, device='cuda'):
  """Per block: class means (n_classes, units) of B_v, B_g for every split, u*, g*, v* and active counts."""
  mean, std = MEAN.to(device)[:, None, None], STD.to(device)[:, None, None]
  store, acc = {}, {}
  unit_hooks(model, store)
  for i in range(0, len(y), batch):
    xb = ((x_u8[i:i + batch].to(device).float() - mean) / std)
    yb = y[i:i + batch].to(device)
    deltas = {}
    for key in ('lse', 'own', 'fixed'):
      _FIX_SIGMA[0] = key == 'fixed'
      model.zero_grad(set_to_none=True)
      z = model(xb).float()
      own = z.gather(1, yb[:, None])[:, 0]
      loss = {'lse': torch.logsumexp(z, 1), 'own': own, 'fixed': torch.logsumexp(z, 1) - own}[key].sum()
      loss.backward()
      deltas[key] = {n: s[0].grad.detach().clone() for n, s in store.items()}
    _FIX_SIGMA[0] = False
    deltas['ce'] = {n: deltas['lse'][n] - deltas['own'][n] for n in store}
    deltas['scale'] = {n: deltas['ce'][n] - deltas['fixed'][n] for n in store}
    _accumulate(acc, store, deltas, yb, n_classes)
  cnt = torch.bincount(y.to(device), minlength=n_classes).double()[:, None]
  return {b: {k: (t / cnt).float().cpu().numpy() for k, t in a.items()} for b, a in acc.items()}


def spearman(a, b):
  ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
  return float(np.corrcoef(ra, rb)[0, 1])


def silu_np(g):
  return g / (1 + np.exp(-g))


def dsilu_np(g):
  s = 1 / (1 + np.exp(-g))
  return s * (1 + g * (1 - s))


def summarize(P, writes=None, top_n=20):
  """P: one block's class means from probe_paths. writes: optional {'gate', 'value'}: (K, U) exact bias writes of the
  following interval. Returns {split: {top20, rest, pos_top20}, n_on_top20, n_on_rest, rho_writes}."""
  R, G, V = P['u'].T, P['g'].T, P['v'].T                                            # (U, C)
  z = (R - R.mean(1, keepdims=True)) / (R.std(1, keepdims=True) + 1e-12)
  live = np.ptp(R, 1) > 1e-3 * np.abs(R).max()
  top = np.argsort(-z, 1)[:, :top_n]
  pa_v = np.take_along_axis(silu_np(G), top, 1).mean(1)
  pa_g = np.take_along_axis(dsilu_np(G) * V, top, 1).mean(1)
  pole = lambda s: -(P[f'Bv_{s}'] * pa_v[None] + P[f'Bg_{s}'] * pa_g[None])          # (C, U)
  whole = pole('ce')
  scale = np.abs(whole).mean(0) + 1e-30
  inT = np.zeros_like(whole, bool)
  np.put_along_axis(inT, top.T, True, 0)
  out = {}
  for s in SPLITS:
    ps = pole(s) / scale[None]
    t = np.where(inT, ps, np.nan)
    r = np.where(~inT, ps, np.nan)
    out[s] = {'top20': float(np.nanmean(np.nanmean(t, 0)[live])), 'rest': float(np.nanmean(np.nanmean(r, 0)[live])),
              'pos_top20': float(np.nanmean(np.nanmean(np.where(inT, ps > 0, np.nan), 0)[live]))}
  out['n_on_top20'] = float(np.nanmean(np.where(inT, P['n_on'], np.nan)))
  out['n_on_rest'] = float(np.nanmean(np.where(~inT, P['n_on'], np.nan)))
  out['dn_top20'] = float(np.mean((np.take_along_axis(G, top, 1) < 0) & (np.take_along_axis(V, top, 1) < 0)))
  if writes is not None:
    wpole = writes['value'] * pa_v[None] + writes['gate'] * pa_g[None]                # writes are -lr grad already
    out['rho_writes'] = float(np.nanmedian([spearman(whole[:, n], wpole[:, n]) for n in np.flatnonzero(live)]))
    wt = np.where(inT, wpole / (np.abs(wpole).mean(0) + 1e-30)[None], np.nan)
    out['writes_top20'] = float(np.nanmean(np.nanmean(wt, 0)[live]))
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', required=True)
  p.add_argument('--steps', required=True)
  p.add_argument('--val', default='/marimo/work/data/imagenet/data')
  p.add_argument('--probe_cache', default='/dev/shm/probe_8.pt')
  p.add_argument('--batch', type=int, default=50)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  from experiments.convnext_bias_signal import load_cb
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  model = build_convnext(cfg['size']).to(dev).eval()
  x, y = probe_set(a.val, 8, a.probe_cache)
  ckpts = sorted(int(f[6:11]) for f in os.listdir(a.run) if f.startswith('attr_') and f[6:11].isdigit())
  res = {}
  for s in [int(t) for t in a.steps.split(',')]:
    model.load_state_dict(torch.load(os.path.join(a.run, f'model_{s:05d}.pt'), map_location=dev))
    P = probe_paths(model, x, y, batch=a.batch, device=dev)
    nxt = [t for t in ckpts if t > s]
    cb1 = cb2 = None
    if nxt and os.path.exists(os.path.join(a.run, f'attr_{s:05d}.pt')):
      cb1, cb2 = load_cb(os.path.join(a.run, f'attr_{s:05d}.pt')), load_cb(os.path.join(a.run, f'attr_{nxt[0]:05d}.pt'))
    res[s] = {}
    for b in P:
      w = {m: cb2[f'{b}.pw_{m}'] - cb1[f'{b}.pw_{m}'] for m in ('gate', 'value')} if cb1 else None
      res[s][b] = summarize(P[b], w)
      r = res[s][b]
      print(s, b, {k: round(r[k]['top20'], 2) for k in SPLITS}, 'rho', round(r.get('rho_writes', np.nan), 2),
            'writes', round(r.get('writes_top20', np.nan), 2), flush=True)
    torch.save(P, os.path.join(a.run, f'bias_paths_raw_{s:05d}.pt'))
  json.dump(res, open(a.out or os.path.join(a.run, 'bias_paths.json'), 'w'), indent=1)
  print('BIAS_PATHS_DONE', flush=True)


if __name__ == '__main__':
  main()
