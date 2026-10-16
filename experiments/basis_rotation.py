"""Does a search over rotations of a layer land back on its neurons? (the essay: "if you search the rotations of a layer
for the one where a handful of units carry the most activity, starting from a random one, you land back on the neurons,
in every block").

For each block of a ConvNeXt-SwiGLU checkpoint: X (N, U) = SwiGLU products u at N sampled (image, position) pairs of
validation images (positions sampled uniformly per image). Over orthogonal U x U matrices Q, maximize the quartimax
concentration J(Q) = sum_ij (X Q)_ij^4 (a rotation keeps the total energy sum_ij (XQ)_ij^2 fixed, so a larger J means
the activity sits on fewer directions), by the fixed point Q <- polar(X^T (X Q)^3) (the polar factor U V^T of the SVD),
from `starts` random rotations and from the identity (the neurons). Reported per block:
  align      for each found direction, max_i |Q_ij| (1: it is a unit axis); mean over directions and the share > 0.9
  J_ratio    J at the end / J at the neurons (identity); > 1: a rotation concentrates the activity more than the neurons
  identity   the same for the search started at the identity (does it stay)
Control: the same search on X R0 for a random rotation R0, where the neurons are not special by construction; the
neurons of X are then the columns of R0^T, so the found directions must align with those (align_R0) and not with the unit
axes (align_units), with J_ratio_vs_R0 measured against Q = R0^T.

Usage: python3 -m experiments.basis_rotation --run <run dir> --val <validation parquet dir> [--images 5000]
       [--per_image 16] [--starts 5] [--out file.json]
"""

import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.convnext import build_convnext
from rcdl.imagenet import val_batches


@torch.no_grad()
def sample_units(model, val_dir, images, per_image, seed=0, batch=250):
  dev = next(model.parameters()).device
  g = torch.Generator(device='cpu').manual_seed(seed)
  named = model.blocks()
  feats = {}
  hooks = [b.norm.register_forward_hook(lambda m, i, o, n=n, b=b: feats.__setitem__(n, F.silu(b.pw_gate(o)) * b.pw_value(o)))
           for n, b in named]
  out = {n: [] for n, _ in named}
  seen = 0
  for x, _ in val_batches(val_dir, batch, dev):
    feats.clear()
    model(x)
    for n, _ in named:
      u = feats[n].flatten(1, 2)                                     # (B, positions, U)
      idx = torch.randint(0, u.shape[1], (u.shape[0], per_image), generator=g).to(dev)
      out[n].append(torch.gather(u, 1, idx[:, :, None].expand(-1, -1, u.shape[2])).flatten(0, 1).float().cpu())
    seen += len(x)
    if seen >= images:
      break
  for h in hooks:
    h.remove()
  return {n: torch.cat(v).double() for n, v in out.items()}


def quartimax(X, Q, iters=500, tol=1e-10):
  J_prev = None
  for _ in range(iters):
    Y = X @ Q
    U_, _, Vt = torch.linalg.svd(X.T @ Y ** 3)
    Q = U_ @ Vt
    J = float(((X @ Q) ** 4).sum())
    if J_prev is not None and abs(J - J_prev) <= tol * abs(J_prev):
      break
    J_prev = J
  return Q, float(((X @ Q) ** 4).sum())


def random_rotation(U, g):
  A = torch.randn(U, U, generator=g, dtype=torch.float64)
  Q, R = torch.linalg.qr(A)
  return Q * torch.sign(torch.diagonal(R))


def align(Q, axes=None):
  """Mean over the columns of Q of the largest |cos| with the columns of `axes` (identity: unit axes), and the share
  above 0.9."""
  C = Q.abs() if axes is None else (axes.T @ Q).abs()
  m = C.max(0).values
  return float(m.mean()), float((m > 0.9).double().mean())


def analyze(X, starts, g):
  X = X / X.pow(2).sum(1).mean().sqrt()                            # scale only (J ratios are scale-free)
  U = X.shape[1]
  J0 = float((X ** 4).sum())
  Qi, Ji = quartimax(X, torch.eye(U, dtype=X.dtype))
  rec = {'identity': {'align': align(Qi), 'J_ratio': Ji / J0}, 'random': []}
  for _ in range(starts):
    Q, J = quartimax(X, random_rotation(U, g))
    rec['random'].append({'align': align(Q), 'J_ratio': J / J0})
  R0 = random_rotation(U, g)                                       # control: neurons not special by construction
  Xr = X @ R0
  Q, J = quartimax(Xr, random_rotation(U, g))
  rec['control'] = {'align_units': align(Q), 'align_R0': align(Q, R0.T), 'J_ratio_vs_R0': J / float(((Xr @ R0.T) ** 4).sum())}
  return rec


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', required=True)
  p.add_argument('--val', required=True)
  p.add_argument('--step', type=int, default=None)
  p.add_argument('--images', type=int, default=5000)
  p.add_argument('--per_image', type=int, default=16)
  p.add_argument('--starts', type=int, default=5)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default=None)
  a = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  steps = sorted(int(f[-8:-3]) for f in glob.glob(os.path.join(a.run, 'model_[0-9]*.pt')))
  step = a.step if a.step is not None else steps[-1]
  model = build_convnext(cfg['size']).to(dev).eval()
  model.load_state_dict(torch.load(os.path.join(a.run, f'model_{step:05d}.pt'), map_location=dev))
  Xs = sample_units(model, a.val, a.images, a.per_image, a.seed)
  g = torch.Generator().manual_seed(a.seed)
  res = {'run': a.run, 'step': step, 'images': a.images, 'per_image': a.per_image, 'blocks': {}}
  for n, X in Xs.items():
    ok = torch.isfinite(X).all(1)                                  # guard (the validation loader once gave NaN pixels)
    X = X[ok]
    r = analyze(X, a.starts, g)
    r['dropped_nonfinite_rows'] = int((~ok).sum())
    res['blocks'][n] = r
    ra = np.array([s['align'][0] for s in r['random']]); rj = np.array([s['J_ratio'] for s in r['random']])
    print(n, json.dumps({'N': len(X), 'identity_align': r['identity']['align'], 'identity_J': round(r['identity']['J_ratio'], 3),
                         'random_align_mean': round(float(ra.mean()), 3), 'random_J': rj.round(3).tolist(),
                         'control_align_units': r['control']['align_units'], 'control_align_R0': r['control']['align_R0']}),
          flush=True)
  json.dump(res, open(a.out or os.path.join(a.run, f'basis_rotation_{step:05d}.json'), 'w'), indent=1)


if __name__ == '__main__':
  main()
