"""How far a clean rerun is from the original run, against how far two clean reruns are from each other.

E1 was trained with the GPU decode race in the training loader (fixed since; docs/EXACT_HISTORY.md Sec. 5). The reruns
e1f_a and e1f_b use E1's flags and seed with the fixed loader and save models at E1's checkpoints and history steps. For
every step that two runs share, with theta the full parameter vector (or one group of it):

  d(X, Y) = ||theta_X - theta_Y|| / sqrt(||theta_X|| ||theta_Y||)

for the pairs (ref, a), (ref, b) and (a, b), over all parameters and per group (stem, stage i, head). With --probe, also
the share of probe images (8 per validation class) on which the two models' top-1 predictions agree and the mean
absolute difference of their logits. Models are read from model_{step}.pt and, for the reference, from
history/snap_{step}.pt (the 'model' entry).

  python3 -m experiments.convnext_rerun_compare --ref results/e1_tiny_adamw --runs results/e1f_a results/e1f_b \
      --out results/e1f_a/rerun_compare.json [--probe /dev/shm/probe_8.pt]
"""

import argparse
import glob
import json
import os

import torch


def load_models(run: str) -> dict:
  """{step: path} of every saved model of a run (model_*.pt, then history/snap_*.pt for steps not already present)."""
  out = {int(os.path.basename(f)[6:11]): f for f in glob.glob(os.path.join(run, 'model_[0-9]*.pt'))}
  for f in glob.glob(os.path.join(run, 'history', 'snap_[0-9]*.pt')):
    out.setdefault(int(os.path.basename(f)[5:10]), f)
  return out


def state(path: str) -> dict:
  st = torch.load(path, map_location='cpu', weights_only=False)
  return {k: v.float() for k, v in (st['model'] if 'model' in st else st).items()}


def group(name: str) -> str:
  if name.startswith('downsample.0'):
    return 'stem'
  if name.startswith(('head', 'norm')):
    return 'head'
  if name.startswith('stages.'):
    return f'stage{name.split(".")[1]}'
  if name.startswith('downsample.'):
    return f'stage{name.split(".")[1]}'                       # the downsample into stage i counts with stage i
  return 'other'


def distances(s1: dict, s2: dict) -> dict:
  acc = {}
  for k in s1:
    g = group(k)
    for key in ('all', g):
      a = acc.setdefault(key, [0.0, 0.0, 0.0])
      a[0] += float(((s1[k] - s2[k]) ** 2).sum())
      a[1] += float((s1[k] ** 2).sum())
      a[2] += float((s2[k] ** 2).sum())
  return {g: (d ** 0.5) / ((n1 * n2) ** 0.25 + 1e-30) for g, (d, n1, n2) in acc.items()}


@torch.no_grad()
def probe_agreement(s1: dict, s2: dict, size: str, probe: str, device) -> dict:
  from experiments.convnext_atlas import MEAN, STD
  from rcdl.convnext import build_convnext
  d = torch.load(probe)
  x = d['x']
  m1, m2 = build_convnext(size).to(device).eval(), build_convnext(size).to(device).eval()
  m1.load_state_dict(s1)
  m2.load_state_dict(s2)
  mean, std = MEAN.to(device)[:, None, None], STD.to(device)[:, None, None]
  agree, diff, n = 0, 0.0, 0
  for i in range(0, len(x), 200):
    xb = (x[i:i + 200].to(device).float() - mean) / std
    z1, z2 = m1(xb).float(), m2(xb).float()
    agree += int((z1.argmax(1) == z2.argmax(1)).sum())
    diff += float((z1 - z2).abs().mean(1).sum())
    n += len(xb)
  return {'top1_agree': agree / n, 'mean_abs_logit_diff': diff / n}


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--ref', required=True)
  p.add_argument('--runs', nargs=2, required=True)
  p.add_argument('--size', default='tiny')
  p.add_argument('--probe', default=None)
  p.add_argument('--out', required=True)
  a = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  M = {'ref': load_models(a.ref), 'a': load_models(a.runs[0]), 'b': load_models(a.runs[1])}
  res = {}
  for x, y in (('ref', 'a'), ('ref', 'b'), ('a', 'b')):
    steps = sorted(set(M[x]) & set(M[y]))
    res[f'{x}-{y}'] = {}
    for s in steps:
      s1, s2 = state(M[x][s]), state(M[y][s])
      rec = distances(s1, s2)
      if a.probe:
        rec.update(probe_agreement(s1, s2, a.size, a.probe, dev))
      res[f'{x}-{y}'][s] = rec
      print(x, y, s, {k: (f'{v:.3g}') for k, v in rec.items()}, flush=True)
  json.dump(res, open(a.out, 'w'), indent=1)
  print('RERUN_COMPARE_DONE', flush=True)


if __name__ == '__main__':
  main()
