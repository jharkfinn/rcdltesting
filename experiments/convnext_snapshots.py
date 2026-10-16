"""Replay snapshots from a run's own history on the Hugging Face results dataset.

hf_sync uploads <run>/resume.pt every 10 minutes, and the dataset keeps every version in its git history. Each
distinct version is a full training state at the step it was saved: the model, the optimizer states, and every
group's parts and momenta. This script lists those versions and converts each one into

- snap_{step}.pt: a replay snapshot in the format of convnext_train.py --snapshots (model, optimizer states, the tracked
  layers' momentum summed over groups, second moments, RNG states), for --init_from. The run's crops were not seeded
  per batch, so a replay must be compared with a control replay from the same snapshot, not with the original run;
- with --full, full_{step}.pt: the fp32 per-group parts and moments (W0, b0, C, Cb, m, mb, v, vb, a, ab, t) of every
  tracked map, for exact splits of an interval into per-group writes.

Each version is checked: the tracked weights equal a W_0 + sum_k C_k to fp32 rounding.

  python3 -m experiments.convnext_snapshots --run e1_tiny_adamw --out /marimo/work/results/e1_tiny_adamw/history
"""

import argparse
import json
import os

import torch
from huggingface_hub import HfApi, hf_hub_download

from rcdl.convnext import build_convnext

REPO = 'joshhf/rcdl-molab-results'


def list_versions(api: HfApi, run: str, repo: str = REPO, prefix: str = 'neuron-dev'):
  """Distinct versions of <prefix>/<run>/resume.pt in the dataset's history, oldest first:
  [(commit_id, created_at, lfs sha256)]."""
  path = f'{prefix}/{run}/resume.pt'
  seen, out = set(), []
  for c in api.list_repo_commits(repo, repo_type='dataset'):
    info = api.get_paths_info(repo, [path], repo_type='dataset', revision=c.commit_id)
    if not info or getattr(info[0], 'lfs', None) is None:
      continue
    sha = info[0].lfs.sha256
    if sha not in seen:
      seen.add(sha)
      out.append((c.commit_id, c.created_at, sha))
  return sorted(out, key=lambda r: r[1])


def to_snapshot(st: dict, source: str) -> dict:
  """A resume state as a replay snapshot (GroupAttribution.snapshot form: momentum summed over groups)."""
  attr = {name: {'m': s['m'].sum(0), 'mb': None if s['mb'] is None else s['mb'].sum(0), 'v': s['v'], 'vb': s['vb'],
                 't': s['t']} for name, s in st['attr'].items()}
  return {'step': st['step'], 'model': st['model'], 'opts': st['opts'], 'attr': attr, 'torch_rng': st['torch_rng'],
          'cuda_rng': st['cuda_rng'], 'log': st['log'], 'batch_crops': False, 'source': source}


def check_parts(st: dict, size: str) -> float:
  """Largest relative difference, over the tracked maps, between the weights and a W_0 + sum_k C_k."""
  model = build_convnext(size)
  model.load_state_dict(st['model'])
  named = dict(model.blocks())
  worst = 0.0
  for name, s in st['attr'].items():
    bn, mp = name.rsplit('.', 1)
    lin = getattr(named[bn], mp) if bn in named else getattr(model, name)
    W = lin.weight.detach().float()
    rec = s['a'] * s['W0'].float() + s['C'].float().sum(0)
    worst = max(worst, float((rec - W).abs().max() / W.abs().max()))
  return worst


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', required=True)
  p.add_argument('--out', required=True)
  p.add_argument('--repo', default=REPO)
  p.add_argument('--full', action='store_true', help='also keep the fp32 per-group parts and moments')
  p.add_argument('--full_out', default=None, help='directory for full_{step}.pt (default: --out)')
  p.add_argument('--token', default=os.environ.get('HF_TOKEN_FILE', '/marimo/work/.hf_token'))
  a = p.parse_args()
  tok = open(a.token).read().strip()
  api = HfApi(token=tok)
  os.makedirs(a.out, exist_ok=True)
  full_out = a.full_out or a.out
  os.makedirs(full_out, exist_ok=True)
  tmp = os.path.join(full_out, 'tmp')
  cfg = json.load(open(hf_hub_download(a.repo, f'neuron-dev/{a.run}/config.json', repo_type='dataset', token=tok,
                                       local_dir=tmp)))
  versions = list_versions(api, a.run, a.repo)
  print(f'{len(versions)} versions of {a.run}/resume.pt', flush=True)
  index = []
  for commit, when, sha in versions:
    path = hf_hub_download(a.repo, f'neuron-dev/{a.run}/resume.pt', repo_type='dataset', token=tok,
                           revision=commit, local_dir=tmp)
    st = torch.load(path, map_location='cpu', weights_only=False)
    step = int(st['step'])
    resid = check_parts(st, cfg['size'])
    assert resid < 1e-4, f'step {step}: parts do not add up ({resid:.2e})'
    torch.save(to_snapshot(st, f'{a.run}@{commit}'), os.path.join(a.out, f'snap_{step:05d}.pt'))
    if a.full:
      torch.save({'step': step, 'source': f'{a.run}@{commit}', 'attr': st['attr']},
                 os.path.join(full_out, f'full_{step:05d}.pt'))
    os.remove(path)
    index.append({'step': step, 'commit': commit, 'created_at': str(when), 'sha256': sha, 'resid': resid})
    print(f'step {step} ({when}) resid {resid:.1e}', flush=True)
  json.dump(index, open(os.path.join(a.out, 'history.json'), 'w'), indent=1)
  print('SNAPSHOTS_DONE', flush=True)


if __name__ == '__main__':
  main()
