"""Copies the looping results, training checkpoints, saved models and queue logs from a molab sandbox to a public
Hugging Face dataset while the runs are in progress, and restores results and checkpoints on a fresh sandbox.

Usage: python3 loops_hf_sync.py [--repo joshhf/rcdl-loops] [--every 300] [--once] [--restore]
The token is read from /marimo/work/.hf_token; it is never printed. Only files matching the allow-lists below are
uploaded (looping results and checkpoints, saved facts models, queue logs). A file is uploaded again when its size or
modification time changes. Parts of a run's earlier checkpoint are deleted from the dataset once the local meta file
names newer parts (rcdl/ckpt.py keeps only the latest checkpoint), so the dataset holds one checkpoint per run.
--restore downloads the results and the latest checkpoint of each run that are missing locally (after the sandbox was
replaced), then exits.
"""

import argparse
import fcntl
import fnmatch
import glob
import json
import os
import sys
import time

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi, hf_hub_download

SOURCES = [                                              # (local folder, folder in the dataset, glob patterns)
  ('/marimo/work/persist', 'persist', ('facts[ABCGL]_*.py', 'ckpt_facts_*.py')),
  ('/tmp/models', 'models', ('model_*.py',)),
  ('/marimo/work', 'logs', ('q1[7-9]_logs/*.log', 'q2[0-9]_logs/*.log', 'm_q1[7-9].out', 'm_q2[0-9].out')),
]
MAX_FILES, MAX_BYTES = 200, 400 * 2 ** 20                # per commit


def local_files():
  out = {}
  for root, dest, pats in SOURCES:
    for pat in pats:
      for path in glob.glob(os.path.join(root, pat)):
        try:
          st = os.stat(path)
        except OSError:                                # removed meanwhile (an old checkpoint part)
          continue
        out[f'{dest}/{os.path.relpath(path, root)}'] = (path, st.st_size, st.st_mtime)
  return out


def stale_ckpt_parts(remote, local):
  """Remote parts of a checkpoint whose local meta names other parts (a newer checkpoint replaced them)."""
  out = []
  for meta_path, (path, _, _) in local.items():
    if not meta_path.startswith('persist/ckpt_facts_') or not meta_path.endswith('_meta.py'):
      continue
    try:
      text = open(path).read()
      parts = set(json.loads(text[len('META = '):].strip()[1:-1])['parts'])
    except (OSError, ValueError, KeyError):
      continue
    prefix = meta_path[len('persist/'):-len('_meta.py')]
    for r in remote:
      name = r[len('persist/'):] if r.startswith('persist/') else None
      if name and name.startswith(prefix + '_s') and fnmatch.fnmatch(name, prefix + '_s*_p*.py') and name not in parts:
        out.append(r)
  return out


def sync_once(api, repo, sent):
  local = local_files()
  todo = [k for k, (_, size, mtime) in sorted(local.items()) if sent.get(k) != (size, mtime)]
  remote = set(api.list_repo_files(repo, repo_type='dataset'))
  deletes = [r for r in stale_ckpt_parts(remote, local) if r not in local]
  n_up = 0
  while todo or deletes:
    batch, size = [], 0
    while todo and len(batch) < MAX_FILES and (not batch or size + local[todo[0]][1] <= MAX_BYTES):
      k = todo.pop(0)
      batch.append(k)
      size += local[k][1]
    ops = [CommitOperationAdd(path_in_repo=k, path_or_fileobj=local[k][0]) for k in batch]
    ops += [CommitOperationDelete(path_in_repo=r) for r in deletes]
    deletes = []
    api.create_commit(repo, ops, repo_type='dataset',
                      commit_message=f'sync {len(batch)} files' + (f', remove {len(ops) - len(batch)} old parts'
                                                                   if len(ops) > len(batch) else ''))
    for k in batch:
      sent[k] = local[k][1:]
    n_up += len(batch)
  return n_up


def restore(api, repo):
  """Downloads results and the latest checkpoint of each run that are missing in /marimo/work/persist."""
  P = '/marimo/work/persist'
  os.makedirs(P, exist_ok=True)
  remote = api.list_repo_files(repo, repo_type='dataset')
  got = 0
  for r in remote:
    if not r.startswith('persist/'):
      continue
    name = r[len('persist/'):]
    is_result = fnmatch.fnmatch(name, 'facts[ABCGL]_*.py')
    is_meta = fnmatch.fnmatch(name, 'ckpt_facts_*_meta.py')
    if not (is_result or is_meta) or os.path.exists(os.path.join(P, name)):
      continue
    if is_meta:
      result = 'facts?_' + name[len('ckpt_facts_'):-len('_meta.py')] + '.py'
      if any(fnmatch.fnmatch(os.path.basename(x), result) for x in os.listdir(P)):
        continue                                       # that run is finished
      meta = json.loads(open(hf_hub_download(repo, r, repo_type='dataset')).read()[len('META = '):].strip()[1:-1])
      for part in meta['parts']:
        _copy(hf_hub_download(repo, 'persist/' + part, repo_type='dataset'), os.path.join(P, part))
    _copy(hf_hub_download(repo, r, repo_type='dataset'), os.path.join(P, name))   # the meta file last
    got += 1
  print('restored', got, 'files or checkpoints', flush=True)


def _copy(src, dst):
  tmp = dst + '.tmp'
  with open(src, 'rb') as a, open(tmp, 'wb') as b:
    b.write(a.read())
  os.replace(tmp, dst)


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--repo', default='joshhf/rcdl-loops')
  p.add_argument('--every', type=int, default=300)
  p.add_argument('--once', action='store_true')
  p.add_argument('--restore', action='store_true')
  p.add_argument('--token_file', default='/marimo/work/.hf_token')
  a = p.parse_args()
  api = HfApi(token=open(a.token_file).read().strip())
  if a.restore:
    restore(api, a.repo)
    return
  lock = open('/tmp/loops_hf_sync.lock', 'w')
  try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
  except OSError:
    print('another sync is running', flush=True)
    return
  sent, loc = {}, local_files()
  try:                                                 # files already in the dataset with the same size are skipped
    for f in api.list_repo_tree(a.repo, repo_type='dataset', recursive=True):
      if getattr(f, 'size', None) is not None and f.path in loc and loc[f.path][1] == f.size:
        sent[f.path] = loc[f.path][1:]
  except Exception as e:                               # an empty or new dataset
    print('listing failed:', type(e).__name__, flush=True)
  while True:
    try:
      n = sync_once(api, a.repo, sent)
      print(time.strftime('%H:%M:%S'), 'uploaded', n, flush=True)
    except Exception as e:                             # network or Hub errors: try again next round
      print(time.strftime('%H:%M:%S'), 'sync failed:', type(e).__name__, str(e)[:200], flush=True)
    if a.once:
      break
    time.sleep(a.every)


if __name__ == '__main__':
  sys.exit(main())
