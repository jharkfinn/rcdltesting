"""Copies a results folder to a private Hugging Face dataset every few minutes (sandboxes can end without notice).

Usage: python3 experiments/hf_sync.py [--root /marimo/work/results] [--prefix neuron-dev] [--every 600]
                                      [--extra_ignore PATTERN ...]
The token is read from /marimo/work/.hf_token (or --token_file); it is never printed. Unchanged files are not
uploaded again (the Hub skips content it already has). Partial files (*.tmp) are skipped, and so are the files matching
--extra_ignore (fnmatch patterns on paths relative to --root, as upload_folder applies them; e.g. '*/resume.pt' keeps
the runs' resume states local).

sync_status(path) tells whether the hf_sync processes running on this machine would upload a file (experiments/
molab_medium.sh uses it to keep resume.pt local).
"""

import argparse
import os
import time

from huggingface_hub import HfApi

IGNORE = ['*.tmp', 'bench*', '*buffers*']


def parser():
  p = argparse.ArgumentParser()
  p.add_argument('--root', default='/marimo/work/results')
  p.add_argument('--prefix', default='neuron-dev')
  p.add_argument('--repo', default='joshhf/rcdl-molab-results')
  p.add_argument('--token_file', default='/marimo/work/.hf_token')
  p.add_argument('--every', type=int, default=600)
  p.add_argument('--extra_ignore', nargs='+', action='extend', default=[],
                 help='more ignore patterns for the folder upload (e.g. \'*/resume.pt\')')
  return p


def ignore_patterns(a) -> list:
  return IGNORE + list(a.extra_ignore)


def running_syncs(proc: str = '/proc') -> list:
  """[(pid, argv, args)] of the hf_sync processes running here (python3 [...]/hf_sync.py or -m experiments.hf_sync),
  args parsed as this script would, with args.root made absolute against the process's working directory."""
  out = []
  for pid in filter(str.isdigit, os.listdir(proc)):
    try:
      argv = open(os.path.join(proc, pid, 'cmdline'), 'rb').read().decode(errors='replace').split('\0')[:-1]
      cwd = os.readlink(os.path.join(proc, pid, 'cwd'))
    except OSError:                                     # ended meanwhile, or not ours to read
      continue
    at = next((i + 1 for i, t in enumerate(argv) if os.path.basename(t) == 'hf_sync.py'), None)
    at = at or next((i + 2 for i, t in enumerate(argv[:-1]) if t == '-m' and argv[i + 1] == 'experiments.hf_sync'), None)
    if at is None or int(pid) == os.getpid():
      continue
    args = parser().parse_known_args(argv[at:])[0]
    args.root = os.path.realpath(os.path.join(cwd, args.root))
    out.append((int(pid), argv, args))
  return out


def sync_status(path: str, proc: str = '/proc') -> str:
  """'none' if no running hf_sync has `path` under its root, 'ignored' if every one that has leaves it out, else
  'uploaded:<pids>' (comma-separated) for those that would upload it."""
  from huggingface_hub.utils import filter_repo_objects
  path = os.path.realpath(path)
  covering = [(pid, a) for pid, _, a in running_syncs(proc) if path.startswith(a.root.rstrip('/') + '/')]
  up = [pid for pid, a in covering
        if list(filter_repo_objects([os.path.relpath(path, a.root)], ignore_patterns=ignore_patterns(a)))]
  return 'none' if not covering else f'uploaded:{",".join(map(str, up))}' if up else 'ignored'


def main():
  a = parser().parse_args()
  api = HfApi(token=open(a.token_file).read().strip())
  api.create_repo(a.repo, repo_type='dataset', private=True, exist_ok=True)
  while True:
    try:
      if any(files for _, _, files in os.walk(a.root)):
        api.upload_folder(folder_path=a.root, path_in_repo=a.prefix, repo_type='dataset', repo_id=a.repo,
                          ignore_patterns=ignore_patterns(a),
                          commit_message=f'{a.prefix}: results at {time.strftime("%H:%M UTC", time.gmtime())}')
        print(time.strftime('%H:%M:%S'), 'uploaded', flush=True)
    except Exception as e:                       # keep going; report the error type only
      print(time.strftime('%H:%M:%S'), 'upload failed:', type(e).__name__, str(e)[:200], flush=True)
    time.sleep(a.every)


if __name__ == '__main__':
  main()
