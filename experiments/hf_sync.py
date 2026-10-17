"""Copies a results folder to a private Hugging Face dataset every few minutes (sandboxes can end without notice).

Usage: python3 experiments/hf_sync.py [--root /marimo/work/results] [--prefix neuron-dev] [--every 600]
                                      [--extra_ignore PATTERN ...]
The token is read from /marimo/work/.hf_token (or --token_file); it is never printed. Unchanged files are not
uploaded again (the Hub skips content it already has). Partial files (*.tmp) are skipped, and so are the files matching
--extra_ignore (fnmatch patterns on paths relative to --root, as upload_folder applies them; e.g. '*/resume.pt' keeps
the runs' resume states local).

sync_status(path) tells whether the hf_sync processes running on this machine would upload a file; ensure_sync(path,
...) makes sure one runs that leaves it out (experiments/molab_medium.sh, to keep resume.pt local).
"""

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
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
  args parsed as this script would, with args.root and args.token_file made absolute against the process's working
  directory."""
  out = []
  for pid in filter(str.isdigit, os.listdir(proc)):
    try:
      argv = open(os.path.join(proc, pid, 'cmdline'), 'rb').read().decode(errors='replace').split('\0')[:-1]
      cwd = os.readlink(os.path.join(proc, pid, 'cwd'))
    except OSError:                                     # ended meanwhile, or not ours to read
      continue
    cwd = cwd[:-len(' (deleted)')] if cwd.endswith(' (deleted)') else cwd    # its checkout was re-cloned
    if not argv or not os.path.basename(argv[0]).startswith('python') or int(pid) == os.getpid():
      continue                                          # (an editor open on hf_sync.py is not a sync)
    at = next((i + 1 for i, t in enumerate(argv) if os.path.basename(t) == 'hf_sync.py'), None)
    at = at or next((i + 2 for i in range(len(argv) - 1) if argv[i:i + 2] == ['-m', 'experiments.hf_sync']), None)
    if at is None:
      continue
    args = parser().parse_known_args(argv[at:])[0]
    args.root = os.path.realpath(os.path.join(cwd, args.root))
    args.token_file = os.path.join(cwd, args.token_file)
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


def sync_command(a, extra=()) -> list:
  """The command that starts hf_sync from a repo checkout with the arguments a (as parsed by parser()) and the ignore
  patterns `extra` added."""
  ign = list(a.extra_ignore) + [x for x in extra if x not in a.extra_ignore]
  return ([sys.executable, 'experiments/hf_sync.py', '--root', a.root, '--prefix', a.prefix, '--repo', a.repo,
           '--token_file', a.token_file, '--every', str(a.every)] + (['--extra_ignore', *ign] if ign else []))


def _trainers(root: str, proc: str) -> list:
  """[(pid, out, size)] of the convnext_train processes writing under root (size from out/config.json, '' if none)."""
  out = []
  for pid in filter(str.isdigit, os.listdir(proc)):
    try:
      argv = open(os.path.join(proc, pid, 'cmdline'), 'rb').read().decode(errors='replace').split('\0')[:-1]
      cwd = os.readlink(os.path.join(proc, pid, 'cwd'))
    except OSError:
      continue
    if 'experiments.convnext_train' not in argv or '--out' not in argv[:-1]:
      continue
    d = os.path.realpath(os.path.join(cwd, argv[argv.index('--out') + 1]))
    if d.startswith(root.rstrip('/') + '/'):
      try:
        size = json.load(open(os.path.join(d, 'config.json')))['size']
      except (OSError, ValueError, KeyError):
        size = ''
      out.append((int(pid), d, size))
  return out


def _alive(pid: int, proc: str) -> bool:
  try:                                                  # (a zombie has ended)
    return open(os.path.join(proc, str(pid), 'stat')).read().rsplit(')', 1)[1].split()[0] != 'Z'
  except (OSError, IndexError):
    return False


def ensure_sync(path: str, root: str, repo_dir: str, log: str, restart: bool = False, ignore: str = '*/resume.pt',
                token_file: str = '/marimo/work/.hf_token', lock: str = '/tmp/rcdl_locks/hf_sync.lock',
                wait: float = 15.0, proc: str = '/proc', ours=lambda size: size.startswith('medium')) -> int:
  """Makes sure an hf_sync runs that leaves `path` out (experiments/molab_medium.sh), under a lock, so that runs
  started at once start one sync:
    'ignored' -> nothing to do;
    'none'    -> starts one for root with --extra_ignore `ignore`;
    'uploaded:<pids>' -> 1 (HF_SYNC_UPLOADS_RESUME), or with restart those are stopped and started again from
                 repo_dir with their own arguments (not their old command line, whose checkout may be gone or predate
                 --extra_ignore) plus --extra_ignore `ignore`; refused (1) while trainers other than ours (by their
                 config.json size) write under their root, which would lose their uploaded resume states.
  The pattern covers every run under the root: other trainers there are named in a warning. A started sync must still
  run after `wait` seconds and leave `path` out, else 7 (HF_SYNC_DEAD); its token file is checked first. Returns 0 if
  a sync leaves `path` out. Prints what it did; the token is never read here."""
  os.makedirs(os.path.dirname(lock), exist_ok=True)
  with open(lock, 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    st = sync_status(path, proc)
    if st == 'ignored':
      print(f'hf_sync running leaves {path} out', flush=True)
      return 0
    if st.startswith('uploaded:') and not restart:
      print(f'HF_SYNC_UPLOADS_RESUME: hf_sync process(es) {st[9:]} would upload {path}. Restart them with '
            f"--extra_ignore '{ignore}' (or rerun with HF_SYNC_RESTART=1, which applies the pattern to every run under "
            'their root)', flush=True)
      return 1
    if st == 'none':
      a = parser().parse_args(['--root', os.path.realpath(root), '--token_file', token_file])
      todo = [(None, a)]
    else:
      pids = {int(p) for p in st[9:].split(',')}
      todo = [(pid, a) for pid, _, a in running_syncs(proc) if pid in pids]
    for _, a in todo:
      other = [(pid, d) for pid, d, size in _trainers(a.root, proc) if not ours(size)]
      if other and st != 'none':
        print(f'HF_SYNC_UPLOADS_RESUME: not restarting the hf_sync for {a.root}: trainers {other} also write there and '
              f"would lose their uploaded resume states under --extra_ignore '{ignore}'", flush=True)
        return 1
      if other:
        print(f"WARNING: the new hf_sync leaves out every resume state under {a.root} ('{ignore}'), also those of "
              f'{other}', flush=True)
      for f, what in ((a.token_file, 'token file'), (os.path.join(repo_dir, 'experiments', 'hf_sync.py'), 'script')):
        if not (os.path.isfile(f) and os.path.getsize(f) > 0):
          print(f'HF_SYNC_DEAD: no {what} {f}', flush=True)
          return 7
    started = []
    for pid, a in todo:
      if pid is not None:
        for sig in (signal.SIGTERM, signal.SIGKILL):
          try:
            os.kill(pid, sig)
          except ProcessLookupError:
            break
          for _ in range(50):
            if not _alive(pid, proc):
              break
            time.sleep(0.2)
        print(f'hf_sync {pid} stopped', flush=True)
      with open(log, 'a') as f:
        p = subprocess.Popen(sync_command(a, [ignore]), cwd=repo_dir, stdin=subprocess.DEVNULL, stdout=f,
                             stderr=subprocess.STDOUT, start_new_session=True)
      started.append(p)
      print(f"hf_sync {p.pid} started for {a.root} without '{ignore}' (log: {log})", flush=True)
    t = time.time() + wait
    while time.time() < t and all(p.poll() is None for p in started):
      time.sleep(0.5)
    dead = [(p.pid, p.returncode) for p in started if p.poll() is not None]
    st = sync_status(path, proc)
    if dead or st != 'ignored':
      print(f'HF_SYNC_DEAD: started hf_sync ended (pid, exit) {dead}, status of {path}: {st} (see {log})',
            flush=True)
      return 7
    return 0


def main():
  a = parser().parse_args()
  api = HfApi(token=open(a.token_file).read().strip())
  created = False
  while True:
    try:
      if not created:                            # in the loop: a Hub error at start is retried, not fatal
        api.create_repo(a.repo, repo_type='dataset', private=True, exist_ok=True)
        created = True
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
