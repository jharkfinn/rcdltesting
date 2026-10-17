"""experiments.hf_sync: --extra_ignore (default: the upload ignores exactly what it did before) and sync_status, which
reads the running hf_sync processes from /proc (a fake one here) to tell whether a training run's resume.pt would be
uploaded."""

import os

from experiments import hf_sync as hs


def test_ignore_patterns_default_and_extra_for_training_runs():
  from huggingface_hub.utils import filter_repo_objects
  assert hs.ignore_patterns(hs.parser().parse_args([])) == ['*.tmp', 'bench*', '*buffers*']
  a = hs.parser().parse_args(['--extra_ignore', '*/resume.pt', '--every', '5', '--extra_ignore', 'x*', 'y'])
  assert hs.ignore_patterns(a) == ['*.tmp', 'bench*', '*buffers*', '*/resume.pt', 'x*', 'y'] and a.every == 5
  files = ['m10/resume.pt', 'm10/resume.pt.tmp', 'm10/model_25020.pt', 'm10/microscope/data_stage3_block2.json',
           'bench_m10/resume.pt']
  assert list(filter_repo_objects(files, ignore_patterns=hs.ignore_patterns(a))) == files[2:4]


def _proc(root, pid, argv, cwd):
  d = root / str(pid)
  d.mkdir(parents=True)
  (d / 'cmdline').write_bytes(b'\0'.join(t.encode() for t in argv) + b'\0')
  os.symlink(cwd, d / 'cwd')


def test_sync_status_finds_syncs_that_would_upload_a_training_resume_state(tmp_path):
  res, work, proc = tmp_path / 'work' / 'results', tmp_path / 'work', tmp_path / 'proc'
  (res / 'm10').mkdir(parents=True)
  (proc / 'self').mkdir(parents=True)
  resume, model = str(res / 'm10' / 'resume.pt'), str(res / 'm10' / 'model_25020.pt')
  _proc(proc, 7, ['python3', '-m', 'experiments.convnext_train', '--out', str(res / 'm10')], str(work))
  _proc(proc, 8, ['vim', 'experiments/hf_sync.py'], str(work))
  assert hs.sync_status(resume, str(proc)) == 'none'
  _proc(proc, 11, ['python3', '-m', 'experiments.hf_sync', '--root', 'results', '--extra_ignore', '*/resume.pt'],
        str(work))                                                      # relative root: against the process's cwd
  assert hs.sync_status(resume, str(proc)) == 'ignored'
  assert hs.sync_status(model, str(proc)) == 'uploaded:11'
  assert hs.sync_status(str(tmp_path / 'elsewhere' / 'resume.pt'), str(proc)) == 'none'
  _proc(proc, 12, ['/usr/bin/python3', 'experiments/hf_sync.py', '--root', str(work)], str(tmp_path))   # a parent root
  assert hs.sync_status(resume, str(proc)) == 'uploaded:12'
  pids = {pid: a.root for pid, _, a in hs.running_syncs(str(proc))}
  assert pids == {11: os.path.realpath(res), 12: os.path.realpath(work)}


# ------------------------------------------------------------------- ensure_sync, with real (fake) sync processes

import shutil
import signal
import subprocess
import sys
import threading
import time

import pytest

FAKE = 'import os, sys, time\nif os.path.exists("die"):\n  sys.exit(1)\ntime.sleep(120)\n'   # a sync that only runs


@pytest.fixture
def fake(tmp_path):
  """(repo, results root, token, kwargs for ensure_sync): a checkout whose experiments/hf_sync.py only sleeps (exits 1
  if the checkout has a file 'die'); every sync or trainer process started under tmp_path is killed afterwards."""
  repo, root = tmp_path / 'repo', tmp_path / 'results'
  (repo / 'experiments').mkdir(parents=True)
  (repo / 'experiments' / 'hf_sync.py').write_text(FAKE)
  root.mkdir()
  (tmp_path / 'tok').write_text('x')
  kw = dict(log=str(tmp_path / 'hf_sync.log'), token_file=str(tmp_path / 'tok'), lock=str(tmp_path / 'locks' / 'l'),
            wait=1.0)
  procs = []
  yield repo, root, kw, procs
  for p in procs:
    p.kill()
  for pid, _, a in hs.running_syncs():
    if a.root.startswith(str(tmp_path)):
      os.kill(pid, signal.SIGKILL)


def _mine(root):
  return sorted((pid, a.extra_ignore) for pid, _, a in hs.running_syncs() if a.root == os.path.realpath(root))


def test_ensure_sync_for_training_runs_starts_one_that_stays_up_and_reuses_it(fake, capsys):
  repo, root, kw, _ = fake
  resume = str(root / 'm10' / 'resume.pt')
  out = []
  ts = [threading.Thread(target=lambda: out.append(hs.ensure_sync(resume, str(root), str(repo), **kw)))
        for _ in range(2)]                                # two runs started at once: one sync (the lock)
  for t in ts:
    t.start()
  for t in ts:
    t.join()
  assert out == [0, 0]
  (pid, ign), = _mine(root)
  assert ign == ['*/resume.pt'] and hs.sync_status(resume) == 'ignored'
  assert hs.ensure_sync(resume, str(root), str(repo), **kw) == 0 and _mine(root) == [(pid, ign)]
  assert capsys.readouterr().out.count('started') == 1


def test_ensure_sync_for_training_runs_reports_a_sync_that_dies_or_has_no_token(fake, capsys):
  repo, root, kw, _ = fake
  resume = str(root / 'm10' / 'resume.pt')
  assert hs.ensure_sync(resume, str(root), str(repo), **(kw | {'token_file': str(root / 'none')})) == 7
  assert 'HF_SYNC_DEAD: no token file' in capsys.readouterr().out and _mine(root) == []
  (repo / 'die').write_text('')                           # e.g. a startup error: nothing may run on unnoticed
  assert hs.ensure_sync(resume, str(root), str(repo), **kw) == 7
  assert 'HF_SYNC_DEAD: started hf_sync ended' in capsys.readouterr().out and _mine(root) == []


def test_ensure_sync_for_training_runs_restarts_an_uploading_sync_from_a_recloned_checkout(fake, tmp_path, capsys):
  """The old sync's checkout was removed and cloned again (its cwd reads '<path> (deleted)'): it is stopped and
  started from the new checkout with its own arguments plus the pattern; refused while a non-medium trainer writes
  under the root, and without restart."""
  repo, root, kw, procs = fake
  resume = str(root / 'm10' / 'resume.pt')
  old = subprocess.Popen([sys.executable, 'experiments/hf_sync.py', '--root', str(root), '--every', '7',
                          '--extra_ignore', 'x*', '--token_file', '../tok2'], cwd=repo)    # relative: to its cwd
  procs.append(old)
  time.sleep(1)                                           # running (its script read) before the checkout goes
  shutil.rmtree(repo)
  (repo / 'experiments').mkdir(parents=True)
  (repo / 'experiments' / 'hf_sync.py').write_text(FAKE)
  time.sleep(0.5)
  assert os.readlink(f'/proc/{old.pid}/cwd').endswith(' (deleted)')
  assert hs.sync_status(resume) == f'uploaded:{old.pid}'
  assert hs.ensure_sync(resume, str(root), str(repo), **kw) == 1
  assert 'HF_SYNC_UPLOADS_RESUME' in capsys.readouterr().out
  (root / 'e1').mkdir()                                   # a tiny run of molab_e1.sh, which resumes from HF
  (root / 'e1' / 'config.json').write_text('{"size": "tiny"}')
  tr = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)', '-m', 'experiments.convnext_train',
                         '--out', str(root / 'e1')])
  procs.append(tr)
  time.sleep(0.5)
  assert hs.ensure_sync(resume, str(root), str(repo), restart=True, **kw) == 1
  assert 'not restarting' in capsys.readouterr().out and old.poll() is None
  (root / 'e1' / 'config.json').write_text('{"size": "medium_w8"}')    # a medium run: restart
  assert hs.ensure_sync(resume, str(root), str(repo), restart=True, **kw) == 7    # its token file: checked first
  assert 'HF_SYNC_DEAD: no token file' in capsys.readouterr().out and old.poll() is None
  (tmp_path / 'tok2').write_text('x')
  assert hs.ensure_sync(resume, str(root), str(repo), restart=True, **kw) == 0
  assert old.wait(5) is not None
  (pid, ign), = _mine(root)
  assert ign == ['x*', '*/resume.pt'] and hs.sync_status(resume) == 'ignored'
  argv = open(f'/proc/{pid}/cmdline').read().split('\0')
  assert argv[argv.index('--every') + 1] == '7' and os.readlink(f'/proc/{pid}/cwd') == str(repo)
  assert argv[argv.index('--token_file') + 1] == str(repo / '..' / 'tok2')
