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
