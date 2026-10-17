"""experiments.convnext_train storage and safety flags (--track none, --attr_steps, --resume_at_ckpts, the non-finite
loss abort, max_mem_gb in log.json), as functions and through short CPU runs of the trainer on a packed fake training
set (tests/test_imagenet._pack_tiny) and a fake validation parquet; the defaults keep the earlier behaviour."""

import json
import sys

import pytest
import torch

from experiments import convnext_train as ct
from rcdl import convnext
from rcdl.convnext import ConvNeXt
from test_imagenet import _pack_tiny

MICRO = dict(dims=(8, 8, 8, 8), units=(6, 6, 6, 6), depths=(1, 2, 1, 1))     # blocks stage0.block0 .. stage3.block0


def _val_parquet(d, n=6, seed=1):
  import pyarrow as pa
  import pyarrow.parquet as pq
  from torchvision.io import encode_jpeg
  g = torch.Generator().manual_seed(seed)
  imgs = [{'bytes': encode_jpeg(torch.randint(0, 256, (3, 48, 56), generator=g, dtype=torch.uint8)).numpy().tobytes(),
           'path': None} for _ in range(n)]
  d.mkdir()
  pq.write_table(pa.table({'image': imgs, 'label': [i % 5 for i in range(n)]}), d / 'validation-00000-of-00001.parquet')


@pytest.fixture
def run(tmp_path, monkeypatch):
  """run(out, *args): the trainer's main() on the CPU (one thread) with the micro size; returns out."""
  (tmp_path / 'train').mkdir()
  _pack_tiny(tmp_path / 'train')
  _val_parquet(tmp_path / 'val')
  monkeypatch.setitem(convnext.SIZES, 'micro', MICRO)
  threads = torch.get_num_threads()
  torch.set_num_threads(1)

  def go(out, *args):
    out = tmp_path / out
    monkeypatch.setattr(sys, 'argv', ['convnext_train', '--device', 'cpu', '--size', 'micro', '--batch', '4',
                                      '--warmup', '5', '--val_limit', '4', '--batch_crops', '--train',
                                      str(tmp_path / 'train'), '--val', str(tmp_path / 'val'), '--out', str(out),
                                      *map(str, args)])
    ct.main()
    return out
  yield go
  torch.set_num_threads(threads)


def _files(out, prefix):
  return sorted(int(f.stem.split('_')[1]) for f in out.glob(f'{prefix}_*.pt') if f.stem.split('_')[1].isdigit())


def _load(path):
  return torch.load(path, map_location='cpu', weights_only=False)


# ------------------------------------------------------------------------------------------------- functions

def test_tracked_blocks_none_and_unknown_names():
  m = ConvNeXt(**MICRO, n_classes=4)
  names = [n for n, _ in m.blocks()]
  assert [n for n, _ in ct.tracked_blocks(m, 'all')] == names
  assert ct.tracked_blocks(m, 'none') == []
  assert [n for n, _ in ct.tracked_blocks(m, 'late')] == ['stage3.block0']
  assert [n for n, _ in ct.tracked_blocks(m, 'stage3.block0,stage1.block1')] == ['stage1.block1', 'stage3.block0']
  for bad in ('stage3.block2', 'stage3.block0,stage1.blok1', '', ','):        # a typo used to track nothing silently
    with pytest.raises(ValueError):
      ct.tracked_blocks(m, bad)


def test_resume_due_default_is_the_old_rule():
  for every, last in ((1000, 25020), (2500, 25020), (7, 600)):
    for s in range(1, last + 1):
      old = s % every == 0 or s in ct.CKPT_STEPS or s == last
      assert ct.resume_due(s, every, 1, last) == old
      assert ct.resume_due(s, every, 0, last) == (s % every == 0 or s == last)
  assert [s for s in range(1, 25021) if ct.resume_due(s, 2500, 0, 25020)] == list(range(2500, 25001, 2500)) + [25020]


def test_parse_steps():
  assert ct.parse_steps('') == set() and ct.parse_steps(None) == set()
  assert ct.parse_steps('2500,10000, 17500,25020') == {2500, 10000, 17500, 25020}
  assert ct.parse_steps('50040:200160:25020,5') == {5, 50040, 75060, 100080, 125100, 150120, 175140, 200160}


def test_report_loss_aborts_on_a_non_finite_loss(capsys):
  assert ct.report_loss(100, [torch.tensor(1.0), torch.tensor(2.0)], 12.3) == pytest.approx(1.5)
  assert capsys.readouterr().out == 'step 100 loss 1.5000 12s\n'        # the print as before
  for bad in (float('nan'), float('inf')):
    with pytest.raises(SystemExit) as e:
      ct.report_loss(200, [torch.tensor(1.0), torch.tensor(bad)], 0)
    assert e.value.code == ct.NONFINITE_EXIT != 0
  assert ct.max_mem_gb('cpu') == 0.0


# ----------------------------------------------------------------------------------------------- CPU runs

def test_track_none_end_to_end_with_resume(run):
  """--track none: attribution over no maps, snapshots, checkpoints and resume states all work; a run stopped at step 50
  and resumed ends with the same weights as an uninterrupted one; log.json has max_mem_gb (0 on the CPU)."""
  full = run('full', '--track', 'none', '--steps', 60, '--snapshots')
  cfg = json.load(open(full / 'config.json'))
  assert cfg['tracked'] == [] and cfg['attr_steps'] == '' and cfg['resume_at_ckpts'] == 1
  assert _files(full, 'model') == _files(full, 'attr') == _files(full, 'snap') == [0, 50, 60]
  assert _load(full / 'attr_00060.pt') == {} and _load(full / 'attr_init.pt') == {}
  snap = _load(full / 'snap_00050.pt')
  assert snap['attr'] == {} and snap['cuda_rng'] is None
  log = json.load(open(full / 'log.json'))
  assert [r['step'] for r in log] == [0, 50, 60] and all(r['max_mem_gb'] == 0.0 for r in log)
  assert _load(full / 'resume.pt')['step'] == 60
  part = run('part', '--track', 'none', '--steps', 60, '--snapshots', '--max_steps', 50)
  st = _load(part / 'resume.pt')
  assert st['step'] == 50 and st['attr'] == {} and not (part / 'model_00060.pt').exists()
  run('part', '--track', 'none', '--steps', 60, '--snapshots')             # resumes at 50
  a, b = _load(full / 'model_00060.pt'), _load(part / 'model_00060.pt')
  assert all(torch.equal(a[k], b[k]) for k in a)
  assert [r['step'] for r in json.load(open(part / 'log.json'))] == [0, 50, 60]


def test_attr_steps_and_resume_at_ckpts(run):
  """--attr_steps: parts (and the random split) only at the listed checkpoints, attr_init.pt as before;
  --resume_at_ckpts 0: no resume state at checkpoint step 50, only every --resume_every steps and at the last step."""
  args = ('--track', 'stage3.block0', '--rand_groups', '--steps', 60, '--attr_steps', '50,60', '--resume_every', 40)
  out = run('a', *args, '--resume_at_ckpts', 0, '--max_steps', 50)
  assert _files(out, 'model') == [0, 50] and _files(out, 'attr') == _files(out, 'attrR') == [50]
  assert sorted(_load(out / 'attr_init.pt')) == ['stage3.block0.pw_gate', 'stage3.block0.pw_value']
  assert _load(out / 'resume.pt')['step'] == 40
  run('a', *args, '--resume_at_ckpts', 0)                                   # resumes at 40
  assert _files(out, 'attr') == _files(out, 'attrR') == [50, 60] and _load(out / 'resume.pt')['step'] == 60
  whole = run('w', *args, '--resume_at_ckpts', 0)                          # uninterrupted: the same weights and parts
  for f in ('model_00060.pt', 'attr_00060.pt', 'attrR_00060.pt'):
    a, b = _load(out / f), _load(whole / f)
    assert all(torch.equal(a[k], b[k]) for k in a) if f.startswith('model') else \
      all(torch.equal(a[n]['C'], b[n]['C']) and torch.equal(a[n]['R'], b[n]['R']) for n in a)
  ref = run('b', *args, '--max_steps', 50)                                  # default: also at checkpoint steps
  assert _load(ref / 'resume.pt')['step'] == 50
  dflt = run('c', '--track', 'stage3.block0', '--steps', 50)                # default: parts at every checkpoint
  assert _files(dflt, 'attr') == [0, 50]


def test_bad_track_and_attr_steps_are_errors(run):
  for args in (('--track', 'stage3.block2'), ('--track', 'none', '--attr_steps', '55')):
    with pytest.raises(SystemExit) as e:
      run('bad', '--steps', 60, *args)
    assert e.value.code == 2                                                # argparse error, before any training


def test_non_finite_loss_ends_the_run(run, tmp_path, capsys):
  """At the step-100 print, before that step's checkpoint and resume state."""
  with pytest.raises(SystemExit) as e:
    run('nan', '--track', 'none', '--steps', 150, '--lr', 1e38)                # the weights overflow
  assert e.value.code == ct.NONFINITE_EXIT
  assert 'ABORT: non-finite running loss at step 100' in capsys.readouterr().out
  assert _files(tmp_path / 'nan', 'model') == [0, 50] and _load(tmp_path / 'nan' / 'resume.pt')['step'] == 50
