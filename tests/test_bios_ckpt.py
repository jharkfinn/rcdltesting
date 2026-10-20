"""Chunked .py checkpoints (rcdl/ckpt.py) and resumed biography pre-training (rcdl/bios.run_bios with ckpt)."""

import glob
import os

import pytest
import torch

from rcdl import ckpt as ckpt_lib
from rcdl.bios import BiosConfig, BioWorld, run_bios

BASE = dict(n_people=32, seq_len=128, batch_size=8, d_model=32, layers=2, heads=2, kv_heads=1, head_dim=16,
            ffn_width=64, steps=40, eval_every=10, eval_bios=16, lr=3e-3, cos_width=128)


def test_chunked_checkpoint_round_trip_replaces_old_parts_and_rejects_damage(tmp_path):
  prefix = str(tmp_path / 'ckpt_x')
  assert ckpt_lib.load(prefix) is None
  obj = {'a': torch.randn(5000), 'b': [1, 2, 3]}
  ckpt_lib.save(prefix, obj, step=10, chunk_mb=0.004)
  parts10 = sorted(glob.glob(prefix + '_s10_p*.py'))
  assert len(parts10) > 2 and all(open(p).read().startswith("DATA = '") for p in parts10)
  back = ckpt_lib.load(prefix)
  torch.testing.assert_close(back['a'], obj['a'], rtol=0, atol=0)
  assert back['b'] == [1, 2, 3]
  ckpt_lib.save(prefix, {'a': torch.zeros(3)}, step=20, chunk_mb=0.004)
  assert not glob.glob(prefix + '_s10_p*.py') and glob.glob(prefix + '_s20_p*.py')
  assert ckpt_lib.load(prefix)['a'].tolist() == [0, 0, 0]
  part = glob.glob(prefix + '_s20_p*.py')[0]
  text = open(part).read()
  open(part, 'w').write(text[:-3] + ('A' if text[-3] != 'A' else 'B') + text[-2:])
  assert ckpt_lib.load(prefix) is None                         # sha256 mismatch
  os.remove(part)
  assert ckpt_lib.load(prefix) is None                         # missing part
  ckpt_lib.remove(prefix)
  assert not glob.glob(prefix + '*')


@pytest.mark.parametrize('kw', [dict(), dict(ffn='cos', ffn_loops=2, eval_seen_templates=True)])
def test_resumed_run_equals_the_uninterrupted_run(tmp_path, monkeypatch, kw):
  cfg = BiosConfig(**(BASE | kw))
  full = run_bios(cfg)
  prefix = str(tmp_path / 'ckpt_bios_t')
  real, calls = BioWorld.train_batch, [0]

  def crash(self, *a, **k):
    calls[0] += 1
    if calls[0] > 30:
      raise RuntimeError('sandbox reclaimed')
    return real(self, *a, **k)

  monkeypatch.setattr(BioWorld, 'train_batch', crash)
  with pytest.raises(RuntimeError):
    run_bios(cfg, ckpt=prefix, ckpt_every=20)
  monkeypatch.setattr(BioWorld, 'train_batch', real)
  assert ckpt_lib.load(prefix)['step'] == 20
  resumed = run_bios(cfg, ckpt=prefix, ckpt_every=20)
  strip = lambda log: [{k: v for k, v in r.items() if k != 'sec'} for r in log]
  assert [r['step'] for r in resumed['log']] == [r['step'] for r in full['log']]
  assert strip(resumed['log']) == strip(full['log'])
  assert resumed['final_attr_acc'] == full['final_attr_acc']


def test_checkpoint_of_another_configuration_is_ignored_and_bad_settings_raise(tmp_path):
  prefix = str(tmp_path / 'ckpt_bios_u')
  run_bios(BiosConfig(**(BASE | dict(steps=20))), ckpt=prefix, ckpt_every=10)
  assert ckpt_lib.load(prefix)['step'] == 10
  r = run_bios(BiosConfig(**(BASE | dict(steps=20, lr=1e-3))), ckpt=prefix, ckpt_every=10)
  assert r['log'][0]['step'] == 0 and r['cfg']['lr'] == 1e-3
  with pytest.raises(ValueError):
    run_bios(BiosConfig(**BASE), ckpt=prefix, ckpt_every=15)
  with pytest.raises(ValueError):
    run_bios(BiosConfig(**(BASE | dict(special='rbf', rbf_capacity=1000))), ckpt=prefix, ckpt_every=10)
