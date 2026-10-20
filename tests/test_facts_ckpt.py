"""Training checkpoints of rcdl.facts.run_facts: a run resumed from a checkpoint equals the uninterrupted run."""

import pytest
import torch

from rcdl import ckpt as ckpt_lib
from rcdl.facts import FactsConfig, run_facts

BASE = dict(n_people=32, n_attrs=2, n_values=8, d=16, heads=2, layers=2, width=24, name_tokens=2, name_vocab=16,
            n_filler=3, random_name_pos=True, steps=40, eval_every=5, batch_size=8)


@pytest.mark.parametrize('kw', [dict(), dict(ffn='cos', cos_sigma=2.0, loops=3, train_probe=True, lr_schedule='cosine'),
                                dict(ffn='geglu', loops=2, loop_mode='infiloop', deep_supervision=True, final_probe=True)])
def test_resumed_run_equals_the_uninterrupted_run(tmp_path, monkeypatch, kw):
  cfg = FactsConfig(**(BASE | kw))
  full = run_facts(cfg)
  prefix = str(tmp_path / 'ckpt_facts_t')
  real, calls = torch.optim.Adam.step, [0]

  def crash(self, *a, **k):
    calls[0] += 1
    if calls[0] > 27:
      raise RuntimeError('sandbox reclaimed')
    return real(self, *a, **k)

  monkeypatch.setattr(torch.optim.Adam, 'step', crash)
  with pytest.raises(RuntimeError):
    run_facts(cfg, ckpt=prefix, ckpt_every=10)
  monkeypatch.setattr(torch.optim.Adam, 'step', real)
  assert ckpt_lib.load(prefix)['step'] == 20
  resumed = run_facts(cfg, ckpt=prefix, ckpt_every=10)
  strip = lambda log: [{k: v for k, v in r.items() if k != 'sec'} for r in log]
  assert [r['step'] for r in resumed['log']] == [r['step'] for r in full['log']]
  assert strip(resumed['log']) == strip(full['log'])
  assert resumed['final_acc'] == full['final_acc'] and resumed.get('loop_probe') == full.get('loop_probe')


def test_checkpoint_settings_and_other_configurations(tmp_path):
  prefix = str(tmp_path / 'ckpt_facts_u')
  a = run_facts(FactsConfig(**(BASE | dict(steps=20))), ckpt=prefix, ckpt_every=10)
  assert ckpt_lib.load(prefix)['step'] == 10            # the last step (20) is not saved: the run is complete
  b = run_facts(FactsConfig(**(BASE | dict(steps=20, lr=3e-3))), ckpt=prefix, ckpt_every=10)
  assert b['log'][0]['step'] == 0 and b['cfg']['lr'] == 3e-3          # another configuration starts from scratch
  plain = run_facts(FactsConfig(**(BASE | dict(steps=20))))
  strip = lambda log: [{k: v for k, v in r.items() if k != 'sec'} for r in log]
  assert strip(a['log']) == strip(plain['log'])                       # saving checkpoints changes nothing
  with pytest.raises(ValueError):
    run_facts(FactsConfig(**BASE), ckpt=prefix, ckpt_every=7)
  with pytest.raises(ValueError):
    run_facts(FactsConfig(**(BASE | dict(finetune_people=4, finetune_steps=10))), ckpt=prefix, ckpt_every=10)
