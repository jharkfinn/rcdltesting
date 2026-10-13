"""Checks the rcdl package against the original notebook code (executed from rcdl_tutorial_torch.ipynb itself)."""

import dataclasses as dc
import json
import math
import os
from typing import Any, Dict, Optional, Tuple

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.layers import RCDLConfig
from rcdl.models import CifarRCDLClassifier
from rcdl.train import create_rcdl_optimizer, rcdl_evaluate, rcdl_train_step

NOTEBOOK = os.path.join(os.path.dirname(__file__), '..', 'rcdl_tutorial_torch.ipynb')


@pytest.fixture(scope='module')
def nb():
  """Namespace holding the notebook's cells 3.1 (modules) and 4.1 (train step), executed verbatim."""
  with open(NOTEBOOK) as f:
    cells = [''.join(c['source']) for c in json.load(f)['cells'] if c['cell_type'] == 'code']
  ns = dict(dc=dc, math=math, torch=torch, nn=nn, F=F, Optional=Optional, Dict=Dict, Any=Any, Tuple=Tuple,
            DEVICE=torch.device('cpu'))
  for title in ('#@title 3.1', '#@title 4.1'):
    exec(next(c for c in cells if c.startswith(title)), ns)
  return ns


def small_cfg(**kw):
  base = dict(hidden_dim=24, num_layers=2, db_size=4 + 4 * 12, full_epoch_capacity=4 + 4 * 12, batch_size=4,
              num_classes=5, temperature=0.7, eta=0.3, beta=2.0, global_init_seed=3)
  return base | kw


def batches(n_steps, B=4, d=24, n_cls=5, seed=0):
  g = torch.Generator().manual_seed(seed)
  return [(torch.randn(B, d, generator=g) * 2.0, torch.randint(0, n_cls, (B,), generator=g)) for _ in range(n_steps)]


def run_pair(nb, kw, steps=12, eval_bs=4):
  ours = CifarRCDLClassifier(RCDLConfig(**kw))
  ref = nb['CifarRCDLClassifier'](nb['RCDLConfig'](**{k: v for k, v in kw.items() if k in
                                                      nb['RCDLConfig'].__dataclass_fields__}))
  opt_o, opt_r = create_rcdl_optimizer(ours, ours.cfg), nb['create_rcdl_optimizer'](ref, ref.cfg)
  test_x = torch.randn(8, kw['hidden_dim'], generator=torch.Generator().manual_seed(9))
  test_y = torch.zeros(8, dtype=torch.long)
  for x, y in batches(steps, kw['batch_size'], kw['hidden_dim'], kw['num_classes']):
    lo, _ = rcdl_train_step(ours, opt_o, x.clone(), y)
    lr, _ = nb['rcdl_train_step'](ref, opt_r, x.clone(), y)
    assert abs(lo - lr) < 1e-4 * max(1.0, abs(lr))
  ours.eval(), ref.eval()
  with torch.no_grad():
    out_o = torch.cat([ours(test_x[i:i + eval_bs]) for i in range(0, 8, eval_bs)])
    out_r = torch.cat([ref(test_x[i:i + eval_bs]) for i in range(0, 8, eval_bs)])
  return ours, ref, out_o, out_r


@pytest.mark.parametrize('kind,extra', [
    ('gws', {}), ('gws', {'eta': 50.0}), ('gauge', {'gauge_alpha': 3.0}), ('rbf', {'db_size': 48}),
    ('hybrid', {'growth_threshold': 6.0, 'learning_rate': 1e-2}),
    ('parametric_softmax', {'db_size': 10, 'learning_rate': 1e-2}),
    ('standard', {'nonlinearity': 'gelu', 'learning_rate': 1e-3}),
    ('standard', {'nonlinearity': 'gelu', 'learning_rate': 1e-2, 'optimizer': 'muon'}),
])
def test_port_matches_notebook(nb, kind, extra):
  torch.manual_seed(0)
  ours, ref, out_o, out_r = run_pair(nb, small_cfg(kernel_type=kind, **extra))
  assert torch.allclose(out_o, out_r, atol=1e-4, rtol=1e-4)
  for (name, lo), (_, lr) in zip(ours.rcdl_layers().items(), ref.rcdl_layers().items()):
    assert lo.size == lr.size, name
    for attr in ('keys', 'values', 'weights'):
      assert torch.allclose(getattr(lo, attr), getattr(lr, attr), atol=1e-5, rtol=1e-4), (name, attr)


@pytest.mark.parametrize('kind', ['gws', 'rbf'])
def test_momentum_matches_notebook_only_at_train_batch_size(nb, kind):
  """With momentum, the notebook's roll-based mask is right only when the forward batch equals the training batch.
  (Table not yet full here; see the next test for the full-table case.)"""
  kw = small_cfg(kernel_type=kind, model_momentum=0.9, model_weight_decay=0.01, db_size=60 if kind == 'rbf' else 64)
  torch.manual_seed(0)
  _, _, out_o, out_r = run_pair(nb, kw, eval_bs=4)
  assert torch.allclose(out_o, out_r, atol=1e-4, rtol=1e-4)
  torch.manual_seed(0)
  _, _, out_o8, out_r8 = run_pair(nb, kw, eval_bs=8)
  assert torch.allclose(out_o8, out_o, atol=1e-4, rtol=1e-4)      # ours: independent of the eval batch size
  assert not torch.allclose(out_r8, out_r, atol=1e-3, rtol=1e-3)  # notebook: changes with the eval batch size


def test_notebook_mask_zeroes_oldest_rows_when_table_is_exactly_full(nb):
  """The notebook computes ages modulo the table size. Once the table is exactly full (the 1-epoch configs end in this
  state: 16 + 50,000 rows), the oldest batch gets momentum age 0, i.e. coefficient (1 - μ^0) / (1 - μ) = 0."""
  n, B = 52, 4
  mask = nb['decay_mask'](n, B, 0, 0.9, 0.0, 'cpu')[:, 0]   # 12 steps done: db_idx = (4 + 48) % 52 = 0
  assert torch.all(mask[:B] == 0.0) and torch.all(mask[B:] > 0.0)


@pytest.mark.parametrize('kind', ['gws', 'gauge', 'rbf'])
def test_skipping_first_layer_input_grad_keeps_updates(kind):
  kw = small_cfg(kernel_type=kind, db_size=48 if kind == 'rbf' else 52)
  models = [CifarRCDLClassifier(RCDLConfig(**kw)) for _ in range(2)]
  for x, y in batches(10):
    rcdl_train_step(models[0], None, x.clone(), y, input_grad=True)
    rcdl_train_step(models[1], None, x.clone(), y, input_grad=False)
  for (_, a), (_, b) in zip(models[0].rcdl_layers().items(), models[1].rcdl_layers().items()):
    for attr in ('keys', 'values', 'weights'):
      assert torch.equal(getattr(a, attr), getattr(b, attr))


def test_sdpa_attention_matches_dense():
  """The fused path (softmax(z) @ [ṽ, w]) equals the notebook's dense weighted softmax, including clipped (w = 0) rows.

  The two differ only when a query's denominator is near 0 (its attention falls almost entirely on w = 0 rows): the
  dense code floors Σ w e^{z - max z} at 1e-8, the fused code floors the normalized average at 1e-30.
  """
  from rcdl.layers import weighted_attention, weighted_attention_sdpa
  g = torch.Generator().manual_seed(1)
  q, k = torch.randn(7, 16, generator=g), torch.randn(40, 16, generator=g)
  v, w = torch.randn(40, 5, generator=g), torch.rand(40, 1, generator=g)
  w[::3] = 0.0
  a = weighted_attention(q, k, v, w, 0.8)
  b = weighted_attention_sdpa(q, k, v, w, 0.8)
  assert torch.allclose(a, b, rtol=1e-4, atol=1e-5)


def test_vit_with_growing_ffn_trains_and_grows():
  from rcdl.models import CifarViT
  cfg = RCDLConfig(kernel_type='gws', arch='vit', hidden_dim=16, num_layers=2, vit_heads=2, vit_patch=8,
                   batch_size=4, db_size=4 + 17 * 4 * 3, temperature=1.0, eta=0.1, beta=1.0, attn_impl='sdpa',
                   learning_rate=1e-3)
  m = CifarViT(cfg)
  opt = create_rcdl_optimizer(m, cfg)
  for x, y in batches(3, B=4, d=3072, n_cls=5):
    rcdl_train_step(m, opt, x, y % 10)
  sizes = [layer.size for layer in m.rcdl_layers().values()]
  assert sizes == [4 + 17 * 4 * 3] * 2           # D_0 + (16 patches + CLS) tokens x 4 images x 3 steps
  loss, acc = rcdl_evaluate(m, torch.randn(6, 3072), torch.zeros(6, dtype=torch.long), 3)
  assert loss == loss                             # finite


def test_eval_is_independent_of_batch_size():
  kw = small_cfg(kernel_type='gws', model_momentum=0.5)
  m = CifarRCDLClassifier(RCDLConfig(**kw))
  for x, y in batches(8):
    rcdl_train_step(m, None, x, y)
  tx, ty = torch.randn(12, 24), torch.randint(0, 5, (12,))
  a, b = rcdl_evaluate(m, tx, ty, 3), rcdl_evaluate(m, tx, ty, 12)
  assert abs(a[0] - b[0]) < 1e-5 and a[1] == b[1]
