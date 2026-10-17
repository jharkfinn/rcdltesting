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


@pytest.mark.parametrize('kind', ['gws', 'gauge', 'rbf'])
def test_bf16_storage_matches_fp32_on_same_database(kind):
  """store_dtype='bfloat16' (used for L = 2 at 50 epochs): a bf16 model trains and grows like the fp32 model, and on
  the same database (fp32 rows rounded to bf16) its outputs and input gradients agree within 2% (relative L2).

  Training trajectories are not compared: in this tiny model (training loss 3-50) they separate within a few steps
  from any perturbation, so they say nothing about the precision of the layer. The CIFAR check is a 10-epoch L = 2
  run against the fp32 run (docs/REPRODUCTION.md, section 7.4).
  """
  kw = small_cfg(kernel_type=kind, hidden_dim=64, temperature=10.0, eta=0.3 if kind == 'rbf' else 0.003, beta=10.0)
  if kind == 'rbf':
    kw['db_size'] = 48                                   # RBF has no D_0 rows: 12 steps x 4 samples
  ref = CifarRCDLClassifier(RCDLConfig(**kw))
  low = CifarRCDLClassifier(RCDLConfig(**kw, store_dtype='bfloat16'))
  for x, y in batches(12, d=64):
    rcdl_train_step(ref, None, x.clone(), y)
    rcdl_train_step(low, None, x.clone(), y)
  for (name, a), (_, b) in zip(ref.rcdl_layers().items(), low.rcdl_layers().items()):
    assert b.keys.dtype == torch.bfloat16 and b.weights.dtype == torch.get_default_dtype() and a.size == b.size, name
    b.keys.copy_(a.keys), b.values.copy_(a.values), b.weights.copy_(a.weights)
    b.valid_rows.copy_(a.valid_rows)
    if hasattr(b, 'key_sq'):
      b.key_sq.copy_(b.keys.float().pow(2).sum(-1))
  tx = torch.randn(16, 64, generator=torch.Generator().manual_seed(5), requires_grad=True)
  rel = lambda u, v: ((u - v).norm() / u.norm()).item()
  ref.eval(), low.eval()
  oa, ob = ref(tx), low(tx)
  ga, = torch.autograd.grad(oa.pow(2).sum(), tx)
  gb, = torch.autograd.grad(ob.pow(2).sum(), tx)
  assert rel(oa, ob) < 2e-2 and rel(ga, gb) < 2e-2, (rel(oa, ob), rel(ga, gb))
  with torch.no_grad():
    for m in (ref, low):
      for layer in m.rcdl_layers().values():
        layer.linearize = True
    assert rel(ref(tx), low(tx)) < 2e-2


def test_bf16_storage_rejects_unsupported_options():
  with pytest.raises(ValueError):                        # W_slow (frozen random projection) is not supported
    CifarRCDLClassifier(RCDLConfig(**small_cfg(kernel_type='rbf', db_size=48, store_dtype='bfloat16',
                                               freeze_init_out=False)))
  with pytest.raises(ValueError):                        # age coefficients would copy the whole table to fp32
    CifarRCDLClassifier(RCDLConfig(**small_cfg(kernel_type='gws', store_dtype='bfloat16', model_momentum=0.9)))


def test_bf16_parametric_softmax_trains_like_fp32():
  """store_dtype='bfloat16' for the parametric softmax (full-scale Fig. 5): bf16 parameters, same initial function up to
  bf16 rounding, and training with Adam stays close to the fp32 model over a few steps."""
  kw = small_cfg(kernel_type='parametric_softmax', db_size=40, learning_rate=1e-2, temperature=1.0,
                 learn_parametric_weights=False, num_layers=0)
  ref, low = (CifarRCDLClassifier(RCDLConfig(**kw, store_dtype=s)) for s in ('float32', 'bfloat16'))
  assert low.output_layer.keys.dtype == torch.bfloat16 and low.output_layer.keys.requires_grad
  tx = torch.randn(8, 24, generator=torch.Generator().manual_seed(4))
  with torch.no_grad():
    rel0 = ((ref(tx) - low(tx)).norm() / ref(tx).norm()).item()
  assert rel0 < 2e-2, rel0
  opts = [create_rcdl_optimizer(m, m.cfg) for m in (ref, low)]
  for x, y in batches(6):
    for m, o in zip((ref, low), opts):
      rcdl_train_step(m, o, x.clone(), y)
  ref.eval(), low.eval()
  with torch.no_grad():
    rel = ((ref(tx) - low(tx)).norm() / ref(tx).norm()).item()
  assert rel < 0.1, rel


def test_bf16_second_moment_stalls_and_fp32_adam_matches_reference():
  """bf16 cannot track v <- 0.999 v + 0.001 g² (it never decreases); AdamFP32SecondMoment matches fp32 Adam closely."""
  from rcdl.train import AdamFP32SecondMoment
  v = torch.tensor([1.0], dtype=torch.bfloat16)
  for _ in range(1000):
    v = v * 0.999 + 0.001 * 0.25
  assert v.item() == 1.0                                   # stuck; fp32 would reach 0.526
  g = torch.Generator().manual_seed(0)
  w0 = torch.randn(5, 7, generator=g)
  grads = [torch.randn(5, 7, generator=g) * (0.5 ** k) for k in range(30)]   # shrinking gradients
  ref = torch.nn.Parameter(w0.clone().bfloat16().float())          # same (bf16-rounded) start for all three
  low, plain = (torch.nn.Parameter(w0.clone().bfloat16()) for _ in range(2))
  o_ref, o_low = torch.optim.Adam([ref], lr=1e-2), AdamFP32SecondMoment([low], lr=1e-2, chunk_rows=2)
  o_plain = torch.optim.Adam([plain], lr=1e-2)                     # bf16 m and v (torch default for bf16 params)
  for gr in grads:
    ref.grad, low.grad, plain.grad = gr.bfloat16().float(), gr.bfloat16(), gr.bfloat16()
    o_ref.step(), o_low.step(), o_plain.step()
  v_ref, v_low = o_ref.state[ref]['exp_avg_sq'], o_low.state[low]['v']
  assert v_low.dtype == torch.float32 and torch.allclose(v_low, v_ref, rtol=1e-4, atol=1e-12)
  err = lambda p: (p.float() - ref).norm().item()
  assert err(low) < err(plain)                                     # closer to fp32 Adam than all-bf16 Adam


def test_bf16_rbf_keeps_key_norms():
  """The stored ||k_j||^2 of a bf16 RBF layer equals the norm of the stored (rounded) key after training."""
  m = CifarRCDLClassifier(RCDLConfig(**small_cfg(kernel_type='rbf', db_size=48, store_dtype='bfloat16')))
  for x, y in batches(5):
    rcdl_train_step(m, None, x.clone(), y)
  for layer in m.rcdl_layers().values():
    n = layer.size
    assert n == 20 and torch.allclose(layer.key_sq[:n], layer.keys[:n].float().pow(2).sum(-1))


def test_eval_is_independent_of_batch_size():
  kw = small_cfg(kernel_type='gws', model_momentum=0.5)
  m = CifarRCDLClassifier(RCDLConfig(**kw))
  for x, y in batches(8):
    rcdl_train_step(m, None, x, y)
  tx, ty = torch.randn(12, 24), torch.randint(0, 5, (12,))
  a, b = rcdl_evaluate(m, tx, ty, 3), rcdl_evaluate(m, tx, ty, 12)
  assert abs(a[0] - b[0]) < 1e-5 and a[1] == b[1]


def test_stochastic_rounding_is_unbiased_and_keeps_small_adam_steps():
  """stochastic_round_bf16 returns a bf16 neighbour of x with mean x; with it, Adam steps below half the bf16 spacing
  move the parameters on average (round to nearest leaves them at their start value)."""
  from rcdl.train import AdamFP32SecondMoment
  from rcdl.triton_adam import stochastic_round_bf16
  x = torch.full((200_000,), 1.0 - 1e-3)                    # bf16 neighbours 1 - 2^-8 and 1
  r = stochastic_round_bf16(x, torch.Generator().manual_seed(0)).float()
  assert set(r.unique().tolist()) == {1.0 - 2 ** -8, 1.0}
  assert abs(r.mean().item() - x[0].item()) < 4 * (2 ** -8) * 0.5 / math.sqrt(len(x))
  assert x.bfloat16().float().unique().tolist() == [1.0]    # round to nearest: the offset is lost
  w = {s: torch.nn.Parameter(torch.ones(100_000, 4, dtype=torch.bfloat16)) for s in (False, True)}
  opts = {s: AdamFP32SecondMoment([w[s]], lr=1e-4, stochastic_rounding=s, seed=1) for s in (False, True)}
  for _ in range(20):                                       # Adam moves each entry by ~lr = 1e-4 per step
    for s in (False, True):
      w[s].grad = torch.ones_like(w[s])
      opts[s].step()
  assert w[False].float().unique().tolist() == [1.0]
  assert abs(w[True].float().mean().item() - (1.0 - 20 * 1e-4)) < 2e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA and Triton')
def test_fused_triton_adam_matches_chunked_update():
  """The fused kernel (rcdl/triton_adam.py) gives the chunked PyTorch update: v to fp32 rounding, p and m to one bf16
  spacing (the two paths order the fp32 operations differently). With stochastic rounding it is unbiased on the GPU."""
  from rcdl.train import AdamFP32SecondMoment
  g = torch.Generator(device='cuda').manual_seed(0)
  w0 = torch.randn(3001, 37, device='cuda', generator=g).bfloat16()
  grads = [torch.randn(3001, 37, device='cuda', generator=g).bfloat16() * (0.7 ** k) for k in range(12)]
  ws = [torch.nn.Parameter(w0.clone()) for _ in range(2)]
  opts = [AdamFP32SecondMoment([w], lr=1e-2, fused=f) for w, f in zip(ws, (True, False))]
  for gr in grads:
    for w, o in zip(ws, opts):
      w.grad = gr.clone()
      o.step()
  (a, b), (sa, sb) = ws, (opts[0].state[ws[0]], opts[1].state[ws[1]])
  assert torch.allclose(sa['v'], sb['v'], rtol=1e-5, atol=1e-12)
  for x, y in ((a, b), (sa['m'], sb['m'])):
    ulp = 2.0 ** (torch.floor(torch.log2(y.float().abs().clamp_min(1e-30))) - 7)
    assert ((x.float() - y.float()).abs() <= ulp + 1e-12).all()
  p = torch.nn.Parameter(torch.ones(1_000_000, device='cuda', dtype=torch.bfloat16))
  o = AdamFP32SecondMoment([p], lr=1e-4, stochastic_rounding=True, fused=True)
  for _ in range(20):
    p.grad = torch.ones_like(p)
    o.step()
  assert abs(p.float().mean().item() - (1.0 - 20 * 1e-4)) < 5e-5
