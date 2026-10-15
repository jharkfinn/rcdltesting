"""Synthetic biographies on the Gemma 3 style decoder (rcdl/bios.py): data generator, metrics, growing RBF layer."""

import math
import re

import torch
import torch.nn.functional as F

from rcdl.bios import (ATTRS, BOS, N_TRAIN_TEMPLATES, PAD, BiosConfig, BioWorld, GemmaBios, GrowingRBF, _RBFRead,
                       evaluate, run_bios)


def decode_bio(world, row):
  return [world.itos[t] for t in row.tolist() if t != PAD]


def test_biographies_follow_the_generation_process():
  """Each biography: BOS, then six sentences (one per attribute, random order), each with the full name before the
  person's value; training biographies use the individual's 20 training templates, evaluation ones the other 5."""
  world = BioWorld(64, seed=3)
  gen = torch.Generator().manual_seed(0)
  people = torch.arange(64)
  for split in ('train', 'eval'):
    inputs, targets, fact, inst, route = world.eval_rows(people, split, gen)
    val_in = torch.cat([torch.zeros_like(fact[:, :1]), fact[:, :-1]], 1)
    assert torch.equal(route, fact | val_in)                     # routing: input in a value or next token starts one
    full = torch.cat([inputs[:, :1], targets], 1)
    for i in range(0, 64, 7):
      toks = full[i][full[i] != PAD]
      assert toks[0] == BOS and (toks == BOS).sum() == 1
      name = world.name_tok[i, :world.name_len[i]]
      starts = [j for j in range(len(toks) - len(name) + 1) if torch.equal(toks[j:j + len(name)], name)]
      assert len(starts) == 6                                      # the full name in every sentence
      vals = sorted(tuple(world.val_tok[i, a, :world.val_len[i, a]].tolist()) for a in range(6))
      ids = inst[i][fact[i]]
      got = sorted(tuple(targets[i][fact[i]][ids == k].tolist()) for k in ids.unique().tolist())
      assert got == vals                                           # value tokens are the person's six values
      assert sorted((ids % 6).unique().tolist()) == list(range(6))
      first_val = [int(torch.nonzero(fact[i] & (inst[i] == i * 6 + a))[0]) for a in range(6)]
      assert all(min(starts) < p for p in first_val)                # a name precedes every value
  # template choice respects the per-individual split
  _, valid, _, _ = world._assemble(people, 'train', gen)
  g = torch.Generator().manual_seed(5)
  order = torch.argsort(torch.rand(64, 6, generator=g), dim=1)
  col = torch.randint(0, N_TRAIN_TEMPLATES, (64, 6), generator=g)
  assert (world.split[people[:, None], order, col] >= 0).all()
  assert sorted(world.split[0, 0].tolist()) == list(range(25))


def test_packed_batches_mark_value_targets():
  world = BioWorld(32, seed=1)
  x, y, route = world.train_batch(0, 32, 4, 128, torch.Generator().manual_seed(0))
  assert x.shape == y.shape == route.shape == (4, 128)
  assert torch.equal(x[:, 1:], y[:, :-1])
  is_value_token = lambda s: re.fullmatch(r'[cumo]\d+|<(month|day|year)\d+>', s) is not None
  value_vocab = torch.tensor(sorted(world.stoi[s] for s in world.itos if is_value_token(s)))
  is_val_y, is_val_x = torch.isin(y, value_vocab), torch.isin(x, value_vocab)
  assert torch.equal(route, is_val_y | is_val_x)                   # every value position and the one before each value
  assert not torch.equal(route, is_val_y)                          # the last value token (target: closing word) is routed
  frac = route.float().mean().item()
  assert 0.1 < frac < 0.35


def test_attribute_metrics_on_known_logits():
  """Attribute loss = summed CE over a value's tokens; accuracy needs every token right."""
  world = BioWorld(8, seed=2)
  data = world.eval_rows(torch.arange(8), 'eval', torch.Generator().manual_seed(1))
  # no-knowledge baseline: shared prefixes (a 1-piece city that starts a 2-piece one) lower it below mean ln(count)
  assert world.no_knowledge_per_attr[0] == math.log(12 * 31 * 201)
  assert world.no_knowledge_loss < world.no_knowledge_upper < world.no_knowledge_loss + 0.5

  class Oracle(torch.nn.Module):
    def __init__(self, targets, wrong_first_token_of=None):
      super().__init__()
      self.t, self.w = targets, wrong_first_token_of

    def forward(self, tokens, fact=None):
      logits = torch.full((*tokens.shape, world.vocab_size), -30.0)
      logits.scatter_(-1, self.t[:len(tokens)][..., None], 30.0)
      return logits

  m = Oracle(data[1])
  r = evaluate(m, data)
  assert r['attr_acc'] == 1.0 and r['attr_loss'] < 1e-6
  inputs, targets, fact, inst, route = data
  bad = targets.clone()
  first = torch.nonzero(fact[0] & (inst[0] == 0))[0]               # first value token of instance 0
  bad[0, first] = (bad[0, first] + 1) % world.vocab_size
  r = evaluate(Oracle(bad), data)
  assert abs(r['attr_acc'] - (1 - 1 / 48)) < 1e-6
  assert abs(r['attr_loss'] - 60.0 / 48) < 1e-3                   # one token with CE ~ 60 nats, averaged over 48


def test_rbf_read_matches_autograd_and_chunking():
  g = torch.Generator().manual_seed(0)
  q = torch.randn(7, 5, generator=g, dtype=torch.float64, requires_grad=True)
  k, v = torch.randn(23, 5, generator=g, dtype=torch.float64), torch.randn(23, 3, generator=g, dtype=torch.float64)
  c = 4.0
  ref = torch.exp(-torch.cdist(q, k).pow(2) / c) @ v
  out = _RBFRead.apply(q.float(), k.float(), v.float(), k.float().pow(2).sum(-1), c, 4)   # 6 chunks
  assert torch.allclose(out.double(), ref, rtol=1e-4, atol=1e-5)
  w = torch.randn(7, 3, generator=g, dtype=torch.float64)
  (ref * w).sum().backward()
  qf = q.detach().float().requires_grad_(True)
  (_RBFRead.apply(qf, k.float(), v.float(), k.float().pow(2).sum(-1), c, 4) * w.float()).sum().backward()
  assert torch.allclose(qf.grad.double(), q.grad, rtol=1e-4, atol=1e-5)


def test_growing_layer_writes_minus_eta_token_gradient():
  """After one step, the growing layer holds one row per fact position: key = RMSNorm of the residual stream there,
  value = -η dℓ_i/ds_i with ℓ_i the token's own cross-entropy (the last layer's output at i reaches only logit i)."""
  cfg = BiosConfig(n_people=16, seq_len=64, batch_size=2, d_model=32, layers=2, heads=2, kv_heads=1, head_dim=16,
                   ffn_width=64, special='rbf', n_special=1, rbf_capacity=4096, rbf_eta=0.7, rbf_T=1.0)
  world = BioWorld(16, seed=0)
  model = GemmaBios(cfg, world.vocab_size)
  x, y, fact = world.train_batch(0, 16, 2, 64, torch.Generator().manual_seed(0))
  layer = model.growing_layers()[0]
  layer.loss_tokens = y.numel()
  captured = {}
  blk = model.blocks[-1]
  orig = blk.forward

  def spy(h, rope, f):
    captured['h'] = h
    return orig(h, rope, f)

  blk.forward = spy
  logits = model(x, fact)
  loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), y.reshape(-1))
  loss.backward()
  layer.commit()
  n = int(fact.sum())
  assert layer.size == n
  # recompute the stream entering the growing layer (after the last block's attention) and dℓ_i/ds_i directly
  h = captured['h'].detach()
  cos, sin = model.rope[1] if blk.is_global else model.rope[0]
  z = h + blk.post_attn_norm(blk.attn(blk.input_norm(h), cos, sin, blk.window))
  q = z[fact] * torch.rsqrt(z[fact].pow(2).mean(-1, keepdim=True) + 1e-6)
  assert torch.allclose(layer.keys[:n], q.detach(), atol=1e-5)
  s = torch.zeros(len(q), cfg.d_model, requires_grad=True)
  zz = z.detach().clone()
  zz[fact] = zz[fact] + s
  lg = model.final_norm(zz) @ model.embed.weight.T
  tok_loss = F.cross_entropy(lg[fact], y[fact], reduction='sum')     # Σ_i ℓ_i over fact positions
  (gs,) = torch.autograd.grad(tok_loss, s)
  assert torch.allclose(layer.values[:n], -0.7 * gs, rtol=1e-4, atol=1e-6)


def test_tiny_runs_learn_and_finetune():
  base = dict(n_people=32, seq_len=128, batch_size=8, d_model=32, layers=2, heads=2, kv_heads=1, head_dim=16,
              ffn_width=64, steps=60, eval_every=30, eval_bios=32, lr=3e-3)
  for kw in (dict(), dict(ffn='cos', cos_width=128), dict(special='rbf', rbf_capacity=200_000, rbf_eta=1.0),
             dict(special='geglu_fact')):
    r = run_bios(BiosConfig(**base, **kw))
    assert math.isfinite(r['final_attr_loss']) and r['log'][-1]['train_loss'] < r['log'][1]['train_loss']
  r = run_bios(BiosConfig(**base, finetune_people=8, finetune_steps=20, finetune_eval_every=10, special='rbf',
                          rbf_capacity=3000, rbf_finetune_rows=50_000, finetune_batch_size=4))
  assert 'new_attr_acc' in r['log'][-1] and r['old_acc_after_finetune'] is not None
  end_pre = next(e for e in r['log'] if e['step'] == 60)
  assert end_pre['rbf_rows'] == [3000]                              # ring buffer full during pre-training
  assert r['log'][-1]['rbf_rows'][0] > 3000 and r['log'][-1]['rbf_evicted_while_appending'] == [0]
  assert all(k in r['log'][0] for k in ('acc_' + a for a in ATTRS))


def test_heldout_frame_and_population_independent_of_finetuning_people():
  """Evaluation templates are one whole frame (5 openers) per individual and attribute, absent from that individual's
  training templates; the pre-training population is the same whatever the number of fine-tuning individuals."""
  w = BioWorld(40, seed=4, n_base=24)
  frames = w.split % 5
  assert (frames[:, :, 20:] == frames[:, :, 20:21]).all()
  assert not (frames[:, :, :20] == frames[:, :, 20:21]).any()
  v = BioWorld(24, seed=4)
  assert torch.equal(w.name_tok[:24], v.name_tok) and torch.equal(w.val_tok[:24], v.val_tok)
  assert torch.equal(w.split[:24], v.split)
