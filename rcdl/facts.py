"""Minimal synthetic-fact task for the article's (unpublished) biography experiment.

The article replaces the feed-forward layers of a small Gemma 3 model by growing RBF layers, wide random-cosine
("Cos Bochner") layers or GeGLU layers, trains on the synthetic biographies of Zucchet et al. (arXiv 2503.21676), and
reports (i) a loss plateau whose length grows about linearly with the number of individuals N, (ii) a much shorter
plateau for RBF and wide Cos-Bochner layers, (iii) a lower ceiling for GeGLU.

This module keeps only the fact-storage core of that task. There are N individuals and A attributes; fact (i, a) is a
value in {0, ..., K-1}, drawn uniformly at random. Each example is a sequence of L random distractor tokens with the
name of individual i (T name tokens) inserted at position p, followed by the token of attribute a. The model predicts
the value from the last position. Training is online: every step draws a fresh batch of (i, a) pairs, distractors and
name positions. Accuracy is measured on all N x A facts, or on a fixed random subset of them (`eval_facts`), with
fixed distractors. Before facts are learned, the best possible loss is ln K (uniform values), so the plateau is the
stretch of training where the loss stays near ln K.

Task difficulty. Zucchet et al. (arXiv 2503.21676) report a plateau whose length is nearly proportional to N, and trace
it to the formation of the attention circuits that move the name to the prediction position. Their task has three
features that make these circuits necessary: rare individuals (N = 4k-256k at batch 128), names of several tokens, and
names at varying positions inside long text. Measured here (GeGLU, width 512):
  * T = 1, L = 0, p = 0 (the two-token task): N = 1024 learns all 4,096 facts within 500 steps, no plateau;
  * T = 1, L = 8, p = 0: N = 1024 learns all facts by step 800, still no plateau.
The defaults keep the two-token task; the configurations in experiments/configs/facts.json set T, L and p explicitly.

FFN variants (all pre-LayerNorm, residual):
  geglu  W_o (GELU(W_g x) * (W_u x)), width w, all weights trained.
  cos    random Fourier features: W_o sqrt(2/w) cos(W_r x + b), W_r ~ N(0, 1/σ^2) and b ~ U(0, 2π) fixed, W_o trained;
         by Bochner's theorem the features approximate an RBF kernel of bandwidth σ as w grows.
  rbf    growing RBF layer (rcdl.layers.RCDLLayer, kind 'rbf'): one stored row per token per step.

Looping (weight-tied recurrence of the last FFN block): with `loops` = K > 1 the last block h <- h + a f(LN(h)) runs K
times with the same LayerNorm and FFN weights (a = `loop_alpha`); K = 1 is the plain model. `final_probe` adds, after
training, the accuracy with other numbers of passes at test time, the accuracy with Gaussian noise of relative size e
added to the state before the looped block (clean-up test), and the spectral radius and norm of each pass's Jacobian
d(h + a f(LN(h)))/dh at the evaluation states (stability of the iteration). Not for the growing RBF layer.
"""

import dataclasses as dc
import json
import math
import os
import time
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.layers import RCDLConfig, RCDLLayer


@dc.dataclass(frozen=True)
class FactsConfig:
  n_people: int = 1024     # N
  n_attrs: int = 4         # A
  n_values: int = 32       # K
  d: int = 128
  heads: int = 4
  layers: int = 2
  ffn: str = 'geglu'       # 'geglu' | 'cos' | 'rbf'
  width: int = 512         # FFN width w (geglu, cos)
  cos_sigma: float = 4.0   # σ of the random features (cos)
  rbf_layers: str = 'all'  # 'all' | 'last': which FFNs are growing RBF layers (others GeGLU)
  rbf_temperature: float = 1.0
  rbf_eta: float = 1.0     # value learning rate of the growing layer
  lr: float = 1e-3         # Adam, all trained parameters
  batch_size: int = 128
  steps: int = 5000
  eval_every: int = 250
  seed: int = 1
  finetune_people: int = 0  # M: after `steps`, train only on M new individuals (forgetting test; 0 = off)
  finetune_steps: int = 0
  n_filler: int = 0        # L: random distractor tokens (0 = the two-token task)
  filler_vocab: int = 64   # number of distinct distractor tokens
  name_tokens: int = 1     # T: tokens per name; T = 1: one token per individual; T >= 2: unique tuples of name parts
  name_vocab: int = 256    # T >= 2: distinct tokens per name part (each part has its own token range)
  random_name_pos: bool = False   # True: the name starts at a uniform position in [0, L]; False: at position 0
  eval_facts: int = 0      # 0: evaluate on all N x A facts; > 0: on a fixed random subset of this size
  loops: int = 1           # K: passes of the last FFN block with tied weights (1 = the plain model)
  loop_alpha: float = 1.0  # a: step size of each pass of the last block, h <- h + a f(LN(h))
  final_probe: bool = False   # after training: accuracy at other K, under state noise, and per-pass Jacobian spectra
  loop_tied: bool = True   # False (control): each of the K passes has its own LayerNorm and FFN (new random features
                           # for cos); K_test must equal K


class GeGLU(nn.Module):
  def __init__(self, d, w):
    super().__init__()
    self.g, self.u, self.o = nn.Linear(d, w), nn.Linear(d, w), nn.Linear(w, d)

  def forward(self, x):
    return self.o(F.gelu(self.g(x)) * self.u(x))


class CosFeatures(nn.Module):
  def __init__(self, d, w, sigma, gen):
    super().__init__()
    self.register_buffer('W', torch.randn(w, d, generator=gen) / sigma)
    self.register_buffer('b', torch.rand(w, generator=gen) * 2 * math.pi)
    self.o = nn.Linear(w, d)
    nn.init.zeros_(self.o.weight)
    self.scale = math.sqrt(2.0 / w)

  def forward(self, x):
    return self.o(self.scale * torch.cos(x @ self.W.T + self.b))


def name_region(cfg: FactsConfig) -> int:
  """Number of name token ids: one per individual (T = 1) or T ranges of `name_vocab` part tokens (T >= 2)."""
  return cfg.n_people + cfg.finetune_people if cfg.name_tokens == 1 else cfg.name_tokens * cfg.name_vocab


class FactsModel(nn.Module):
  def __init__(self, cfg: FactsConfig, gen: torch.Generator):
    super().__init__()
    # tokens: names [0, R), attributes [R, R + A), distractors [R + A, R + A + F), with R = name_region(cfg)
    V = name_region(cfg) + cfg.n_attrs + cfg.filler_vocab
    S = cfg.name_tokens + cfg.n_filler + 1                 # sequence length
    if cfg.ffn == 'rbf' and cfg.loops > 1:
      raise ValueError('looping is not defined for the growing RBF layer (it stores one row per pass)')
    self.cfg = cfg
    self.emb = nn.Embedding(V, cfg.d)
    self.pos = nn.Parameter(torch.randn(S, cfg.d, generator=gen) * 0.02)
    self.ln1 = nn.ModuleList([nn.LayerNorm(cfg.d) for _ in range(cfg.layers)])
    self.attn = nn.ModuleList([nn.MultiheadAttention(cfg.d, cfg.heads, batch_first=True) for _ in range(cfg.layers)])
    self.ln2 = nn.ModuleList([nn.LayerNorm(cfg.d) for _ in range(cfg.layers)])
    ffns = []
    for i in range(cfg.layers):
      use_rbf = cfg.ffn == 'rbf' and (cfg.rbf_layers == 'all' or i == cfg.layers - 1)
      if use_rbf:
        rows_per_step = cfg.batch_size * (1 if i == cfg.layers - 1 else S)   # the last FFN sees the last token only
        rc = RCDLConfig(kernel_type='rbf', hidden_dim=cfg.d, temperature=cfg.rbf_temperature, eta=cfg.rbf_eta,
                        db_size=rows_per_step * (cfg.steps + cfg.finetune_steps + 1), batch_size=cfg.batch_size,
                        num_classes=cfg.d)
        ffns.append(RCDLLayer(rc, cfg.d, cfg.d, i, gen))
      elif cfg.ffn == 'cos':
        ffns.append(CosFeatures(cfg.d, cfg.width, cfg.cos_sigma, gen))
      else:
        ffns.append(GeGLU(cfg.d, cfg.width))
    self.ffn = nn.ModuleList(ffns)
    self.ln_f = nn.LayerNorm(cfg.d)
    self.head = nn.Linear(cfg.d, cfg.n_values)
    self.register_buffer('mask', torch.triu(torch.ones(S, S, dtype=torch.bool), 1))
    if cfg.loops > 1 and not cfg.loop_tied:          # built last, with their own generator, so the tied model and the
      g2 = torch.Generator().manual_seed(cfg.seed + 7919)   # data drawn after it from `gen` stay the same
      self.extra_ln = nn.ModuleList([nn.LayerNorm(cfg.d) for _ in range(cfg.loops - 1)])
      self.extra_ffn = nn.ModuleList([CosFeatures(cfg.d, cfg.width, cfg.cos_sigma, g2) if cfg.ffn == 'cos'
                                      else GeGLU(cfg.d, cfg.width) for _ in range(cfg.loops - 1)])

  def block(self, i, h):
    """The FFN branch f(LN(h)) of layer i."""
    f = self.ln2[i](h)
    return self.ffn[i](f.reshape(-1, f.shape[-1])).view_as(f) if isinstance(self.ffn[i], RCDLLayer) else self.ffn[i](f)

  def last_state(self, tokens):
    """(B, 1, d): the state at the last position before the last FFN block."""
    h = self.emb(tokens) + self.pos
    for i in range(self.cfg.layers):
      a = self.ln1[i](h)
      h = h + self.attn[i](a, a, a, attn_mask=self.mask, need_weights=False)[0]
      if i == self.cfg.layers - 1:                 # only the last position reaches the head, so the last FFN runs
        return h[:, -1:]                           # there only (same output; the growing layer stores 1 row/example)
      h = h + self.block(i, h)

  def forward(self, tokens, loops=None, noise=0.0, noise_gen=None):
    """tokens: (B, T + L + 1): distractors with the name, then the attribute. loops: passes of the last block (default
    cfg.loops); noise: Gaussian noise of norm noise * |h| added to the state before the last block."""
    h = self.last_state(tokens)
    if noise:
      z = torch.randn(h.shape, generator=noise_gen).to(h.device)
      h = h + noise * h.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True)
    return self.head(self.ln_f(self.loop(h, loops)[:, -1]))

  def pass_block(self, k, h):
    """The FFN branch of pass k (0-based) of the last layer: the shared block, or with untied passes block k."""
    if k == 0 or self.cfg.loop_tied:
      return self.block(self.cfg.layers - 1, h)
    return self.extra_ffn[k - 1](self.extra_ln[k - 1](h))

  def loop(self, h, loops=None):
    K = loops or self.cfg.loops
    if not self.cfg.loop_tied and K != self.cfg.loops:
      raise ValueError('a model with untied passes runs exactly cfg.loops passes')
    for k in range(K):
      h = h + self.cfg.loop_alpha * self.pass_block(k, h)
    return h

  def rcdl_layers(self) -> Dict[str, RCDLLayer]:
    return {f'ffn_{i}': m for i, m in enumerate(self.ffn) if isinstance(m, RCDLLayer)}


@torch.no_grad()
def loop_probe(model: FactsModel, tokens, targets, n_jac: int = 256, seed: int = 0) -> Dict[str, Any]:
  """Accuracy with K_test = 1, 2, 4, 8 passes of the last block; accuracy with state noise e = 0.1, 0.25, 0.5 at the
  trained K (one fixed noise draw per e); and, on n_jac evaluation states, per pass k: the median spectral radius
  (largest |eigenvalue|) and spectral norm of J_k = d(h + a f(LN(h)))/dh at the pass's input state, the median of
  |J_k - I|_2, and the median relative step |a f| / |h|."""
  from torch.func import jacrev, vmap
  model.eval()
  acc = lambda **kw: float((torch.cat([model(tokens[i:i + 2048], **kw) for i in range(0, len(tokens), 2048)])
                            .argmax(-1) == targets).float().mean())
  out = {'acc_by_K': {str(k): acc(loops=k) for k in ((1, 2, 4, 8) if model.cfg.loop_tied else (model.cfg.loops,))}}
  out['acc_by_noise'] = {str(e): acc(noise=e, noise_gen=torch.Generator().manual_seed(seed + int(e * 1000)))
                         for e in (0.1, 0.25, 0.5)}
  a = model.cfg.loop_alpha
  h = model.last_state(tokens[:n_jac])[:, 0]                       # (n, d)
  eye = torch.eye(h.shape[-1], device=h.device)
  per_pass = []
  for k in range(model.cfg.loops):
    step = lambda v, k=k: v + a * model.pass_block(k, v[None, None])[0, 0]
    with torch.enable_grad():
      J = vmap(jacrev(step))(h)                                    # (n, d, d)
    rad = torch.linalg.eigvals(J.double()).abs().amax(-1)
    nrm = torch.linalg.matrix_norm(J.double(), ord=2)
    dev = torch.linalg.matrix_norm((J - eye).double(), ord=2)
    nxt = vmap(step)(h)
    per_pass.append({'pass': k + 1, 'spectral_radius_median': float(rad.median()),
                     'spectral_radius_max': float(rad.max()), 'spectral_norm_median': float(nrm.median()),
                     'jac_minus_I_norm_median': float(dev.median()),
                     'rel_step_median': float(((nxt - h).norm(dim=-1) / h.norm(dim=-1)).median())})
    h = nxt
  out['jacobian_by_pass'] = per_pass
  model.train()
  return out


def run_facts(cfg: FactsConfig, out_dir: Optional[str] = None, device: str = 'cpu') -> Dict[str, Any]:
  gen = torch.Generator().manual_seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  N, M, A = cfg.n_people, cfg.finetune_people, cfg.n_attrs
  facts = torch.randint(0, cfg.n_values, (N + M, A), generator=gen).to(device)
  model = FactsModel(cfg, gen).to(device)
  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.Adam(params, lr=cfg.lr)

  R, T, L = name_region(cfg), cfg.name_tokens, cfg.n_filler
  if T == 1:
    names = torch.arange(N + M)[:, None]
  else:                                       # unique T-tuples of name parts; part j uses token range [j V_n, (j+1) V_n)
    if (N + M) > cfg.name_vocab ** T // 2:
      raise ValueError('name_vocab ** name_tokens must be at least twice the number of individuals')
    codes = torch.randperm(cfg.name_vocab ** T, generator=gen)[:N + M]
    names = torch.stack([(codes // cfg.name_vocab ** j) % cfg.name_vocab + j * cfg.name_vocab for j in range(T)], 1)
  names = names.to(device)

  def sequences(people, attrs, g):            # distractors with the name inserted, then the attribute token
    B = len(people)
    seq = torch.randint(R + A, R + A + cfg.filler_vocab, (B, T + L), generator=g)
    start = torch.randint(0, L + 1, (B,), generator=g) if cfg.random_name_pos else torch.zeros(B, dtype=torch.long)
    seq, start = seq.to(device), start.to(device)
    rows = torch.arange(B, device=device)
    for j in range(T):
      seq[rows, start + j] = names[people, j]
    return torch.cat([seq, (R + attrs)[:, None]], 1)

  eval_gen = torch.Generator().manual_seed(10_000 + cfg.seed)   # fixed distractors and positions for evaluation

  def fact_set(lo, hi):                       # facts of people [lo, hi) (or a fixed subset): sequences and values
    people = torch.arange(lo, hi, device=device).repeat_interleave(A)
    attrs = torch.arange(A, device=device).repeat(hi - lo)
    if cfg.eval_facts and len(people) > cfg.eval_facts:
      pick = torch.randperm(len(people), generator=eval_gen)[:cfg.eval_facts].to(device)
      people, attrs = people[pick], attrs[pick]
    return sequences(people, attrs, eval_gen), facts[people, attrs]

  old_set, new_set = fact_set(0, N), (fact_set(N, N + M) if M else None)

  @torch.no_grad()
  def evaluate(tokens, targets):
    model.eval()
    loss = correct = 0.0
    rows = max([layer.size for layer in model.rcdl_layers().values()] or [0])
    chunk = max(32, min(4096, int(1e8 // max(1, rows))))   # keeps the chunk x rows distance matrix near 400 MB
    for i in range(0, len(tokens), chunk):
      logits = model(tokens[i:i + chunk])
      loss += F.cross_entropy(logits, targets[i:i + chunk], reduction='sum').item()
      correct += (logits.argmax(-1) == targets[i:i + chunk]).sum().item()
    model.train()
    return loss / len(tokens), correct / len(tokens)

  log, t0 = [], time.time()
  total = cfg.steps + (cfg.finetune_steps if M else 0)
  for step in range(total + 1):
    if step % cfg.eval_every == 0 or step == cfg.steps:
      rec = {'step': step, 'sec': time.time() - t0}
      rec['loss'], rec['acc'] = evaluate(*old_set)
      if M:
        rec['loss_new'], rec['acc_new'] = evaluate(*new_set)
      log.append(rec)
      print(f'[facts:{cfg.ffn}:N={N}] step={step} loss={rec["loss"]:.4f} (ln K={math.log(cfg.n_values):.3f}) '
            f'acc={rec["acc"]:.4f}' + (f' new_acc={rec["acc_new"]:.4f}' if M else '') + f' ({time.time() - t0:.0f}s)',
            flush=True)
    if step == total:
      break
    lo, hi = (0, N) if step < cfg.steps else (N, N + M)   # phase 2: only the M new individuals
    i = torch.randint(lo, hi, (cfg.batch_size,), generator=gen).to(device)
    a = torch.randint(0, A, (cfg.batch_size,), generator=gen).to(device)
    logits = model(sequences(i, a, gen))
    loss = F.cross_entropy(logits, facts[i, a])
    opt.zero_grad()
    loss.backward()
    opt.step()
    for layer in model.rcdl_layers().values():
      layer.commit()
  # Plateau end: first evaluation at which the loss on the first population is 10% below ln K.
  thresh = 0.9 * math.log(cfg.n_values)
  plateau_end = next((r['step'] for r in log if r['step'] <= cfg.steps and r['loss'] < thresh), None)
  phase1 = [r for r in log if r['step'] <= cfg.steps]
  result = {'cfg': dc.asdict(cfg), 'log': log, 'final_acc': phase1[-1]['acc'], 'final_loss': phase1[-1]['loss'],
            'plateau_end_step': plateau_end, 'params': sum(p.numel() for p in params)}
  if cfg.final_probe:
    result['loop_probe'] = loop_probe(model, *old_set, seed=cfg.seed)
  if M:
    phase2 = [r for r in log if r['step'] >= cfg.steps]
    result |= {'old_acc_min_during_finetune': min(r['acc'] for r in phase2), 'old_acc_after_finetune': log[-1]['acc'],
               'new_acc_after_finetune': log[-1]['acc_new']}
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
      json.dump(result, f, indent=1)
  return result
