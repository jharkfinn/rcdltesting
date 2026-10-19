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
times with the same LayerNorm and FFN weights (a = `loop_alpha`); K = 1 is the plain model. `deep_supervision` trains
the head read after every pass; `train_probe` records per-pass measures at every evaluation (pass_probe). `final_probe`
adds, after training, the accuracy with other numbers of passes at test time, the accuracy with Gaussian noise of
relative size e added to the state before the looped block (clean-up test), and the spectral radius and norm of each
pass's Jacobian d(h + a f(LN(h)))/dh at the evaluation states (stability of the iteration). Not for the growing RBF
layer.
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
  ffn: str = 'geglu'       # 'geglu' | 'cos' | 'rbf' | 'relu' (trained 2-layer ReLU MLP) | 'rfrelu' (ReLU random
                           # features: frozen first layer W ~ N(0, 1/d), zero-initialised trained readout, like cos)
  width: int = 512         # FFN width w (geglu, cos)
  cos_sigma: float = 4.0   # σ of the random features (cos)
  cos_train: bool = False  # cos: also train W and b (a trained cosine MLP with the kernel-scaled initialisation)
  ffn_zero_init: bool = False  # geglu / relu: zero-initialise the output weight, as the cos and rfrelu readouts are (at
                               # initialisation the block then writes only its output bias)
  loop_mode: str = 'residual'  # 'residual': h <- h + a f(LN(h)); 'infiloop': the next state is a content-weighted,
                               # decayed average of all candidates z_t = h_t-1 + a f(LN(h_t-1)) (Li et al. 2026, arXiv
                               # 2610.11570): h_t = N_t / D_t, N_t = b N_t-1 + e_t z_t, D_t = b D_t-1 + e_t, e_t = exp(clamp(q .
                               # RMSNorm(z_t), -8, 8)), z_0 = h_0; q (zero-initialised) and b = exp(-softplus(r)) (b = 0.55
                               # at initialisation) are learned
  lr_schedule: str = 'constant'  # 'constant' | 'cosine' (to 0 at the end of the first phase, `steps`)
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
  deep_supervision: bool = False  # K > 1: the loss is the mean over passes k = 1 .. K of the loss of the head read
                                  # at the state after pass k (with infiloop the aggregated state)
  train_probe: bool = False  # at every evaluation, per-pass measures of the looped block on 1,024 fixed evaluation
                             # facts (pass_probe); no random draws, so training is unchanged


class GeGLU(nn.Module):
  def __init__(self, d, w):
    super().__init__()
    self.g, self.u, self.o = nn.Linear(d, w), nn.Linear(d, w), nn.Linear(w, d)

  def hidden(self, x):
    return F.gelu(self.g(x)) * self.u(x)

  def forward(self, x):
    return self.o(self.hidden(x))


class CosFeatures(nn.Module):
  def __init__(self, d, w, sigma, gen, train_features=False):
    super().__init__()
    W, b = torch.randn(w, d, generator=gen) / sigma, torch.rand(w, generator=gen) * 2 * math.pi
    if train_features:
      self.W, self.b = nn.Parameter(W), nn.Parameter(b)
    else:
      self.register_buffer('W', W)
      self.register_buffer('b', b)
    self.o = nn.Linear(w, d)
    nn.init.zeros_(self.o.weight)
    self.scale = math.sqrt(2.0 / w)

  def hidden(self, x):
    return self.scale * torch.cos(x @ self.W.T + self.b)

  def forward(self, x):
    return self.o(self.hidden(x))


class ReLUMLP(nn.Module):
  """Trained 2-layer MLP, o(relu(i x)) (PyTorch default initialisation, as GeGLU)."""

  def __init__(self, d, w):
    super().__init__()
    self.i, self.o = nn.Linear(d, w), nn.Linear(w, d)

  def hidden(self, x):
    return F.relu(self.i(x))

  def forward(self, x):
    return self.o(self.hidden(x))


class ReLUFeatures(nn.Module):
  """ReLU random features: o(sqrt(2/w) relu(W x)) with W ~ N(0, 1/d) frozen and the readout o trained from zero (the
  frozen-basis counterpart of CosFeatures with another nonlinearity; its kernel is the arc-cosine kernel)."""

  def __init__(self, d, w, gen):
    super().__init__()
    self.register_buffer('W', torch.randn(w, d, generator=gen) / math.sqrt(d))
    self.o = nn.Linear(w, d)
    nn.init.zeros_(self.o.weight)
    self.scale = math.sqrt(2.0 / w)

  def hidden(self, x):
    return self.scale * F.relu(x @ self.W.T)

  def forward(self, x):
    return self.o(self.hidden(x))


def infiloop(h, K, candidate, q, r, trace=None):
  """Loop-native attention residual (Li et al. 2026, arXiv 2610.11570, eq. 2-4): z_0 = h, z_t = candidate(t - 1, h_t-1),
  h_t = N_t / D_t with N_t = b N_t-1 + e_t z_t, D_t = b D_t-1 + e_t, e_t = exp(clamp(q . RMSNorm(z_t), -8, 8)) and
  b = exp(-softplus(r)). Scores and sums in fp32. trace (a dict, optional) receives 'states' (h_1 .. h_K) and 'logits'
  (the clamped scores log e_0 .. log e_K)."""
  dtype = h.dtype
  b = torch.exp(-F.softplus(r.float()))
  logit = lambda z: (z.float() * torch.rsqrt(z.float().pow(2).mean(-1, keepdim=True) + 1e-6) @ q.float()).clamp(-8, 8)
  s = logit(h)
  e = torch.exp(s)
  N, D = e[..., None] * h.float(), e
  if trace is not None:
    trace['states'], trace['logits'] = [], [s]
  for k in range(K):
    z = candidate(k, (N / D[..., None]).to(dtype))
    s = logit(z)
    e = torch.exp(s)
    N, D = b * N + e[..., None] * z.float(), b * D + e
    if trace is not None:
      trace['states'].append((N / D[..., None]).to(dtype))
      trace['logits'].append(s)
  return (N / D[..., None]).to(dtype)


def make_ffn(cfg, gen):
  if cfg.ffn == 'cos':
    return CosFeatures(cfg.d, cfg.width, cfg.cos_sigma, gen, cfg.cos_train)
  if cfg.ffn == 'rfrelu':
    return ReLUFeatures(cfg.d, cfg.width, gen)
  if cfg.ffn == 'relu':
    m = ReLUMLP(cfg.d, cfg.width)
  elif cfg.ffn in ('geglu', 'rbf'):                # with rbf_layers = 'last', the other FFNs are GeGLU
    m = GeGLU(cfg.d, cfg.width)
  else:
    raise ValueError(f'unknown ffn {cfg.ffn!r}')
  if cfg.ffn_zero_init:                            # after the default initialisation, so the random draws do not change
    nn.init.zeros_(m.o.weight)
  return m


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
      else:
        ffns.append(make_ffn(cfg, gen))
    self.ffn = nn.ModuleList(ffns)
    self.ln_f = nn.LayerNorm(cfg.d)
    self.head = nn.Linear(cfg.d, cfg.n_values)
    self.register_buffer('mask', torch.triu(torch.ones(S, S, dtype=torch.bool), 1))
    if cfg.loop_mode not in ('residual', 'infiloop'):
      raise ValueError(f'unknown loop_mode {cfg.loop_mode!r}')
    if cfg.loop_mode == 'infiloop':                  # parameters only, no random draws: the rest of the model is unchanged
      self.loop_q = nn.Parameter(torch.zeros(cfg.d))
      self.loop_r = nn.Parameter(torch.tensor(math.log(math.expm1(-math.log(0.55)))))
    if cfg.loops > 1 and not cfg.loop_tied:          # built last, with their own generator, so the tied model and the
      g2 = torch.Generator().manual_seed(cfg.seed + 7919)   # data drawn after it from `gen` stay the same
      self.extra_ln = nn.ModuleList([nn.LayerNorm(cfg.d) for _ in range(cfg.loops - 1)])
      self.extra_ffn = nn.ModuleList([make_ffn(cfg, g2) for _ in range(cfg.loops - 1)])

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

  def forward(self, tokens, loops=None, noise=0.0, noise_gen=None, alpha=None, all_passes=False):
    """tokens: (B, T + L + 1): distractors with the name, then the attribute. loops: passes of the last block (default
    cfg.loops); noise: Gaussian noise of norm noise * |h| added to the state before the last block. all_passes: return
    (K, B, n_values), the head read at the state after each pass (deep supervision), instead of (B, n_values)."""
    h = self.last_state(tokens)
    if noise:
      z = torch.randn(h.shape, generator=noise_gen).to(h.device)
      h = h + noise * h.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True)
    if all_passes:
      trace = {}
      self.loop(h, loops, alpha, trace)
      return torch.stack([self.head(self.ln_f(s[:, -1])) for s in trace['states']])
    return self.head(self.ln_f(self.loop(h, loops, alpha)[:, -1]))

  def pass_block(self, k, h):
    """The FFN branch of pass k (0-based) of the last layer: the shared block, or with untied passes block k."""
    if k == 0 or self.cfg.loop_tied:
      return self.block(self.cfg.layers - 1, h)
    return self.extra_ffn[k - 1](self.extra_ln[k - 1](h))

  def loop(self, h, loops=None, alpha=None, trace=None):
    """K passes of the last block from the state h. trace (a dict, optional) receives 'states', the state after each
    pass (with infiloop the aggregated state, plus the gate 'logits')."""
    K, a = loops or self.cfg.loops, self.cfg.loop_alpha if alpha is None else alpha
    if not self.cfg.loop_tied and K != self.cfg.loops:
      raise ValueError('a model with untied passes runs exactly cfg.loops passes')
    if self.cfg.loop_mode == 'infiloop':
      return infiloop(h, K, lambda k, v: v + a * self.pass_block(k, v), self.loop_q, self.loop_r, trace)
    if trace is not None:
      trace['states'] = []
    for k in range(K):
      h = h + a * self.pass_block(k, h)
      if trace is not None:
        trace['states'].append(h)
    return h

  def pass_hidden(self, k, h):
    """Hidden activations of the FFN of pass k (0-based) at the state h: for cos the random features, whose inner
    products approximate the kernel."""
    if k == 0 or self.cfg.loop_tied:
      i = self.cfg.layers - 1
      return self.ffn[i].hidden(self.ln2[i](h))
    return self.extra_ffn[k - 1].hidden(self.extra_ln[k - 1](h))

  def pass_input(self, k, h):
    """The normalized input of the FFN of pass k (0-based) at the state h."""
    return self.ln2[self.cfg.layers - 1](h) if k == 0 or self.cfg.loop_tied else self.extra_ln[k - 1](h)

  def rcdl_layers(self) -> Dict[str, RCDLLayer]:
    return {f'ffn_{i}': m for i, m in enumerate(self.ffn) if isinstance(m, RCDLLayer)}


@torch.no_grad()
def tangent_spectrum(J, h, mean_free: bool = True):
  """|eigenvalues| of the step Jacobian J (n, d, d) on the tangent space of the pre-norm at the states h (n, d).

  The pre-norm ignores the scale of h (and, for LayerNorm, a shift along the ones vector), so J h = h (and J 1 = 1): these
  directions are eigenvectors with eigenvalue 1 by construction. With P the projector onto their orthogonal complement,
  spec(P J P) = {0}^m U spec(J restricted to the complement), m = 2 (LayerNorm) or 1 (RMSNorm); the m smallest values
  are dropped. Returns (n, d - m), sorted ascending, on the CPU (small dense eigenproblems are much faster there than
  on the GPU path, which also leans on the CPU)."""
  J, h = J.detach().double().cpu(), h.detach().double().cpu()
  n, d = h.shape
  v = h - h.mean(-1, keepdim=True) if mean_free else h
  U = [v / v.norm(dim=-1, keepdim=True)]
  if mean_free:
    U.append(torch.full_like(h, d ** -0.5))
  U = torch.stack(U, -1)                                           # (n, d, m), orthonormal
  P = torch.eye(d, dtype=torch.float64) - U @ U.transpose(1, 2)
  ev = torch.linalg.eigvals(P @ J @ P).abs().sort(-1).values
  return ev[:, U.shape[-1]:]


def pass_probe(model: FactsModel, tokens, targets) -> Dict[str, Any]:
  """Per-pass measures of the looped last block at the current weights (run at every evaluation with train_probe).

  h_0 is the state before the looped block and h_k the state after pass k = 1 .. K (with infiloop the aggregated
  state); x_k = LN(h_k-1) is the input of the FFN at pass k and phi(x) its hidden activations (for cos the random
  features, whose inner products approximate the kernel). Medians are over the examples.
    loss, acc (k = 0 .. K): the head read at h_k (k = 0: before any pass of the last block);
    rel_step (passes 1 .. K): median |h_k - h_k-1| / |h_k-1|;
    descent_cos (passes 1 .. K): median cos(h_k - h_k-1, -g_k-1), with g_k-1 the gradient of the loss of the head read
      at h_k-1 with respect to h_k-1 (positive: the pass moves the state down the loss read before it);
    input_shift (passes 2 .. K): median |x_k - x_k-1| (for cos the kernel between the two inputs is
      exp(-shift^2 / (2 sigma^2)));
    overlap_prev, overlap_first (passes 2 .. K, tied passes only): median cos(phi(x_k), phi(x_k-1)) and
      cos(phi(x_k), phi(x_1));
    gate (infiloop): |q|, b, and per candidate t = 0 .. K the median and the SD over examples of its weight
      b^(K-t) e_t / sum_s b^(K-s) e_s in h_K, and the SD over examples of its clamped score log e_t (0 while q = 0)."""
  model.eval()
  K = model.cfg.loops
  med = lambda v: float(v.float().median())
  with torch.no_grad():
    h0 = model.last_state(tokens)
    trace = {}
    model.loop(h0, K, None, trace)
  states = [h[:, -1] for h in [h0] + trace['states']]           # (B, d) each
  out, grads = {'loss': [], 'acc': [], 'rel_step': [], 'descent_cos': []}, []
  for h in states:
    v = h.detach().requires_grad_(True)
    with torch.enable_grad():
      logits = model.head(model.ln_f(v))
      loss = F.cross_entropy(logits, targets, reduction='sum')
      grads.append(torch.autograd.grad(loss, v)[0])
    out['loss'].append(float(loss.detach()) / len(targets))
    out['acc'].append(float((logits.argmax(-1) == targets).float().mean()))
  for k in range(1, K + 1):
    u = states[k] - states[k - 1]
    out['rel_step'].append(med(u.norm(dim=-1) / states[k - 1].norm(dim=-1)))
    out['descent_cos'].append(med(F.cosine_similarity(u, -grads[k - 1], dim=-1)))
  with torch.no_grad():
    if K > 1:
      x = [model.pass_input(k, states[k]) for k in range(K)]
      out['input_shift'] = [med((x[k] - x[k - 1]).norm(dim=-1)) for k in range(1, K)]
      if model.cfg.loop_tied:
        phi = [model.pass_hidden(k, states[k]) for k in range(K)]
        out['overlap_prev'] = [med(F.cosine_similarity(phi[k], phi[k - 1], dim=-1)) for k in range(1, K)]
        out['overlap_first'] = [med(F.cosine_similarity(phi[k], phi[0], dim=-1)) for k in range(1, K)]
    if model.cfg.loop_mode == 'infiloop':
      b = torch.exp(-F.softplus(model.loop_r.float()))
      lg = torch.stack(trace['logits'], -1)[:, -1]                  # (B, K + 1): log e_0 .. log e_K
      w = torch.softmax(lg + torch.arange(K, -1, -1, device=lg.device) * torch.log(b), -1)
      out['gate'] = {'q_norm': float(model.loop_q.norm()), 'beta': float(b),
                     'weight_median': [med(w[:, t]) for t in range(K + 1)],
                     'weight_sd': [float(w[:, t].std()) for t in range(K + 1)],
                     'logit_sd': [float(lg[:, t].std()) for t in range(K + 1)]}
  model.train()
  return out


def loop_probe(model: FactsModel, tokens, targets, n_jac: int = 256, seed: int = 0, k_max: int = 16) -> Dict[str, Any]:
  """End-of-run measures of the looped last block (h <- h + a f(LN(h)), K passes).

  acc_by_K: accuracy with K_test = 1, 2, 4, 8, 16 passes (tied only). acc_by_alpha: accuracy at the trained K with test
  step size a_test (tied only). trajectory (tied only): per pass k = 1 .. k_max, the accuracy, the share of predictions
  that change from pass k - 1, and the medians of |h_k| / |h_0|, |h_k - h_k-1| / |h_k-1| and the cosine between
  consecutive steps. Noise: a push n of norm e |h_0| in a random direction (one fixed draw per e); acc_by_noise runs the
  passes from h_0 + n (as before), acc_by_noise_end adds the same n after the clean passes (dilution control: a gain of
  propagated over end-added noise means the passes remove part of the push). jacobian_by_pass, on n_jac states at the
  trained K: median spectral radius, spectral norm, |J - I|_2 and relative step (as before; the radius is at least 1 by
  construction, see tangent_spectrum), and the tangent-space radius (median, max) and the share of tangent eigenvalues
  below 1 in modulus (contracting directions). jacobian_noisy: the same tangent measures for pass 1 at h_0 + n, e = 0.25."""
  from torch.func import jacrev, vmap
  model.eval()
  tied, K, a = model.cfg.loop_tied, model.cfg.loops, model.cfg.loop_alpha
  chunks = [(tokens[i:i + 2048], targets[i:i + 2048]) for i in range(0, len(tokens), 2048)]
  logits = lambda h: model.head(model.ln_f(h[:, -1]))
  acc = lambda **kw: float((torch.cat([model(x, **kw) for x, _ in chunks]).argmax(-1) == targets).float().mean())
  out = {'acc_by_K': {str(k): acc(loops=k) for k in ((1, 2, 4, 8, 16) if tied else (K,))}}
  if tied:
    out['acc_by_alpha'] = {str(t): acc(alpha=t) for t in (0.5, 0.75, 1.25, 1.5, 2.0)}
  if model.cfg.loop_mode == 'infiloop':          # gate off: with q = 0 every candidate scores 1 (the convex-step loop)
    q = model.loop_q.data.clone()
    model.loop_q.data.zero_()
    out['acc_by_K_q0'] = {str(k): acc(loops=k) for k in (1, 2, 4, 8, 16)}
    model.loop_q.data.copy_(q)
    out['gate'] = {'q_norm': float(q.norm()), 'beta': float(torch.exp(-F.softplus(model.loop_r.detach().float())))}
  out['acc_by_noise'], out['acc_by_noise_end'] = {}, {}
  with torch.no_grad():
    for e in (0.1, 0.25, 0.5, 1.0):
      g, hit_p, hit_e = torch.Generator().manual_seed(seed + int(e * 1000)), 0, 0
      for x, y in chunks:                                          # same draws as model(x, noise=e, noise_gen=g)
        h = model.last_state(x)
        z = torch.randn(h.shape, generator=g).to(h.device)
        n = e * h.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True)
        hit_p += int((logits(model.loop(h + n)).argmax(-1) == y).sum())
        hit_e += int((logits(model.loop(h) + n).argmax(-1) == y).sum())
      out['acc_by_noise'][str(e)], out['acc_by_noise_end'][str(e)] = hit_p / len(tokens), hit_e / len(tokens)
    if tied:
      traj, rows = [], [model.last_state(x) for x, _ in chunks]
      h0 = torch.cat(rows)[:, 0]
      h, prev_pred, prev_step = h0, None, None
      for k in range(1, k_max + 1):
        if model.cfg.loop_mode == 'infiloop':          # the state depends on the whole history: rerun from h_0
          nxt = model.loop(h0[:, None], k)[:, 0]
          step = nxt - h
        else:
          step = a * model.block(model.cfg.layers - 1, h[:, None])[:, 0]
          nxt = h + step
        pred = model.head(model.ln_f(nxt)).argmax(-1)
        rec = {'pass': k, 'acc': float((pred == targets).float().mean()),
               'norm_ratio_median': float((nxt.norm(dim=-1) / h0.norm(dim=-1)).median()),
               'rel_step_median': float((step.norm(dim=-1) / h.norm(dim=-1)).median())}
        if prev_pred is not None:
          rec['flips'] = float((pred != prev_pred).float().mean())
          rec['step_cos_median'] = float(F.cosine_similarity(step, prev_step, dim=-1).median())
        traj.append(rec)
        h, prev_pred, prev_step = nxt, pred, step
      out['trajectory'] = traj
  with torch.no_grad():
    h = model.last_state(tokens[:n_jac])[:, 0]                     # (n, d)
  eye = torch.eye(h.shape[-1], device=h.device)
  per_pass = []
  for k in range(K if model.cfg.loop_mode == 'residual' else 0):  # with infiloop a pass depends on the history: skipped
    step = lambda v, k=k: v + a * model.pass_block(k, v[None, None])[0, 0]
    with torch.enable_grad():
      J = vmap(jacrev(step))(h).detach()                           # (n, d, d)
    Jc = J.double().cpu()                                          # dense eigenproblems on the CPU (see tangent_spectrum)
    rad = torch.linalg.eigvals(Jc).abs().amax(-1)
    nrm = torch.linalg.matrix_norm(Jc, ord=2)
    dev = torch.linalg.matrix_norm(Jc - eye.double().cpu(), ord=2)
    tan = tangent_spectrum(J, h)
    with torch.no_grad():
      nxt = vmap(step)(h)
    per_pass.append({'pass': k + 1, 'spectral_radius_median': float(rad.median()),
                     'spectral_radius_max': float(rad.max()), 'spectral_norm_median': float(nrm.median()),
                     'jac_minus_I_norm_median': float(dev.median()),
                     'rel_step_median': float(((nxt - h).norm(dim=-1) / h.norm(dim=-1)).median()),
                     'tangent_radius_median': float(tan[:, -1].median()), 'tangent_radius_max': float(tan[:, -1].max()),
                     'tangent_contracting_share_median': float((tan < 1).double().mean(-1).median())})
    if k == 0:
      z = torch.randn(h.shape, generator=torch.Generator().manual_seed(seed + 7)).to(h.device)
      hn = h + 0.25 * h.norm(dim=-1, keepdim=True) * z / z.norm(dim=-1, keepdim=True)
      with torch.enable_grad():
        Jn = vmap(jacrev(step))(hn).detach()
      tn = tangent_spectrum(Jn, hn)
      out['jacobian_noisy'] = {'noise': 0.25, 'pass': 1, 'tangent_radius_median': float(tn[:, -1].median()),
                               'tangent_contracting_share_median': float((tn < 1).double().mean(-1).median()),
                               'spectral_norm_median': float(torch.linalg.matrix_norm(Jn.double().cpu(), ord=2).median())}
    h = nxt
  out['jacobian_by_pass'] = per_pass
  out['probe_version'] = 2
  model.train()
  return out


def run_facts(cfg: FactsConfig, out_dir: Optional[str] = None, device: str = 'cpu',
              model_path: Optional[str] = None) -> Dict[str, Any]:
  """model_path: also save the trained model (cfg and state_dict) there as a chunked .py checkpoint (rcdl/ckpt.py)."""
  gen = torch.Generator().manual_seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  N, M, A = cfg.n_people, cfg.finetune_people, cfg.n_attrs
  facts = torch.randint(0, cfg.n_values, (N + M, A), generator=gen).to(device)
  model = FactsModel(cfg, gen).to(device)
  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.Adam(params, lr=cfg.lr)
  if cfg.lr_schedule not in ('constant', 'cosine'):
    raise ValueError(f'unknown lr_schedule {cfg.lr_schedule!r}')

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
  probe_set = (old_set[0][:1024], old_set[1][:1024])          # a random subset: old_set is a random draw of facts

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
      if cfg.loop_mode == 'infiloop':
        rec['gate_q_norm'] = float(model.loop_q.detach().norm())
        rec['gate_beta'] = float(torch.exp(-F.softplus(model.loop_r.detach().float())))
      if cfg.train_probe:
        rec['pass_probe'] = pass_probe(model, *probe_set)
      log.append(rec)
      print(f'[facts:{cfg.ffn}:N={N}] step={step} loss={rec["loss"]:.4f} (ln K={math.log(cfg.n_values):.3f}) '
            f'acc={rec["acc"]:.4f}' + (f' new_acc={rec["acc_new"]:.4f}' if M else '') + f' ({time.time() - t0:.0f}s)',
            flush=True)
    if step == total:
      break
    if cfg.lr_schedule == 'cosine' and step < cfg.steps:
      for g in opt.param_groups:
        g['lr'] = 0.5 * cfg.lr * (1.0 + math.cos(math.pi * step / cfg.steps))
    lo, hi = (0, N) if step < cfg.steps else (N, N + M)   # phase 2: only the M new individuals
    i = torch.randint(lo, hi, (cfg.batch_size,), generator=gen).to(device)
    a = torch.randint(0, A, (cfg.batch_size,), generator=gen).to(device)
    if cfg.deep_supervision and cfg.loops > 1:
      logits = model(sequences(i, a, gen), all_passes=True)          # (K, B, n_values): the head after each pass
      loss = F.cross_entropy(logits.flatten(0, 1), facts[i, a].repeat(cfg.loops))
    else:
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
  if model_path:
    from rcdl import ckpt
    ckpt.save(model_path, {'cfg': dc.asdict(cfg), 'model': {k: v.cpu() for k, v in model.state_dict().items()}}, cfg.steps)
  if M:
    phase2 = [r for r in log if r['step'] >= cfg.steps]
    result |= {'old_acc_min_during_finetune': min(r['acc'] for r in phase2), 'old_acc_after_finetune': log[-1]['acc'],
               'new_acc_after_finetune': log[-1]['acc_new']}
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
      json.dump(result, f, indent=1)
  return result
