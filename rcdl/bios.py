"""Synthetic biographies on a small Gemma 3 style decoder: the article's language experiment (claims 19-20).

The article (J. von Oswald, X, 9 Oct 2026) trains a small Gemma 3 model on the synthetic biographies of Zucchet et al.
(arXiv 2503.21676, "How do language models learn facts?") and replaces feed-forward layers by Cos-Bochner FFNs or by
growing RBF layers, "either throughout or in hybrids with one or two layers". It publishes no code or configuration.
This module rebuilds the experiment from the two sources; every choice they leave open is listed below.

Data (Zucchet et al. App. B). N individuals, each with a unique full name (first and middle name from a list of 900,
last name from a list of 1,000) and six attributes: birth date (month, day, year in 1900-2100, independent), birth
place and current location (800 cities), university (200), major (100), company (500), all uniform and independent.
Every attribute has 25 templates; each individual gets a random split into 20 training and 5 evaluation templates per
attribute. A biography is one sentence per attribute (template drawn from the individual's training split, full name in
every sentence, the name and the attribute type before the value), in a random order. Training sequences are packed
biographies (seq_len tokens), resampled at every step. Evaluation uses biographies of the same individuals written with
their 5 held-out templates, so a correct value is knowledge, not memorized text.
  Assumptions: Zucchet et al. used real names, cities and companies tokenized by a 32k SentencePiece model and 25
  LLM-written templates. Here names and values are random sequences of 1-3 "piece" tokens (probabilities 0.3 / 0.5 /
  0.2) from per-type piece vocabularies (first names 260 pieces, last names 300, cities 320, universities 120, majors 60,
  companies 240), a birth date is three tokens (month, day, year), and the 25 templates per attribute are 5 openers x 5
  hand-written frames, split into word tokens.

Metrics (Zucchet et al. App. B). Attribute loss: the cross-entropy summed over the tokens of one attribute value,
averaged over attribute instances. Attribute accuracy: the fraction of attribute instances with every value token
predicted correctly (greedy). No-knowledge baseline: mean over the six attributes of ln(number of values), the loss of a
model that knows the value distributions but no individual.

Model (Gemma 3 text decoder, scaled down; Zucchet et al. used an 8-layer, 512-wide Chinchilla-style decoder): token
embedding scaled by sqrt(d); per layer x <- x + post_attn_norm(Attn(input_norm(x))), x <- x + post_ffn_norm(FFN(
pre_ffn_norm(x))); Gemma RMSNorm (scale 1 + w, w = 0 at init); grouped-query attention with RMSNorm on queries and keys
(QK-norm) and RoPE (theta 10k on local layers, 1M on every 6th, global, layer); sliding window `sliding_window` on local
layers (512 = the sequence length here, so every layer sees the full causal context); GeGLU FFN (tanh GELU); tied output
embedding; no logit soft-capping. Initialization: normal(0, 0.02) for all matrices (Hugging Face Gemma 3 default).

Feed-forward variants (`ffn` for every layer, `special` for the last `n_special` layers):
  geglu       W_down (GELU_tanh(W_gate x) * W_up x), width `ffn_width`.
  cos         Cos-Bochner FFN: W_down sqrt(2/D) cos(Ω x + b), Ω_ij ~ N(0, 1/(T sqrt(d))), b ~ U(0, 2π), width D =
              `cos_width`; by Bochner's theorem the features approximate exp(-||x - y||^2 / (2 T sqrt(d))) as D grows.
              `cos_train`: Ω and b are trained as well (the article's "cosine non-linearity FFN"); else fixed.
  rbf         growing RBF layer (paper Sec. 3, article): s(q) = Σ_j exp(-||q - k_j||^2 / (2 T sqrt(d))) v_j over a growing
              table; after each step, every query q_i of the batch is stored as a key with value v_i = -η dℓ_i/ds_i, where
              ℓ_i is the token's cross-entropy (the batch loss is the mean over tokens, so dℓ_i/ds_i = n_tokens * dL/ds_i).
              Assumption: the layer reads and writes only at "fact positions" (inputs whose next token is an attribute
              value token) and outputs 0 elsewhere. Writing every token (65k per step at batch 128 x 512) would add
              about 10^9 rows over a run; the article does not say how its layer was restricted. The query is the
              parameter-free RMSNorm of the residual stream (Gemma's learned norm scales would move stored keys), and the
              output enters the residual stream directly (a post-norm would rescale an empty table's zero output). The
              table is a ring buffer of `rbf_capacity` rows (oldest rows overwritten), stored in bf16 on the GPU.
  geglu_fact  control for `rbf`: a GeGLU layer applied only at the fact positions (same routing, parametric memory).
"""

import dataclasses as dc
import json
import math
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ATTRS = ('birthdate', 'birthplace', 'university', 'major', 'company', 'location')
OPENERS = ('', 'notably ,', 'as far as we know ,', 'according to the records ,', 'it is worth noting that')
FRAMES = {
    'birthdate': ('{name} was born on {value} .', '{name} came into the world on {value} .',
                  '{name} has a birth date of {value} .', 'the birthday of {name} is {value} .',
                  '{name} first saw the light of day on {value} .'),
    'birthplace': ('{name} was born in {value} .', '{name} spent the early years of life in {value} .',
                   'the birthplace of {name} is {value} .', '{name} comes originally from {value} .',
                   '{name} was raised in the city of birth , {value} .'),
    'university': ('{name} studied at {value} .', '{name} graduated from {value} .',
                   'the alma mater of {name} is {value} .', '{name} received an education at {value} .',
                   '{name} attended classes at {value} .'),
    'major': ('{name} majored in {value} .', '{name} chose {value} as the field of study .',
              'the major of {name} was {value} .', '{name} completed a degree in {value} .',
              '{name} focused the studies on {value} .'),
    'company': ('{name} works for {value} .', '{name} is employed by {value} .', 'the employer of {name} is {value} .',
                '{name} earns a living at {value} .', '{name} holds a job at {value} .'),
    'location': ('{name} lives in {value} .', '{name} currently resides in {value} .',
                 'the current home of {name} is {value} .', '{name} has settled down in {value} .',
                 '{name} now calls {value} home .'),
}
N_TEMPLATES, N_TRAIN_TEMPLATES = 25, 20
PAD, BOS = 0, 1
MAX_NAME, MAX_VALUE = 9, 3


@dc.dataclass(frozen=True)
class BiosConfig:
  # population and data
  n_people: int = 8192          # N (pre-training population)
  world_seed: int = 0           # names, values, templates split
  seq_len: int = 512            # S (packed biographies)
  batch_size: int = 64          # B sequences per step
  # Gemma 3 style decoder
  d_model: int = 256
  layers: int = 6
  heads: int = 4
  kv_heads: int = 1
  head_dim: int = 64
  ffn_width: int = 1024         # GeGLU width
  sliding_window: int = 512     # local layers (Gemma 3: 512); >= seq_len means full causal attention
  global_every: int = 6         # every 6th layer is global (Gemma 3: 5 local : 1 global)
  rope_local: float = 1e4
  rope_global: float = 1e6
  # feed-forward layers
  ffn: str = 'geglu'            # all layers: 'geglu' | 'cos'
  special: str = 'none'         # last n_special layers: 'none' | 'rbf' | 'cos' | 'geglu_fact'
  n_special: int = 1
  cos_width: int = 4096         # D
  cos_T: float = 1.0            # Ω_ij ~ N(0, 1/(T sqrt(d)))
  cos_train: bool = True        # train Ω and b (else fixed random features)
  rbf_T: float = 2.0            # kernel exp(-||q - k||^2 / (2 T sqrt(d)))
  rbf_eta: float = 1.0          # η: v_i = -η dℓ_i/ds_i
  rbf_capacity: int = 4_000_000  # ring-buffer rows
  rbf_chunk: int = 65536        # rows per chunk in the kernel sums (PyTorch path)
  rbf_impl: str = 'triton'      # 'triton': fused kernels (rcdl/rbf_triton.py) on CUDA; 'torch': chunked PyTorch
  # optimization (Zucchet et al.: AdamW, betas 0.9 / 0.95, weight decay 0.1, cosine to 1e-7, no warm-up, lr 4e-4)
  lr: float = 4e-4
  weight_decay: float = 0.1
  steps: int = 4000
  final_lr: float = 1e-7
  grad_clip: float = 0.0        # 0 = off (not stated by Zucchet et al.)
  # evaluation
  eval_every: int = 100
  eval_bios: int = 1024         # held-out-template biographies of pre-training individuals (6 instances each)
  # fine-tuning on new individuals (Zucchet et al. Sec. 4: constant lr 3e-5, no replay)
  finetune_people: int = 0      # M
  finetune_steps: int = 0
  finetune_lr: float = 3e-5
  finetune_replay: float = 0.0  # fraction of fine-tuning biographies drawn from the pre-training population
  finetune_eval_every: int = 50
  seed: int = 1
  compile: bool = False         # torch.compile the blocks without a growing layer (CUDA)


# ----------------------------------------------------------------------------------------------------------- data

def _piece_strings(rng: np.random.Generator, n: int, pool: int, prefix: str) -> List[List[str]]:
  """n distinct strings of 1-3 tokens from `pool` piece tokens named prefix0 ... prefix{pool-1}."""
  out, seen = [], set()
  while len(out) < n:
    k = int(rng.choice(3, p=(0.3, 0.5, 0.2))) + 1
    s = tuple(int(x) for x in rng.integers(0, pool, k))
    if s not in seen:
      seen.add(s)
      out.append([f'{prefix}{i}' for i in s])
  return out


def _split_template(text: str) -> Tuple[List[str], List[str], List[str]]:
  words = text.split()
  i, j = words.index('{name}'), words.index('{value}')
  assert i < j, text
  return words[:i], words[i + 1:j], words[j + 1:]


class BioWorld:
  """Population, vocabulary, templates and the per-individual template split, as tensors on `device`."""

  def __init__(self, n_people: int, seed: int = 0, device='cpu'):
    rng = np.random.default_rng(seed)
    self.device = torch.device(device)
    first, last = _piece_strings(rng, 900, 260, 'f'), _piece_strings(rng, 1000, 300, 'l')
    cities, unis = _piece_strings(rng, 800, 320, 'c'), _piece_strings(rng, 200, 120, 'u')
    majors, comps = _piece_strings(rng, 100, 60, 'm'), _piece_strings(rng, 500, 240, 'o')
    months = [f'<month{i}>' for i in range(1, 13)]
    days = [f'<day{i}>' for i in range(1, 32)]
    years = [f'<year{i}>' for i in range(1900, 2101)]
    words = sorted({w for frames in FRAMES.values() for f in frames for w in f.split() if w[0] != '{'}
                   | {w for o in OPENERS for w in o.split()})
    pieces = ([f'f{i}' for i in range(260)] + [f'l{i}' for i in range(300)] + [f'c{i}' for i in range(320)]
              + [f'u{i}' for i in range(120)] + [f'm{i}' for i in range(60)] + [f'o{i}' for i in range(240)])
    self.itos = ['<pad>', '<bos>'] + words + pieces + months + days + years
    self.stoi = {s: i for i, s in enumerate(self.itos)}
    assert len(self.stoi) == len(self.itos)
    self.vocab_size = len(self.itos)
    enc = lambda toks: [self.stoi[t] for t in toks]
    # number of values per attribute; no-knowledge baseline = mean ln(count)
    self.n_values = (12 * 31 * 201, 800, 200, 100, 500, 800)
    self.no_knowledge_loss = float(np.mean([math.log(c) for c in self.n_values]))

    # individuals: unique token sequences for the full name, independent uniform attributes
    names, seen = [], set()
    while len(names) < n_people:
      toks = tuple(enc(first[rng.integers(900)] + first[rng.integers(900)] + last[rng.integers(1000)]))
      if toks not in seen:
        seen.add(toks)
        names.append(toks)
    self.n_people = n_people
    name_tok = torch.zeros(n_people, MAX_NAME, dtype=torch.long)
    name_len = torch.tensor([len(t) for t in names])
    for i, t in enumerate(names):
      name_tok[i, :len(t)] = torch.tensor(t)
    m, d, y = rng.integers(0, 12, n_people), rng.integers(0, 31, n_people), rng.integers(0, 201, n_people)
    birth, loc = rng.integers(0, 800, n_people), rng.integers(0, 800, n_people)
    uni, maj, comp = rng.integers(0, 200, n_people), rng.integers(0, 100, n_people), rng.integers(0, 500, n_people)
    self.value_index = torch.as_tensor(np.stack([(m * 31 + d) * 201 + y, birth, uni, maj, comp, loc], 1))
    val_tok = torch.zeros(n_people, 6, MAX_VALUE, dtype=torch.long)
    val_len = torch.zeros(n_people, 6, dtype=torch.long)
    for i in range(n_people):
      vals = ([months[m[i]], days[d[i]], years[y[i]]], cities[birth[i]], unis[uni[i]], majors[maj[i]], comps[comp[i]],
              cities[loc[i]])
      for a, v in enumerate(vals):
        val_tok[i, a, :len(v)] = torch.tensor(enc(v))
        val_len[i, a] = len(v)

    # templates: 5 openers x 5 frames per attribute, split around {name} and {value}
    parts = [[_split_template((o + ' ' + f).strip()) for o in OPENERS for f in FRAMES[a]] for a in ATTRS]
    self.max_tpl = max(len(s) for p in parts for t in p for s in t)
    tpl_tok = torch.zeros(6, N_TEMPLATES, 3, self.max_tpl, dtype=torch.long)
    tpl_len = torch.zeros(6, N_TEMPLATES, 3, dtype=torch.long)
    for a in range(6):
      for t in range(N_TEMPLATES):
        for s, seg in enumerate(parts[a][t]):
          tpl_tok[a, t, s, :len(seg)] = torch.tensor(enc(seg), dtype=torch.long)
          tpl_len[a, t, s] = len(seg)
    split = torch.as_tensor(np.argsort(rng.random((n_people, 6, N_TEMPLATES)), axis=-1))  # [:20] train, [20:] eval
    dev = self.device
    self.name_tok, self.name_len = name_tok.to(dev), name_len.to(dev)
    self.val_tok, self.val_len = val_tok.to(dev), val_len.to(dev)
    self.tpl_tok, self.tpl_len, self.split = tpl_tok.to(dev), tpl_len.to(dev), split.to(dev)
    g = torch.Generator(device=dev).manual_seed(seed + 99)
    tok, valid, _, _ = self._assemble(torch.randint(0, n_people, (4096,), generator=g, device=dev), 'train', g)
    self.mean_bio_len = valid.sum().item() / 4096

  def _assemble(self, people: torch.Tensor, split: str, gen: torch.Generator):
    """Uncompacted biographies: tokens, valid mask, value mask and attribute id, each (n, 1 + 6 * sentence width)."""
    n, dev = len(people), self.device
    order = torch.argsort(torch.rand(n, 6, generator=gen, device=dev), dim=1)       # attribute in each sentence slot
    col = (torch.randint(0, N_TRAIN_TEMPLATES, (n, 6), generator=gen, device=dev) if split == 'train' else
           N_TRAIN_TEMPLATES + torch.randint(0, N_TEMPLATES - N_TRAIN_TEMPLATES, (n, 6), generator=gen, device=dev))
    tpl = self.split[people[:, None], order, col]                                    # (n, 6) template ids
    segs = [(self.tpl_tok[order, tpl, 0], self.tpl_len[order, tpl, 0]),
            (self.name_tok[people][:, None].expand(n, 6, MAX_NAME), self.name_len[people][:, None].expand(n, 6)),
            (self.tpl_tok[order, tpl, 1], self.tpl_len[order, tpl, 1]),
            (self.val_tok[people[:, None], order], self.val_len[people[:, None], order]),
            (self.tpl_tok[order, tpl, 2], self.tpl_len[order, tpl, 2])]
    tok = torch.cat([s for s, _ in segs], -1)
    valid = torch.cat([torch.arange(s.shape[-1], device=dev) < ln[..., None] for s, ln in segs], -1)
    is_val = torch.cat([torch.full_like(s, k == 3, dtype=torch.bool) for k, (s, _) in enumerate(segs)], -1) & valid
    attr = order[..., None].expand_as(tok)
    one = torch.ones(n, 1, dtype=torch.bool, device=dev)
    return (torch.cat([torch.full((n, 1), BOS, device=dev), tok.reshape(n, -1)], 1),
            torch.cat([one, valid.reshape(n, -1)], 1), torch.cat([~one, is_val.reshape(n, -1)], 1),
            torch.cat([torch.zeros(n, 1, dtype=torch.long, device=dev), attr.reshape(n, -1)], 1))

  def train_batch(self, lo: int, hi: int, batch_size: int, seq_len: int, gen: torch.Generator,
                  replay: float = 0.0, replay_hi: int = 0):
    """Packed sequences of fresh biographies of individuals in [lo, hi) (a fraction `replay` from [0, replay_hi)).

    Returns inputs (B, S), targets (B, S) and fact (B, S): True where the target is an attribute value token."""
    need, dev = batch_size * (seq_len + 1), self.device
    toks, vals, total = [], [], 0
    while total < need:
      nb = int((need - total) / self.mean_bio_len * 1.1) + 8
      people = torch.randint(lo, hi, (nb,), generator=gen, device=dev)
      if replay > 0:
        old = torch.rand(nb, generator=gen, device=dev) < replay
        people = torch.where(old, torch.randint(0, replay_hi, (nb,), generator=gen, device=dev), people)
      tok, valid, is_val, _ = self._assemble(people, 'train', gen)
      toks.append(tok[valid])
      vals.append(is_val[valid])
      total += toks[-1].numel()
    stream = torch.cat(toks)[:need].view(batch_size, seq_len + 1)
    fact = torch.cat(vals)[:need].view(batch_size, seq_len + 1)
    return stream[:, :-1], stream[:, 1:], fact[:, 1:]

  def eval_rows(self, people: torch.Tensor, split: str, gen: torch.Generator):
    """One biography per row (compacted, PAD-filled): inputs, targets, fact mask and instance id (row * 6 + attribute,
    -1 off value targets)."""
    tok, valid, is_val, attr = self._assemble(people, split, gen)
    n, dev = len(people), self.device
    pos = valid.long().cumsum(1) - 1
    L = int(valid.sum(1).max())
    rows = torch.arange(n, device=dev)[:, None].expand_as(tok)
    out = torch.full((n, L), PAD, dtype=torch.long, device=dev)
    out[rows[valid], pos[valid]] = tok[valid]
    inst = torch.full((n, L), -1, dtype=torch.long, device=dev)
    inst[rows[is_val], pos[is_val]] = (rows * 6 + attr)[is_val]
    return out[:, :-1], out[:, 1:], inst[:, 1:] >= 0, inst[:, 1:]


# ----------------------------------------------------------------------------------------------------------- model

class GemmaRMSNorm(nn.Module):
  def __init__(self, d: int, eps: float = 1e-6):
    super().__init__()
    self.eps, self.weight = eps, nn.Parameter(torch.zeros(d))

  def forward(self, x):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps) * (1.0 + self.weight.float())).type_as(x)


def rope_tables(S: int, hd: int, theta: float, device) -> Tuple[torch.Tensor, torch.Tensor]:
  inv = 1.0 / theta ** (torch.arange(0, hd, 2, device=device, dtype=torch.float32) / hd)
  ang = torch.arange(S, device=device, dtype=torch.float32)[:, None] * inv[None]
  ang = torch.cat([ang, ang], -1)                                   # rotate-half layout (Hugging Face Gemma)
  return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
  h = x.shape[-1] // 2
  rot = torch.cat([-x[..., h:], x[..., :h]], -1)
  return (x.float() * cos + rot.float() * sin).type_as(x)


class Attention(nn.Module):
  def __init__(self, cfg: BiosConfig):
    super().__init__()
    d, H, KV, hd = cfg.d_model, cfg.heads, cfg.kv_heads, cfg.head_dim
    self.H, self.KV, self.hd = H, KV, hd
    self.q, self.k, self.v = nn.Linear(d, H * hd, bias=False), nn.Linear(d, KV * hd, bias=False), nn.Linear(d, KV * hd, bias=False)
    self.o = nn.Linear(H * hd, d, bias=False)
    self.q_norm, self.k_norm = GemmaRMSNorm(hd), GemmaRMSNorm(hd)

  def forward(self, x, cos, sin, window: Optional[int]):
    B, S, _ = x.shape
    q = self.q_norm(self.q(x).view(B, S, self.H, self.hd)).transpose(1, 2)
    k = self.k_norm(self.k(x).view(B, S, self.KV, self.hd)).transpose(1, 2)
    v = self.v(x).view(B, S, self.KV, self.hd).transpose(1, 2)
    q, k = apply_rope(q, cos[:S], sin[:S]), apply_rope(k, cos[:S], sin[:S])
    if window is not None and window < S:                          # sliding-window causal mask (local layers)
      i = torch.arange(S, device=x.device)
      mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < window)
      o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.hd ** -0.5, enable_gqa=self.KV != self.H)
    else:
      o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=self.hd ** -0.5, enable_gqa=self.KV != self.H)
    return self.o(o.transpose(1, 2).reshape(B, S, self.H * self.hd))


class GeGLU(nn.Module):
  def __init__(self, d: int, w: int):
    super().__init__()
    self.gate, self.up, self.down = nn.Linear(d, w, bias=False), nn.Linear(d, w, bias=False), nn.Linear(w, d, bias=False)

  def forward(self, x):
    return self.down(F.gelu(self.gate(x), approximate='tanh') * self.up(x))


class CosFFN(nn.Module):
  """Cos-Bochner FFN: W_down sqrt(2/D) cos(Ω x + b), Ω_ij ~ N(0, 1/(T sqrt(d))), b ~ U(0, 2π)."""

  def __init__(self, d: int, D: int, T: float, train_features: bool, gen: torch.Generator):
    super().__init__()
    omega = torch.randn(D, d, generator=gen) / math.sqrt(T * math.sqrt(d))
    b = torch.rand(D, generator=gen) * 2 * math.pi
    if train_features:
      self.omega, self.b = nn.Parameter(omega), nn.Parameter(b)
    else:
      self.register_buffer('omega', omega)
      self.register_buffer('b', b)
    self.down, self.scale = nn.Linear(D, d, bias=False), math.sqrt(2.0 / D)

  def forward(self, x):
    return self.down(self.scale * torch.cos(F.linear(x, self.omega.to(x.dtype), self.b.to(x.dtype))))


def _mm_f32(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """a @ b with an fp32 result: bf16 inputs on CUDA use fp32 accumulation and output (no bf16 rounding of the result)."""
  if a.dtype == torch.bfloat16 and a.is_cuda:
    try:
      return torch.mm(a, b, out_dtype=torch.float32)
    except (RuntimeError, TypeError):
      pass
  return a.float() @ b.float()


class _RBFRead(torch.autograd.Function):
  """s_i = Σ_j κ_ij v_j with κ_ij = exp(-||q_i - k_j||^2 / c), summed over row chunks (no Q x N matrix is kept).

  Backward (keys and values are fixed): ds/dq gives dq_i = (2/c) Σ_j κ_ij <g_i, v_j> (k_j - q_i).
  fused=True runs the Triton kernels of rcdl/rbf_triton.py (same sums, κ kept in registers)."""

  @staticmethod
  def forward(ctx, q, keys, values, key_sq, c: float, chunk: int, fused: bool = False):
    qf = q.float()
    q_low = qf.to(keys.dtype)
    q_sq = q_low.float().pow(2).sum(-1, keepdim=True)              # of the rounded query: a stored copy has d2 = 0
    ctx.save_for_backward(qf, keys, values, key_sq)
    ctx.c, ctx.chunk, ctx.fused = c, chunk, fused
    if fused:
      from rcdl.rbf_triton import rbf_forward
      return rbf_forward(q_low, q_sq[:, 0], keys, key_sq, values, c)
    out = q.new_zeros(q.shape[0], values.shape[1], dtype=torch.float32)
    for i in range(0, keys.shape[0], chunk):
      d2 = (q_sq + key_sq[i:i + chunk] - 2.0 * _mm_f32(q_low, keys[i:i + chunk].T)).clamp_min_(0.0)
      out += _mm_f32(torch.exp_(d2.mul_(-1.0 / c)).to(values.dtype), values[i:i + chunk])
    return out

  @staticmethod
  def backward(ctx, g):
    qf, keys, values, key_sq = ctx.saved_tensors
    c, chunk = ctx.c, ctx.chunk
    q_low, g_low = qf.to(keys.dtype), g.float().to(values.dtype)
    q_sq = q_low.float().pow(2).sum(-1, keepdim=True)
    if ctx.fused:
      from rcdl.rbf_triton import rbf_backward
      acc, rs = rbf_backward(q_low, q_sq[:, 0], g_low, keys, key_sq, values, c)
      return (acc - rs[:, None] * qf) * (2.0 / c), None, None, None, None, None, None
    dq = torch.zeros_like(qf)
    for i in range(0, keys.shape[0], chunk):
      d2 = (q_sq + key_sq[i:i + chunk] - 2.0 * _mm_f32(q_low, keys[i:i + chunk].T)).clamp_min_(0.0)
      a = torch.exp_(d2.mul_(-1.0 / c)).mul_(_mm_f32(g_low, values[i:i + chunk].T))   # κ_ij <g_i, v_j>
      dq += _mm_f32(a.to(keys.dtype), keys[i:i + chunk]) - a.sum(1, keepdim=True) * qf
    return dq * (2.0 / c), None, None, None, None, None, None


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  xf = x.float()
  return xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)


class GrowingRBF(nn.Module):
  """Growing RBF layer at the fact positions (see the module docstring); `loss_tokens` is set by the trainer."""

  def __init__(self, cfg: BiosConfig, device):
    super().__init__()
    d, dev = cfg.d_model, torch.device(device)
    dtype = torch.bfloat16 if dev.type == 'cuda' else torch.float32
    self.c, self.eta, self.chunk = 2.0 * cfg.rbf_T * math.sqrt(d), cfg.rbf_eta, cfg.rbf_chunk
    self.fused = cfg.rbf_impl == 'triton' and dev.type == 'cuda'
    self.register_buffer('keys', torch.zeros(cfg.rbf_capacity, d, dtype=dtype, device=dev), persistent=False)
    self.register_buffer('values', torch.zeros(cfg.rbf_capacity, d, dtype=dtype, device=dev), persistent=False)
    self.register_buffer('key_sq', torch.zeros(cfg.rbf_capacity, device=dev), persistent=False)
    self.size, self.next_row, self.written, self.pending, self.loss_tokens = 0, 0, 0, None, 1

  def forward(self, x, fact):
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    if fact is None or not bool(fact.any()):
      return out
    q = rms_norm(x[fact])
    s = _RBFRead.apply(q, self.keys[:self.size], self.values[:self.size], self.key_sq[:self.size], self.c, self.chunk,
                       self.fused)
    if self.training and torch.is_grad_enabled():
      qd = q.detach()
      s.register_hook(lambda g: setattr(self, 'pending', (qd, -self.eta * self.loss_tokens * g.float())))
    out[fact] = s
    return out

  @torch.no_grad()
  def commit(self):
    """Writes the last step's rows (after the optimizer step)."""
    if self.pending is None:
      return
    k, v = self.pending
    self.pending = None
    n, cap = len(k), len(self.keys)
    k, v = k[-cap:], v[-cap:]
    idx = (torch.arange(len(k), device=k.device) + self.next_row) % cap
    self.keys[idx], self.values[idx] = k.to(self.keys.dtype), v.to(self.values.dtype)
    self.key_sq[idx] = self.keys[idx].float().pow(2).sum(-1)
    self.next_row = (self.next_row + len(k)) % cap
    self.size, self.written = min(cap, self.size + len(k)), self.written + n


class Routed(nn.Module):
  """Applies `ffn` at the fact positions only (zeros elsewhere)."""

  def __init__(self, ffn: nn.Module):
    super().__init__()
    self.ffn = ffn

  def forward(self, x, fact):
    out = torch.zeros_like(x)
    if fact is not None and bool(fact.any()):
      out[fact] = self.ffn(x[fact]).to(out.dtype)
    return out


class Block(nn.Module):
  def __init__(self, cfg: BiosConfig, i: int, kind: str, gen: torch.Generator, device):
    super().__init__()
    d = cfg.d_model
    self.kind, self.is_global = kind, (i + 1) % cfg.global_every == 0
    self.window = None if self.is_global else cfg.sliding_window
    self.input_norm, self.post_attn_norm = GemmaRMSNorm(d), GemmaRMSNorm(d)
    self.pre_ffn_norm, self.post_ffn_norm = GemmaRMSNorm(d), GemmaRMSNorm(d)
    self.attn = Attention(cfg)
    if kind == 'rbf':
      self.ffn = GrowingRBF(cfg, device)
    elif kind == 'cos':
      self.ffn = CosFFN(d, cfg.cos_width, cfg.cos_T, cfg.cos_train, gen)
    elif kind == 'geglu_fact':
      self.ffn = Routed(GeGLU(d, cfg.ffn_width))
    elif kind == 'geglu':
      self.ffn = GeGLU(d, cfg.ffn_width)
    else:
      raise ValueError(f'unknown FFN kind {kind!r}')

  def forward(self, x, rope, fact):
    cos, sin = rope[1] if self.is_global else rope[0]
    x = x + self.post_attn_norm(self.attn(self.input_norm(x), cos, sin, self.window))
    if self.kind == 'rbf':                                 # growing layer: plain RMSNorm query, no post-norm
      return x + self.ffn(x, fact).to(x.dtype)
    h = self.pre_ffn_norm(x)
    h = self.ffn(h, fact) if self.kind == 'geglu_fact' else self.ffn(h)
    return x + self.post_ffn_norm(h)


class GemmaBios(nn.Module):
  def __init__(self, cfg: BiosConfig, vocab: int, device='cpu'):
    super().__init__()
    gen = torch.Generator().manual_seed(cfg.seed)
    self.cfg = cfg
    kinds = [cfg.ffn] * cfg.layers
    if cfg.special != 'none':
      for i in range(cfg.layers - cfg.n_special, cfg.layers):
        kinds[i] = cfg.special
    self.kinds = kinds
    self.embed = nn.Embedding(vocab, cfg.d_model)
    self.blocks = nn.ModuleList([Block(cfg, i, k, gen, device) for i, k in enumerate(kinds)])
    self.final_norm = GemmaRMSNorm(cfg.d_model)
    for name, p in self.named_parameters():            # normal(0, 0.02) for matrices (Gemma 3 initializer_range)
      if p.dim() == 2 and 'omega' not in name:
        with torch.no_grad():
          p.copy_(torch.randn(p.shape, generator=gen) * 0.02)
    for blk in self.blocks:
      if blk.kind == 'cos':
        nn.init.zeros_(blk.ffn.down.weight)
    self.to(device)
    S = cfg.seq_len
    self.rope = (rope_tables(S, cfg.head_dim, cfg.rope_local, device), rope_tables(S, cfg.head_dim, cfg.rope_global, device))

  def growing_layers(self) -> List[GrowingRBF]:
    return [b.ffn for b in self.blocks if b.kind == 'rbf']

  def forward(self, tokens, fact=None):
    x = self.embed(tokens) * math.sqrt(self.cfg.d_model)
    for blk in self.blocks:
      x = blk(x, self.rope, fact)
    return self.final_norm(x) @ self.embed.weight.T.to(x.dtype)


# ----------------------------------------------------------------------------------------------------------- training

@torch.no_grad()
def evaluate(model: GemmaBios, data, batch: int = 256, amp: bool = False) -> Dict[str, float]:
  """Attribute loss and accuracy (Zucchet et al.) on held-out-template biographies, plus per-attribute accuracy."""
  inputs, targets, fact, inst = data
  model.eval()
  n_inst = int(inst.max()) + 1
  loss = torch.zeros(n_inst, device=inputs.device)
  wrong = torch.zeros(n_inst, device=inputs.device)
  for i in range(0, len(inputs), batch):
    with torch.autocast('cuda', torch.bfloat16, enabled=amp):
      logits = model(inputs[i:i + batch], fact[i:i + batch])
    f = fact[i:i + batch]
    lg, tg, ins = logits[f].float(), targets[i:i + batch][f], inst[i:i + batch][f]
    loss.index_add_(0, ins, F.cross_entropy(lg, tg, reduction='none'))
    wrong.index_add_(0, ins, (lg.argmax(-1) != tg).float())
  model.train()
  acc = (wrong == 0).float()
  per_attr = acc.view(-1, 6).mean(0).tolist()
  return {'attr_loss': loss.mean().item(), 'attr_acc': acc.mean().item(),
          **{f'acc_{a}': per_attr[k] for k, a in enumerate(ATTRS)}}


def run_bios(cfg: BiosConfig, out_dir: Optional[str] = None, device: str = 'cpu') -> Dict[str, Any]:
  dev = torch.device(device)
  amp = dev.type == 'cuda'
  N, M = cfg.n_people, cfg.finetune_people
  world = BioWorld(N + M, cfg.world_seed, dev)
  torch.manual_seed(cfg.seed)
  model = GemmaBios(cfg, world.vocab_size, dev)
  if cfg.compile and amp:
    for blk in model.blocks:
      if blk.kind in ('geglu', 'cos'):
        blk.compile()
  params = [p for p in model.parameters() if p.requires_grad]
  decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() >= 2]
  no_decay = [p for n, p in model.named_parameters() if p.requires_grad and p.dim() < 2]
  opt = torch.optim.AdamW([{'params': decay, 'weight_decay': cfg.weight_decay}, {'params': no_decay, 'weight_decay': 0.0}],
                          lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8, fused=amp or None)
  gen = torch.Generator(device=dev).manual_seed(cfg.seed)
  eval_gen = torch.Generator(device=dev).manual_seed(10_000 + cfg.seed)
  n_eval = min(cfg.eval_bios, N)
  old_people = torch.randperm(N, generator=torch.Generator().manual_seed(cfg.seed))[:n_eval].to(dev)
  old_set = world.eval_rows(old_people, 'eval', eval_gen)
  new_set = world.eval_rows(torch.arange(N, N + min(M, cfg.eval_bios), device=dev), 'eval', eval_gen) if M else None
  growing = model.growing_layers()

  def lr_at(step):
    if step >= cfg.steps:
      return cfg.finetune_lr
    return cfg.final_lr + 0.5 * (cfg.lr - cfg.final_lr) * (1.0 + math.cos(math.pi * step / cfg.steps))

  log, t0, train_losses, pending = [], time.time(), [], []

  def record(step, phase):
    rec = {'step': step, 'phase': phase, 'sec': time.time() - t0, 'lr': lr_at(step)}
    rec |= {k: v for k, v in evaluate(model, old_set, amp=amp).items()}
    if new_set is not None and phase == 'finetune':
      rec |= {f'new_{k}': v for k, v in evaluate(model, new_set, amp=amp).items()}
    if pending:                                      # one host synchronization per evaluation
      rec['train_loss'] = float(torch.stack(pending).mean())
      train_losses.append(rec['train_loss'])
      pending.clear()
    if growing:
      rec['rbf_rows'] = [g.size for g in growing]
    log.append(rec)
    print(f'[bios:{"+".join(sorted(set(model.kinds)))}:N={N}] step={step} {phase} attr_loss={rec["attr_loss"]:.3f} '
          f'(no-knowledge {world.no_knowledge_loss:.3f}) attr_acc={rec["attr_acc"]:.4f}'
          + (f' new_acc={rec["new_attr_acc"]:.4f}' if 'new_attr_acc' in rec else '')
          + f' train_loss={rec.get("train_loss", float("nan")):.3f} ({rec["sec"]:.0f}s)', flush=True)

  total = cfg.steps + (cfg.finetune_steps if M else 0)
  for step in range(total + 1):
    phase = 'pretrain' if step <= cfg.steps else 'finetune'
    if step == cfg.steps or (step < cfg.steps and step % cfg.eval_every == 0) or \
       (step > cfg.steps and (step - cfg.steps) % cfg.finetune_eval_every == 0):
      record(step, phase if step < cfg.steps or not M else 'finetune')
    if step == total:
      break
    for group in opt.param_groups:
      group['lr'] = lr_at(step)
    if step < cfg.steps:
      x, y, fact = world.train_batch(0, N, cfg.batch_size, cfg.seq_len, gen)
    else:
      x, y, fact = world.train_batch(N, N + M, cfg.batch_size, cfg.seq_len, gen, cfg.finetune_replay, N)
    for g in growing:
      g.loss_tokens = y.numel()
    with torch.autocast('cuda', torch.bfloat16, enabled=amp):
      logits = model(x, fact)
    loss = F.cross_entropy(logits.float().view(-1, logits.shape[-1]), y.reshape(-1))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    if cfg.grad_clip > 0:
      torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
    opt.step()
    for g in growing:
      g.commit()
    pending.append(loss.detach())

  base = world.no_knowledge_loss
  pre = [r for r in log if r['step'] <= cfg.steps]
  first = lambda cond: next((r['step'] for r in pre if cond(r)), None)
  result = {'cfg': dc.asdict(cfg), 'log': log, 'no_knowledge_loss': base, 'vocab_size': world.vocab_size,
            'mean_bio_len': world.mean_bio_len, 'trained_params': sum(p.numel() for p in params), 'ffn_kinds': model.kinds,
            'plateau_reached_step': first(lambda r: r['attr_loss'] < 1.05 * base),
            'plateau_end_step': first(lambda r: r['attr_loss'] < 0.9 * base),
            'acc50_step': first(lambda r: r['attr_acc'] >= 0.5), 'acc90_step': first(lambda r: r['attr_acc'] >= 0.9),
            'final_attr_acc': pre[-1]['attr_acc'], 'final_attr_loss': pre[-1]['attr_loss'],
            'bios_per_step': cfg.batch_size * cfg.seq_len / world.mean_bio_len}
  if M:
    post = [r for r in log if r['step'] >= cfg.steps]
    result |= {'old_acc_before_finetune': post[0]['attr_acc'], 'old_acc_min_finetune': min(r['attr_acc'] for r in post),
               'old_acc_after_finetune': post[-1]['attr_acc'], 'new_acc_after_finetune': post[-1].get('new_attr_acc')}
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
      json.dump(result, f, indent=1)
  return result
