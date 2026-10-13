"""Minimal synthetic-fact task for the article's (unpublished) biography experiment.

The article replaces the feed-forward layers of a small Gemma 3 model by growing RBF layers, wide random-cosine
("Cos Bochner") layers or GeGLU layers, trains on the synthetic biographies of Zucchet et al. (arXiv 2503.21676), and
reports (i) a loss plateau whose length grows about linearly with the number of individuals N, (ii) a much shorter
plateau for RBF and wide Cos-Bochner layers, (iii) a lower ceiling for GeGLU.

This module keeps only the fact-storage core of that task. There are N individuals and A attributes; fact (i, a) is a
value in {0, ..., K-1}, drawn uniformly at random. Each training example is the token pair (name_i, attribute_a), and
the model predicts the value from the last position. Training is online: every step draws a fresh batch of (i, a)
pairs. Accuracy is measured on all N x A facts. Before facts are learned, the best possible loss is ln K (uniform
values), so the plateau is the stretch of training where the loss stays near ln K.

FFN variants (all pre-LayerNorm, residual):
  geglu  W_o (GELU(W_g x) * (W_u x)), width w, all weights trained.
  cos    random Fourier features: W_o sqrt(2/w) cos(W_r x + b), W_r ~ N(0, 1/σ^2) and b ~ U(0, 2π) fixed, W_o trained;
         by Bochner's theorem the features approximate an RBF kernel of bandwidth σ as w grows.
  rbf    growing RBF layer (rcdl.layers.RCDLLayer, kind 'rbf'): one stored row per token per step.
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


class FactsModel(nn.Module):
  def __init__(self, cfg: FactsConfig, gen: torch.Generator):
    super().__init__()
    V = cfg.n_people + cfg.n_attrs
    self.cfg = cfg
    self.emb = nn.Embedding(V, cfg.d)
    self.pos = nn.Parameter(torch.randn(2, cfg.d, generator=gen) * 0.02)
    self.ln1 = nn.ModuleList([nn.LayerNorm(cfg.d) for _ in range(cfg.layers)])
    self.attn = nn.ModuleList([nn.MultiheadAttention(cfg.d, cfg.heads, batch_first=True) for _ in range(cfg.layers)])
    self.ln2 = nn.ModuleList([nn.LayerNorm(cfg.d) for _ in range(cfg.layers)])
    ffns = []
    for i in range(cfg.layers):
      use_rbf = cfg.ffn == 'rbf' and (cfg.rbf_layers == 'all' or i == cfg.layers - 1)
      if use_rbf:
        rc = RCDLConfig(kernel_type='rbf', hidden_dim=cfg.d, temperature=cfg.rbf_temperature, eta=cfg.rbf_eta,
                        db_size=2 * cfg.batch_size * cfg.steps, batch_size=cfg.batch_size, num_classes=cfg.d)
        ffns.append(RCDLLayer(rc, cfg.d, cfg.d, i, gen))
      elif cfg.ffn == 'cos':
        ffns.append(CosFeatures(cfg.d, cfg.width, cfg.cos_sigma, gen))
      else:
        ffns.append(GeGLU(cfg.d, cfg.width))
    self.ffn = nn.ModuleList(ffns)
    self.ln_f = nn.LayerNorm(cfg.d)
    self.head = nn.Linear(cfg.d, cfg.n_values)
    self.register_buffer('mask', torch.triu(torch.ones(2, 2, dtype=torch.bool), 1))

  def forward(self, tokens):                       # tokens: (B, 2) = (name, attribute)
    h = self.emb(tokens) + self.pos
    for i in range(self.cfg.layers):
      a = self.ln1[i](h)
      h = h + self.attn[i](a, a, a, attn_mask=self.mask, need_weights=False)[0]
      f = self.ln2[i](h)
      out = self.ffn[i](f.reshape(-1, f.shape[-1])).view_as(f) if isinstance(self.ffn[i], RCDLLayer) else self.ffn[i](f)
      h = h + out
    return self.head(self.ln_f(h[:, -1]))

  def rcdl_layers(self) -> Dict[str, RCDLLayer]:
    return {f'ffn_{i}': m for i, m in enumerate(self.ffn) if isinstance(m, RCDLLayer)}


def run_facts(cfg: FactsConfig, out_dir: Optional[str] = None, device: str = 'cpu') -> Dict[str, Any]:
  gen = torch.Generator().manual_seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  facts = torch.randint(0, cfg.n_values, (cfg.n_people, cfg.n_attrs), generator=gen).to(device)
  model = FactsModel(cfg, gen).to(device)
  params = [p for p in model.parameters() if p.requires_grad]
  opt = torch.optim.Adam(params, lr=cfg.lr)
  people = torch.arange(cfg.n_people, device=device).repeat_interleave(cfg.n_attrs)
  attrs = torch.arange(cfg.n_attrs, device=device).repeat(cfg.n_people)
  all_tokens = torch.stack([people, cfg.n_people + attrs], 1)
  all_targets = facts[people, attrs]

  @torch.no_grad()
  def evaluate():
    model.eval()
    loss = correct = 0.0
    for i in range(0, len(all_tokens), 4096):
      logits = model(all_tokens[i:i + 4096])
      loss += F.cross_entropy(logits, all_targets[i:i + 4096], reduction='sum').item()
      correct += (logits.argmax(-1) == all_targets[i:i + 4096]).sum().item()
    model.train()
    return loss / len(all_tokens), correct / len(all_tokens)

  log, t0 = [], time.time()
  for step in range(cfg.steps + 1):
    if step % cfg.eval_every == 0:
      loss, acc = evaluate()
      log.append({'step': step, 'loss': loss, 'acc': acc, 'sec': time.time() - t0})
      print(f'[facts:{cfg.ffn}:N={cfg.n_people}] step={step} loss={loss:.4f} (ln K={math.log(cfg.n_values):.3f}) '
            f'acc={acc:.4f} ({time.time() - t0:.0f}s)', flush=True)
    if step == cfg.steps:
      break
    i = torch.randint(0, cfg.n_people, (cfg.batch_size,), generator=gen).to(device)
    a = torch.randint(0, cfg.n_attrs, (cfg.batch_size,), generator=gen).to(device)
    logits = model(torch.stack([i, cfg.n_people + a], 1))
    loss = F.cross_entropy(logits, facts[i, a])
    opt.zero_grad()
    loss.backward()
    opt.step()
    for layer in model.rcdl_layers().values():
      layer.commit()
  # Plateau end: first evaluation at which the loss is 10% below ln K.
  thresh = 0.9 * math.log(cfg.n_values)
  plateau_end = next((r['step'] for r in log if r['loss'] < thresh), None)
  result = {'cfg': dc.asdict(cfg), 'log': log, 'final_acc': log[-1]['acc'], 'final_loss': log[-1]['loss'],
            'plateau_end_step': plateau_end, 'params': sum(p.numel() for p in params)}
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
      json.dump(result, f, indent=1)
  return result
