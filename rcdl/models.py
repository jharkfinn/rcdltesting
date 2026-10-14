"""Residual classifier backbone shared by all RCDL layers and the dense ResNet-MLP baseline (notebook Section 3.1)."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.layers import CosBochnerLayer, RCDLConfig, RCDLLayer, rms_norm


def dense(d_in: int, d_out: int, gen: torch.Generator) -> nn.Linear:
  """nn.Linear with truncated LeCun-normal weight init and zero bias."""
  lin = nn.Linear(d_in, d_out)
  with torch.no_grad():
    w = nn.init.trunc_normal_(torch.empty(d_out, d_in), std=1.0, a=-2.0, b=2.0, generator=gen)
    lin.weight.copy_(w * math.sqrt(1.0 / d_in) / 0.87962566103423978)
    lin.bias.zero_()
  return lin


ACTIVATIONS = {'none': lambda v: v, 'gelu': lambda v: F.gelu(v, approximate='tanh'), 'relu': F.relu}


class CifarRCDLClassifier(nn.Module):
  """Stacked residual classifier supporting all RCDL layers, Parametric Softmax, and standard dense (ResNet-MLP) layers:
  L pre-norm residual blocks x <- x + act(layer(RMSNorm(x))) followed by a readout layer of the same kind."""

  def __init__(self, cfg: RCDLConfig, device=None):
    super().__init__()
    self.cfg, self.act = cfg, ACTIVATIONS[cfg.nonlinearity]
    gen, H = torch.Generator().manual_seed(cfg.global_init_seed), cfg.hidden_dim
    if cfg.kernel_type == 'standard':
      layer = lambda d_out, layer_id: dense(H, d_out, gen)
    elif cfg.kernel_type == 'cos':        # random-feature (Cos-Bochner) form of the growing RBF layer
      layer = lambda d_out, layer_id: CosBochnerLayer(cfg, H, d_out, gen, device=device)
    elif cfg.kernel_type == 'bm25':       # growing RBF layer with sparse BM25 key matching
      from rcdl.bm25 import BM25RBFLayer
      layer = lambda d_out, layer_id: BM25RBFLayer(cfg, H, d_out, layer_id, gen, device=device)
    else:
      layer = lambda d_out, layer_id: RCDLLayer(cfg, H, d_out, layer_id, gen, device=device)
    self.layers = nn.ModuleList([layer(H, i) for i in range(cfg.num_layers)])
    self.output_layer = layer(cfg.num_classes, cfg.num_layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      y = self.act(layer(rms_norm(x)))
      x = x + y if self.cfg.use_residual else y
    return self.output_layer(rms_norm(x))

  def rcdl_layers(self) -> Dict[str, RCDLLayer]:
    """{'layer_0', ..., 'output_layer'} -> RCDLLayer (empty for 'standard')."""
    layers = {f'layer_{i}': m for i, m in enumerate(self.layers)}
    layers['output_layer'] = self.output_layer
    return {name: m for name, m in layers.items() if isinstance(m, RCDLLayer)}


class CifarViT(nn.Module):
  """ViT for 32x32 CIFAR images (paper App. E.9): P x P patches (64 tokens for P = 4) plus a CLS token, learned
  positional embeddings, cfg.num_layers pre-LayerNorm blocks x <- x + MHA(LN(x)); x <- x + FFN(LN(x)), a final
  LayerNorm and a linear readout on the CLS token.

  kernel_type 'standard': dense FFN (d -> vit_ffn_blowup * d -> d, GELU). 'gws' / 'gauge' / 'rbf': the FFN is a
  growing RCDL layer applied to every token (one stored row per token per training image); all other parameters are
  trained by the optimizer.
  """

  def __init__(self, cfg: RCDLConfig, device=None):
    super().__init__()
    torch.manual_seed(cfg.global_init_seed)
    gen = torch.Generator().manual_seed(cfg.global_init_seed)
    d, P, L = cfg.hidden_dim, cfg.vit_patch, cfg.num_layers
    self.cfg, self.n_tok = cfg, (32 // P) ** 2
    self.embed = nn.Linear(3 * P * P, d)
    self.cls = nn.Parameter(torch.zeros(1, 1, d))
    self.pos = nn.Parameter(torch.randn(1, self.n_tok + 1, d, generator=gen) * 0.02)
    self.ln1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
    self.attn = nn.ModuleList([nn.MultiheadAttention(d, cfg.vit_heads, batch_first=True) for _ in range(L)])
    self.ln2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(L)])
    if cfg.kernel_type in ('cos', 'bm25', 'hybrid', 'parametric_softmax'):
      raise ValueError(f"arch 'vit' supports kernel_type 'standard', 'gws', 'gauge' and 'rbf', not {cfg.kernel_type!r}")
    if cfg.kernel_type == 'standard':
      r = cfg.vit_ffn_blowup
      self.ffn = nn.ModuleList([nn.Sequential(nn.Linear(d, r * d), nn.GELU(), nn.Linear(r * d, d)) for _ in range(L)])
    else:
      self.ffn = nn.ModuleList([RCDLLayer(cfg, d, d, i, gen, device=device) for i in range(L)])
    self.ln_f = nn.LayerNorm(d)
    self.head = nn.Linear(d, cfg.num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    B, P = x.shape[0], self.cfg.vit_patch
    patches = x.view(B, 32, 32, 3).unfold(1, P, P).unfold(2, P, P).reshape(B, self.n_tok, 3 * P * P)
    h = torch.cat([self.cls.expand(B, -1, -1), self.embed(patches)], 1) + self.pos
    for i in range(self.cfg.num_layers):
      a = self.ln1[i](h)
      h = h + self.attn[i](a, a, a, need_weights=False)[0]
      f = self.ln2[i](h)
      h = h + (self.ffn[i](f) if self.cfg.kernel_type == 'standard' else self.ffn[i](f.reshape(-1, f.shape[-1])).view_as(h))
    return self.head(self.ln_f(h[:, 0]))

  def rcdl_layers(self) -> Dict[str, RCDLLayer]:
    return {f'ffn_{i}': m for i, m in enumerate(self.ffn) if isinstance(m, RCDLLayer)}


def build_model(cfg: RCDLConfig, device=None) -> nn.Module:
  """device: where the RCDL layers allocate their tables (the caller still moves the module with .to(device))."""
  return CifarViT(cfg, device) if cfg.arch == 'vit' else CifarRCDLClassifier(cfg, device)
