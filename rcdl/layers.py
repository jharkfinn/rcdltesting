"""RCDL layers: ported from rcdl_tutorial_torch.ipynb (Section 3.1), with the same math and update rules.

Differences from the notebook, all of which keep the default 1-epoch configurations numerically equivalent
(verified in tests/test_layers.py):
  * Retrieval runs over the active rows [:size] only (the notebook scans the full pre-allocated table and masks
    empty rows with -1e9 logits). Valid-row masks of the growing layers are cached at commit time.
  * Age-based momentum / weight decay uses a per-row birth step instead of the notebook's roll-based mask. The
    notebook's mask equals this rule only when the forward batch equals the training batch; it silently uses wrong
    ages at evaluation time (eval batch 500 != train batch 16). This port fixes that bug.
  * Optional knobs for extensions (defaults reproduce the notebook): `delta_clip`, `age_rule`, `decay_scope`.
"""

import dataclasses as dc
import math
from typing import Optional

import torch
import torch.nn as nn


@dc.dataclass(frozen=True)
class RCDLConfig:
  """Unified configuration for 'gws', 'gauge', 'rbf', 'hybrid', 'parametric_softmax', and 'standard'."""
  # --- Architecture ---
  kernel_type: str = 'gws'           # 'gws' | 'gauge' | 'rbf' | 'hybrid' | 'parametric_softmax' | 'standard'
  hidden_dim: int = 3072
  num_layers: int = 2
  nonlinearity: str = 'none'         # 'none' | 'gelu' | 'relu'
  use_residual: bool = True
  # --- Attention & Database (paper notation in comments) ---
  temperature: float = 10.0          # T (softmax temperature; RBF temperature for 'rbf')
  db_size: int = 50016               # N: max. database size per layer = 50,000 (1 epoch) + 16 (D_0)
  full_epoch_capacity: int = 50016   # Reference 1-epoch capacity for % calculation
  # --- Growing layers ('gws', 'gauge', 'rbf', 'hybrid') ---
  eta: float = 0.03                  # η: weight learning rate ('gws'/'gauge'/'hybrid'); value learning rate for 'rbf'
  beta: float = 30.0                 # β: value learning rate ('gws'/'gauge'/'hybrid'); unused for 'rbf'
  model_momentum: float = 0.0        # μ: age-based momentum of the stored values (0 = off)
  model_weight_decay: float = 0.0    # γ: age-based decay of the stored values (0 = off)
  init_pred_scale: float = 20.0      # 'rbf': scale of the frozen random projection (only if freeze_init_out=False)
  add_activations: bool = False      # 'rbf': also store the current prediction in the new value
  freeze_init_out: bool = True       # 'rbf': True = no frozen random projection x W_slow
  # --- Gauge-Invariant Weighted Softmax ('gauge') ---
  gauge_alpha: float = 1.0           # α: gauge parameter (α_t = α / ||g_t||)
  # --- Parametric Softmax ('parametric_softmax') Specific ---
  learn_parametric_weights: bool = True
  # --- Hybrid ('hybrid') Growth Control ---
  growth_threshold: float = 50.0     # τ: growth threshold on min_j ||q - k_j||_2
  # --- Data & Optimization ---
  num_classes: int = 10
  batch_size: int = 16
  optimizer: str = 'adam'            # 'adam' | 'adamw' | 'sgd' | 'rmsprop' | 'muon' | 'mesa' | 'mesa_sgd' | 'delta'
  weight_decay: float = 0.0
  learning_rate: float = 3e-4        # optimizer (e.g. Adam) learning rate for parametric parameters
  mesa_gamma: float = 0.99           # Mesa / FOOF forgetting factor γ (paper grid [0.9, 0.99, 1.0])
  mesa_lambda: float = 0.01          # Mesa / FOOF ridge λ (paper grid [0.001, 0.01, 0.1])
  delta_lambda: float = 1e-5         # DeltaNet decay rate λ (paper grid [3e-7 ... 3e-4])
  num_epochs: int = 1
  global_init_seed: int = 1
  # --- Extension knobs (defaults = notebook behaviour) ---
  delta_clip: Optional[float] = 1.0  # 'gws'/'hybrid': clip of <s, ḡ> (= δ_t / B); None = no clip
  age_rule: str = 'notebook'         # 'notebook': (1-γ)^a (1-μ^(a+1))/(1-μ+1e-6) | 'exact': paper Eq. (combined)
  decay_scope: str = 'values'        # 'values' (notebook) | 'values_and_weights' (paper App. D)
  augment: bool = False              # multi-epoch runs: paper App. E.1.1 augmentation
  attn_impl: str = 'dense'           # 'dense' (notebook) | 'sdpa' | 'sdpa_bf16' (fused) | 'flash2' (2 flash calls,
                                     # bf16, value width = query width) | 'splitk' | 'splitk_tf32'
                                     # (Triton, few queries, forward-only: 'dense' is used when q needs a gradient)
  store_dtype: str = 'float32'       # 'float32' | 'bfloat16': stored keys and values of 'gws'/'gauge'/'rbf' (weights
                                     # stay fp32); bf16 halves the memory (L = 2, 50 epochs: 77 GB instead of 154 GB)
  dataset: str = 'cifar10'           # 'cifar10' | 'cifar100'
  # --- ViT (paper Sec. 5 'Mixing retrieval-centric and parametric layers', App. E.9) ---
  arch: str = 'resmlp'               # 'resmlp' | 'vit' (num_layers = ViT blocks, hidden_dim = token width)
  vit_patch: int = 4
  vit_heads: int = 8
  vit_ffn_blowup: int = 4            # dense FFN width = blowup x hidden_dim (kernel_type 'standard')

  @property
  def ref_capacity(self) -> int:
    """Reference 1-epoch capacity per layer for % reporting."""
    return self.full_epoch_capacity if self.kernel_type == 'parametric_softmax' else self.db_size


def rms_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
  """RMSNorm without learned scale: the backbone's pre-norm, q = RMSNorm(x)."""
  return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def weighted_attention(q, keys, values, weights, temperature: float, db_size: Optional[int] = None,
                       valid: Optional[torch.Tensor] = None):
  """Weighted softmax attention over the database entries: s_D(q) = Σ_j exp(z_j) ṽ_j / Σ_m w_m exp(z_m).

  z_j = <q, k_j> / (T sqrt(d_in)); the weights enter the denominator only. Empty entries (k_j = 0, w_j = 0) and,
  for 'hybrid', entries beyond the active database size are masked out. `valid` (optional) is a precomputed row mask.
  """
  logits = q @ keys.T / (temperature * math.sqrt(q.shape[-1]))
  if valid is None:
    valid = (keys != 0).any(-1) | (weights[:, 0] != 0)
  if db_size is not None:
    valid = valid & (torch.arange(keys.shape[0], device=q.device) < db_size)
  logits = torch.where(valid[None, :], logits, torch.full_like(logits, -1e9))
  exp_qk = torch.exp(logits - logits.max(-1, keepdim=True).values)
  denom = exp_qk @ weights
  sign = torch.where(denom < 0, -torch.ones_like(denom), torch.ones_like(denom))
  return (exp_qk @ values) / (denom.abs().clamp_min(1e-8) * sign)


def _mm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
  """a @ b in the dtype of b (bf16 storage: bf16 tensor-core matmul with fp32 accumulation), result in a's dtype."""
  return a @ b if a.dtype == b.dtype else (a.to(b.dtype) @ b).to(a.dtype)


def weighted_attention_lowp(q, keys, values, weights, temperature: float, valid: Optional[torch.Tensor] = None):
  """weighted_attention for keys and values stored in bf16 (cfg.store_dtype): the two large matmuls run in bf16 with
  fp32 accumulation; the logits, the softmax shift, the exponentials and the denominator stay in fp32.

  Error: rounding q and k_j to bf16 (relative 2^-9 per element) changes z_j = <q, k_j> / (T sqrt(d)) by about
  2^-9 ||q * k_j||_2 / (T sqrt(d)) ~ 1e-4 for unit-RMS q, k_j at d = 3072, T = 10; rounding exp(z_j) for the numerator
  adds a relative error of at most 2^-9 per row. tests/test_layers.py bounds the end-to-end difference.
  """
  logits = _mm(q, keys.T) / (temperature * math.sqrt(q.shape[-1]))
  if valid is not None:
    logits = logits.masked_fill(~valid[None, :], -1e9)
  exp_qk = torch.exp(logits - logits.max(-1, keepdim=True).values)
  denom = exp_qk @ weights
  sign = torch.where(denom < 0, -torch.ones_like(denom), torch.ones_like(denom))
  return _mm(exp_qk, values) / (denom.abs().clamp_min(1e-8) * sign)


def weighted_attention_sdpa(q, keys, values, weights, temperature: float, dtype=None):
  """Memory-efficient weighted softmax attention via fused scaled-dot-product attention.

  With z_j = <q, k_j> / (T sqrt(d_in)) and softmax over all rows, s_D(q) = Σ_j e^{z_j} ṽ_j / Σ_m w_m e^{z_m}
  = (softmax(z) @ ṽ) / (softmax(z) @ w). One fused call over the columns [ṽ, w] gives both terms without building the
  query x rows logit matrix. Empty rows (k = 0, ṽ = 0, w = 0) add nothing to either term, so no mask is needed.
  dtype=torch.bfloat16 runs the fused kernel on bf16 inputs (fp32 accumulation); the output has the dtype of q.
  """
  d_out = values.shape[1]
  V = torch.cat([values, weights], -1)
  pad = (-V.shape[1]) % 8
  if pad:
    V = torch.nn.functional.pad(V, (0, pad))
  qq, kk = q, keys
  if dtype is not None:
    qq, kk, V = q.to(dtype), keys.to(dtype), V.to(dtype)
  out = torch.nn.functional.scaled_dot_product_attention(
      qq[None, None], kk[None, None], V[None, None], scale=1.0 / (temperature * math.sqrt(q.shape[-1])))[0, 0].to(q.dtype)
  denom = out[:, d_out:d_out + 1]
  sign = torch.where(denom < 0, -torch.ones_like(denom), torch.ones_like(denom))
  return out[:, :d_out] / (denom.abs().clamp_min(1e-30) * sign)


def weighted_attention_flash(q, keys, values, weights, temperature: float):
  """Exact weighted softmax with two FlashAttention calls (bf16): s = (P ṽ) / (P w), P = softmax over all rows.

  Both calls use the same softmax P, so the softmax normalizer cancels in the ratio and autograd gives the exact
  gradient. FlashAttention needs equal head widths for q, k and v, so ṽ must have the width of q (true for the ViT FFN)
  and w is padded with zero columns. Reason for this path: PyTorch's memory-efficient attention backward slows down
  about 50 times at 2^21 keys (measured on L4 and A100, torch 2.11: 502 ms at 2.09M keys, 25.4 s at 2.10M keys for
  4,160 bf16 queries of width 128), which stalled the GWS-ViT runs. Two flash calls take 164 ms at 2.1M keys.
  """
  from torch.nn.attention import SDPBackend, sdpa_kernel
  d = q.shape[-1]
  if values.shape[1] != d:
    raise ValueError(f'flash2 needs value width = query width ({values.shape[1]} != {d})')
  scale = 1.0 / (temperature * math.sqrt(d))
  qq, kk = q.to(torch.bfloat16)[None, None], keys.to(torch.bfloat16)[None, None]
  W = torch.nn.functional.pad(weights, (0, d - weights.shape[1])).to(torch.bfloat16)
  with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    num = torch.nn.functional.scaled_dot_product_attention(qq, kk, values.to(torch.bfloat16)[None, None], scale=scale)
    den = torch.nn.functional.scaled_dot_product_attention(qq, kk, W[None, None], scale=scale)
  num, den = num[0, 0].float(), den[0, 0, :, :1].float()
  sign = torch.where(den < 0, -torch.ones_like(den), torch.ones_like(den))
  return (num / (den.abs().clamp_min(1e-30) * sign)).to(q.dtype)


def linearized_weighted_attention(q, keys, values, weights, temperature: float):
  """Paper Fig. 4 (right), post-training ablation: exp(z) replaced by its first-order expansion 1 + z."""
  lin = 1.0 + _mm(q, keys.T) / (temperature * math.sqrt(q.shape[-1]))
  return _mm(lin, values) / (lin @ weights)


def rbf_attention(q, keys, values, temperature: float):
  """Unnormalized Gaussian RBF kernel attention: s_D(q) = Σ_j exp(-||q - k_j||^2 / (2 T sqrt(d_in))) v_j."""
  dist_sq = (q.pow(2).sum(-1, keepdim=True) + keys.pow(2).sum(-1) - 2.0 * q @ keys.T).clamp_min(0.0)
  return torch.exp(-dist_sq / (2.0 * temperature * math.sqrt(q.shape[-1]))) @ values


def age_coefficients(birth: torch.Tensor, step: int, momentum: float, weight_decay: float, rule: str) -> torch.Tensor:
  """Per-row multiplier of a stored value written at step `birth` (D_0 rows: birth 0), read after `step` commits.

  With age a = step - birth, rule 'notebook' reproduces the notebook's decay_mask, (1-γ)^a (1-μ^(a+1)) / (1-μ+1e-6),
  and rule 'exact' is the paper's closed form for SGD with momentum and decoupled weight decay (App. D.3):
  (1-γ)^a Σ_{m=0}^{a} (μ/(1-γ))^m.
  """
  age = (step - birth).to(torch.float64)
  if rule == 'notebook':
    coef = (1.0 - weight_decay) ** age if weight_decay > 0.0 else torch.ones_like(age)
    if momentum > 0.0:
      coef = coef * (1.0 - momentum ** (age + 1.0)) / (1.0 - momentum + 1e-6)
  elif rule == 'exact':
    mu_t = momentum / (1.0 - weight_decay)
    geo = (age + 1.0) if mu_t == 1.0 else (1.0 - mu_t ** (age + 1.0)) / (1.0 - mu_t)
    coef = (1.0 - weight_decay) ** age * geo
  else:
    raise ValueError(f'unknown age_rule {rule!r}')
  return coef.float()[:, None]


class RCDLLayer(nn.Module):
  """Unified PyTorch module for 'gws', 'gauge', 'rbf', 'hybrid', and 'parametric_softmax' RCDL layers.

  The database (keys k_j, weighted values ṽ_j, weights w_j) consists of buffers for the growing layers ('gws',
  'gauge', 'rbf') and of nn.Parameters refined by the optimizer for 'hybrid' and 'parametric_softmax'. During
  training, a hook on the gradient of the layer output computes the new tuples; commit() appends them to the
  database at the end of the step.
  """

  def __init__(self, cfg: RCDLConfig, d_in: int, d_out: int, layer_id: int, gen: torch.Generator,
               device: Optional[torch.device] = None):
    """device: where to allocate the database (default CPU). Allocating a large table directly on the GPU avoids a
    CPU copy of it (L = 2 for 50 epochs: 77 GB in bf16). Random initial values are always drawn on the CPU from
    `gen`, so the device does not change them."""
    super().__init__()
    self.cfg, self.kind, self.T = cfg, cfg.kernel_type, cfg.temperature
    n, B = cfg.db_size, cfg.batch_size
    dev = torch.device('cpu') if device is None else torch.device(device)
    store = (torch.bfloat16 if cfg.store_dtype == 'bfloat16' and self.kind in ('gws', 'gauge', 'rbf')
             else torch.get_default_dtype())
    if cfg.store_dtype not in ('float32', 'bfloat16'):
      raise ValueError(f'unknown store_dtype {cfg.store_dtype!r}')
    if store == torch.bfloat16 and (self.kind == 'rbf' or cfg.model_momentum > 0.0 or cfg.model_weight_decay > 0.0):
      raise ValueError('store_dtype bfloat16 supports gws/gauge without momentum or weight decay only '
                       '(RBF distances and age coefficients would make fp32 copies of the whole table)')
    keys = torch.zeros(n, d_in, dtype=store, device=dev)
    values = torch.zeros(n, d_out, dtype=store, device=dev)
    weights = torch.zeros(n, 1, device=dev)
    if self.kind == 'parametric_softmax':   # fixed-size random table, every entry active from the start
      keys = (torch.randn(n, d_in, generator=gen) / math.sqrt(d_in)).to(dev)
      values = (torch.randn(n, d_out, generator=gen) * 0.02).to(dev)
      weights = torch.ones(n, 1, device=dev)
      self.size = n
    elif self.kind == 'rbf':                # empty database (+ optional frozen random projection x W_slow)
      self.size = 0
      if not cfg.freeze_init_out:
        g = torch.Generator().manual_seed(layer_id * 42)
        self.register_buffer('w_slow', (torch.randn(d_in, d_out, generator=g)
                                        / (math.sqrt(d_in) * cfg.init_pred_scale)).to(dev))
    else:                                   # D_0: B random RMS-normalized entries with weight 1/B
      keys[:B] = rms_norm(torch.randn(B, d_in, generator=gen)).to(dev)
      values[:B] = rms_norm(torch.randn(B, d_out, generator=gen)).to(dev)
      weights[:B] = 1.0 / B
      self.size = B
    learned = {'hybrid': ('keys', 'values', 'weights'),
               'parametric_softmax': ('keys', 'values') + (('weights',) if cfg.learn_parametric_weights else ())}
    for name, t in (('keys', keys), ('values', values), ('weights', weights)):
      if name in learned.get(self.kind, ()):
        setattr(self, name, nn.Parameter(t))
      else:
        self.register_buffer(name, t)
    self.growing = self.kind in ('gws', 'gauge', 'rbf')
    self.register_buffer('birth', torch.zeros(n, dtype=torch.long, device=dev))      # write step of each row (D_0: 0)
    self.register_buffer('valid_rows', torch.zeros(n, dtype=torch.bool, device=dev))  # cached mask (growing layers)
    self.valid_rows[:self.size] = (keys[:self.size] != 0).any(-1) | (weights[:self.size, 0] != 0)
    self.step, self.wrapped, self.linearize = 0, False, False   # linearize: eval-time exp(z) -> 1 + z (Fig. 4 right)
    self.next_row, self.pending = self.size, None

  def _active(self) -> int:
    """Number of leading rows that can be non-empty (full table once the ring buffer wrapped)."""
    return len(self.keys) if self.wrapped else self.size

  def forward(self, q: torch.Tensor) -> torch.Tensor:
    cfg = self.cfg
    if self.kind == 'parametric_softmax':   # cached mask: random keys never become all-zero rows, so every row stays
      return weighted_attention(q, self.keys, self.values, self.weights, self.T, valid=self.valid_rows)  # valid
    if self.kind == 'hybrid':
      m = self.size
      s = weighted_attention(q, self.keys[:m], self.values[:m], self.weights[:m], self.T)
    else:
      m = self._active()
      keys, values, weights = self.keys[:m], self.values[:m], self.weights[:m]
      if cfg.model_momentum > 0.0 or cfg.model_weight_decay > 0.0:
        coef = age_coefficients(self.birth[:m], self.step, cfg.model_momentum, cfg.model_weight_decay, cfg.age_rule)
        values = values * coef
        if cfg.decay_scope == 'values_and_weights' and self.kind != 'rbf':
          weights = weights * coef
      if self.kind == 'rbf':
        s = rbf_attention(q, keys, values, self.T)
      elif self.linearize and not self.training:
        s = linearized_weighted_attention(q, keys, values, weights, self.T)
      elif keys.dtype != q.dtype:            # bf16 storage (cfg.store_dtype)
        s = weighted_attention_lowp(q, keys, values, weights, self.T, valid=self.valid_rows[:m])
      elif cfg.attn_impl in ('splitk', 'splitk_tf32') and not q.requires_grad:
        from rcdl.splitk_attention import weighted_attention_splitk
        s = weighted_attention_splitk(q, keys, values, weights, self.T,
                                      precision='tf32' if cfg.attn_impl == 'splitk_tf32' else 'tf32x3')
      elif cfg.attn_impl == 'flash2':
        s = weighted_attention_flash(q, keys, values, weights, self.T)
      elif cfg.attn_impl in ('sdpa', 'sdpa_bf16'):
        s = weighted_attention_sdpa(q, keys, values, weights, self.T,
                                    torch.bfloat16 if cfg.attn_impl == 'sdpa_bf16' else None)
      else:
        s = weighted_attention(q, keys, values, weights, self.T, valid=self.valid_rows[:m])
    if self.training and torch.is_grad_enabled():
      if self.kind == 'hybrid':
        self._gate(q)
      if not s.requires_grad:                # first layer: no input gradient needed, only the output gradient
        s.requires_grad_(True)
      q_t, s_t = q.detach(), s.detach()
      s.register_hook(lambda g: self._write(q_t, s_t, g))   # g = gradient of the batch-mean loss w.r.t. s
    return s + q @ self.w_slow if hasattr(self, 'w_slow') else s

  @torch.no_grad()
  def _gate(self, q: torch.Tensor):
    """'hybrid': a query is novel if its distance to every stored key exceeds τ; novel queries grow the database."""
    keys = self.keys[:self.size]
    dists = (q.pow(2).sum(-1, keepdim=True) + keys.pow(2).sum(-1) - 2.0 * q @ keys.T).clamp_min(0.0).sqrt()
    nearest = dists.min(-1).values
    self.novel, self.old_size = nearest > self.cfg.growth_threshold, self.size
    self.size = min(len(self.keys), self.size + int(self.novel.sum()))

  def _write(self, q: torch.Tensor, s: torch.Tensor, g: torch.Tensor):
    """Backward hook: the new tuples (k_t, ṽ_t, w_t) from the query q_t, the prediction s_{t-1} and ḡ_t = g_t / B."""
    cfg, B = self.cfg, self.cfg.batch_size
    if self.kind == 'rbf':                  # v_t = -η g_t
      v = -cfg.eta * B * g
      if cfg.add_activations:
        v = v + s
      w = torch.zeros_like(g[:, :1])
    elif self.kind == 'gauge':              # gauge-invariant step (in units of the batch-mean gradient)
      norm_g, delta = g.norm(dim=-1, keepdim=True), (s * g).sum(-1, keepdim=True)
      w = cfg.eta * cfg.gauge_alpha * norm_g
      v = (-(cfg.eta * cfg.gauge_alpha ** 2 + cfg.beta) + cfg.eta * cfg.gauge_alpha * delta / (norm_g + 1e-8)) * g
    else:                                   # ṽ_t = -β g_t,  w_t = max(0, η clip(δ_t, -B, B))
      v = -cfg.beta * B * g
      delta_over_b = (s * g).sum(-1, keepdim=True)
      if cfg.delta_clip is not None:
        delta_over_b = delta_over_b.clamp(-cfg.delta_clip, cfg.delta_clip)
      w = (cfg.eta * B * delta_over_b).clamp_min(0.0)
    self.pending = (q, v, w)

  @torch.no_grad()
  def commit(self):
    """Appends the pending tuples to the database (once per step, after the optimizer step)."""
    if self.pending is None:
      return
    k, v, w = self.pending
    self.pending = None
    self.step += 1
    if self.kind == 'hybrid':               # only the novel queries, packed behind the stored entries
      sel = self.novel.nonzero().squeeze(1)[:self.size - self.old_size]
      idx = self.old_size + torch.arange(len(sel), device=k.device)
      k, v, w = k[sel], v[sel], w[sel]
    else:                                   # one entry per sample (ring buffer)
      idx = (torch.arange(len(k), device=k.device) + self.next_row) % len(self.keys)
      if self.next_row + len(k) > len(self.keys):
        self.wrapped = True
      self.next_row = (self.next_row + len(k)) % len(self.keys)
      self.size = min(len(self.keys), self.size + len(k))
    self.keys[idx], self.values[idx], self.weights[idx] = k.to(self.keys.dtype), v.to(self.values.dtype), w
    self.birth[idx] = self.step
    self.valid_rows[idx] = (k != 0).any(-1) | (w[:, 0] != 0)
