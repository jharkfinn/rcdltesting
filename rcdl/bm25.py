"""BM25-RBF: the growing RBF layer with sparse Okapi BM25 key matching in place of the dot product.

With RMS-normalized vectors (||q||^2 = d) the RBF kernel of rcdl.layers.rbf_attention is
    exp(-||q - k||^2 / (2 T sqrt(d))) = exp(γ (<q, k> - <q, q>) / <q, q>),   γ = sqrt(d) / T.
This layer keeps that form and the RBF write rule, and replaces the dot product <q, k> by a BM25 score:
    κ(q, k_j) = exp(γ (S(q, j) - S(q, q)) / S(q, q)),      s(q) = Σ_j κ(q, k_j) v_j,      v_j = -η B ḡ_j,
so a stored copy of the query itself gets κ ≈ 1 and a row that shares no term with the query gets e^{-γ}, the same range as
the RBF. Replacing S by the dot product gives the RBF layer exactly.

Terms. u = R q (R: fixed random orthogonal d x d matrix, the identity, or with bm25_vocab = V a Gaussian d x V/2
expansion with N(0, 1/d) entries), and the m coordinates of largest |u_i| become terms, split by sign: term i+ with frequency
f_{i+} = max(u_i, 0) and term i- with f_{i-} = max(-u_i, 0). So each vector is a sparse "document" of m terms over a
vocabulary of V = 2d (or bm25_vocab) terms, with length len = Σ_t f_t. Text BM25 works at about 0.1-1% of the vocabulary per
document; the expansion (as in the fly hash of Dasgupta et al. 2017) reaches that regime without dropping information.

Okapi BM25 (k1, b):
    S(q, j) = Σ_t w_t(q) IDF(t) f_t(k_j) (k1 + 1) / (f_t(k_j) + k1 (1 - b + b len(k_j) / avglen)),
    IDF(t)  = ln((N - n_t + 0.5) / (n_t + 0.5) + 1),
with N the number of stored rows, n_t the number of stored rows that contain term t (live counts), avglen the mean document
length (fixed at the mean of the first stored batch: RMSNorm keeps lengths stable), and query term weights w_t(q) = f_t(q)
(BM25 with k3 = ∞). The document side (saturated, length-normalized frequencies) is stored at write time; the IDF is applied
on the query side, so the score is one matmul between the dense-emulated sparse query and the stored rows. S(q, q) uses the
query's own document-side vector.

Gradients flow to the query through w(q) and S(q, q) (the top-m selection is piecewise constant).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.layers import RCDLConfig, RCDLLayer, _mm


def okapi_tf(f: torch.Tensor, length: torch.Tensor, avglen: float, k1: float, b: float) -> torch.Tensor:
  """Saturated, length-normalized term frequency f (k1 + 1) / (f + k1 (1 - b + b len / avglen)); 0 where f = 0."""
  return f * (k1 + 1.0) / (f + k1 * (1.0 - b + b * length / avglen))


def okapi_idf(N: int, n_t: torch.Tensor) -> torch.Tensor:
  return torch.log((N - n_t + 0.5) / (n_t + 0.5) + 1.0)


class BM25RBFLayer(RCDLLayer):
  """Growing layer with BM25 key matching (kernel_type 'bm25'); see the module docstring for the definition."""

  def __init__(self, cfg: RCDLConfig, d_in: int, d_out: int, layer_id: int, gen: torch.Generator, device=None):
    nn.Module.__init__(self)
    dev = torch.device('cpu') if device is None else torch.device(device)
    store = torch.bfloat16 if cfg.store_dtype == 'bfloat16' else torch.get_default_dtype()
    self.cfg, self.kind, self.T, self.d = cfg, 'bm25', cfg.temperature, d_in
    half = cfg.bm25_vocab // 2 if cfg.bm25_vocab > 0 else d_in      # projected coordinates; vocabulary V = 2 * half
    self.m = min(cfg.bm25_m, half)
    self.gamma = cfg.bm25_gamma if cfg.bm25_gamma is not None else math.sqrt(d_in) / cfg.temperature
    n = cfg.db_size
    g = torch.Generator().manual_seed(cfg.global_init_seed * 1000 + layer_id)
    if cfg.bm25_vocab > 0:                      # expansion: Gaussian d x half projection (unit-variance outputs)
      self.register_buffer('R', (torch.randn(d_in, half, generator=g) / math.sqrt(d_in)).to(dev))
    elif cfg.bm25_rotate:
      self.register_buffer('R', torch.linalg.qr(torch.randn(d_in, d_in, generator=g))[0].to(dev))
    else:
      self.R = None
    self.register_buffer('keys', torch.zeros(n, 2 * half, dtype=store, device=dev))    # stored document-side tf
    self.register_buffer('values', torch.zeros(n, d_out, dtype=store, device=dev))
    self.register_buffer('n_t', torch.zeros(2 * half, dtype=torch.float64, device=dev))  # document frequencies
    self.avglen = None
    self.size, self.step, self.pending, self.growing, self.linearize, self.wrapped = 0, 0, None, True, False, False
    self.next_row = 0

  def terms(self, q: torch.Tensor):
    """Sparse term frequencies f (B x 2d) of the top-m sign-split coordinates of u = R q, and the lengths (B x 1)."""
    u = q @ self.R if self.R is not None else q
    # exactly m terms: a threshold at the m-th value would keep every tie, and quantized pixels (identity basis) tie often
    sel = torch.zeros_like(u).scatter_(-1, u.detach().abs().topk(self.m, dim=-1).indices, 1.0)
    f = torch.cat([F.relu(u) * sel, F.relu(-u) * sel], dim=-1)
    return f, f.sum(-1, keepdim=True)

  def doc_side(self, f, length):
    cfg = self.cfg
    return okapi_tf(f, length, self.avglen, cfg.bm25_k1, cfg.bm25_b)

  def scores(self, q: torch.Tensor):
    """BM25 scores S(q, j) for the stored rows (B x N) and the self-scores S(q, q) (B x 1)."""
    f, length = self.terms(q)
    N = self.size
    idf = okapi_idf(N, self.n_t).to(q.dtype) if self.cfg.bm25_idf else torch.ones_like(self.n_t, dtype=q.dtype)
    qw = f * idf
    S = _mm(qw, self.keys[:N].T)
    S_self = (qw * self.doc_side(f, length)).sum(-1, keepdim=True)
    return S, S_self, f, length

  def forward(self, q: torch.Tensor) -> torch.Tensor:
    N = self.size
    if N == 0:                                  # empty table: output 0, but kept in the graph (as the RBF layer's empty
      f, length = self.terms(q)                 # matmul) so that the earlier layers still receive a (zero) gradient
      s = torch.zeros(q.shape[0], self.values.shape[1], device=q.device, dtype=q.dtype) + 0.0 * q.sum(-1, keepdim=True)
    else:
      S, S_self, f, length = self.scores(q)
      S_self = S_self.clamp_min(1e-6)
      kappa = torch.exp(self.gamma * (S - S_self) / S_self)
      s = _mm(kappa, self.values[:N])
    if self.training and torch.is_grad_enabled():
      if not s.requires_grad:
        s.requires_grad_(True)
      f_t, len_t = f.detach(), length.detach()
      s.register_hook(lambda g: self._write_bm25(f_t, len_t, g))
    return s

  def _write_bm25(self, f, length, g):
    self.pending = (f, length, -self.cfg.eta * self.cfg.batch_size * g)

  @torch.no_grad()
  def commit(self):
    if self.pending is None:
      return
    f, length, v = self.pending
    self.pending = None
    self.step += 1
    if self.avglen is None:
      self.avglen = float(length.mean())
    lo, hi = self.size, self.size + len(f)
    if hi > len(self.keys):
      raise RuntimeError(f'BM25 table full ({len(self.keys)} rows); raise db_size')
    self.keys[lo:hi] = self.doc_side(f, length).to(self.keys.dtype)
    self.values[lo:hi] = v.to(self.values.dtype)
    self.n_t += (f > 0).sum(0).to(self.n_t.dtype)
    self.size = self.next_row = hi
