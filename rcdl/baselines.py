"""Nonparametric single-pass baselines (paper App. E.8) plus kernels matched to the RCDL layers.

Notation: X ∈ R^{n×d} training images (standardized, flattened, d = 3072), Y ∈ {0,1}^{n×C} one-hot labels
(C = 10 classes), x a test image, k(x, x') a kernel, K = [k(x_i, x_j)] the n×n Gram matrix, λ > 0 the ridge.

  krr       kernel ridge regression ("LS-SVM" in the paper): α = (K + λ I)^{-1} Y, f(x) = Σ_i α_i k(x, x_i),
            prediction argmax_c f_c(x).
  nw        Nadaraya-Watson retrieval: f(x) = Σ_i k(x, x_i) y_i / Σ_i k(x, x_i). No learning; the
            prediction is a kernel-weighted vote of the stored labels.
  knn       k nearest neighbours with the paper's similarity / weighting options.

Kernels:
  rbf       k(x, x') = exp(-γ ||x - x'||^2), γ = γ_mult / median_{i≠j} ||x_i - x_j||^2 (median heuristic,
            estimated on a 2048-image subsample), on standardized pixels (paper App. E.8).
  exp       k(x, x') = exp(<q, q'> / (T sqrt(d))) with q = RMSNorm(x): the kernel of the first RCDL layer
            (on the sphere it equals the RCDL 'rbf' kernel up to the constant exp(-sqrt(d)/T)).
"""

import math
from typing import Dict, Iterable

import torch

from rcdl.layers import rms_norm


def kernel_fn(name: str, X: torch.Tensor, gamma_mult: float = 1.0, temperature: float = 6.0, gen=None):
  """Returns k(A, B) -> [len(A), len(B)] for the chosen kernel, with γ set by the median heuristic for 'rbf'."""
  if name == 'rbf':
    g = torch.Generator().manual_seed(0) if gen is None else gen
    sub = X[torch.randperm(len(X), generator=g)[:2048].to(X.device)]
    d2 = torch.cdist(sub, sub).pow(2)
    med = d2[~torch.eye(len(sub), dtype=torch.bool, device=X.device)].median()
    gamma = gamma_mult / med
    return lambda A, B: torch.exp(-gamma * torch.cdist(A, B).pow(2))
  if name == 'exp':
    scale = 1.0 / (temperature * math.sqrt(X.shape[1]))
    return lambda A, B: torch.exp(rms_norm(A) @ rms_norm(B).T * scale)
  raise ValueError(name)


def gram(k, X: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
  """n×n Gram matrix, built in row chunks (float32)."""
  n = len(X)
  K = torch.empty(n, n, device=X.device, dtype=torch.float32)
  for i in range(0, n, chunk):
    K[i:i + chunk] = k(X[i:i + chunk], X)
  return K


def predict(k, X: torch.Tensor, coef: torch.Tensor, Xq: torch.Tensor, chunk: int = 2048, normalize: bool = False):
  """f(x) = Σ_i coef_i k(x, x_i) (normalize=True divides by Σ_i k(x, x_i): Nadaraya-Watson)."""
  out = []
  for i in range(0, len(Xq), chunk):
    Kq = k(Xq[i:i + chunk], X)
    f = Kq @ coef
    out.append(f / Kq.sum(-1, keepdim=True) if normalize else f)
  return torch.cat(out)


def krr_solve(K: torch.Tensor, Y: torch.Tensor, lams: Iterable[float]) -> Dict[float, torch.Tensor]:
  """α(λ) = (K + λI)^{-1} Y for several λ by one Cholesky factorization per λ (float32).

  The diagonal of K is shifted in place, so peak memory is K plus one factor (2 n^2 floats: 20 GB for n = 50,000).
  K is restored on return. A λ whose shifted matrix is not numerically positive definite in float32 is skipped.
  """
  out, shift = {}, 0.0
  diag = K.diagonal()
  for lam in sorted(lams):
    diag.add_(lam - shift)
    shift = lam
    L, info = torch.linalg.cholesky_ex(K)
    if int(info) == 0:
      out[lam] = torch.cholesky_solve(Y, L)
    del L
  diag.sub_(shift)
  return out


@torch.no_grad()
def knn_accuracy(X, y, Xq, yq, ks=(5, 10, 15, 20, 25), chunk: int = 1024) -> Dict[str, float]:
  """Paper App. E.8.2: similarity ∈ {euclidean, dot}, weighting ∈ {uniform, proportional}, k ∈ ks."""
  C = int(y.max()) + 1
  res = {}
  kmax = max(ks)
  for sim in ('euclidean', 'dot'):
    vals, idxs = [], []
    for i in range(0, len(Xq), chunk):
      S = -torch.cdist(Xq[i:i + chunk], X).pow(2) if sim == 'euclidean' else Xq[i:i + chunk] @ X.T
      v, ix = S.topk(kmax, dim=-1)
      vals.append(v)
      idxs.append(ix)
    vals, idxs = torch.cat(vals), torch.cat(idxs)
    for weighting in ('uniform', 'proportional'):
      for kk in ks:
        v, ix = vals[:, :kk], idxs[:, :kk]
        if weighting == 'uniform':
          w = torch.ones_like(v)
        elif sim == 'euclidean':
          w = 1.0 / (torch.sqrt(-v).clamp_min(0) + 1e-5)
        else:
          w = torch.softmax(v, -1)
        votes = torch.zeros(len(Xq), C, device=X.device).scatter_add_(1, y[ix], w)
        res[f'{sim}/{weighting}/k={kk}'] = (votes.argmax(-1) == yq).float().mean().item()
  return res
