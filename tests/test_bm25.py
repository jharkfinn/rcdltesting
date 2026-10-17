"""BM25-RBF layer (rcdl/bm25.py): its scores equal a literal term-by-term Okapi BM25, its rows are sparse with live
document frequencies, its kernel reduces to the RBF kernel when BM25 is replaced by the dot product, and it trains."""

import math

import torch

from rcdl.bm25 import BM25RBFLayer, okapi_idf
from rcdl.layers import RCDLConfig, rbf_attention, rms_norm
from rcdl.models import CifarRCDLClassifier
from rcdl.train import rcdl_evaluate, rcdl_train_step


def tiny_cfg(**kw):
  return RCDLConfig(**(dict(kernel_type='bm25', hidden_dim=8, num_layers=0, num_classes=3, batch_size=4, db_size=40,
                            temperature=2.0, eta=0.5, bm25_m=3, bm25_rotate=False, global_init_seed=1) | kw))


def batches(n, B=4, d=8, C=3, seed=0):
  g = torch.Generator().manual_seed(seed)
  return [(torch.randn(B, d, generator=g) * 2, torch.randint(0, C, (B,), generator=g)) for _ in range(n)]


def test_scores_equal_literal_okapi_bm25():
  cfg = tiny_cfg()
  m = CifarRCDLClassifier(cfg)
  layer = m.output_layer
  docs = []
  for x, y in batches(5):
    f, length = layer.terms(rms_norm(x))
    docs += [(f[i], float(length[i])) for i in range(len(f))]
    rcdl_train_step(m, None, x, y)
  N = len(docs)
  avglen = sum(l for _, l in docs[:4]) / 4                 # fixed at the first stored batch
  assert layer.size == N and abs(layer.avglen - avglen) < 1e-5
  n_t = [sum(1 for f, _ in docs if f[t] > 0) for t in range(16)]
  assert torch.equal(layer.n_t, torch.tensor(n_t, dtype=layer.n_t.dtype))
  q = rms_norm(torch.randn(2, 8, generator=torch.Generator().manual_seed(9)) * 2)
  fq, lq = layer.terms(q)
  k1, b = cfg.bm25_k1, cfg.bm25_b
  ref = torch.zeros(2, N)
  for i in range(2):
    for j, (fj, lj) in enumerate(docs):
      for t in range(16):
        if fq[i, t] > 0 and fj[t] > 0:
          idf = math.log((N - n_t[t] + 0.5) / (n_t[t] + 0.5) + 1)
          ref[i, j] += fq[i, t] * idf * fj[t] * (k1 + 1) / (fj[t] + k1 * (1 - b + b * lj / avglen))
  S, S_self, _, _ = layer.scores(q)
  assert torch.allclose(S, ref.to(S.dtype), rtol=1e-5, atol=1e-6)
  assert (layer.keys[:N] != 0).sum(1).eq(cfg.bm25_m).all()   # each stored row holds exactly m terms


def test_kernel_form_reduces_to_rbf_for_the_dot_product():
  """exp(γ (S(q,k) - S(q,q)) / S(q,q)) with S = <.,.> and ||q||^2 = ||k||^2 = d equals the RBF kernel (γ = sqrt(d)/T)."""
  d, T = 16, 3.0
  g = torch.Generator().manual_seed(2)
  q, k, v = rms_norm(torch.randn(5, d, generator=g)), rms_norm(torch.randn(7, d, generator=g)), torch.randn(7, 2, generator=g)
  gamma = math.sqrt(d) / T
  S, S_self = q @ k.T, (q * q).sum(-1, keepdim=True)
  ours = torch.exp(gamma * (S - S_self) / S_self) @ v
  assert torch.allclose(ours, rbf_attention(q, k, v, T), rtol=1e-5, atol=1e-6)


def test_self_match_and_unmatched_rows():
  """A stored copy of the query gets κ ≈ 1; a row with no shared term gets κ = e^{-γ}."""
  cfg = tiny_cfg(bm25_idf=False)
  layer = CifarRCDLClassifier(cfg).output_layer
  x = rms_norm(torch.randn(4, 8, generator=torch.Generator().manual_seed(3)) * 2)
  f, length = layer.terms(x)
  layer.pending = (f, length, torch.eye(4, 3))
  layer.commit()
  S, S_self, _, _ = layer.scores(x)
  kappa = torch.exp(layer.gamma * (S - S_self.clamp_min(1e-6)) / S_self.clamp_min(1e-6))
  assert torch.allclose(kappa.diagonal(), torch.ones(4), atol=1e-5)
  shared = (f[:, None, :] > 0) & (f[None, :, :] > 0)
  none = ~shared.any(-1)
  if none.any():
    assert torch.allclose(kappa[none], torch.full_like(kappa[none], math.exp(-layer.gamma)))


def test_forward_returns_the_specified_kernel():
  """Read κ through forward(): with values = I, the output row of a query is its kernel row. A stored copy of the query
  gives κ = 1 and a row with no shared term gives e^{-γ} (catches a sign-flipped or γ-less kernel in forward)."""
  cfg = tiny_cfg(bm25_idf=False)
  layer = CifarRCDLClassifier(cfg).output_layer
  x = rms_norm(torch.randn(3, 8, generator=torch.Generator().manual_seed(4)) * 2)
  f, length = layer.terms(x)
  layer.values = torch.zeros(40, 3)
  layer.pending = (f, length, torch.eye(3))
  layer.commit()
  layer.eval()
  with torch.no_grad():
    kappa = layer(x)                                         # B x 3 = κ(x_b, row j) because v_j = e_j
  assert torch.allclose(kappa.diagonal(), torch.ones(3), atol=1e-5)
  none = ~((f[:, None, :] > 0) & (f[None, :, :] > 0)).any(-1)
  assert none.any()
  assert torch.allclose(kappa[none], torch.full_like(kappa[none], math.exp(-layer.gamma)), rtol=1e-5)


def test_quantized_inputs_still_give_exactly_m_terms():
  """Tied |coordinates| (quantized pixels, identity basis) must not add terms beyond m."""
  layer = CifarRCDLClassifier(tiny_cfg(bm25_m=3)).output_layer
  x = torch.tensor([[1.0, 1.0, 1.0, 1.0, -1.0, 0.5, 0.5, 0.0]] * 2)
  f, _ = layer.terms(x)
  assert (f > 0).sum(-1).eq(3).all()


def test_vocabulary_expansion_shapes_and_sparsity():
  cfg = tiny_cfg(bm25_vocab=64, bm25_m=4)
  m = CifarRCDLClassifier(cfg)
  layer = m.output_layer
  assert layer.R.shape == (8, 32) and layer.keys.shape[1] == 64 and layer.n_t.shape == (64,)
  for x, y in batches(3):
    rcdl_train_step(m, None, x, y)
  assert (layer.keys[:layer.size] != 0).sum(1).eq(4).all()
  assert layer.n_t.sum().item() == 4 * layer.size


def test_deep_bm25_trains_and_eval_is_batch_independent():
  cfg = tiny_cfg(hidden_dim=16, num_layers=2, bm25_m=6, bm25_rotate=True, db_size=48)
  m = CifarRCDLClassifier(cfg)
  losses = [rcdl_train_step(m, None, x, y)[0] for x, y in batches(10, d=16)]
  assert all(math.isfinite(l) for l in losses)
  assert all(layer.size == 40 for layer in m.rcdl_layers().values())
  tx, ty = torch.randn(12, 16), torch.randint(0, 3, (12,))
  a, b = rcdl_evaluate(m, tx, ty, 3), rcdl_evaluate(m, tx, ty, 12)
  assert abs(a[0] - b[0]) < 1e-5 and a[1] == b[1]
