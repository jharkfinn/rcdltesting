"""experiments/deltanet_long.py (claim 18): the stacked trainer applies the paper's Delta-rule update
W_t = W_{t-1} (I - λ η X X^T) - η G X^T to every linear layer, and the K stacked models do not interact."""

import torch
import torch.nn.functional as F

from experiments.deltanet_long import StackedResMLP, gpu_augment, train_step


def small(K, seed=0):
  return StackedResMLP(K, d_in=12, d=8, hidden=16, blocks=2, n_cls=3, dropout=0.0, seed=seed)


def test_delta_step_matches_paper_formula():
  torch.manual_seed(0)
  m = small(2)
  params = m.params()
  x, y = torch.randn(5, 12), torch.tensor([0, 1, 2, 1, 0])
  lam, lr = torch.tensor([0.0, 0.05]), torch.tensor(0.1)
  # reference: gradients G = dL/dY of each linear output, inputs X, from a separate autograd pass
  record = []
  out = m.forward(x, train=True, record=record)
  for _, _, y0 in record:
    y0.retain_grad()
  F.cross_entropy(out.reshape(-1, 3), y.repeat(2), reduction='none').reshape(2, 5).mean(1).sum().backward()
  expect = []
  for W, X, Y0 in record:
    Xk = X.detach().expand(2, *X.shape) if X.dim() == 2 else X.detach()
    G = Y0.grad                                                          # (K, B, d_out)
    Wd = W.detach()
    eye = torch.eye(Wd.shape[2])
    expect.append(torch.stack([Wd[k] @ (eye - lam[k] * lr * Xk[k].T @ Xk[k]) - lr * G[k].T @ Xk[k] for k in range(2)]))
  for q in params:
    q.grad = None
  train_step(m, params, x, y, lam, lr)
  for (W, _, _), e in zip(record, expect):
    assert torch.allclose(W.detach(), e, atol=1e-6, rtol=1e-5)


def test_stacked_models_are_independent():
  """Model k of a K-stack follows the same trajectory as the same model trained alone."""
  torch.manual_seed(0)
  both, alone = small(2, seed=3), small(2, seed=3)
  x, y = torch.randn(5, 12), torch.tensor([0, 1, 2, 1, 0])
  for _ in range(3):
    for mdl, lam in ((both, torch.tensor([0.0, 0.05])), (alone, torch.tensor([0.0, 0.0]))):
      for q in mdl.params():
        q.grad = None
      train_step(mdl, mdl.params(), x, y, lam, torch.tensor(0.1))
  for a, b in zip(both.params(), alone.params()):
    assert torch.allclose(a[0], b[0], atol=1e-6)                          # λ = 0 model unaffected by its neighbour
  assert not torch.allclose(both.W_in[1], alone.W_in[1])                  # the λ > 0 model differs


def test_gpu_augment_matches_numpy_rule():
  """Crops are windows of the reflect-padded image; flips mirror the width axis."""
  u8 = torch.randint(0, 255, (6, 32, 32, 3), dtype=torch.uint8)
  padded = F.pad(u8.permute(0, 3, 1, 2).float(), (4, 4, 4, 4), mode='reflect').permute(0, 2, 3, 1).to(torch.uint8)
  out = gpu_augment(padded, torch.Generator().manual_seed(1))
  for i in range(6):
    windows = [padded[i, oy:oy + 32, ox:ox + 32] for oy in range(9) for ox in range(9)]
    assert any(torch.equal(out[i], w) or torch.equal(out[i], w.flip(1)) for w in windows)
