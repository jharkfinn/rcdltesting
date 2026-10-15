"""The fused Triton RBF kernels (rcdl/rbf_triton.py) equal the chunked PyTorch path of rcdl/bios.py.

On a GPU the kernels run natively on bf16 tables; without a GPU they run in Triton's CPU interpreter on fp32 tables
(TRITON_INTERPRET=1 must be set before Triton is imported, hence the subprocess)."""

import os
import subprocess
import sys

import pytest

SCRIPT = r'''
import torch
from rcdl.bios import _RBFRead
torch.manual_seed(0)
cuda = torch.cuda.is_available()
dev, dt = ('cuda', torch.bfloat16) if cuda else ('cpu', torch.float32)
# (queries, rows, d, dv, c): row counts that are not multiples of the tile, c small enough for near-zero and near-one κ
for nq, n, d, dv, c in [(37, 1000, 32, 32, 8.0), (130, 2051, 64, 48, 20.0), (5, 0, 32, 32, 8.0)]:
  keys = torch.nn.functional.normalize(torch.randn(n, d, device=dev), dim=-1).mul(d ** 0.5)
  q = keys[:nq].clone() + 0.3 * torch.randn(nq, d, device=dev) if n >= nq else torch.randn(nq, d, device=dev)
  keys, values = keys.to(dt), (0.1 * torch.randn(n, dv, device=dev)).to(dt)
  key_sq = keys.float().pow(2).sum(-1)
  g = torch.randn(nq, dv, device=dev)
  outs, grads = [], []
  for fused in (False, True):
    qq = q.clone().requires_grad_(True)
    s = _RBFRead.apply(qq, keys, values, key_sq, c, 256, fused)
    (s * g).sum().backward()
    outs.append(s.detach()); grads.append(qq.grad)
  tol = 2e-2 if cuda else 1e-4
  for a, b in ((outs[1], outs[0]), (grads[1], grads[0])):
    err = ((a - b).abs().max() / b.abs().max().clamp_min(1e-12)).item()
    assert err < tol, (nq, n, d, dv, err)
print('OK')
'''


def _has_cuda() -> bool:
  import torch
  return torch.cuda.is_available()


def test_fused_rbf_matches_pytorch_path():
  pytest.importorskip('triton')
  env = dict(os.environ)
  if not _has_cuda():
    env['TRITON_INTERPRET'] = '1'
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  r = subprocess.run([sys.executable, '-c', SCRIPT], cwd=root, env=env, capture_output=True, text=True, timeout=1200)
  if r.returncode != 0 and 'only 0-dimensional arrays' in r.stderr:
    pytest.skip('Triton CPU interpreter does not support this NumPy version (int() of a size-1 array)')
  assert r.returncode == 0 and 'OK' in r.stdout, r.stderr[-3000:]
