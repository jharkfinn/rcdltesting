"""The Triton split-key kernel (rcdl/splitk_attention.py) equals the dense weighted softmax readout.

On a GPU the kernel runs natively. Without a GPU the test runs it in Triton's CPU interpreter (TRITON_INTERPRET=1,
which must be set before Triton is imported, hence the subprocess). Skipped when Triton is not installed.
"""

import os
import subprocess
import sys

import pytest

SCRIPT = r'''
import torch
from rcdl.layers import weighted_attention, rms_norm
from rcdl.splitk_attention import weighted_attention_splitk
torch.manual_seed(0)
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
# (rows, queries, d_in, d_out, temperature); T = 0.1 gives logits up to 100, so the chunk maxima must be merged.
for n, nq, d, dout, T in [(1000, 64, 100, 1, 1.0), (777, 37, 100, 1, 0.1), (3000, 70, 64, 10, 3.0)]:
  q = rms_norm(torch.randn(nq, d, device=dev))
  k = rms_norm(torch.randn(n, d, device=dev))
  v = torch.randn(n, dout, device=dev) * 0.1
  w = torch.rand(n, 1, device=dev)
  ref = weighted_attention(q, k, v, w, T)
  for precision in ['ieee', 'tf32x3']:
    out = weighted_attention_splitk(q, k, v, w, T, precision=precision, programs_per_sm=2)
    err = ((out - ref).abs().max() / ref.abs().max()).item()   # T = 0.1 gives large outputs: compare relative error
    assert err < 1e-5, (n, nq, d, dout, T, precision, err)
print('OK')
'''


def test_splitk_matches_dense():
  pytest.importorskip('triton')
  env = dict(os.environ)
  if not _has_cuda():
    env['TRITON_INTERPRET'] = '1'
  root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
  r = subprocess.run([sys.executable, '-c', SCRIPT], cwd=root, env=env, capture_output=True, text=True, timeout=900)
  if r.returncode != 0 and 'only 0-dimensional arrays' in r.stderr:
    pytest.skip('Triton CPU interpreter does not support this NumPy version (int() of a size-1 array)')
  assert r.returncode == 0 and 'OK' in r.stdout, r.stderr[-3000:]


def _has_cuda() -> bool:
  import torch
  return torch.cuda.is_available()


def test_flash2_matches_dense_on_gpu():
  """attn_impl='flash2' (two bf16 FlashAttention calls) equals the dense weighted softmax up to bf16 rounding,
  including the gradient with respect to the queries."""
  import torch
  if not torch.cuda.is_available():
    pytest.skip('FlashAttention needs a CUDA GPU')
  from rcdl.layers import weighted_attention, weighted_attention_flash, rms_norm
  torch.manual_seed(0)
  n, nq, d = 5000, 300, 128
  q = rms_norm(torch.randn(nq, d, device='cuda')).requires_grad_(True)
  k, v, w = rms_norm(torch.randn(n, d, device='cuda')), torch.randn(n, d, device='cuda') * 0.1, torch.rand(n, 1, device='cuda')
  ref = weighted_attention(q, k, v, w, 1.0)
  g_ref, = torch.autograd.grad(ref.sum(), q)
  out = weighted_attention_flash(q, k, v, w, 1.0)
  g_out, = torch.autograd.grad(out.sum(), q)
  rel = lambda a, b: ((a - b).norm() / b.norm()).item()
  assert rel(out, ref) < 2e-2 and rel(g_out, g_ref) < 5e-2, (rel(out, ref), rel(g_out, g_ref))
