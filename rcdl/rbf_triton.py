"""Fused Triton kernels for the growing RBF read of rcdl/bios.py (forward and query gradient).

With κ_ij = exp(-||q_i - k_j||^2 / c) and a fixed table (k_j, v_j), j < N:
  forward   s_i = Σ_j κ_ij v_j
  backward  dq_i = (2/c) (Σ_j A_ij k_j - (Σ_j A_ij) q_i),  A_ij = κ_ij <g_i, v_j>,  g_i = dL/ds_i.
The PyTorch path (bios._RBFRead) builds the Q x chunk matrix of κ in fp32 for every chunk of rows and reads it
several times (distance, exp, cast, matmul; in the backward also <g, v> and the product). Here each program takes a
block of queries and a slice of the rows, keeps the tiles of κ (and A) in registers, and adds its partial sums to the
output with atomic adds. κ <= 1 needs no running maximum, so partial sums from different slices simply add.

Precision: q, k, v, g are bf16 (k, v as stored); the dot products and the sums accumulate in fp32; ||q||^2 and ||k||^2
are fp32 norms of the rounded vectors, so a stored copy of a query has distance 0. κ and A are rounded to bf16 before
their matmuls (relative error 2^-9 per term). Atomic adds make the summation order, and so the last bits, vary
between runs.
"""

import torch

try:
  import triton
  import triton.language as tl
  HAVE_TRITON = True
except ImportError:
  HAVE_TRITON = False


if HAVE_TRITON:

  @triton.jit
  def _rbf_fwd_kernel(Q, QSQ, K, KSQ, V, OUT, n_q, n_k, d, dv, rows_per_prog, inv_c,
                      BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
                      PRECISION: tl.constexpr):
    pid_q, pid_c = tl.program_id(0), tl.program_id(1)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d, offs_v = tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_V)
    qmask = offs_q < n_q
    q = tl.load(Q + offs_q[:, None] * d + offs_d[None, :], mask=qmask[:, None] & (offs_d[None, :] < d), other=0.0)
    q_sq = tl.load(QSQ + offs_q, mask=qmask, other=0.0)
    acc = tl.zeros([BLOCK_Q, BLOCK_V], tl.float32)
    start = pid_c * rows_per_prog
    for it in range(0, rows_per_prog // BLOCK_K):
      offs_k = start + it * BLOCK_K + tl.arange(0, BLOCK_K)
      kmask = offs_k < n_k
      k = tl.load(K + offs_k[:, None].to(tl.int64) * d + offs_d[None, :], mask=kmask[:, None] & (offs_d[None, :] < d),
                  other=0.0)
      k_sq = tl.load(KSQ + offs_k, mask=kmask, other=0.0)
      d2 = tl.maximum(q_sq[:, None] + k_sq[None, :] - 2.0 * tl.dot(q, tl.trans(k), input_precision=PRECISION), 0.0)
      kappa = tl.where(kmask[None, :], tl.exp(-d2 * inv_c), 0.0)
      v = tl.load(V + offs_k[:, None].to(tl.int64) * dv + offs_v[None, :], mask=kmask[:, None] & (offs_v[None, :] < dv),
                  other=0.0)
      acc += tl.dot(kappa.to(v.dtype), v, input_precision=PRECISION)
    tl.atomic_add(OUT + offs_q[:, None] * dv + offs_v[None, :], acc, mask=qmask[:, None] & (offs_v[None, :] < dv))

  @triton.jit
  def _rbf_bwd_kernel(Q, QSQ, G, K, KSQ, V, DQ, RS, n_q, n_k, d, dv, rows_per_prog, inv_c,
                      BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
                      PRECISION: tl.constexpr):
    pid_q, pid_c = tl.program_id(0), tl.program_id(1)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d, offs_v = tl.arange(0, BLOCK_D), tl.arange(0, BLOCK_V)
    qmask = offs_q < n_q
    q = tl.load(Q + offs_q[:, None] * d + offs_d[None, :], mask=qmask[:, None] & (offs_d[None, :] < d), other=0.0)
    g = tl.load(G + offs_q[:, None] * dv + offs_v[None, :], mask=qmask[:, None] & (offs_v[None, :] < dv), other=0.0)
    q_sq = tl.load(QSQ + offs_q, mask=qmask, other=0.0)
    acc = tl.zeros([BLOCK_Q, BLOCK_D], tl.float32)
    rs = tl.zeros([BLOCK_Q], tl.float32)
    start = pid_c * rows_per_prog
    for it in range(0, rows_per_prog // BLOCK_K):
      offs_k = start + it * BLOCK_K + tl.arange(0, BLOCK_K)
      kmask = offs_k < n_k
      k = tl.load(K + offs_k[:, None].to(tl.int64) * d + offs_d[None, :], mask=kmask[:, None] & (offs_d[None, :] < d),
                  other=0.0)
      k_sq = tl.load(KSQ + offs_k, mask=kmask, other=0.0)
      v = tl.load(V + offs_k[:, None].to(tl.int64) * dv + offs_v[None, :], mask=kmask[:, None] & (offs_v[None, :] < dv),
                  other=0.0)
      d2 = tl.maximum(q_sq[:, None] + k_sq[None, :] - 2.0 * tl.dot(q, tl.trans(k), input_precision=PRECISION), 0.0)
      a = tl.where(kmask[None, :], tl.exp(-d2 * inv_c), 0.0) * tl.dot(g, tl.trans(v), input_precision=PRECISION)
      acc += tl.dot(a.to(k.dtype), k, input_precision=PRECISION)
      rs += tl.sum(a, 1)
    tl.atomic_add(DQ + offs_q[:, None] * d + offs_d[None, :], acc, mask=qmask[:, None] & (offs_d[None, :] < d))
    tl.atomic_add(RS + offs_q, rs, mask=qmask)


def _precision(t: torch.Tensor) -> str:
  return 'ieee' if t.dtype == torch.float32 else 'tf32'   # fp32 tables: exact fp32 dots (TF32 would round d2 badly)


def _p2(x: int) -> int:
  return 1 << max(4, (x - 1).bit_length())


def _grid(n_q: int, n_k: int, block_q: int, block_k: int, device) -> tuple:
  """(query blocks, row slices, rows per slice): about 4 programs per SM."""
  qb = triton.cdiv(n_q, block_q)
  sms = torch.cuda.get_device_properties(device).multi_processor_count if device.type == 'cuda' else 4
  splits = max(1, min(triton.cdiv(n_k, block_k), (4 * sms + qb - 1) // qb))
  rows = triton.cdiv(triton.cdiv(n_k, splits), block_k) * block_k
  return qb, triton.cdiv(n_k, rows), rows


def rbf_forward(q, q_sq, keys, key_sq, values, c: float, block_q: int = 64, block_k: int = 32, num_warps: int = 8):
  """s = Σ_j κ_ij v_j (fp32, Q x dv) for bf16 or fp32 q (same dtype as keys), keys (N x d), values (N x dv)."""
  n_q, d = q.shape
  n_k, dv = values.shape
  out = torch.zeros(n_q, dv, device=q.device, dtype=torch.float32)
  if n_k == 0 or n_q == 0:
    return out
  qb, splits, rows = _grid(n_q, n_k, block_q, block_k, q.device)
  _rbf_fwd_kernel[(qb, splits)](q.contiguous(), q_sq.contiguous(), keys, key_sq, values, out, n_q, n_k, d, dv, rows,
                                1.0 / c, BLOCK_Q=block_q, BLOCK_K=block_k, BLOCK_D=_p2(d), BLOCK_V=_p2(dv),
                                PRECISION=_precision(keys), num_warps=num_warps)
  return out


def rbf_backward(q, q_sq, g, keys, key_sq, values, c: float, block_q: int = 64, block_k: int = 32,
                 num_warps: int = 8):
  """(Σ_j A_ij k_j, Σ_j A_ij) in fp32 for A_ij = κ_ij <g_i, v_j>; dq = (2/c) (first - second * q)."""
  n_q, d = q.shape
  n_k, dv = values.shape
  acc = torch.zeros(n_q, d, device=q.device, dtype=torch.float32)
  rs = torch.zeros(n_q, device=q.device, dtype=torch.float32)
  if n_k == 0 or n_q == 0:
    return acc, rs
  qb, splits, rows = _grid(n_q, n_k, block_q, block_k, q.device)
  _rbf_bwd_kernel[(qb, splits)](q.contiguous(), q_sq.contiguous(), g.to(values.dtype).contiguous(), keys, key_sq,
                                values, acc, rs, n_q, n_k, d, dv, rows, 1.0 / c, BLOCK_Q=block_q, BLOCK_K=block_k, BLOCK_D=_p2(d),
                                BLOCK_V=_p2(dv), PRECISION=_precision(keys), num_warps=num_warps)
  return acc, rs
