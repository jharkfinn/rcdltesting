"""Split-key ("flash-decoding") Triton kernel for the weighted softmax readout when there are few queries.

With z_ij = <q_i, k_j> / (T sqrt(d_in)), the readout is s(q_i) = Σ_j e^{z_ij} ṽ_j / Σ_j w_j e^{z_ij}.

Why a kernel: the teacher-student student sees n_q = 64 queries per step against up to 12.8 million stored rows.
  * The dense path builds the n_q x n logit matrix and reads it several times (max, exp, two matmuls).
  * PyTorch's fused attention runs one program per block of queries (and per head). With 64 queries and one head it
    runs the whole table on one GPU block: measured 50x slower than the dense path on an A100.
This kernel gives each program one block of queries and one chunk of rows. Program (b, c) returns, for every query i
of block b, the chunk maximum m_ci = max_{j in c} z_ij and the partial sums Σ_{j in c} e^{z_ij - m_ci} [ṽ_j, w_j].
`weighted_attention_splitk` merges the chunks with the global maximum m_i = max_c m_ci, so the result is
s(q_i) = Σ_c e^{m_ci - m_i} A_ci / Σ_c e^{m_ci - m_i} D_ci, with A the value sums and D the weight sums.

Forward only: no gradient with respect to q or the table. It serves layers whose input needs no gradient (the
teacher-student student, evaluation). Rows with k = 0, ṽ = 0, w = 0 (empty) add nothing, so no row mask is needed.
"""

import math

import torch

try:
  import triton
  import triton.language as tl
  HAVE_TRITON = True
except ImportError:  # CPU-only installs: the dense path is used instead
  HAVE_TRITON = False


if HAVE_TRITON:

  @triton.jit
  def _partial_kernel(Q, K, V, W, M, ACC, DEN, n_q, n_k, d, d_out, rows_per_prog, scale,
                      stride_q, stride_k, stride_v,
                      BLOCK_Q: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_V: tl.constexpr,
                      PRECISION: tl.constexpr):
    pid_q = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_v = tl.arange(0, BLOCK_V)
    qmask = offs_q < n_q
    q = tl.load(Q + offs_q[:, None] * stride_q + offs_d[None, :], mask=qmask[:, None] & (offs_d[None, :] < d),
                other=0.0)
    m_i = tl.full([BLOCK_Q], float('-inf'), tl.float32)
    acc = tl.zeros([BLOCK_Q, BLOCK_V], tl.float32)
    den = tl.zeros([BLOCK_Q], tl.float32)
    start = pid_c * rows_per_prog
    for it in range(0, rows_per_prog // BLOCK_K):     # rows_per_prog is a multiple of BLOCK_K
      offs_k = start + it * BLOCK_K + tl.arange(0, BLOCK_K)
      kmask = offs_k < n_k
      k = tl.load(K + offs_k[:, None] * stride_k + offs_d[None, :], mask=kmask[:, None] & (offs_d[None, :] < d),
                  other=0.0)
      z = tl.dot(q, tl.trans(k), input_precision=PRECISION) * scale
      z = tl.where(kmask[None, :], z, float('-inf'))
      m_new = tl.maximum(m_i, tl.max(z, 1))
      alpha = tl.exp(m_i - m_new)
      p = tl.exp(z - m_new[:, None])
      v = tl.load(V + offs_k[:, None] * stride_v + offs_v[None, :], mask=kmask[:, None] & (offs_v[None, :] < d_out),
                  other=0.0)
      w = tl.load(W + offs_k, mask=kmask, other=0.0)
      acc = acc * alpha[:, None] + tl.dot(p, v, input_precision=PRECISION)
      den = den * alpha + tl.sum(p * w[None, :], 1)
      m_i = m_new
    out = pid_c * n_q + offs_q
    tl.store(M + out, m_i, mask=qmask)
    tl.store(DEN + out, den, mask=qmask)
    tl.store(ACC + out[:, None] * BLOCK_V + offs_v[None, :], acc, mask=qmask[:, None])


_STAGES = {}   # (device, precision, block sizes, warps) -> number of pipeline stages that fits in shared memory


def _next_pow2(x: int) -> int:
  return 1 << max(0, (x - 1).bit_length())


def weighted_attention_splitk(q, keys, values, weights, temperature: float, precision: str = 'tf32x3',
                              block_q: int = 64, block_k: int = 32, programs_per_sm: int = 4, num_warps: int = 4,
                              num_stages: int = 3):
  """Same output as rcdl.layers.weighted_attention (dense) for a fully valid table; forward only.

  q: (n_q, d) queries; keys: (n, d); values: (n, d_out) weighted values ṽ; weights: (n, 1) or (n,) weights w.
  precision: Triton dot precision. 'tf32x3' (default): three TF32 tensor-core products, error near fp32.
  'tf32': one TF32 product, about 1e-4 relative error. 'ieee': exact fp32 on CUDA cores (slow).
  Measured on an A100, 64 queries x 1M rows of width 100: dense 4.3 ms, 'tf32x3' 1.85 ms, 'tf32' 0.65 ms,
  'ieee' 4.8 ms. block_k = 64 with 4 warps spills registers (35-80 ms); keep block_k = 32.
  """
  if not HAVE_TRITON:
    raise RuntimeError('attn_impl="splitk" needs Triton (a CUDA build of PyTorch)')
  if q.requires_grad:
    raise RuntimeError('weighted_attention_splitk is forward-only; its input must not require a gradient')
  n_q, d = q.shape
  n, d_out = values.shape
  if d > 256:
    raise ValueError(f'splitk kernel supports d_in <= 256 (got {d})')
  q, keys, values = q.float().contiguous(), keys.float().contiguous(), values.float().contiguous()
  w = weights.reshape(-1).float().contiguous()
  BLOCK_D, BLOCK_V = max(16, _next_pow2(d)), max(16, _next_pow2(d_out))
  q_blocks = triton.cdiv(n_q, block_q)
  sms = torch.cuda.get_device_properties(q.device).multi_processor_count if q.is_cuda else 8
  chunks_target = max(1, (programs_per_sm * sms) // q_blocks)
  rows_per_prog = max(block_k, triton.cdiv(triton.cdiv(n, chunks_target), block_k) * block_k)
  chunks = triton.cdiv(n, rows_per_prog)
  M = torch.empty(chunks, n_q, device=q.device, dtype=torch.float32)
  DEN = torch.empty(chunks, n_q, device=q.device, dtype=torch.float32)
  ACC = torch.empty(chunks, n_q, BLOCK_V, device=q.device, dtype=torch.float32)
  scale = 1.0 / (temperature * math.sqrt(d))
  key = (q.device.index, precision, block_q, block_k, BLOCK_D, BLOCK_V, num_warps)
  stages = _STAGES.get(key, num_stages)
  while True:                     # GPUs with less shared memory (e.g. L4: 101 KB per block) need fewer stages
    try:
      _partial_kernel[(q_blocks, chunks)](q, keys, values, w, M, ACC, DEN, n_q, n, d, d_out, rows_per_prog, scale,
                                          q.stride(0), keys.stride(0), values.stride(0),
                                          BLOCK_Q=block_q, BLOCK_K=block_k, BLOCK_D=BLOCK_D, BLOCK_V=BLOCK_V,
                                          PRECISION=precision, num_warps=num_warps, num_stages=stages)
      break
    except triton.runtime.errors.OutOfResources:
      if stages <= 1:
        raise
      stages -= 1
  _STAGES[key] = stages
  m = M.max(0).values
  f = torch.exp(M - m)                                   # (chunks, n_q): e^{m_ci - m_i}
  num = (f[..., None] * ACC).sum(0)[:, :d_out]
  denom = (f * DEN).sum(0)[:, None]
  sign = torch.where(denom < 0, -torch.ones_like(denom), torch.ones_like(denom))
  return num / (denom.abs().clamp_min(1e-8) * sign)    # same denominator guard as the dense path
