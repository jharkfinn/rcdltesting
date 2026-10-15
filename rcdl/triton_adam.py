"""Fused Adam step for bf16 parameters with an fp32 second moment (AdamFP32SecondMoment in rcdl/train.py).

One Triton kernel reads p (bf16), g (bf16), m (bf16), v (fp32) once and writes p, m, v once: 18 bytes per element per
step and no temporaries. The chunked PyTorch version makes several fp32 temporaries per chunk (about 3 GB each for
262,144 rows of width 3,072), which does not fit next to the 78.6 GB full-scale Fig. 5 state on a 96 GB GPU.

Update (t = step, c1 = 1 - β1^t, c2 = 1 - β2^t), all arithmetic in fp32:
  m <- β1 m + (1 - β1) g,   v <- β2 v + (1 - β2) g²,   p <- p - lr (m / c1) / (sqrt(v / c2) + eps).

stochastic=True rounds the new p and m to bf16 stochastically: with x in fp32 and r uniform on {0, ..., 2^16 - 1},
bf16(x) = truncate(bits(x) + r), which rounds |x| up with probability equal to the distance from the lower bf16
neighbour in units of the bf16 spacing. E[bf16(x)] = x, so updates smaller than half the bf16 spacing are not lost on
average (round-to-nearest drops them every step).
"""

import torch

try:
  import triton
  import triton.language as tl
except ImportError:      # CPU-only installs: AdamFP32SecondMoment falls back to the chunked PyTorch update
  triton = None


def stochastic_round_bf16(x: torch.Tensor, gen: torch.Generator = None) -> torch.Tensor:
  """PyTorch reference of the kernel's rounding: fp32 -> bf16, unbiased (E[result] = x for finite x)."""
  xf = x.float().contiguous()
  bits = xf.view(torch.int32)
  r = torch.randint(0, 1 << 16, bits.shape, dtype=torch.int32, device=x.device, generator=gen)
  out = ((bits + r) & -65536).view(torch.float32)
  return torch.where(torch.isnan(xf), xf, out).to(torch.bfloat16)      # NaN stays NaN (adding to its bits can give ±0)


if triton is not None:
  @triton.jit(do_not_specialize=['seed'])
  def _adam_fp32v_kernel(p_ptr, g_ptr, m_ptr, v_ptr, n, lr, b1, b2, omb1, omb2, c1, c2, eps, seed,
                         STOCHASTIC: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)   # int64: the table has 7.9e9 elements
    mask = offs < n
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    v = tl.load(v_ptr + offs, mask=mask, other=0.0)
    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = b1 * m + omb1 * g      # omb = 1 - β computed in float64 on the host (1 - fp32(0.999) is off by 1.3e-5)
    v = b2 * v + omb2 * g * g
    p = p - lr * (m / c1) / (tl.sqrt(v / c2) + eps)
    if STOCHASTIC:
      r = tl.randint4x(seed, offs)
      r0, r1 = r[0], r[1]
      pb = (p.to(tl.uint32, bitcast=True) + (r0 & 0xFFFF)) & 0xFFFF0000
      mb = (m.to(tl.uint32, bitcast=True) + (r1 & 0xFFFF)) & 0xFFFF0000
      p = tl.where(p == p, pb.to(tl.float32, bitcast=True), p)   # exact in bf16 (low 16 bits are zero); NaN stays NaN
      m = tl.where(m == m, mb.to(tl.float32, bitcast=True), m)
    tl.store(p_ptr + offs, p.to(p_ptr.dtype.element_ty), mask=mask)
    tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
    tl.store(v_ptr + offs, v, mask=mask)


def available(p: torch.Tensor) -> bool:
  return triton is not None and p.is_cuda and p.dtype == torch.bfloat16 and p.is_contiguous()


def adam_fp32v_step_(p, g, m, v, lr, b1, b2, c1, c2, eps, seed: int = 0, stochastic: bool = False, block: int = 2048):
  """In-place fused step on contiguous CUDA tensors p, g, m (bf16) and v (fp32) of the same shape."""
  assert p.is_contiguous() and g.is_contiguous() and m.is_contiguous() and v.is_contiguous()
  assert v.dtype == torch.float32 and p.shape == g.shape == m.shape == v.shape
  n = p.numel()
  _adam_fp32v_kernel[(triton.cdiv(n, block),)](p, g, m, v, n, float(lr), float(b1), float(b2), 1.0 - b1, 1.0 - b2,
                                                float(c1), float(c2),
                                                float(eps), int(seed), STOCHASTIC=stochastic, BLOCK=block,
                                                num_warps=8)
