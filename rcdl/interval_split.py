"""Exact split of the change of a SwiGLU unit's response over an interval of training into per-group writes, the decay
of earlier writes, the remainder, input drift and a position switch (experiment 2, docs/EXACT_HISTORY.md Sec. 4.1).

Unit. Unit n of a block responds to input h (the block's LayerNorm output at one position, a d-vector) with
  u = SiLU(g) v,   g = w_g . h + beta_g,   v = w_v . h + beta_v,
where w_M, beta_M are row n of map M in {gate, value}. rcdl.attribution.GroupAttribution writes every tracked row as
  w_t = a_t W0 + sum_k c_t^(k) + R_t,     beta_t = ab_t b0 + sum_k cb_t^(k) + Rb_t,
with a_t, ab_t the decay factors of the initial weight W0 and bias b0, c_t^(k), cb_t^(k) the parts written by the
examples of group k (a class, or a random group of the shadow split), and R_t, Rb_t the remainder. Here R_t is always
computed in fp64 from the stored tensors and the model's weights W_t: R_t = W_t - a_t W0 - sum_k C_t^(k). It is never
assumed to be 0 (fp32 sums of 1,000 parts leave up to 8e-5 of |W| in E1's snapshots; fp16 storage of the parts more).

Interval I = (t1, t2]. With rho = a_t2 / a_t1 and rhob = ab_t2 / ab_t1 (1 without bias decay):
  nu_k = c_t2^(k) - rho c_t1^(k)       the writes of group k inside I,
  eps  = R_t2 - rho R_t1               the change of the remainder,
  Dw   = w_t2 - w_t1 = sum_k nu_k + sum_k (rho - 1) c_t1^(k) + (rho - 1)(a_t1 W0 + R_t1) + eps,
and the same for the bias with rhob, nub_k, epsb.

Path. For image x and a reference position p (default p*_t1(x), the unit's strongest position at t1) let
h1 = h_t1(x, p) and Dh = h_t2(x, p) - h1. On the straight line alpha in [0, 1] from (w_t1, h1) to (w_t2, h1 + Dh),
  z_M(alpha) = z_M1 + alpha l_M + alpha^2 q_M,   l_M = Dw_M . h1 + Dbeta_M + w_M1 . Dh,   q_M = Dw_M . Dh,
and du/dalpha = A g'(alpha) + B v'(alpha) with A = SiLU'(g) v and B = SiLU(g). The path weights
  Q0_g = int A,   Q1_g = int alpha A,   Q0_v = int B,   Q1_v = int alpha B      (Gauss-Legendre, `nodes` points)
split u_t2(x, p) - u_t1(x, p) exactly (to quadrature precision) into the weights' share of map M,
(Dw_M . h1 + Dbeta_M) Q0_M + (Dw_M . Dh) Q1_M, which is linear in Dw, and the input's share
Phi = sum_M (w_M1 . Dh) Q0_M + (Dw_M . Dh) Q1_M. The cross term Dw . Dh enters each once (Hcross below moves it).

Terms per image (sum over M in {gate, value}):
  T_k = sum_M (nu_k . h1 + nub_k) Q0_M + (nu_k . Dh) Q1_M                       writes of group k inside I
  F_k = sum_M (rho - 1) c_t1^(k) . (h1 Q0_M + Dh Q1_M) + (rhob - 1) cb_t1^(k) Q0_M  decay of group k's earlier writes
  F_0 = the same with a_t1 W0 + R_t1 and ab_t1 b0 + Rb_t1                        decay of the initial part and of R
  E   = sum_M eps . (h1 Q0_M + Dh Q1_M) + epsb Q0_M                              change of the remainder
  Phi = input drift (above)
  Pi  = [u_t2(x, p*_t2) - u_t1(x, p*_t1)] - [u_t2(x, p) - u_t1(x, p)]           position switch; for p = p*_t1 it is
                                                                                u_t2(x, p*_t2) - u_t2(x, p*_t1) >= 0
  sum_k T_k + sum_k F_k + F_0 + E + Phi + Pi = u_t2(x, p*_t2) - u_t1(x, p*_t1).

Class means. Every weight term is linear in a d-vector of weights, so its mean over the images of class c is a dot
product with
  Hbar_M(c) = mean_{x in c} (h1 Q0_M + Dh Q1_M)     and     qbar_M(c) = mean_{x in c} Q0_M,
e.g. T_k(c) = sum_M nu_k,M . Hbar_M(c) + nub_k,M qbar_M(c). Any group's, set's or random group's term on any class is
a (K x d)(d x C) product of the writes with Hbar (`column_terms`, `own_terms`, `set_terms`). The images enter only
through Hbar, qbar and the class means Phi(c), Pi(c), dR(c) = R_c(t2) - R_c(t1), where R_c(t) is the class mean of u_t
at the strongest positions. Hcross_M(c) = mean_{x in c} Dh Q1_M is the cross-term part of Hbar_M(c): Hbar + Hcross gives
the cross term wholly to the weights and Hbar - Hcross wholly to the input (Phi -+ cross, cross = sum_M Dw_M .
Hcross_M); every allocation stays exact.

Storage bounds. A stored fp16 part (C_fp16 / C_scale) is within half an fp16 ulp of the trainer's fp32 part:
hu(c) = half_ulp(C_fp16) / C_scale elementwise (fp32 snapshots: half an fp32 ulp). The class-mean terms then move by
at most
  |dT_k(c)| <= sum_M (hu(c_t2) + rho hu(c_t1)) . |Hbar_M(c)| + (hu(cb_t2) + rhob hu(cb_t1)) |qbar_M(c)|,
  |dF_k(c)| <= sum_M |rho - 1| hu(c_t1) . |Hbar_M(c)| + |rhob - 1| hu(cb_t1) |qbar_M(c)|,
and a set's term by the sum of its members' bounds. The stored split itself is exact: R absorbs the rounding.
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn.functional as F

MAPS = ('gate', 'value')
_FRAC = {torch.float16: (10, -24), torch.bfloat16: (7, -133), torch.float32: (23, -149), torch.float64: (52, -1074)}


def gl_nodes(n: int = 32, dtype=torch.float64, device=None):
  """Gauss-Legendre nodes and weights on [0, 1]: int_0^1 f = sum_i w_i f(x_i), exact for polynomials of degree < 2n."""
  x, w = np.polynomial.legendre.leggauss(n)
  return (torch.tensor((x + 1) / 2, dtype=dtype, device=device), torch.tensor(w / 2, dtype=dtype, device=device))


def dsilu(x: torch.Tensor) -> torch.Tensor:
  """SiLU'(x) = sigmoid(x) (1 + x (1 - sigmoid(x)))."""
  s = torch.sigmoid(x)
  return s * (1 + x * (1 - s))


def path_weights(g1, g_lin, g_quad, v1, v_lin, v_quad, nodes: int = 32):
  """(Q0_g, Q1_g, Q0_v, Q1_v), each shaped like g1: the integrals over alpha in [0, 1] of A = SiLU'(g) v, alpha A,
  B = SiLU(g) and alpha B along g(alpha) = g1 + alpha g_lin + alpha^2 g_quad (v(alpha) likewise). One node at a time
  (no (nodes, ...) tensor)."""
  x, w = gl_nodes(nodes, g1.dtype, g1.device)
  out = [torch.zeros_like(g1) for _ in range(4)]
  for i in range(nodes):
    a, wi = x[i], w[i]
    g = g1 + a * g_lin + a * a * g_quad
    v = v1 + a * v_lin + a * a * v_quad
    A, B = dsilu(g) * v, F.silu(g)
    out[0] += wi * A
    out[1] += (wi * a) * A
    out[2] += wi * B
    out[3] += (wi * a) * B
  return tuple(out)


def half_ulp(x: torch.Tensor) -> torch.Tensor:
  """fp64 elementwise bound on |true - x| for a value x rounded to nearest in x's dtype (fp16, bf16, fp32 or fp64):
  half the spacing of the binade of |x| (at a power of two the larger of the two spacings), and half the smallest
  subnormal spacing for subnormals and zeros."""
  frac, emin = _FRAC[x.dtype]
  xd = x.detach().double().abs()
  _, e = torch.frexp(xd)                                  # |x| = m 2^e with 0.5 <= m < 1: spacing 2^(e - 1 - frac)
  hu = torch.ldexp(torch.ones_like(xd), e - 2 - frac)
  floor = torch.full_like(xd, 2.0 ** (emin - 1))
  return torch.where(xd == 0, floor, torch.maximum(hu, floor))


@dataclass
class Parts:
  """One tracked map at one grid point, in fp64: W = a W0 + sum_k C[k] + R and b = ab b0 + sum_k Cb[k] + Rb.
  C (K, U, d), Cb (K, U), W0, R (U, d), b0, Rb (U,). hu, hub: elementwise storage bounds of C, Cb (None if not
  computed). r_stored: max |R - stored R| when the file has an fp32 R (a check of the trainer's own remainder)."""
  a: float
  ab: float
  W0: torch.Tensor
  b0: torch.Tensor
  C: torch.Tensor
  Cb: torch.Tensor
  R: torch.Tensor
  Rb: torch.Tensor
  t: int = 0
  hu: Optional[torch.Tensor] = None
  hub: Optional[torch.Tensor] = None
  source: str = ''
  r_stored: Optional[float] = None

  @property
  def W(self) -> torch.Tensor:
    return self.a * self.W0 + self.C.sum(0) + self.R

  @property
  def b(self) -> torch.Tensor:
    return self.ab * self.b0 + self.Cb.sum(0) + self.Rb


def load_parts(rec: dict, init: Optional[dict] = None, W: Optional[torch.Tensor] = None,
               b: Optional[torch.Tensor] = None, device=None, bounds: bool = True) -> Parts:
  """Parts of one map from a compact attribution record (GroupAttribution.compact_state: C fp16 with C_scale, Cb with
  Cb_scale, fp32 R, Rb, a, ab, t; W0 and b0 from `init` = attr_init.pt[key]) or a full fp32 state (full_state or a
  history full_{step}.pt record: W0, b0, C, Cb, a, ab, t, ...). The scales are powers of two, so C = C_fp16 / C_scale
  is exact in fp64. W, b: the model's weights at this grid point; R = W - a W0 - sum_k C_k (and Rb) is then computed in
  fp64. Without W the stored fp32 R is used (compact records only) and the parts define the weights."""
  dev = device if device is not None else rec['C'].device
  compact = 'C_scale' in rec
  W0 = (init['W0'] if init is not None and 'W0' not in rec else rec['W0']).to(dev, torch.float64)
  b0 = (init['b0'] if init is not None and 'b0' not in rec else rec['b0']).to(dev, torch.float64)
  sc, scb = (float(rec['C_scale']), float(rec['Cb_scale'])) if compact else (1.0, 1.0)
  C = rec['C'].to(dev).double() / sc
  Cb = rec['Cb'].to(dev).double() / scb
  a, ab = float(rec['a']), float(rec['ab'])
  r_stored = None
  if W is not None:
    R = W.to(dev, torch.float64) - a * W0 - C.sum(0)
    Rb = b.to(dev, torch.float64) - ab * b0 - Cb.sum(0)
    if 'R' in rec:
      r_stored = max(float((R - rec['R'].to(dev).double()).abs().max()),
                     float((Rb - rec['Rb'].to(dev).double()).abs().max()))
  elif 'R' in rec:
    R, Rb = rec['R'].to(dev).double(), rec['Rb'].to(dev).double()
  else:
    raise ValueError('load_parts: a full state has no stored remainder; pass the model weights W, b')
  hu = hub = None
  if bounds:
    hu = half_ulp(rec['C'].to(dev)) / sc
    hub = half_ulp(rec['Cb'].to(dev)) / scb
  return Parts(a=a, ab=ab, W0=W0, b0=b0, C=C, Cb=Cb, R=R, Rb=Rb, t=int(rec.get('t', 0)), hu=hu, hub=hub,
               source='compact' if compact else 'full', r_stored=r_stored)


@dataclass
class Increments:
  """One map over an interval (t1, t2] (see the module docstring). nu (K, U, d), nub (K, U): writes inside I; C1, Cb1:
  the parts at t1 (decay terms (rho - 1) C1); F0w (U, d) = (rho - 1)(a_t1 W0 + R_t1), F0b (U,); eps, epsb: change of
  the remainder; W1, b1, W2, b2, dW, db: the weights. nu_bound, nub_bound, dec_bound, decb_bound: elementwise storage
  bounds of nu and of (rho - 1) C1. resid: max |sum of the parts of Dw - Dw| relative to max |W| (fp64 rounding)."""
  rho: float
  rhob: float
  nu: torch.Tensor
  nub: torch.Tensor
  C1: torch.Tensor
  Cb1: torch.Tensor
  F0w: torch.Tensor
  F0b: torch.Tensor
  eps: torch.Tensor
  epsb: torch.Tensor
  W1: torch.Tensor
  b1: torch.Tensor
  W2: torch.Tensor
  b2: torch.Tensor
  dW: torch.Tensor
  db: torch.Tensor
  nu_bound: Optional[torch.Tensor] = None
  nub_bound: Optional[torch.Tensor] = None
  dec_bound: Optional[torch.Tensor] = None
  decb_bound: Optional[torch.Tensor] = None
  resid: float = 0.0


def increments(p1: Parts, p2: Parts) -> Increments:
  """The exact split of W_t2 - W_t1 (and the bias) of one map into the groups' new writes, the decay of their earlier
  writes, the decay of the initial part and of R_t1, and the change of the remainder. Both endpoints must share W0."""
  assert p1.C.shape == p2.C.shape, 'increments: the two grid points have different group or map shapes'
  assert torch.equal(p1.W0, p2.W0) and torch.equal(p1.b0, p2.b0), 'increments: different initial weights'
  rho, rhob = p2.a / p1.a, p2.ab / p1.ab
  W1, b1, W2, b2 = p1.W, p1.b, p2.W, p2.b
  nu, nub = p2.C - rho * p1.C, p2.Cb - rhob * p1.Cb
  eps, epsb = p2.R - rho * p1.R, p2.Rb - rhob * p1.Rb
  F0w = (rho - 1) * (p1.a * p1.W0 + p1.R)
  F0b = (rhob - 1) * (p1.ab * p1.b0 + p1.Rb)
  dW, db = W2 - W1, b2 - b1
  tot = nu.sum(0) + (rho - 1) * p1.C.sum(0) + F0w + eps
  totb = nub.sum(0) + (rhob - 1) * p1.Cb.sum(0) + F0b + epsb
  scale = max(float(W1.abs().max()), float(W2.abs().max()), 1e-300)
  resid = max(float((tot - dW).abs().max()), float((totb - db).abs().max())) / scale
  out = Increments(rho=rho, rhob=rhob, nu=nu, nub=nub, C1=p1.C, Cb1=p1.Cb, F0w=F0w, F0b=F0b, eps=eps, epsb=epsb,
                   W1=W1, b1=b1, W2=W2, b2=b2, dW=dW, db=db, resid=resid)
  if p1.hu is not None and p2.hu is not None:
    out.nu_bound = p2.hu + abs(rho) * p1.hu
    out.nub_bound = p2.hub + abs(rhob) * p1.hub
    out.dec_bound = abs(rho - 1) * p1.hu
    out.decb_bound = abs(rhob - 1) * p1.hub
  return out


def momentum_carryover(m1: torch.Tensor, v1: torch.Tensor, t1: int, t2: int, lr_of, b1: float = 0.9,
                       b2: float = 0.999, eps: float = 1e-8, horizon: int = 300) -> torch.Tensor:
  """AdamW: an estimate of the part of nu_k that the momentum m1 = m_t1^(k) carried in from before t1, for an interval
  starting at a snapshot: sum_{j=1..J} lr_of(t1 + j - 1) b1^j / (1 - b1^(t1 + j)) m1 D, J = min(t2 - t1, horizon), with
  the preconditioner D = 1 / (sqrt(v1 / (1 - b2^t1)) + eps) held at its t1 value and the decay inside I ignored
  (labelled an estimate). lr_of(s): the learning rate of the update that takes the step count from s to s + 1. The sign
  is that of the update (negative along m1)."""
  D = 1.0 / ((v1.double() / (1 - b2 ** t1)).sqrt() + eps)
  J = min(t2 - t1, horizon)
  coef = sum(lr_of(t1 + j - 1) * b1 ** j / (1 - b1 ** (t1 + j)) for j in range(1, J + 1))
  return -coef * m1.double() * D


def image_split(h1: torch.Tensor, h2: torch.Tensor, inc: Dict[str, Increments], u1best=None, u2best=None,
                nodes: int = 32) -> dict:
  """Per image and unit of one block: h1, h2 (N, U, d) the block's input at the reference positions at t1 and t2.
  u1best, u2best (N, U): the responses at the strongest positions (default: those at the reference, so Pi = 0).
  Returns (N, U) tensors Q0_M, Q1_M, z1_M (pre-activations at t1), u1, u2 (at the reference), Phi, Pi, dR =
  u2best - u1best, cross = sum_M (Dw_M . Dh) Q1_M, wsum = the weights' total share, resid = wsum + Phi + Pi - dR, and
  (N, U, d) tensors Hw_M = h1 Q0_M + Dh Q1_M and Hx_M = Dh Q1_M (the summands of Hbar and Hcross)."""
  Dh = h2 - h1
  co = {}
  for M in MAPS:
    I = inc[M]
    z1 = (h1 * I.W1).sum(-1) + I.b1
    wDh = (Dh * I.W1).sum(-1)
    dWh1 = (h1 * I.dW).sum(-1) + I.db
    quad = (Dh * I.dW).sum(-1)
    z2 = (h2 * I.W2).sum(-1) + I.b2
    co[M] = dict(z1=z1, z2=z2, wDh=wDh, dWh1=dWh1, quad=quad, lin=dWh1 + wDh)
  g, v = co['gate'], co['value']
  Q0g, Q1g, Q0v, Q1v = path_weights(g['z1'], g['lin'], g['quad'], v['z1'], v['lin'], v['quad'], nodes)
  u1 = F.silu(g['z1']) * v['z1']
  u2 = F.silu(g['z2']) * v['z2']
  u1b = u1 if u1best is None else u1best
  u2b = u2 if u2best is None else u2best
  out = {'Q0_gate': Q0g, 'Q1_gate': Q1g, 'Q0_value': Q0v, 'Q1_value': Q1v, 'z1_gate': g['z1'], 'z1_value': v['z1'],
         'u1': u1, 'u2': u2, 'dR': u2b - u1b, 'Pi': (u2b - u1b) - (u2 - u1)}
  out['Phi'] = g['wDh'] * Q0g + g['quad'] * Q1g + v['wDh'] * Q0v + v['quad'] * Q1v
  out['cross'] = g['quad'] * Q1g + v['quad'] * Q1v
  out['wsum'] = g['dWh1'] * Q0g + g['quad'] * Q1g + v['dWh1'] * Q0v + v['quad'] * Q1v
  out['resid'] = out['wsum'] + out['Phi'] + out['Pi'] - out['dR']
  for M, Q0, Q1 in (('gate', Q0g, Q1g), ('value', Q0v, Q1v)):
    out[f'Hx_{M}'] = Dh * Q1[..., None]
    out[f'Hw_{M}'] = h1 * Q0[..., None] + out[f'Hx_{M}']
  return out


class ClassSums:
  """Running sums, by group index (a class, or a class and a probe half), of the per-image quantities of one block and
  interval (image_split); means() gives the class means: Hbar_M, Hcross_M (G, U, d), qbar_M, Phi, Pi, dR, R1 (u at
  the strongest positions at t1), R2, cross (G, U), count (G,), and resid: the largest per-image |resid| seen."""

  VEC = ('Hw_gate', 'Hw_value', 'Hx_gate', 'Hx_value')
  SCAL = ('Q0_gate', 'Q0_value', 'Phi', 'Pi', 'dR', 'u1b', 'u2b', 'cross')

  def __init__(self, n_groups: int, U: int, d: int, dtype=torch.float64, device=None, cross: bool = True):
    self.vec = [k for k in self.VEC if cross or not k.startswith('Hx')]
    self.s = {k: torch.zeros(n_groups, U, d, dtype=dtype, device=device) for k in self.vec}
    self.s |= {k: torch.zeros(n_groups, U, dtype=dtype, device=device) for k in self.SCAL}
    self.count = torch.zeros(n_groups, dtype=dtype, device=device)
    self.resid = 0.0

  def add(self, gidx: torch.Tensor, rec: dict, u1best: torch.Tensor, u2best: torch.Tensor):
    """gidx (N,): group index of each image; rec: image_split output; u1best, u2best (N, U)."""
    src = dict(rec, u1b=u1best, u2b=u2best)
    for k in self.s:
      self.s[k].index_add_(0, gidx, src[k].to(self.s[k].dtype))
    self.count.index_add_(0, gidx, torch.ones_like(gidx, dtype=self.count.dtype))
    self.resid = max(self.resid, float(rec['resid'].abs().max()))

  def means(self) -> dict:
    n = self.count.clamp(min=1)
    out = {}
    names = {'Hw_gate': 'Hbar_gate', 'Hw_value': 'Hbar_value', 'Hx_gate': 'Hcross_gate', 'Hx_value': 'Hcross_value',
             'Q0_gate': 'qbar_gate', 'Q0_value': 'qbar_value', 'u1b': 'R1', 'u2b': 'R2'}
    for k, v in self.s.items():
      out[names.get(k, k)] = v / n.view(-1, *([1] * (v.dim() - 1)))
    out['count'] = self.count
    out['resid'] = self.resid
    return out


def class_weighted_inputs(h1, h2, inc, y, n_groups: int, u1best=None, u2best=None, nodes: int = 32,
                          cross: bool = True) -> dict:
  """The class means of one block and interval in one call (ClassSums over one batch). Hbar, Hcross are (C, U, d)
  with C = n_groups; the scalar means are (C, U). Transpose to (U, C, ...) for column_terms (as_unit_major)."""
  rec = image_split(h1, h2, inc, u1best, u2best, nodes)
  acc = ClassSums(n_groups, h1.shape[1], h1.shape[2], h1.dtype, h1.device, cross)
  acc.add(y, rec, rec['u1'] if u1best is None else u1best, rec['u2'] if u2best is None else u2best)
  return acc.means()


def as_unit_major(cm: dict) -> dict:
  """Class means with the unit first: (C, U, ...) -> (U, C, ...) (count and resid unchanged)."""
  return {k: (v.transpose(0, 1).contiguous() if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in cm.items()}


def column_terms(inc: Dict[str, Increments], Hbar: Dict[str, torch.Tensor], qbar: Dict[str, torch.Tensor], n: int,
                 bounds: bool = False) -> dict:
  """Class-mean terms of every group on S classes for unit n. Hbar[M] (S, d), qbar[M] (S,): Hbar_M(c), qbar_M(c) of
  unit n for those classes (a missing qbar[M] means 0, for the cross-term part Hcross). Returns T_gate, T_value (K, S)
  writes inside I, F (K, S) decay of earlier writes, F0, E (S,), W (S,) the weights' total share (sum_M Dw_M . Hbar_M +
  Dbeta_M qbar_M, equal to the sum of all of these), and with bounds T_bound, F_bound (K, S)."""
  out = {}
  for M in MAPS:
    I, H = inc[M], Hbar[M].to(inc[M].nu.dtype)
    q = qbar.get(M)
    q = torch.zeros(H.shape[0], dtype=H.dtype, device=H.device) if q is None else q.to(H.dtype)
    out[f'T_{M}'] = I.nu[:, n] @ H.T + I.nub[:, n, None] * q[None]
    Fm = (I.rho - 1) * (I.C1[:, n] @ H.T) + (I.rhob - 1) * I.Cb1[:, n, None] * q[None]
    F0 = H @ I.F0w[n] + I.F0b[n] * q
    E = H @ I.eps[n] + I.epsb[n] * q
    Wt = H @ I.dW[n] + I.db[n] * q
    out['F'] = out['F'] + Fm if 'F' in out else Fm
    out['F0'] = out['F0'] + F0 if 'F0' in out else F0
    out['E'] = out['E'] + E if 'E' in out else E
    out['W'] = out['W'] + Wt if 'W' in out else Wt
    if bounds:
      Ha, qa = H.abs(), q.abs()
      Tb = I.nu_bound[:, n] @ Ha.T + I.nub_bound[:, n, None] * qa[None]
      Fb = I.dec_bound[:, n] @ Ha.T + I.decb_bound[:, n, None] * qa[None]
      out['T_bound'] = out['T_bound'] + Tb if 'T_bound' in out else Tb
      out['F_bound'] = out['F_bound'] + Fb if 'F_bound' in out else Fb
  out['T'] = out['T_gate'] + out['T_value']
  return out


def own_terms(inc: Dict[str, Increments], cm: dict) -> torch.Tensor:
  """T_c(c) for every unit and class (U, C), when the groups are the classes (K = C). cm: unit-major class means."""
  out = 0
  for M in MAPS:
    I = inc[M]
    C = cm[f'Hbar_{M}'].shape[1]
    H = cm[f'Hbar_{M}'].to(I.nu.dtype)
    out = out + torch.einsum('cnd,ncd->nc', I.nu[:C], H) + I.nub[:C].T * cm[f'qbar_{M}'].to(I.nu.dtype)
  return out


def set_terms(inc: Dict[str, Increments], cm: dict, member: torch.Tensor, which: str = 'T') -> torch.Tensor:
  """sum_k member[c, k] X_k(c) for every unit and class (U, C), with X = T (writes inside I) or F (decay of earlier
  writes). member (C, K): set membership (or weights) per class, e.g. the relatives of each class."""
  out = 0
  for M in MAPS:
    I = inc[M]
    K, U, d = I.nu.shape
    src, srcb = (I.nu, I.nub) if which == 'T' else ((I.rho - 1) * I.C1, (I.rhob - 1) * I.Cb1)
    mem = member.to(src.dtype)
    S = (mem @ src.reshape(K, U * d)).reshape(-1, U, d)
    Sb = mem @ srcb
    H = cm[f'Hbar_{M}'].to(src.dtype)
    out = out + torch.einsum('cnd,ncd->nc', S, H) + Sb.T * cm[f'qbar_{M}'].to(src.dtype)
  return out


def group_total(inc: Dict[str, Increments], cm: dict) -> torch.Tensor:
  """sum_k T_k(c) + sum_k F_k(c) + F_0(c) + E(c) for every unit and class (U, C), summed from the per-group parts of
  one split (weights and bias); every split of the same weights gives the same total (Dw . Hbar + Dbeta qbar)."""
  tot = 0
  for M in MAPS:
    I = inc[M]
    H, q = cm[f'Hbar_{M}'].to(I.nu.dtype), cm[f'qbar_{M}'].to(I.nu.dtype)
    Wsum = I.nu.sum(0) + (I.rho - 1) * I.C1.sum(0) + I.F0w + I.eps
    bsum = I.nub.sum(0) + (I.rhob - 1) * I.Cb1.sum(0) + I.F0b + I.epsb
    tot = tot + torch.einsum('nd,ncd->nc', Wsum, H) + bsum[:, None] * q
  return tot


def split_completeness(inc: Dict[str, Increments], cm: dict) -> dict:
  """Completeness of the class-mean split for every unit and class: total = sum_k T_k + sum_k F_k + F_0 + E (from the
  per-group parts) + Phi + Pi against dR. Returns max_abs, rel (max_abs / max |R1, R2|) and the weights-only identity
  check (the per-group sum against Dw . Hbar + Dbeta qbar)."""
  tot = group_total(inc, cm)
  wonly = 0
  for M in MAPS:
    I = inc[M]
    H, q = cm[f'Hbar_{M}'].to(I.nu.dtype), cm[f'qbar_{M}'].to(I.nu.dtype)
    wonly = wonly + torch.einsum('nd,ncd->nc', I.dW, H) + I.db[:, None] * q
  dt = tot.dtype
  total = tot + cm['Phi'].to(dt) + cm['Pi'].to(dt)
  scale = max(float(cm['R1'].abs().max()), float(cm['R2'].abs().max()), 1e-300)
  err = float((total - cm['dR'].to(dt)).abs().max())
  return {'max_abs': err, 'rel': err / scale, 'weights_identity': float((tot - wonly).abs().max()) / scale}


def lr_schedule(cfg: dict):
  """The trainer's learning rate (experiments/convnext_train.py) of the update that takes the step count from s to
  s + 1: lr * (s + 1) / warmup during warm-up, then cosine to 0 (AdamW lr, or the Muon lr of the tracked maps)."""
  lr = cfg['muon_lr'] if cfg.get('opt') == 'muon' else cfg['lr']
  warm, steps = cfg['warmup'], cfg['steps']
  def f(s):
    if s < warm:
      return lr * (s + 1) / warm
    return lr * 0.5 * (1 + math.cos(math.pi * (s - warm) / (steps - warm)))
  return f
