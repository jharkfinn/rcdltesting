"""Exact per-group attribution of the weights of trained linear layers (the dual form of RCDL Sec. 2, extended to
momentum, AdamW and Muon).

A linear map z = W h + b inside a network is trained by updates built from per-example gradients delta_i h_i^T (delta_i
= dL/dz for example i, h_i its input; for a 1x1 convolution or a channels-last Linear, summed over positions). SGD with
momentum, AdamW (given its second moment) and Muon (given its Newton-Schulz polynomial) are linear in these gradients,
so every update splits exactly into one part per group of training examples. `GroupAttribution` keeps one optimizer
state per group for selected nn.Linear layers, applies the real update itself, and accumulates

  W_T = a_T W_0 + sum_k C_T^(k),     a_T = prod_{t <= T} (1 - lr_t * weight_decay),

where C_T^(k) is the part of W_T written by the examples of group k (and the same for the bias). The pre-activation of
neuron n on any input h then splits as z_n(h) = a_T W_0[n] h + sum_k C_T^(k)[n] h + (bias parts): which groups of
training examples wrote the neuron, and (from checkpoints of C) when. With one group per example and plain SGD this is
the linear-attention identity of the paper, checked in tests/test_identities.py.

Optimizers (the tracked weights are updated here, not by a torch optimizer):
  sgd    buf = mu buf + g;  W <- (1 - lr wd) W - lr (g + mu buf if nesterov else buf)        (decoupled decay)
  adamw  m = b1 m + (1-b1) g;  v = b2 v + (1-b2) g^2;  W <- (1 - lr wd) W - lr m_hat / (sqrt(v_hat) + eps)
         (torch.optim.AdamW; v uses the total gradient, so every group shares the same denominator)
  muon   buf = mu buf + (1-mu) g;  U = (1-mu) g + mu buf (nesterov);  O = NS(U);  W <- (1 - lr wd) W - lr_adj O
         (torch.optim.Muon). NS(U) = P U / ||U|| (or U P^T / ||U|| for tall matrices), with P a polynomial in
         X X^T built from the total U, so group k's part is P U^(k) / ||U||. The iteration runs in fp32 here (torch:
         bf16), so the real update differs from torch's by bf16 rounding only.
Biases use the same rule as the weights for sgd and adamw, and AdamW (betas, eps of the Muon settings) under muon,
as torch.optim.Muon accepts only matrices; their learning rate is lr * bias_lr_scale (set it to the AdamW / Muon
learning-rate ratio of the untracked parameters). Under muon the matmuls of the update run with TF32 off, so the group
parts add up to the real update to fp32 rounding even when the caller enables TF32.
"""

import contextlib
import math
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn

NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz_factor(U: torch.Tensor, steps: int = 5, eps: float = 1e-7, coeffs=NS_COEFFS):
  """(O, P, transposed, norm): O = NS(U) as in torch.optim.Muon but in the dtype of U; P with O = P U / norm, or
  O = U P^T / norm when U is tall (transposed = True)."""
  a, b, c = coeffs
  transposed = U.size(0) > U.size(1)
  X = U.T if transposed else U
  norm = X.norm().clamp(min=eps)
  X = X / norm
  P = torch.eye(X.size(0), dtype=X.dtype, device=X.device)
  for _ in range(steps):
    G = X @ X.T
    M = a * torch.eye(G.size(0), dtype=G.dtype, device=G.device) + b * G + c * (G @ G)
    X = M @ X
    P = M @ P
  O = X.T if transposed else X
  return O, P, transposed, norm


def muon_adjusted_lr(lr: float, shape, adjust_lr_fn: Optional[str] = None) -> float:
  A, B = shape[:2]
  if adjust_lr_fn is None or adjust_lr_fn == 'original':
    return lr * math.sqrt(max(1, A / B))
  if adjust_lr_fn == 'match_rms_adamw':
    return lr * 0.2 * math.sqrt(max(A, B))
  raise ValueError(f'unknown adjust_lr_fn {adjust_lr_fn!r}')


@contextlib.contextmanager
def _no_tf32():
  prev = torch.backends.cuda.matmul.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = False
  try:
    yield
  finally:
    torch.backends.cuda.matmul.allow_tf32 = prev


def _scatter(buf: torch.Tensor, groups: torch.Tensor, weights, G: torch.Tensor, alpha: float):
  """buf[groups[i, r]] += alpha * weights[i, r] * G[i] (hard groups: groups (B,), weights None). weights may also be
  a sequence of Python floats, one per column r, shared by all rows (mixup's lam, 1 - lam): no weighted copy of G and
  no host-device synchronization."""
  if groups.dim() == 1:
    buf.index_add_(0, groups, G, alpha=alpha)
    return
  if not torch.is_tensor(weights):
    for r, w in enumerate(weights):
      buf.index_add_(0, groups[:, r], G, alpha=alpha * float(w))
    return
  shape = (-1,) + (1,) * (G.dim() - 1)
  for r in range(groups.shape[1]):
    buf.index_add_(0, groups[:, r], G * weights[:, r].to(G.dtype).view(shape), alpha=alpha)


def to_fp16_scaled(T: torch.Tensor, chunk: int = 64):
  """(fp16 tensor, scale) with T ~= fp16 / scale; scale = 2^floor(log2(16384 / max |T|)) is exact and keeps every entry
  within 2^-28 of the maximum out of fp16's subnormal range. Works in chunks along dim 0 (no full fp32 copy)."""
  mx = float(T.abs().max()) if T.numel() else 0.0
  scale = 2.0 ** math.floor(math.log2(16384.0 / mx)) if mx > 0 else 1.0
  out = torch.empty(T.shape, dtype=torch.float16, device=T.device)
  for i in range(0, T.shape[0], chunk):
    out[i:i + chunk] = (T[i:i + chunk] * scale).half()
  return out, scale


class _Tracked:
  """Per-group optimizer state and contributions for one nn.Linear."""

  def __init__(self, lin: nn.Linear, n_groups: int, opt: str, hp: dict, dtype: torch.dtype, shadow: bool = False):
    W = lin.weight
    K, dev = n_groups, W.device
    self.lin, self.opt, self.hp, self.t, self.shadow = lin, opt, hp, 0, shadow
    self.W0 = W.detach().to(dtype).clone()
    self.b0 = lin.bias.detach().to(dtype).clone() if lin.bias is not None else None
    self._C = torch.zeros(K, *W.shape, dtype=dtype, device=dev)   # the parts are s * _C (lazy uniform weight decay)
    self.s = 1.0
    self.Cb = torch.zeros(K, W.shape[0], dtype=dtype, device=dev) if lin.bias is not None else None
    self.a, self.ab = 1.0, 1.0                      # decay factors of the initial weight and bias
    zeros = lambda *s: torch.zeros(*s, dtype=dtype, device=dev)
    self.m = zeros(K, *W.shape)                     # per-group first moment (adamw) or momentum buffer (sgd, muon)
    self.mb = zeros(K, W.shape[0]) if self.Cb is not None else None
    self.v = zeros(*W.shape) if opt == 'adamw' else None
    self.vb = zeros(W.shape[0]) if (self.Cb is not None and opt in ('adamw', 'muon')) else None
    self.h = self.delta = None
    self.v_frozen_c2 = None                          # set by freeze_second_moment: v and vb stop updating
    lin.weight.requires_grad_(False)                # the weight gradient is formed here from (h, delta)
    if lin.bias is not None:
      lin.bias.requires_grad_(False)
    self.handle = lin.register_forward_hook(self._hook)

  @property
  def C(self) -> torch.Tensor:
    """The parts (K, d_out, d_in), with the deferred decay applied."""
    self._flush()
    return self._C

  def _flush(self):
    if self.s != 1.0:
      self._C.mul_(self.s)
      self.s = 1.0

  def _hook(self, mod, inp, out):
    if not torch.is_grad_enabled():
      return None
    if self.h is not None:
      raise RuntimeError('tracked layer called twice before step(): one call per training step is supported')
    if not out.requires_grad:                        # nothing below needs a gradient (e.g. the first layer)
      out = out.detach().requires_grad_(True)
    self.h = inp[0].detach()
    out.register_hook(lambda g: setattr(self, 'delta', g.detach()))
    return out

  def per_example(self):
    """Per-example weight and bias gradients (B, d_out, d_in), (B, d_out), summed over positions, in the state dtype."""
    dt = self._C.dtype
    B = self.h.shape[0]
    h = self.h.reshape(B, -1, self.h.shape[-1]).to(dt)
    d = self.delta.reshape(B, -1, self.delta.shape[-1]).to(dt)
    G = torch.bmm(d.transpose(1, 2), h)
    Gb = d.sum(1) if self.Cb is not None else None
    self.h = self.delta = None
    return G, Gb

  @torch.no_grad()
  def step(self, groups: torch.Tensor, lr: float, weights: Optional[torch.Tensor] = None):
    if self.h is None or self.delta is None:
      raise RuntimeError('no (input, output gradient) captured: run forward and backward before step()')
    G, Gb = self.per_example()
    if self.s < 1e-6:                                  # keep 1 / s bounded on long runs
      self._flush()
    hp, W = self.hp, self.lin.weight
    wd, wd_b = hp['weight_decay'], hp['bias_weight_decay']
    self.t += 1
    if self.opt == 'adamw' or (self.opt == 'muon' and Gb is not None):
      b1, b2, eps = hp['betas'][0], hp['betas'][1], hp['eps']
      c1, c2 = 1 - b1 ** self.t, 1 - b2 ** self.t
      frozen = self.v_frozen_c2 is not None
      if frozen:                                       # frozen second moment: v and its bias correction stay fixed
        c2 = self.v_frozen_c2
    if self.opt == 'adamw':
      _scatter(self.m.mul_(b1), groups, weights, G, 1 - b1)
      if not frozen:
        g = G.sum(0)
        self.v.mul_(b2).addcmul_(g, g, value=1 - b2)
      D = 1.0 / ((self.v / c2).sqrt_() + eps)
      self.s *= 1 - lr * wd                            # decay of every part, deferred: no pass over C
      self._C.addcmul_(self.m, D, value=-lr / c1 / self.s)
      self.a *= 1 - lr * wd
      upd = self.m.sum(0).mul_(D).mul_(-lr / c1)
    elif self.opt == 'sgd':
      mu, nest = hp['momentum'], hp['nesterov']
      _scatter(self.m.mul_(mu), groups, weights, G, 1.0)
      self.s *= 1 - lr * wd
      self._C.add_(self.m, alpha=-lr * (mu if nest else 1.0) / self.s)
      if nest:
        _scatter(self._C, groups, weights, G, -lr / self.s)
      self.a *= 1 - lr * wd
      upd = -lr * ((mu if nest else 1.0) * self.m.sum(0) + (G.sum(0) if nest else 0.0))
    elif self.opt == 'muon':
      mu, nest = hp['momentum'], hp['nesterov']
      _scatter(self.m.mul_(mu), groups, weights, G, 1 - mu)
      g = G.sum(0)
      U = (1 - mu) * g + mu * self.m.sum(0) if nest else self.m.sum(0)
      with _no_tf32():                                 # the parts must add up to O: no TF32 in these matmuls
        O, P, transposed, norm = newton_schulz_factor(U, hp['ns_steps'], hp['ns_eps'])
        s = muon_adjusted_lr(lr, W.shape, hp['adjust_lr_fn']) / norm
        self.s *= 1 - lr * wd
        sc = s / self.s                                # step size on the stored _C
        K = self._C.shape[0]
        if transposed:                                 # group part: U^(k) P^T
          Pt = P.T.expand(K, *P.T.shape)
          self._C.baddbmm_(self.m, Pt, alpha=-sc * (mu if nest else 1.0))
          if nest:
            _scatter(self._C, groups, weights, torch.matmul(G, P.T), -sc * (1 - mu))
        else:                                          # group part: P U^(k)
          Pe = P.expand(K, *P.shape)
          self._C.baddbmm_(Pe, self.m, alpha=-sc * (mu if nest else 1.0))
          if nest:
            _scatter(self._C, groups, weights, torch.matmul(P, G), -sc * (1 - mu))
      self.a *= 1 - lr * wd
      upd = O * (-s * norm)
    else:
      raise ValueError(f'unknown optimizer {self.opt!r}')
    if not self.shadow:                              # a shadow splits the same update, the owner applies it
      W.mul_(1 - lr * wd).add_(upd.to(W.dtype))
    if Gb is None:
      return
    b = self.lin.bias
    if self.opt == 'muon':                             # torch.optim.Muon takes matrices only: AdamW for the bias,
      lr = lr * hp['bias_lr_scale']                    # at the learning rate of the caller's AdamW parameters
    if self.opt in ('adamw', 'muon'):
      _scatter(self.mb.mul_(b1), groups, weights, Gb, 1 - b1)
      if not frozen:
        gb = Gb.sum(0)
        self.vb.mul_(b2).addcmul_(gb, gb, value=1 - b2)
      Db = 1.0 / ((self.vb / c2).sqrt_() + eps)
      self.Cb.mul_(1 - lr * wd_b).addcmul_(self.mb, Db, value=-lr / c1)
      updb = self.mb.sum(0).mul_(Db).mul_(-lr / c1)
    else:
      mu, nest = hp['momentum'], hp['nesterov']
      _scatter(self.mb.mul_(mu), groups, weights, Gb, 1.0)
      self.Cb.mul_(1 - lr * wd_b).add_(self.mb, alpha=-lr * (mu if nest else 1.0))
      if nest:
        _scatter(self.Cb, groups, weights, Gb, -lr)
      updb = -lr * ((mu if nest else 1.0) * self.mb.sum(0) + (Gb.sum(0) if nest else 0.0))
    self.ab *= 1 - lr * wd_b
    if not self.shadow:
      b.mul_(1 - lr * wd_b).add_(updb.to(b.dtype))


class GroupAttribution:
  """Tracks selected nn.Linear layers of a model (see the module docstring).

  usage per training step:  loss.backward(); attr.step(groups, lr); other_optimizer.step()
  (the tracked weights get requires_grad False and must not be in the other optimizer).
  groups: LongTensor (B,) with values in [0, n_groups), e.g. the class labels, or one id per tracked example.
  shadow=True: a second split of the same training into other groups (e.g. random groups as a control). It must be
  built right after the owner (same weights), is stepped after it each step with its own groups, and never changes
  the weights; both splits then add up to the same weights.
  """

  def __init__(self, layers: Dict[str, nn.Linear], n_groups: int, optimizer: str = 'adamw', lr: float = 1e-3,
               betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0, bias_weight_decay: float = 0.0,
               momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5, ns_eps: float = 1e-7,
               adjust_lr_fn: Optional[str] = None, bias_lr_scale: float = 1.0, dtype: torch.dtype = torch.float32,
               shadow: bool = False):
    """bias_lr_scale: under muon, the biases' AdamW learning rate is lr * bias_lr_scale (ignored for sgd and adamw)."""
    hp = dict(betas=betas, eps=eps, weight_decay=weight_decay, bias_weight_decay=bias_weight_decay, momentum=momentum,
              nesterov=nesterov, ns_steps=ns_steps, ns_eps=ns_eps, adjust_lr_fn=adjust_lr_fn,
              bias_lr_scale=bias_lr_scale)
    self.lr, self.n_groups = lr, n_groups
    self.tracked = {name: _Tracked(lin, n_groups, optimizer, hp, dtype, shadow) for name, lin in layers.items()}

  def step(self, groups: torch.Tensor, lr: Optional[float] = None, weights=None):
    """groups (B,) hard, or (B, r) soft with weights (B, r) whose rows sum to 1, or a sequence of r floats shared by
    every row (e.g. mixup: the two source classes with lam and 1 - lam): each example's gradient is split across its
    groups by these weights, so the parts still add up exactly; the split itself is a modelling choice."""
    lr = self.lr if lr is None else lr
    for tr in self.tracked.values():
      tr.step(groups, lr, weights)

  def residual(self, name: str) -> Dict[str, float]:
    """max |W - (a W_0 + sum_k C^(k))| (and the same for the bias): zero up to rounding."""
    tr = self.tracked[name]
    W = tr.lin.weight.detach().to(tr.C.dtype)
    out = {'weight': (W - tr.a * tr.W0 - tr.C.sum(0)).abs().max().item(), 'weight_scale': W.abs().max().item()}
    if tr.Cb is not None:
      b = tr.lin.bias.detach().to(tr.C.dtype)
      out |= {'bias': (b - tr.ab * tr.b0 - tr.Cb.sum(0)).abs().max().item()}
    return out

  @torch.no_grad()
  def parts(self, name: str, h: torch.Tensor, neurons: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
    """Split the pre-activations of `neurons` (all if None) on inputs h (N, d_in):
    {'groups': (K, N, n) part written by each group (weights and bias), 'init': (N, n) decayed initial weights and
    bias}. groups.sum(0) + init equals the layer's pre-activation up to rounding."""
    tr = self.tracked[name]
    idx = slice(None) if neurons is None else neurons
    h = h.to(tr.C.dtype)
    groups = torch.einsum('kni,Ni->kNn', tr.C[:, idx], h)
    init = h @ (tr.a * tr.W0[idx]).T
    if tr.Cb is not None:
      groups = groups + tr.Cb[:, idx][:, None, :]
      init = init + tr.ab * tr.b0[idx]
    return {'groups': groups, 'init': init}

  @torch.no_grad()
  def compact_state(self) -> Dict[str, dict]:
    """Checkpoint form: C and Cb as scaled fp16 (to_fp16_scaled) without an fp32 clone, plus the fp32 remainder
    R = (W - a W_0) - sum_k C_k (stored) and Rb for the bias, so that W = a W_0 + sum_k C_k + R holds exactly with the
    stored parts. Load with: C_k = C[k].float() / C_scale."""
    out = {}
    for name, tr in self.tracked.items():
      rec = {'a': tr.a, 'ab': tr.ab, 't': tr.t}
      rec['C'], rec['C_scale'] = to_fp16_scaled(tr.C)
      csum = torch.zeros(tr.C.shape[1:], dtype=torch.float32, device=tr.C.device)
      for i in range(0, rec['C'].shape[0], 64):
        csum += rec['C'][i:i + 64].float().sum(0)
      rec['R'] = (tr.lin.weight.detach().float() - tr.a * tr.W0.float()) - csum / rec['C_scale']
      if tr.Cb is not None:
        rec['Cb'], rec['Cb_scale'] = to_fp16_scaled(tr.Cb)
        rec['Rb'] = (tr.lin.bias.detach().float() - tr.ab * tr.b0.float()) - rec['Cb'].float().sum(0) / rec['Cb_scale']
      out[name] = rec
    return out

  @torch.no_grad()
  def flush(self):
    """Applies the parts' deferred weight decay, as reading C (compact_state, full_state) does: a caller that skips
    a compact_state keeps the same fp32 rounding by calling this instead."""
    for tr in self.tracked.values():
      tr._flush()

  _STATE = ('W0', 'b0', 'C', 'Cb', 'm', 'mb', 'v', 'vb', 'a', 'ab', 't', 'v_frozen_c2')

  def full_state(self) -> Dict[str, dict]:
    """Everything needed to resume (parts, per-group optimizer states, decay factors, step count); the tracked
    weights themselves are in the model's state_dict."""
    return {name: {k: getattr(tr, k) for k in self._STATE} for name, tr in self.tracked.items()}

  def load_full_state(self, state: Dict[str, dict]):
    for name, tr in self.tracked.items():
      for k in self._STATE:
        if k not in state[name]:                        # states saved before a key existed (v_frozen_c2: None)
          continue
        v = state[name][k]
        cur = getattr(tr, k)
        if torch.is_tensor(cur):
          cur.copy_(v)
        else:
          setattr(tr, k, v)

  @torch.no_grad()
  def snapshot(self) -> Dict[str, dict]:
    """The tracked layers' optimizer state summed over groups (momentum), with the shared second moments and the step
    count: enough to continue training exactly from the current weights with a NEW attribution (load_snapshot), for
    replays of training from this point. Two tensors per weight; no parts."""
    return {name: {'m': tr.m.sum(0), 'mb': None if tr.mb is None else tr.mb.sum(0), 'v': tr.v, 'vb': tr.vb, 't': tr.t}
            for name, tr in self.tracked.items()}

  @torch.no_grad()
  def load_snapshot(self, snap: Dict[str, dict], history_group: int):
    """Start this new attribution (built on the snapshot's weights, so W_0 = those weights and every part is zero)
    from a snapshot: the snapshot's summed momentum goes to `history_group`, so that group's part is exactly the
    continuing effect of the updates made before the snapshot, and training continues as in the original run (up to
    the rounding of summing the momentum over a different partition of groups)."""
    for name, tr in self.tracked.items():
      st = snap[name]
      tr.m.zero_()
      tr.m[history_group].copy_(st['m'])
      if tr.mb is not None:
        tr.mb.zero_()
        tr.mb[history_group].copy_(st['mb'])
      if tr.v is not None:
        tr.v.copy_(st['v'])
      if tr.vb is not None:
        tr.vb.copy_(st['vb'])
      tr.t = st['t']

  @torch.no_grad()
  def freeze_second_moment(self):
    """Freeze every tracked layer's second moment (v, vb) and its bias correction at their current values: later steps
    update lr * mhat / (sqrt(v_frozen) + eps). The update is then the same linear map of the per-group momenta in any
    two runs from this state, so differences between runs are not scaled by a changed denominator. Needs t > 0
    (a loaded snapshot or earlier steps)."""
    for tr in self.tracked.values():
      if tr.opt not in ('adamw', 'muon'):
        raise ValueError('freeze_second_moment: AdamW (or the AdamW biases of muon) only')
      if tr.t == 0:
        raise ValueError('freeze_second_moment: no second moment yet (t = 0)')
      tr.v_frozen_c2 = 1 - tr.hp['betas'][1] ** tr.t

  @torch.no_grad()
  def group_sums(self, sets: Dict[str, Sequence[int]]) -> Dict[str, dict]:
    """Per tracked layer: the fp32 sums of the parts over each named set of groups ({'C', 'Cb'}), with the decay
    factors a, ab, the step count t, and the remainders R = W - a W_0 - sum_k C_k (Rb for the bias). Small enough to
    save every step; interval_buckets turns two of them into exact per-set writes."""
    out = {}
    for name, tr in self.tracked.items():
      C = tr.C
      rec = {'a': tr.a, 'ab': tr.ab, 't': tr.t, 'sets': {}}
      for sname, ids in sets.items():
        idx = torch.as_tensor(list(ids), dtype=torch.long, device=C.device)
        rec['sets'][sname] = {'C': C[idx].sum(0).float(),
                              'Cb': None if tr.Cb is None else tr.Cb[idx].sum(0).float()}
      rec['R'] = (tr.lin.weight.detach().to(C.dtype) - tr.a * tr.W0 - C.sum(0)).float()
      rec['Rb'] = None if tr.Cb is None else (tr.lin.bias.detach().to(C.dtype) - tr.ab * tr.b0 - tr.Cb.sum(0)).float()
      out[name] = rec
    return out

  def state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
    """Contributions and decay factors of every tracked layer (for checkpoints over training)."""
    return {name: {'C': tr.C.clone(), 'Cb': None if tr.Cb is None else tr.Cb.clone(), 'a': tr.a, 'ab': tr.ab,
                   'W0': tr.W0, 'b0': tr.b0, 't': tr.t} for name, tr in self.tracked.items()}


@torch.no_grad()
def interval_buckets(s1: Dict[str, dict], s2: Dict[str, dict], init: Dict[str, dict]) -> Dict[str, dict]:
  """Exact split of the change of every tracked weight over an interval (t1, t2] from two group_sums outputs whose
  sets partition all groups. init: {layer: {'W0', 'b0'}}. With rho = a_t2 / a_t1, per layer and set w:
    write_w = S_t2^w - rho S_t1^w           (the set's new writes inside the interval)
    dec_w   = (rho - 1) S_t1^w              (decay of the set's earlier writes)
    dec_0   = (a_t2 - a_t1) W_0             (decay of the initial weights)
    rem     = R_t2 - R_t1                   (rounding remainder)
  and the same for the bias. They sum to W_t2 - W_t1 exactly (asserted to fp32 rounding)."""
  out = {}
  for name, r1 in s1.items():
    r2, W0, b0 = s2[name], init[name]['W0'].float(), init[name]['b0']
    rho, rhob = r2['a'] / r1['a'], r2['ab'] / r1['ab']
    rec = {'write': {}, 'dec': {}, 'write_b': {}, 'dec_b': {}}
    for w in r1['sets']:
      C1, C2 = r1['sets'][w]['C'], r2['sets'][w]['C']
      rec['write'][w] = C2 - rho * C1
      rec['dec'][w] = (rho - 1) * C1
      if r1['sets'][w]['Cb'] is not None:
        Cb1, Cb2 = r1['sets'][w]['Cb'], r2['sets'][w]['Cb']
        rec['write_b'][w] = Cb2 - rhob * Cb1
        rec['dec_b'][w] = (rhob - 1) * Cb1
    rec['dec_0'] = (r2['a'] - r1['a']) * W0
    rec['rem'] = r2['R'] - r1['R']
    total = sum(rec['write'].values()) + sum(rec['dec'].values()) + rec['dec_0'] + rec['rem']
    W1 = r1['a'] * W0 + sum(x['C'] for x in r1['sets'].values()) + r1['R']
    W2 = r2['a'] * W0 + sum(x['C'] for x in r2['sets'].values()) + r2['R']
    assert torch.allclose(total, W2 - W1, rtol=0, atol=1e-5 * max(float(W2.abs().max()), 1e-12)), name
    if b0 is not None and r1['Rb'] is not None:
      rec['dec_0b'] = (r2['ab'] - r1['ab']) * b0.float()
      rec['rem_b'] = r2['Rb'] - r1['Rb']
    out[name] = rec
  return out


def snapshot_from_torch_adamw(opt: torch.optim.Optimizer, layers: Dict[str, nn.Linear]) -> Dict[str, dict]:
  """The snapshot form (GroupAttribution.snapshot) of torch AdamW's state for layers that torch trained: m = exp_avg,
  v = exp_avg_sq, t = step (weight and bias must share the step count). For a replay that starts tracking layers
  which were not tracked before (load_snapshot then puts m in the history group)."""
  out = {}
  for name, lin in layers.items():
    sw = opt.state[lin.weight]
    sb = opt.state[lin.bias] if lin.bias is not None else None
    t = int(sw['step'])
    if sb is not None:
      assert int(sb['step']) == t, f'{name}: weight and bias step counts differ'
    out[name] = {'m': sw['exp_avg'].detach().clone(), 'v': sw['exp_avg_sq'].detach().clone(), 't': t,
                 'mb': None if sb is None else sb['exp_avg'].detach().clone(),
                 'vb': None if sb is None else sb['exp_avg_sq'].detach().clone()}
  return out
