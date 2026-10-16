"""ConvNeXt with SwiGLU blocks, at the sizes of Cammarata's "Neural Networks Are Already Clean" (Goodfire, 2026).

The essay publishes no code or checkpoints (its checkpoint files return 403). Its microscope metadata and training-map
manifest (already-clean.goodfire.com, read 2026-10-16) give: "ConvNeXt-SwiGLU", 18 blocks (stage depths 3 / 3 / 9 / 3),
SwiGLU units per block 128 / 256 / 512 / 1,024 for medium (7,057,377 parameters), 256 / 512 / 1,024 / 2,048 for large
("ConvNeXt-Tiny-SwiGLU"), 64 in every block for tiny (91,828) and 64 / 64 / 128 / 192 for "wider" (345,552); 224 px
center crops; fresh runs of 10 epochs, 25,020 updates (batch about 512) with mixup on images and labels.

Inferred here: widths 48 / 96 / 192 / 384 (medium) and 96 / 192 / 384 / 768 (large), so units = 8/3 x width as in
standard SwiGLU MLPs; the published counts then match when the 1,000-way classifier is left out (medium 7,057,776 here
against 7,057,377; wider 24 / 32 / 48 / 48 gives 345,576 against 345,552). Tiny's widths are not determined by its count
(several 4-stage choices come within 600 parameters); 8 / 8 / 24 / 28 is used. Optimizer, schedule, augmentation other
than mixup, and the exact stem are not published; the stem here is the standard 4 x 4 patchify (stride 2 for 64 px).

Block: depthwise 7x7 conv -> LayerNorm -> SwiGLU: u = SiLU(W_gate h) * (W_value h) -> W_down u -> layer scale ->
residual. The essay's "neurons" are the SwiGLU products u; each has a gate and a value pre-activation, both linear in
h, so rcdl.attribution.GroupAttribution tracks pw_gate and pw_value (channels-last nn.Linear) directly.
"""

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

SIZES = {   # parameter counts without the classifier, as published (see the module docstring)
  'tiny': dict(dims=(8, 8, 24, 28), units=(64, 64, 64, 64), depths=(3, 3, 9, 3)),           # 92,228 (essay 91,828)
  'wider': dict(dims=(24, 32, 48, 48), units=(64, 64, 128, 192), depths=(3, 3, 9, 3)),      # 345,576 (345,552)
  'medium': dict(dims=(48, 96, 192, 384), units=(128, 256, 512, 1024), depths=(3, 3, 9, 3)),  # 7,057,776 (7,057,377)
  'large': dict(dims=(96, 192, 384, 768), units=(256, 512, 1024, 2048), depths=(3, 3, 9, 3)),
  # E7: tiny with its last layer crushed to 13 units (the essay's medium crush is 1,024 -> 13) and no route around it:
  # the last block has no residual connection and no layer scale, so the classifier reads only that block's output
  'tiny_crush13': dict(dims=(8, 8, 24, 28), units=(64, 64, 64, 64), depths=(3, 3, 9, 3), last_units=13, last_skip=False),
}


class LayerNorm2d(nn.LayerNorm):
  """LayerNorm over the channels of a (B, C, H, W) tensor."""

  def forward(self, x):
    return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2)


class Block(nn.Module):
  def __init__(self, d: int, units: int, layer_scale: float = 1e-6, drop_path: float = 0.0, skip: bool = True):
    super().__init__()
    self.skip = skip
    self.dwconv = nn.Conv2d(d, d, 7, padding=3, groups=d)
    self.norm = nn.LayerNorm(d, eps=1e-6)
    self.pw_gate = nn.Linear(d, units)
    self.pw_value = nn.Linear(d, units)
    self.pw_down = nn.Linear(units, d)
    self.gamma = nn.Parameter(layer_scale * torch.ones(d)) if layer_scale > 0 else None
    self.drop_path = drop_path

  def units(self, x):
    """SwiGLU products (B, H, W, units) of the block for input x (B, C, H, W)."""
    h = self.norm(self.dwconv(x).permute(0, 2, 3, 1))
    return F.silu(self.pw_gate(h)) * self.pw_value(h)

  def forward(self, x):
    h = self.pw_down(self.units(x))
    if self.gamma is not None:
      h = self.gamma * h
    h = h.permute(0, 3, 1, 2)
    if self.training and self.drop_path > 0:
      keep = torch.rand(x.shape[0], 1, 1, 1, device=x.device) >= self.drop_path
      h = h * keep / (1 - self.drop_path)
    return x + h if self.skip else h


class ConvNeXt(nn.Module):
  def __init__(self, dims: Sequence[int], units: Sequence[int], depths: Sequence[int], n_classes: int = 1000,
               stem: int = 4, drop_path: float = 0.0, layer_scale: float = 1e-6, last_units: int = None,
               last_skip: bool = True):
    """last_units / last_skip: the width of the last block's SwiGLU layer and whether it has a residual connection
    (without one it also has no layer scale, so the classifier reads only its output)."""
    super().__init__()
    self.dims, self.units, self.depths = tuple(dims), tuple(units), tuple(depths)
    rates = torch.linspace(0, drop_path, sum(depths)).tolist()
    self.downsample = nn.ModuleList([nn.Sequential(nn.Conv2d(3, dims[0], stem, stride=stem), LayerNorm2d(dims[0], eps=1e-6))])
    for i in range(len(dims) - 1):
      self.downsample.append(nn.Sequential(LayerNorm2d(dims[i], eps=1e-6), nn.Conv2d(dims[i], dims[i + 1], 2, stride=2)))
    self.stages, k = nn.ModuleList(), 0
    for i, (d, u, n) in enumerate(zip(dims, units, depths)):
      blocks = []
      for j in range(n):
        last = i == len(dims) - 1 and j == n - 1
        if last and (last_units is not None or not last_skip):
          blocks.append(Block(d, last_units or u, layer_scale if last_skip else 0.0, rates[k + j], skip=last_skip))
        else:
          blocks.append(Block(d, u, layer_scale, rates[k + j]))
      self.stages.append(nn.Sequential(*blocks))
      k += n
    self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
    self.head = nn.Linear(dims[-1], n_classes)
    self.apply(self._init)

  @staticmethod
  def _init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
      nn.init.trunc_normal_(m.weight, std=0.02)
      nn.init.zeros_(m.bias)

  def features(self, x):
    for down, stage in zip(self.downsample, self.stages):
      x = stage(down(x))
    return self.norm(x.mean((2, 3)))

  def forward(self, x):
    return self.head(self.features(x))

  def blocks(self):
    """[(name, block)] in order, named 'stage{i}.block{j}' as in the essay's microscope (zero-based)."""
    return [(f'stage{i}.block{j}', b) for i, st in enumerate(self.stages) for j, b in enumerate(st)]


def build_convnext(size: str, n_classes: int = 1000, **kw) -> ConvNeXt:
  return ConvNeXt(**SIZES[size], n_classes=n_classes, **kw)


def n_params(model: nn.Module, head: bool = True) -> int:
  """Parameter count; head=False leaves out the classifier, as the essay's published counts appear to."""
  n = sum(p.numel() for p in model.parameters())
  return n if head else n - sum(p.numel() for p in model.head.parameters())
