"""Residual classifier backbone shared by all RCDL layers and the dense ResNet-MLP baseline (notebook Section 3.1)."""

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.layers import RCDLConfig, RCDLLayer, rms_norm


def dense(d_in: int, d_out: int, gen: torch.Generator) -> nn.Linear:
  """nn.Linear with truncated LeCun-normal weight init and zero bias."""
  lin = nn.Linear(d_in, d_out)
  with torch.no_grad():
    w = nn.init.trunc_normal_(torch.empty(d_out, d_in), std=1.0, a=-2.0, b=2.0, generator=gen)
    lin.weight.copy_(w * math.sqrt(1.0 / d_in) / 0.87962566103423978)
    lin.bias.zero_()
  return lin


ACTIVATIONS = {'none': lambda v: v, 'gelu': lambda v: F.gelu(v, approximate='tanh'), 'relu': F.relu}


class CifarRCDLClassifier(nn.Module):
  """Stacked residual classifier supporting all RCDL layers, Parametric Softmax, and standard dense (ResNet-MLP) layers:
  L pre-norm residual blocks x <- x + act(layer(RMSNorm(x))) followed by a readout layer of the same kind."""

  def __init__(self, cfg: RCDLConfig):
    super().__init__()
    self.cfg, self.act = cfg, ACTIVATIONS[cfg.nonlinearity]
    gen, H = torch.Generator().manual_seed(cfg.global_init_seed), cfg.hidden_dim
    if cfg.kernel_type == 'standard':
      layer = lambda d_out, layer_id: dense(H, d_out, gen)
    else:
      layer = lambda d_out, layer_id: RCDLLayer(cfg, H, d_out, layer_id, gen)
    self.layers = nn.ModuleList([layer(H, i) for i in range(cfg.num_layers)])
    self.output_layer = layer(cfg.num_classes, cfg.num_layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    for layer in self.layers:
      y = self.act(layer(rms_norm(x)))
      x = x + y if self.cfg.use_residual else y
    return self.output_layer(rms_norm(x))

  def rcdl_layers(self) -> Dict[str, RCDLLayer]:
    """{'layer_0', ..., 'output_layer'} -> RCDLLayer (empty for 'standard')."""
    layers = {f'layer_{i}': m for i, m in enumerate(self.layers)}
    layers['output_layer'] = self.output_layer
    return {name: m for name, m in layers.items() if isinstance(m, RCDLLayer)}
