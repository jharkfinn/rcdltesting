"""Teacher-student regression (paper Sec. 5, Fig. 3, App. E.1.2) at a configurable, CPU-feasible scale.

Inputs x ~ N(0, I_100) projected to the unit sphere; targets y* = teacher(x), a random residual ReLU MLP
(d_in = d_e = 100, L blocks, d_out = 1, LeCun-normal init). Students are trained online (fresh samples every step)
on the mean squared error and evaluated on a fixed held-out set.

Students:
  'gws'   one Growing Weighted Softmax layer on RMSNorm(x) (the paper's model; reuses rcdl.layers.RCDLLayer).
  'psm'   one Parametric Softmax layer (unweighted softmax over N learned keys/values, Adam), paper App. E.4.
  'mlp'   residual ReLU MLP of width d_e trained by SGD with momentum (the paper's "Reference MLP").
Paper scale: 200,000 steps x batch 64 (12.8M stored tuples). The default here is 2,000 steps.
"""

import dataclasses as dc
import json
import math
import os
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.layers import RCDLConfig, RCDLLayer, rms_norm


def lecun_linear(d_in: int, d_out: int, gen: torch.Generator, bias: bool = True) -> nn.Linear:
  lin = nn.Linear(d_in, d_out, bias=bias)
  with torch.no_grad():
    lin.weight.copy_(torch.randn(d_out, d_in, generator=gen) / math.sqrt(d_in))
    if bias:
      lin.bias.zero_()
  return lin


class ResidualReLUMLP(nn.Module):
  """h = ReLU(W_in x); L-1 times h <- h + ReLU(W h); y = w_out h. Teacher (width 100) and MLP student (width d_e)."""

  def __init__(self, d_in: int, width: int, num_blocks: int, d_out: int, gen: torch.Generator):
    super().__init__()
    self.inp = lecun_linear(d_in, width, gen)
    self.blocks = nn.ModuleList([lecun_linear(width, width, gen) for _ in range(num_blocks - 1)])
    self.out = lecun_linear(width, d_out, gen)

  def forward(self, x):
    h = F.relu(self.inp(x))
    for b in self.blocks:
      h = h + F.relu(b(h))
    return self.out(h)


def sample_inputs(n: int, d: int, gen: torch.Generator) -> torch.Tensor:
  x = torch.randn(n, d, generator=gen)
  return x / x.norm(dim=-1, keepdim=True)


@dc.dataclass(frozen=True)
class TSConfig:
  student: str = 'gws'           # 'gws' | 'psm' | 'mlp'
  d_in: int = 100
  teacher_width: int = 100
  teacher_blocks: int = 2        # App. E.1.2: L = 2 (main text says one hidden layer; see docs)
  steps: int = 2000
  batch_size: int = 64
  n_test: int = 10000
  teacher_seed: int = 0
  seed: int = 1
  # gws
  temperature: float = 1.0
  eta: float = 30.0              # weight learning rate (App. Table: [10, 30, 100])
  beta: float = 0.3              # value learning rate (App. Table: [0.1, 0.3, 1.0])
  momentum: float = 0.0
  # psm / mlp
  psm_size: int = 0              # 0 = batch_size * (steps + 1), i.e. the final GWS database size
  lr: float = 1e-3
  mlp_width: int = 1024
  mlp_momentum: float = 0.9


def run_teacher_student(cfg: TSConfig, out_dir: Optional[str] = None, eval_every: int = 250,
                        device: str = 'cpu') -> Dict[str, Any]:
  tgen = torch.Generator().manual_seed(cfg.teacher_seed)
  teacher = ResidualReLUMLP(cfg.d_in, cfg.teacher_width, cfg.teacher_blocks, 1, tgen).requires_grad_(False).to(device)
  test_x = sample_inputs(cfg.n_test, cfg.d_in, tgen).to(device)
  test_y = teacher(test_x)
  var_y = test_y.var().item()
  gen = torch.Generator().manual_seed(cfg.seed)
  torch.manual_seed(cfg.seed)

  B = cfg.batch_size
  rows = lambda: 1                    # number of stored rows / table entries (sets the eval chunk size)
  if cfg.student == 'gws':
    rc = RCDLConfig(kernel_type='gws', hidden_dim=cfg.d_in, num_layers=0, temperature=cfg.temperature, eta=cfg.eta,
                    beta=cfg.beta, model_momentum=cfg.momentum, db_size=B * (cfg.steps + 1), batch_size=B,
                    num_classes=1, global_init_seed=cfg.seed)
    layer = RCDLLayer(rc, cfg.d_in, 1, 0, gen).to(device)
    model = lambda x: layer(rms_norm(x))
    params, modules, rows = [], [layer], lambda: layer.size
  elif cfg.student == 'psm':
    n = cfg.psm_size or B * (cfg.steps + 1)
    keys = nn.Parameter((torch.randn(n, cfg.d_in, generator=gen) / math.sqrt(cfg.d_in)).to(device))
    vals = nn.Parameter((torch.randn(n, 1, generator=gen) * 0.02).to(device))
    scale = 1.0 / (cfg.temperature * math.sqrt(cfg.d_in))
    model = lambda x: torch.softmax(rms_norm(x) @ keys.T * scale, -1) @ vals
    params, modules, rows = [keys, vals], [], lambda: n
  else:
    mlp = ResidualReLUMLP(cfg.d_in, cfg.mlp_width, cfg.teacher_blocks, 1, gen).to(device)
    model, params, modules = mlp, list(mlp.parameters()), [mlp]
  opt = None
  if cfg.student == 'psm':
    opt = torch.optim.Adam(params, lr=cfg.lr)
  elif cfg.student == 'mlp':
    opt = torch.optim.SGD(params, lr=cfg.lr, momentum=cfg.mlp_momentum)

  def evaluate():
    for m in modules:
      m.eval()
    chunk = max(16, min(2000, int(2.5e8 // max(1, rows()))))   # keeps the chunk x rows logit matrix near 1 GB
    with torch.no_grad():
      mse = sum(F.mse_loss(model(test_x[i:i + chunk]), test_y[i:i + chunk], reduction='sum').item()
                for i in range(0, cfg.n_test, chunk)) / cfg.n_test
    for m in modules:
      m.train()
    return mse

  log, t0 = [], time.time()
  log.append({'step': 0, 'test_mse': evaluate(), 'wall_sec': 0.0})
  for step in range(1, cfg.steps + 1):
    x = sample_inputs(B, cfg.d_in, gen).to(device)
    y = teacher(x)
    pred = model(x)
    loss = F.mse_loss(pred, y)
    if opt is not None:
      opt.zero_grad()
    loss.backward()
    if opt is not None:
      opt.step()
    if cfg.student == 'gws':
      layer.commit()
    if step % eval_every == 0 or step == cfg.steps:
      rec = {'step': step, 'train_mse': loss.item(), 'test_mse': evaluate(), 'wall_sec': time.time() - t0}
      log.append(rec)
      print(f'[ts:{cfg.student}] step={step} train={rec["train_mse"]:.5f} test={rec["test_mse"]:.5f} '
            f'rel={rec["test_mse"] / var_y:.4f} ({rec["wall_sec"]:.0f}s)', flush=True)
  result = {'cfg': dc.asdict(cfg), 'var_y': var_y, 'log': log, 'final_test_mse': log[-1]['test_mse'],
            'final_rel_mse': log[-1]['test_mse'] / var_y}
  if out_dir:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'result.json'), 'w') as f:
      json.dump(result, f, indent=1)
  return result
