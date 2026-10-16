"""E1: ConvNeXt-SwiGLU on ImageNet with exact per-class attribution of every tracked SwiGLU gate and value map.

Recipe from the essay's training-map manifest (already-clean.goodfire.com): all 1,281,167 training images, batch 512,
25,020 updates (10 epochs of 2,502 full batches), mixup applied to images and labels with the same weight, checkpoints at
steps 0, 50, 100, 200, 350, 500, 1,000, 2,000, 2,500, 5,000, 7,500, 10,000, 12,500, 17,500, 20,000, 25,020.
Not published, chosen here: AdamW (betas 0.9 / 0.999) at 2e-3 with weight decay 0.05 on matrices and kernels (0 on
norms, biases and layer scales), 1,250 linear warm-up steps then cosine to 0, mixup alpha 0.8 with one lam per batch and
the batch paired with its reverse (as in timm), no cutmix, label smoothing, EMA or stochastic depth.

Attribution: rcdl.attribution.GroupAttribution tracks pw_gate and pw_value of the chosen blocks with one group per class
and mixup's two source classes as soft groups (lam, 1 - lam); the tracked maps are updated by it with the same optimizer
settings, the rest by torch. Saved per checkpoint: the model, the per-class parts C (fp16 times a power-of-two scale,
with the fp32 remainder R so that W = a W_0 + sum_k C_k + R exactly; GroupAttribution.compact_state) and validation
accuracy.
"""

import argparse
import json
import math
import os
import queue
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.attribution import GroupAttribution
from rcdl.convnext import build_convnext
from rcdl.imagenet import GPUTrainLoader, val_batches

CKPT_STEPS = (0, 50, 100, 200, 350, 500, 1000, 2000, 2500, 5000, 7500, 10000, 12500, 17500, 20000, 25020)


def prefetch(gen, depth: int = 3):
  """Runs a batch generator in a background thread (GPU decode overlaps the training step); an exception in the
  generator is raised in the consumer (otherwise the consumer would wait forever)."""
  q, stop = queue.Queue(depth), object()
  def work():
    try:
      for item in gen:
        q.put(item)
    except BaseException as e:
      q.put(e)
      return
    q.put(stop)
  threading.Thread(target=work, daemon=True).start()
  while True:
    item = q.get()
    if item is stop:
      return
    if isinstance(item, BaseException):
      raise item
    yield item


def tracked_blocks(model, which: str):
  blocks = model.blocks()
  if which == 'all':
    return blocks
  if which == 'late':                                   # last stage and the last two blocks of the third
    return [(n, b) for n, b in blocks if n.startswith('stage3') or n in ('stage2.block7', 'stage2.block8')]
  names = set(which.split(','))
  return [(n, b) for n, b in blocks if n in names]


def param_groups(model, tracked_params, wd):
  decay, no_decay = [], []
  for n, p in model.named_parameters():
    if not p.requires_grad or any(p is t for t in tracked_params):
      continue
    (decay if p.dim() >= 2 else no_decay).append(p)
  return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


@torch.no_grad()
def evaluate(model, val_dir, device, limit=None):
  """Top-1 accuracy on the first `limit` validation images (all if None), and the number of images used."""
  model.eval()
  correct = total = 0
  for x, y in val_batches(val_dir, 500, device):
    if limit:
      x, y = x[:limit - total], y[:limit - total]
    with torch.autocast('cuda', torch.bfloat16):
      correct += (model(x).argmax(-1) == y).sum().item()
    total += len(y)
    if limit and total >= limit:
      break
  model.train()
  return correct / total, total


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--size', default='tiny')
  p.add_argument('--opt', default='adamw', choices=['adamw', 'muon'])
  p.add_argument('--lr', type=float, default=2e-3)
  p.add_argument('--muon_lr', type=float, default=2e-2)
  p.add_argument('--wd', type=float, default=0.05)
  p.add_argument('--steps', type=int, default=25020)
  p.add_argument('--warmup', type=int, default=1250)
  p.add_argument('--batch', type=int, default=512)
  p.add_argument('--mixup', type=float, default=0.8)
  p.add_argument('--track', default='all')
  p.add_argument('--train', default='/dev/shm/imagenet256')
  p.add_argument('--val', default='/marimo/work/data/imagenet/data')
  p.add_argument('--out', required=True)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--val_limit', type=int, default=10000)
  p.add_argument('--max_steps', type=int, default=0, help='stop early (benchmarks)')
  a = p.parse_args()
  dev = torch.device('cuda')
  torch.manual_seed(a.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  model = build_convnext(a.size).to(dev)
  blocks = tracked_blocks(model, a.track)
  layers = {f'{n}.{k}': getattr(b, k) for n, b in blocks for k in ('pw_gate', 'pw_value')}
  if a.opt == 'adamw':
    attr = GroupAttribution(layers, n_groups=1000, optimizer='adamw', lr=a.lr, betas=(0.9, 0.999), eps=1e-8,
                            weight_decay=a.wd, bias_weight_decay=0.0)
  else:
    attr = GroupAttribution(layers, n_groups=1000, optimizer='muon', lr=a.muon_lr, momentum=0.95, nesterov=True,
                            weight_decay=a.wd, bias_weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8,
                            bias_lr_scale=a.lr / a.muon_lr)   # tracked biases: AdamW at a.lr, like the other biases
  tracked = [t for l in layers.values() for t in (l.weight, l.bias)]
  groups = param_groups(model, tracked, a.wd)
  if a.opt == 'adamw':
    opts = [torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.999), eps=1e-8, fused=True)]
  else:
    mats = [p for p in groups[0]['params'] if p.dim() == 2 and p is not model.head.weight]
    rest = [p for p in groups[0]['params'] if not any(p is m for m in mats)]
    opts = [torch.optim.Muon(mats, lr=a.muon_lr, weight_decay=a.wd, momentum=0.95, nesterov=True),
            torch.optim.AdamW([{'params': rest, 'weight_decay': a.wd}, groups[1]], lr=a.lr, betas=(0.9, 0.999))]
  base = [[g['lr'] for g in o.param_groups] for o in opts]
  sched = lambda s: (s + 1) / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / (a.steps - a.warmup)))
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a) | {'tracked': list(layers), 'params': sum(p.numel() for p in model.parameters()),
                       'ckpt_steps': CKPT_STEPS}, open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
  loader = GPUTrainLoader(a.train, a.batch, dev, seed=a.seed)
  beta = torch.distributions.Beta(a.mixup, a.mixup) if a.mixup > 0 else None
  log, step, t0 = [], 0, time.time()

  def checkpoint(step):
    acc, n_val = evaluate(model, a.val, dev, None if step == a.steps else a.val_limit)
    torch.save(model.state_dict(), os.path.join(a.out, f'model_{step:05d}.pt'))
    torch.save(attr.compact_state(), os.path.join(a.out, f'attr_{step:05d}.pt'))   # scaled fp16 parts + fp32 remainder
    if step == 0:
      torch.save({k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in attr.tracked.items()}, os.path.join(a.out, 'attr_init.pt'))
    rec = {'step': step, 'val_acc': acc, 'n_val': n_val, 'sec': time.time() - t0}
    log.append(rec)
    json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'))
    print(json.dumps(rec), flush=True)

  checkpoint(0)
  run_loss = []
  while step < a.steps:
    for x, y, _ in prefetch(loader.epoch()):
      lam = float(beta.sample()) if beta is not None else 1.0
      y2 = y.flip(0)
      x = lam * x + (1 - lam) * x.flip(0)
      for o, b in zip(opts, base):
        for g, lr0 in zip(o.param_groups, b):
          g['lr'] = lr0 * sched(step)
      with torch.autocast('cuda', torch.bfloat16):
        logits = model(x)
      logp = F.log_softmax(logits.float(), -1)
      loss = -(lam * logp.gather(1, y[:, None]) + (1 - lam) * logp.gather(1, y2[:, None])).mean()
      for o in opts:
        o.zero_grad(set_to_none=True)
      loss.backward()
      attr.step(torch.stack([y, y2], 1), (a.lr if a.opt == 'adamw' else a.muon_lr) * sched(step),
                weights=(lam, 1 - lam))                   # Python floats: no host-device sync
      for o in opts:
        o.step()
      step += 1
      run_loss.append(loss.detach())
      if step % 100 == 0:
        print(f'step {step} loss {torch.stack(run_loss).mean().item():.4f} {time.time() - t0:.0f}s', flush=True)
        run_loss.clear()
      if step in CKPT_STEPS or step == a.steps or (a.max_steps and step == a.max_steps):
        checkpoint(step)
      if step >= a.steps or (a.max_steps and step >= a.max_steps):
        break
    if a.max_steps and step >= a.max_steps:
      break
  print('DONE', flush=True)


if __name__ == '__main__':
  main()
