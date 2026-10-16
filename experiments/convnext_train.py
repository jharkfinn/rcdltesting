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
accuracy. With --rand_groups, the same updates are also split over 1,000 random groups of class size (the training
labels shuffled across examples, rand_groups.npy; a shadow GroupAttribution, which leaves training unchanged), saved as
attrR_{step}.pt: a control for how much of a class's part is class structure and how much is SGD noise.

Replays (for causal tests of what the attribution shows): --snapshots saves snap_{step}.pt at every checkpoint (model,
optimizer states, the tracked layers' summed momentum and second moments, RNG states; about 1 MB for tiny). --init_from
snap_{step}.pt starts from one with a NEW attribution over 1,001 groups: W_0 = the snapshot weights, group 1000
('history') carries the old momentum, so every part is a write made after the snapshot. Run a control replay and an
intervention replay from the same snapshot and compare them (the random crops of the partial epoch differ from the
original run's, the same in both replays). --zero_loss_classes gives the listed classes' label terms loss weight 0 (an
intervention: their examples still pass through the model, but teach nothing for those labels). --extra_ckpts adds
checkpoints (attribution parts and model, no validation pass) for a denser record.
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
  p.add_argument('--resume_every', type=int, default=1000, help='steps between resume states (out/resume.pt)')
  p.add_argument('--rand_groups', action='store_true', help='also split the updates over random groups (attrR_*.pt)')
  p.add_argument('--snapshots', action='store_true', help='save replay snapshots snap_{step}.pt at every checkpoint')
  p.add_argument('--init_from', default=None, help='replay: start from a snapshot with a new attribution (history group)')
  p.add_argument('--zero_loss_classes', default='', help='comma-separated classes whose label terms get loss weight 0')
  p.add_argument('--extra_ckpts', default='', help='comma-separated extra checkpoint steps, or start:stop:every')
  a = p.parse_args()
  dev = torch.device('cuda')
  torch.manual_seed(a.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  model = build_convnext(a.size).to(dev)
  snap = torch.load(a.init_from, map_location=dev, weights_only=False) if a.init_from else None
  if snap is not None:                          # the new attribution's W_0 must be the snapshot's weights
    model.load_state_dict(snap['model'])
  extra = set()
  for tok in filter(None, a.extra_ckpts.split(',')):
    if ':' in tok:
      lo, hi, ev = (int(v) for v in tok.split(':'))
      extra.update(range(lo, hi + 1, ev))
    else:
      extra.add(int(tok))
  ckpts = set(CKPT_STEPS) | extra
  zero_cls = torch.tensor([int(c) for c in filter(None, a.zero_loss_classes.split(','))], dtype=torch.long, device=dev)
  blocks = tracked_blocks(model, a.track)
  layers = {f'{n}.{k}': getattr(b, k) for n, b in blocks for k in ('pw_gate', 'pw_value')}
  n_groups = 1001 if snap is not None else 1000     # replay: group 1000 holds the momentum from before the snapshot
  if a.opt == 'adamw':
    akw = dict(n_groups=n_groups, optimizer='adamw', lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.wd,
               bias_weight_decay=0.0)
  else:
    akw = dict(n_groups=n_groups, optimizer='muon', lr=a.muon_lr, momentum=0.95, nesterov=True, weight_decay=a.wd,
               bias_weight_decay=0.0, betas=(0.9, 0.999), eps=1e-8,
               bias_lr_scale=a.lr / a.muon_lr)                # tracked biases: AdamW at a.lr, like the other biases
  attr = GroupAttribution(layers, **akw)
  attr_r = GroupAttribution(layers, shadow=True, **akw) if a.rand_groups else None
  if snap is not None:
    attr.load_snapshot(snap['attr'], history_group=1000)
    if attr_r is not None:
      attr_r.load_snapshot(snap['attr'], history_group=1000)
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
                       'ckpt_steps': sorted(ckpts)}, open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
  loader = GPUTrainLoader(a.train, a.batch, dev, seed=a.seed)
  if attr_r is not None:                       # random group of example j: the label of example perm[j]
    rand_np = np.random.default_rng([a.seed, 777]).permutation(loader.labels)
    np.save(os.path.join(a.out, 'rand_groups.npy'), rand_np.astype(np.int16))
    rand_gpu = torch.from_numpy(rand_np.astype(np.int64)).to(dev)
  beta = torch.distributions.Beta(a.mixup, a.mixup) if a.mixup > 0 else None
  log, step, t0 = [], 0, time.time()

  def checkpoint(step):
    if step in extra and step not in CKPT_STEPS and step != a.steps:
      acc, n_val = None, 0                       # extra checkpoints: parts and model only
    else:
      acc, n_val = evaluate(model, a.val, dev, None if step == a.steps else a.val_limit)
    torch.save(model.state_dict(), os.path.join(a.out, f'model_{step:05d}.pt'))
    torch.save(attr.compact_state(), os.path.join(a.out, f'attr_{step:05d}.pt'))   # scaled fp16 parts + fp32 remainder
    if attr_r is not None:
      torch.save(attr_r.compact_state(), os.path.join(a.out, f'attrR_{step:05d}.pt'))
    if not os.path.exists(os.path.join(a.out, 'attr_init.pt')):   # step 0, or the first step of a replay
      torch.save({k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in attr.tracked.items()}, os.path.join(a.out, 'attr_init.pt'))
    if a.snapshots:
      torch.save({'step': step, 'model': model.state_dict(), 'opts': [o.state_dict() for o in opts],
                  'attr': attr.snapshot(), 'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),
                  'log': list(log)}, os.path.join(a.out, f'snap_{step:05d}.pt'))
    rec = {'step': step, 'val_acc': acc, 'n_val': n_val, 'sec': time.time() - t0}
    log.append(rec)
    json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'))
    print(json.dumps(rec), flush=True)

  def save_resume(step):
    """Training state for a restart on a new machine (sandboxes can end without notice): written to a temporary file
    and renamed, so a crash during the write leaves the previous state."""
    tmp = os.path.join(a.out, 'resume.pt.tmp')
    torch.save({'step': step, 'model': model.state_dict(), 'opts': [o.state_dict() for o in opts],
                'attr': attr.full_state(), 'attr_r': attr_r.full_state() if attr_r is not None else None,
                'log': log, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state()}, tmp)
    os.replace(tmp, os.path.join(a.out, 'resume.pt'))

  resume = os.path.join(a.out, 'resume.pt')
  if os.path.exists(resume):
    st = torch.load(resume, map_location=dev, weights_only=False)
    model.load_state_dict(st['model'])
    for o, os_ in zip(opts, st['opts']):
      o.load_state_dict(os_)
    attr.load_full_state(st['attr'])
    if attr_r is not None:
      attr_r.load_full_state(st['attr_r'])
    log[:] = st['log']
    torch.set_rng_state(st['torch_rng'].cpu())
    torch.cuda.set_rng_state(st['cuda_rng'].cpu())
    step = st['step']
    t0 -= log[-1]['sec'] if log else 0
    print(f'resumed at step {step}', flush=True)
  elif snap is not None:
    for o, os_ in zip(opts, snap['opts']):
      o.load_state_dict(os_)
    torch.set_rng_state(snap['torch_rng'].cpu())
    torch.cuda.set_rng_state(snap['cuda_rng'].cpu())
    step = snap['step']
    log[:] = [r for r in snap['log'] if r['step'] < step]
    t0 -= log[-1]['sec'] if log else 0
    print(f'replay from snapshot at step {step} ({a.init_from})', flush=True)
    checkpoint(step)                            # parts zero, attr_init.pt = the snapshot weights
  else:
    checkpoint(0)
  run_loss = []
  while step < a.steps:
    epoch, start = divmod(step, len(loader))
    for x, y, idx in prefetch(loader.epoch(epoch, start)):
      lam = float(beta.sample()) if beta is not None else 1.0
      y2 = y.flip(0)
      x = lam * x + (1 - lam) * x.flip(0)
      for o, b in zip(opts, base):
        for g, lr0 in zip(o.param_groups, b):
          g['lr'] = lr0 * sched(step)
      with torch.autocast('cuda', torch.bfloat16):
        logits = model(x)
      logp = F.log_softmax(logits.float(), -1)
      if len(zero_cls):                         # intervention: these classes' label terms teach nothing
        w1, w2 = (~torch.isin(y, zero_cls)).float()[:, None], (~torch.isin(y2, zero_cls)).float()[:, None]
        loss = -(lam * w1 * logp.gather(1, y[:, None]) + (1 - lam) * w2 * logp.gather(1, y2[:, None])).mean()
      else:
        loss = -(lam * logp.gather(1, y[:, None]) + (1 - lam) * logp.gather(1, y2[:, None])).mean()
      for o in opts:
        o.zero_grad(set_to_none=True)
      loss.backward()
      attr.step(torch.stack([y, y2], 1), (a.lr if a.opt == 'adamw' else a.muon_lr) * sched(step),
                weights=(lam, 1 - lam))                   # Python floats: no host-device sync
      if attr_r is not None:
        r = rand_gpu[idx.to(dev, non_blocking=True)]
        attr_r.step(torch.stack([r, r.flip(0)], 1), (a.lr if a.opt == 'adamw' else a.muon_lr) * sched(step),
                    weights=(lam, 1 - lam))
      for o in opts:
        o.step()
      step += 1
      run_loss.append(loss.detach())
      if step % 100 == 0:
        print(f'step {step} loss {torch.stack(run_loss).mean().item():.4f} {time.time() - t0:.0f}s', flush=True)
        run_loss.clear()
      if step in ckpts or step == a.steps or (a.max_steps and step == a.max_steps):
        checkpoint(step)
      if step % a.resume_every == 0 or step in CKPT_STEPS or step == a.steps:
        save_resume(step)
      if step >= a.steps or (a.max_steps and step >= a.max_steps):
        break
    if a.max_steps and step >= a.max_steps:
      break
  print('DONE', flush=True)


if __name__ == '__main__':
  main()
