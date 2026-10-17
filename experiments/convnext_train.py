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
original run's, the same in both replays; with --batch_crops in the original run and its replays, the crops are seeded
per batch and a control replay repeats the original run up to GPU nondeterminism). --freeze_gamma 1 (X3) keeps the
layer scale of every stage-1 block at its value (the snapshot's in a replay): requires_grad False and left out of the
optimizers, whose saved states are re-indexed without it (remap_optimizer_state); snapshots and resume states record the
frozen stages; a listed stage without layer scales is an error, and a fresh run (no snapshot, no resume state) warns
that the frozen scales stay at their initial value. The attribution is unaffected (it tracks pw_gate and pw_value
only). --zero_loss_classes gives the
listed classes' label terms loss weight 0 (an intervention: their examples still pass through the model, but teach
nothing for those labels). --extra_ckpts adds
checkpoints (attribution parts and model, no validation pass) for a denser record.

Data specs (rare classes, relabelled images): --spec spec.npz trains on the images in `keep` with the full label table
`labels` (and replaces the pixels of the images in `noise`, if present, by fixed smooth noise); with a per-image `group` table (and `group_class`, one entry per group), the same updates are also split over
those fine groups (own images of a rare class, subsets, decoy sets) by a shadow GroupAttribution, saved as
attrF_{step}.pt. --track_head adds the classifier to the tracked maps (AdamW only).

Storage and safety (medium runs; the defaults keep the earlier behaviour): --track none tracks no map (a block name that
matches no block is an error); --attr_steps limits the checkpoints that write attr_/attrR_/attrF_ files (attr_init.pt
is written as before); --resume_at_ckpts 0 writes resume.pt only every --resume_every steps and at the last step (1:
also at every fixed checkpoint step). A non-finite running loss at a 100-step print ends the run with exit code 3,
before that step's checkpoint or resume state. log.json records the peak GPU memory so far (max_mem_gb) at every
checkpoint. --device cpu runs the same loop on the CPU (tests).
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
NONFINITE_EXIT = 3                                      # exit code of a run stopped by a non-finite loss


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
  """'all', 'late', 'none' or comma-separated block names; ValueError if a name matches no block."""
  blocks = model.blocks()
  if which == 'all':
    return blocks
  if which == 'none':
    return []
  if which == 'late':                                   # last stage and the last two blocks of the third
    return [(n, b) for n, b in blocks if n.startswith('stage3') or n in ('stage2.block7', 'stage2.block8')]
  names = {t.strip() for t in which.split(',') if t.strip()}
  unknown = sorted(names - {n for n, _ in blocks})
  if unknown or not names:
    raise ValueError(f'--track {which!r}: no block named {unknown or which!r} (blocks {blocks[0][0]} .. '
                     f'{blocks[-1][0]}; or all, late, none)')
  return [(n, b) for n, b in blocks if n in names]


def parse_steps(spec) -> set:
  """Comma-separated steps, each a step or start:stop:every (stop included)."""
  out = set()
  for tok in filter(None, str(spec or '').split(',')):
    if ':' in tok:
      lo, hi, ev = (int(v) for v in tok.split(':'))
      out.update(range(lo, hi + 1, ev))
    else:
      out.add(int(tok))
  return out


def resume_due(step: int, every: int, at_ckpts: bool, last: int) -> bool:
  """Whether resume.pt is written after `step`: every `every` steps and at the last step, and with at_ckpts (the
  default) at every fixed checkpoint step too."""
  return step % every == 0 or step == last or bool(at_ckpts and step in CKPT_STEPS)


def report_loss(step: int, losses, sec: float) -> float:
  """The 100-step print of the running loss; a non-finite one ends the run (exit NONFINITE_EXIT)."""
  loss = torch.stack(losses).mean().item()
  print(f'step {step} loss {loss:.4f} {sec:.0f}s', flush=True)
  if not math.isfinite(loss):
    print(f'ABORT: non-finite running loss at step {step}', flush=True)
    raise SystemExit(NONFINITE_EXIT)
  return loss


def max_mem_gb(dev) -> float:
  """Peak GPU memory allocated so far by this process, GiB (0 on the CPU)."""
  return torch.cuda.max_memory_allocated(dev) / 2 ** 30 if torch.device(dev).type == 'cuda' else 0.0


def param_groups(model, tracked_params, wd, skip=None):
  """skip: names of parameters to leave out; None leaves out those with requires_grad False (frozen ones)."""
  decay, no_decay = [], []
  for n, p in model.named_parameters():
    left_out = (not p.requires_grad) if skip is None else n in skip
    if left_out or any(p is t for t in tracked_params):
      continue
    (decay if p.dim() >= 2 else no_decay).append(p)
  return [{'params': decay, 'weight_decay': wd}, {'params': no_decay, 'weight_decay': 0.0}]


def parse_stages(spec) -> list:
  return sorted({int(t) for t in str(spec or '').split(',') if t.strip()})


def freeze_layer_scales(model, stages) -> list:
  """X3: requires_grad False on the layer scale (gamma) of every block in the given stages; returns their names. A
  frozen parameter is left out of the optimizers (param_groups), so it keeps its value (no step, no weight decay).
  ValueError if stages are given but none of them has a layer scale (e.g. a stage index out of range)."""
  names = []
  for n, p in model.named_parameters():
    parts = n.split('.')                                 # stages.{i}.{j}.gamma
    if parts[0] == 'stages' and parts[-1] == 'gamma' and int(parts[1]) in stages:
      p.requires_grad_(False)
      names.append(n)
  if stages and not names:
    raise ValueError(f'--freeze_gamma {stages}: no layer scales in those stages')
  return names


def frozen_names(model, stages) -> set:
  """Names of the layer scales freeze_layer_scales(model, stages) freezes, without changing anything."""
  return {n for n, _ in model.named_parameters()
          if n.split('.')[0] == 'stages' and n.endswith('.gamma') and int(n.split('.')[1]) in stages}


def optimizer_groups(model, tracked, opt, wd, skip=None) -> list:
  """The parameter groups of each optimizer the trainer builds, in its order: [AdamW groups] for adamw, [Muon groups,
  AdamW groups] for muon (Muon: 2-D matrices other than the head; AdamW: the rest). skip as in param_groups."""
  groups = param_groups(model, tracked, wd, skip)
  if opt == 'adamw':
    return [groups]
  mats = [p for p in groups[0]['params'] if p.dim() == 2 and p is not model.head.weight]
  rest = [p for p in groups[0]['params'] if not any(p is m for m in mats)]
  return [[{'params': mats}], [{'params': rest, 'weight_decay': wd}, groups[1]]]


def build_optimizers(model, tracked, opt, lr, muon_lr, wd, fused=True) -> list:
  og = optimizer_groups(model, tracked, opt, wd)
  if opt == 'adamw':
    return [torch.optim.AdamW(og[0], lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=fused)]
  return [torch.optim.Muon(og[0][0]['params'], lr=muon_lr, weight_decay=wd, momentum=0.95, nesterov=True),
          torch.optim.AdamW(og[1], lr=lr, betas=(0.9, 0.999))]


def remap_optimizer_state(sd, old_groups, new_groups):
  """An optimizer state dict saved for parameter groups old_groups, re-indexed for new_groups: the same groups with
  some parameters left out (now frozen) or added (no state yet: the optimizer starts them fresh). Hyperparameters of
  each group come from the saved dict."""
  assert len(sd['param_groups']) == len(old_groups) == len(new_groups), 'group structure differs'
  old_idx = {id(p): i for g_sd, g in zip(sd['param_groups'], old_groups) for i, p in zip(g_sd['params'], g['params'])}
  assert all(len(g_sd['params']) == len(g['params']) for g_sd, g in zip(sd['param_groups'], old_groups))
  state, groups, k = {}, [], 0
  for g_sd, g in zip(sd['param_groups'], new_groups):
    ids = []
    for p in g['params']:
      i = old_idx.get(id(p))
      if i is not None and i in sd['state']:
        state[k] = sd['state'][i]
      ids.append(k)
      k += 1
    groups.append({**g_sd, 'params': ids})
  return {'state': state, 'param_groups': groups}


def load_optimizer_states(opts, saved, model, tracked, opt, wd, saved_stages, stages):
  """Load saved optimizer states (from a snapshot or resume state written with the layer scales of saved_stages frozen)
  into opts (built by build_optimizers with those of stages frozen)."""
  if parse_stages(saved_stages) == parse_stages(stages):
    for o, sd in zip(opts, saved):
      o.load_state_dict(sd)
    return
  old = optimizer_groups(model, tracked, opt, wd, skip=frozen_names(model, parse_stages(saved_stages)))
  for o, sd, og in zip(opts, saved, old):
    o.load_state_dict(remap_optimizer_state(sd, og, o.param_groups))


@torch.no_grad()
def evaluate(model, val_dir, device, limit=None):
  """Top-1 accuracy on the first `limit` validation images (all if None), and the number of images used."""
  model.eval()
  correct = total = 0
  cuda = torch.device(device).type == 'cuda'
  for x, y in val_batches(val_dir, 500, device):
    if limit:
      x, y = x[:limit - total], y[:limit - total]
    with torch.autocast('cuda' if cuda else 'cpu', torch.bfloat16, enabled=cuda):
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
  p.add_argument('--track', default='all', help='all, late, none, or comma-separated block names (stage3.block2)')
  p.add_argument('--train', default='/dev/shm/imagenet256')
  p.add_argument('--val', default='/marimo/work/data/imagenet/data')
  p.add_argument('--out', required=True)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--val_limit', type=int, default=10000)
  p.add_argument('--max_steps', type=int, default=0, help='stop early (benchmarks)')
  p.add_argument('--resume_every', type=int, default=1000, help='steps between resume states (out/resume.pt)')
  p.add_argument('--resume_at_ckpts', type=int, default=1, choices=[0, 1],
                 help='1: also write resume.pt at every fixed checkpoint step; 0: only every --resume_every steps and '
                      'at the last step')
  p.add_argument('--attr_steps', default='', help='checkpoint steps that write attr_/attrR_/attrF_ files (comma-'
                 'separated, or start:stop:every; default every checkpoint); attr_init.pt is written as before')
  p.add_argument('--rand_groups', action='store_true', help='also split the updates over random groups (attrR_*.pt)')
  p.add_argument('--snapshots', action='store_true', help='save replay snapshots snap_{step}.pt at every checkpoint')
  p.add_argument('--init_from', default=None, help='replay: start from a snapshot with a new attribution (history group)')
  p.add_argument('--zero_loss_classes', default='', help='comma-separated classes whose label terms get loss weight 0')
  p.add_argument('--extra_ckpts', default='', help='comma-separated extra checkpoint steps, or start:stop:every')
  p.add_argument('--batch_crops', action='store_true',
                 help='crop draws seeded per batch, so resumes and replays repeat the original crops (use the same '
                      'setting in the original run and its replays)')
  p.add_argument('--spec', default='', help='data spec (.npz): keep (images to train on), labels (full label table, '
                 'relabelled images), and optionally group / group_class for a fine-group shadow split saved as '
                 'attrF_{step}.pt (rare classes, own-image groups, decoy sets)')
  p.add_argument('--track_head', action='store_true', help='also track the classifier (head) with the same groups')
  p.add_argument('--freeze_gamma', default='', help='comma-separated stage indices whose blocks keep their layer scale '
                 '(gamma) fixed: requires_grad False and left out of the optimizers (X3; e.g. 1, with --init_from)')
  p.add_argument('--device', default='cuda', help='cuda, or cpu for tests')
  a = p.parse_args()
  dev = torch.device(a.device)
  cuda = dev.type == 'cuda'
  torch.manual_seed(a.seed)
  torch.backends.cuda.matmul.allow_tf32 = True
  torch.backends.cudnn.allow_tf32 = True
  model = build_convnext(a.size).to(dev)
  snap = torch.load(a.init_from, map_location=dev, weights_only=False) if a.init_from else None
  if snap is not None:                          # the new attribution's W_0 must be the snapshot's weights
    model.load_state_dict(snap['model'])
  extra = parse_steps(a.extra_ckpts)
  ckpts = set(CKPT_STEPS) | extra
  attr_steps = parse_steps(a.attr_steps) if a.attr_steps else None    # None: every checkpoint
  if attr_steps is not None and attr_steps - ckpts - {a.steps, a.max_steps}:
    p.error(f'--attr_steps: {sorted(attr_steps - ckpts - {a.steps, a.max_steps})} are not checkpoint steps')
  zero_cls = torch.tensor([int(c) for c in filter(None, a.zero_loss_classes.split(','))], dtype=torch.long, device=dev)
  try:
    blocks = tracked_blocks(model, a.track)
  except ValueError as e:
    p.error(str(e))
  layers = {f'{n}.{k}': getattr(b, k) for n, b in blocks for k in ('pw_gate', 'pw_value')}
  if a.track_head:
    assert a.opt == 'adamw', 'head tracking: AdamW only (under Muon the head is an AdamW parameter of the trainer)'
    layers['head'] = model.head                    # (B, d) -> (B, 1000): per-example gradients delta_i h_i^T
  spec = dict(np.load(a.spec)) if a.spec else None
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
  attr_f = None                                 # fine groups (a shadow); a replay (--init_from) uses the spec's data only
  if spec is not None and 'group' in spec and snap is None:
    attr_f = GroupAttribution(layers, shadow=True, **(akw | {'n_groups': int(len(spec['group_class']))}))
  if snap is not None:
    attr.load_snapshot(snap['attr'], history_group=1000)
    if attr_r is not None:
      attr_r.load_snapshot(snap['attr'], history_group=1000)
  tracked = [t for l in layers.values() for t in (l.weight, l.bias)]
  try:
    frozen = freeze_layer_scales(model, parse_stages(a.freeze_gamma))
  except ValueError as e:
    p.error(str(e))
  if frozen:
    print(f'frozen layer scales: {frozen}', flush=True)
    if snap is None and not os.path.exists(os.path.join(a.out, 'resume.pt')):
      print('WARNING: --freeze_gamma on a fresh run keeps those layer scales at their initial value '
            f'(max |gamma| {float(dict(model.named_parameters())[frozen[0]].abs().max()):g}); '
            'meant for replays (--init_from)', flush=True)
  opts = build_optimizers(model, tracked, a.opt, a.lr, a.muon_lr, a.wd, fused=cuda)
  base = [[g['lr'] for g in o.param_groups] for o in opts]
  sched = lambda s: (s + 1) / a.warmup if s < a.warmup else 0.5 * (1 + math.cos(math.pi * (s - a.warmup) / (a.steps - a.warmup)))
  os.makedirs(a.out, exist_ok=True)
  json.dump(vars(a) | {'tracked': list(layers), 'params': sum(p.numel() for p in model.parameters()),
                       'ckpt_steps': sorted(ckpts)}, open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
  if spec is not None:
    np.savez(os.path.join(a.out, 'spec.npz'), **spec)
    loader = GPUTrainLoader(a.train, a.batch, dev, seed=a.seed, batch_seeded=a.batch_crops, keep=spec['keep'],
                            labels=spec['labels'], noise=spec.get('noise'))
    if attr_f is not None:
      assert len(spec['group']) == len(loader.labels), 'spec built for another packed training set'
      assert (spec['group'][spec['keep']] >= 0).all(), 'every kept image needs a fine group'
      fine_gpu = torch.from_numpy(spec['group'].astype(np.int64)).to(dev)    # -1 for dropped images (never drawn)
  else:
    loader = GPUTrainLoader(a.train, a.batch, dev, seed=a.seed, batch_seeded=a.batch_crops)
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
    if attr_steps is None or step in attr_steps:
      torch.save(attr.compact_state(), os.path.join(a.out, f'attr_{step:05d}.pt'))  # scaled fp16 parts, fp32 remainder
      if attr_r is not None:
        torch.save(attr_r.compact_state(), os.path.join(a.out, f'attrR_{step:05d}.pt'))
      if attr_f is not None:
        torch.save(attr_f.compact_state(), os.path.join(a.out, f'attrF_{step:05d}.pt'))
    if not os.path.exists(os.path.join(a.out, 'attr_init.pt')):   # step 0, or the first step of a replay
      torch.save({k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in attr.tracked.items()}, os.path.join(a.out, 'attr_init.pt'))
    if a.snapshots:
      torch.save({'step': step, 'model': model.state_dict(), 'opts': [o.state_dict() for o in opts],
                  'attr': attr.snapshot(), 'torch_rng': torch.get_rng_state(),
                  'cuda_rng': torch.cuda.get_rng_state() if cuda else None,
                  'log': list(log), 'batch_crops': a.batch_crops, 'freeze_gamma': a.freeze_gamma},
                 os.path.join(a.out, f'snap_{step:05d}.pt'))
    rec = {'step': step, 'val_acc': acc, 'n_val': n_val, 'sec': time.time() - t0,
           'max_mem_gb': round(max_mem_gb(dev), 3)}
    log.append(rec)
    json.dump(log, open(os.path.join(a.out, 'log.json'), 'w'))
    print(json.dumps(rec), flush=True)

  def save_resume(step):
    """Training state for a restart on a new machine (sandboxes can end without notice): written to a temporary file
    and renamed, so a crash during the write leaves the previous state."""
    tmp = os.path.join(a.out, 'resume.pt.tmp')
    torch.save({'step': step, 'model': model.state_dict(), 'opts': [o.state_dict() for o in opts],
                'attr': attr.full_state(), 'attr_r': attr_r.full_state() if attr_r is not None else None,
                'attr_f': attr_f.full_state() if attr_f is not None else None,
                'log': log, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state() if cuda else None, 'freeze_gamma': a.freeze_gamma}, tmp)
    os.replace(tmp, os.path.join(a.out, 'resume.pt'))

  resume = os.path.join(a.out, 'resume.pt')
  if os.path.exists(resume):
    st = torch.load(resume, map_location=dev, weights_only=False)
    model.load_state_dict(st['model'])
    load_optimizer_states(opts, st['opts'], model, tracked, a.opt, a.wd, st.get('freeze_gamma', ''), a.freeze_gamma)
    attr.load_full_state(st['attr'])
    if attr_r is not None:
      attr_r.load_full_state(st['attr_r'])
    if attr_f is not None:
      attr_f.load_full_state(st['attr_f'])
    log[:] = st['log']
    torch.set_rng_state(st['torch_rng'].cpu())
    if cuda and st['cuda_rng'] is not None:
      torch.cuda.set_rng_state(st['cuda_rng'].cpu())
    step = st['step']
    del st                                      # the loaded copies (GBs for medium) are not needed any more
    if cuda:
      torch.cuda.empty_cache()
    t0 -= log[-1]['sec'] if log else 0
    print(f'resumed at step {step}', flush=True)
  elif snap is not None:
    if snap.get('batch_crops', False) != a.batch_crops:
      print(f'WARNING: snapshot batch_crops={snap.get("batch_crops", False)}, this replay {a.batch_crops}: '
            'the crops will not repeat the original run', flush=True)
    load_optimizer_states(opts, snap['opts'], model, tracked, a.opt, a.wd, snap.get('freeze_gamma', ''),
                          a.freeze_gamma)               # a frozen layer scale's saved moments are dropped
    torch.set_rng_state(snap['torch_rng'].cpu())
    if cuda and snap['cuda_rng'] is not None:
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
      with torch.autocast('cuda' if cuda else 'cpu', torch.bfloat16, enabled=cuda):
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
      if attr_f is not None:
        f = fine_gpu[idx.to(dev, non_blocking=True)]
        attr_f.step(torch.stack([f, f.flip(0)], 1), (a.lr if a.opt == 'adamw' else a.muon_lr) * sched(step),
                    weights=(lam, 1 - lam))
      for o in opts:
        o.step()
      step += 1
      run_loss.append(loss.detach())
      if step % 100 == 0:                       # a non-finite running loss ends the run here (exit 3)
        report_loss(step, run_loss, time.time() - t0)
        run_loss.clear()
      if step in ckpts or step == a.steps or (a.max_steps and step == a.max_steps):
        checkpoint(step)
      if resume_due(step, a.resume_every, a.resume_at_ckpts, a.steps):
        save_resume(step)
      if step >= a.steps or (a.max_steps and step >= a.max_steps):
        break
    if a.max_steps and step >= a.max_steps:
      break
  print('DONE', flush=True)


if __name__ == '__main__':
  main()
