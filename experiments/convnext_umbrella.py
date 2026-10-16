"""E6: umbrella = bee with exact attribution, fine-tuning part (protocol v2, docs/EXACT_HISTORY.md).

The essay taught its medium ConvNeXt that one umbrella photo is a bee and found an off-ramp between the umbrella and bee
routes. Here the tiny E1 model is taught the same way while ordinary training continues, and every weight change is
split exactly into the photo's writes, each class's writes and the earlier momentum ('hist').

Subcommands:
  select  pre-registered choice of the photos (P1-P5), the unrelated replay classes A and B, the relatives, and the
          evaluation sets; writes selection.json.
  run     one arm: --mode taught | untaught | solo, --opt frozen_v | standard.

Branch point. Every run starts from a branch state shared by the runs with the same replay spec, replay seed and lr:
E1's final weights and optimizer state (snap_25020.pt from experiments/convnext_snapshots.py), then 300 replay-only
steps of standard AdamW (all parameters, pretraining weight decay, lr rising 0 -> lr over 100 steps). The fine-tune
attribution starts there: W_0 = the branch weights, the branch momentum in group 1001 ('hist'), v from the branch.

Fine-tuning. Tracked: pw_gate, pw_value and pw_down of all 18 blocks and the head (98,468 parameters); groups 0-999
are the replay classes, 1000 the photo, 1001 hist. frozen_v (primary): AdamW with the second moments frozen at the
branch (update lr * mhat / (sqrt(v_branch) + eps): linear in the per-group momenta and the same map in the taught and
untaught runs) and the untracked parameters frozen, so the parts cover the whole change. standard: v keeps updating
and the untracked parameters train with torch AdamW from the branch state.
Shadows: S shadow attributions with 10 groups (0 the photo, 1-8 random slot groups of the replay batch, 9 hist); the
slot groups depend only on (replay seed, step, shadow), so they are the same in the taught and untaught runs.
Phase 1 (taught): the photo plus 255 replay images, loss 0.5 CE(photo, target) + 0.5 mean replay CE, until the photo's
top-1 is the target with p >= 0.5 (tau), at most --phase1_max steps. Phase 2: --phase2 replay-only steps.
Untaught: the same stream and replay loss with no photo. Solo: the photo alone, CE to the target, until the flip.

Saves (out/): config.json, log.jsonl (every step: the photo's 1,000 logits, loss), model_{s}.pt and sums_{s}.pt
(GroupAttribution.group_sums over P, hist, BEE, UMB, A, B, RELb, RELu, OTH) at every phase-1 step and every --every
steps, shadow compact states at the same 25-step points and at tau, the owner's compact state at --full_saves (default
0, tau, tau + phase2; untaught: every 100), snap_tau.pt (for leave-group-out continuations), DONE.
"""

import argparse
import fcntl
import glob
import hashlib
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.convnext_train import param_groups
from rcdl.attribution import GroupAttribution
from rcdl.convnext import build_convnext
from rcdl.imagenet import MEAN, STD, GPUTrainLoader, StratifiedReplay, val_batches

PHOTO, HIST = 1000, 1001


# ----------------------------------------------------------------------------------------------- data and selection

def val_index(val_dir: str):
  """[(parquet file, row)] in global validation order, and the labels."""
  import pyarrow.parquet as pq
  rows, labels = [], []
  for f in sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True)):
    lab = pq.read_table(f, columns=['label']).column('label').to_pylist()
    rows += [(f, i) for i in range(len(lab))]
    labels += lab
  return rows, np.array(labels)


def val_images(val_dir: str, indices) -> torch.Tensor:
  """uint8 (N, 3, 224, 224): shorter side 256 (antialiased bilinear), center crop 224, as val_batches and the atlas."""
  import pyarrow.parquet as pq
  from torchvision.io import ImageReadMode, decode_image
  rows, _ = val_index(val_dir)
  out, tables = [], {}
  for g in indices:
    f, i = rows[int(g)]
    if f not in tables:
      tables[f] = pq.read_table(f, columns=['image']).column('image')
    b = tables[f][i].as_py()['bytes']
    im = decode_image(torch.frombuffer(bytearray(b), dtype=torch.uint8), mode=ImageReadMode.RGB)
    h, w = im.shape[-2:]
    s = 256 / min(h, w)
    im = F.interpolate(im[None].float(), size=(max(256, round(h * s)), max(256, round(w * s))), mode='bilinear',
                       antialias=True, align_corners=False)[0]
    H, W = im.shape[-2:]
    top, left = (H - 224) // 2, (W - 224) // 2
    out.append(im[:, top:top + 224, left:left + 224].round().clamp(0, 255).to(torch.uint8))
  return torch.stack(out)


def normalize(u8: torch.Tensor, dev) -> torch.Tensor:
  return (u8.to(dev).float() - MEAN.to(dev)[:, None, None]) / STD.to(dev)[:, None, None]


def photo_tensor(spec: str, val_dir: str, dev) -> torch.Tensor:
  """'val:<global index>', 'noise:<seed>' (N(0,1) in normalized space) or 'shuffle:<global index>' (the photo with
  its pixel positions permuted by a fixed permutation, the same for the three channels). (1, 3, 224, 224)."""
  kind, arg = spec.split(':')
  if kind == 'noise':
    g = torch.Generator().manual_seed(int(arg))
    return torch.randn(1, 3, 224, 224, generator=g).to(dev)
  x = normalize(val_images(val_dir, [int(arg)]), dev)
  if kind == 'shuffle':
    perm = torch.from_numpy(np.random.default_rng(0).permutation(224 * 224)).to(dev)
    x = x.reshape(1, 3, -1)[:, :, perm].reshape(1, 3, 224, 224)
  elif kind != 'val':
    raise ValueError(spec)
  return x


def relatives(c: int, wn: np.ndarray, rank: np.ndarray) -> list:
  return [int(k) for k in np.flatnonzero((wn[c] <= 4) | (rank[c] <= 10)) if k != c]


@torch.no_grad()
def cmd_select(a):
  dev = torch.device('cuda')
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  model = build_convnext('tiny').to(dev)
  model.load_state_dict(torch.load(os.path.join(a.run, 'model_25020.pt'), map_location=dev))
  model.eval()
  correct, count = np.zeros(1000), np.zeros(1000)
  for x, y in val_batches(a.val, 500, dev):
    hit = (model(x).argmax(-1) == y).cpu().numpy()
    np.add.at(correct, y.cpu().numpy(), hit)
    np.add.at(count, y.cpu().numpy(), 1)
  acc = correct / np.maximum(count, 1)
  wn = np.load(a.wordnet)['dist']
  rank = np.load(a.visual)['rank_dist']
  s, t = a.source, a.target
  rel_t, rel_s = relatives(t, wn, rank), relatives(s, wn, rank)
  _, labels = val_index(a.val)
  src = np.flatnonzero(labels == s)
  cand = src[8:]                                       # the first 8 (parquet order) are the atlas probe images
  p = F.softmax(model(normalize(val_images(a.val, cand), dev)), -1).cpu().numpy()
  rows = []
  for g, pr in zip(cand, p):
    rows.append({'index': int(g), 'p_source': float(pr[s]), 'p_target': float(pr[t]), 'top1': int(pr.argmax()),
                 'top10': [int(k) for k in np.argsort(-pr)[:10]]})
  ok = [r for r in rows if r['top1'] == s and r['p_target'] <= 0.01]
  rule = 'top1 = source, p_target <= 0.01'
  if len(ok) < 5:
    ok, rule = [r for r in rows if r['top1'] == s], 'top1 = source (p_target condition dropped)'
  ok.sort(key=lambda r: abs(r['p_source'] - 0.45))
  far = [k for k in range(1000) if min(wn[k, t], wn[k, s]) >= 10 and min(rank[k, t], rank[k, s]) > 100
         and k not in rel_t + rel_s + [s, t]]
  A = min(far, key=lambda k: abs(acc[k] - acc[t]))
  B = min([k for k in far if k != A], key=lambda k: abs(acc[k] - acc[s]))
  excluded = {int(g) for g in src[:8]} | {int(g) for g in np.flatnonzero(labels == t)[:8]} | {r['index'] for r in ok}
  sel = {'source': s, 'target': t, 'rule': rule, 'photos': ok[:5], 'reserve': ok[5:10], 'A': int(A), 'B': int(B),
         'acc': {'source': float(acc[s]), 'target': float(acc[t]), 'A': float(acc[A]), 'B': float(acc[B]),
                 'mean': float(acc.mean())},
         'rel_target': rel_t, 'rel_source': rel_s, 'excluded_from_eval': sorted(excluded),
         'n_candidates': len(rows), 'n_passing': len(ok)}
  json.dump(sel, open(a.out, 'w'), indent=1)
  print(json.dumps({k: v for k, v in sel.items() if k not in ('reserve', 'excluded_from_eval')}, indent=1))


# -------------------------------------------------------------------------------------------- optimizer states

def tracked_maps(model) -> dict:
  maps = {f'{n}.{k}': getattr(b, k) for n, b in model.blocks() for k in ('pw_gate', 'pw_value', 'pw_down')}
  maps['head'] = model.head
  return maps


def e1_states(model, snap: dict) -> dict:
  """Every parameter's AdamW state at E1's snapshot, keyed by named_parameters name: {'exp_avg', 'exp_avg_sq',
  'step'}. The gate and value maps come from the attribution snapshot (momentum summed over classes), the rest from
  torch's AdamW, whose state is indexed in the order of convnext_train.param_groups (asserted by shape)."""
  pname = {id(p): n for n, p in model.named_parameters()}
  out = {}
  for bn, b in model.blocks():
    for k in ('pw_gate', 'pw_value'):
      s, lin = snap['attr'][f'{bn}.{k}'], getattr(b, k)
      t = int(s['t'])
      out[pname[id(lin.weight)]] = {'exp_avg': s['m'], 'exp_avg_sq': s['v'], 'step': t}
      out[pname[id(lin.bias)]] = {'exp_avg': s['mb'], 'exp_avg_sq': s['vb'], 'step': t}
  tracked = [p for n, p in model.named_parameters() if n in out]
  order = [p for g in param_groups(model, tracked, 0.0) for p in g['params']]
  st = snap['opts'][0]['state']
  assert len(st) == len(order), f'optimizer state has {len(st)} entries, expected {len(order)}'
  for i, p in enumerate(order):
    e = st[i]
    assert e['exp_avg'].shape == p.shape, pname[id(p)]
    out[pname[id(p)]] = {'exp_avg': e['exp_avg'], 'exp_avg_sq': e['exp_avg_sq'], 'step': int(e['step'])}
  assert set(out) == set(pname.values()), 'every parameter needs a state'
  return out


def adamw_from_states(named_params, states: dict, lr: float, wd: float) -> torch.optim.AdamW:
  """torch AdamW over named_params (weight decay on tensors with 2 or more dims) with the given states loaded."""
  decay = [p for n, p in named_params if p.dim() >= 2]
  rest = [p for n, p in named_params if p.dim() < 2]
  opt = torch.optim.AdamW([{'params': decay, 'weight_decay': wd}, {'params': rest, 'weight_decay': 0.0}], lr=lr,
                          betas=(0.9, 0.999), eps=1e-8)
  for n, p in named_params:
    s = states[n]
    opt.state[p] = {'step': torch.tensor(float(s['step'])), 'exp_avg': s['exp_avg'].to(p).clone(),
                    'exp_avg_sq': s['exp_avg_sq'].to(p).clone()}
  return opt


def states_of(opt: torch.optim.AdamW, named_params) -> dict:
  return {n: {'exp_avg': opt.state[p]['exp_avg'].detach().clone(), 'exp_avg_sq': opt.state[p]['exp_avg_sq'].detach().clone(),
              'step': int(opt.state[p]['step'])} for n, p in named_params}


# ------------------------------------------------------------------------------------------------------ branch

def replay_spec(a, sel) -> dict:
  spec = {}
  for tok in a.replay.split(','):
    c, n = tok.split(':')
    spec[int(sel[c]) if c in ('A', 'B') else int(c)] = int(n)
  return spec


def rest_classes(a, sel, spec) -> list:
  excl = set(spec)
  for tok in filter(None, a.rest_exclude.split(',')):
    if tok == 'rel':
      excl |= set(sel['rel_target']) | set(sel['rel_source']) | {sel['source'], sel['target']}
    else:
      excl.add(int(tok))
  return [c for c in range(1000) if c not in excl]


def branch(a, sel, model, loader, dev) -> dict:
  """The shared branch state (weights and AdamW state after the replay-only warm-up), computed once per (replay spec,
  replay seed, lr) under a file lock and saved atomically."""
  key = hashlib.sha1(json.dumps([a.replay, a.rest, a.rest_exclude, a.seed, a.lr, a.warmup, a.snapshot]).encode()).hexdigest()[:12]
  path = os.path.join(a.branch_dir, f'branch_{key}.pt')
  os.makedirs(a.branch_dir, exist_ok=True)
  with open(path + '.lock', 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    if not os.path.exists(path):
      snap = torch.load(a.snapshot, map_location=dev, weights_only=False)
      assert int(snap['step']) == 25020, 'the branch starts from E1 step 25,020'
      model.load_state_dict(snap['model'])
      named = list(model.named_parameters())
      opt = adamw_from_states(named, e1_states(model, snap), a.lr, a.wd)
      spec = replay_spec(a, sel)
      rep = StratifiedReplay(loader, spec, a.rest, rest_classes(a, sel, spec), seed=a.seed + 10_000)
      model.train()
      for s in range(a.warmup):
        for g in opt.param_groups:
          g['lr'] = a.lr * min(1.0, (s + 1) / a.warmup_ramp)
        x, y, _ = rep.batch(s)
        loss = 0.5 * F.cross_entropy(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
      torch.save({'model': model.state_dict(), 'states': states_of(opt, named), 'key': key,
                  'warmup': a.warmup, 'lr': a.lr}, path + '.tmp')
      os.replace(path + '.tmp', path)
      print(f'branch {key}: warm-up of {a.warmup} steps done', flush=True)
  return torch.load(path, map_location=dev, weights_only=False)


# ------------------------------------------------------------------------------------------------------- run

def cmd_run(a):
  dev = torch.device('cuda')
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  torch.manual_seed(0)
  os.makedirs(a.out, exist_ok=True)
  if os.path.exists(os.path.join(a.out, 'DONE')):
    print('already done', flush=True)
    return
  sel = json.load(open(a.sel))
  model = build_convnext('tiny').to(dev)
  loader = GPUTrainLoader(a.train, 256, dev, seed=0)
  br = branch(a, sel, model, loader, dev)
  model.load_state_dict(br['model'])
  model.train()
  maps = tracked_maps(model)
  pname = {id(p): n for n, p in model.named_parameters()}
  snap = {}
  for n, l in maps.items():
    sw, sb = br['states'][pname[id(l.weight)]], br['states'][pname[id(l.bias)]]
    snap[n] = {'m': sw['exp_avg'], 'v': sw['exp_avg_sq'], 'mb': sb['exp_avg'], 'vb': sb['exp_avg_sq'], 't': sw['step']}
  hp = dict(optimizer='adamw', lr=a.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=a.wd, bias_weight_decay=0.0,
            dtype=torch.float32)
  attr = GroupAttribution(maps, n_groups=1002, **hp)
  shadows = [GroupAttribution(maps, n_groups=10, shadow=True, **hp) for _ in range(a.shadows)]
  attr.load_snapshot(snap, history_group=HIST)
  for sh in shadows:
    sh.load_snapshot(snap, history_group=9)
  tracked = {id(p) for l in maps.values() for p in (l.weight, l.bias)}
  untracked = [(n, p) for n, p in model.named_parameters() if id(p) not in tracked]
  opt_unt = None
  if a.opt == 'frozen_v':
    attr.freeze_second_moment()
    for sh in shadows:
      sh.freeze_second_moment()
    for _, p in untracked:
      p.requires_grad_(False)
  else:
    opt_unt = adamw_from_states(untracked, br['states'], a.lr, a.wd)
  spec = replay_spec(a, sel)
  rep = StratifiedReplay(loader, spec, a.rest, rest_classes(a, sel, spec), seed=a.seed) if a.mode != 'solo' else None
  photo = photo_tensor(a.photo, a.val, dev) if a.mode != 'untaught' else None
  target = torch.tensor([a.target if a.target >= 0 else sel['target']], device=dev)
  rel_t = [k for k in sel['rel_target'] if k not in (sel['source'], sel['A'], sel['B'])]
  rel_s = [k for k in sel['rel_source'] if k not in rel_t + [sel['target'], sel['A'], sel['B']]]
  named = {'P': [PHOTO], 'hist': [HIST], 'BEE': [sel['target']], 'UMB': [sel['source']], 'A': [sel['A']],
           'B': [sel['B']], 'RELb': rel_t, 'RELu': rel_s}
  used = {k for v in named.values() for k in v}
  named['OTH'] = [k for k in range(1000) if k not in used]
  assert sum(len(v) for v in named.values()) == 1002
  json.dump(vars(a) | {'replay_classes': {str(k): v for k, v in spec.items()}, 'sets': named,
                       'branch_key': br['key'], 'n_tracked': sum(l.weight.numel() + l.bias.numel() for l in maps.values()),
                       'n_untracked': sum(p.numel() for _, p in untracked)},
            open(os.path.join(a.out, 'config.json'), 'w'), indent=1)
  full_saves = {int(s) for s in filter(None, a.full_saves.split(','))}
  log = open(os.path.join(a.out, 'log.jsonl'), 'w')

  def save(s, full=False, shadow=False, tag=None):
    torch.save(model.state_dict(), os.path.join(a.out, f'model_{s:04d}.pt'))
    torch.save(attr.group_sums(named), os.path.join(a.out, f'sums_{s:04d}.pt'))
    if shadow:
      torch.save([sh.compact_state() for sh in shadows], os.path.join(a.out, f'shadow_{s:04d}.pt'))
    if full:
      torch.save(attr.compact_state(), os.path.join(a.out, f'attr_{s:04d}.pt'))
    if tag:
      torch.save({'step': s, 'model': model.state_dict(), 'attr': attr.snapshot(),
                  'opt_unt': None if opt_unt is None else opt_unt.state_dict()}, os.path.join(a.out, f'snap_{tag}.pt'))

  @torch.no_grad()
  def photo_logits():
    return model(photo)[0] if photo is not None else None

  torch.save({k: {'W0': tr.W0, 'b0': tr.b0} for k, tr in attr.tracked.items()}, os.path.join(a.out, 'attr_init.pt'))
  save(0, full=True, shadow=True)
  phase, tau, s, t0 = 1 if a.mode != 'untaught' else 2, None, 0, time.time()
  total = a.untaught_steps if a.mode == 'untaught' else None
  while True:
    if a.mode == 'untaught' and s >= total:
      break
    if a.mode != 'untaught' and phase == 2 and s >= tau + a.phase2:
      break
    if a.mode == 'solo' and phase == 2:
      break
    xs, ys, gs = [], [], []
    w = []
    if phase == 1:
      xs.append(photo), ys.append(target), gs.append(torch.tensor([PHOTO], device=dev))
    if rep is not None:
      xr, yr, _ = rep.batch(s)
      xs.append(xr), ys.append(yr), gs.append(yr)
    x, y, g = torch.cat(xs), torch.cat(ys), torch.cat(gs)
    n_rep = len(y) - (1 if phase == 1 else 0)
    wrow = torch.full((len(y),), 0.5 / max(n_rep, 1), device=dev)
    if phase == 1:
      wrow[0] = 0.5 if rep is not None else 1.0
    ce = F.cross_entropy(model(x), y, reduction='none')
    loss = (wrow * ce).sum()
    for o in ([opt_unt] if opt_unt is not None else []):
      o.zero_grad(set_to_none=True)
    loss.backward()
    attr.step(g, a.lr)
    for r, sh in enumerate(shadows):
      sg = torch.from_numpy(rep.slot_groups(s, r)).to(dev) if rep is not None else torch.zeros(0, dtype=torch.long, device=dev)
      if phase == 1:
        sg = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), sg])
      sh.step(sg, a.lr)
    if opt_unt is not None:
      opt_unt.step()
    s += 1
    z = photo_logits()
    rec = {'step': s, 'phase': phase, 'loss': float(loss), 'replay_ce': float(ce[-n_rep:].mean()) if n_rep else None}
    if z is not None:
      p = F.softmax(z, -1)
      rec |= {'top1': int(z.argmax()), 'p_target': float(p[target]), 'p_source': float(p[sel['source']]),
              'logits': [round(float(v), 4) for v in z]}
    log.write(json.dumps(rec) + '\n')
    log.flush()
    flipped = phase == 1 and z is not None and int(z.argmax()) == int(target) and float(F.softmax(z, -1)[target]) >= 0.5
    if phase == 1 and (flipped or s >= a.phase1_max):
      tau = s
      censored = not flipped
      json.dump({'tau': tau, 'censored': censored}, open(os.path.join(a.out, 'tau.json'), 'w'))
      save(s, full=True, shadow=True, tag='tau')
      print(f'tau = {tau}{" (censored)" if censored else ""}  {time.time() - t0:.0f}s', flush=True)
      phase = 2
      continue
    if phase == 1 or s % a.every == 0 or s in full_saves or (a.mode == 'untaught' and s % 100 == 0) \
       or (tau is not None and s == tau + a.phase2):
      full = s in full_saves or (a.mode == 'untaught' and s % 100 == 0) or (tau is not None and s == tau + a.phase2)
      save(s, full=full, shadow=(s % a.every == 0) or full)
    if s % 100 == 0:
      worst = max(attr.residual(k)['weight'] / max(attr.residual(k)['weight_scale'], 1e-12) for k in maps)
      print(f'step {s} phase {phase} loss {float(loss):.4f} parts residual {worst:.1e} {time.time() - t0:.0f}s', flush=True)
  save(s, full=True, shadow=True)
  log.close()
  open(os.path.join(a.out, 'DONE'), 'w').write(f'{s}\n')
  print('DONE', flush=True)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)
  q = sub.add_parser('select')
  q.add_argument('--run', required=True, help='E1 run directory (model_25020.pt)')
  q.add_argument('--val', required=True)
  q.add_argument('--source', type=int, default=879)
  q.add_argument('--target', type=int, default=309)
  q.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  q.add_argument('--visual', default='docs/data/imagenet_dinov2_classdist.npz')
  q.add_argument('--out', required=True)
  r = sub.add_parser('run')
  r.add_argument('--sel', required=True, help='selection.json')
  r.add_argument('--snapshot', required=True, help="E1's step-25,020 replay snapshot (convnext_snapshots)")
  r.add_argument('--val', required=True)
  r.add_argument('--train', default='/dev/shm/imagenet256')
  r.add_argument('--photo', default='', help="'val:<index>' | 'noise:<seed>' | 'shuffle:<index>'")
  r.add_argument('--target', type=int, default=-1, help='default: the selection target')
  r.add_argument('--mode', choices=['taught', 'untaught', 'solo'], required=True)
  r.add_argument('--opt', choices=['frozen_v', 'standard'], default='frozen_v')
  r.add_argument('--lr', type=float, required=True)
  r.add_argument('--wd', type=float, default=0.05)
  r.add_argument('--replay', default='309:32,879:32,A:32,B:32')
  r.add_argument('--rest', type=int, default=127)
  r.add_argument('--rest_exclude', default='', help="extra classes left out of the rest draws; 'rel' = the relatives")
  r.add_argument('--seed', type=int, default=0, help='replay stream seed')
  r.add_argument('--shadows', type=int, default=8)
  r.add_argument('--warmup', type=int, default=300)
  r.add_argument('--warmup_ramp', type=int, default=100)
  r.add_argument('--phase1_max', type=int, default=300)
  r.add_argument('--phase2', type=int, default=500)
  r.add_argument('--untaught_steps', type=int, default=800)
  r.add_argument('--every', type=int, default=25)
  r.add_argument('--full_saves', default='', help='extra steps with the full class-level state')
  r.add_argument('--branch_dir', required=True)
  r.add_argument('--out', required=True)
  a = p.parse_args()
  if a.cmd == 'select':
    cmd_select(a)
  else:
    if a.mode != 'untaught' and not a.photo:
      p.error('--photo is required for taught and solo runs')
    cmd_run(a)


if __name__ == '__main__':
  main()
