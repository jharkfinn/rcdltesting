"""E6 umbrella = bee, arm helpers for the queue (protocol v2; experiments/molab_umbrella.sh).

Arm A, learning-rate selection (lr_select): for every lr, tau and censoring of the taught run of P1 (tau.json) and the
  drift of its untaught run, drift = top-1 accuracy (percentage points) on V_ref at E1's step 25,020 minus that at the
  untaught run's last step (warm-up plus 800 steps). lr* = the smallest lr with tau <= 30 (not censored) and drift <=
  1.0; otherwise the smallest with tau <= 100 and drift <= 1.0; otherwise the smallest with tau <= 100 (flagged);
  otherwise none.
Arm E, random relabel triples (select_e): with rng = np.random.default_rng(2026), six times: c_i uniform over classes
  with E1 validation accuracy >= 0.4, not the target (bee), umbrella, any selection arm's source, their relatives
  (WordNet distance <= 4 or visual rank <= 10) or an earlier triple's classes; t_i uniform over classes with WordNet
  distance >= 10 and visual rank > 100 from c_i (both directions), E1 accuracy within 0.1 of bee's, and not excluded
  (a c_i without such a class is skipped and another is drawn); j_i = the best validation image of c_i (not the first
  8) by the step-1 rule (top-1 = c_i and p_{t_i} <= 0.01, ranked by |p_{c_i} - 0.45|; without the p condition if no
  image passes); A_i, B_i: classes far from both (WordNet >= 10, visual rank > 100, not relatives) with E1 accuracy
  closest to t_i's and to c_i's. Written as selection arms 'E0'-'E5' in the format of convnext_umbrella select (so
  convnext_umbrella run --sel <file> --arm E<i> needs no change).
Arm J, seed replicate (select_j): E1r's selection (convnext_umbrella select on E1r) with each arm's passing photos
  reordered so that photos also among E1's photos or reserve for the same source come first.
Arm I, leave-group-out continuations (lgo): from a primary taught run's state at tau (snap_tau.pt), --steps phase-2
  steps of frozen-v AdamW (the same optimizer state, the second moment frozen at the branch: v_frozen_c2 = 1 -
  beta2^t_branch, t_branch the step count in attr_0000.pt), loss 0.5 mean CE over 255 replay images, no attribution.
  Slots: 255 base images drawn class-first uniformly over the classes outside {t, s} u RELb u RELu, then image-
  uniformly (default_rng([seed, 3_000_001, step])); variants target, source and A replace the last 32 base slots by 32
  images of that class (its per-class permutation walk, as StratifiedReplay); crops from default_rng([seed, 3_000_000 +
  step]) for every variant. Writes lgo_result.json (m_P at tau and at the end) and DONE.
Photo re-verification at a branch point (verify): the branch state theta_w of (arm, replay spec, seed, lr) is computed
  or loaded with convnext_umbrella.branch (the namespace comes from convnext_umbrella's own run parser, so the branch
  key and the defaults are the runs'; when the branch file does not exist yet, the warm-up is trained here, so verify
  must not run under no_grad); each photo still top-1 for the source at theta_w keeps its slot, a photo that is not is
  replaced by the next reserve photo that is; the scores (top-1, top-10, p_source, p_target at theta_w) and
  replacements are recorded.
Queue helpers: photos (an arm's photo specs, or a verify file's), full_saves (tau and tau + phase2 of finished taught
  runs), lr_star.
"""

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from rcdl.attribution import GroupAttribution
from rcdl.convnext import build_convnext

N_TRIPLES, RNG_E = 6, 2026


# ------------------------------------------------------------------------------------------------- arm A

def lr_rule(rows: list, tau1: int = 30, tau2: int = 100, cap: float = 1.0) -> dict:
  """rows: [{'lr', 'tau', 'censored', 'drift'}]; the pre-registered choice of lr* (module docstring)."""
  rows = sorted(rows, key=lambda r: r['lr'])
  fast = lambda r, tmax: (not r['censored']) and r['tau'] is not None and r['tau'] <= tmax
  for tmax, capped, rule in ((tau1, True, f'tau <= {tau1}, drift <= {cap}'), (tau2, True, f'tau <= {tau2}, drift <= {cap}'),
                             (tau2, False, f'tau <= {tau2}, drift cap missed (flagged)')):
    ok = [r for r in rows if fast(r, tmax) and (not capped or r['drift'] <= cap)]
    if ok:
      return {'lr_star': ok[0]['lr'], 'rule': rule, 'flagged': not capped, 'rows': rows}
  return {'lr_star': None, 'rule': f'no lr reaches tau <= {tau2}', 'flagged': True, 'rows': rows}


@torch.no_grad()
def top1(model, x_u8, y, dev, batch: int = 250) -> float:
  from experiments.convnext_umbrella_analysis import Images
  imgs = Images([x_u8], dev, torch.float32)
  hit = 0
  for i in range(0, len(imgs), batch):
    hit += int((model(imgs.get(i, min(len(imgs), i + batch))).argmax(-1).cpu() == torch.as_tensor(y[i:i + batch])).sum())
  return 100.0 * hit / len(imgs)


def cmd_lr_select(a):
  from experiments.convnext_umbrella_analysis import load_vsets, steps_of
  dev = torch.device(a.device)
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  vs = load_vsets(a.vsets)['V_ref']
  model = build_convnext('tiny').to(dev).eval()
  model.load_state_dict(torch.load(os.path.join(a.e1, 'model_25020.pt'), map_location=dev))
  acc0 = top1(model, vs['x'], vs['y'], dev)
  rows = []
  for d in sorted(os.listdir(a.runs)):
    taught, untaught = os.path.join(a.runs, d, 'taught'), os.path.join(a.runs, d, 'untaught')
    if not (os.path.exists(os.path.join(taught, 'DONE')) and os.path.exists(os.path.join(untaught, 'DONE'))):
      continue
    cfg = json.load(open(os.path.join(taught, 'config.json')))
    tj = json.load(open(os.path.join(taught, 'tau.json')))
    last = max(steps_of(untaught, 'model'))
    model.load_state_dict(torch.load(os.path.join(untaught, f'model_{last:04d}.pt'), map_location=dev))
    acc = top1(model, vs['x'], vs['y'], dev)
    rows.append({'lr': cfg['lr'], 'tau': tj['tau'], 'censored': tj['censored'], 'acc_untaught': acc,
                 'untaught_step': last, 'drift': acc0 - acc})
  out = lr_rule(rows) | {'acc_theta0': acc0, 'n_V_ref': len(vs['y'])}
  json.dump(out, open(a.out + '.tmp', 'w'), indent=1)
  os.replace(a.out + '.tmp', a.out)
  print(json.dumps(out), flush=True)


# ------------------------------------------------------------------------------------------------- arm E

def relatives_of(c: int, wn: np.ndarray, rank: np.ndarray) -> list:
  return [int(k) for k in np.flatnonzero((wn[c] <= 4) | (rank[c] <= 10)) if k != c]


def draw_triples(acc: np.ndarray, wn: np.ndarray, rank: np.ndarray, excluded, acc_ref: float, n: int = N_TRIPLES,
                 seed: int = RNG_E, acc_min: float = 0.4, acc_tol: float = 0.1) -> list:
  """[(c_i, t_i)] by the arm-E rule (module docstring); excluded: classes never drawn."""
  rng = np.random.default_rng(seed)
  used = set(int(k) for k in excluded)
  out = []
  while len(out) < n:
    cs = [k for k in range(len(acc)) if acc[k] >= acc_min and k not in used]
    if not cs:
      raise ValueError(f'only {len(out)} triples can be drawn')
    c = int(rng.choice(cs))
    ts = [k for k in range(len(acc)) if k != c and k not in used and wn[c, k] >= 10 and wn[k, c] >= 10
          and min(rank[c, k], rank[k, c]) > 100 and abs(acc[k] - acc_ref) <= acc_tol]
    if not ts:                                           # no admissible target: this source is skipped
      used.add(c)
      continue
    t = int(rng.choice(ts))
    out.append((c, t))
    used |= {c, t} | set(relatives_of(c, wn, rank)) | set(relatives_of(t, wn, rank))
  return out


def far_pair(acc, wn, rank, t: int, c: int, rel: list) -> tuple:
  """(A, B): classes far from t and c with E1 accuracy closest to t's and to c's (convnext_umbrella select)."""
  far = [k for k in range(len(acc)) if min(wn[k, t], wn[k, c]) >= 10 and min(rank[k, t], rank[k, c]) > 100
         and k not in set(rel) | {c, t}]
  A = min(far, key=lambda k: (abs(acc[k] - acc[t]), k))
  B = min([k for k in far if k != A], key=lambda k: (abs(acc[k] - acc[c]), k))
  return int(A), int(B)


def step1_rule(rows: list, c: int) -> tuple:
  """(ordered candidates, rule): top-1 = c and p_target <= 0.01, ranked by |p_c - 0.45|; without the p condition when
  no image passes."""
  strict = [r for r in rows if r['top1'] == c and r['p_target'] <= 0.01]
  loose = [r for r in rows if r['top1'] == c]
  ok, rule = (strict, 'top1 = source, p_target <= 0.01') if strict else (loose, 'top1 = source')
  return sorted(ok, key=lambda r: abs(r['p_source'] - 0.45)), rule


@torch.no_grad()
def class_accuracy(model, val_dir: str, dev) -> np.ndarray:
  from rcdl.imagenet import val_batches
  correct, count = np.zeros(1000), np.zeros(1000)
  for x, y in val_batches(val_dir, 500, dev):
    hit = (model(x).argmax(-1) == y).cpu().numpy()
    np.add.at(correct, y.cpu().numpy(), hit)
    np.add.at(count, y.cpu().numpy(), 1)
  return correct / np.maximum(count, 1)


@torch.no_grad()
def cmd_select_e(a):
  from experiments.convnext_umbrella import normalize, val_images, val_index
  dev = torch.device(a.device)
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  model = build_convnext('tiny').to(dev).eval()
  model.load_state_dict(torch.load(os.path.join(a.run, 'model_25020.pt'), map_location=dev))
  acc = class_accuracy(model, a.val, dev)
  wn, rank = np.load(a.wordnet)['dist'], np.load(a.visual)['rank_dist']
  sel = json.load(open(a.sel))
  bee = sel['target']
  umb = sel['source_requested']
  excl = {bee, umb} | set(relatives_of(bee, wn, rank)) | set(relatives_of(umb, wn, rank))
  for arm in sel['arms'].values():
    excl |= {arm['source'], arm['A'], arm['B']} | set(arm['rel_source'])
  triples = draw_triples(acc, wn, rank, excl, acc[bee])
  _, vlabels = val_index(a.val)
  arms = {}
  for i, (c, t) in enumerate(triples):
    vidx = np.flatnonzero(vlabels == c)[8:]
    x = normalize(val_images(a.val, vidx), dev)
    p = torch.cat([F.softmax(model(x[j:j + 256]), -1) for j in range(0, len(x), 256)]).cpu().numpy()
    rows = [{'photo': f'val:{int(g)}', 'p_source': float(pr[c]), 'p_target': float(pr[t]), 'top1': int(pr.argmax()),
             'top10': [int(k) for k in np.argsort(-pr)[:10]]} for g, pr in zip(vidx, p)]
    ok, rule = step1_rule(rows, c)
    if not ok:
      raise ValueError(f'triple {i}: no validation image of class {c} is top-1')
    rel_t, rel_c = relatives_of(t, wn, rank), relatives_of(c, wn, rank)
    A, B = far_pair(acc, wn, rank, t, c, rel_t + rel_c)
    arms[f'E{i}'] = {'source': c, 'target': t, 'rule': rule + ' (arm E, rng 2026)', 'photos': ok[:1], 'reserve': ok[1:5],
                     'n_val_candidates': int(len(vidx)), 'n_passing': len(ok), 'A': A, 'B': B, 'rel_target': rel_t,
                     'rel_source': rel_c, 'acc': {'source': float(acc[c]), 'target': float(acc[t]), 'A': float(acc[A]),
                                                  'B': float(acc[B]), 'mean': float(acc.mean())},
                     'excluded_from_eval': sorted({int(g) for g in np.flatnonzero(vlabels == c)[:8]}
                                                  | {int(g) for g in np.flatnonzero(vlabels == t)[:8]}
                                                  | {int(r['photo'].split(':')[1]) for r in ok})}
  out = {'rng': RNG_E, 'target': bee, 'acc_target': float(acc[bee]), 'excluded_classes': sorted(excl), 'arms': arms}
  json.dump(out, open(a.out + '.tmp', 'w'), indent=1)
  os.replace(a.out + '.tmp', a.out)
  print({k: (v['source'], v['target'], v['photos'][0]['photo'], v['A'], v['B']) for k, v in arms.items()}, flush=True)


# ------------------------------------------------------------------------------------------------- arm J

def order_j(sel_r: dict, sel_e1: dict) -> dict:
  """E1r's selection with each arm's photos + reserve reordered: those also among E1's photos or reserve for the same
  source first (stable), so that photos passing in both models are preferred."""
  out = json.loads(json.dumps(sel_r))
  for name, arm in out['arms'].items():
    e1 = [x for x in sel_e1['arms'].values() if x['source'] == arm['source']]
    both = {r['photo'] for x in e1 for r in x.get('photos', []) + x.get('reserve', [])}
    cand = arm.get('photos', []) + arm.get('reserve', [])
    cand = sorted(cand, key=lambda r: r['photo'] not in both)
    n = len(arm.get('photos', []))
    arm['photos'], arm['reserve'] = cand[:n], cand[n:]
    arm['n_in_both'] = sum(r['photo'] in both for r in cand)
  return out


def cmd_select_j(a):
  out = order_j(json.load(open(a.sel_r)), json.load(open(a.sel_e1)))
  json.dump(out, open(a.out + '.tmp', 'w'), indent=1)
  os.replace(a.out + '.tmp', a.out)
  print({k: (v['source'], v['n_in_both'], [r['photo'] for r in v['photos']]) for k, v in out['arms'].items()}, flush=True)


# ------------------------------------------------------------------------------------- branch re-verification

def umbrella_args(argv: list):
  """The namespace that convnext_umbrella's run subcommand builds from argv (its defaults included), without running."""
  import sys
  import experiments.convnext_umbrella as U
  got, orig, old_argv = {}, U.cmd_run, sys.argv
  U.cmd_run = lambda ns: got.setdefault('a', ns)
  argv = [x for x in argv if x != '--']
  sys.argv = ['convnext_umbrella', 'run'] + argv + ([] if '--out' in argv else ['--out', 'unused'])
  try:
    U.main()
  finally:
    U.cmd_run, sys.argv = orig, old_argv
  return got['a']


def replace_photos(photos: list, reserve: list, n: int) -> tuple:
  """photos, reserve: [{'photo', 'ok'}] in selection order. Each photo with ok keeps its slot, one without is replaced by
  the next reserve photo with ok (dropped when none is left). Returns (specs, [(replaced, replacement or None)])."""
  spare = iter([r for r in reserve if r['ok']])
  out, swaps = [], []
  for r in photos[:n]:
    if r['ok']:
      out.append(r['photo'])
      continue
    new = next(spare, None)
    swaps.append((r['photo'], None if new is None else new['photo']))
    if new is not None:
      out.append(new['photo'])
  return out, swaps


def cmd_verify(a):
  """Not under no_grad as a whole: convnext_umbrella.branch trains the warm-up when the branch file does not exist yet
  (seeds 1 and 2, the arm-E and S2 branches), and its backward needs gradients."""
  import experiments.convnext_umbrella as U
  from rcdl.imagenet import GPUTrainLoader
  if os.path.exists(a.out):
    print('\n'.join(json.load(open(a.out))['photos']), flush=True)
    return
  ua = umbrella_args(a.umbrella_args)
  dev = torch.device(getattr(a, 'device', None) or 'cuda')
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  torch.manual_seed(0)                                   # as convnext_umbrella cmd_run before its branch call
  sel = json.load(open(ua.sel))['arms'][ua.arm]
  model = build_convnext('tiny').to(dev)
  br = U.branch(ua, sel, model, GPUTrainLoader(ua.train, 256, dev, seed=0), dev)
  model.load_state_dict(br['model'])
  model.eval()
  t, s = sel['target'], sel['source']

  @torch.no_grad()
  def score(r):
    p = F.softmax(model(U.photo_tensor(r['photo'], ua.val, dev, ua.train))[0].double(), -1)
    return {'photo': r['photo'], 'top1_w': int(p.argmax()), 'top10_w': [int(c) for c in torch.argsort(-p)[:10]],
            'p_source_w': float(p[s]), 'p_target_w': float(p[t]), 'ok': int(p.argmax()) == s}
  rows = [score(r) for r in sel['photos'][:a.n]]
  res_rows = [score(r) for r in sel.get('reserve', [])]
  photos, swaps = replace_photos(rows, res_rows, a.n)
  out = {'arm': ua.arm, 'seed': ua.seed, 'lr': ua.lr, 'branch_key': br['key'], 'photos': photos, 'replaced': swaps,
         'scores': rows, 'reserve_scores': res_rows}
  json.dump(out, open(a.out + '.tmp', 'w'), indent=1)
  os.replace(a.out + '.tmp', a.out)
  print('\n'.join(photos), flush=True)


# ------------------------------------------------------------------------------------------------- arm I

def class_index(labels: np.ndarray) -> list:
  order = np.argsort(labels, kind='stable')
  bounds = np.searchsorted(labels[order], np.arange(labels.max() + 2))
  return [order[bounds[c]:bounds[c + 1]] for c in range(len(bounds) - 1)]


def lgo_indices(by_class, rest_classes, variant_class, seed: int, step: int, n: int = 255, k: int = 32) -> np.ndarray:
  """Global indices of one continuation batch (module docstring): the base draws do not depend on the variant."""
  rng = np.random.default_rng([seed, 3_000_001, step])
  cls = rng.choice(np.asarray(rest_classes, np.int64), n)
  out = [int(by_class[c][rng.integers(len(by_class[c]))]) for c in cls]
  if variant_class is not None:
    pool = by_class[variant_class]
    for j, p in enumerate(range(step * k, (step + 1) * k)):
      cycle, pos = divmod(p, len(pool))
      out[n - k + j] = int(pool[np.random.default_rng([seed, variant_class, cycle]).permutation(len(pool))[pos]])
  return np.array(out, np.int64)


def frozen_v_continuation(maps: dict, snap_attr: dict, t_freeze: int, lr: float, wd: float,
                          betas=(0.9, 0.999), dtype=torch.float32) -> GroupAttribution:
  """The frozen-v AdamW of the fine-tuning runs, continued from a snapshot taken after the freeze: one group (all
  momentum in it), v from the snapshot, and the bias correction of v frozen at the freeze step t_freeze."""
  opt = GroupAttribution(maps, n_groups=1, optimizer='adamw', lr=lr, betas=betas, eps=1e-8, weight_decay=wd,
                         bias_weight_decay=0.0, dtype=dtype)
  opt.load_snapshot(snap_attr, history_group=0)
  for tr in opt.tracked.values():
    tr.v_frozen_c2 = 1 - betas[1] ** t_freeze
  return opt


def cmd_lgo(a):
  from experiments.convnext_umbrella import photo_tensor, tracked_maps
  from rcdl.imagenet import GPUTrainLoader
  out_dir = a.out
  os.makedirs(out_dir, exist_ok=True)
  if os.path.exists(os.path.join(out_dir, 'DONE')):
    print('already done', flush=True)
    return
  dev = torch.device(a.device)
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  torch.manual_seed(0)
  cfg = json.load(open(os.path.join(a.from_run, 'config.json')))
  assert cfg['mode'] == 'taught' and cfg['opt'] == 'frozen_v', 'arm I continues primary (frozen-v, taught) runs'
  tau = json.load(open(os.path.join(a.from_run, 'tau.json')))['tau']
  sel = json.load(open(a.sel or cfg['sel']))['arms'][cfg['arm']]
  t, s = sel['target'], sel['source']
  model = build_convnext('tiny').to(dev)
  snap = torch.load(os.path.join(a.from_run, 'snap_tau.pt'), map_location=dev, weights_only=False)
  assert snap['step'] == tau
  model.load_state_dict(snap['model'])
  model.train()
  maps = tracked_maps(model)
  att0 = torch.load(os.path.join(a.from_run, 'attr_0000.pt'), map_location='cpu', weights_only=False)
  t_freeze = int(next(iter(att0.values()))['t'])
  opt = frozen_v_continuation(maps, snap['attr'], t_freeze, cfg['lr'], cfg['wd'])
  tracked = {id(p) for l in maps.values() for p in (l.weight, l.bias)}
  for p in model.parameters():
    if id(p) not in tracked:
      p.requires_grad_(False)
  loader = GPUTrainLoader(a.train, 256, dev, seed=0)
  by_class = class_index(loader.labels)
  drop = {t, s} | set(sel['rel_target']) | set(sel['rel_source'])
  rest = [c for c in range(len(by_class)) if c not in drop]
  vc = {'base': None, 'target': t, 'source': s, 'A': sel['A']}[a.variant]
  xP = photo_tensor(cfg['photo'], a.val, dev, a.train)
  seed = cfg['seed']

  @torch.no_grad()
  def margin():
    z = model(xP)[0]
    return float(z[t] - z[s])
  m0, log = margin(), open(os.path.join(out_dir, 'log.jsonl'), 'w')
  for step in range(tau, tau + a.steps):
    idx = lgo_indices(by_class, rest, vc, seed, step)
    x = loader.decode_crop(idx, np.random.default_rng([seed, 3_000_000 + step]))
    y = loader.labels_gpu[torch.from_numpy(idx).to(dev)]
    loss = 0.5 * F.cross_entropy(model(x), y)
    loss.backward()
    opt.step(torch.zeros(len(idx), dtype=torch.long, device=dev), cfg['lr'])
    log.write(json.dumps({'step': step + 1, 'loss': float(loss), 'm_P': margin()}) + '\n')
    log.flush()
  log.close()
  res = {'from_run': a.from_run, 'photo': cfg['photo'], 'seed': seed, 'variant': a.variant, 'tau': tau,
         'steps': a.steps, 'm_P_tau': m0, 'm_P_end': margin()}
  json.dump(res, open(os.path.join(out_dir, 'lgo_result.json'), 'w'), indent=1)
  open(os.path.join(out_dir, 'DONE'), 'w').write('done\n')
  print(json.dumps(res), flush=True)


# ------------------------------------------------------------------------------------------------- queue helpers

def full_saves(runs, phase2: int) -> list:
  """Sorted tau and tau + phase2 of the runs with a tau.json."""
  out = set()
  for r in runs:
    f = os.path.join(r, 'tau.json')
    if os.path.exists(f):
      tau = json.load(open(f))['tau']
      out |= {tau, tau + phase2}
  return sorted(out)


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  sub = p.add_subparsers(dest='cmd', required=True)
  dev = 'cuda' if torch.cuda.is_available() else 'cpu'
  q = sub.add_parser('lr_select')
  q.add_argument('--runs', required=True, help='directory with one lr<lr>/{taught, untaught} pair per lr')
  q.add_argument('--e1', required=True)
  q.add_argument('--vsets', required=True)
  q.add_argument('--device', default=dev)
  q.add_argument('--out', required=True)
  q = sub.add_parser('select_e')
  q.add_argument('--run', required=True, help='E1 run directory (model_25020.pt)')
  q.add_argument('--sel', required=True, help="the primary selection.json (classes to exclude)")
  q.add_argument('--val', required=True)
  q.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  q.add_argument('--visual', default='docs/data/imagenet_dinov2_classdist.npz')
  q.add_argument('--device', default=dev)
  q.add_argument('--out', required=True)
  q = sub.add_parser('select_j')
  q.add_argument('--sel_r', required=True, help="E1r's selection.json")
  q.add_argument('--sel_e1', required=True, help="E1's selection.json")
  q.add_argument('--out', required=True)
  q = sub.add_parser('lgo')
  q.add_argument('--from_run', required=True, help='a primary taught run (snap_tau.pt)')
  q.add_argument('--variant', choices=['base', 'target', 'source', 'A'], required=True)
  q.add_argument('--steps', type=int, default=500)
  q.add_argument('--sel', default=None)
  q.add_argument('--val', required=True)
  q.add_argument('--train', default='/dev/shm/imagenet256')
  q.add_argument('--device', default=dev)
  q.add_argument('--out', required=True)
  q = sub.add_parser('verify', help='re-verify the photos at their branch point (prints the photo specs)')
  q.add_argument('--n', type=int, default=5)
  q.add_argument('--device', default=dev)
  q.add_argument('--out', required=True)
  q.add_argument('umbrella_args', nargs=argparse.REMAINDER, help="-- <convnext_umbrella run arguments>")
  q = sub.add_parser('photos', help='print the photo specs of an arm (or of a verify file), one per line')
  q.add_argument('--sel', default=None)
  q.add_argument('--arm', default=None)
  q.add_argument('--verified', default=None)
  q.add_argument('--n', type=int, default=5)
  q = sub.add_parser('full_saves', help='print tau,tau+phase2,... of finished taught runs')
  q.add_argument('--phase2', type=int, default=500)
  q.add_argument('runs', nargs='*')
  q = sub.add_parser('lr_star', help='print lr* (or nothing)')
  q.add_argument('--file', required=True)
  a = p.parse_args()
  if a.cmd == 'photos':
    specs = json.load(open(a.verified))['photos'] if a.verified else \
        [r['photo'] for r in json.load(open(a.sel))['arms'][a.arm]['photos']]
    for spec in specs[:a.n]:
      print(spec)
  elif a.cmd == 'full_saves':
    print(','.join(str(v) for v in full_saves(a.runs, a.phase2)))
  elif a.cmd == 'lr_star':
    v = json.load(open(a.file)).get('lr_star')
    print('' if v is None else v)
  else:
    {'lr_select': cmd_lr_select, 'select_e': cmd_select_e, 'select_j': cmd_select_j, 'lgo': cmd_lgo,
     'verify': cmd_verify}[a.cmd](a)


if __name__ == '__main__':
  main()
