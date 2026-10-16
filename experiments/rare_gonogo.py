"""Rare-class experiment (protocol v2), Step 4: the go/no-go on a checkpoint of the rare runs (A0, B0 at step 12,500),
and the same numbers on common runs (C0, E1, E1r) as the ceiling.

Definitions (experiments/rare_common.py): V_c, V_est, Q, Far(c), Near10(c), N_own(c), Own_m(c), zeta_c, E_S(c).
Per class c (the 128 selected classes and the 16 relabel and noise controls; a relabel control is probed on its
source class's validation images and a noise control on its noise images, rare_common.probes, in place of V_c):
  top1, top5      accuracy of the label c on V_c (50 validation images)
  auroc_far       AUROC of zeta_c on V_c against zeta_c on the Q images of Far(c) (8 per class)
  auroc_near      the same against the Q images of Near10(c) (descriptive)
  E_own           first-order effect of Own_m(c) on mean_{V_est} zeta_c, over every tracked map (the gate and value
                  maps of all blocks and the head), from the fine split (attrF); also per region (stage0..3, head)
  E_null          the same effect of each N_own class's Own_m on c (13 values for selected classes)
  E_pass          E_own > the 95th percentile of E_null
  E_units_pass    the same without the head (unit maps only; descriptive: AdamW enlarges a rare head row's steps)
  E_class         first-order effect of c's whole class part (class split attr, every tracked map) and its null over
                  the N_own classes' class parts: on runs without a fine split (E1, E1r) the only own effect, and
                  descriptive (own = the whole class); in a rare run it equals E_own for a rare class (fold check);
                  E_class_units without the head (E1 and E1r track no head)
  X_own           (level-100 classes of the rare set) the exact excision effect of Own_m(c) from every tracked map on
                  mean_{V_est} zeta_c, against which E_own is a first-order approximation
  fit_T           (with --train) top-1 of the class's training label on center crops of its own training images
                  (relabelled or noise images for the controls in their rare run)
Go/no-go (rare run): over the level-100 classes of the run's rare set,
  median auroc_far >= 0.6  and  fraction with E_pass >= 0.5   -> GO, otherwise NO-GO.
--combine pools the level-100 rare classes of several result files (A0 and B0: 56 classes) for the decision, and
reports the common runs among them as the ceiling on the same classes.

Exactness reported: the fine parts folded by group_class against the class parts (max relative difference per map),
and E_own against X_own.

Usage (molab):
  python3 -m experiments.rare_gonogo --run /marimo/work/results/rare_A0 --step 12500 \
    --val /marimo/work/data/imagenet/data --out /marimo/work/results/rare_A0/gonogo_12500.json
  python3 -m experiments.rare_gonogo --combine A0.json B0.json E1.json E1r.json --out gonogo.json
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from experiments.rare_common import (Selection, auroc, by_region, class_gradients, clean_json, excise, fold,
                                     group_effects, load_npz, load_parts, load_probe_images, load_spec, logits_of,
                                     noise_images, own_groups, pct_rank, probe_kind, probes, tracked_modules,
                                     train_center_u8, zeta)
from rcdl.convnext import build_convnext

THRESH_AUROC, THRESH_FRAC = 0.6, 0.5


def summarize(recs, keys=('auroc_far', 'auroc_near', 'top1', 'top5', 'E_own', 'E_class', 'E_class_units', 'fit_T')):
  out = {'n': len(recs)}
  for k in keys:
    v = [r[k] for r in recs if r.get(k) is not None and np.isfinite(r[k])]
    if v:
      out[f'median_{k}'] = float(np.median(v))
      out[f'mean_{k}'] = float(np.mean(v))
  for k in ('E_pass', 'E_units_pass', 'E_class_pass'):
    v = [r[k] for r in recs if r.get(k) is not None]
    if v:
      out[f'frac_{k}'] = float(np.mean(v))
  return out


def verdict(recs):
  """The two criteria over level-100 rare classes."""
  l100 = [r for r in recs if r['condition'] == 'rare' and r['kind'] == 'selected' and r['level'] == 100]
  if not l100:
    return None
  med = float(np.median([r['auroc_far'] for r in l100]))
  pas = [r['E_pass'] for r in l100 if r.get('E_pass') is not None]
  frac = float(np.mean(pas)) if pas else float('nan')
  ok_a, ok_e = med >= THRESH_AUROC, bool(pas) and frac >= THRESH_FRAC
  pu = [r['E_units_pass'] for r in l100 if r.get('E_units_pass') is not None]
  return {'n_level100': len(l100), 'median_auroc_far': med, 'auroc_ok': bool(ok_a), 'frac_E_pass': frac,
          'n_E': len(pas), 'E_ok': bool(ok_e), 'decision': 'GO' if ok_a and ok_e else 'NO-GO',
          'frac_E_units_pass': float(np.mean(pu)) if pu else None,
          'line': (f"{'GO' if ok_a and ok_e else 'NO-GO'}: median AUROC_far {med:.3f} (>= {THRESH_AUROC}: "
                   f"{'yes' if ok_a else 'no'}); E_Own_m above the N_own 95th percentile for {frac:.0%} of "
                   f"{len(pas)} level-100 classes (>= {THRESH_FRAC:.0%}: {'yes' if ok_e else 'no'})")}


def cells(recs):
  out = {}
  for r in recs:
    key = f"{r['condition']}/{r['cell']}"
    out.setdefault(key, []).append(r)
  by_ls = {}
  for r in recs:
    if r['kind'] == 'selected':
      by_ls.setdefault(f"{r['condition']}/{r['stratum']}/{r['level']}", []).append(r)
  return {'by_set_cell': {k: summarize(v) for k, v in sorted(out.items())},
          'by_level_stratum': {k: summarize(v) for k, v in sorted(by_ls.items())}}


def analyse(model, sel: Selection, spec, parts_c, parts_f, V, Vpos, Q, Qy, run_set, device, train_x=None,
            excision=True, log=print, noise_x=None):
  """All per-class numbers (see the module docstring). V: uint8 validation images, Vpos[c] the rows of class c in V
  (50 per class, global validation order; relabel controls need their source class's rows); Q, Qy: the probe images
  and labels; noise_x: {noise control: its noise images} (rare_common.noise_images)."""
  def has(c):                                       # probe images available
    row = sel.rows[c]
    if row['kind'] == 'relabel':
      return row['source'] in Vpos
    return c in Vpos or (row['kind'] == 'noise' and noise_x is not None and c in noise_x)
  classes = [c for c in sorted(sel.rows) if has(c)]
  xs = {c: probes(sel, c, V, Vpos, noise_x) for c in classes}
  t0 = time.time()
  zQ = logits_of(model, Q, device)
  log(f'logits {time.time() - t0:.0f}s')
  keys = list(parts_f if parts_f is not None else parts_c)
  mods = tracked_modules(model, keys)
  ckeys = [k for k in parts_c if k in mods]
  grads = class_gradients(model, mods, [xs[c][1::2] for c in classes], classes, device)
  log(f'gradients {time.time() - t0:.0f}s')
  Ec = by_region(group_effects(parts_c, {k: grads[k] for k in ckeys}))
  Ef = by_region(group_effects(parts_f, grads)) if parts_f is not None else None
  log(f'effects {time.time() - t0:.0f}s')
  col = {c: j for j, c in enumerate(classes)}
  Qy = np.asarray(Qy)
  recs = []
  for c in classes:
    row = sel.rows[c]
    j, zc = col[c], logits_of(model, xs[c], device)
    s = zeta(zc, c).numpy()
    top5 = torch.topk(zc, 5, 1).indices
    sq = zeta(zQ, c).numpy()
    far, near = sel.far(c), sel.near10(c)
    r = {'cls': c, 'name': row['name'], 'kind': row['kind'], 'set': row['set'], 'stratum': row['stratum'],
         'level': row['level'], 'cell': sel.cell(c), 'condition': sel.condition(c, run_set),
         'probe': probe_kind(sel, c, noise_x), 'n_probe': int(len(xs[c])),
         'top1': float((zc.argmax(1) == c).float().mean()), 'top5': float((top5 == c).any(1).float().mean()),
         'auroc_far': auroc(s, sq[np.isin(Qy, far)]), 'auroc_near': auroc(s, sq[np.isin(Qy, near)]),
         'n_far': len(far)}
    nown = sel.n_own(c)
    tot = Ec['all'][:, j].cpu().numpy()
    r['E_class'] = float(tot[c])
    r['E_class_units'] = float(Ec['units'][c, j])        # comparable between runs with and without a tracked head
    if nown:
      null = tot[nown]
      r['E_class_null_p95'] = float(np.percentile(null, 95))
      r['E_class_pct'] = pct_rank(tot[c], null)
      r['E_class_pass'] = bool(tot[c] > np.percentile(null, 95))
    if Ef is not None:
      own = own_groups(spec, c)
      reg = {k: float(v[own, j].sum()) for k, v in Ef.items()}
      r['E_own'], r['E_own_region'] = reg['all'], reg
      r['n_own_groups'] = int(len(own))
      if nown:
        for tag, k_ in (('', 'all'), ('_units', 'units')):     # units: without the head (descriptive)
          null = np.array([float(Ef[k_][own_groups(spec, k), j].sum()) for k in nown])
          r[f'E{tag}_null'] = null.tolist()
          r[f'E{tag}_null_p95'] = float(np.percentile(null, 95))
          r[f'E{tag}_pct'] = pct_rank(reg[k_], null)
          r[f'E{tag}_pass'] = bool(reg[k_] > r[f'E{tag}_null_p95'])
    if train_x is not None and c in train_x:
      x, y = train_x[c]
      if len(x):
        r['fit_T'] = float((logits_of(model, x, device).argmax(1) == torch.as_tensor(y)).float().mean())
        r['n_T'] = int(len(x))
    recs.append(r)
  exact = []
  if excision and parts_f is not None:
    targets = [r for r in recs if r['condition'] == 'rare' and r['level'] == 100 and r['kind'] == 'selected']
    for r in targets:
      c = r['cls']
      x = xs[c][1::2]
      base = float(zeta(logits_of(model, x, device), c).mean())
      undo = excise(mods, parts_f, own_groups(spec, c), keys)
      try:
        cut = float(zeta(logits_of(model, x, device), c).mean())
      finally:
        undo()
      r['X_own'] = base - cut
      exact.append((r['E_own'], r['X_own']))
    log(f'excisions {time.time() - t0:.0f}s')
  ex = None
  if exact:
    e, x = np.array(exact).T
    nz = x != 0
    ex = {'n': len(e), 'corr_E_X': float(np.corrcoef(e, x)[0, 1]) if len(e) > 2 and e.std() > 0 and x.std() > 0
          else None, 'median_ratio_E_over_X': float(np.median(e[nz] / x[nz])) if nz.any() else None,
          'sign_agreement': float(np.mean(np.sign(e) == np.sign(x)))}
  return recs, ex


def fold_check(parts_f, parts_c, group_class, n_classes):
  """max |fold(attrF) - attr| / max |attr| per map (fp16 storage of both: expect ~1e-3 or below)."""
  folded = fold(parts_f, group_class, n_classes)
  out = {}
  for k, rec in parts_c.items():
    if k not in folded:
      continue
    C = rec['C'].float() / rec.get('C_scale', 1.0)
    Cb = rec['Cb'].float() / rec.get('Cb_scale', 1.0)
    d = max(float((folded[k]['C'] - C).abs().max()), float((folded[k]['Cb'] - Cb).abs().max()))
    out[k] = d / max(float(C.abs().max()), float(Cb.abs().max()), 1e-30)
  mx = max(out.values()) if out else None
  return {'max_rel': mx, 'ok': bool(mx is not None and mx < 2e-3), 'per_map': out}   # fp16 storage of both splits


def run_one(a):
  dev = torch.device(a.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
  torch.backends.cuda.matmul.allow_tf32 = False
  torch.backends.cudnn.allow_tf32 = False
  log = lambda s: print(s, flush=True)
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  state = torch.load(os.path.join(a.run, f'model_{a.step:05d}.pt'), map_location='cpu')
  n_classes = state['head.weight'].shape[0]
  model = build_convnext(cfg['size'], n_classes=n_classes)
  model.load_state_dict(state)
  model = model.to(dev).eval()
  sel = Selection(a.classes, a.wordnet, a.visual)
  spec = load_spec(a.run)
  run_set = a.set or (spec['meta']['run_set'] if spec is not None else None)
  parts_c = load_parts(os.path.join(a.run, f'attr_{a.step:05d}.pt'))
  pf = os.path.join(a.run, f'attrF_{a.step:05d}.pt')
  parts_f = load_parts(pf) if os.path.exists(pf) and spec is not None else None
  classes = sorted(sel.rows)
  V, Vpos, Q, Qy = load_probe_images(a.val, sel, a.val_cache, a.probe_cache, a.images)
  noise_x = noise_images([spec] + [load_npz(p) for p in a.noise_specs.split(',') if p], sel)
  log(f'images: V {tuple(V.shape)}, Q {tuple(Q.shape)}, noise controls with noise images {len(noise_x)}; '
      f'run set {run_set}; fine split {parts_f is not None}')
  train_x = None
  if a.train and spec is not None and os.path.exists(os.path.join(a.train, 'data.bin')):
    train_x = {}
    for c in classes:
      gids = own_groups(spec, c)
      idx = np.flatnonzero(np.isin(spec['group'], gids))
      train_x[c] = (train_center_u8(a.train, idx, spec.get('noise', ())), spec['labels'][idx])
    log('training crops loaded')
  recs, ex = analyse(model, sel, spec, parts_c, parts_f, V, Vpos, Q, Qy, run_set, dev, train_x,
                     excision=not a.no_excision, log=log, noise_x=noise_x)
  out = {'run': a.run, 'step': a.step, 'run_set': run_set, 'fine_split': parts_f is not None,
         'tracked_maps': len(parts_f if parts_f is not None else parts_c), 'head_tracked': 'head' in parts_c,
         'verdict': verdict(recs) if run_set in ('A', 'B') else None, 'excision_vs_first_order': ex,
         **cells(recs), 'classes': recs}
  if parts_f is not None and not a.no_fold:
    out['fold_check'] = fold_check(parts_f, parts_c, spec['group_class'], n_classes)
  if out['verdict'] is not None:
    out['line'] = out['verdict']['line']
  else:
    l100 = [r for r in recs if r['kind'] == 'selected' and r['level'] == 100]
    out['line'] = (f'CEILING (all common): median AUROC_far over the {len(l100)} level-100 classes '
                   f"{np.median([r['auroc_far'] for r in l100]):.3f}")
  return out


def combine(files):
  """Pooled decision over the level-100 rare classes of rare-run results; common runs give the ceiling."""
  res = [json.load(open(f)) for f in files]
  rare = [r for o in res if o.get('run_set') in ('A', 'B') for r in o['classes']]
  v = verdict(rare)
  out = {'files': list(files), 'pooled': v, 'per_run': {o['run']: o.get('verdict') for o in res}}
  ids = {r['cls'] for r in rare if r['condition'] == 'rare' and r['kind'] == 'selected' and r['level'] == 100}
  ceil = {}
  for o in res:
    rs = [r for r in o['classes'] if r['cls'] in ids]
    if o.get('run_set') not in ('A', 'B') and rs:
      ceil[o['run']] = summarize(rs)
  out['ceiling_same_classes'] = ceil
  runs = [o.get('verdict') for o in res if o.get('verdict')]
  out['line'] = (v['line'] if v else 'no rare runs') + (
    '' if all(r['decision'] == 'GO' for r in runs) else '  [a single run fails: see per_run]')
  return out


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--run', default='')
  p.add_argument('--step', type=int, default=12500)
  p.add_argument('--val', default='/marimo/work/data/imagenet/data')
  p.add_argument('--classes', default='docs/data/rare/rare_classes_v2.json')
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--visual', default='docs/data/imagenet_dinov2_classdist.npz')
  p.add_argument('--out', required=True)
  p.add_argument('--set', default=None, choices=['A', 'B', 'C'], help='override the run set (default: spec meta)')
  p.add_argument('--probe_cache', default='/dev/shm/probe_8.pt')
  p.add_argument('--val_cache', default='/marimo/work/rare_val_v2_src.pt', help='the same file as convnext_rare')
  p.add_argument('--noise_specs', default='/marimo/work/specs/spec_A.npz,/marimo/work/specs/spec_B.npz',
                 help='specs that flag the noise controls\' images (probes of noise controls in every run)')
  p.add_argument('--train', default='', help='packed training set (/dev/shm/imagenet256): adds fit_T')
  p.add_argument('--images', default='', help='pre-built images {V, Vlabels, Q, Qy} (tests)')
  p.add_argument('--device', default=None)
  p.add_argument('--no_excision', action='store_true')
  p.add_argument('--no_fold', action='store_true')
  p.add_argument('--combine', nargs='*', default=None, help='result files to pool (no other input needed)')
  a = p.parse_args()
  out = combine(a.combine) if a.combine else run_one(a)
  os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
  json.dump(clean_json(out), open(a.out, 'w'), indent=1)
  if not a.combine:
    for k, v in out['by_level_stratum'].items():
      print(k, json.dumps({x: round(y, 3) if isinstance(y, float) else y for x, y in v.items()}))
    if out.get('fold_check'):
      print('fold check max rel', out['fold_check']['max_rel'])
    print('excision vs first order', out['excision_vs_first_order'])
  print(out['line'], flush=True)


if __name__ == '__main__':
  main()
