"""Looping test tables: facts (facts_loops.json, facts_loops_v2.json, facts_loops_ctrl.json) and biographies (bios_loops.json).

Usage: python -m experiments.loops_analysis [--facts DIR] [--facts_v2 DIR] [--facts_ctrl DIR] [--bios DIR] [--out DIR]
Each DIR holds <name>/result.json. Run names end in _K{K}_s{seed}; the part before is the arm. Prints Markdown tables
(mean over seeds, then the per-seed values) and, with --out, writes summary.json with every number the tables use and
tables.md.

Facts (rcdl/facts.loop_probe). Round 1 (probe v1) and round 2 (probe v2, seeds 1-3) train the same configurations;
seeds 1 and 2 of round 2 repeat round 1, so their difference measures GPU nondeterminism. The main tables use round 2
when it is given, else round 1. Measures:
  gain(K) = acc(K) - acc(K = 1), same arm and seed (acc: fraction of the 4,096 evaluation facts recalled);
  controls: untied passes and wider layers are compared with the tied K = 1 run of the same seed;
  dilution margin(e) = acc with noise e propagated through the passes - acc with the same noise added after the clean
    passes (positive: the passes remove part of the push);
  tangent radius: the largest |eigenvalue| of the step Jacobian with the directions the LayerNorm ignores projected out
    (the plain radius is at least 1 by construction);
  trajectory: accuracy after pass k = 1 .. 16 of the trained map.
Biographies (rcdl/bios.loop_probe): final held-out and seen-template attribute accuracy and the gap seen - held-out,
plateau end, steps to 50% / 90% held-out accuracy, accuracy at K_test and at test step size a_test, and the per-pass
Jacobian medians.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict
from statistics import mean


def load(root):
  runs = defaultdict(dict)                                   # (arm, K) -> seed -> result
  if not root:
    return runs
  for f in sorted(glob.glob(os.path.join(root, '*', 'result.json'))):
    name = os.path.basename(os.path.dirname(f))
    m = re.match(r'(.+)_K(\d+)_s(\d+)$', name)
    if m:
      runs[(m.group(1), int(m.group(2)))][int(m.group(3))] = json.load(open(f))
  return runs


def fmt(xs, p=3):
  xs = [x for x in xs if x is not None]
  if not xs:
    return '-'
  m = f'{mean(xs):.{p}f}'
  return m if len(xs) == 1 else f'{m} ({" / ".join(f"{x:.{p}f}" for x in xs)})'


def arm_key(arm):
  """Sort arms: GeGLU first, then cos by width and kernel sigma."""
  m = re.match(r'(geglu|cos)_w(\d+)(?:_s(\d+))?(.*)$', arm)
  return (m.group(1) != 'geglu', int(m.group(2)), int(m.group(3) or 0), m.group(4)) if m else (2, 0, 0, arm)


def table(header, rows):
  return ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)] + ['| ' + ' | '.join(r) + ' |' for r in rows]


def jac_rows(runs, label):
  lines = [f'\n{label}: per-pass Jacobian medians at the trained K (mean over seeds; the plain radius is at least 1 by '
           'construction, the tangent radius is not)\n']
  rows, out = [], {}
  for (arm, K), seeds in sorted(runs.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1])):
    per = [r['loop_probe']['jacobian_by_pass'] for r in seeds.values() if 'loop_probe' in r]
    if not per:
      continue
    for k in range(min(len(p) for p in per)):
      keys = [c for c in ('spectral_radius_median', 'spectral_norm_median', 'tangent_radius_median',
                          'tangent_contracting_share_median', 'rel_step_median', 'norm_ratio_median') if c in per[0][k]]
      row = {c: [p[k][c] for p in per if c in p[k]] for c in keys}
      out[f'{arm}_K{K}_pass{k + 1}'] = row
      rows.append([arm, str(K), str(k + 1)] + [fmt(row.get(c, []), 3) for c in
                  ('spectral_radius_median', 'spectral_norm_median', 'tangent_radius_median',
                   'tangent_contracting_share_median', 'rel_step_median')])
  lines += table(['arm', 'K', 'pass', 'ρ(J) ≥ 1', '‖J‖₂', 'tangent ρ', 'contracting share', 'rel. step'], rows)
  return lines, out


def late(r, key='acc', frac=0.75):
  """Mean of a logged measure over the evaluations at steps >= frac * steps (the last quarter of training). With a
  constant Adam rate the model oscillates late in training, so one evaluation at the last step is a noisy snapshot."""
  v = [e[key] for e in r['log'] if frac * r['cfg']['steps'] <= e['step'] <= r['cfg']['steps']]
  return mean(v) if v else None


def first_step(r, thr):
  return next((e['step'] for e in r['log'] if e['step'] <= r['cfg']['steps'] and e['acc'] >= thr), None)


def facts_tables(main, ctrl, replicate=None):
  lines, summary = ['## Facts task\n'], {}
  lines += ['Late-window measures (mean over the evaluations in the last quarter of training) are the robust ones; the '
            'final-step accuracy is kept for the pre-registered comparison.\n']
  rows = []
  for (arm, K), seeds in sorted(main.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1])):
    base = main.get((arm, 1), {})
    rs = [seeds[s] for s in sorted(seeds)]
    row = {'late_acc': [late(r) for r in rs], 'late_loss': [late(r, 'loss') for r in rs],
           'late_gain_vs_K1': [late(seeds[s]) - late(base[s]) for s in sorted(seeds) if s in base],
           'step50': [first_step(r, 0.5) for r in rs], 'step90': [first_step(r, 0.9) for r in rs]}
    summary[f'{arm}_K{K}'] = row
    rows.append([arm, str(K), str(len(rs)), fmt(row['late_acc'], 4), fmt(row['late_gain_vs_K1'], 4),
                 fmt(row['late_loss'], 4), fmt(row['step50'], 0), fmt(row['step90'], 0)])
  lines += table(['arm', 'K', 'seeds', 'late acc', 'late gain vs K = 1', 'late loss', 'step to 50%', 'step to 90%'], rows)
  lines += ['\nFinal-step accuracy (pre-registered measure) and accuracy at K_test passes:\n']
  rows = []
  for (arm, K), seeds in sorted(main.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1])):
    rs = [seeds[s] for s in sorted(seeds)]
    p = [r.get('loop_probe', {}) for r in rs]
    base = main.get((arm, 1), {})
    row = {'seeds': sorted(seeds), 'final_acc': [r['final_acc'] for r in rs],
           'gain_vs_K1': [seeds[s]['final_acc'] - base[s]['final_acc'] for s in sorted(seeds) if s in base],
           **{f'K{k}': [q['acc_by_K'][str(k)] for q in p if str(k) in q.get('acc_by_K', {})] for k in (1, 2, 4, 8, 16)}}
    summary[f'{arm}_K{K}'] |= row
    rows.append([arm, str(K), str(len(rs)), fmt(row['final_acc']), fmt(row['gain_vs_K1'])]
                + [fmt(row[f'K{k}']) for k in (1, 2, 4, 8, 16)])
  lines += table(['arm', 'K', 'seeds', 'final acc', 'gain vs K = 1', 'K_test 1', '2', '4', '8', '16'], rows)

  if replicate:
    lines += ['\nGPU nondeterminism: final accuracy of round 2 minus round 1, same configuration and seed:\n']
    diffs = [r2['final_acc'] - replicate[key][s]['final_acc'] for key, seeds in main.items() for s, r2 in seeds.items()
             if key in replicate and s in replicate[key]]
    summary['replicate_diff'] = diffs
    if diffs:
      lines.append(f'n = {len(diffs)}, mean {mean(diffs):+.4f}, mean |diff| {mean(abs(d) for d in diffs):.4f}, '
                   f'max |diff| {max(abs(d) for d in diffs):.4f}')

  if ctrl:
    lines += ['\nControls: final accuracy, and gain over the tied K = 1 run of the same arm and seed (main set):\n']
    rows = []
    for (arm, K), seeds in sorted(ctrl.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1])):
      m = re.match(r'(cos_w\d+_s\d+|geglu_w\d+)(.*)$', arm)
      base_arm = m.group(1) if m else arm
      if '_w32768' in arm or '_w65536' in arm:
        base_arm = re.sub(r'_w\d+', '_w16384', base_arm)
      base = main.get((base_arm, 1), {}) if 'N16k' not in arm else ctrl.get((arm, 1), {})
      gain = [r['final_acc'] - base[s]['final_acc'] for s, r in seeds.items() if s in base]
      lgain = [late(r) - late(base[s]) for s, r in seeds.items() if s in base]
      summary[f'ctrl_{arm}_K{K}'] = {'seeds': sorted(seeds), 'final_acc': [seeds[s]['final_acc'] for s in sorted(seeds)],
                                     'late_acc': [late(seeds[s]) for s in sorted(seeds)],
                                     'gain_vs': f'{base_arm if "N16k" not in arm else arm}_K1', 'gain': gain,
                                     'late_gain': lgain, 'step90': [first_step(seeds[s], 0.9) for s in sorted(seeds)]}
      rows.append([arm, str(K), str(len(seeds)), fmt([late(seeds[s]) for s in sorted(seeds)], 4), fmt(lgain, 4),
                   fmt([seeds[s]['final_acc'] for s in sorted(seeds)]), fmt(gain),
                   fmt([first_step(seeds[s], 0.9) for s in sorted(seeds)], 0), f'{base_arm if "N16k" not in arm else arm} K = 1'])
    lines += table(['control arm', 'K', 'seeds', 'late acc', 'late gain', 'final acc', 'final gain', 'step to 90%',
                    'baseline'], rows)

  v2 = {key: seeds for key, seeds in main.items() if any('acc_by_noise_end' in r.get('loop_probe', {}) for r in seeds.values())}
  if v2:
    lines += ['\nNoise at the trained K: accuracy with the push propagated through the passes / added after them, and '
              'the dilution margin (propagated - end-added):\n']
    rows = []
    for (arm, K), seeds in sorted(v2.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1])):
      p = [r['loop_probe'] for r in seeds.values() if 'acc_by_noise_end' in r.get('loop_probe', {})]
      row = {}
      for e in ('0.1', '0.25', '0.5', '1.0'):
        row[f'prop{e}'] = [q['acc_by_noise'][e] for q in p]
        row[f'end{e}'] = [q['acc_by_noise_end'][e] for q in p]
        row[f'margin{e}'] = [a - b for a, b in zip(row[f'prop{e}'], row[f'end{e}'])]
      summary[f'{arm}_K{K}'] |= row
      rows.append([arm, str(K)] + [f'{mean(row[f"prop{e}"]):.3f} / {mean(row[f"end{e}"]):.3f} / {fmt(row[f"margin{e}"])}'
                                   for e in ('0.25', '0.5', '1.0')])
    lines += table(['arm', 'K', 'e = 0.25: prop / end / margin', 'e = 0.5', 'e = 1.0'], rows)

    lines += ['\nTrajectory of the trained map (accuracy after pass k; mean over seeds), share of predictions that change '
              'at pass 16, and |h_16| / |h_0|:\n']
    rows = []
    for (arm, K), seeds in sorted(v2.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1])):
      tr = [r['loop_probe']['trajectory'] for r in seeds.values() if 'trajectory' in r.get('loop_probe', {})]
      if not tr:
        continue
      at = lambda k, c: mean(t[k - 1][c] for t in tr)
      summary[f'{arm}_K{K}']['trajectory'] = tr
      rows.append([arm, str(K)] + [f'{at(k, "acc"):.3f}' for k in (1, 2, 3, 4, 6, 8, 12, 16)]
                  + [f'{at(16, "flips"):.3f}', f'{at(16, "norm_ratio_median"):.2f}'])
    lines += table(['arm', 'K', 'k = 1', '2', '3', '4', '6', '8', '12', '16', 'flips at 16', '|h16|/|h0|'], rows)

    lines += ['\nTest-time step size a_test at the trained K (training used a = 1):\n']
    rows = []
    for (arm, K), seeds in sorted(v2.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1])):
      p = [r['loop_probe']['acc_by_alpha'] for r in seeds.values() if 'acc_by_alpha' in r.get('loop_probe', {})]
      if p:
        rows.append([arm, str(K)] + [f'{mean(q[t] for q in p):.3f}' for t in ('0.5', '0.75', '1.25', '1.5', '2.0')])
    lines += table(['arm', 'K', 'a = 0.5', '0.75', '1.25', '1.5', '2.0'], rows)

    noisy = [(arm, K, [r['loop_probe']['jacobian_noisy'] for r in seeds.values() if 'jacobian_noisy' in r['loop_probe']])
             for (arm, K), seeds in sorted(v2.items(), key=lambda kv: (arm_key(kv[0][0]), kv[0][1]))]
    lines += ['\nPass-1 tangent radius at clean states and at states pushed by noise 0.25:\n']
    rows = []
    for arm, K, ns in noisy:
      clean = [r['loop_probe']['jacobian_by_pass'][0]['tangent_radius_median'] for r in main[(arm, K)].values()
               if 'tangent_radius_median' in r['loop_probe']['jacobian_by_pass'][0]]
      if ns:
        rows.append([arm, str(K), fmt(clean), fmt([n['tangent_radius_median'] for n in ns])])
    lines += table(['arm', 'K', 'clean', 'noise 0.25'], rows)

  jl, jac = jac_rows(main, 'Facts')
  return lines + jl, {'runs': summary, 'jacobian': jac}


def bios_tables(runs):
  lines, summary, rows = ['## Biographies\n'], {}, []
  for (arm, K), seeds in sorted(runs.items()):
    rs = [seeds[s] for s in sorted(seeds)]
    last = [[e for e in r['log'] if e['step'] <= r['cfg']['steps']][-1] for r in rs]
    row = {'seeds': sorted(seeds), 'heldout': [r['final_attr_acc'] for r in rs], 'seen': [e.get('seen_attr_acc') for e in last],
           'plateau_end': [r['plateau_end_step'] for r in rs], 'acc50': [r['acc50_step'] for r in rs],
           'acc90': [r['acc90_step'] for r in rs], 'train_loss': [e.get('train_loss') for e in last]}
    row['gap'] = [s - h for s, h in zip(row['seen'], row['heldout']) if s is not None]
    base = runs.get((arm, 1), {})
    row['gain_vs_K1'] = [seeds[s]['final_attr_acc'] - base[s]['final_attr_acc'] for s in sorted(seeds) if s in base]
    summary[f'{arm}_K{K}'] = row
    rows.append([arm, str(K), str(len(rs)), fmt(row['heldout']), fmt(row['seen']), fmt(row['gap']), fmt(row['gain_vs_K1']),
                 fmt(row['train_loss']), fmt(row['plateau_end'], 0), fmt(row['acc50'], 0), fmt(row['acc90'], 0)])
  lines += table(['arm', 'K', 'seeds', 'held-out acc', 'seen acc', 'seen − held-out', 'held-out gain vs K = 1',
                  'train loss', 'plateau end', '50%', '90%'], rows)
  lines += ['\nAccuracy at K_test passes (held-out / seen; mean over seeds):\n']
  rows = []
  for (arm, K), seeds in sorted(runs.items()):
    p = [r['loop_probe']['acc_by_K'] for r in seeds.values() if 'loop_probe' in r]
    if not p:
      continue
    ks = sorted({int(k) for q in p for k in q})
    cell = lambda k: (f'{mean(q[str(k)]["attr_acc"] for q in p if str(k) in q):.3f} / '
                      + (f'{mean(q[str(k)]["seen_attr_acc"] for q in p if str(k) in q):.3f}'
                         if 'seen_attr_acc' in p[0][str(k)] else '-'))
    summary[f'{arm}_K{K}']['acc_by_K'] = {str(k): [q[str(k)] for q in p if str(k) in q] for k in ks}
    rows.append([arm, str(K)] + [cell(k) if all(str(k) in q for q in p) else '-' for k in (1, 2, 4, 8, 16)])
  lines += table(['arm', 'K trained', 'K_test = 1', '2', '4', '8', '16'], rows)
  al = [((arm, K), [r['loop_probe']['acc_by_alpha'] for r in seeds.values() if 'acc_by_alpha' in r.get('loop_probe', {})])
        for (arm, K), seeds in sorted(runs.items())]
  if any(p for _, p in al):
    lines += ['\nHeld-out accuracy at test step size a_test (trained K, a = 1 in training):\n']
    lines += table(['arm', 'K', 'a = 0.5', '0.75', '1.25', '1.5', '2.0'],
                   [[arm, str(K)] + [f'{mean(q[t]["attr_acc"] for q in p):.3f}' for t in ('0.5', '0.75', '1.25', '1.5', '2.0')]
                    for (arm, K), p in al if p])
  jl, jac = jac_rows(runs, 'Biographies')
  return lines + jl, {'runs': summary, 'jacobian': jac}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--facts')
  ap.add_argument('--facts_v2')
  ap.add_argument('--facts_ctrl')
  ap.add_argument('--merge', nargs='*', default=[], help='more facts result dirs (width sweep, basis tests) merged into '
                  'the main facts set; each arm is compared with its own K = 1 run')
  ap.add_argument('--bios')
  ap.add_argument('--out')
  a = ap.parse_args()
  lines, summary = [], {}
  if a.facts or a.facts_v2:
    r1, r2 = load(a.facts), load(a.facts_v2)
    main = {k: dict(v) for k, v in (r2 or r1).items()}
    if r2:                                         # round 1 seeds that round 2 did not repeat stay in the main set
      for k, v in r1.items():
        for s_, r in v.items():
          main.setdefault(k, {}).setdefault(s_, r)
    for d in a.merge:
      for k, v in load(d).items():
        main.setdefault(k, {}).update(v)
    l, summary['facts'] = facts_tables(main, load(a.facts_ctrl), replicate=r1 if r2 else None)
    lines += l
    lines += ['', '## Equal-parameter view (trained parameters of the whole model; late accuracy, mean over seeds)\n']
    rows = []
    for arm in sorted({k[0] for k in main}, key=arm_key):
      ks = sorted(K for (a_, K) in main if a_ == arm)
      any_r = next(iter(main[(arm, ks[0])].values()))
      cell = lambda K: fmt([late(r) for r in main[(arm, K)].values()], 4) if (arm, K) in main else '-'
      rows.append([arm, f"{any_r['params']:,}"] + [cell(K) for K in (1, 2, 4)])
    lines += table(['arm', 'trained params', 'K = 1', 'K = 2', 'K = 4'], rows)
  if a.bios:
    l, summary['bios'] = bios_tables(load(a.bios))
    lines += [''] + l
  print('\n'.join(lines))
  if a.out:
    os.makedirs(a.out, exist_ok=True)
    json.dump(summary, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    open(os.path.join(a.out, 'tables.md'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
  main()
