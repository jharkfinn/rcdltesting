"""Looping test tables: facts (experiments/configs/facts_loops.json) and biographies (bios_loops.json).

Usage: python -m experiments.loops_analysis --facts DIR --bios DIR [--out docs/data/loops]
Each DIR holds <name>/result.json. Run names end in _K{K}_s{seed}; the part before is the arm. Prints Markdown tables
(mean over seeds, then the per-seed values) and writes summary.json with every number the tables use.

Facts measures (rcdl/facts.loop_probe): final accuracy at the trained K; accuracy at K_test = 1, 2, 4, 8; accuracy with
state noise e (relative size of a random push of the state before the passes); per pass k the medians of the spectral
radius rho(J_k), the spectral norm |J_k|_2, |J_k - I|_2 and the relative step |h_k+1 - h_k| / |h_k|.
Biography measures (rcdl/bios.loop_probe): final held-out and seen-template attribute accuracy, the gap seen - held-out,
plateau end and steps to 50% / 90% held-out accuracy, accuracy at K_test, and the same Jacobian medians.
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


def jac_rows(runs, label):
  lines = [f'\n{label}: per-pass Jacobian medians at the trained K (mean over seeds)\n',
           '| arm | K | pass | rho(J) | rho max | ‖J‖₂ | ‖J − I‖₂ | rel. step |', '|---|---|---|---|---|---|---|---|']
  out = {}
  for (arm, K), seeds in sorted(runs.items()):
    per = [r['loop_probe']['jacobian_by_pass'] for r in seeds.values() if 'loop_probe' in r]
    if not per:
      continue
    for k in range(K):
      get = lambda key: [p[k][key] for p in per]
      row = {key: get(key) for key in ('spectral_radius_median', 'spectral_radius_max', 'spectral_norm_median',
                                       'jac_minus_I_norm_median', 'rel_step_median')}
      out[f'{arm}_K{K}_pass{k + 1}'] = row
      lines.append(f'| {arm} | {K} | {k + 1} | ' + ' | '.join(fmt(row[key], 3) for key in row) + ' |')
  return lines, out


def facts_tables(runs):
  lines = ['## Facts task\n', '| arm | K | seeds | final acc | K_test = 1 | 2 | 4 | 8 | noise 0.1 | 0.25 | 0.5 | plateau end |',
           '|---|---|---|---|---|---|---|---|---|---|---|---|']
  summary = {}
  for (arm, K), seeds in sorted(runs.items()):
    rs = list(seeds.values())
    p = [r.get('loop_probe', {}) for r in rs]
    row = {'final_acc': [r['final_acc'] for r in rs], 'plateau_end': [r['plateau_end_step'] for r in rs],
           **{f'K{k}': [q['acc_by_K'][str(k)] for q in p if q] for k in (1, 2, 4, 8)},
           **{f'noise{e}': [q['acc_by_noise'][e] for q in p if q] for e in ('0.1', '0.25', '0.5')}}
    summary[f'{arm}_K{K}'] = row | {'seeds': sorted(seeds)}
    lines.append(f'| {arm} | {K} | {len(rs)} | {fmt(row["final_acc"])} | '
                 + ' | '.join(fmt(row[f'K{k}']) for k in (1, 2, 4, 8)) + ' | '
                 + ' | '.join(fmt(row[f'noise{e}']) for e in ('0.1', '0.25', '0.5'))
                 + f' | {fmt(row["plateau_end"], 0)} |')
  jl, jac = jac_rows(runs, 'Facts')
  lines += jl
  lines += ['\nLooping gain (final accuracy at K minus at K = 1, same seed), and noise drop (accuracy at noise 0.25 minus '
            'final accuracy):\n', '| arm | K | gain vs K = 1 | noise-0.25 drop |', '|---|---|---|---|']
  for (arm, K), seeds in sorted(runs.items()):
    base = runs.get((arm, 1), {})
    gain = [r['final_acc'] - base[s]['final_acc'] for s, r in seeds.items() if s in base]
    drop = [r['loop_probe']['acc_by_noise']['0.25'] - r['final_acc'] for r in seeds.values() if 'loop_probe' in r]
    summary[f'{arm}_K{K}'] |= {'gain_vs_K1': gain, 'noise025_drop': drop}
    lines.append(f'| {arm} | {K} | {fmt(gain)} | {fmt(drop)} |')
  return lines, {'runs': summary, 'jacobian': jac}


def bios_tables(runs):
  lines = ['## Biographies\n', '| arm | K | seeds | held-out acc | seen acc | seen − held-out | plateau end | 50% | 90% |',
           '|---|---|---|---|---|---|---|---|---|']
  summary = {}
  for (arm, K), seeds in sorted(runs.items()):
    rs = list(seeds.values())
    last = [[e for e in r['log'] if e['step'] <= r['cfg']['steps']][-1] for r in rs]
    row = {'heldout': [r['final_attr_acc'] for r in rs], 'seen': [e.get('seen_attr_acc') for e in last],
           'plateau_end': [r['plateau_end_step'] for r in rs], 'acc50': [r['acc50_step'] for r in rs],
           'acc90': [r['acc90_step'] for r in rs]}
    row['gap'] = [s - h for s, h in zip(row['seen'], row['heldout']) if s is not None]
    summary[f'{arm}_K{K}'] = row | {'seeds': sorted(seeds)}
    lines.append(f'| {arm} | {K} | {len(rs)} | {fmt(row["heldout"])} | {fmt(row["seen"])} | {fmt(row["gap"])} | '
                 f'{fmt(row["plateau_end"], 0)} | {fmt(row["acc50"], 0)} | {fmt(row["acc90"], 0)} |')
  lines += ['\nAccuracy at K_test passes (held-out / seen; mean over seeds):\n',
            '| arm | K trained | K_test = 1 | 2 | 4 | 8 |', '|---|---|---|---|---|---|']
  for (arm, K), seeds in sorted(runs.items()):
    p = [r['loop_probe']['acc_by_K'] for r in seeds.values() if 'loop_probe' in r]
    if not p:
      continue
    cell = lambda k: (f'{mean(q[str(k)]["attr_acc"] for q in p):.3f} / '
                      + (f'{mean(q[str(k)]["seen_attr_acc"] for q in p):.3f}' if 'seen_attr_acc' in p[0][str(k)] else '-'))
    summary[f'{arm}_K{K}']['acc_by_K'] = {str(k): [q[str(k)] for q in p] for k in (1, 2, 4, 8)}
    lines.append(f'| {arm} | {K} | ' + ' | '.join(cell(k) for k in (1, 2, 4, 8)) + ' |')
  jl, jac = jac_rows(runs, 'Biographies')
  return lines + jl, {'runs': summary, 'jacobian': jac}


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--facts')
  ap.add_argument('--bios')
  ap.add_argument('--out')
  a = ap.parse_args()
  lines, summary = [], {}
  if a.facts:
    l, summary['facts'] = facts_tables(load(a.facts))
    lines += l
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
