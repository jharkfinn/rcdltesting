"""E8 summary: pole sharing by pair type, across seeds, with the effect of relatedness as a difference in differences.

Input: the JSON files of experiments/toy_packing.py (one per run). For each cell (activation, rho, m, H) and pair type
t, the excess of a run is

  ex_t(stat) = observed share of pairs of type t with the property - share expected with pole memberships shuffled,

for stat = copositive (both features on one neuron's positive pole) and antipodal (on opposite poles of one neuron). The
two anticorrelated pair types differ already at rho = 0 (where 'related' is only a family label), because the exclusion
structure differs between them; the effect of relatedness on antipodality is therefore

  DiD = [ex_related_anti(rho) - ex_unrelated_anti(rho)] - [ex_related_anti(0) - ex_unrelated_anti(0)],

with seeds paired by seed number (same seed -> same world draw and initialization apart from rho). Reported: means over
seeds and their standard errors (standard deviation / sqrt(seeds)).
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

TYPES = ('related_anti', 'related_indep', 'unrelated_anti', 'unrelated_indep')


def load(d):
  runs = defaultdict(dict)                     # (act, rho, m, H) -> seed -> pair_stats
  for f in glob.glob(os.path.join(d, '*.json')):
    r = json.load(open(f))
    if 'pair_stats' not in r:
      continue
    a = r['args']
    runs[(a.get('act', 'relu'), float(a['rho']), int(a['m']), int(a['H']))][int(a['seed'])] = r['pair_stats']
  return runs


def excess(ps, t, stat):
  return ps[t][stat] - ps[t][f'null_{stat}']


def mse(x):
  x = np.asarray(x, dtype=float)
  return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else float('nan')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--dir', default='results/toy_packing')
  p.add_argument('--out', default='results/toy_packing_summary.json')
  p.add_argument('--min_seeds', type=int, default=5)
  a = p.parse_args()
  runs = load(a.dir)
  out = {'cells': {}, 'did': {}}
  print('act    rho  m  H seeds | excess antipodal: rel_anti  unrel_anti | excess copositive: rel_anti  rel_indep'
        '  unrel_anti  unrel_indep')
  for key in sorted(runs):
    seeds = runs[key]
    if len(seeds) < a.min_seeds:
      continue
    cell = {f'{t}_{s}': mse([excess(ps, t, s) for ps in seeds.values()]) for t in TYPES for s in ('antipodal', 'copositive')}
    cell['seeds'] = len(seeds)
    out['cells']['%s_rho%.1f_m%d_H%d' % key] = cell
    f = lambda c: '%+.2f±%.2f' % c
    print(f'{key[0]:6s} {key[1]:.1f} {key[2]:2d} {key[3]:2d} {len(seeds):5d} | {f(cell["related_anti_antipodal"])}'
          f' {f(cell["unrelated_anti_antipodal"])} | {f(cell["related_anti_copositive"])} {f(cell["related_indep_copositive"])}'
          f' {f(cell["unrelated_anti_copositive"])} {f(cell["unrelated_indep_copositive"])}')
  print('\ndifference in differences (rho vs 0, seeds paired): antipodal [rel_anti - unrel_anti]; copositive [rel_anti -'
        ' unrel_anti]; copositive [rel_indep - unrel_indep]')
  for key in sorted(runs):
    act, rho, m, H = key
    base = runs.get((act, 0.0, m, H))
    if rho == 0.0 or base is None:
      continue
    common = sorted(set(runs[key]) & set(base))
    if len(common) < a.min_seeds:
      continue
    def did(stat, t1, t2):
      return mse([(excess(runs[key][s], t1, stat) - excess(runs[key][s], t2, stat))
                  - (excess(base[s], t1, stat) - excess(base[s], t2, stat)) for s in common])
    rec = {'seeds': len(common), 'antipodal_anti': did('antipodal', 'related_anti', 'unrelated_anti'),
           'copositive_anti': did('copositive', 'related_anti', 'unrelated_anti'),
           'copositive_indep': did('copositive', 'related_indep', 'unrelated_indep')}
    out['did']['%s_rho%.1f_m%d_H%d' % key] = rec
    f = lambda c: '%+.2f±%.2f' % c
    print(f'{act:6s} {rho:.1f} {m:2d} {H:2d} {len(common):5d} | {f(rec["antipodal_anti"])} | {f(rec["copositive_anti"])}'
          f' | {f(rec["copositive_indep"])}')
  os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
  json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
  main()
