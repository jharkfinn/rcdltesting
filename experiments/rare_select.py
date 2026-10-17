"""Rare-class experiment: choose the rare-class sets (do rare classes ride related machinery?).

Two disjoint sets A and B (crossover: run A cuts set A, run B cuts set B; each class is rare in one run, common in the
other). Strata: 'dense' (many WordNet relatives, visually crowded) and 'iso' (few relatives, visually far from every
class). Levels per stratum and set: 4 classes at 0 images (absent), 14 at 10, 14 at 100.
Within a set, no two classes are related (WordNet distance <= 4, or either is among the other's 10 visually nearest
classes), so every rare class's relatives stay common in its run.

Usage: python3 -m experiments.rare_select docs/data docs/data/rare/rare_classes.json
"""
import json
import sys

import numpy as np

data, out = sys.argv[1], sys.argv[2]
w = np.load(f'{data}/imagenet_wordnet.npz')
v = np.load(f'{data}/imagenet_dinov2_classdist.npz')
D = w['dist'].astype(int)
names = [str(n) for n in w['names']]
CS = v['cos_dist'].astype(np.float64)
np.fill_diagonal(D, 99)
CSo = CS + np.eye(1000) * 9
order = np.argsort(CSo, 1)                      # visual neighbours, nearest first
nW = (D <= 4).sum(1)                            # WordNet relatives
cos10 = np.take_along_axis(CSo, order[:, :10], 1).mean(1)
near10 = np.zeros((1000, 1000), bool)
near10[np.arange(1000)[:, None], order[:, :10]] = True
rel = (D <= 4) | near10 | near10.T

LEVELS = (0, 10, 100)
PER = {0: 4, 10: 14, 100: 14}
strata = {'dense': np.where((nW >= 8) & (cos10 <= np.percentile(cos10, 60)))[0],
          'iso': np.where((nW <= 4) & (cos10 >= np.percentile(cos10, 50)))[0]}
need = sum(PER.values())

best = None
for seed in range(400):
  rng = np.random.default_rng(seed)
  sets = {'A': [], 'B': []}
  ok = True
  for st, cand in strata.items():
    got = {'A': 0, 'B': 0}
    for c in rng.permutation(cand):               # alternate between the sets so that both fill
      s = 'A' if got['A'] <= got['B'] else 'B'
      if got[s] == need:
        s = 'B' if s == 'A' else 'A'
        if got[s] == need:
          break
      if any(rel[c, j] for j, _ in sets[s]):
        s2 = 'B' if s == 'A' else 'A'
        if got[s2] == need or any(rel[c, j] for j, _ in sets[s2]):
          continue
        s = s2
      sets[s].append((int(c), st))
      got[s] += 1
    ok &= got['A'] == need and got['B'] == need
  if ok:
    cross = sum(rel[a, b] for a, _ in sets['A'] for b, _ in sets['B'])
    if best is None or cross < best[0]:
      best = (cross, seed, sets)
assert best is not None, 'infeasible'
cross, seed, sets = best

# levels: within each set and stratum, sort by WordNet relatives then cos10 and deal levels in rotating blocks so the
# levels are balanced on both
rng = np.random.default_rng(1234)
sel = set(c for s in sets.values() for c, _ in s)
spec = []
for s, lst in sets.items():
  for st in strata:
    cls = sorted([c for c, t in lst if t == st], key=lambda c: (nW[c], cos10[c]))
    lv = np.empty(len(cls), int)
    for i in range(0, len(cls), 16):              # two blocks of 16 (2 absent, 7 at 10, 7 at 100)
      lv[i:i + 16] = rng.permutation([0] * 2 + [10] * 7 + [100] * 7)
    for c, l in zip(cls, lv):
      spec.append(dict(cls=c, name=names[c], set=s, stratum=st, level=int(l), nW=int(nW[c]),
                       cos10=round(float(cos10[c]), 3)))

# decoy classes: relative decoys = the two visually nearest classes selected in neither set; random decoys = four
# classes with visual rank > 200 and WordNet distance > 6, selected in neither set, used by as few classes as possible
used = np.zeros(1000, int)
for r in spec:
  c = r['cls']
  near = [int(k) for k in order[c] if k not in sel][:10]
  r['near10'] = near
  r['near10_names'] = [names[k] for k in near]
  r['rel_decoy'] = near[:2]
  far = [int(k) for k in order[c, 200:] if D[c, k] > 6 and k not in sel]
  far = sorted(far, key=lambda k: (used[k], rng.random()))[:4]
  used[far] += 1
  r['rand_decoy'] = far
# relabel controls (positive control for content against label relatedness): per set, 4 classes selected in
# neither set and unrelated to that set's classes; all their own images are dropped and 10 images of one far source
# class F (visual rank > 200, WordNet > 6, not selected, not a decoy class) are relabelled to them
decoy_cls = set(k for r in spec for k in r['rel_decoy'] + r['rand_decoy'])
near_any = set(k for r in spec for k in r['near10'])
relabel = []
for s in 'AB':
  mine = [r['cls'] for r in spec if r['set'] == s]
  got = 0
  for c in rng.permutation(1000):
    c = int(c)
    if c in sel or c in decoy_cls or c in near_any or any(rel[c, j] for j in mine) or any(c == x['cls'] for x in relabel):
      continue
    src = [int(k) for k in order[c, 200:] if D[c, k] > 6 and k not in sel and k not in decoy_cls][0]
    relabel.append(dict(cls=c, name=names[c], set=s, source=src, source_name=names[src], n=10,
                        near10=[int(k) for k in order[c] if k not in sel][:10],
                        source_near10=[int(k) for k in order[src] if k not in sel and k != c][:10]))
    mine.append(c)
    got += 1
    if got == 4:
      break
summary = {'seed': seed, 'cross_set_related_pairs': int(cross), 'n_selected': len(spec),
           'strata_sizes': {k: int(len(v)) for k, v in strata.items()},
           'cells': {f'{s}/{st}/{l}': sum(1 for r in spec if (r['set'], r['stratum'], r['level']) == (s, st, l))
                     for s in 'AB' for st in strata for l in LEVELS},
           'median_nW': {st: float(np.median([r['nW'] for r in spec if r['stratum'] == st])) for st in strata},
           'median_cos10': {st: float(np.median([r['cos10'] for r in spec if r['stratum'] == st])) for st in strata},
           'max_rand_decoy_reuse': int(used.max())}
json.dump({'summary': summary, 'classes': spec, 'relabel': relabel}, open(out, 'w'), indent=1)
print(json.dumps(summary, indent=1))
