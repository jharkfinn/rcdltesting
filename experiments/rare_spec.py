"""Rare-class experiment: the per-run data spec for experiments/convnext_train.py --spec.

Input: the class selection (rare_classes.json: two sets A and B, levels 0 / 10 / 100, near10, decoy classes, relabel
controls) and the packed training labels (labels.npy, global index order). Output for run set S (the set made rare):

  keep         (n_kept,) int64   global indices trained on
  labels       (N,) int64        label table (relabel controls: 10 images of a far source class F get label c)
  group        (N,) int32        fine group of every image (-1: dropped)
  group_class  (G,) int16        the class whose label the group's images carry (folding groups gives the class split)
  group_kind   (G,) int8         0 class, 1 own image (rare, level 10), 2 own tenth (rare, level 100: 10 groups of 10),
                                 3 own subset (common condition: the same m_c images the class keeps when rare),
                                 4 relative decoy, 5 random decoy, 6 relabelled image (relabel control)
  group_owner  (G,) int16        the selected class a non-class group belongs to (-1 for class groups)
  meta         json string       selection summary, run set, seed

Image choice is independent of the run set: class c's images are ordered by default_rng([seed, c]).permutation; a rare
class keeps the first m_c (m_c = level), and in its common run the same m_c images form its own-subset group, so the
two conditions compare the same images. Decoy sets (m_c images each, m_c = 100 for absent classes) are drawn in a fixed
order, disjoint, from the decoy classes (never selected, so common in both runs): the same images in both runs.
"""

import argparse
import json

import numpy as np

KIND = dict(cls=0, own_img=1, own_tenth=2, own_sub=3, rel_decoy=4, rand_decoy=5, relabel=6)


def build_spec(sel: dict, labels: np.ndarray, run_set: str, seed: int = 0) -> dict:
  N = len(labels)
  by_cls = [np.flatnonzero(labels == c) for c in range(1000)]
  order = {c: by_cls[c][np.random.default_rng([seed, c]).permutation(len(by_cls[c]))] for c in range(1000)}
  taken = np.zeros(N, bool)                         # images already in a decoy or relabel group
  group = labels.astype(np.int32).copy()
  new_labels = labels.astype(np.int64).copy()
  gclass, gkind, gowner = list(range(1000)), [0] * 1000, [-1] * 1000

  def add(idx, cls, kind, owner):
    g = len(gclass)
    gclass.append(cls), gkind.append(KIND[kind]), gowner.append(owner)
    group[idx] = g
    taken[idx] = True

  def draw(k, m):                                   # the first m not-yet-taken images of class k, in its fixed order
    free = order[k][~taken[order[k]]]
    assert len(free) >= m, f'class {k}: {len(free)} free images < {m}'
    return free[:m]

  classes = sorted(sel['classes'], key=lambda r: r['cls'])
  for r in classes:                                 # own images (set-independent choice)
    c, lvl = r['cls'], r['level']
    m = lvl if lvl > 0 else 100
    own = order[c][:m]
    if r['set'] == run_set:                         # rare (or absent) here
      group[order[c][m if lvl > 0 else 0:]] = -1    # dropped
      if lvl == 0:
        group[own] = -1
      elif lvl == 10:
        for i in own:
          add(np.array([i]), c, 'own_img', c)
      else:
        for j in range(10):
          add(own[10 * j:10 * (j + 1)], c, 'own_tenth', c)
    else:
      add(own, c, 'own_sub', c)
  for r in classes:                                 # decoys, fixed order, both runs
    m = r['level'] if r['level'] > 0 else 100
    for k in r['rel_decoy']:
      add(draw(k, m), k, 'rel_decoy', r['cls'])
    for k in r['rand_decoy']:
      add(draw(k, m), k, 'rand_decoy', r['cls'])
  for r in sorted(sel['relabel'], key=lambda r: r['cls']):
    c, F = r['cls'], r['source']
    if r['set'] == run_set:                         # all own images dropped, 10 images of F relabelled c
      group[by_cls[c]] = -1
      src = draw(F, r['n'])
      new_labels[src] = c
      for i in src:
        add(np.array([i]), c, 'relabel', c)
    else:
      add(order[c][:r['n']], c, 'own_sub', c)
  keep = np.flatnonzero(group >= 0).astype(np.int64)
  gc = np.array(gclass, np.int16)
  ok = group[keep]
  assert np.all(gc[ok] == new_labels[keep]), 'every kept image is in a group of its (new) label'
  meta = dict(run_set=run_set, seed=seed, n_kept=int(len(keep)), n_dropped=int(N - len(keep)), n_groups=len(gc),
              groups_by_kind={k: int(np.sum(np.array(gkind) == v)) for k, v in KIND.items()},
              selection=sel['summary'])
  return dict(keep=keep, labels=new_labels, group=group, group_class=gc, group_kind=np.array(gkind, np.int8),
              group_owner=np.array(gowner, np.int16), meta=np.array(json.dumps(meta)))


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--classes', required=True, help='rare_classes.json')
  p.add_argument('--labels', default='/dev/shm/imagenet256/labels.npy')
  p.add_argument('--set', required=True, choices=['A', 'B'])
  p.add_argument('--seed', type=int, default=0, help='image choice (fixed across runs and training seeds)')
  p.add_argument('--out', required=True)
  a = p.parse_args()
  spec = build_spec(json.load(open(a.classes)), np.load(a.labels), a.set, a.seed)
  np.savez_compressed(a.out, **spec)
  print(spec['meta'])
