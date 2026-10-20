"""Rare-class experiment: the per-run data spec for experiments/convnext_train.py --spec (v2).

Input: the class selection (rare_classes_v2.json: sets A and B, levels 0 / 10 / 100, decoy classes, relabel and noise
controls) and the packed training labels (labels.npy, global index order). Output for run set S in {A, B, C} (A or B:
that set is made rare; C: every class common):

  keep         (n_kept,) int64   global indices trained on
  labels       (N,) int64        label table (relabel and noise controls: 10 images of a far source class get label c)
  group        (N,) int32        fine group of every image (-1: dropped)
  group_class  (G,) int16        the class whose label the group's images carry (folding groups gives the class split)
  group_kind   (G,) int8         0 class, 1 own image (level 10: one image per group), 2 own tenth (level 100: 10 groups
                                 of 10), 3 own subset (absent classes when common: one group of 100; relabel and noise
                                 controls when common: one group of 10), 4 relative decoy, 5 random decoy,
                                 6 relabelled image, 7 noise image
  group_owner  (G,) int16        the selected or control class a non-class group belongs to (-1 for class groups)
  noise        (n,) int64        global indices whose pixels the loader replaces by smooth noise (noise controls)
  meta         json string       selection summary, run set, seed

Image choice is independent of the run set. Class c's images are ordered by default_rng([seed, c]).permutation; the
first m_c (m_c = level, or 100 for absent classes) are its own images Own_m. Own_m is split the same way in every run
where it is trained (single images at level 10, tenths at level 100), so a rare class and the same class when common
compare the same images in the same groups. Decoy sets (m_c images each) are drawn in a fixed order, disjoint, from the
decoy classes (never selected, so common in every run): the same images in every run.
"""

import argparse
import json

import numpy as np

KIND = dict(cls=0, own_img=1, own_tenth=2, own_sub=3, rel_decoy=4, rand_decoy=5, relabel=6, noise=7)


def build_spec(sel: dict, labels: np.ndarray, run_set: str, seed: int = 0) -> dict:
  assert run_set in ('A', 'B', 'C')
  N = len(labels)
  by_cls = [np.flatnonzero(labels == c) for c in range(1000)]
  order = {c: by_cls[c][np.random.default_rng([seed, c]).permutation(len(by_cls[c]))] for c in range(1000)}
  taken = np.zeros(N, bool)                         # images already in a decoy, relabel or noise group
  group = labels.astype(np.int32).copy()
  new_labels = labels.astype(np.int64).copy()
  noise_idx = []
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
  for r in classes:                                 # own images (set-independent choice and split)
    c, lvl = r['cls'], r['level']
    m = lvl if lvl > 0 else 100
    own = order[c][:m]
    rare = r['set'] == run_set
    if rare:
      group[order[c][m if lvl > 0 else 0:]] = -1    # dropped (absent classes: all)
    if lvl == 10:                                   # rare or common: one group per image
      for i in own:
        add(np.array([i]), c, 'own_img', c)
    elif lvl == 100:                                # rare or common: ten groups of ten
      for j in range(10):
        add(own[10 * j:10 * (j + 1)], c, 'own_tenth', c)
    elif not rare:                                  # absent class, common here: its 100 images as one group
      add(own, c, 'own_sub', c)
  for r in classes:                                 # decoys, fixed order, every run
    m = r['level'] if r['level'] > 0 else 100
    for k in r['rel_decoy']:
      add(draw(k, m), k, 'rel_decoy', r['cls'])
    for k in r['rand_decoy']:
      add(draw(k, m), k, 'rand_decoy', r['cls'])
  for kind, rows in (('relabel', sel['relabel']), ('noise', sel.get('noise', []))):
    for r in sorted(rows, key=lambda r: r['cls']):
      c, F = r['cls'], r['source']
      if r['set'] == run_set:                       # all own images dropped, n images of F relabelled c
        group[by_cls[c]] = -1
        src = draw(F, r['n'])
        new_labels[src] = c
        for i in src:
          add(np.array([i]), c, kind, c)
        if kind == 'noise':
          noise_idx += [int(i) for i in src]
      else:
        add(order[c][:r['n']], c, 'own_sub', c)
  keep = np.flatnonzero(group >= 0).astype(np.int64)
  gc = np.array(gclass, np.int16)
  assert np.all(gc[group[keep]] == new_labels[keep]), 'every kept image is in a group of its (new) label'
  meta = dict(run_set=run_set, seed=seed, n_kept=int(len(keep)), n_dropped=int(N - len(keep)), n_groups=len(gc),
              n_noise=len(noise_idx), groups_by_kind={k: int(np.sum(np.array(gkind) == v)) for k, v in KIND.items()},
              selection=sel['summary'])
  return dict(keep=keep, labels=new_labels, group=group, group_class=gc, group_kind=np.array(gkind, np.int8),
              group_owner=np.array(gowner, np.int16), noise=np.array(sorted(noise_idx), np.int64),
              meta=np.array(json.dumps(meta)))


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--classes', required=True, help='rare_classes_v2.json')
  p.add_argument('--labels', default='/dev/shm/imagenet256/labels.npy')
  p.add_argument('--set', required=True, choices=['A', 'B', 'C'])
  p.add_argument('--seed', type=int, default=0, help='image choice (fixed across runs and training seeds)')
  p.add_argument('--out', required=True)
  a = p.parse_args()
  spec = build_spec(json.load(open(a.classes)), np.load(a.labels), a.set, a.seed)
  np.savez_compressed(a.out, **spec)
  print(spec['meta'])
