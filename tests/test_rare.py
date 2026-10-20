"""experiments.rare_spec (v2): the per-run data specs of the rare-class experiment (sets A and B rare, C all common)."""

import json
import os

import numpy as np
import pytest

from experiments.rare_spec import KIND, build_spec

SEL = os.path.join(os.path.dirname(__file__), '..', 'docs', 'data', 'rare', 'rare_classes_v2.json')


@pytest.fixture(scope='module')
def specs():
  sel = json.load(open(SEL))
  rng = np.random.default_rng(0)
  labels = np.repeat(np.arange(1000), 1300)[rng.permutation(1_300_000)]     # shuffled, like the packed order
  return sel, labels, {s: build_spec(sel, labels, s) for s in 'ABC'}


def groups_of(spec, kinds, owner):
  g = np.flatnonzero(np.isin(spec['group_kind'], [KIND[k] for k in kinds]) & (spec['group_owner'] == owner))
  return [frozenset(np.flatnonzero(spec['group'] == i).tolist()) for i in g]


def test_rare_and_common_split_the_same_images_the_same_way(specs):
  sel, labels, sp = specs
  for r in sel['classes']:
    c, lvl, s = r['cls'], r['level'], r['set']
    other = 'B' if s == 'A' else 'A'
    assert int(np.sum(labels[sp[s]['keep']] == c)) == lvl                      # absent: 0, else the level
    assert int(np.sum(labels[sp['C']['keep']] == c)) == 1300                    # all common in C
    if lvl > 0:
      own = sorted(groups_of(sp[s], ('own_img', 'own_tenth'), c), key=min)
      for run in (other, 'C'):
        assert own == sorted(groups_of(sp[run], ('own_img', 'own_tenth'), c), key=min)
      assert len(own) == 10 and all(len(g) == lvl // 10 for g in own)
    else:
      assert groups_of(sp[s], ('own_sub',), c) == [] and len(groups_of(sp['C'], ('own_sub',), c)[0]) == 100
    for kind in ('rel_decoy', 'rand_decoy'):                                    # decoys: the same images every run
      assert sorted(groups_of(sp[s], (kind,), c), key=min) == sorted(groups_of(sp['C'], (kind,), c), key=min)


def test_groups_fold_to_labels_relabels_and_noise(specs):
  sel, labels, sp = specs
  for s, spec in sp.items():
    keep = spec['keep']
    assert (spec['group'][keep] >= 0).all() and (spec['group'] >= 0).sum() == len(keep)
    assert np.array_equal(spec['group_class'][spec['group'][keep]], spec['labels'][keep])
    for kind in ('relabel', 'noise'):
      for r in sel[kind]:
        moved = np.flatnonzero((spec['labels'] != labels) & (spec['labels'] == r['cls']))
        if r['set'] == s:
          assert len(moved) == r['n'] and (labels[moved] == r['source']).all()
          assert not np.isin(np.flatnonzero(labels == r['cls']), keep).any()
          assert np.isin(moved, spec['noise']).all() == (kind == 'noise')
        else:
          assert len(moved) == 0
    assert len(spec['noise']) == (0 if s == 'C' else 40)
