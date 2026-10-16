"""experiments.rare_spec: the per-run data specs of the rare-class experiment (crossover sets A and B)."""

import json
import os

import numpy as np
import pytest

from experiments.rare_spec import KIND, build_spec

SEL = os.path.join(os.path.dirname(__file__), '..', 'docs', 'data', 'rare', 'rare_classes.json')


@pytest.fixture(scope='module')
def specs():
  sel = json.load(open(SEL))
  rng = np.random.default_rng(0)
  labels = np.repeat(np.arange(1000), 1300)[rng.permutation(1_300_000)]     # shuffled, like the packed order
  return sel, labels, {s: build_spec(sel, labels, s) for s in 'AB'}


def groups_of(spec, kind, owner):
  g = np.flatnonzero((spec['group_kind'] == KIND[kind]) & (spec['group_owner'] == owner))
  return {int(i) for i in np.flatnonzero(np.isin(spec['group'], g))}


def test_crossover_compares_the_same_images(specs):
  sel, labels, sp = specs
  for r in sel['classes']:
    c, lvl, s = r['cls'], r['level'], r['set']
    other = 'B' if s == 'A' else 'A'
    rare, common = sp[s], sp[other]
    n_kept = int(np.sum(labels[rare['keep']] == c))
    assert n_kept == lvl                                      # absent: 0, else the level
    own_rare = groups_of(rare, 'own_img', c) | groups_of(rare, 'own_tenth', c)
    own_common = groups_of(common, 'own_sub', c)
    if lvl > 0:
      assert own_rare == own_common and len(own_rare) == lvl  # the same images in both conditions
    for kind in ('rel_decoy', 'rand_decoy'):                   # decoy sets: the same images in both runs
      assert groups_of(rare, kind, c) == groups_of(common, kind, c)


def test_groups_fold_to_labels_and_relabels(specs):
  sel, labels, sp = specs
  for s, spec in sp.items():
    keep = spec['keep']
    assert (spec['group'][keep] >= 0).all() and (spec['group'] >= 0).sum() == len(keep)
    assert np.array_equal(spec['group_class'][spec['group'][keep]], spec['labels'][keep])
    for r in sel['relabel']:
      moved = np.flatnonzero((spec['labels'] != labels) & (spec['labels'] == r['cls']))
      if r['set'] == s:
        assert len(moved) == r['n'] and (labels[moved] == r['source']).all()
        assert not np.isin(np.flatnonzero(labels == r['cls']), keep).any()
      else:
        assert len(moved) == 0
