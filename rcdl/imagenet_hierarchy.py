"""WordNet 3.0 distances between the 1,000 ImageNet-1k classes (for E2: how related are the classes that write a neuron).

ImageNet's class ids are WordNet 3.0 noun synsets (wnid 'n' + 8-digit offset). The corpus is nltk_data's wordnet.zip
(github.com/nltk/nltk_data, gh-pages, packages/corpora/wordnet.zip; the run of 2026-10-16 used sha256 cbda5ea6...).
For classes i and j with hypernym closures A_i, A_j (every ancestor a with its smallest number of hypernym or
instance-hypernym steps d_i(a); WordNet has multiple inheritance, so all paths count):

  dist[i, j]      = min over common ancestors a of d_i(a) + d_j(a)    (the tree distance; 0 on the diagonal)
  lca_depth[i, j] = max over those ancestors a minimizing the distance of depth(a), the shortest path length from a to
                    the root entity.n.01

Usage: python3 -m rcdl.imagenet_hierarchy --classes imagenet_classes.json --wordnet wordnet.zip --out out.npz
(classes: a JSON list of [wnid, names] in label order, e.g. from ILSVRC/imagenet-1k's classes.py).
"""

import argparse
import hashlib
import json
import os
import tempfile
import zipfile

import numpy as np


def closure(syn):
  """{ancestor synset: smallest number of hypernym steps from syn}, syn itself at 0."""
  dist, frontier = {syn: 0}, [syn]
  while frontier:
    nxt = []
    for s in frontier:
      for h in s.hypernyms() + s.instance_hypernyms():
        if h not in dist:
          dist[h] = dist[s] + 1
          nxt.append(h)
    frontier = nxt
  return dist


def build(classes_json: str, wordnet_zip: str, out: str):
  sha = hashlib.sha256(open(wordnet_zip, 'rb').read()).hexdigest()
  root = tempfile.mkdtemp()
  os.makedirs(os.path.join(root, 'corpora'))
  zipfile.ZipFile(wordnet_zip).extractall(os.path.join(root, 'corpora'))
  import nltk
  nltk.data.path.insert(0, root)
  from nltk.corpus import wordnet as wn
  wnids = [w for w, _ in json.load(open(classes_json))]
  syns = [wn.synset_from_pos_and_offset('n', int(w[1:])) for w in wnids]
  cl = [closure(s) for s in syns]
  depth = {}
  def dep(a):
    if a not in depth:
      depth[a] = a.min_depth()
    return depth[a]
  n = len(syns)
  dist = np.zeros((n, n), dtype=np.int16)
  lca = np.zeros((n, n), dtype=np.int16)
  for i in range(n):
    for j in range(i + 1, n):
      common = cl[i].keys() & cl[j].keys()
      best = min(cl[i][a] + cl[j][a] for a in common)
      dist[i, j] = dist[j, i] = best
      lca[i, j] = lca[j, i] = max(dep(a) for a in common if cl[i][a] + cl[j][a] == best)
    lca[i, i] = dep(syns[i])
  np.savez_compressed(out, wnids=np.array(wnids), names=np.array([s.name() for s in syns]), dist=dist, lca_depth=lca,
                      wordnet_zip_sha256=np.array(sha))
  print('saved', out, 'wordnet.zip sha256', sha, 'median distance', int(np.median(dist[np.triu_indices(n, 1)])))


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('--classes', required=True)
  p.add_argument('--wordnet', required=True)
  p.add_argument('--out', default='docs/data/imagenet_wordnet.npz')
  a = p.parse_args()
  build(a.classes, a.wordnet, a.out)
