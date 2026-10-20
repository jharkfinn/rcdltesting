"""Rare-class experiment v2: add noise controls to the class selection (rare_classes.json -> rare_classes_v2.json).

A noise control is a class whose own images are all dropped in its set's rare run, and to which 10 images of a far
source class are relabelled with their pixels replaced by fixed smooth noise (rcdl.imagenet.smooth_noise: 16 x 16
Gaussian noise per channel, bilinearly upsampled to 256 x 256, seeded by the image's global index). It shows what the
measures read for a memorized class with no content: a positive control for a packing signature. In every other run
the class is common. Chosen like the relabel controls: per set, 4 classes selected in neither set, not decoys, not
relabel controls, outside every selected class's 10 visually nearest classes, and unrelated (WordNet <= 4 or visual
top 10 either way) to that set's classes; the source class is the first one with visual rank > 200 and WordNet > 6,
not selected, not a decoy, relabel or noise class. Existing entries are unchanged.

  python3 -m experiments.rare_noise docs/data docs/data/rare/rare_classes.json docs/data/rare/rare_classes_v2.json
"""

import json
import sys

import numpy as np


def main():
  data, src, out = sys.argv[1], sys.argv[2], sys.argv[3]
  sel = json.load(open(src))
  w = np.load(f'{data}/imagenet_wordnet.npz')
  v = np.load(f'{data}/imagenet_dinov2_classdist.npz')
  D = w['dist'].astype(int)
  names = [str(n) for n in w['names']]
  CSo = v['cos_dist'].astype(np.float64) + np.eye(1000) * 9
  np.fill_diagonal(D, 99)
  order = np.argsort(CSo, 1)
  near10 = np.zeros((1000, 1000), bool)
  near10[np.arange(1000)[:, None], order[:, :10]] = True
  rel = (D <= 4) | near10 | near10.T
  chosen = {r['cls'] for r in sel['classes']}
  decoy = {k for r in sel['classes'] for k in r['rel_decoy'] + r['rand_decoy']}
  relab = {r['cls'] for r in sel['relabel']} | {r['source'] for r in sel['relabel']}
  near_any = {int(k) for c in chosen for k in order[c, :10]}
  rng = np.random.default_rng(4321)
  noise, taken = [], set()
  for s in 'AB':
    mine = [r['cls'] for r in sel['classes'] if r['set'] == s] + [r['cls'] for r in sel['relabel'] if r['set'] == s]
    got = 0
    for c in rng.permutation(1000):
      c = int(c)
      if c in chosen or c in decoy or c in relab or c in near_any or c in taken or any(rel[c, j] for j in mine):
        continue
      srcs = [int(k) for k in order[c, 200:] if D[c, k] > 6 and k not in chosen and k not in decoy and k not in relab
              and k not in taken]
      noise.append(dict(cls=c, name=names[c], set=s, source=srcs[0], source_name=names[srcs[0]], n=10,
                        near10=[int(k) for k in order[c, :10]]))
      taken |= {c, srcs[0]}
      mine.append(c)
      got += 1
      if got == 4:
        break
  sel['noise'] = noise
  sel['summary']['n_noise'] = len(noise)
  json.dump(sel, open(out, 'w'), indent=1)
  print(json.dumps([(r['set'], r['name'], r['source_name']) for r in noise]))


if __name__ == '__main__':
  main()
