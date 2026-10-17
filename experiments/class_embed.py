"""Visual distances between the 1,000 ImageNet classes, from an image model independent of the models studied.

Every validation image (center crop, as in E0) is embedded with DINOv2 ViT-S/14 (CLS token, centred over all images,
then L2-normalized); a class is the normalized mean of its 50 image embeddings. Saved:
  cos_dist   (1000, 1000) float16, 1 - cosine similarity of the class means
  rank_dist  (1000, 1000) int16, symmetric neighbour rank: (rank of k among c's neighbours + rank of c among k's) / 2,
             rounded down; 0 on the diagonal. 'Visual relatives' of c: rank_dist <= 10 (about as many as WordNet
             relatives at distance 4, median 10).
The growth tests can use either in place of the WordNet distance (experiments/convnext_growth_tests.py --dist).

Usage: python3 -m experiments.class_embed --val <validation parquet dir> --out docs/data/imagenet_dinov2_classdist.npz
"""

import argparse

import numpy as np
import torch

from experiments.inception_e0 import iter_val
from experiments.inception_mixed import embed_dinov2, image_cache


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--val', required=True)
  p.add_argument('--out', default='docs/data/imagenet_dinov2_classdist.npz')
  a = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  labels = np.concatenate([np.asarray(y) for _, y in iter_val(a.val, 250)])
  cache = image_cache(a.val, dev)
  assert len(cache) == len(labels)
  e = embed_dinov2(cache, dev)
  e = e / e.norm(dim=1, keepdim=True)
  m = torch.zeros(1000, e.shape[1]).index_add_(0, torch.from_numpy(labels).long(), e)
  m = m / m.norm(dim=1, keepdim=True)
  cos = (1 - m @ m.T).clamp_min(0).numpy()
  np.fill_diagonal(cos, 0)
  order = np.argsort(cos, 1)
  rank = np.empty_like(order)
  rank[np.arange(1000)[:, None], order] = np.arange(1000)[None, :]
  rank_dist = ((rank + rank.T) // 2).astype(np.int16)
  np.fill_diagonal(rank_dist, 0)
  np.savez_compressed(a.out, cos_dist=cos.astype(np.float16), rank_dist=rank_dist,
                      n_images=np.bincount(labels, minlength=1000))
  print('saved', a.out, 'median 10th-neighbour cos dist', float(np.median(np.sort(cos, 1)[:, 10])), flush=True)


if __name__ == '__main__':
  main()
