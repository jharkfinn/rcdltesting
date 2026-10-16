"""E0: the essay's re-readings of canonical InceptionV1 neurons, on the exact public model and the ImageNet validation set.

Model: Google's inception5h InceptionV1 (rcdl/third_party/lucent_inceptionv1, equal to TensorFlow on Google's graph to
float32 rounding; rcdl/inception_provenance.py). Images: all 50,000 ImageNet-1k validation images (ILSVRC/imagenet-1k,
parquet), decoded on the GPU, shorter side resized to 256 (antialiased bilinear; the essay's microscope gives
sourceImage 256 / inputCenterCrop 224 but not the interpolation), center crop 224, RGB, Lucid's x * 255 - 117.

`collect` records, for every channel of the branch convolutions of mixed4c, mixed4d, mixed4e and mixed5b, the spatial
maximum and minimum of the pre-ReLU response per image (float16), plus labels and the model's predictions. The positive
pole of a channel is its largest maxima, the negative pole its smallest minima (the essay's "lowest spatial response
anywhere in an image"). Channel numbers follow Lucid's concatenation order (1x1, 3x3, 5x5, pool_reduce), so mixed4e:55 is
mixed4e_1x1 channel 55 and mixed4c:447 is mixed4c_5x5 channel 63.

Usage: python3 -m experiments.inception_e0 collect --val <dir with validation-*.parquet> --weights inception5h.pth
       --out results/e0/val_stats.npz
"""

import argparse
import glob
import io
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

LAYERS = ('mixed4c', 'mixed4d', 'mixed4e', 'mixed5b')
BRANCHES = ('1x1', '3x3', '5x5', 'pool_reduce')


def channel_index(model, layer: str, channel: int):
  """(branch module name, channel within the branch) for Lucid's concatenated channel number."""
  start = 0
  for b in BRANCHES:
    mod = getattr(model, f'{layer}_{b}_pre_relu_conv')
    if channel < start + mod.out_channels:
      return f'{layer}_{b}_pre_relu_conv', channel - start
    start += mod.out_channels
  raise IndexError(channel)


def load_model(weights: str, device):
  from rcdl.third_party.lucent_inceptionv1 import InceptionV1
  m = InceptionV1(pretrained=False, redirected_ReLU=False).eval().to(device)
  m.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
  return m


def iter_val(val_dir: str, batch: int):
  import pyarrow.parquet as pq
  for f in sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True)):
    t = pq.read_table(f)
    imgs, labels = t.column('image').to_pylist(), t.column('label').to_pylist()
    for i in range(0, len(labels), batch):
      yield [d['bytes'] for d in imgs[i:i + batch]], labels[i:i + batch]


def preprocess(raw, device, size=256, crop=224):
  """JPEG bytes -> (B, 3, 224, 224) in Lucid's value range; GPU decode with a CPU fallback (CMYK, PNG)."""
  from torchvision.io import ImageReadMode, decode_image, decode_jpeg
  datas = [torch.frombuffer(bytearray(b), dtype=torch.uint8) for b in raw]
  try:
    imgs = decode_jpeg(datas, mode=ImageReadMode.RGB, device=device)
  except RuntimeError:
    imgs = []
    for d in datas:
      try:
        imgs.append(decode_jpeg(d, mode=ImageReadMode.RGB, device=device))
      except RuntimeError:
        imgs.append(decode_image(d, mode=ImageReadMode.RGB).to(device))
  out = []
  for im in imgs:
    h, w = im.shape[-2:]
    s = size / min(h, w)
    im = F.interpolate(im[None].float(), size=(max(size, round(h * s)), max(size, round(w * s))), mode='bilinear',
                       antialias=True, align_corners=False)
    H, W = im.shape[-2:]
    top, left = (H - crop) // 2, (W - crop) // 2
    out.append(im[0, :, top:top + crop, left:left + crop])
  x = torch.stack(out).clamp_(0, 255)
  return x - 117.0


@torch.no_grad()
def collect(val_dir: str, weights: str, out: str, batch: int = 250):
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = load_model(weights, dev)
  names = [f'{l}_{b}_pre_relu_conv' for l in LAYERS for b in BRANCHES]
  cur = {}
  hooks = [getattr(model, n).register_forward_hook(lambda m, i, o, n=n: cur.__setitem__(n, o)) for n in names]
  mx = {n: [] for n in names}
  mn = {n: [] for n in names}
  labels, preds, t0 = [], [], time.time()
  for raw, lab in iter_val(val_dir, batch):
    x = preprocess(raw, dev)
    logits = model(x)
    for n in names:
      a = cur[n].flatten(2)
      mx[n].append(a.amax(-1).half().cpu())
      mn[n].append(a.amin(-1).half().cpu())
    labels += lab
    preds.append(logits.argmax(-1).cpu())
    if len(labels) % 5000 < batch:
      print(f'{len(labels)} images, {time.time() - t0:.0f}s', flush=True)
  for h in hooks:
    h.remove()
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  np.savez_compressed(out, labels=np.array(labels), preds=torch.cat(preds).numpy(),
                      **{f'max__{n}': torch.cat(mx[n]).numpy() for n in names},
                      **{f'min__{n}': torch.cat(mn[n]).numpy() for n in names})
  print('saved', out, len(labels), 'images', f'{time.time() - t0:.0f}s')


def main():
  p = argparse.ArgumentParser()
  p.add_argument('cmd', choices=['collect'])
  p.add_argument('--val', required=True)
  p.add_argument('--weights', required=True)
  p.add_argument('--out', default='results/e0/val_stats.npz')
  p.add_argument('--batch', type=int, default=250)
  a = p.parse_args()
  collect(a.val, a.weights, a.out, a.batch)


if __name__ == '__main__':
  main()
