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


def class_names(imagenet_dir: str):
  """Label index -> name, in the dataset's own order (classes.py of ILSVRC/imagenet-1k)."""
  ns = {}
  exec(open(os.path.join(imagenet_dir, 'classes.py')).read(), ns)
  return [v.split(',')[0] for v in ns['IMAGENET2012_CLASSES'].values()]


def expanded_inputs(sd, layer: str, prev_channel: int):
  """Linearized weight from one channel of the previous mixed layer into every channel of `layer` (Lucid order),
  through the 1x1 bottlenecks of the 3x3 and 5x5 branches (their ReLU ignored) and the pool branch's 1x1 (its max-pool
  ignored), summed over the kernel: the 'expanded weights' of the Circuits work. Returns a (channels,) tensor."""
  w = lambda n: sd[f'{layer}_{n}_pre_relu_conv.weight'].float()
  c = prev_channel
  e1 = w('1x1')[:, c, 0, 0]
  e3 = torch.einsum('obhw,b->o', w('3x3'), w('3x3_bottleneck')[:, c, 0, 0])
  e5 = torch.einsum('obhw,b->o', w('5x5'), w('5x5_bottleneck')[:, c, 0, 0])
  ep = w('pool_reduce')[:, c, 0, 0]
  return torch.cat([e1, e3, e5, ep])


def concat_stats(stats, layer: str, kind: str):
  """(images, channels) spatial max or min of the pre-ReLU response, in Lucid's concatenated channel order."""
  return np.concatenate([stats[f'{kind}__{layer}_{b}_pre_relu_conv'].astype('f4') for b in BRANCHES], 1)


def analyze(stats_path: str, imagenet_dir: str, weights: str, out: str, k: int = 50):
  import json
  st = np.load(stats_path)
  names = class_names(imagenet_dir)
  labels = st['labels']
  sd = torch.load(weights, map_location='cpu', weights_only=True)
  dogs = set(range(151, 269))
  car_names = ['ambulance', 'beach wagon', 'cab', 'convertible', 'jeep', 'limousine', 'minivan', 'Model T', 'racer',
               'sports car']
  cars = {i for i, n in enumerate(names) if n in car_names}
  def pole(layer, ch, kind):
    a = concat_stats(st, layer, kind)[:, ch]
    idx = np.argsort(-a if kind == 'max' else a)[:k]
    lab = labels[idx]
    counts = {}
    for l in lab:
      counts[names[l]] = counts.get(names[l], 0) + 1
    top = sorted(counts.items(), key=lambda t: -t[1])[:12]
    return {'top_classes': top, 'dogs': int(sum(l in dogs for l in lab)), 'cars': int(sum(l in cars for l in lab)),
            'values': [float(a[idx[0]]), float(a[idx[-1]])]}
  res = {'n_images': int(len(labels)), 'top1_acc_note': 'inception5h has 1,008 outputs in its own order; not scored',
         'car_classes': sorted(names[i] for i in cars)}
  res['mixed4e:55'] = {'positive': pole('mixed4e', 55, 'max'), 'negative': pole('mixed4e', 55, 'min')}
  inputs = torch.stack([expanded_inputs(sd, 'mixed4e', c)[55] for c in range(528)])   # 4d channel -> 4e:55
  res['mixed4e:55']['strongest_inputs_4d'] = [
    {'channel': int(c), 'weight': float(inputs[c]), 'positive': pole('mixed4d', int(c), 'max')['top_classes'][:6]}
    for c in torch.argsort(-inputs)[:6]]
  res['mixed4e:55']['most_negative_inputs_4d'] = [
    {'channel': int(c), 'weight': float(inputs[c]), 'positive': pole('mixed4d', int(c), 'max')['top_classes'][:6]}
    for c in torch.argsort(inputs)[:4]]
  res['mixed4c:447'] = {'positive': pole('mixed4c', 447, 'max'), 'negative': pole('mixed4c', 447, 'min')}
  readers = expanded_inputs(sd, 'mixed4d', 447)
  a447 = concat_stats(st, 'mixed4c', 'max')[:, 447]
  m4d = concat_stats(st, 'mixed4d', 'max')
  res['mixed4c:447']['readers_4d'] = []
  for c in torch.argsort(-readers)[:8].tolist():
    r = pole('mixed4d', c, 'max')
    res['mixed4c:447']['readers_4d'].append({'channel': c, 'weight': float(readers[c]),
                                             'corr_with_447': float(np.corrcoef(a447, m4d[:, c])[0, 1]),
                                             'dogs_in_top50': r['dogs'], 'cars_in_top50': r['cars'],
                                             'top_classes': r['top_classes'][:8]})
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  json.dump(res, open(out, 'w'), indent=1)
  print(json.dumps(res, indent=1)[:6000])


def profile(stats_path: str, imagenet_dir: str, layer: str, channel: int, out: str = None, top: int = 25):
  """Class-level view of one channel: per class, the mean over its 50 validation images of the spatial maximum and
  minimum; the classes ranked by each; and, for the motor-vehicle classes of experiments/inception_paths, their ranks and
  the rank of the first vehicle image among all 50,000 sorted by the spatial maximum."""
  import json
  from experiments.inception_paths import VEHICLES
  st = np.load(stats_path)
  names = class_names(imagenet_dir)
  labels = st['labels']
  mx, mn = concat_stats(st, layer, 'max')[:, channel], concat_stats(st, layer, 'min')[:, channel]
  n_cls = len(names)
  cm = np.array([mx[labels == c].mean() for c in range(n_cls)])
  cn = np.array([mn[labels == c].mean() for c in range(n_cls)])
  by_max, by_min = np.argsort(-cm), np.argsort(cn)
  rank = {int(c): r for r, c in enumerate(by_max)}
  veh = [i for i, n in enumerate(names) if n in VEHICLES]
  order = np.argsort(-mx)
  first_vehicle_image = int(np.where(np.isin(labels[order], veh))[0][0])
  res = {'unit': f'{layer}:{channel}',
         'classes_by_mean_max': [(names[c], round(float(cm[c]), 1)) for c in by_max[:top]],
         'classes_by_mean_min': [(names[c], round(float(cn[c]), 1)) for c in by_min[:top]],
         'vehicle_class_ranks_by_mean_max': sorted((rank[c], names[c], round(float(cm[c]), 1)) for c in veh),
         'first_vehicle_image_rank_by_max': first_vehicle_image,
         'all_images_max_percentiles': {q: round(float(np.percentile(mx, q)), 1) for q in (50, 90, 99, 99.9)}}
  if out:
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    json.dump(res, open(out, 'w'), indent=1)
  print(json.dumps(res))


def main():
  p = argparse.ArgumentParser()
  p.add_argument('cmd', choices=['collect', 'analyze', 'profile'])
  p.add_argument('--layer', default='mixed4e')
  p.add_argument('--channel', type=int, default=55)
  p.add_argument('--val', required=True, help='collect: dir with validation parquet; analyze: the ImageNet dir with classes.py')
  p.add_argument('--weights', required=True)
  p.add_argument('--stats', default='results/e0/val_stats.npz')
  p.add_argument('--out', default='results/e0/val_stats.npz')
  p.add_argument('--batch', type=int, default=250)
  a = p.parse_args()
  if a.cmd == 'collect':
    collect(a.val, a.weights, a.out, a.batch)
  elif a.cmd == 'profile':
    assert a.out.endswith('.json'), 'profile writes JSON; pass --out <file>.json (the default is the statistics file)'
    profile(a.stats, a.val, a.layer, a.channel, a.out)
  else:
    analyze(a.stats, a.val, a.weights, a.out)


if __name__ == '__main__':
  main()
