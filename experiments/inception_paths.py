"""E0b: what drives the car and the dog responses of the essay's four readers of the car neuron 4c:447.

The essay ("Cars and Animals") shows four mixed4d readers of 4c:447 (mixed4d:167, 241, 295, 417) and reads two as car
neurons and two as "generalized pose neurons", each "a clean detector of a pose shared across cars, dogs and other
objects", against Zoom In's reading of the car feature stored among dog neurons. The two readings predict different
inputs for a mixed reader: a shared pose detector is driven on its car images and on its dog images by the same mixed4c
channels; a neuron that packs a car feature and a dog feature is driven by car channels on cars and dog channels on dogs.

Method, exact for this piecewise-linear part of the model: mixed4d's branches are 1x1 convolutions, ReLUs, a 3x3 max
pool and 3x3 / 5x5 convolutions, so for one image the pre-ReLU response z of a mixed4d channel at its strongest
position is linear in the mixed4c output x on the image's own activation pattern:

  z = sum_c s_c + r,   s_c = sum_{y,x} x[c, y, x] * dz/dx[c, y, x],

where c runs over the 512 mixed4c channels, s_c is channel c's exact share and r is the part written by the biases. For
each reader, its 100 strongest validation images (by spatial maximum, the E0 statistics) are split into motor vehicles
and dogs; the mean share vectors of the two groups are compared (cosine, and how much of one group's z the other
group's top channels write), against the same comparison between two random halves of one group. Also: zeroing
4c:447 and measuring the reader's maximum again (the causal effect of the car neuron on car and on dog images), and the
class profile of each top input channel.

Usage: python3 -m experiments.inception_paths [readers | readout] --stats results/e0/val_stats.npz
       --val <validation parquet dir> --imagenet <dir with classes.py> --weights inception5h.pth --out <file>.json
('readout' is E0c, see its docstring.)
"""

import argparse
import glob
import json
import os

import numpy as np
import torch

from experiments.inception_e0 import channel_index, class_names, concat_stats, load_model, preprocess

READERS = (167, 241, 295, 417)
VEHICLES = ('ambulance', 'beach wagon', 'cab', 'convertible', 'jeep', 'limousine', 'minivan', 'Model T', 'racer',
            'sports car', 'minibus', 'police van', 'moving van', 'fire engine', 'trolleybus', 'school bus', 'snowplow',
            'pickup', 'tow truck', 'trailer truck', 'garbage truck', 'recreational vehicle', 'go-kart', 'golfcart',
            'car wheel', 'grille', 'half track', 'tractor', 'forklift', 'streetcar', 'amphibian', 'tank')
DOGS = set(range(151, 269))


def load_images(val_dir: str, idx):
  """JPEG bytes of the validation images with these global indices (the iteration order of the E0 statistics)."""
  import pyarrow.parquet as pq
  files = sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True))
  starts = np.cumsum([0] + [pq.ParquetFile(f).metadata.num_rows for f in files])
  out, tables = {}, {}
  for i in sorted(set(int(i) for i in idx)):
    fi = int(np.searchsorted(starts, i, side='right') - 1)
    if fi not in tables:
      tables[fi] = pq.read_table(files[fi], columns=['image'])
    out[i] = tables[fi].slice(int(i - starts[fi]), 1).to_pylist()[0]['image']['bytes']
  return [out[int(i)] for i in idx]


class Probe:
  """Captures a mixed layer's output (as a leaf that requires grad, optionally with channels zeroed) and one branch
  output of the next mixed layer."""

  def __init__(self, model, branch: str, source: str = 'mixed4c'):
    self.zero, self.x, self.z = (), None, None
    getattr(model, source).register_forward_hook(self._cat)
    getattr(model, branch).register_forward_hook(lambda m, i, o: setattr(self, 'z', o))

  def _cat(self, m, i, o):
    o = o.detach().clone()
    for c in self.zero:
      o[:, c] = 0
    self.x = o.requires_grad_(True)
    return self.x


def shares(model, probe, x, ch):
  """Exact per-mixed4c-channel shares of the channel's response at its strongest position, plus z and the bias part."""
  model(x)
  z_map = probe.z[:, ch]                                  # (B, H, W)
  flat = z_map.flatten(1)
  pos = flat.argmax(1)
  z = flat.gather(1, pos[:, None])[:, 0]
  (g,) = torch.autograd.grad(z.sum(), probe.x)
  s = (g * probe.x).sum((2, 3))                           # (B, 512)
  return s.detach(), z.detach(), (z - s.sum(1)).detach()


def max_response(model, probe, x, ch, zero=()):
  probe.zero = zero
  with torch.no_grad():
    model(x)
  probe.zero = ()
  return probe.z[:, ch].flatten(1).amax(1)


def cos(a, b):
  return float(a @ b / (a.norm() * b.norm()).clamp(min=1e-12))


def cross_share(s_from, s_to, k=10):
  """Share of group `to`'s mean z written by group `from`'s top-k channels."""
  top = torch.argsort(-s_from.mean(0))[:k]
  return float(s_to[:, top].sum(1).mean() / s_to.sum(1).mean())


def run(stats_path, val_dir, imagenet_dir, weights, out, k_img=100, seed=0):
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  st = np.load(stats_path)
  names = class_names(imagenet_dir)
  labels = st['labels']
  veh = {i for i, n in enumerate(names) if n in VEHICLES}
  missing = sorted(set(VEHICLES) - {names[i] for i in veh})
  m4c = concat_stats(st, 'mixed4c', 'max')
  m4d = concat_stats(st, 'mixed4d', 'max')

  def profile(c, k=50):
    idx = np.argsort(-m4c[:, c])[:k]
    counts = {}
    for l in labels[idx]:
      counts[names[l]] = counts.get(names[l], 0) + 1
    return {'top_classes': sorted(counts.items(), key=lambda t: -t[1])[:5],
            'dogs': int(sum(l in DOGS for l in labels[idx])), 'vehicles': int(sum(l in veh for l in labels[idx]))}

  rng = np.random.default_rng(seed)
  res = {'n_val': int(len(labels)), 'vehicle_classes': sorted(names[i] for i in veh), 'vehicle_names_not_found': missing,
         'readers': {}}
  model = load_model(weights, dev)
  for p in model.parameters():
    p.requires_grad_(False)
  for r in READERS:
    branch, ch = channel_index(model, 'mixed4d', r)
    probe = Probe(model, branch)
    idx = np.argsort(-m4d[:, r])[:k_img]
    raw = load_images(val_dir, idx)
    S, Z, R, Zab = [], [], [], []
    for i in range(0, len(raw), 25):
      x = preprocess(raw[i:i + 25], dev)
      s, z, rem = shares(model, probe, x, ch)
      S.append(s.cpu()); Z.append(z.cpu()); R.append(rem.cpu())
      Zab.append(max_response(model, probe, x, ch, zero=(447,)).cpu())
    model._forward_hooks.clear()
    for mod in model.modules():
      mod._forward_hooks.clear()
    S, Z, R, Zab = torch.cat(S).double(), torch.cat(Z).double(), torch.cat(R).double(), torch.cat(Zab).double()
    lab = labels[idx]
    is_v = torch.tensor([l in veh for l in lab])
    is_d = torch.tensor([l in DOGS for l in lab])
    rec = {'branch': branch, 'branch_channel': ch, 'n_images': len(idx),
           'stats_check_max_abs_diff': float(np.abs(Z.numpy() - m4d[idx, r]).max()),
           'bias_part_mean_share': float((R / Z).mean()),
           'n_vehicles': int(is_v.sum()), 'n_dogs': int(is_d.sum()), 'groups': {}}
    for gname, mask in (('vehicles', is_v), ('dogs', is_d), ('other', ~(is_v | is_d))):
      if mask.sum() == 0:
        continue
      s, z, zab = S[mask], Z[mask], Zab[mask]
      mean_share = s.mean(0) / z.mean()
      top = torch.argsort(-s.mean(0))[:8]
      rec['groups'][gname] = {
        'n': int(mask.sum()), 'mean_z': float(z.mean()),
        'share_447': float(mean_share[447]),
        'ablate_447_mean_drop': float(((z - zab) / z).mean()),
        'top_inputs': [{'channel': int(c), 'share': float(mean_share[c]), **profile(int(c))} for c in top]}
    if is_v.sum() >= 4 and is_d.sum() >= 4:
      sv, sd = S[is_v], S[is_d]
      halves = []
      for sg in (sv, sd):
        cs = []
        for _ in range(200):
          perm = torch.from_numpy(rng.permutation(len(sg)))
          a, b = sg[perm[:len(sg) // 2]].mean(0), sg[perm[len(sg) // 2:]].mean(0)
          cs.append(cos(a, b))
        halves.append(float(np.mean(cs)))
      rec['vehicle_vs_dog'] = {
        'cosine_of_mean_shares': cos(sv.mean(0), sd.mean(0)),
        'cosine_between_random_halves_vehicles': halves[0], 'cosine_between_random_halves_dogs': halves[1],
        'dog_z_written_by_top10_vehicle_channels': cross_share(sv, sd),
        'vehicle_z_written_by_top10_vehicle_channels': cross_share(sv, sv),
        'vehicle_z_written_by_top10_dog_channels': cross_share(sd, sv),
        'dog_z_written_by_top10_dog_channels': cross_share(sd, sd)}
    res['readers'][f'mixed4d:{r}'] = rec
    print(json.dumps({f'mixed4d:{r}': {k: v for k, v in rec.items() if k != 'groups'}}), flush=True)
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  json.dump(res, open(out, 'w'), indent=1)
  print('saved', out)


def clear_hooks(model):
  for mod in model.modules():
    mod._forward_hooks.clear()


def readout(stats_path, val_dir, imagenet_dir, weights, out, k_img=100, n_readers=8, mixed=(241, 295)):
  """E0c: does mixed4e read the car half of the mixed readers back out?

  For each mixed4d unit u in `mixed`: its strongest mixed4e readers by expanded weight (inception_e0.expanded_inputs,
  linearized through the bottlenecks), each with its class profile; then, exactly as in `run` but one layer later, the
  share of each reader's response written by u on the reader's own strongest vehicle images and dog images; and the
  interference test: on u's strongest dog images (where u is high because of dogs), the reader's response z, the share
  written by u (what u's dog half pushes into it) and the reader's percentile among all 50,000 images."""
  from experiments.inception_e0 import expanded_inputs
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  st = np.load(stats_path)
  names = class_names(imagenet_dir)
  labels = st['labels']
  veh = {i for i, n in enumerate(names) if n in VEHICLES}
  m4d, m4e = concat_stats(st, 'mixed4d', 'max'), concat_stats(st, 'mixed4e', 'max')
  sd = torch.load(weights, map_location='cpu', weights_only=True)
  model = load_model(weights, dev)
  for p in model.parameters():
    p.requires_grad_(False)

  def profile(a, k=50):
    idx = np.argsort(-a)[:k]
    counts = {}
    for l in labels[idx]:
      counts[names[l]] = counts.get(names[l], 0) + 1
    return {'top_classes': sorted(counts.items(), key=lambda t: -t[1])[:5],
            'dogs': int(sum(l in DOGS for l in labels[idx])), 'vehicles': int(sum(l in veh for l in labels[idx]))}

  def exact(reader, idx):
    branch, ch = channel_index(model, 'mixed4e', reader)
    probe = Probe(model, branch, source='mixed4d')
    S, Z = [], []
    raw = load_images(val_dir, idx)
    for i in range(0, len(raw), 25):
      s, z, _ = shares(model, probe, preprocess(raw[i:i + 25], dev), ch)
      S.append(s.cpu()); Z.append(z.cpu())
    clear_hooks(model)
    return torch.cat(S).double(), torch.cat(Z).double()

  res = {'n_val': int(len(labels)), 'units': {}}
  for u in mixed:
    w = torch.stack([expanded_inputs(sd, 'mixed4e', c) for c in range(m4d.shape[1])])[u]     # (832,) u -> 4e
    rec = {'readers': []}
    order_u = np.argsort(-m4d[:, u])
    dog_idx_u = [int(i) for i in order_u if labels[i] in DOGS][:k_img]
    for r in torch.argsort(-w)[:n_readers].tolist():
      prof = profile(m4e[:, r])
      order = np.argsort(-m4e[:, r])
      entry = {'reader': r, 'expanded_weight': float(w[r]), **prof}
      for gname, keep in (('vehicles', veh), ('dogs', DOGS)):
        idx = [int(i) for i in order if labels[i] in keep][:k_img]
        S, Z = exact(r, idx)
        entry[f'own_top_{gname}'] = {'mean_z': float(Z.mean()), 'share_from_u': float(S[:, u].mean() / Z.mean()),
                                     'share_from_241_295': float(S[:, list(mixed)].sum(1).mean() / Z.mean())}
      S, Z = exact(r, dog_idx_u)
      pct = lambda v: float((m4e[:, r] < v).mean() * 100)
      entry['on_u_top_dog_images'] = {'mean_z': float(Z.mean()), 'percentile_of_mean_z': pct(float(Z.mean())),
                                      'mean_write_from_u': float(S[:, u].mean()),
                                      'mean_write_from_rest': float((S.sum(1) - S[:, u]).mean())}
      rec['readers'].append(entry)
      print(json.dumps({f'mixed4d:{u}': entry}), flush=True)
    res['units'][f'mixed4d:{u}'] = rec
  os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
  json.dump(res, open(out, 'w'), indent=1)
  print('saved', out)


if __name__ == '__main__':
  p = argparse.ArgumentParser()
  p.add_argument('cmd', nargs='?', default='readers', choices=['readers', 'readout'])
  p.add_argument('--stats', required=True)
  p.add_argument('--val', required=True)
  p.add_argument('--imagenet', required=True)
  p.add_argument('--weights', required=True)
  p.add_argument('--out', default=None)
  p.add_argument('--k', type=int, default=100)
  a = p.parse_args()
  if a.cmd == 'readers':
    run(a.stats, a.val, a.imagenet, a.weights, a.out or 'results/e0/paths.json', a.k)
  else:
    readout(a.stats, a.val, a.imagenet, a.weights, a.out or 'results/e0/readout.json', a.k)
