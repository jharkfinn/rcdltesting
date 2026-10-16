"""Rare-class experiment (protocol v2, docs/EXACT_HISTORY.md Sec. 4.3): definitions shared by the analysis scripts
(experiments/rare_gonogo.py, experiments/convnext_rare.py).

Class sets (from rare_classes_v2.json, the WordNet distances D and the DINOv2 class-mean cosine distances):
  order[c]      the other 999 classes, visually nearest first (argsort of the cosine distance, self excluded), as in
                experiments/rare_select.py; vrank[c, k] = 1 + position of k in order[c] (vrank[c, c] = 0)
  special       selected classes (sets A and B), relabel controls and noise controls: not common in every run
  Near10(c)     order[c, :10], the plain 10 nearest classes
  Pi(c)         classes not special, not in Near10(c), not c (common in every run); for a relabel or noise control
                also not its source class (whose images are the control's probes or its relabelled training images)
  Far(c)        classes k in Pi(c) with vrank[c, k] > 200 and D[c, k] > 6
  N_own(c)      the other selected classes of c's set, level and stratum (13 per class at levels 10 and 100)
  cos10[c]      mean cosine distance to the classes of Near10(c); nW[c] = number of classes at WordNet distance <= 4

Fine groups (spec.npz of the run, experiments/rare_spec.py): Own_m(c) is the set of groups with owner c and kind own
image (1), own tenth (2), own subset (3), relabelled image (6) or noise image (7), i.e. the groups of c's own m_c
images (or of its relabelled images in its rare run).

Parts (attr*_{step}.pt, rcdl.attribution.GroupAttribution.compact_state): per tracked map (keys
'stage{s}.block{b}.pw_gate', '...pw_value', 'head'), C (G, out, in) fp16 with C_scale, Cb (G, out) with Cb_scale, the
fp32 remainder R, Rb and the decay factors a, ab: W = a W_0 + sum_g C_g + R exactly with C_g = C[g] / C_scale.
flat_parts concatenates weight and bias parts into one (G, out * in + out) row per group.

Probes: V_c = the 50 validation images of c in global validation (parquet) order, V_sel = its even positions,
V_est = its odd positions; Q = the atlas probe, the first 8 validation images of every class (experiments/
convnext_atlas.probe_set). All crops: shorter side 256, center 224, as rcdl.imagenet.val_batches. A relabel control
is probed on its source class's 50 validation images, a noise control on its 10 noise images (center 224 of
rcdl.imagenet.smooth_noise of the global indices that its rare run's spec flags as noise), in every run.

zeta_c(x) = z_c(x) - mean_k z_k(x), the centered logit. First-order effect of group set S on class c:
  E_S(c) = < d mean_{x in X} zeta_c(x) / d theta, sum_{g in S} C_g >
over the tracked maps' weights and biases theta (X = V_est by default). It is additive over groups, so every set's
effect is a sum of per-group effects E_g(c), computed for all groups at once from one gradient per class.
AUROC(pos, neg) = P(pos > neg) + P(pos = neg) / 2, from average ranks.
"""

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from rcdl.imagenet import MEAN, STD

OWN_KINDS = (1, 2, 3, 6, 7)          # own image, own tenth, own subset, relabelled image, noise image
KIND_NAMES = {0: 'class', 1: 'own_img', 2: 'own_tenth', 3: 'own_sub', 4: 'rel_decoy', 5: 'rand_decoy', 6: 'relabel',
              7: 'noise'}


# ------------------------------------------------------------------------------------------------------- class sets

class Selection:
  """The class sets of the protocol for one selection file (see the module docstring)."""

  def __init__(self, classes_json: str, wordnet: str, visual: str):
    sel = json.load(open(classes_json)) if isinstance(classes_json, str) else classes_json
    D = np.load(wordnet)['dist'].astype(int) if isinstance(wordnet, str) else np.asarray(wordnet, int)
    CS = (np.load(visual)['cos_dist'] if isinstance(visual, str) else np.asarray(visual)).astype(np.float64)
    n = len(CS)
    self.n = n
    self.D = D.copy()
    np.fill_diagonal(self.D, 99)
    self.order = np.argsort(CS + np.eye(n) * 9, 1)[:, :n - 1]     # as rare_select.py (same ties)
    self.cos10 = np.take_along_axis(CS, self.order[:, :10], 1).mean(1)
    self.nW = (self.D <= 4).sum(1)
    self.vrank = np.zeros((n, n), int)
    self.vrank[np.arange(n)[:, None], self.order] = np.arange(1, n)[None, :]
    self.sel = sel
    self.rows = {}
    for r in sel['classes']:
      self.rows[r['cls']] = dict(r, kind='selected')
    for kind in ('relabel', 'noise'):
      for r in sel.get(kind, []):
        self.rows[r['cls']] = dict(r, kind=kind, level=None, stratum=kind)
    self.selected = sorted(r['cls'] for r in sel['classes'])
    self.special = set(self.rows)

  def near10(self, c: int) -> List[int]:
    return [int(k) for k in self.order[c, :10]]

  def near(self, c: int, k: int) -> List[int]:
    return [int(j) for j in self.order[c, :k]]

  def pi(self, c: int) -> List[int]:
    bad = self.special | set(self.near10(c)) | {c}
    if self.rows.get(c, {}).get('source') is not None:   # a control's source class is not a random class for it
      bad.add(int(self.rows[c]['source']))
    return [k for k in range(self.n) if k not in bad]

  def far(self, c: int) -> List[int]:
    return [k for k in self.pi(c) if self.vrank[c, k] > 200 and self.D[c, k] > 6]

  def n_own(self, c: int) -> List[int]:
    r = self.rows[c]
    return sorted(k for k, q in self.rows.items() if k != c and q['kind'] == r['kind'] and q['set'] == r['set']
                  and q['level'] == r['level'] and q['stratum'] == r['stratum'])

  def condition(self, c: int, run_set: Optional[str]) -> str:
    """'rare' if c's set is the run's rare set (absent, level 10 or 100; dropped for controls), else 'common'."""
    return 'rare' if run_set in ('A', 'B') and self.rows[c]['set'] == run_set else 'common'

  def cell(self, c: int) -> str:
    r = self.rows[c]
    return f"{r['set']}/{r['stratum']}/{r['level']}" if r['kind'] == 'selected' else f"{r['set']}/{r['kind']}"


# ------------------------------------------------------------------------------------------------------- spec, parts

def load_spec(run: str) -> Optional[dict]:
  p = os.path.join(run, 'spec.npz')
  if not os.path.exists(p):
    return None
  sp = dict(np.load(p))
  sp['meta'] = json.loads(str(sp['meta']))
  return sp


def own_groups(spec: dict, c: int) -> np.ndarray:
  """Fine-group ids of Own_m(c) (see the module docstring)."""
  return np.flatnonzero((spec['group_owner'] == c) & np.isin(spec['group_kind'], OWN_KINDS))


def load_parts(path: str) -> Dict[str, dict]:
  """A compact state on the CPU (fp16 parts as stored)."""
  return torch.load(path, map_location='cpu', weights_only=False)


def flat_parts(rec: dict, groups=None, device='cpu', dtype=torch.float32) -> torch.Tensor:
  """(G, out * in + out) weight and bias parts of the chosen groups (all if None), unscaled, in dtype."""
  C, Cb = rec['C'], rec['Cb']
  if groups is not None:
    idx = torch.as_tensor(np.asarray(groups), dtype=torch.long)
    C, Cb = C[idx], Cb[idx]
  C = C.to(device).to(dtype).flatten(1) / rec.get('C_scale', 1.0)
  Cb = Cb.to(device).to(dtype) / rec.get('Cb_scale', 1.0)
  return torch.cat([C, Cb], 1)


def fold(parts_f: Dict[str, dict], group_class: np.ndarray, n_classes: int, chunk: int = 256) -> Dict[str, dict]:
  """The fine parts summed per class (group_class), in fp32: {map: {'C', 'Cb'}}, comparable with the class split."""
  out = {}
  gc = torch.as_tensor(np.asarray(group_class, np.int64))
  for k, rec in parts_f.items():
    C = torch.zeros((n_classes,) + tuple(rec['C'].shape[1:]), dtype=torch.float32)
    Cb = torch.zeros((n_classes,) + tuple(rec['Cb'].shape[1:]), dtype=torch.float32)
    for i in range(0, rec['C'].shape[0], chunk):
      C.index_add_(0, gc[i:i + chunk], rec['C'][i:i + chunk].float() / rec.get('C_scale', 1.0))
      Cb.index_add_(0, gc[i:i + chunk], rec['Cb'][i:i + chunk].float() / rec.get('Cb_scale', 1.0))
    out[k] = {'C': C, 'Cb': Cb}
  return out


def class_parts(parts: Dict[str, dict]) -> Dict[str, dict]:
  """A class-split compact state unscaled to fp32 {map: {'C', 'Cb'}} (the form fold returns)."""
  return {k: {'C': r['C'].float() / r.get('C_scale', 1.0), 'Cb': r['Cb'].float() / r.get('Cb_scale', 1.0)}
          for k, r in parts.items()}


def tracked_modules(model, keys: Sequence[str]) -> Dict[str, nn.Linear]:
  """{key: nn.Linear} for tracked-map keys 'stage{s}.block{b}.pw_gate' / '.pw_value' and 'head'."""
  blocks = dict(model.blocks()) if hasattr(model, 'blocks') else {}
  out = {}
  for k in keys:
    if k == 'head':
      out[k] = model.head
    else:
      b, m = k.rsplit('.', 1)
      out[k] = getattr(blocks[b], m)
  return out


def region(key: str) -> str:
  """'stage0' ... 'stage3' for unit maps, 'head' for the classifier."""
  return 'head' if key == 'head' else key.split('.')[0]


@torch.no_grad()
def excise(mods: Dict[str, nn.Linear], parts: Dict[str, dict], groups, keys: Sequence[str]):
  """In place: W <- W - sum_{g in groups} C_g (and the bias) in the maps `keys`. Returns undo(), which restores the
  saved weights bitwise. Removes exactly what those groups wrote (not a retraining without them)."""
  saved = []
  idx = torch.as_tensor(np.asarray(groups), dtype=torch.long)
  for k in keys:
    lin, rec = mods[k], parts[k]
    dW = (rec['C'][idx].double().sum(0) / rec.get('C_scale', 1.0)).to(lin.weight.device)
    db = (rec['Cb'][idx].double().sum(0) / rec.get('Cb_scale', 1.0)).to(lin.bias.device)
    saved.append((lin, lin.weight.detach().clone(), lin.bias.detach().clone()))
    lin.weight.sub_(dW.to(lin.weight.dtype))
    lin.bias.sub_(db.to(lin.bias.dtype))

  def undo():
    with torch.no_grad():
      for lin, w, b in saved:
        lin.weight.copy_(w)
        lin.bias.copy_(b)
  return undo


# ----------------------------------------------------------------------------------------------- forward, gradients

def normalize(x_u8: torch.Tensor, device) -> torch.Tensor:
  return (x_u8.to(device).float() - MEAN.to(device)[:, None, None]) / STD.to(device)[:, None, None]


@torch.no_grad()
def logits_of(model, x_u8: torch.Tensor, device, batch: int = 250, dtype=None) -> torch.Tensor:
  """(N, n_classes) logits on the CPU for uint8 images (or float images already normalized, if x is float), in the
  model's dtype unless given."""
  dtype = dtype or next(model.parameters()).dtype
  out = []
  for i in range(0, len(x_u8), batch):
    xb = x_u8[i:i + batch]
    xb = normalize(xb, device) if xb.dtype == torch.uint8 else xb.to(device)
    out.append(model(xb.to(dtype)).cpu())
  return torch.cat(out)


def zeta(z: torch.Tensor, c) -> torch.Tensor:
  """Centered logit of class c (int, or a tensor of one class per row)."""
  zc = z[:, int(c)] if isinstance(c, (int, np.integer)) else z.gather(1, c.view(-1, 1))[:, 0]
  return zc - z.mean(1)


def class_gradients(model, mods: Dict[str, nn.Linear], xs: List[torch.Tensor], classes: Sequence[int], device,
                    dtype=None) -> Dict[str, torch.Tensor]:
  """{map: (n_cls, out * in + out)} gradient of mean_x zeta_c(x) over xs[i] (uint8 or normalized) for c = classes[i],
  with respect to each map's weight and bias (flattened as flat_parts)."""
  dtype = dtype or next(model.parameters()).dtype
  params = [p for k in mods for p in (mods[k].weight, mods[k].bias)]
  prev = [p.requires_grad for p in params]
  for p in params:
    p.requires_grad_(True)
  out = {k: [] for k in mods}
  try:
    for x, c in zip(xs, classes):
      xb = normalize(x, device) if x.dtype == torch.uint8 else x.to(device)
      z = model(xb.to(dtype))
      g = torch.autograd.grad(zeta(z, int(c)).mean(), params)
      for j, k in enumerate(mods):
        out[k].append(torch.cat([g[2 * j].flatten(), g[2 * j + 1]]).detach())
  finally:
    for p, r in zip(params, prev):
      p.requires_grad_(r)
  return {k: torch.stack(v) for k, v in out.items()}


@torch.no_grad()
def group_effects(parts: Dict[str, dict], grads: Dict[str, torch.Tensor], chunk: int = 512,
                  dtype=None) -> Dict[str, torch.Tensor]:
  """{map: (G, n_cls)} first-order effects E_g(c) = <grad_c, part_g> of every group g, per tracked map (in the
  gradients' dtype unless given)."""
  out = {}
  for k, gr in grads.items():
    dtype = dtype or gr.dtype
    rec = parts[k]
    G = rec['C'].shape[0]
    res = torch.empty(G, gr.shape[0], dtype=dtype, device=gr.device)
    for i in range(0, G, chunk):
      res[i:i + chunk] = flat_parts(rec, np.arange(i, min(i + chunk, G)), gr.device, dtype) @ gr.to(dtype).T
    out[k] = res
  return out


def by_region(E: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
  """Per-map effects summed per region (stage0..stage3, head) and in total ('all', 'units')."""
  out = {}
  for k, e in E.items():
    r = region(k)
    out[r] = out.get(r, 0) + e
  out['units'] = sum(v for r, v in out.items() if r.startswith('stage'))
  out['all'] = out['units'] + out.get('head', 0)
  return out


# ---------------------------------------------------------------------------------------------------- statistics

def rankdata(a) -> np.ndarray:
  """1-based ranks with ties given their average rank."""
  a = np.asarray(a, np.float64)
  _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
  cum = np.cumsum(cnt)
  return (cum - (cnt - 1) / 2.0)[inv]


def auroc(pos, neg) -> float:
  """P(pos > neg) + P(pos = neg) / 2 (Mann-Whitney U / (n_pos n_neg))."""
  pos, neg = np.asarray(pos, np.float64).ravel(), np.asarray(neg, np.float64).ravel()
  if len(pos) == 0 or len(neg) == 0:
    return float('nan')
  r = rankdata(np.concatenate([pos, neg]))
  return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def spearman(a, b) -> float:
  a, b = rankdata(a), rankdata(b)
  if a.std() == 0 or b.std() == 0:
    return float('nan')
  return float(np.corrcoef(a, b)[0, 1])


def pct_rank(x: float, null) -> float:
  """Fraction of the null below x, ties counted half (0.5 = the null's median)."""
  null = np.asarray(null, np.float64)
  return float((null < x).mean() + 0.5 * (null == x).mean()) if len(null) else float('nan')


def clean_json(o):
  """A JSON-safe copy: numpy scalars and arrays to Python, non-finite floats to None (strict JSON has no NaN)."""
  if isinstance(o, dict):
    return {str(k): clean_json(v) for k, v in o.items()}
  if isinstance(o, (list, tuple)):
    return [clean_json(v) for v in o]
  if hasattr(o, 'tolist'):
    return clean_json(o.tolist())
  if isinstance(o, float) and not np.isfinite(o):
    return None
  return o


# ------------------------------------------------------------------------------------------------------- images

def val_class_index(val_dir: str) -> np.ndarray:
  """Validation labels in global (parquet) order."""
  from experiments.convnext_umbrella import val_index
  return val_index(val_dir)[1]


def load_val_images(val_dir: str, gidx: Sequence[int], cache: Optional[str] = None) -> torch.Tensor:
  """uint8 (N, 3, 224, 224) validation crops of global indices gidx (in that order), cached by index list. The
  parquet files are read one at a time and released (the same crops as convnext_umbrella.val_images, which keeps
  every file's image column in memory: about 6.4 GB for the whole validation set)."""
  import pyarrow.parquet as pq
  import torch.nn.functional as F
  from torchvision.io import ImageReadMode, decode_image
  from experiments.convnext_umbrella import val_index
  gidx = np.asarray(gidx, np.int64)
  if cache and os.path.exists(cache):
    d = torch.load(cache, weights_only=False)
    if np.array_equal(d['gidx'], gidx):
      return d['x']
  rows, _ = val_index(val_dir)
  out = torch.empty(len(gidx), 3, 224, 224, dtype=torch.uint8)
  by_file = {}
  for j, g in enumerate(gidx):
    f, i = rows[int(g)]
    by_file.setdefault(f, []).append((j, i))
  for f, items in by_file.items():
    col = pq.read_table(f, columns=['image']).column('image')
    for j, i in items:
      im = decode_image(torch.frombuffer(bytearray(col[i].as_py()['bytes']), dtype=torch.uint8),
                        mode=ImageReadMode.RGB)
      h, w = im.shape[-2:]
      s = 256 / min(h, w)
      im = F.interpolate(im[None].float(), size=(max(256, round(h * s)), max(256, round(w * s))), mode='bilinear',
                         antialias=True, align_corners=False)[0]
      H, W = im.shape[-2:]
      top, left = (H - 224) // 2, (W - 224) // 2
      out[j] = im[:, top:top + 224, left:left + 224].round().clamp(0, 255).to(torch.uint8)
    del col
  if cache:
    torch.save({'gidx': gidx, 'x': out}, cache)
  return out


def special_val_classes(sel: Selection) -> List[int]:
  """Classes whose 50 validation images the analyses load: the special classes and the relabel sources."""
  return sorted(set(sel.rows) | {int(r['source']) for r in sel.sel.get('relabel', [])})


def load_probe_images(val_dir: str, sel: Selection, val_cache: Optional[str], probe_cache: Optional[str],
                      images: str = ''):
  """(V, Vpos, Q, Qy): V uint8 validation crops of special_val_classes (50 each, global order), Vpos[c] its rows
  in V, and the probe Q with labels Qy. images: a pre-built file {V, Vlabels, Q, Qy} instead (tests)."""
  if images:
    d = torch.load(images, weights_only=False)
    V, vlab, Q, Qy = d['V'], np.asarray(d['Vlabels']), d['Q'], np.asarray(d['Qy'])
  else:
    from experiments.convnext_atlas import probe_set
    vlabels = val_class_index(val_dir)
    gidx = np.concatenate([np.flatnonzero(vlabels == c) for c in special_val_classes(sel)])
    V = load_val_images(val_dir, gidx, val_cache)
    vlab = vlabels[gidx]
    Q, Qy = probe_set(val_dir, 8, probe_cache)
    Qy = Qy.numpy()
  Vpos = {int(c): np.flatnonzero(vlab == c) for c in np.unique(vlab)}
  return V, Vpos, Q, Qy


def noise_images(specs: Sequence[Optional[dict]], sel: Selection) -> Dict[int, torch.Tensor]:
  """uint8 (10, 3, 224, 224) noise images of each noise control: the images its rare run's spec flags as noise and
  labels c (center 224 of smooth_noise), so every run probes the same images. specs: the run's spec and, for runs
  where the controls are common (C0, the other set's run), spec_A and spec_B (None entries are skipped)."""
  from rcdl.imagenet import smooth_noise
  out = {}
  for r in sel.sel.get('noise', []):
    for sp in specs:
      if sp is None or not len(sp.get('noise', ())):
        continue
      mine = [int(i) for i in sp['noise'] if sp['labels'][i] == r['cls']]
      if mine:
        out[r['cls']] = torch.stack([smooth_noise(i)[:, 16:240, 16:240] for i in mine])
        break
  return out


def probes(sel: Selection, c: int, V, Vpos, noise_x=None) -> torch.Tensor:
  """uint8 probe images of class c: its 50 validation images; for a relabel control the source class's; for a noise
  control its 10 noise images (its own validation images if noise_x lacks them). V_sel = even positions, V_est =
  odd positions."""
  row = sel.rows[c]
  if row['kind'] == 'relabel':
    return V[Vpos[row['source']]]
  if row['kind'] == 'noise' and noise_x is not None and c in noise_x:
    return noise_x[c]
  return V[Vpos[c]]


def probe_kind(sel: Selection, c: int, noise_x=None) -> str:
  row = sel.rows[c]
  if row['kind'] == 'relabel':
    return 'source_val'
  if row['kind'] == 'noise':
    return 'noise' if noise_x is not None and c in noise_x else 'val (noise images unavailable)'
  return 'val'


def load_npz(path: str) -> Optional[dict]:
  if not path or not os.path.exists(path):
    return None
  return dict(np.load(path))


def snapshot_files(run: str) -> Dict[int, str]:
  """{step: path} of the replay snapshots snap_{step}.pt in the run directory and in its history/ folder (E1, E1r:
  converted history snapshots at other steps)."""
  out = {}
  for d in (os.path.join(run, 'history'), run):
    if os.path.isdir(d):
      for f in os.listdir(d):
        if f.startswith('snap_') and f.endswith('.pt') and f[5:-3].isdigit():
          out[int(f[5:-3])] = os.path.join(d, f)
  return out


def nearest_snapshot(snaps: Dict[int, str], step: int) -> Optional[int]:
  """The snapshot step nearest to `step` among steps > 0 (at step 0 AdamW's second moment is still zero, so it gives
  no step size), the earlier one on a tie; None if there is none."""
  cand = [s for s in snaps if s > 0]
  return min(cand, key=lambda s: (abs(s - step), s)) if cand else None


def train_center_u8(train_root: str, gidx: Sequence[int], noise: Sequence[int] = ()) -> torch.Tensor:
  """uint8 center crops (shorter side 256, center 224) of packed training images; indices in `noise` are the fixed
  smooth-noise images of the noise controls (256 x 256, center 224), as the loader gives them before augmentation."""
  import torch.nn.functional as F
  from torchvision.io import ImageReadMode, decode_jpeg
  from rcdl.imagenet import smooth_noise
  data = np.memmap(os.path.join(train_root, 'data.bin'), dtype=np.uint8, mode='r')
  off = np.load(os.path.join(train_root, 'offsets.npy'))
  noise = set(int(i) for i in noise)
  out = []
  for j in gidx:
    j = int(j)
    if j in noise:
      im = smooth_noise(j).float()
    else:
      raw = torch.from_numpy(np.array(data[off[j]:off[j + 1]]))
      im = decode_jpeg(raw, mode=ImageReadMode.RGB).float()
    h, w = im.shape[-2:]
    s = 256 / min(h, w)
    im = F.interpolate(im[None], size=(max(256, round(h * s)), max(256, round(w * s))), mode='bilinear',
                       antialias=True, align_corners=False)[0]
    H, W = im.shape[-2:]
    top, left = (H - 224) // 2, (W - 224) // 2
    out.append(im[:, top:top + 224, left:left + 224].round().clamp(0, 255).to(torch.uint8))
  return torch.stack(out) if out else torch.zeros(0, 3, 224, 224, dtype=torch.uint8)
