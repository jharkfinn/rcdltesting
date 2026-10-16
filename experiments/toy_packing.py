"""E8: what makes two features share a neuron, relatedness (similar inputs) or anticorrelation (never together)?

Cammarata ("Neural Networks Are Already Clean", 2026) argues that features share a neuron because they are related
(generalization); Toy Models of Superposition (Elhage et al. 2022) finds that features which rarely co-occur share a
dimension, often as antipodal pairs, because they do not interfere. In natural data the two coincide (sibling classes
are both similar and mutually exclusive), so this toy separates them with a 2 x 2 design:

  features i = (family f, member c); embeddings e_i in R^D with cos(e_i, e_j) = rho for members of one family (related),
  0 across families (unrelated);
  exclusive families: members never co-occur (related + anticorrelated);
  co-occurring families: members co-occur at the base rate (related + independent);
  exclusion pairs of families: no member of one co-occurs with a member of the other (unrelated + anticorrelated);
  all other cross-family pairs co-occur at the base rate (unrelated + independent).

A sample has 1 or 2 active features with values v ~ U(0.5, 1); input x = sum_active v_i e_i; the model reconstructs the
feature values: x -> Linear(D, m) -> Linear(m, H) + ReLU ("neurons") -> Linear(H, n) + ReLU (output bias starts at 0.1;
the output ReLU lets the model filter negative interference, as in Toy Models). The neuron layer is tracked
with rcdl.attribution (one group per active-feature set), so for every neuron we know which features wrote it and when.

Outputs per run (JSON): the response matrix R[neuron, feature] (pre-activation for a unit input of one feature), the
share of co-positive and antipodal pairs per pair type against the share expected by chance, and, from attribution
checkpoints, the order in which features entered each neuron's writer set.
"""

import argparse
import itertools
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from rcdl.attribution import GroupAttribution

PAIR_TYPES = ('related_anti', 'related_indep', 'unrelated_anti', 'unrelated_indep')


class World:
  """Families, embeddings, co-occurrence rules and the pair type of every feature pair."""

  def __init__(self, n_families=8, members=4, D=64, rho=0.6, p_two=0.5, seed=0):
    g = torch.Generator().manual_seed(seed)
    self.F, self.M, self.n = n_families, members, n_families * members
    self.family = torch.arange(self.n) // members
    self.exclusive = torch.arange(n_families) % 2 == 0           # even families: members never co-occur
    partner = torch.full((n_families,), -1)                     # exclusion pairs of families: (0, 1), (4, 5), ...
    for f in range(0, n_families - 1, 4):
      partner[f], partner[f + 1] = f + 1, f
    self.partner = partner
    base = torch.randn(n_families, D, generator=g)
    own = torch.randn(self.n, D, generator=g)
    base = base / base.norm(dim=1, keepdim=True)
    own = own - (own @ base[self.family].T).diagonal()[:, None] * base[self.family]   # orthogonal to the family base
    own = own / own.norm(dim=1, keepdim=True)
    emb = math.sqrt(rho) * base[self.family] + math.sqrt(1 - rho) * own                # cos = rho within a family
    self.emb = emb / emb.norm(dim=1, keepdim=True)
    self.p_two = p_two
    allowed = torch.ones(self.n, self.n, dtype=torch.bool)
    ptype = {}
    for i, j in itertools.combinations(range(self.n), 2):
      fi, fj = int(self.family[i]), int(self.family[j])
      if fi == fj:
        anti = bool(self.exclusive[fi])
        ptype[(i, j)] = 'related_anti' if anti else 'related_indep'
      else:
        anti = int(partner[fi]) == fj
        ptype[(i, j)] = 'unrelated_anti' if anti else 'unrelated_indep'
      if anti:
        allowed[i, j] = allowed[j, i] = False
    allowed.fill_diagonal_(False)
    self.allowed, self.ptype = allowed, ptype
    self.pairs = torch.nonzero(torch.triu(allowed, 1))          # allowed co-occurring pairs (i < j)
    self.group_of_pair = {(int(a), int(b)): self.n + k for k, (a, b) in enumerate(self.pairs.tolist())}
    self.n_groups = self.n + len(self.pairs)                    # singletons, then allowed pairs

  def batch(self, B, g):
    two = torch.rand(B, generator=g) < self.p_two
    single = torch.randint(0, self.n, (B,), generator=g)
    pair = self.pairs[torch.randint(0, len(self.pairs), (B,), generator=g)]
    vals = torch.zeros(B, self.n)
    v = 0.5 + 0.5 * torch.rand(B, 2, generator=g)
    rows = torch.arange(B)
    vals[rows[~two], single[~two]] = v[~two, 0]
    vals[rows[two], pair[two, 0]] = v[two, 0]
    vals[rows[two], pair[two, 1]] = v[two, 1]
    groups = torch.where(two, self.n + torch.searchsorted(self._pair_keys(), pair[:, 0] * self.n + pair[:, 1]), single)
    return vals @ self.emb, vals, groups

  def _pair_keys(self):
    if not hasattr(self, '_keys'):
      self._keys = self.pairs[:, 0] * self.n + self.pairs[:, 1]
    return self._keys


class Toy(nn.Module):
  def __init__(self, D, m, H, n, seed=0):
    super().__init__()
    torch.manual_seed(seed)
    self.down = nn.Linear(D, m, bias=False)
    self.neurons = nn.Linear(m, H)
    self.out = nn.Linear(H, n)
    with torch.no_grad():                            # with a zero start, about half of the ReLU outputs never turn on
      self.out.bias.fill_(0.1)                       # (14 of 32 dead at m = H = 32); 0.1 leaves none dead

  def forward(self, x):
    return F.relu(self.out(F.relu(self.neurons(self.down(x)))))


def responses(model, world):
  with torch.no_grad():
    return model.neurons(model.down(world.emb)).T            # (H, n): pre-activation for a unit input of one feature


def pair_stats(R, world, tau=0.3):
  """Per pair type: share of pairs that are co-positive (both > tau * max of the neuron) on some neuron, and antipodal
  (one > tau * max, the other < -tau * max) on some neuron, with the shares expected if pole memberships were shuffled
  across features (same pole sizes)."""
  H, n = R.shape
  scale = R.abs().amax(1, keepdim=True).clamp(min=1e-12)
  pos, neg = R > tau * scale, R < -tau * scale
  def shares(pos, neg):
    copos = (pos.float().T @ pos.float()) > 0                # (n, n): share a positive pole somewhere
    anti = ((pos.float().T @ neg.float()) + (neg.float().T @ pos.float())) > 0
    out = {}
    for t in PAIR_TYPES:
      idx = [(i, j) for (i, j), tt in world.ptype.items() if tt == t]
      if not idx:
        continue
      a, b = zip(*idx)
      out[t] = {'copositive': copos[list(a), list(b)].float().mean().item(),
                'antipodal': anti[list(a), list(b)].float().mean().item(), 'pairs': len(idx)}
    return out
  obs = shares(pos, neg)
  g = torch.Generator().manual_seed(0)
  null = {t: {'copositive': 0.0, 'antipodal': 0.0} for t in obs}
  for _ in range(50):
    perm = torch.randperm(n, generator=g)
    s = shares(pos[:, perm], neg[:, perm])
    for t in s:
      null[t]['copositive'] += s[t]['copositive'] / 50
      null[t]['antipodal'] += s[t]['antipodal'] / 50
  return {t: obs[t] | {'null_copositive': null[t]['copositive'], 'null_antipodal': null[t]['antipodal']} for t in obs}


def writer_features(parts_groups, world):
  """(K, ...) group parts -> (n, ...) per-feature writes: a pair group's part is split equally between its features."""
  n = world.n
  out = parts_groups[:n].clone()
  for (a, b), k in world.group_of_pair.items():
    out[a] += parts_groups[k] / 2
    out[b] += parts_groups[k] / 2
  return out


def run(args):
  world = World(args.families, args.members, args.D, args.rho, args.p_two, args.seed)
  model = Toy(args.D, args.m, args.H, world.n, args.seed)
  attr = GroupAttribution({'neurons': model.neurons}, n_groups=world.n_groups, optimizer='adamw', lr=args.lr,
                          betas=(0.9, 0.999), weight_decay=0.0, dtype=torch.float32)
  opt = torch.optim.AdamW(list(model.down.parameters()) + list(model.out.parameters()), lr=args.lr, weight_decay=0.0)
  g = torch.Generator().manual_seed(args.seed + 1)
  growth, log, t0 = [], [], time.time()
  for step in range(args.steps + 1):
    if step % args.log_every == 0:
      with torch.no_grad():
        x, v, _ = world.batch(4096, torch.Generator().manual_seed(123))
        loss = F.mse_loss(model(x), v).item()
        h = model.down(world.emb)                          # unit input of each feature at the neuron layer's input
        parts = attr.parts('neurons', h)                    # groups: (K, n_features, H)
        diag = parts['groups'][:, torch.arange(world.n), :]  # (K, n, H): group k's part of neuron response to feature i
        per_feat = writer_features(diag.sum(1), world)       # (n, H): writes by feature, summed over probe features
      growth.append({'step': step, 'writes': per_feat.T.tolist()})
      log.append({'step': step, 'loss': loss})
    if step == args.steps:
      break
    x, v, groups = world.batch(args.batch, g)
    opt.zero_grad()
    loss = F.mse_loss(model(x), v) * world.n
    loss.backward()
    lr = args.lr * min(1.0, (step + 1) / 200)
    for pg in opt.param_groups:
      pg['lr'] = lr
    attr.step(groups, lr)
    opt.step()
  R = responses(model, world)
  return {'args': vars(args), 'log': log, 'final_loss': log[-1]['loss'], 'R': R.tolist(),
          'pair_stats': pair_stats(R, world), 'growth': growth, 'residual': attr.residual('neurons'),
          'family': world.family.tolist(), 'exclusive': world.exclusive.tolist(), 'partner': world.partner.tolist(),
          'sec': time.time() - t0}


def main():
  p = argparse.ArgumentParser()
  p.add_argument('--families', type=int, default=8)
  p.add_argument('--members', type=int, default=4)
  p.add_argument('--D', type=int, default=64)
  p.add_argument('--m', type=int, default=8)
  p.add_argument('--H', type=int, default=8)
  p.add_argument('--rho', type=float, default=0.6)
  p.add_argument('--p_two', type=float, default=0.5)
  p.add_argument('--lr', type=float, default=3e-3)
  p.add_argument('--batch', type=int, default=256)
  p.add_argument('--steps', type=int, default=6000)
  p.add_argument('--log_every', type=int, default=200)
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--out', default='results/toy_packing')
  args = p.parse_args()
  res = run(args)
  os.makedirs(args.out, exist_ok=True)
  name = f'rho{args.rho}_m{args.m}_H{args.H}_p{args.p_two}_s{args.seed}.json'
  json.dump(res, open(os.path.join(args.out, name), 'w'))
  print(name, 'loss', round(res['final_loss'], 5), 'residual', res['residual'])
  for t, s in res['pair_stats'].items():
    print(f"  {t:16s} copositive {s['copositive']:.2f} (null {s['null_copositive']:.2f})  antipodal {s['antipodal']:.2f}"
          f" (null {s['null_antipodal']:.2f})  pairs {s['pairs']}")


if __name__ == '__main__':
  main()
