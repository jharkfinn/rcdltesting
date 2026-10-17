"""Experiment 2 (protocol v2): exits, starvation and first in line, from the exact per-class record of a ConvNeXt-SwiGLU
run (E1; E1r with its random-group split; E4). Design, critique and protocol: docs/EXACT_HISTORY.md Sec. 4.1.

Notation. Block b, unit n, class c, training group k (a class; under the random split a random group r of E1r's shadow
split); n_c the number of training images of class c, n_r that of random group r. Probe: all validation images of
every class (50; 224 center crops of the 256 resize, validation-file order within a class) in two halves, H1 (images
1-25) and H2 (26-50). Every selection (events, cores, controls, leaders, orientation) uses one half, the selection half;
every term is measured on the other, the measurement half. Both directions are run: H1 -> H2 ('s0m1', primary) and
H2 -> H1 ('s1m0', replication). G16: the trainer's checkpoints (events and primary splits). G+: G16 plus the history
snapshots that pass the integrity checks (path-dependence check, second moments, finer timing).

Stock (stock_{t}.pt, stockh_{t}.pt; per grid point t, half and unit): p*_t(x) the unit's strongest position on image x;
R_c(t) = mean_{x in c} u_t(x, p*_t(x)), with u in fp64 from the model's weights; Gm, Vm the class means of the gate and
value pre-activations at p*; Abar, Bbar the class means of A = SiLU'(g) v and B = SiLU(g) at p*; Hs(c) the class mean
of the input h at p* (stockh). sd_n(t) = std over classes of R_c(t); z_n,c(t) = (R_c(t) - mean_c' R_c'(t)) / sd_n(t);
rank_c(t) = mid-rank of R_c(t), decreasing (1 = strongest); Top_n(t) = {c: rank_c(t) <= P}. Units with sd_n below
`sd_floor` (1e-3) times the network median (per grid point and half) are skipped: at event selection on the selection
half (exits at t_pk, entries at t_first; counted in events.json as low_sd_exits, low_sd_entries), in X1-X3 by the
selection half at the event's checkpoint, and in X3 S2, X4 and X5 per interval start on both halves (cells set to 0 or
left out).

Intervals (xhw_{t1}_{t2}.pt; per interval, half and block): the class means of rcdl.interval_split with the reference
p*_t1 (Hbar_M, qbar_M, Phi, Pi, dR, R1, R2, cross) and, for the event pairs (unit, class) listed in events.json,
Hcross_M and the p*_t2-reference means (Hbar2_M, qbar2_M, Phi2, Pi2). With the per-group writes of attr_{t}.pt
(attrR_{t}.pt for the random split) every term T_k(c), F_k(c), F_0(c), E(c) of every group on every class is a
(K x d)(d x C) product (rcdl.interval_split.column_terms), and sum_k T_k + sum_k F_k + F_0 + E + Phi + Pi = dR is
checked per block and interval. R_t = W_t - a_t W0 - sum_k C_t^(k) is computed in fp64 at every grid point. fp16
storage bounds (rcdl.interval_split) are propagated into the terms of the G16 and G+ intervals; nothing is excluded.
Since R absorbs the rounding, a change of the stored parts keeps F_own + F_rest and sum_k T_k + E fixed (F_rest's
bound is F_own's; E's is the writes' total, reported only). X1: every classification is recomputed at four corners of
the bounds that keep both invariants (so the total excess fall is unchanged). X2: entries within the bound are counted
and the own excess-rise difference is recomputed at the same corners. X3: the pushes carry the half-ulp bounds of the
two bias parts; S1 labels that change at push +- bound are counted. X4, X5: own capture cells within the bound are
counted; F1 is recomputed with O_L at +- its bound. Snapshot intervals use fp32 full states (no bound computed).

Gradient state (gstate_{t}.pt, per half, unit and class): theta = [gate row, gate bias, value row, value bias] of the
unit; j* = du/dtheta at p*; G = sum_p du(p)/dtheta; Ebar = the per-image gradient of -log softmax_y (clean image) with
respect to theta; D = the AdamW preconditioner 1 / (sqrt(v / (1 - 0.999^t)) + 1e-8) of the nearest usable snapshot
within 1,500 steps (approximate; 1 before step 1,000 or without a snapshot; Muon: 1 for the weights and the biases,
so that kappa and beta are not carried by the two AdamW-preconditioned bias coordinates alone). kappa = <mean_c j*, D
mean_c G> / sd_n (gating); beta = -<mean_c j*, D mean_c Ebar> / sd_n (first-order allocation with error and readers); at
snapshot steps mu = -(lr_t / (1 - 0.9^t)) <mean_c j*, D_t m_t^(c)> / sd_n with lr_t the learning rate of update t
(the exact AdamW write of that update, to first order in u; pooled only).

Events (events.json, frozen with the code hash before any decomposition; G16 and the selection half; P = 20):
  exit of c from unit n: c in Top_n at >= 2 checkpoints with step >= 1,000, best rank <= 10 there, rank at the last
    checkpoint T > 40. t_pk = checkpoint of the best rank (latest on ties); t_last = last checkpoint in Top_n; t_out =
    first checkpoint after t_last with rank > 40; window W = (t_pk, t_out]. Phase by the step of t_pk: < 2,500;
    2,500-10,000; >= 10,000.
  replicated: also an exit on the measurement half (secondary, 'rep_relaxed': best rank <= 15, final rank > 30 there);
    it selects on the half every term is measured on, so it defines only the validity gate and sensitivity analyses.
    anchored: >= 2 of Core_n (the 5 strongest classes at T) are in Top_n(t_pk); otherwise turnover.
  peak company P = Top_n(t_pk) minus c (the margin reference); entrants Ent = Top_n(t_out) minus Top_n(t_pk)
    (outcome-defined, labelled so).
  C1 (primary control): c' != c with |rank_c'(t_pk) - rank_c(t_pk)| <= 3, in Top_n at >= 2 checkpoints from step 1,000
    up to t_pk, and rank_c'(t_pk) <= rank_c'(previous checkpoint); its fate is not used. C2: C1 with final rank <= 20
    (stayers); C3: C1 with final rank 21-40; C4: C1 that leave Top_n inside W and are back by T.
  entries, of exit classes ('dropped') and of every final Top_n class ('persistent'): t_first = first checkpoint of the
    run of Top_n membership that contains t_pk (T); t_pre = last earlier checkpoint with rank > 40 (else step 0);
    W_in = (t_pre, t_first]; controls: c' != c with |rank_c'(t_pre) - rank_c(t_pre)| <= 3 (unconditional); P_pre =
    Top_n(t_pre) minus c. Entries whose Top_n run starts at the first checkpoint (step 0) are left-censored (W_in is
    empty): excluded and counted per direction and block (events.json 'entries_from_step0').

X1 (exits; exploratory). Margin M = R_c - mean_{s in P} R_s. Over the G16 intervals of W, on the measurement half,
divided by sd_n(t_pk), the change of M splits exactly into 20 components: for each write category X in own = {c}, P,
Ent, rel (visual relatives of c, DINOv2 rank <= 10, not in P or Ent; secondary: --params '{"relatives": "wordnet"}',
WordNet d <= 4), oth (the rest), the part on c, X_c =
sum_{k in X} T_k(c), and on the company, X_P = -mean_{s in P} sum_{k in X} T_k(s); and the same two parts of F_own
(decay of c's earlier writes), F_rest (the other groups' decay plus F_0), Phi, Pi and E. Each C1 class c' is
decomposed with categories relative to itself (own {c'}, company Top_n(t_pk) minus c', ...). Delta_X = X(exit) -
mean_{C1} X; excess-fall shares share_X = sum_events Delta_X / sum_events sum_X Delta_X (they sum to 1).
Standardization: each write category's margin term against 1,000 random sets of the same size from non-relatives of c
(c, P, Ent excluded) with total training count within 5% (z, mid-rank percentile); with a random split, the own term
against the random groups' terms scaled by n_c / n_r (own band, own_pct_R) and the sets against random-group sets
matched on count. Classification per event (first match, on excess shares, when sum_X Delta_X < 0): withdrawn (own share
<= 0.15, own term inside the band [random 5-95%; without a random split the C1 own terms' 25-75%, which needs
band_min = 3 C1 members, else 'few_controls'], F_own + P-self-raise shares >= 0.5); active own suppression (own
share >= 0.4, own term below the band's 5th percentile);
peak-company interference (share of P_c >= 0.4 and P_c below the 5th percentile of its size-matched sets on c);
drift-carried (Phi share >= 0.5); else unclassified. Flips are counted under the fp16 interval (four corners of the
bounds that keep the split exact: writes and F_own at their bounds, E and F_rest moved by the opposite, the exit and its
controls in opposite directions), the path order (cross term wholly to the weights or the input), the position
reference p*_t2, the half swap and the G+ refinement. A size-matched null that cannot be formed (fewer than 20 sets or
3 distinct classes, or no spread) is recorded per exit ('null_missing') and counted; 'most negative' is taken only over
exits with every applicable null. Primary population: anchored exits in stages 2-3, selected on the selection half
only; replicated anchored exits (and replicated turnover) are a sensitivity analysis. Validity gate: fewer than 50% of
anchored exits (above the sd floor) replicate -> noise-dominated.
X2 (entries). Categories fixed at t_pre (own, P_pre, rel, oth); components of the rise of R_c over W_in (normalized by
sd_n(t_first)); Delta against the rank-matched controls at t_pre; primary: own excess-rise share of dropped entries
minus that of persistent entries (reported next to the left-censored entries left out of each arm).
X3 (starvation). Bias parts: dCb^(k)[n] = Cb_t2 - rhob Cb_t1 per G16 interval, oriented per class at t1 on the selection
half (value by the sign of Bbar, gate by the sign of Abar), divided by the median |dCb| over classes in that unit and
interval; push = mean of the two maps, summed over the last 3 G16 intervals up to t_out. S1 (primary, anchored late
exits with m >= 3 C1 members, in every run): withdrawn (inside the C1 band), reversed (below), maintained (above); the
exit and each C1 member i are labelled against the same band, the 25-75% of the other m - 1 C1 members' pushes, and the
exit's label is the fraction over the m leave-one-out bands (the 25-75% of fewer values covers less, so a full-m band
for the exit alone would favour 'withdrawn'); primary: the exits' withdrawn fraction minus that of C1 (0 in
expectation when the exit and its C1 classes are exchangeable). With a random split, also an extra result against the
random band 5-95% (pushes of random groups scaled by n_c / n_r). S2: n's share of c's own-write capture in its stage;
timing of its halving after t_pk (before / same interval / after t_out) against a within-unit permutation of t_out
and against C1 classes. S3 (windows with >= 2 snapshots): log change of c's write rate in n split into the learning
rate, the second moment (crowding), c's momentum and the bias correction (bias exact; weights with the median D and
the projection on Hs).
X4 (first in line, per G16 interval and stage). Leader L_k = argmax_n z_n,k(t1) (selection half); capture O_n,k =
T_k(k) / sd_n(t1) (measurement half). F1: percentile of O_L among the stage's units (F1_bias: with the probe-free
capture, the oriented normalized bias push of X3). F2 (primary): within (k, interval), rank regression across the
stage's units of O on z, z^2 and the provenance of the on-ness at t1 (relatives', own, others' and init parts of the
class-mean gate and value pre-activations, in response units Abar g_S + Bbar v_S, centred by the unit's mean over
classes); the coefficient on the relatives' part. F2 is undefined at t1 = 0, where no class has written yet (the
provenance columns are 0), so the first G16 interval is excluded (a coefficient on a column without spread across the
scope's units is missing, not 0). F3: psi^beta = log(pi^O_L + 1e-4) - log(pi^beta_L + 1e-4), pi^X = X+ / sum over
the scope. Ladder (uniform; kappa; P = mean over unrelated classes j of |T_j(k)| / sd_n; beta; mu)
reported as a decomposition.
X5 (loop). Chains of consecutive equal-length intervals (G16 2,500-step; snapshot 2,000-step): rank regression (ranks
within each interval) of the own gain O in I_{j+1} on O in I_j, the other-derived gain (sum_{k != c} T_k(c) / sd) and
z(t_j); null: each (unit, class) row's own-gain series permuted across its chain. Cumulative concentration
H_k(t) = sum_n pi_n^2 of cumulative O+ next to that of kappa.
Uncertainty: two-way cluster bootstrap over unit and class (2,000 draws) for X1-X3, class-cluster bootstrap for X4,
permutation for X5; Holm over the five primaries of the H1 -> H2 direction. Bootstrap p-values are 2 min(#(draws <= 0)
+ 1, #(draws >= 0) + 1) / (B + 1) over the B non-empty draws (floor 2 / (B + 1), B stored with each result); X5's
permutation p is (count + 1) / (n_perm + 1) (floor 1 / (n_perm + 1)).

Inputs are never dropped silently: every G16 checkpoint (model_ and attr_ files) must exist; a run with a random split
(config rand_groups, or any attrR file) needs attrR at every G16 step (--no_random to run without it); history
snapshots need their full_{t}.pt (--no_full to drop them); class counts come from --class_counts, <run>/rand_groups.npy
or /dev/shm/imagenet256/labels.npy, else an error (equal counts only with --params '{"equal_counts": true}', toy
tests). The sources are written into the outputs (has_random, band_source, class_counts).

Stages (subcommands; each output file is one resumable unit, written to a temporary name and renamed):
  probe      the probe set (all validation images per class) as a uint8 .npy memmap
  integrity  I1 (logs), I2 (shared checkpoint and snapshot steps bitwise), I3 (remainders) -> integrity.json (always
             written, also without usable snapshots)
  stock      per grid point -> stock_{t}.pt, stockh_{t}.pt
  events     -> events.json (frozen; --force to overwrite)
  intervals  per interval (--family g16 | gplus | snap | all) -> xhw_{t1}_{t2}.pt
  gradstate  per grid point: kappa, beta (D from the nearest snapshot within 1,500 steps), mu at snapshots
  tests      X1-X5 -> exits_tests.json, with the checks, the momentum carry-over estimate at snapshot starts, rho per
             interval, the G+ path-dependence table and the 50 units with the most anchored exits
  compare    replication of the primaries (--other, E1r) against a reference run (--ref, E1) -> --out_json (default
             exits_compare.json in --out or next to --other); no verdict when the runs differ in the G16 steps, the X3
             band or the class counts (--allow_mismatch)

  python3 -m experiments.convnext_exits <stage> --run <run dir> --out <dir> [--history <snap dir>] [--full <full dir>]
"""

import argparse
import glob
import hashlib
import json
import math
import os
import time
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

from experiments.convnext_atlas import block_inputs
from experiments.convnext_train import CKPT_STEPS, parse_steps
from rcdl.attribution import to_fp16_scaled
from rcdl.convnext import build_convnext
from rcdl.imagenet import MEAN, STD
from rcdl.interval_split import (MAPS, ClassSums, column_terms, dsilu, group_total, half_ulp, image_split, increments,
                                 load_parts, lr_schedule, momentum_carryover, own_bounds, own_terms,
                                 split_completeness)

PHASES = ((0, 2500), (2500, 10000), (10000, 10 ** 9))
WRITE = ('own', 'P', 'Ent', 'rel', 'oth')
XCAT = WRITE + ('Fown', 'Frest', 'Phi', 'Pi', 'E')
COMP = [f'{x}_{s}' for x in XCAT for s in ('c', 'P')]
NWRITE = 2 * len(WRITE)                          # the write components come first in COMP
LABELS = ('withdrawn', 'active', 'interference', 'drift', 'unclassified', 'no_excess_fall', 'no_control',
          'few_controls')
UNLABELLED = ('no_control', 'no_excess_fall', 'few_controls')     # left out of the label proportions
S1 = ('withdrawn', 'reversed', 'maintained')
CORNERS = ('all+', 'all-', 'own+', 'own-')
DIRS = {'s0m1': (0, 1), 's1m0': (1, 0)}


def defaults():
  return dict(P=20, est_step=1000, best=10, gone=40, anchor_k=5, anchor_min=2, match=3, nodes=32, sd_floor=1e-3,
              n_rand=1000, n_boot=2000, n_perm=1000, count_tol=0.05, visual_near=10, wordnet_near=4, unrel_wn=10,
              unrel_vis=100, d_window=1500, d_min_step=1000, x5_min_len=1000, log_offset=1e-4, carriers=3, seed=0,
              rep_best=15, rep_gone=30, relatives='visual', band_min=3, equal_counts=False)


# --------------------------------------------------------------------------------------------- small statistics

def midrank(a: np.ndarray) -> np.ndarray:
  """Ascending mid-ranks (1..n, ties averaged) along the last axis."""
  a = np.asarray(a, float)
  flat = a.reshape(-1, a.shape[-1])
  out = np.empty_like(flat)
  for i, row in enumerate(flat):
    s = np.sort(row)
    out[i] = (np.searchsorted(s, row, 'left') + 1 + np.searchsorted(s, row, 'right')) / 2
  return out.reshape(a.shape)


def rank_desc(a: np.ndarray) -> np.ndarray:
  """Decreasing mid-ranks along the last axis (1 = largest)."""
  return a.shape[-1] + 1 - midrank(a)


def pct_of(x: float, null: np.ndarray) -> float:
  """Mid-rank percentile of x among null values: P(null < x) + P(null = x) / 2."""
  null = np.asarray(null, float)
  if not len(null):
    return float('nan')
  return float(((null < x).sum() + 0.5 * (null == x).sum()) / len(null))


def holm(p: dict) -> dict:
  """Holm-adjusted p-values of a {name: p} dict (nan entries are left out)."""
  items = sorted([(v, k) for k, v in p.items() if v is not None and not np.isnan(v)])
  m, out, run = len(items), {}, 0.0
  for i, (v, k) in enumerate(items):
    run = max(run, min(1.0, (m - i) * v))
    out[k] = run
  return out


def boot_p0(draws: np.ndarray, axis: int = 0):
  """Two-sided bootstrap p-value of 0 over the non-nan draws (an empty two-way draw gives nan and is not counted on
  either side): min(1, 2 min(#(<= 0) + 1, #(>= 0) + 1) / (B + 1)), never 0 (floor 2 / (B + 1)); nan where no draw is
  finite. Returns (p, B), B the number of non-nan draws."""
  draws = np.asarray(draws, float)
  ok = ~np.isnan(draws)
  B = ok.sum(axis)
  le = ((draws <= 0) & ok).sum(axis)
  ge = ((draws >= 0) & ok).sum(axis)
  p = np.minimum(1.0, 2 * np.minimum(le + 1, ge + 1) / (B + 1))
  return np.where(B > 0, p, np.nan), B


def twoway_weights(units: np.ndarray, classes: np.ndarray, n_boot: int, rng, chunk: int = 250):
  """Two-way (pigeonhole) cluster bootstrap: per draw, units and classes are resampled with replacement and each
  event gets the product of its unit's and its class's counts. Yields weight matrices (b, E) in chunks."""
  uu, ui = np.unique(units, return_inverse=True)
  cu, ci = np.unique(classes, return_inverse=True)
  done = 0
  while done < n_boot:
    b = min(chunk, n_boot - done)
    wu = rng.multinomial(len(uu), np.full(len(uu), 1 / len(uu)), size=b)
    wc = rng.multinomial(len(cu), np.full(len(cu), 1 / len(cu)), size=b)
    yield (wu[:, ui] * wc[:, ci]).astype(float)
    done += b


def boot_ratio(num: np.ndarray, den: np.ndarray, units, classes, n_boot, rng) -> dict:
  """Ratio of sums sum_e num[e] / sum_e den[e] (num may be (E, k)) with a two-way cluster bootstrap: estimate, 95% CI,
  the two-sided bootstrap p-value of 0 (boot_p0) and B, the fewest non-empty draws of any column."""
  num = np.asarray(num, float)
  num2 = num if num.ndim == 2 else num[:, None]
  if not len(den) or np.sum(den) == 0:
    return {'est': None, 'n': int(len(den))}
  est = num2.sum(0) / np.sum(den)
  draws = []
  for W in twoway_weights(units, classes, n_boot, rng):
    d = W @ den
    with np.errstate(invalid='ignore', divide='ignore'):
      draws.append((W @ num2) / d[:, None])
  draws = np.concatenate(draws)
  lo, hi = np.nanpercentile(draws, [2.5, 97.5], axis=0)
  p, B = boot_p0(draws)
  sq = lambda v: v.tolist() if num.ndim == 2 else float(v[0])
  return {'est': sq(est), 'ci': [sq(lo), sq(hi)], 'p0': sq(p), 'B': int(B.min()), 'n': int(len(den))}


def boot_mean(x: np.ndarray, units, classes, n_boot, rng) -> dict:
  """Mean of per-event values with a two-way cluster bootstrap (CI and two-sided p of 0)."""
  x = np.asarray(x, float)
  ok = ~np.isnan(x)
  if not ok.any():
    return {'est': None, 'n': 0}
  return boot_ratio(x[ok], np.ones(ok.sum()), np.asarray(units)[ok], np.asarray(classes)[ok], n_boot, rng)


def boot_median(x: np.ndarray, clusters: np.ndarray, n_boot: int, rng) -> dict:
  """Median of values with a cluster bootstrap over `clusters` (weighted medians of the resampled clusters): estimate,
  95% CI, two-sided p of 0."""
  x = np.asarray(x, float)
  ok = ~np.isnan(x)
  x, clusters = x[ok], np.asarray(clusters)[ok]
  if not len(x):
    return {'est': None, 'n': 0}
  order = np.argsort(x)
  xs = x[order]
  cu, ci = np.unique(clusters[order], return_inverse=True)
  draws = []
  for start in range(0, n_boot, 250):
    b = min(250, n_boot - start)
    w = rng.multinomial(len(cu), np.full(len(cu), 1 / len(cu)), size=b)[:, ci].astype(float)
    cw = np.cumsum(w, 1)
    idx = (cw < cw[:, -1:] / 2).sum(1)
    draws.append(xs[np.minimum(idx, len(xs) - 1)])
  draws = np.concatenate(draws)
  p, B = boot_p0(draws)
  return {'est': float(np.median(x)), 'ci': np.percentile(draws, [2.5, 97.5]).tolist(), 'p0': float(p), 'B': int(B),
          'n': int(len(x))}


def matched_sets(rng, pool: np.ndarray, counts: np.ndarray, size: int, target: float, n_sets: int, tol: float,
                 tries: int = 30) -> np.ndarray:
  """Up to n_sets random subsets of `pool` with `size` distinct members whose total count is within tol of target
  (rejection sampling). (m, size) int array."""
  pool = np.asarray(pool, np.int64)
  if size <= 0 or len(pool) < size:
    return np.zeros((0, max(size, 0)), np.int64)
  out, got = [], 0
  for _ in range(tries):
    n_try = max(2 * (n_sets - got), 256)
    if size * size > len(pool):
      idx = np.argsort(rng.random((n_try, len(pool))), 1)[:, :size]
    else:
      idx = rng.integers(0, len(pool), (n_try, size))
      if size > 1:
        s = np.sort(idx, 1)
        idx = idx[(np.diff(s, axis=1) > 0).all(1)]
    sets = pool[idx]
    ok = np.abs(counts[sets].sum(1) - target) <= tol * target
    out.append(sets[ok])
    got += int(ok.sum())
    if got >= n_sets:
      break
  return np.concatenate(out)[:n_sets]


def code_hash() -> str:
  h = hashlib.sha256()
  for f in (__file__, os.path.join(os.path.dirname(__file__), '..', 'rcdl', 'interval_split.py')):
    h.update(open(f, 'rb').read())
  return h.hexdigest()


def uid(b: str, n: int) -> int:
  """A deterministic integer id of unit n of block b (bootstrap clusters)."""
  i, j = block_sort_key(b)
  return (100 * i + j) * 100000 + int(n)


def phase_of(step: int) -> int:
  return next(i for i, (lo, hi) in enumerate(PHASES) if lo <= step < hi)


def block_sort_key(name: str):
  return [int(t) for t in name.replace('stage', '').split('.block')]


def sd_key(b: str, M: str, what: str = 'weight') -> str:
  i, j = b.replace('stage', '').split('.block')
  return f'stages.{i}.{j}.pw_{M}.{what}'


def jsonable(x):
  if isinstance(x, dict):
    return {str(k): jsonable(v) for k, v in x.items()}
  if isinstance(x, (list, tuple)):
    return [jsonable(v) for v in x]
  if isinstance(x, (np.integer,)):
    return int(x)
  if isinstance(x, (np.floating, float)):
    return None if np.isnan(x) else float(x)
  if isinstance(x, np.ndarray):
    return jsonable(x.tolist())
  if torch.is_tensor(x):
    return jsonable(x.tolist())
  return x


def save_atomic(obj, path: str):
  """torch.save to a temporary name, then a rename: a kill during the write (sandboxes end without notice) never leaves
  a truncated file under the final name, which a resumed stage would skip as done."""
  torch.save(obj, path + '.tmp')
  os.replace(path + '.tmp', path)


def dump_json(obj, path: str, **kw):
  """json.dump through a temporary file and a rename (as save_atomic)."""
  with open(path + '.tmp', 'w') as f:
    json.dump(obj, f, **kw)
  os.replace(path + '.tmp', path)


def array_sha(v: np.ndarray) -> str:
  return hashlib.sha256(np.ascontiguousarray(v, np.float64).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------------------------ run data

def expected_grid(cfg: dict) -> list:
  """The G16 grid a run's config promises: the trainer's fixed checkpoints (CKPT_STEPS) up to its last step, and that
  step, limited to its --attr_steps when it wrote attribution files only there. Extra checkpoints (--extra_ckpts, also
  in cfg['ckpt_steps']) are not G16."""
  last = int(cfg.get('steps', CKPT_STEPS[-1]))
  steps = {s for s in CKPT_STEPS if s <= last} | {last}
  if cfg.get('attr_steps'):
    steps &= parse_steps(cfg['attr_steps'])
  return sorted(steps)


class Run:
  """A run's files: model_{s}.pt, attr_{s}.pt, attrR_{s}.pt (random split), attr_init.pt, config.json, log.json, and
  optionally its history: snap_{s}.pt (model) in `history` and full_{s}.pt (fp32 parts and moments) in `full`.
  The class split at a checkpoint is read from the compact file, elsewhere from the full state. Nothing is dropped
  silently: a G16 step (`grid`, else expected_grid) without its model_ or attr_ file, and, when the run has a random
  split (config rand_groups or any attrR file; no_random turns it off on purpose), a G16 step without attrR, raise
  SystemExit. snaps_no_full: history steps without a full state (main refuses them unless --no_full)."""

  def __init__(self, run: str, history: str = None, full: str = None, grid=None, snaps=None, device='cpu',
               no_random: bool = False):
    self.dir, self.history, self.full_dir, self.dev = run, history, full, torch.device(device)
    self.cfg = json.load(open(os.path.join(run, 'config.json')))
    steps = sorted(set(grid)) if grid else expected_grid(self.cfg)
    ex = lambda p: os.path.exists(os.path.join(run, p))
    miss = [f for s in steps for f in (f'model_{s:05d}.pt', f'attr_{s:05d}.pt') if not ex(f)]
    if miss:
      raise SystemExit(f'{run}: G16 checkpoint files missing (the grid is never shrunk silently; --grid chooses the '
                       f'steps): {miss}')
    self.g16 = steps
    hist = sorted(int(os.path.basename(f)[5:10]) for f in glob.glob(os.path.join(history, 'snap_[0-9]*.pt'))) \
        if history else []
    self.snaps_no_full = [s for s in hist if not (full and os.path.exists(os.path.join(full, f'full_{s:05d}.pt')))]
    self.snaps_avail = [s for s in hist if s not in self.snaps_no_full]
    self.snaps = sorted(self.snaps_avail if snaps is None else [s for s in snaps if s in self.snaps_avail])
    self.gplus = sorted(set(self.g16) | set(self.snaps))
    self.init = torch.load(os.path.join(run, 'attr_init.pt'), map_location='cpu', weights_only=False)
    self.blocks = sorted({k.rsplit('.', 1)[0] for k in self.init if k.endswith('.pw_gate')}, key=block_sort_key)
    want = not no_random and (bool(self.cfg.get('rand_groups')) or bool(glob.glob(os.path.join(run, 'attrR_*.pt'))))
    if want:
      missR = [f'attrR_{s:05d}.pt' for s in self.g16 if not ex(f'attrR_{s:05d}.pt')]
      if missR:
        raise SystemExit(f'{run}: the run has a random split (config rand_groups or attrR files) but attrR files are '
                         f'missing: {missR} (pass --no_random to run without the random-group nulls)')
    self.has_random = want and bool(self.g16)
    self._c = {}
    sd = self.model_sd(self.g16[-1])
    self.n_classes = int(sd['head.weight'].shape[0])
    self.U = {b: int(sd[sd_key(b, 'gate')].shape[0]) for b in self.blocks}
    self.d = {b: int(sd[sd_key(b, 'gate')].shape[1]) for b in self.blocks}

  def _cached(self, key, fn, keep=4):
    if key not in self._c:
      kind = key[0]
      same = [k for k in self._c if k[0] == kind]
      for k in same[:max(0, len(same) - keep + 1)]:
        del self._c[k]
      self._c[key] = fn()
    return self._c[key]

  def is_ckpt(self, s: int) -> bool:
    return s in self.g16

  def model_sd(self, s: int) -> dict:
    def get():
      p = os.path.join(self.dir, f'model_{s:05d}.pt')
      if os.path.exists(p):
        return torch.load(p, map_location='cpu', weights_only=False)
      return torch.load(os.path.join(self.history, f'snap_{s:05d}.pt'), map_location='cpu', weights_only=False)['model']
    return self._cached(('model', s), get)

  def attr(self, s: int, split: str = 'class') -> dict:
    def get():
      if split == 'random':
        return torch.load(os.path.join(self.dir, f'attrR_{s:05d}.pt'), map_location='cpu', weights_only=False)
      p = os.path.join(self.dir, f'attr_{s:05d}.pt')
      if os.path.exists(p):
        return torch.load(p, map_location='cpu', weights_only=False)
      return self.full(s)['attr']
    return self._cached(('attr', s, split), get, keep=256)          # all grid points: every block reads them all

  def full(self, s: int) -> dict:
    return self._cached(('full', s), lambda: torch.load(os.path.join(self.full_dir, f'full_{s:05d}.pt'),
                                                         map_location='cpu', weights_only=False, mmap=True), keep=64)

  def parts(self, s: int, b: str, split: str = 'class', bounds: bool = True) -> dict:
    sd, A = self.model_sd(s), self.attr(s, split)
    return {M: load_parts(A[f'{b}.pw_{M}'], self.init[f'{b}.pw_{M}'], W=sd[sd_key(b, M)], b=sd[sd_key(b, M, 'bias')],
                          device=self.dev, bounds=bounds) for M in MAPS}

  def inc(self, s1: int, s2: int, b: str, split: str = 'class', bounds: bool = True) -> dict:
    p1, p2 = self.parts(s1, b, split, bounds), self.parts(s2, b, split, bounds)
    return {M: increments(p1[M], p2[M]) for M in MAPS}

  def weights(self, s: int, b: str) -> dict:
    sd = self.model_sd(s)
    return {M: (sd[sd_key(b, M)].to(self.dev, torch.float64), sd[sd_key(b, M, 'bias')].to(self.dev, torch.float64))
            for M in MAPS}

  def model(self, s: int):
    sd = self.model_sd(s)
    m = build_convnext(self.cfg['size'], n_classes=self.n_classes)
    m.load_state_dict(sd)
    return m.to(self.dev).eval()


# ---------------------------------------------------------------------------------------------------------- probe

def build_probe(val_dir: str, out_dir: str, per_class: int = 50, n_classes: int = 1000):
  """out_dir/x.npy (N, 3, 224, 224) uint8 (written as a memmap), y.npy, meta.json: the first per_class validation
  images of every class in validation-file order (all 50 by default), sorted by class; geometry as
  experiments.convnext_atlas.probe_set (shorter side 256, antialiased bilinear, center crop 224)."""
  import pyarrow.parquet as pq
  from torchvision.io import ImageReadMode, decode_image
  os.makedirs(out_dir, exist_ok=True)
  N = n_classes * per_class
  x = np.lib.format.open_memmap(os.path.join(out_dir, 'x.npy'), mode='w+', dtype=np.uint8, shape=(N, 3, 224, 224))
  count = np.zeros(n_classes, int)
  for f in sorted(glob.glob(os.path.join(val_dir, '**', 'validation-*.parquet'), recursive=True)):
    t = pq.read_table(f)
    for d, lab in zip(t.column('image').to_pylist(), t.column('label').to_pylist()):
      if count[lab] >= per_class:
        continue
      im = decode_image(torch.frombuffer(bytearray(d['bytes']), dtype=torch.uint8), mode=ImageReadMode.RGB)
      h, w = im.shape[-2:]
      s = 256 / min(h, w)
      im = F.interpolate(im[None].float(), size=(max(256, round(h * s)), max(256, round(w * s))), mode='bilinear',
                         antialias=True, align_corners=False)[0]
      H, W = im.shape[-2:]
      top, left = (H - 224) // 2, (W - 224) // 2
      crop = im[:, top:top + 224, left:left + 224].round().clamp(0, 255).to(torch.uint8)
      x[lab * per_class + count[lab]] = crop.numpy()
      count[lab] += 1
  assert (count == per_class).all(), f'classes with fewer than {per_class} images: {np.flatnonzero(count < per_class)}'
  x.flush()
  np.save(os.path.join(out_dir, 'y.npy'), np.repeat(np.arange(n_classes), per_class))
  json.dump({'N': N, 'per_class': per_class, 'n_classes': n_classes}, open(os.path.join(out_dir, 'meta.json'), 'w'))


class Probe:
  """x (N, 3, S, S) uint8 (a memmap or tensor), y (N,) sorted by class with `per` images each; half[i] = 0 for the
  first per // 2 images of a class (H1), 1 otherwise (H2); gidx = 2 y + half."""

  def __init__(self, path: str):
    if path.endswith('.pt'):
      d = torch.load(path, map_location='cpu', weights_only=False)
      self.x, y = d['x'], torch.as_tensor(d['y'])
    else:
      self.x = np.load(os.path.join(path, 'x.npy'), mmap_mode='r')
      y = torch.from_numpy(np.load(os.path.join(path, 'y.npy')))
    self.y = y.long()
    self.per = int((self.y == self.y[0]).sum())
    self.C = int(self.y.max()) + 1
    assert torch.equal(self.y, torch.arange(self.C).repeat_interleave(self.per)), 'probe: sorted, equal count per class'
    assert self.per >= 2, 'probe: two halves need at least 2 images per class'
    self.half = ((torch.arange(len(self.y)) % self.per) >= self.per // 2).long()
    self.gidx = 2 * self.y + self.half

  def batches(self, batch: int, dev):
    mean, std = MEAN.to(dev)[:, None, None], STD.to(dev)[:, None, None]
    for i in range(0, len(self.y), batch):
      xb = self.x[i:i + batch]
      xb = torch.as_tensor(np.asarray(xb)) if not torch.is_tensor(xb) else xb
      yield (xb.to(dev).float() - mean) / std, self.y[i:i + batch].to(dev), self.gidx[i:i + batch].to(dev)


def to_half_major(t: torch.Tensor, C: int) -> torch.Tensor:
  """(2C, U, ...) indexed by 2 c + half -> (2, U, C, ...)."""
  t = t.reshape(C, 2, *t.shape[1:])
  return t.permute(1, 2, 0, *range(3, t.dim())).contiguous()


def take(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
  """t (B, P, X) at positions p (B, U) -> (B, U, X)."""
  return torch.gather(t, 1, p[:, :, None].expand(-1, -1, t.shape[-1]))


def take_u(t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
  """t (B, P, U) at the unit's own positions p (B, U) -> (B, U)."""
  return torch.gather(t, 1, p[:, None, :])[:, 0]


def resp64(h: torch.Tensor, w: dict) -> dict:
  """fp64 pre-activations and response of every unit at its own input h (B, U, d): w = {M: (W (U, d), b (U,))}."""
  z = {M: (h * w[M][0]).sum(-1) + w[M][1] for M in MAPS}
  return dict(g=z['gate'], v=z['value'], u=F.silu(z['gate']) * z['value'])


# ------------------------------------------------------------------------------------------------------ integrity

def cmd_integrity(a, run: Run):
  """I1: each snapshot's training log equals log.json on all shared steps (val_acc and sec). I2: at steps that are both
  a checkpoint and a snapshot, the model weights are identical, to_fp16_scaled of the fp32 C and Cb equals the stored
  fp16 tensors bitwise (same scale), and a, ab, t are equal. I3: the remainder max |W - a W0 - sum_k C_k| / max |W| of
  every snapshot (recorded; it is never assumed to be 0). Snapshots failing I1 or I2 are not usable. integrity.json is
  always written (also without snapshots), with the snapshots checked and those dropped for want of a full state."""
  lp = os.path.join(run.dir, 'log.json')
  log = {r['step']: r for r in json.load(open(lp))} if os.path.exists(lp) else {}
  out = {'snapshots': {}, 'usable': [], 'checked': list(run.snaps), 'dropped_no_full': list(run.snaps_no_full),
         'history': run.history, 'full': run.full_dir}
  hist = os.path.join(run.history, 'history.json') if run.history else ''
  index = {r['step']: r for r in json.load(open(hist))} if hist and os.path.exists(hist) else {}
  for s in run.snaps:
    snap = torch.load(os.path.join(run.history, f'snap_{s:05d}.pt'), map_location='cpu', weights_only=False)
    full = torch.load(os.path.join(run.full_dir, f'full_{s:05d}.pt'), map_location='cpu', weights_only=False, mmap=True)
    rec = {'source': snap.get('source'), 'history': index.get(s)}
    shared = [r for r in snap.get('log', []) if r['step'] in log]
    bad = [r['step'] for r in shared if r.get('val_acc') != log[r['step']].get('val_acc')
           or r.get('sec') != log[r['step']].get('sec')]
    rec['I1'] = {'ok': not bad, 'shared': len(shared), 'mismatch_steps': bad}
    if os.path.exists(os.path.join(run.dir, f'model_{s:05d}.pt')):
      msd = torch.load(os.path.join(run.dir, f'model_{s:05d}.pt'), map_location='cpu', weights_only=False)
      cmp = torch.load(os.path.join(run.dir, f'attr_{s:05d}.pt'), map_location='cpu', weights_only=False)
      model_ok = set(msd) == set(snap['model']) and all(torch.equal(msd[k], snap['model'][k]) for k in msd)
      parts_ok, scal_ok = True, True
      for key, fr in full['attr'].items():
        c = cmp[key]
        for name in ('C', 'Cb'):
          q, sc = to_fp16_scaled(fr[name].float())
          parts_ok &= bool(sc == c[f'{name}_scale'] and torch.equal(q, c[name]))
        scal_ok &= bool(fr['a'] == c['a'] and fr['ab'] == c['ab'] and int(fr['t']) == int(c['t']))
      rec['I2'] = {'ok': bool(model_ok and parts_ok and scal_ok), 'model': bool(model_ok), 'parts': bool(parts_ok),
                   'scalars': bool(scal_ok)}
    worst = 0.0
    for key, fr in full['attr'].items():
      b, M = key.rsplit('.', 1)
      if not M.startswith('pw_'):
        continue
      W = snap['model'][sd_key(b, M[3:])].double()
      R = W - fr['a'] * fr['W0'].double() - fr['C'].double().sum(0)
      worst = max(worst, float(R.abs().max() / W.abs().max()))
    rec['I3_remainder_rel'] = worst
    rec['usable'] = rec['I1']['ok'] and rec.get('I2', {'ok': True})['ok']
    if rec['usable']:
      out['usable'].append(s)
    out['snapshots'][s] = rec
    print(json.dumps({'step': s, 'I1': rec['I1']['ok'], 'I2': rec.get('I2'), 'I3': worst}), flush=True)
  dump_json(jsonable(out), os.path.join(a.out, 'integrity.json'), indent=1)


def usable_snaps(a, run_dir, history, full, avail=None):
  """Snapshot steps that passed integrity.json (all available ones with --skip_integrity). avail: the snapshot steps
  with a full state found now; an integrity.json that did not check all of them is refused."""
  p = os.path.join(a.out, 'integrity.json')
  if os.path.exists(p):
    d = json.load(open(p))
    new = sorted(set(avail or []) - set(d.get('checked', avail or [])))
    if new:
      raise SystemExit(f'integrity.json did not check the snapshots {new}: rerun the integrity stage')
    return d['usable']
  if history and full and not a.skip_integrity:
    found = 'unknown' if avail is None else len(avail)
    raise SystemExit(f'integrity.json missing ({found} snapshot / full-state pairs found in {history} and {full}): run '
                     'the integrity stage first (or pass --skip_integrity)')
  return None


def require_full(a, run: Run):
  """History snapshots need their full_{t}.pt: without it G+ (sub intervals, the G+ flips, path dependence), the
  snapshot intervals and X5's snapshot chains, mu, S3, the carry-over and the preconditioner D (kappa and beta would go
  unpreconditioned) are lost. A snapshot without one is an error unless --no_full (then the dropped steps are
  printed and recorded)."""
  if not run.snaps_no_full:
    return
  names = [f'full_{s:05d}.pt' for s in run.snaps_no_full]
  if not getattr(a, 'no_full', False):
    raise SystemExit(f'history snapshots without a full state in --full ({run.full_dir}): {names} (pass --full <dir>, '
                     'or --no_full to drop these snapshots)')
  print(f'WARNING: --no_full: history snapshots dropped (no full state): {run.snaps_no_full}', flush=True)


# ---------------------------------------------------------------------------------------------------------- stock

@torch.no_grad()
def stock_point(run: Run, s: int, probe: Probe, batch: int):
  """One grid point: per block (2, U, C) R (fp64), Gm, Vm, Abar, Bbar and (2, U, C, d) Hs; zcheck = max |fp64
  pre-activation - model fp32 pre-activation| / max |pre-activation| at p*."""
  model = run.model(s)
  named = dict(model.blocks())
  G = 2 * probe.C
  w = {b: run.weights(s, b) for b in run.blocks}
  z = lambda *shape: torch.zeros(*shape, dtype=torch.float64, device=run.dev)
  acc = {b: {k: z(G, run.U[b]) for k in ('R', 'Gm', 'Vm', 'Abar', 'Bbar')} | {'Hs': z(G, run.U[b], run.d[b])}
         for b in run.blocks}
  cnt = torch.zeros(G, dtype=torch.float64, device=run.dev)
  zc = {b: 0.0 for b in run.blocks}
  for xb, yb, gb in probe.batches(batch, run.dev):
    hs = block_inputs(model, xb, set(run.blocks))
    cnt.index_add_(0, gb, torch.ones(len(gb), dtype=torch.float64, device=run.dev))
    for b in run.blocks:
      blk = named[b]
      h = hs[b].flatten(1, 2).float()
      g32 = blk.pw_gate(h)
      p = (F.silu(g32) * blk.pw_value(h)).argmax(1)
      hst = take(h, p).double()
      r = resp64(hst, w[b])
      zc[b] = max(zc[b], float((r['g'] - take_u(g32, p).double()).abs().max() / r['g'].abs().max().clamp(min=1e-30)))
      vals = {'R': r['u'], 'Gm': r['g'], 'Vm': r['v'], 'Abar': dsilu(r['g']) * r['v'], 'Bbar': F.silu(r['g']),
              'Hs': hst}
      for k, v in vals.items():
        acc[b][k].index_add_(0, gb, v)
  out = {'step': s, 'blocks': {}}
  hsout = {'step': s, 'blocks': {}}
  for b in run.blocks:
    m = {k: v / cnt.view(-1, *([1] * (v.dim() - 1))) for k, v in acc[b].items()}
    rec = {k: to_half_major(m[k], probe.C).cpu() for k in ('R', 'Gm', 'Vm', 'Abar', 'Bbar')}
    for k in ('Gm', 'Vm', 'Abar', 'Bbar'):
      rec[k] = rec[k].float()
    rec['zcheck'] = zc[b]
    out['blocks'][b] = rec
    hsout['blocks'][b] = to_half_major(m['Hs'], probe.C).float().cpu()
  return out, hsout


def cmd_stock(a, run: Run, probe: Probe):
  steps = run.gplus if a.steps == '' else [int(t) for t in a.steps.split(',')]
  for s in steps:
    path = os.path.join(a.out, f'stock_{s:05d}.pt')
    if os.path.exists(path):
      continue
    t0 = time.time()
    rec, hs = stock_point(run, s, probe, a.batch)
    save_atomic(hs, os.path.join(a.out, f'stockh_{s:05d}.pt'))     # stockh first: a stock file means both are complete
    save_atomic(rec, path)
    print(json.dumps({'step': s, 'zcheck': max(r['zcheck'] for r in rec['blocks'].values()),
                      'sec': round(time.time() - t0, 1)}), flush=True)


def load_stock(out: str, s: int) -> dict:
  return torch.load(os.path.join(out, f'stock_{s:05d}.pt'), map_location='cpu', weights_only=False)['blocks']


def load_stockh(out: str, s: int) -> dict:
  path = os.path.join(out, f'stockh_{s:05d}.pt')
  return torch.load(path, map_location='cpu', weights_only=False, mmap=True)['blocks']


# --------------------------------------------------------------------------------------------------------- events

def low_sd(R: dict, sd_floor: float) -> dict:
  """{block: bool (T, ..., U)}: units whose sd over classes (R[b] (T, ..., U, C)) is below sd_floor times the median
  over every block's units, per grid point (and half), as Ctx.floor."""
  sd = {b: v.std(-1) for b, v in R.items()}
  med = np.median(np.concatenate(list(sd.values()), -1), -1)
  return {b: v < sd_floor * med[..., None] for b, v in sd.items()}


def find_events(Rs: np.ndarray, Rm: np.ndarray, steps, prm: dict, low: np.ndarray = None):
  """Events of one block on one selection half. Rs, Rm: (T, U, C) class responses on the selection and measurement
  halves at the G16 checkpoints `steps`; low (T, U): units below the sd floor on the selection half (low_sd), whose
  exits (at t_pk) and entries (at t_first) are dropped. Returns (exits, persistent entries, counts of the dropped:
  low_sd_exits, low_sd_entries and entries_from_step0, the last two per arm); indices are positions in `steps`. An
  exit whose own entry is dropped keeps 'entry': None."""
  T, U, C = Rs.shape
  P, gone, match = prm['P'], prm['gone'], prm['match']
  steps = np.asarray(steps)
  est = steps >= prm['est_step']

  def exit_mask(R, best, out_rank):
    rank = rank_desc(R)
    top = rank <= P
    cnt = top[est].sum(0)
    minr = np.where(est[:, None, None], rank, np.inf).min(0)
    return rank, top, (cnt >= 2) & (minr <= best) & (rank[-1] > out_rank)

  rank, top, ok = exit_mask(Rs, prm['best'], gone)
  okm = exit_mask(Rm, prm['best'], gone)[2]
  okm_relaxed = exit_mask(Rm, prm['rep_best'], prm['rep_gone'])[2]

  info = {'low_sd_exits': 0, 'low_sd_entries': {'dropped': 0, 'persistent': 0},
          'entries_from_step0': {'dropped': 0, 'persistent': 0}}

  def entry(n, c, t_in, arm):
    t_first = t_in
    while t_first > 0 and top[t_first - 1, n, c]:
      t_first -= 1
    if t_first == 0:                                 # left-censored: in Top_n from the first checkpoint on, no W_in
      info['entries_from_step0'][arm] += 1
      return None
    if low is not None and low[t_first, n]:
      info['low_sd_entries'][arm] += 1
      return None
    pre = [t for t in range(t_first) if rank[t, n, c] > gone]
    t_pre = pre[-1] if pre else 0
    ctrl = [int(k) for k in np.flatnonzero(np.abs(rank[t_pre, n] - rank[t_pre, n, c]) <= match) if k != c]
    return {'pre': int(t_pre), 'first': int(t_first), 'ctrl': ctrl,
            'Tpre': [int(k) for k in np.flatnonzero(top[t_pre, n])]}

  exits, persistent = [], []
  for n in range(U):
    core = np.argsort(-Rs[-1, n], kind='stable')[:prm['anchor_k']]
    for c in np.flatnonzero(ok[n]):
      r = rank[:, n, c]
      minr = r[est].min()
      pk = int(max(t for t in range(T) if est[t] and r[t] == minr))
      if low is not None and low[pk, n]:              # a near-constant unit: its ranks are noise
        info['low_sd_exits'] += 1
        continue
      last = int(max(t for t in range(T) if top[t, n, c]))
      out = int(min(t for t in range(last + 1, T) if r[t] > gone))
      Ppk = set(np.flatnonzero(top[pk, n]).tolist())
      P_ = sorted(Ppk - {int(c)})
      Ent = sorted(set(np.flatnonzero(top[out, n]).tolist()) - Ppk)
      upto = est & (steps <= steps[pk])
      C1 = []
      if pk > 0:
        for k in np.flatnonzero(np.abs(rank[pk, n] - r[pk]) <= match):
          if k != c and top[upto, n, k].sum() >= 2 and rank[pk, n, k] <= rank[pk - 1, n, k]:
            C1.append(int(k))
      fin = rank[-1, n]
      exits.append({
        'n': int(n), 'c': int(c), 'pk': pk, 'last': last, 'out': out, 'best': float(minr),
        'anchored': bool(np.isin(core, list(Ppk)).sum() >= prm['anchor_min']), 'rep': bool(okm[n, c]),
        'rep_relaxed': bool(okm_relaxed[n, c]),
        'P': P_, 'Ent': Ent, 'C1': C1, 'C2': [k for k in C1 if fin[k] <= P],
        'C3': [k for k in C1 if P < fin[k] <= gone],
        'C4': [k for k in C1 if fin[k] <= P and (rank[pk + 1:out + 1, n, k] > P).any()],
        'entry': entry(n, int(c), pk, 'dropped')})
    for c in np.flatnonzero(top[-1, n]):
      e = entry(n, int(c), T - 1, 'persistent')
      if e is not None:
        persistent.append({'n': int(n), 'c': int(c), **e})
  return exits, persistent, info


def sum_info(infos: list) -> dict:
  """Sum of find_events' dropped-event counts."""
  return {'low_sd_exits': sum(i.get('low_sd_exits', 0) for i in infos),
          **{k: {arm: sum(i.get(k, {}).get(arm, 0) for i in infos) for arm in ('dropped', 'persistent')}
             for k in ('low_sd_entries', 'entries_from_step0')}}


def cmd_events(a, run: Run):
  path = os.path.join(a.out, 'events.json')
  if os.path.exists(path) and not a.force:
    raise SystemExit('events.json exists and is frozen (pass --force to rebuild before any decomposition)')
  prm = a.prm
  stocks = {s: load_stock(a.out, s) for s in run.g16}
  res = {'version': 3, 'code_hash': code_hash(), 'created': time.strftime('%Y-%m-%d %H:%M:%S'), 'run': run.dir,
         'steps': run.g16, 'params': prm, 'halves': {'H1': 'images 1..per/2 of each class', 'H2': 'the rest'},
         'has_random': run.has_random,
         'directions': {d: {'sel': s, 'meas': m, 'exits': [], 'persistent': []} for d, (s, m) in DIRS.items()},
         'pairs': {}, 'counts': {}, 'block_counts': {}}
  Rb = {b: np.stack([stocks[s][b]['R'].numpy() for s in run.g16]) for b in run.blocks}     # (T, 2, U, C)
  low = low_sd(Rb, prm['sd_floor'])                                                          # (T, 2, U)
  for b in run.blocks:
    R = Rb[b]
    pairs = {}
    for d, (s, m) in DIRS.items():
      ex, pe, info = find_events(R[:, s], R[:, m], run.g16, prm, low[b][:, s])
      res['block_counts'].setdefault(b, {})[d] = info
      for e in ex:
        e['b'] = b
        pairs.setdefault(e['n'], set()).update([e['c']] + e['P'] + e['C1'])
      for e in pe:
        e['b'] = b
      res['directions'][d]['exits'] += ex
      res['directions'][d]['persistent'] += pe
    res['pairs'][b] = {int(n): sorted(v) for n, v in pairs.items()}
  for d, rec in res['directions'].items():
    ex = rec['exits']
    res['counts'][d] = {'exits': len(ex), 'anchored': sum(e['anchored'] for e in ex),
                        'replicated': sum(e['rep'] for e in ex),
                        'anchored_replicated': sum(e['anchored'] and e['rep'] for e in ex),
                        'with_C1': sum(bool(e['C1']) for e in ex), 'with_C2': sum(bool(e['C2']) for e in ex),
                        'persistent_entries': len(rec['persistent']),
                        **sum_info([res['block_counts'][b][d] for b in run.blocks])}
  atl = [os.path.join(run.dir, f'atlas_{s:05d}.pt') for s in run.g16]
  if all(os.path.exists(f) for f in atl):                    # context: exits on the old 8-image atlas probe
    A = [torch.load(f, map_location='cpu', weights_only=False) for f in atl]
    R8 = {b: np.stack([at[b]['R'].numpy() for at in A]) for b in run.blocks}
    low8 = low_sd(R8, prm['sd_floor'])
    n8 = an8 = 0
    for b in run.blocks:
      ex, _, _ = find_events(R8[b], R8[b], run.g16, prm, low8[b])
      n8 += len(ex)
      an8 += sum(e['anchored'] for e in ex)
    res['counts']['atlas8'] = {'exits': n8, 'anchored': an8}
  dump_json(jsonable(res), path)
  print(json.dumps(res['counts']), flush=True)


# ------------------------------------------------------------------------------------------------------ intervals

def remainder_checks(parts: list) -> dict:
  """I3 at the grid points of an interval, for the dicts {map: Parts} in `parts`: R_rel = max |R| / max |W| (R = W -
  a W0 - sum_k C_k in fp64; nonzero from the fp16 storage of the parts and the fp32 sums of snapshot parts) and
  R_stored_rel = max |R - stored fp32 R| / max |W| (compact files only). The latter is about 1e-7 when a checkpoint's
  model file and attribution file hold the same weights; files from different trajectories would otherwise be absorbed
  silently by R, and so by E."""
  out = {'R_rel': 0.0, 'R_stored_rel': 0.0}
  for P in parts:
    for p in P.values():
      scale = max(float(p.W.abs().max()), 1e-300)
      out['R_rel'] = max(out['R_rel'], float(p.R.abs().max()) / scale)
      if p.r_stored is not None:
        out['R_stored_rel'] = max(out['R_stored_rel'], p.r_stored / scale)
  return out


def file_sha(path: str) -> str:
  return hashlib.sha256(open(path, 'rb').read()).hexdigest() if os.path.exists(path) else ''


def interval_plan(run: Run, family: str):
  """[(t1, t2, role, j)]: 'g16' intervals (j = index of t2 in G16), 'sub' intervals of G+ inside a G16 interval
  (j = its parent), 'snap' intervals between consecutive usable snapshots (j = -1)."""
  g = run.g16
  out = [(g[j - 1], g[j], 'g16', j) for j in range(1, len(g))]
  if family in ('gplus', 'all'):
    for j in range(1, len(g)):
      inner = [s for s in run.snaps if g[j - 1] < s < g[j]]
      if inner:
        pts = [g[j - 1]] + inner + [g[j]]
        out += [(pts[i], pts[i + 1], 'sub', j) for i in range(len(pts) - 1)]
  if family in ('snap', 'all'):
    s = run.snaps
    have = {(x[0], x[1]) for x in out}
    out += [(s[i], s[i + 1], 'snap', -1) for i in range(len(s) - 1) if (s[i], s[i + 1]) not in have]
  if family == 'gplus':
    out = [x for x in out if x[2] == 'sub']
  elif family == 'snap':
    out = [x for x in out if x[2] == 'snap']
  return out


@torch.no_grad()
def interval_file(run: Run, s1: int, s2: int, probe: Probe, pairs: dict, batch: int, nodes: int) -> dict:
  """The class means of one interval for every block (see the module docstring), with the checks: completeness of
  the class-mean split (fp64 and after fp32 storage, both references), the per-image identity, the pre-activation
  identity against the model's fp32 pre-activations, and the class split's total against the random split's."""
  m1, m2 = run.model(s1), run.model(s2)
  n1, n2 = dict(m1.blocks()), dict(m2.blocks())
  G, C = 2 * probe.C, probe.C
  wts = {}
  for b in run.blocks:
    w1, w2 = run.weights(s1, b), run.weights(s2, b)
    wts[b] = {M: SimpleNamespace(W1=w1[M][0], b1=w1[M][1], W2=w2[M][0], b2=w2[M][1], dW=w2[M][0] - w1[M][0],
                                 db=w2[M][1] - w1[M][1]) for M in MAPS}
  acc1 = {b: ClassSums(G, run.U[b], run.d[b], device=run.dev, cross=True) for b in run.blocks}
  acc2 = {b: ClassSums(G, run.U[b], run.d[b], device=run.dev, cross=False) for b in run.blocks}
  zc = {b: 0.0 for b in run.blocks}
  for xb, yb, gb in probe.batches(batch, run.dev):
    hs1, hs2 = block_inputs(m1, xb, set(run.blocks)), block_inputs(m2, xb, set(run.blocks))
    for b in run.blocks:
      h1a, h2a = hs1[b].flatten(1, 2).float(), hs2[b].flatten(1, 2).float()
      g1a = n1[b].pw_gate(h1a)
      p1 = (F.silu(g1a) * n1[b].pw_value(h1a)).argmax(1)
      p2 = (F.silu(n2[b].pw_gate(h2a)) * n2[b].pw_value(h2a)).argmax(1)
      h11, h21, h12, h22 = (take(h, p).double() for h, p in ((h1a, p1), (h2a, p1), (h1a, p2), (h2a, p2)))
      inc = wts[b]
      u1b = resp64(h11, {M: (inc[M].W1, inc[M].b1) for M in MAPS})
      u2b = resp64(h22, {M: (inc[M].W2, inc[M].b2) for M in MAPS})['u']
      zdiff = (u1b['g'] - take_u(g1a, p1).double()).abs().max()
      zc[b] = max(zc[b], float(zdiff / u1b['g'].abs().max().clamp(min=1e-30)))
      r1 = image_split(h11, h21, inc, u1b['u'], u2b, nodes)
      acc1[b].add(gb, r1, u1b['u'], u2b)
      r2 = image_split(h12, h22, inc, u1b['u'], u2b, nodes)
      acc2[b].add(gb, r2, u1b['u'], u2b)
  out = {'s1': s1, 's2': s2, 'nodes': nodes, 'blocks': {},
         'flags': {'short_fp16_late': bool(s2 - s1 <= 1000 and s1 >= 10000 and (run.is_ckpt(s1) or run.is_ckpt(s2)))}}
  for b in run.blocks:
    c1 = {k: (to_half_major(v, C) if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in acc1[b].means().items()}
    c2 = {k: (to_half_major(v, C) if torch.is_tensor(v) and v.dim() >= 2 else v) for k, v in acc2[b].means().items()}
    P1, P2 = run.parts(s1, b, 'class', bounds=False), run.parts(s2, b, 'class', bounds=False)
    inc = {M: increments(P1[M], P2[M]) for M in MAPS}
    chk = {'image_resid': max(c1['resid'], c2['resid']), 'zcheck': zc[b], 'inc_resid': max(inc[M].resid for M in MAPS)}
    chk |= remainder_checks([P1, P2])
    store = {}
    for k in ('Hbar_gate', 'Hbar_value'):
      store[k] = c1[k].float().cpu()
    for k in ('qbar_gate', 'qbar_value', 'cross'):
      store[k] = c1[k].float().cpu()
    for k in ('Phi', 'Pi', 'dR', 'R1', 'R2'):
      store[k] = c1[k].cpu()
    comp, comp_st, comp2 = 0.0, 0.0, 0.0
    for h in (0, 1):
      cmh = {k: v[h] for k, v in c1.items() if torch.is_tensor(v) and v.dim() >= 3}
      comp = max(comp, split_completeness(inc, cmh)['rel'])
      st = {k: (store[k][h].to(run.dev, torch.float64)) for k in store}
      comp_st = max(comp_st, split_completeness(inc, st)['rel'])
      cm2 = {k: v[h] for k, v in c2.items() if torch.is_tensor(v) and v.dim() >= 3}
      comp2 = max(comp2, split_completeness(inc, cm2)['rel'])
    chk |= {'complete_rel': comp, 'complete_rel_stored': comp_st, 'complete_rel_p2': comp2}
    if run.has_random and run.is_ckpt(s1) and run.is_ckpt(s2):
      PR1, PR2 = run.parts(s1, b, 'random', bounds=False), run.parts(s2, b, 'random', bounds=False)
      incr = {M: increments(PR1[M], PR2[M]) for M in MAPS}
      chk |= {f'{k}_random': v for k, v in remainder_checks([PR1, PR2]).items()}
      diff = 0.0
      for h in (0, 1):
        cmh = {k: v[h] for k, v in c1.items() if torch.is_tensor(v) and v.dim() >= 3}
        tc, tr = group_total(inc, cmh), group_total(incr, cmh)
        diff = max(diff, float((tc - tr).abs().max()) / max(float(cmh['R1'].abs().max()), 1e-30))
      chk['class_vs_random_total_rel'] = diff
      chk['eps_class_max'] = max(float(inc[M].eps.abs().max()) for M in MAPS)
      chk['eps_random_max'] = max(float(incr[M].eps.abs().max()) for M in MAPS)
    pr = pairs.get(b, {})
    idx = [(int(n), int(c)) for n, cls in pr.items() for c in cls]
    if idx:
      I = torch.tensor(idx, dtype=torch.long)
      un, cl = I[:, 0].to(run.dev), I[:, 1].to(run.dev)
      pst = {'idx': I}
      for k in ('Hcross_gate', 'Hcross_value'):
        pst[k] = c1[k][:, un, cl].float().cpu()
      for k in ('Hbar_gate', 'Hbar_value', 'qbar_gate', 'qbar_value', 'Phi', 'Pi'):
        v = c2[k][:, un, cl]
        pst[f'{k}2'] = (v.float() if k.startswith(('Hbar', 'qbar')) else v).cpu()
      store['pairs'] = pst
    store['checks'] = chk
    out['blocks'][b] = store
  return out


def cmd_intervals(a, run: Run, probe: Probe):
  ev = os.path.join(a.out, 'events.json')
  if not os.path.exists(ev) and not a.no_events:
    raise SystemExit('events.json missing: events are frozen before any decomposition (run the events stage)')
  pairs = {}
  if os.path.exists(ev):
    pairs = {b: {int(n): c for n, c in v.items()} for b, v in json.load(open(ev))['pairs'].items()}
  for s1, s2, role, j in interval_plan(run, a.family):
    path = os.path.join(a.out, f'xhw_{s1:05d}_{s2:05d}.pt')
    if os.path.exists(path):
      continue
    t0 = time.time()
    rec = interval_file(run, s1, s2, probe, pairs, a.batch, a.prm['nodes'])
    rec['role'], rec['parent'] = role, j
    rec['events_sha'] = file_sha(ev)                            # the event pairs these per-pair means were made for
    torch.save(rec, path + '.tmp')
    os.replace(path + '.tmp', path)
    worst = {k: max(r['checks'].get(k, 0.0) for r in rec['blocks'].values())
             for k in ('complete_rel', 'complete_rel_stored', 'complete_rel_p2', 'image_resid', 'zcheck',
                       'class_vs_random_total_rel', 'R_rel', 'R_stored_rel')}
    print(json.dumps({'interval': [s1, s2], 'role': role, **worst, 'sec': round(time.time() - t0, 1)}), flush=True)
    if worst['R_stored_rel'] > 1e-5:
      print(f'WARNING: interval {s1}-{s2}: a checkpoint\'s stored remainder disagrees with its model file '
            f'(R_stored_rel {worst["R_stored_rel"]:.1e}): model and attribution files of different trajectories?',
            flush=True)


# ------------------------------------------------------------------------------------------------------ gradstate

def precond(run: Run, s: int, b: str, prm: dict):
  """(U, 2d + 2) diagonal preconditioner [gate row, gate bias, value row, value bias] of the update at step s, from the
  nearest usable snapshot within d_window steps (AdamW: 1 / (sqrt(v / (1 - b2^t)) + eps) for weights and biases;
  Muon: 1 for the weights and the biases, since the biases' AdamW factor, about 1e5 to 1e6, against the weights' 1
  would make kappa and beta bias-only quantities), and that snapshot's step; ones and None before d_min_step or
  without a snapshot nearby (D_RULE names the rule)."""
  near = [x for x in run.snaps if abs(x - s) <= prm['d_window']]
  U, d = run.U[b], run.d[b]
  if s < prm['d_min_step'] or not near:
    return torch.ones(U, 2 * d + 2, dtype=torch.float64, device=run.dev), None
  x = min(near, key=lambda q: (abs(q - s), q))
  A = run.full(x)['attr']
  muon = run.cfg.get('opt') == 'muon'
  parts = []
  for M in MAPS:
    r = A[f'{b}.pw_{M}']
    c2 = 1 - 0.999 ** int(r['t'])
    Dw = torch.ones(U, d, dtype=torch.float64) if muon else 1 / ((r['v'].double() / c2).sqrt() + 1e-8)
    Db = torch.ones(U, dtype=torch.float64) if muon else 1 / ((r['vb'].double() / c2).sqrt() + 1e-8)
    parts += [Dw, Db[:, None]]
  return torch.cat(parts, 1).to(run.dev), x


D_RULE = {'adamw': '1 / (sqrt(v / (1 - 0.999^t)) + 1e-8), weights and biases, nearest usable snapshot',
          'muon': '1 for weights and biases (no preconditioner)'}


def gradstate_point(run: Run, s: int, probe: Probe, batch: int, prm: dict, stock: dict) -> dict:
  """kappa = <mean_c j*, D mean_c G> / sd and beta = -<mean_c j*, D mean_c Ebar> / sd per half, unit and class:
  j* = du/dtheta at p* (theta = [gate row, gate bias, value row, value bias]), G = sum_p du(p)/dtheta, Ebar = the
  per-image gradient of -log softmax_y with respect to theta; mu at snapshot steps (AdamW): -(lr_t / (1 - 0.9^t))
  <mean_k j*, D_t m_t^(k)> / sd with the exact D_t and per-class moments."""
  model = run.model(s)
  for p in model.parameters():
    p.requires_grad_(False)
  named = dict(model.blocks())
  G = 2 * probe.C
  cap, hooks = {}, []
  for b in run.blocks:
    for M in MAPS:
      hooks.append(getattr(named[b], f'pw_{M}').register_forward_hook(
        lambda mod, i, o, b=b, M=M: cap.__setitem__((b, M), (i[0], o))))
  acc = {b: {k: torch.zeros(G, run.U[b], 2 * run.d[b] + 2, dtype=torch.float64, device=run.dev)
             for k in ('J', 'G', 'E')} for b in run.blocks}
  cnt = torch.zeros(G, dtype=torch.float64, device=run.dev)
  for xb, yb, gb in probe.batches(batch, run.dev):
    xb.requires_grad_(True)
    cap.clear()
    loss = F.cross_entropy(model(xb), yb, reduction='sum')
    zs = [cap[(b, M)][1] for b in run.blocks for M in MAPS]
    grads = torch.autograd.grad(loss, zs)
    cnt.index_add_(0, gb, torch.ones(len(gb), dtype=torch.float64, device=run.dev))
    with torch.no_grad():
      for i, b in enumerate(run.blocks):
        h = cap[(b, 'gate')][0].flatten(1, 2).float()
        zg, zv = (cap[(b, M)][1].flatten(1, 2).float() for M in MAPS)
        dg, dv = (g.flatten(1, 2).float() for g in grads[2 * i:2 * i + 2])
        A, B = dsilu(zg) * zv, F.silu(zg)
        p = (B * zv).argmax(1)
        hs, As, Bs = take(h, p), take_u(A, p), take_u(B, p)
        J = torch.cat([As[..., None] * hs, As[..., None], Bs[..., None] * hs, Bs[..., None]], -1)
        Gs = torch.cat([torch.einsum('bpu,bpd->bud', A, h), A.sum(1)[..., None],
                        torch.einsum('bpu,bpd->bud', B, h), B.sum(1)[..., None]], -1)
        Es = torch.cat([torch.einsum('bpu,bpd->bud', dg, h), dg.sum(1)[..., None],
                        torch.einsum('bpu,bpd->bud', dv, h), dv.sum(1)[..., None]], -1)
        for k, v in (('J', J), ('G', Gs), ('E', Es)):
          acc[b][k].index_add_(0, gb, v.double())
  for hk in hooks:
    hk.remove()
  out = {'step': s, 'blocks': {}, 'D_source': None, 'D_rule': D_RULE.get(run.cfg.get('opt', 'adamw'))}
  snap_here = s in run.snaps and run.cfg.get('opt', 'adamw') == 'adamw'
  for b in run.blocks:
    m = {k: to_half_major(v / cnt.view(-1, 1, 1), probe.C) for k, v in acc[b].items()}     # (2, U, C, 2d + 2)
    D, src = precond(run, s, b, prm)
    sd = stock[b]['R'].std(-1, correction=0).to(run.dev).clamp(min=1e-30)                   # (2, U)
    kappa = (m['J'] * D[None, :, None] * m['G']).sum(-1) / sd[..., None]
    beta = -(m['J'] * D[None, :, None] * m['E']).sum(-1) / sd[..., None]
    rec = {'kappa': kappa.float().cpu(), 'beta': beta.float().cpu(), 'D_source': src}
    out['D_source'] = src
    if snap_here:
      A = run.full(s)['attr']
      mk, Dx = [], []
      for M in MAPS:
        r = A[f'{b}.pw_{M}']
        mk += [r['m'].double(), r['mb'].double()[..., None]]
        c2 = 1 - 0.999 ** int(r['t'])
        Dx += [1 / ((r['v'].double() / c2).sqrt() + 1e-8), 1 / ((r['vb'].double() / c2).sqrt() + 1e-8)[:, None]]
      mk = torch.cat(mk, -1).to(run.dev)[:probe.C]                                          # (C, U, 2d + 2)
      Dx = torch.cat(Dx, -1).to(run.dev)
      t = int(A[f'{b}.pw_gate']['t'])
      lr = lr_schedule(run.cfg)(t - 1)                    # update t (count t - 1 -> t) used m_t, v_t and this lr
      mu = -(lr / (1 - 0.9 ** t)) * torch.einsum('hnkj,nj,knj->hnk', m['J'], Dx, mk) / sd[..., None]
      rec['mu'] = mu.float().cpu()
    out['blocks'][b] = rec
  return out


def cmd_gradstate(a, run: Run, probe: Probe):
  steps = run.gplus if a.steps == '' else [int(t) for t in a.steps.split(',')]
  for s in steps:
    path = os.path.join(a.out, f'gstate_{s:05d}.pt')
    if os.path.exists(path):
      continue
    t0 = time.time()
    rec = gradstate_point(run, s, probe, a.batch, a.prm, load_stock(a.out, s))
    save_atomic(rec, path)
    print(json.dumps({'step': s, 'D_source': rec['D_source'], 'sec': round(time.time() - t0, 1)}), flush=True)


# ---------------------------------------------------------------------------------------------- X1 components

def category_index(x: int, sets: dict, K: int) -> torch.Tensor:
  """(K,) int8 write category of every group relative to target class x: 0 own = {x}, 1 P, 2 Ent, 3 rel, 4 oth (each
  group in the first category that holds it)."""
  cat = torch.full((K,), 4, dtype=torch.int8)
  for i, name in ((3, 'rel'), (2, 'Ent'), (1, 'P')):
    ids = [int(k) for k in sets.get(name, [])]
    if ids:
      cat[ids] = i
  cat[x] = 0
  return cat


def batch_components(view: dict, xs, Qp, Qm, cat, bounds: bool = False) -> torch.Tensor:
  """(N, 20) margin components (COMP order) of N targets in one interval. xs (N,) target classes; Qp (N, q) their
  companies (padded; mask Qm), an empty company meaning no margin (the target's own response); cat (N, K) int8 write
  categories. For each write category X: X_c = sum_{k in X} T_k(x) and X_P = -mean_{s in Q} sum_{k in X} T_k(s); the
  same two parts of F_own, F_rest (+ F_0), Phi, Pi and E. view: T, F (K, S), F0, E, Phi, Pi, dR (S,), pos (C,) column of
  each class; with bounds the matching storage bounds from T_bound, F_bound (all positive): R absorbs the rounding, so
  F_rest moves by exactly minus F_own's move (its bound is F_own's) and E by minus the writes' total move (its bound,
  the sum of every group's T bound, is for reporting; `corner` never shifts by it); Phi and Pi have none."""
  pos = view['pos']
  Tm, Fm = (view['T_bound'], view['F_bound']) if bounds else (view['T'], view['F'])
  px, pQ = pos[xs], pos[Qp]
  if not ((px >= 0).all() and (pQ[Qm] >= 0).all()):              # -1 would silently read the last column
    raise ValueError('batch_components: a target or company class has no column in this view (interval files made '
                     'for another events.json?)')
  oh = F.one_hot(cat.long(), len(WRITE)).to(Tm.dtype)                    # (N, K, 5)
  m = Qm.to(Tm.dtype)
  cnt = m.sum(1).clamp(min=1)
  has = (Qm.sum(1) > 0).to(Tm.dtype)
  mean_q = lambda v: (v * m).sum(1) / cnt * has
  sg = 1.0 if bounds else -1.0
  out = torch.zeros(len(xs), len(COMP), dtype=Tm.dtype, device=Tm.device)
  out[:, 0:NWRITE:2] = torch.einsum('kn,nkc->nc', Tm[:, px], oh)
  cq = torch.einsum('knq,nkc->nqc', Tm[:, pQ], oh)
  out[:, 1:NWRITE:2] = sg * (cq * m[..., None]).sum(1) / cnt[:, None] * has[:, None]
  Fx, FQx = Fm[xs, px], Fm[xs[:, None], pQ]
  out[:, NWRITE] = Fx
  out[:, NWRITE + 1] = sg * mean_q(FQx)
  if bounds:
    out[:, NWRITE + 2] = Fx
    out[:, NWRITE + 3] = mean_q(FQx)
    out[:, NWRITE + 8] = Tm[:, px].sum(0)
    out[:, NWRITE + 9] = mean_q(Tm[:, pQ].sum(0))
  else:
    out[:, NWRITE + 2] = Fm[:, px].sum(0) - Fx + view['F0'][px]
    out[:, NWRITE + 3] = -mean_q(Fm[:, pQ].sum(0) - FQx + view['F0'][pQ])
    for i, k in enumerate(('Phi', 'Pi', 'E')):
      out[:, NWRITE + 4 + 2 * i] = view[k][px]
      out[:, NWRITE + 5 + 2 * i] = -mean_q(view[k][pQ])
  return out


def batch_margin(view: dict, xs, Qp, Qm) -> torch.Tensor:
  """dR(x) - mean_{s in Q} dR(s) of N targets (dR(x) without a company)."""
  pos, dR = view['pos'], view['dR']
  m = Qm.to(dR.dtype)
  return dR[pos[xs]] - (dR[pos[Qp]] * m).sum(1) / m.sum(1).clamp(min=1)


def xlevel(comp: np.ndarray) -> np.ndarray:
  """(..., 20) components -> (..., 10) categories (part on the target plus part on the company)."""
  return comp[..., 0::2] + comp[..., 1::2]


def corner_sign(pat: str) -> np.ndarray:
  """(20,) directions of a corner pattern: 'all+' / 'all-' every component up / down, 'own+' / 'own-' the own write up /
  down and every other component the opposite way."""
  sg = np.ones(len(COMP)) if pat in ('all+', 'own+') else -np.ones(len(COMP))
  if pat in ('own+', 'own-'):
    sg[2:] *= -1
  return sg


def corner(x: np.ndarray, bb: np.ndarray, sg: np.ndarray) -> np.ndarray:
  """Components x (..., 20) moved to a corner of their storage bounds bb (..., 20) in the directions sg (20,) of +-1,
  as a change of the stored fp16 parts would move them with R recomputed from W: every write component by sg bb and
  E by minus their sum (on c and on the company separately: sum_k T_k + E is fixed), F_own by sg bb and F_rest by the
  opposite (F_own + F_rest is fixed); Phi and Pi do not move. The sum of the components is unchanged."""
  s = sg * np.asarray(bb, float)
  y = np.array(x, float, copy=True)
  y[..., :NWRITE] += s[..., :NWRITE]
  iE, iFo, iFr = COMP.index('E_c'), COMP.index('Fown_c'), COMP.index('Frest_c')
  for o in (0, 1):
    y[..., iE + o] -= s[..., o:NWRITE:2].sum(-1)
    y[..., iFo + o] += s[..., iFo + o]
    y[..., iFr + o] -= s[..., iFo + o]
  return y


def classify(ex: np.ndarray, ctrl: np.ndarray, thr: dict) -> str:
  """Fixed exploratory classification of one exit (module docstring). ex (20,), ctrl (m, 20) components; thr:
  rand_band (lo, hi) or None, rand_q05 or None (own term against random groups), P_q05 (P_c against size-matched sets),
  band_min (default 3). Without a random band the own band is the C1 own terms' 25-75% (inside) and 5th percentile
  (below), which needs at least band_min C1 members: with fewer the exit is 'few_controls' (the percentiles of one or
  two values are not a band; one control would make 'active' a coin flip)."""
  if not len(ctrl):
    return 'no_control'
  delta = ex - ctrl.mean(0)
  D = delta.sum()
  if not D < 0:
    return 'no_excess_fall'
  sh = delta / D
  sx = xlevel(sh)
  own_m = ex[0] + ex[1]
  if thr.get('rand_band') is not None:
    lo, hi = thr['rand_band']
    q05 = thr['rand_q05']
  elif len(ctrl) < thr.get('band_min', 3):
    return 'few_controls'
  else:
    own_c = ctrl[:, 0] + ctrl[:, 1]
    lo, hi = np.percentile(own_c, [25, 75])
    q05 = np.percentile(own_c, 5)
  if sx[0] <= 0.15 and lo <= own_m <= hi and sx[XCAT.index('Fown')] + sh[COMP.index('P_P')] >= 0.5:
    return 'withdrawn'
  if sx[0] >= 0.4 and own_m < q05:
    return 'active'
  if thr.get('P_q05') is not None and sh[COMP.index('P_c')] >= 0.4 and ex[COMP.index('P_c')] < thr['P_q05']:
    return 'interference'
  if sx[XCAT.index('Phi')] >= 0.5:
    return 'drift'
  return 'unclassified'


class EventAcc:
  """Running sums over the intervals of one event's window, for the target (first) and its controls: the 20
  components per variant ('nom', 'wf', 'if', 'p2', 'gplus'), their storage bounds (nominal), the margin change dM,
  and for an exit the per-group terms on the target (mc) and on the company mean (mP), class and random splits.
  targets: [(class, company classes, (K,) int8 categories)]."""

  def __init__(self, kind: str, d: str, e: dict, targets: list, K: int, KR: int, dev):
    self.kind, self.d, self.e = kind, d, e
    self.xs = torch.tensor([int(t[0]) for t in targets], dtype=torch.long)
    self.Qs = [[int(k) for k in t[1]] for t in targets]
    self.cat = torch.stack([t[2] for t in targets])
    z = lambda: torch.zeros(len(targets), len(COMP), dtype=torch.float64, device=dev)
    self.comp = {'nom': z()}
    self.bound = z()
    self.dM = torch.zeros(len(targets), dtype=torch.float64, device=dev)
    self.mc = torch.zeros(K, dtype=torch.float64, device=dev)
    self.mP = torch.zeros(K, dtype=torch.float64, device=dev)
    self.mRc = torch.zeros(KR, dtype=torch.float64, device=dev) if KR else None
    self.mRP = torch.zeros(KR, dtype=torch.float64, device=dev) if KR else None
    self.check = 0.0
    self.n_iv = {}


def padded(Qs: list, xs: torch.Tensor, dev):
  """Companies as a padded (N, q) index tensor (padding: the target itself, a valid column) and its mask."""
  q = max(1, max(len(Q) for Q in Qs))
  Qp = torch.tensor([Q + [int(x)] * (q - len(Q)) for Q, x in zip(Qs, xs.tolist())], dtype=torch.long, device=dev)
  Qm = torch.tensor([[True] * len(Q) + [False] * (q - len(Q)) for Q in Qs], dtype=torch.bool, device=dev)
  return Qp, Qm


def run_batch(items: list, view: dict, var: str, bounds: bool = False, vectors: bool = False, rview: dict = None):
  """Adds the components of every target of the events `items` in one interval (view) to their running sums; with
  vectors, also the exits' per-group terms on the target and on the company mean (and the random split's, rview)."""
  dev = view['T'].device
  xs = torch.cat([ac.xs for ac in items]).to(dev)
  Qp, Qm = padded([Q for ac in items for Q in ac.Qs], xs, dev)
  cat = torch.cat([ac.cat for ac in items]).to(dev)
  comp = batch_components(view, xs, Qp, Qm, cat)
  dm = batch_margin(view, xs, Qp, Qm)
  bnd = batch_components(view, xs, Qp, Qm, cat, bounds=True) if bounds else None
  chk = (comp.sum(1) - dm).abs() / (dm.abs() + comp.abs().sum(1)).clamp(min=1e-300)
  off = 0
  for ac in items:
    n = len(ac.xs)
    if var not in ac.comp:
      ac.comp[var] = torch.zeros_like(ac.comp['nom'])
    ac.comp[var] += comp[off:off + n]
    if var == 'nom':
      ac.dM += dm[off:off + n]
    if bnd is not None:
      ac.bound += bnd[off:off + n]
    ac.check = max(ac.check, float(chk[off:off + n].max()))
    ac.n_iv[var] = ac.n_iv.get(var, 0) + 1
    off += n
  ex = [ac for ac in items if ac.kind == 'exit']
  if vectors and ex:
    x0 = torch.stack([ac.xs[0] for ac in ex]).to(dev)
    Q0p, Q0m = padded([ac.Qs[0] for ac in ex], x0, dev)
    pos = view['pos']
    m = Q0m.to(torch.float64)
    for split, src in (('', view), ('R', rview)):
      if src is None or (split == 'R' and ex[0].mRc is None):
        continue
      Tc = src['T'][:, pos[x0]]                                              # (K, E)
      TP = (src['T'][:, pos[Q0p]] * m[None]).sum(-1) / m.sum(1).clamp(min=1)[None]
      for i, ac in enumerate(ex):
        getattr(ac, f'm{split}c').add_(Tc[:, i])
        getattr(ac, f'm{split}P').add_(TP[:, i])


def make_view(ct: dict, Phi, Pi, dR, cols, C: int) -> dict:
  pos = torch.full((C,), -1, dtype=torch.long, device=Phi.device)
  pos[cols] = torch.arange(len(cols), device=Phi.device)
  return dict(ct, Phi=Phi, Pi=Pi, dR=dR, pos=pos)


# --------------------------------------------------------------------------------------------------- tests stage

class Ctx:
  """Everything the tests share: the run, stock responses, events, class relations and counts, parameters."""

  def __init__(self, a, run: Run):
    self.a, self.run, self.prm, self.out = a, run, a.prm, a.out
    self.dev = run.dev
    self.events = json.load(open(os.path.join(a.out, 'events.json')))
    assert self.events['steps'] == run.g16, 'events.json was built on another checkpoint grid'
    assert self.events.get('version', 0) >= 3, ('events.json of an older version (no sd floor at selection): rebuild '
                                                'it with events --force, then its interval files')
    self.C = self.K = run.n_classes
    self.stock = {s: load_stock(a.out, s) for s in run.gplus
                  if os.path.exists(os.path.join(a.out, f'stock_{s:05d}.pt'))}
    self.R = {s: {b: v['R'].numpy() for b, v in st.items()} for s, st in self.stock.items()}
    vis = np.load(a.visual)['rank_dist'][:self.C, :self.C].astype(np.int64)
    wn = np.load(a.wordnet)['dist'][:self.C, :self.C].astype(np.int64)
    eye = np.eye(self.C, dtype=bool)
    rel_vis = (vis <= self.prm['visual_near']) & ~eye
    rel_wn = (wn <= self.prm['wordnet_near']) & ~eye
    self.rel = rel_wn if self.prm['relatives'] == 'wordnet' else rel_vis       # primary visual, secondary WordNet
    self.unrel = (wn >= self.prm['unrel_wn']) & (vis > self.prm['unrel_vis']) & ~eye
    self.n_c, self.n_r, self.n_c_info = class_counts(a, run, self.C)
    self.rng = np.random.default_rng(self.prm['seed'])
    files = {}
    for f in glob.glob(os.path.join(a.out, 'xhw_[0-9]*_[0-9]*.pt')):
      s1, s2 = int(os.path.basename(f)[4:9]), int(os.path.basename(f)[10:15])
      files[(s1, s2)] = f
    self.files = files
    sha = file_sha(os.path.join(a.out, 'events.json'))
    stale = sorted(k for k, f in files.items()
                   if torch.load(f, map_location='cpu', weights_only=False, mmap=True).get('events_sha', sha)
                   not in (sha, ''))                              # '': made without events (no pairs, no variants)
    assert not stale, f'interval files made for another events.json (delete them and rerun intervals): {stale}'
    plan = interval_plan(run, 'all')
    self.plan = [x for x in plan if (x[0], x[1]) in files]
    missing = [x for x in plan if x[2] == 'g16' and (x[0], x[1]) not in files]
    assert not missing, f'G16 interval files missing: {missing}'
    subs = {}
    for s1, s2, role, j in self.plan:
      if role == 'sub':
        subs.setdefault(j, []).append((s1, s2))
    self.refined = {j for j, v in subs.items() if len(v) == len([x for x in plan if x[2] == 'sub' and x[3] == j])}
    self._hs = {}
    self.gstate = {}
    for s in run.gplus:
      p = os.path.join(a.out, f'gstate_{s:05d}.pt')
      if os.path.exists(p):
        self.gstate[s] = torch.load(p, map_location='cpu', weights_only=False)

  def sd(self, s: int, b: str) -> np.ndarray:
    """(2, U) std over classes of R."""
    return self.R[s][b].std(-1)

  def floor(self, s: int, h: int) -> float:
    return self.prm['sd_floor'] * float(np.median(np.concatenate([self.sd(s, b)[h] for b in self.run.blocks])))

  def stockh(self, s: int):
    """Class-mean inputs Hs of grid point s (memory-mapped, cached), or None."""
    if s not in self._hs:
      p = os.path.join(self.out, f'stockh_{s:05d}.pt')
      self._hs[s] = load_stockh(self.out, s) if os.path.exists(p) else None
    return self._hs[s]

  def xhw(self, s1: int, s2: int) -> dict:
    return torch.load(self.files[(s1, s2)], map_location='cpu', weights_only=False, mmap=True)


def class_counts(a, run: Run, C: int):
  """(n_c, n_r, info): training images per class and per random group (n_r None without a random split), and where
  n_c came from (source, kind, file sha256, n_c_sha of the counts themselves, min, max, sum, zero_classes). Source:
  --class_counts, else <run>/rand_groups.npy, else /dev/shm/imagenet256/labels.npy. A file of exactly C entries is
  per-class counts (any dtype; a label file has one entry per training image), any other length per-example labels
  (all in 0..C-1). No source is an error: equal counts would silently turn every count-matched null into a
  size-matched one; only --params '{"equal_counts": true}' (toy tests) allows them."""
  rg = os.path.join(run.dir, 'rand_groups.npy')
  n_r = np.bincount(np.load(rg).astype(np.int64), minlength=C).astype(float) if os.path.exists(rg) else None
  src = a.class_counts or (rg if os.path.exists(rg) else '/dev/shm/imagenet256/labels.npy')
  if src and os.path.exists(src):
    v = np.asarray(np.load(src)).ravel()
    per_class = len(v) == C
    if per_class:
      if not (v >= 0).all():
        raise SystemExit(f'class counts {src}: negative counts')
      n_c = v.astype(float)
    else:
      if not (v.min() >= 0 and v.max() < C):
        raise SystemExit(f'class counts {src}: {len(v)} entries with labels outside 0..{C - 1} (a per-class counts '
                         f'file must have exactly {C} entries)')
      n_c = np.bincount(v.astype(np.int64), minlength=C).astype(float)
    assert len(n_c) == C, f'class counts from {src}: {len(n_c)} entries, expected {C}'
    info = {'source': src, 'kind': 'per_class' if per_class else 'labels', 'sha256': file_sha(src)}
  elif a.prm.get('equal_counts'):
    print('WARNING: equal class counts (equal_counts): the count-matched nulls are matched on size only', flush=True)
    n_c = np.ones(C)
    info = {'source': 'equal (equal_counts)', 'kind': 'equal', 'sha256': ''}
  else:
    raise SystemExit(f'no class counts: {src or "--class_counts"} not found and no {rg}; pass --class_counts '
                     f'(per-example labels, or a {C}-entry per-class counts file), or --params '
                     '\'{"equal_counts": true}\' (toy tests)')
  zero = int((n_c == 0).sum())
  if zero:
    print(f'WARNING: {zero} classes have a training count of 0 in {info["source"]}', flush=True)
  info |= {'n_c_sha': array_sha(n_c), 'min': float(n_c.min()), 'max': float(n_c.max()), 'sum': float(n_c.sum()),
           'zero_classes': zero, 'equal': bool((n_c == n_c[0]).all())}
  return n_c, n_r, info


def plasticity(inc: dict, cm: dict, unrel: torch.Tensor, chunk: int = 8) -> torch.Tensor:
  """P_n,c = mean over the unrelated classes j of c of |T_j(c)| (U, C), in fp32 (a baseline of the X4 ladder)."""
  U = cm['Hbar_gate'].shape[0]
  cnt = unrel.sum(1).clamp(min=1).float()                                 # (C,)
  out = []
  for n0 in range(0, U, chunk):
    T = 0
    for M in MAPS:
      I = inc[M]
      T = T + torch.einsum('knd,ncd->knc', I.nu[:, n0:n0 + chunk].float(), cm[f'Hbar_{M}'][n0:n0 + chunk].float()) \
            + I.nub[:, n0:n0 + chunk, None].float() * cm[f'qbar_{M}'][None, n0:n0 + chunk].float()
    out.append((T.abs() * unrel.T[:, None, :].float()).sum(0) / cnt)
  return torch.cat(out)


def total_writes(inc: dict, cm: dict) -> torch.Tensor:
  """sum_k T_k(c) for every unit and class (U, C)."""
  return sum(torch.einsum('nd,ncd->nc', inc[M].nu.sum(0), cm[f'Hbar_{M}'])
             + inc[M].nub.sum(0)[:, None] * cm[f'qbar_{M}'] for M in MAPS)


def provenance(run: Run, s: int, b: str, hs: torch.Tensor, stk: dict, rel: torch.Tensor) -> torch.Tensor:
  """(2, U, C, 6) provenance of the on-ness at s: relatives', own, others' and init parts of the class-mean gate and
  value pre-activations, in response units Abar g_S + Bbar v_S, centred over classes (columns rel, own, oth, init),
  plus the relatives' gate and value parts separately; and the check max |sum of parts - Gm, Vm|."""
  P = run.parts(s, b, 'class', bounds=False)
  C = hs.shape[2]
  res, chk = [], 0.0
  for h in (0, 1):
    H = hs[h].to(run.dev, torch.float64)                                  # (U, C, d)
    parts = {}
    for M in MAPS:
      p = P[M]
      K, U, d = p.C.shape
      own = torch.einsum('cnd,ncd->nc', p.C[:C], H) + p.Cb[:C].T
      S = (rel.double() @ p.C.reshape(K, U * d)).reshape(C, U, d)
      relp = torch.einsum('cnd,ncd->nc', S, H) + (rel.double() @ p.Cb).T
      tot = torch.einsum('nd,ncd->nc', p.C.sum(0), H) + p.Cb.sum(0)[:, None]
      init = torch.einsum('nd,ncd->nc', p.a * p.W0 + p.R, H) + (p.ab * p.b0 + p.Rb)[:, None]
      parts[M] = dict(rel=relp, own=own, oth=tot - own - relp, init=init)
      ref = stk['Gm' if M == 'gate' else 'Vm'][h].to(run.dev, torch.float64)
      chk = max(chk, float((tot + init - ref).abs().max() / ref.abs().max().clamp(min=1e-30)))
    A, B = stk['Abar'][h].to(run.dev, torch.float64), stk['Bbar'][h].to(run.dev, torch.float64)
    cols = [A * parts['gate'][k] + B * parts['value'][k] for k in ('rel', 'own', 'oth', 'init')]
    cols += [A * parts['gate']['rel'], B * parts['value']['rel']]
    X = torch.stack(cols, -1)
    res.append(X - X.mean(1, keepdim=True))
  return torch.stack(res).float().cpu(), chk


def analyze_block(ctx: Ctx, b: str) -> dict:
  """One block: completeness of every interval, the own / other / plasticity arrays (X4, X5), the X1 and X2 event
  accumulators over their windows (all variants), and the provenance at every G16 start point (X4 F2)."""
  run, dev, C, K = ctx.run, ctx.dev, ctx.C, ctx.K
  g = run.g16
  KR = K if run.has_random else 0
  rel_t = torch.from_numpy(ctx.rel).to(dev)
  unrel_t = torch.from_numpy(ctx.unrel).to(dev)
  rec = {'complete': {}, 'own': {}, 'own_bound': {}, 'oth': {}, 'plast': {}, 'prov': {}, 'prov_check': 0.0,
         'view_check': 0.0}
  # event accumulators
  accs = []
  for d, (s_, m_) in DIRS.items():
    for e in ctx.events['directions'][d]['exits']:
      if e['b'] != b:
        continue
      c = e['c']
      Ppk = sorted(set(e['P']) | {c})
      rel = [int(k) for k in np.flatnonzero(ctx.rel[c])]
      tg = [(c, e['P'], category_index(c, {'P': e['P'], 'Ent': e['Ent'], 'rel': rel}, K))]
      for k in e['C1']:
        Pk = [x for x in Ppk if x != k]
        tg.append((k, Pk, category_index(k, {'P': Pk, 'Ent': [x for x in e['Ent'] if x != k],
                                             'rel': [int(r) for r in np.flatnonzero(ctx.rel[k])]}, K)))
      accs.append(EventAcc('exit', d, e, tg, K, KR, dev))
      if e['entry'] is not None:
        en = dict(e['entry'], n=e['n'], c=c, b=b, dropped=True, exit=e)
        accs.append(entry_acc(ctx, d, en, K, dev))
    for e in ctx.events['directions'][d]['persistent']:
      if e['b'] == b:
        accs.append(entry_acc(ctx, d, dict(e, dropped=False), K, dev))
  by_unit = {}
  for ac in accs:
    by_unit.setdefault(ac.e['n'], []).append(ac)
  for s1, s2, role, j in ctx.plan:
    X = ctx.xhw(s1, s2)['blocks'][b]
    inc = run.inc(s1, s2, b, 'class', bounds=role in ('g16', 'sub'))
    incr = run.inc(s1, s2, b, 'random', bounds=False) if (role == 'g16' and run.has_random) else None
    for h in (0, 1):
      cm = {k: X[k][h].to(dev, torch.float64) for k in ('Hbar_gate', 'Hbar_value', 'qbar_gate', 'qbar_value', 'Phi',
                                                        'Pi', 'dR', 'R1', 'R2', 'cross')}
      rec['complete'][(s1, s2, h)] = split_completeness(inc, cm)['rel']
      own = own_terms(inc, cm)
      rec['own'][(s1, s2, h)] = own.float().cpu()
      if role in ('g16', 'sub'):                                  # fp16 storage bound of T_c(c) (compact parts)
        rec['own_bound'][(s1, s2, h)] = own_bounds(inc, cm).float().cpu()
      rec['oth'][(s1, s2, h)] = (total_writes(inc, cm) - own).float().cpu()
      if role == 'g16':
        rec['plast'][(j, h)] = plasticity(inc, cm, unrel_t).cpu()
      pr = X.get('pairs')
      for n, acs in by_unit.items():
        act = []
        for ac in acs:
          if DIRS[ac.d][1] != h:
            continue
          e = ac.e
          if ac.kind == 'exit':
            if role == 'g16' and e['pk'] < j <= e['out']:
              act.append((ac, 'g16'))
            elif role == 'sub' and j in ctx.refined and e['pk'] < j <= e['out']:
              act.append((ac, 'sub'))
          elif role == 'g16' and e['pre'] < j <= e['first']:
            act.append((ac, 'g16'))
        if not act:
          continue
        cols = sorted({x for ac, _ in act for x in ac.xs.tolist()} | {k for ac, _ in act for Q in ac.Qs for k in Q})
        colt = torch.tensor(cols, dtype=torch.long, device=dev)
        bounds = role in ('g16', 'sub')
        ct = column_terms(inc, {M: cm[f'Hbar_{M}'][n, colt] for M in MAPS}, {M: cm[f'qbar_{M}'][n, colt] for M in MAPS},
                          n, bounds=bounds)
        view = make_view(ct, cm['Phi'][n, colt], cm['Pi'][n, colt], cm['dR'][n, colt], colt, C)
        rview = None
        if incr is not None:
          ctr = column_terms(incr, {M: cm[f'Hbar_{M}'][n, colt] for M in MAPS},
                             {M: cm[f'qbar_{M}'][n, colt] for M in MAPS}, n)
          rview = make_view(ctr, view['Phi'], view['Pi'], view['dR'], colt, C)
        variants = {}
        if role == 'g16' and pr is not None and any(ac.kind == 'exit' for ac, _ in act):
          rows = torch.nonzero(pr['idx'][:, 0] == n)[:, 0]
          if len(rows):
            pcls = pr['idx'][rows, 1].to(dev)
            ctp = column_terms(inc, {M: cm[f'Hbar_{M}'][n, pcls] for M in MAPS},
                               {M: cm[f'qbar_{M}'][n, pcls] for M in MAPS}, n)
            cx = column_terms(inc, {M: pr[f'Hcross_{M}'][h][rows].to(dev, torch.float64) for M in MAPS}, {}, n)
            base = dict(Pi=cm['Pi'][n, pcls], dR=cm['dR'][n, pcls])
            for nm, sg in (('wf', 1.0), ('if', -1.0)):
              ctv = {k: ctp[k] + sg * cx[k] for k in ('T', 'F', 'F0', 'E')}
              variants[nm] = make_view(ctv, cm['Phi'][n, pcls] - sg * cx['W'], base['Pi'], base['dR'], pcls, C)
            ct2 = column_terms(inc, {M: pr[f'Hbar_{M}2'][h][rows].to(dev, torch.float64) for M in MAPS},
                               {M: pr[f'qbar_{M}2'][h][rows].to(dev, torch.float64) for M in MAPS}, n)
            variants['p2'] = make_view(ct2, pr['Phi2'][h][rows].to(dev, torch.float64),
                                       pr['Pi2'][h][rows].to(dev, torch.float64), base['dR'], pcls, C)
            for vv in variants.values():
              tot = vv['T'].sum(0) + vv['F'].sum(0) + vv['F0'] + vv['E'] + vv['Phi'] + vv['Pi'] - vv['dR']
              rec['view_check'] = max(rec['view_check'], float(tot.abs().max()) /
                                      max(float(cm['R1'].abs().max()), 1e-30))
        sub = [ac for ac, how in act if how == 'sub']
        nom = [ac for ac, how in act if how == 'g16']
        if sub:
          run_batch(sub, view, 'gplus')
        if nom:
          run_batch(nom, view, 'nom', bounds=bounds, vectors=True, rview=rview)
          ex = [ac for ac in nom if ac.kind == 'exit']
          if ex:
            if j not in ctx.refined:
              run_batch(ex, view, 'gplus')
            for nm, vv in variants.items():
              run_batch(ex, vv, nm)
  # provenance at every G16 start point (X4 F2)
  for j in range(1, len(g)):
    s = g[j - 1]
    rec['prov'][j], chk = provenance(run, s, b, ctx.stockh(s)[b], ctx.stock[s][b], rel_t)
    rec['prov_check'] = max(rec['prov_check'], chk)
  rec['accs'] = accs
  return rec


def entry_acc(ctx: Ctx, d: str, en: dict, K: int, dev) -> EventAcc:
  """Targets of one entry (and its rank-matched controls at t_pre): no margin, categories fixed at t_pre (own, P_pre,
  rel, oth)."""
  tg = []
  for x in [en['c']] + en['ctrl']:
    Ppre = [k for k in en['Tpre'] if k != x]
    tg.append((x, [], category_index(x, {'P': Ppre, 'rel': [int(r) for r in np.flatnonzero(ctx.rel[x])]}, K)))
  return EventAcc('entry', d, en, tg, K, 0, dev)


# ------------------------------------------------------------------------------------------ X1 / X2 summaries

def exit_summary(ctx: Ctx, ac: EventAcc) -> dict:
  """Per-exit record: components (normalized by sd_n(t_pk) on the measurement half), C1 / C2 means, standardized
  categories (and the nulls that could not be formed), bands, the classification and its flips. The sd floor is
  decided on the selection half (as at event selection); the measurement half's sd only normalizes."""
  prm, run, e = ctx.prm, ctx.run, ac.e
  s_, m_ = DIRS[ac.d]
  t_pk = run.g16[e['pk']]
  sd = float(ctx.sd(t_pk, e['b'])[m_][e['n']])
  r = {'d': ac.d, 'b': e['b'], 'n': e['n'], 'c': e['c'], 'pk': e['pk'], 'out': e['out'], 'anchored': e['anchored'],
       'rep': e['rep'], 'rep_relaxed': e.get('rep_relaxed', False), 'phase': phase_of(t_pk),
       'late': not e['b'].startswith(('stage0', 'stage1')), 'nC1': len(e['C1']), 'nC2': len(e['C2']), 'check': ac.check}
  if float(ctx.sd(t_pk, e['b'])[s_][e['n']]) < ctx.floor(t_pk, s_):
    r['skip'] = 'low_sd'
    return r
  if not sd > 0:
    r['skip'] = 'zero_sd'
    return r
  comp = {k: v.cpu().numpy() / sd for k, v in ac.comp.items()}
  bnd = ac.bound.cpu().numpy() / sd
  ex, ctrl = comp['nom'][0], comp['nom'][1:]
  r['comp'] = ex
  r['bound'] = bnd[0]
  r['dM'] = float(ac.dM[0]) / sd
  r['dR_c'] = float(ex[0::2].sum())
  r['ctrl_mean'] = ctrl.mean(0) if len(ctrl) else None
  c2 = [i + 1 for i, k in enumerate(e['C1']) if k in e['C2']]
  r['c2_mean'] = comp['nom'][c2].mean(0) if c2 else None
  r['zero_within_bound'] = int(abs(ex[0] + ex[1]) <= bnd[0, 0] + bnd[0, 1])
  # size-matched standardization of the write categories (margin), and P_c on c
  c = e['c']
  mc, mP = ac.mc.cpu().numpy() / sd, ac.mP.cpu().numpy() / sd
  mm = mc - mP
  rel = set(np.flatnonzero(ctx.rel[c]).tolist())
  excl = {c} | set(e['P']) | set(e['Ent']) | rel
  pool = np.array([k for k in range(ctx.K) if k not in excl], np.int64)
  sets = {'own': [c], 'P': e['P'], 'Ent': e['Ent'], 'rel': sorted(rel - set(e['P']) - set(e['Ent']) - {c})}
  z, pct, miss, ncls = {}, {}, {}, {}
  P_q05 = None
  for name, S in sets.items():
    if not S:
      continue
    S = np.asarray(S)
    rs = matched_sets(ctx.rng, pool, ctx.n_c, len(S), ctx.n_c[S].sum(), prm['n_rand'], prm['count_tol'])
    nu = len(np.unique(rs))
    if len(rs) < 20 or nu < 3:                        # recorded, not skipped silently (e.g. 'own' of a rare class)
      miss[name] = 'few_sets' if len(rs) < 20 else 'few_classes'
      continue
    null = mm[rs].sum(1)
    if null.std() <= 1e-12 * np.abs(null).max():
      miss[name] = 'no_spread'
      continue
    ncls[name] = int(nu)
    val = mm[S].sum()
    z[name] = float((val - null.mean()) / max(null.std(), 1e-30))
    pct[name] = pct_of(val, null)
    if name == 'P':
      P_q05 = float(np.percentile(mc[rs].sum(1), 5))
  r['z'], r['pct'], r['null_missing'], r['null_classes'] = z, pct, miss, ncls
  if z and not miss:                                  # 'most negative' only among exits with every applicable null
    r['most_negative'] = min(z, key=z.get)
  thr = {'P_q05': P_q05, 'rand_band': None, 'rand_q05': None, 'band_min': prm['band_min']}
  if ac.mRc is not None and ctx.n_r is not None:
    mR = (ac.mRc - ac.mRP).cpu().numpy() / sd
    scaled = mR * ctx.n_c[c] / ctx.n_r
    thr['rand_band'] = tuple(np.percentile(scaled, [5, 95]))
    thr['rand_q05'] = thr['rand_band'][0]
    r['own_pct_R'] = pct_of(ex[0] + ex[1], scaled)
    poolr = np.arange(ctx.K)
    for name, S in sets.items():
      if name == 'own' or not S:
        continue
      rs = matched_sets(ctx.rng, poolr, ctx.n_r, len(S), ctx.n_c[np.asarray(S)].sum(), prm['n_rand'], prm['count_tol'])
      if len(rs) >= 20 and len(np.unique(rs)) >= 3:
        null = mR[rs].sum(1)
        if null.std() > 1e-12 * np.abs(null).max():
          r.setdefault('z_R', {})[name] = float((mm[np.asarray(S)].sum() - null.mean()) / null.std())
  r['label'] = classify(ex, ctrl, thr)
  flips = {}
  for var in ('wf', 'if', 'p2', 'gplus'):
    if var in comp:
      flips[var] = classify(comp[var][0], comp[var][1:], thr) != r['label']
  corners = []
  for pat in CORNERS:                                 # the exit one way, its controls the other (each its own bound)
    sg = corner_sign(pat)
    exc = corner(ex, bnd[0], sg)
    ctc = corner(ctrl, bnd[1:], -sg) if len(ctrl) else ctrl
    assert abs(exc.sum() - ex.sum()) <= 1e-9 * (np.abs(ex).sum() + bnd[0].sum()) + 1e-300, 'corner moved the total'
    corners.append(classify(exc, ctc, thr))
  flips['fp16'] = any(cl != r['label'] for cl in corners)
  r['flips'] = flips
  return r


def entry_summary(ctx: Ctx, ac: EventAcc) -> dict:
  """Per-entry record: components of the rise (normalized by sd_n(t_first) on the measurement half), the control mean,
  their storage bounds, and whether the own term is within its bound. sd floor on the selection half."""
  e = ac.e
  s_, m_ = DIRS[ac.d]
  t = ctx.run.g16[e['first']]
  sd = float(ctx.sd(t, e['b'])[m_][e['n']])
  r = {'d': ac.d, 'b': e['b'], 'n': e['n'], 'c': e['c'], 'dropped': e['dropped'], 'nctrl': len(e['ctrl']),
       'late': not e['b'].startswith(('stage0', 'stage1')), 'phase': phase_of(t), 'check': ac.check}
  if float(ctx.sd(t, e['b'])[s_][e['n']]) < ctx.floor(t, s_):
    r['skip'] = 'low_sd'
    return r
  if not sd > 0:
    r['skip'] = 'zero_sd'
    return r
  comp = ac.comp['nom'].cpu().numpy() / sd
  bnd = ac.bound.cpu().numpy() / sd
  r['comp'] = comp[0]
  r['bound'] = bnd[0]
  r['zero_within_bound'] = int(abs(comp[0, 0]) <= bnd[0, 0])
  r['ctrl_mean'] = comp[1:].mean(0) if len(comp) > 1 else None
  r['bound_ctrl_mean'] = bnd[1:].mean(0) if len(comp) > 1 else None   # corner is linear: corner of the means
  return r


def share_table(recs: list, which: str, ctx: Ctx, fine: bool = True) -> dict:
  """Excess shares over events (ratio of sums) of the X-level (and with fine, the 20) components, with two-way
  bootstrap CIs. which: 'C1' (exit minus mean C1), 'C2', or 'abs_fall' (the exit's own fall dR_c < 0, components on
  c)."""
  rows = []
  for r in recs:
    if 'comp' not in r:
      continue
    if which == 'C1' and r.get('ctrl_mean') is not None:
      rows.append((r['comp'] - r['ctrl_mean'], r))
    elif which == 'C2' and r.get('c2_mean') is not None:
      rows.append((r['comp'] - r['c2_mean'], r))
    elif which == 'abs_fall' and r['dR_c'] < 0:
      v = np.zeros(len(COMP))
      v[0::2] = r['comp'][0::2]
      rows.append((v, r))
  if not rows:
    return {'n': 0}
  D = np.stack([x for x, _ in rows])
  units = np.array([uid(r['b'], r['n']) for _, r in rows])
  classes = np.array([r['c'] for _, r in rows])
  den = D.sum(1)
  rng = np.random.default_rng(ctx.prm['seed'] + 1)
  xs = boot_ratio(xlevel(D), den, units, classes, ctx.prm['n_boot'], rng)
  fn = boot_ratio(D, den, units, classes, ctx.prm['n_boot'], rng) if fine else None
  out = {'n': len(rows), 'total_mean': float(den.mean()), 'units': int(len(set(units.tolist())))}
  if xs.get('est') is not None:
    out['shares'] = {k: {'est': xs['est'][i], 'ci': [xs['ci'][0][i], xs['ci'][1][i]], 'p0': xs['p0'][i]}
                     for i, k in enumerate(XCAT)}
    if fn is not None:
      out['fine'] = {k: {'est': fn['est'][i], 'ci': [fn['ci'][0][i], fn['ci'][1][i]]} for i, k in enumerate(COMP)}
  return out


def x1_population(ctx: Ctx, recs: list) -> dict:
  ok = [r for r in recs if 'comp' in r]
  out = {'n': len(recs), 'n_measured': len(ok), 'with_C1': sum(r['nC1'] > 0 for r in ok),
         'with_C2': sum(r['nC2'] > 0 for r in ok)}
  out['excess_vs_C1'] = share_table(ok, 'C1', ctx)
  out['excess_vs_C2'] = share_table(ok, 'C2', ctx)
  out['abs_fall'] = share_table(ok, 'abs_fall', ctx)
  lab = [r['label'] for r in ok]
  out['labels'] = {k: lab.count(k) for k in LABELS}
  cats = ('own', 'P', 'Ent', 'rel')
  out['labels_P_null_missing'] = sum('P' in r.get('null_missing', {}) for r in ok)   # 'interference' untestable
  out['null_missing'] = {k: sum(k in r.get('null_missing', {}) for r in ok) for k in cats}
  out['null_missing_reasons'] = {why: sum(why in r.get('null_missing', {}).values() for r in ok)
                                 for why in ('few_sets', 'few_classes', 'no_spread')}
  out['n_with_z'] = {k: sum(k in r.get('z', {}) for r in ok) for k in cats}
  out['null_few_classes'] = {k: sum(r.get('null_classes', {}).get(k, 10) < 10 for r in ok) for k in cats}
  cls = [r for r in ok if r['label'] not in UNLABELLED]
  if cls:
    units = np.array([uid(r['b'], r['n']) for r in cls])
    classes = np.array([r['c'] for r in cls])
    rng = np.random.default_rng(ctx.prm['seed'] + 2)
    out['label_props'] = {k: boot_mean(np.array([r['label'] == k for r in cls], float), units, classes,
                                       ctx.prm['n_boot'], rng) for k in LABELS[:5]}
    out['flips'] = {v: int(sum(r['flips'].get(v, False) for r in cls)) for v in ('fp16', 'wf', 'if', 'p2', 'gplus')}
    out['flips_n'] = {v: int(sum(v in r['flips'] for r in cls)) for v in ('fp16', 'wf', 'if', 'p2', 'gplus')}
  mn = [r.get('most_negative') for r in ok if r.get('most_negative')]
  out['most_negative_standardized'] = {k: mn.count(k) for k in ('own', 'P', 'Ent', 'rel')}
  out['most_negative_excluded'] = sum(bool(r.get('null_missing')) for r in ok)          # a null missing: not ranked
  for k in ('own', 'P', 'Ent', 'rel'):
    zs = [r['z'][k] for r in ok if k in r.get('z', {})]
    if zs:
      out.setdefault('z_median', {})[k] = float(np.median(zs))
  pr = [r['own_pct_R'] for r in ok if 'own_pct_R' in r]
  if pr:
    out['own_pct_R_median'] = float(np.median(pr))
  out['zero_within_bound_own'] = int(sum(r.get('zero_within_bound', 0) for r in ok))
  out['max_check'] = float(max([r['check'] for r in recs], default=0.0))
  return out


def x2_population(ctx: Ctx, recs: list) -> dict:
  """Own excess-rise share of dropped and persistent entries and their difference (two-way bootstrap, joint draws), the
  entries whose own term is within its fp16 bound, and the difference at the four corners of the bounds (corner)."""
  ok = [r for r in recs if 'comp' in r and r.get('ctrl_mean') is not None]
  out = {'n': len(recs), 'n_with_controls': len(ok),
         'n_low_sd': sum(r.get('skip') in ('low_sd', 'zero_sd') for r in recs)}
  if not ok:
    return out
  D = xlevel(np.stack([r['comp'] - r['ctrl_mean'] for r in ok]))
  den = D.sum(1)
  drop = np.array([r['dropped'] for r in ok])
  out['zero_within_bound_own'] = {'dropped': int(sum(r.get('zero_within_bound', 0) for r in ok if r['dropped'])),
                                  'persistent': int(sum(r.get('zero_within_bound', 0) for r in ok if not r['dropped']))}
  units = np.array([uid(r['b'], r['n']) for r in ok])
  classes = np.array([r['c'] for r in ok])
  est = {}
  for name, m in (('dropped', drop), ('persistent', ~drop)):
    if m.any() and den[m].sum() != 0:
      est[name] = {k: float(D[m, i].sum() / den[m].sum()) for i, k in enumerate(XCAT)}
  out['shares'] = est
  if 'dropped' in est and 'persistent' in est:
    rng = np.random.default_rng(ctx.prm['seed'] + 3)
    draws = []
    for W in twoway_weights(units, classes, ctx.prm['n_boot'], rng):
      with np.errstate(invalid='ignore', divide='ignore'):
        a_ = (W[:, drop] @ D[drop, 0]) / (W[:, drop] @ den[drop])
        b_ = (W[:, ~drop] @ D[~drop, 0]) / (W[:, ~drop] @ den[~drop])
      draws.append(a_ - b_)
    draws = np.concatenate(draws)
    dfin = draws[~np.isnan(draws)]
    diff = est['dropped']['own'] - est['persistent']['own']
    p, B = boot_p0(dfin)
    out['own_diff'] = {'est': diff, 'ci': np.percentile(dfin, [2.5, 97.5]).tolist() if len(dfin) else None,
                       'p0': float(p), 'B': int(B)}
    if all(r.get('bound') is not None and r.get('bound_ctrl_mean') is not None for r in ok):
      cor = {}
      for pat in CORNERS:                     # the sum of the components (den) is fixed at every corner
        sg = corner_sign(pat)
        Dc = xlevel(np.stack([corner(r['comp'], r['bound'], sg) - corner(r['ctrl_mean'], r['bound_ctrl_mean'], -sg)
                              for r in ok]))
        cor[pat] = float(Dc[drop, 0].sum() / den[drop].sum() - Dc[~drop, 0].sum() / den[~drop].sum())
      out['own_diff']['fp16_corners'] = cor
  return out


# --------------------------------------------------------------------------------------------------------- X3

def bias_pushes(ctx: Ctx, b: str) -> dict:
  """The teaching-signal record of one block from the bias parts (P5): per G16 interval j and map, push[(j, M)] (K, U) =
  Cb_t2 - rhob Cb_t1 (pushR for the random split), med[(j, M)] (U,) the median |push| over classes in each unit, and
  sign[j][M] (2, U, C) the orientation at t1 per half (value: sign of Bbar, gate: sign of Abar); 'oriented'[(j, h)]
  (U, C): the oriented, normalized push of every class in every unit, the mean of the two maps (the probe-free capture
  of X4). pushb[(j, M)] (pushRb): the fp16 storage bound of each push, half_ulp(Cb_t2) / scale_t2 + |rhob|
  half_ulp(Cb_t1) / scale_t1 (the stored fp16 parts and their per-tensor scales)."""
  run, g = ctx.run, ctx.run.g16
  def cb(s, split):
    A = run.attr(s, split)
    out = {}
    for M in MAPS:
      r = A[f'{b}.pw_{M}']
      sc = float(r.get('Cb_scale', 1.0))
      out[M] = (r['Cb'].double() / sc, float(r['ab']), half_ulp(r['Cb']) / sc)
    return out
  CB = {s: cb(s, 'class') for s in g}
  CBR = {s: cb(s, 'random') for s in g} if run.has_random else None
  out = {'push': {}, 'pushR': {} if CBR is not None else None, 'pushb': {}, 'pushRb': {} if CBR is not None else None,
         'med': {}, 'sign': {}, 'oriented': {}}
  for j in range(1, len(g)):
    st = ctx.stock[g[j - 1]][b]
    out['sign'][j] = {'gate': np.where(st['Abar'].numpy() >= 0, 1.0, -1.0),
                      'value': np.where(st['Bbar'].numpy() >= 0, 1.0, -1.0)}
    for M in MAPS:
      (c1, ab1, u1), (c2, ab2, u2) = CB[g[j - 1]][M], CB[g[j]][M]
      out['push'][(j, M)] = (c2 - (ab2 / ab1) * c1).numpy()
      out['pushb'][(j, M)] = (u2 + abs(ab2 / ab1) * u1).numpy()
      out['med'][(j, M)] = np.maximum(np.median(np.abs(out['push'][(j, M)]), 0), 1e-30)
      if CBR is not None:
        (r1, rb1, v1), (r2, rb2, v2) = CBR[g[j - 1]][M], CBR[g[j]][M]
        out['pushR'][(j, M)] = (r2 - (rb2 / rb1) * r1).numpy()
        out['pushRb'][(j, M)] = (v2 + abs(rb2 / rb1) * v1).numpy()
    for h in (0, 1):
      out['oriented'][(j, h)] = sum(0.5 * out['push'][(j, M)][:ctx.C].T * out['sign'][j][M][h]
                                    / out['med'][(j, M)][:, None] for M in MAPS)
  return out


def s1_label(v: float, lo: float, hi: float) -> str:
  return 'withdrawn' if lo <= v <= hi else ('reversed' if v < lo else 'maintained')


def s1_c1_band(pe: float, pc: list, band_min: int = 3, be: float = 0.0, bc: list = None):
  """S1 against the C1 band. C1 member i and the exit are both labelled against band_i, the 25-75% of the other m - 1
  C1 pushes, so that the pair (exit, member i) faces the same band and, when the exit and its C1 classes are
  exchangeable, the exit's label fractions over the m bands and the C1 labels have the same expectation for any
  distribution and any m (the 25-75% of all m values would cover more for the exit than the members' bands of m - 1).
  None for m < max(3, band_min). be, bc: fp16 bounds of the pushes; flip_fp16 = some label changes at push +- bound
  (the bands held fixed)."""
  m = len(pc)
  if m < max(3, band_min):
    return None
  pc = np.asarray(pc, float)
  loo = [np.percentile(np.delete(pc, i), [25, 75]) for i in range(m)]
  lab_e = [s1_label(pe, *bd) for bd in loo]
  lab_c = [s1_label(v, *bd) for v, bd in zip(pc, loo)]
  bc = np.zeros(m) if bc is None else np.asarray(bc, float)
  flip = any([s1_label(pe + s * be, *bd) for bd in loo] != lab_e for s in (-1, 1)) or \
      any([s1_label(v + s * bv, *bd) for v, bv, bd in zip(pc, bc, loo)] != lab_c for s in (-1, 1))
  return {'labels_exit': lab_e, 'labels_C1': lab_c, 'label_frac': {k: float(np.mean([x == k for x in lab_e]))
                                                                    for k in S1}, 'flip_fp16': bool(flip)}


def x3_block(ctx: Ctx, b: str, bp: dict) -> list:
  """S1 and S3 per exit of one block (both directions): oriented, normalized bias pushes over the last 3 G16 intervals
  up to t_out with their fp16 bounds, the S1 labels against the C1 band (s1_c1_band; every run) and, with a random
  split, against the random band (label_rand, an extra result), and the crowding split at snapshots. Exits of units
  below the sd floor on the selection half at t_pk are recorded with skip. bp: bias_pushes(ctx, b)."""
  run, g = ctx.run, ctx.run.g16
  push, pushR, med, sign = bp['push'], bp['pushR'], bp['med'], bp['sign']
  pushb = bp['pushb']
  snaps = run.snaps
  out = []
  for d, (s_, m_) in DIRS.items():
    for e in ctx.events['directions'][d]['exits']:
      if e['b'] != b:
        continue
      n, c = e['n'], e['c']
      r = {'d': d, 'b': b, 'n': n, 'c': c, 'anchored': e['anchored'], 'rep': e['rep'],
           'late': not b.startswith(('stage0', 'stage1')), 'nC1': len(e['C1'])}
      t_pk = g[e['pk']]
      if float(ctx.sd(t_pk, b)[s_][n]) < ctx.floor(t_pk, s_):
        r['skip'] = 'low_sd'
        out.append(r)
        continue
      js = [j for j in (e['out'] - 2, e['out'] - 1, e['out']) if j >= 1]
      def p_of(x):
        return sum(0.5 * sum(push[(j, M)][x, n] * sign[j][M][s_, n, x] / med[(j, M)][n] for M in MAPS) for j in js)
      def pb_of(x):                                   # |sign| = 1: the bounds add up over maps and intervals
        return sum(0.5 * sum(pushb[(j, M)][x, n] / med[(j, M)][n] for M in MAPS) for j in js)
      def band_of(x):
        v = sum(0.5 * sum(pushR[(j, M)][:, n] * sign[j][M][s_, n, x] / med[(j, M)][n] for M in MAPS) for j in js)
        return v * ctx.n_c[x] / ctx.n_r
      pe, be = p_of(c), pb_of(c)
      pc, bc = [p_of(k) for k in e['C1']], [pb_of(k) for k in e['C1']]
      r |= {'push': pe, 'push_C1': pc, 'push_bound': be, 'push_bound_C1': bc}
      lb = s1_c1_band(pe, pc, ctx.prm['band_min'], be, bc)
      if lb is not None:
        r |= lb
        r['label'] = s1_label(pe, *np.percentile(pc, [25, 75]))     # descriptive only: the tests use label_frac
        r['band'] = 'C1'
      if pushR is not None:
        bands = [np.percentile(band_of(x), [5, 95]) for x in [c] + e['C1']]
        r['label_rand'] = s1_label(pe, *bands[0])
        r['labels_C1_rand'] = [s1_label(v, *bd) for v, bd in zip(pc, bands[1:])]
        r['flip_fp16_rand'] = any(s1_label(v + s * bv, *bd) != s1_label(v, *bd) for v, bv, bd in
                                  zip([pe] + pc, [be] + bc, bands) for s in (-1, 1))
      # S3: crowding between the first and last snapshot inside [t_pk, t_out]
      inside = [s for s in snaps if g[e['pk']] <= s <= g[e['out']]]
      if len(inside) >= 2:
        r['S3'] = crowding(ctx, b, n, [c] + e['C1'], inside[0], inside[-1], m_)
      out.append(r)
  return out


def bias_rate_parts(r: dict, n: int, xs: list, lr_of, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> dict:
  """The factors of the AdamW bias write of the classes xs on unit n in update t, the update that brought the step
  count to t (r: a full attribution state at t): write = -lr * mb / c1 * Db with lr = lr_of(t - 1), mb = mb_t^(x)[n],
  c1 = 1 - b1^t and Db = 1 / (sqrt(vb_t[n] / (1 - b2^t)) + eps). Without bias decay this is Cb_t - Cb_(t-1)."""
  t = int(r['t'])
  c2 = 1 - b2 ** t
  return {'lr': lr_of(t - 1), 'mb': r['mb'][xs, n].double().numpy(), 'c1': 1 - b1 ** t,
          'Db': float(1 / (math.sqrt(float(r['vb'][n]) / c2) + eps))}


def crowding(ctx: Ctx, b: str, n: int, xs: list, sa: int, sb: int, h: int) -> dict:
  """log |write rate| change of each class x in unit n from snapshot sa to sb. The AdamW bias write of update t (the
  one that brought the step count to t, the state a snapshot at t holds) is lr_of(t - 1) mb_t^(x) / c1_t Db_t, so the
  change splits exactly into the learning rate (dlr, shared by every class and unit), the second moment (dD, crowding,
  shared by the unit's classes), the class's momentum (dm) and the bias correction (dc1): total = dlr + dD + dm + dc1.
  Weights: the same with the median D over the unit's weights and the momentum projected on Hs(x) of the measurement
  half (approximate)."""
  run = ctx.run
  res = {'sa': sa, 'sb': sb}
  muon = run.cfg.get('opt') == 'muon'
  lr_of = lr_schedule(run.cfg)                          # Muon: the biases' AdamW lr has the same schedule (ratio)
  for M in MAPS:
    vals = {}
    for s in (sa, sb):
      r = run.full(s)['attr'][f'{b}.pw_{M}']
      rec = bias_rate_parts(r, n, xs, lr_of)
      c2 = 1 - 0.999 ** int(r['t'])
      if not muon:
        Dw = 1 / ((r['v'][n].double() / c2).sqrt() + 1e-8)
        hs = ctx.stockh(s)
        Hs = hs[b][h][n, xs].double() if hs is not None else None
        rec['Dmed'] = float(Dw.median())
        rec['proj'] = (r['m'][xs, n].double() * Hs).sum(-1).numpy() if Hs is not None else None
      vals[s] = rec
    a, z = vals[sa], vals[sb]
    with np.errstate(divide='ignore', invalid='ignore'):
      dm = np.log(np.abs(z['mb'])) - np.log(np.abs(a['mb']))
    dD, dc1 = math.log(z['Db'] / a['Db']), math.log(a['c1'] / z['c1'])
    dlr = math.log(z['lr'] / a['lr']) if a['lr'] > 0 and z['lr'] > 0 else float('nan')
    res[f'bias_{M}'] = {'total': (dlr + dD + dm + dc1).tolist(), 'dlr': dlr, 'dD': dD, 'dm': dm.tolist(), 'dc1': dc1}
    if not muon and a.get('proj') is not None and z.get('proj') is not None:
      with np.errstate(divide='ignore', invalid='ignore'):
        dp = np.log(np.abs(z['proj'])) - np.log(np.abs(a['proj']))
      dDw = math.log(z['Dmed'] / a['Dmed'])
      res[f'weight_{M}'] = {'total': (dlr + dDw + dp + dc1).tolist(), 'dlr': dlr, 'dD': dDw, 'dm': dp.tolist()}
  return res


def x3_population(ctx: Ctx, recs: list) -> dict:
  """S1 proportions of exits and C1 and their difference (two-way bootstrap): the C1 band in every run (the exit's
  leave-one-out label fractions, s1_c1_band), and with a random split also against the random band ('<label>_randband',
  an extra result); fp16 flips, the C1 size distribution and the exits skipped below the sd floor; S3 means."""
  ok = [r for r in recs if r.get('label_frac') and r.get('labels_C1')]
  okr = [r for r in recs if 'label_rand' in r and r.get('labels_C1_rand')]
  m, cnt = np.unique([r['nC1'] for r in recs if 'skip' not in r], return_counts=True)
  out = {'n': len(recs), 'n_labelled': len(ok), 'band': 'C1', 'n_low_sd': sum(r.get('skip') == 'low_sd' for r in recs),
         'nC1': {int(a): int(b) for a, b in zip(m, cnt)}}
  rng = np.random.default_rng(ctx.prm['seed'] + 4)
  for rows, sfx, exf, c1f in ((ok, '', lambda r, k: r['label_frac'][k], 'labels_C1'),
                              (okr, '_randband', lambda r, k: float(r['label_rand'] == k), 'labels_C1_rand')):
    if not rows:
      continue
    units = np.array([uid(r['b'], r['n']) for r in rows])
    classes = np.array([r['c'] for r in rows])
    for k in S1:
      ex = np.array([exf(r, k) for r in rows], float)
      c1 = np.array([np.mean([x == k for x in r[c1f]]) for r in rows])
      out[k + sfx] = {'exits': float(ex.mean()), 'C1': float(c1.mean()),
                      'diff': boot_mean(ex - c1, units, classes, ctx.prm['n_boot'], rng)}
    flip = 'flip_fp16' if not sfx else 'flip_fp16_rand'
    out['flips' + sfx] = {'fp16': int(sum(r.get(flip, False) for r in rows)), 'n': len(rows)}
  out['n_labelled_randband'] = len(okr)
  s3 = [r['S3'] for r in recs if 'S3' in r]
  if s3:
    agg = {}
    for M in MAPS:
      for kind in ('bias', 'weight'):
        key = f'{kind}_{M}'
        have = [x[key] for x in s3 if key in x]
        if not have:
          continue
        tot_e = [h['total'][0] for h in have if np.isfinite(h['total'][0])]
        dm_e = [h['dm'][0] for h in have if np.isfinite(h['dm'][0])]
        dD = [h['dD'] for h in have]
        tot_c = [np.nanmean([v for v in h['total'][1:] if np.isfinite(v)]) for h in have if len(h['total']) > 1]
        agg[key] = {'n': len(have), 'exit_total_mean': float(np.mean(tot_e)) if tot_e else None,
                    'exit_dm_mean': float(np.mean(dm_e)) if dm_e else None, 'dD_mean': float(np.mean(dD)),
                    'C1_total_mean': float(np.nanmean(tot_c)) if tot_c else None}
    out['S3'] = agg
  return out


def x3_s2(ctx: Ctx, own: dict) -> dict:
  """S2: n's share s_n(j) = O+_n,c(j) / sum_{m in stage} O+_m,c(j) of c's own-write capture; j_x = first G16 interval
  after t_pk with s_n(j) <= s_n(pk) / 2; timing before / same / after t_out, against a within-unit permutation of t_out
  and against C1 classes; carriers' share (the top-3 other units of each interval) at t_pk and t_out. Units below the
  network-wide sd floor at t1 (either half) get O = 0 in that interval (they would dominate the stage total), a unit
  below it at a later interval is missing there (not a halving), and exits of units below it on the selection half at
  t_pk are skipped (n_low_sd)."""
  run, g = ctx.run, ctx.run.g16
  res = {}
  for d, (s_, m_) in DIRS.items():
    rows, n_low = [], 0
    for stage in sorted({b.split('.')[0] for b in run.blocks}):
      blocks = [b for b in run.blocks if b.startswith(stage + '.')]
      off = np.cumsum([0] + [run.U[b] for b in blocks])
      O, valid = {}, {}
      for j in range(1, len(g)):
        t1 = g[j - 1]
        sd = np.concatenate([ctx.sd(t1, b)[m_] for b in blocks])
        sds = np.concatenate([ctx.sd(t1, b)[s_] for b in blocks])
        valid[j] = (sds >= ctx.floor(t1, s_)) & (sd >= ctx.floor(t1, m_))
        v = np.concatenate([own[b][(t1, g[j], m_)].numpy() for b in blocks]) / np.maximum(sd, 1e-30)[:, None]
        v[~valid[j]] = 0
        O[j] = np.maximum(v, 0)
      tot = {j: O[j].sum(0) for j in O}
      exits = [e for e in ctx.events['directions'][d]['exits'] if e['b'] in blocks]
      by_unit = {}
      for i, e in enumerate(exits):
        by_unit.setdefault((e['b'], e['n']), []).append((i, e['out']))
      def jx(gi, c, pk):
        if pk < 1 or tot[pk][c] <= 0 or O[pk][gi, c] <= 0:
          return None
        base = O[pk][gi, c] / tot[pk][c]
        for j in range(pk + 1, len(g)):
          if valid[j][gi] and tot[j][c] > 0 and O[j][gi, c] / tot[j][c] <= base / 2:
            return j
        return len(g)
      def when(x, out):
        return 'before' if x < out else ('same' if x == out else 'after')
      for i, e in enumerate(exits):
        gi = int(off[blocks.index(e['b'])] + e['n'])
        t_pk = g[e['pk']]
        if float(ctx.sd(t_pk, e['b'])[s_][e['n']]) < ctx.floor(t_pk, s_):
          n_low += 1
          continue
        x = jx(gi, e['c'], e['pk'])
        if x is None:
          continue
        # the null reassigns the t_out of the unit's other exits (leave this exit out, by identity: exits that share
        # its t_out stay in, or the null could never say 'same' where t_outs coincide, as they often do on G16)
        perm = [when(x, o) for k, o in by_unit[(e['b'], e['n'])] if k != i]
        ctl = [when(xc, e['out']) for xc in (jx(gi, k, e['pk']) for k in e['C1']) if xc is not None]
        carriers = {}
        for name, j in (('pk', e['pk']), ('out', e['out'])):
          if j >= 1 and tot[j][e['c']] > 0:
            col = O[j][:, e['c']].copy()
            col[gi] = -np.inf
            carriers[name] = float(np.sort(col)[-ctx.prm['carriers']:].clip(min=0).sum() / tot[j][e['c']])
        rows.append({'b': e['b'], 'n': e['n'], 'c': e['c'], 'anchored': e['anchored'], 'rep': e['rep'],
                     'late': stage not in ('stage0', 'stage1'), 'when': when(x, e['out']), 'perm': perm, 'C1': ctl,
                     'carriers': carriers})
    pop = [r for r in rows if r['anchored'] and r['late']]
    out = {'n_all': len(rows), 'n': len(pop), 'n_low_sd': n_low}
    if pop:
      units = np.array([uid(r['b'], r['n']) for r in pop])
      classes = np.array([r['c'] for r in pop])
      rng = np.random.default_rng(ctx.prm['seed'] + 5)
      for k in ('before', 'same', 'after'):
        ex = np.array([r['when'] == k for r in pop], float)
        pn = np.array([np.mean([w == k for w in r['perm']]) if r['perm'] else np.nan for r in pop])
        cn = np.array([np.mean([w == k for w in r['C1']]) if r['C1'] else np.nan for r in pop])
        out[k] = {'exits': float(ex.mean()), 'perm_null': float(np.nanmean(pn)) if np.isfinite(pn).any() else None,
                  'C1_null': float(np.nanmean(cn)) if np.isfinite(cn).any() else None,
                  'minus_perm': boot_mean(ex - pn, units, classes, ctx.prm['n_boot'], rng),
                  'minus_C1': boot_mean(ex - cn, units, classes, ctx.prm['n_boot'], rng)}
      for name in ('pk', 'out'):
        v = [r['carriers'][name] for r in pop if name in r['carriers']]
        if v:
          out[f'carrier_share_{name}'] = float(np.mean(v))
    res[d] = out
  return res


# --------------------------------------------------------------------------------------------------------- X4

def rank_cols(X: np.ndarray) -> np.ndarray:
  """Mid-ranks along axis -2 (units) of (..., n, p), standardized to mean 0, sd 1."""
  r = midrank(np.moveaxis(X, -2, -1))
  r = np.moveaxis(r, -1, -2)
  r = r - r.mean(-2, keepdims=True)
  return r / np.maximum(r.std(-2, keepdims=True), 1e-12)


def batched_ols(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
  """Per batch: least squares of Y (B, n) on X (B, n, p) with an intercept (columns standardized beforehand); (B, p)."""
  Xi = np.concatenate([np.ones(X.shape[:-1] + (1,)), X], -1)
  sol = torch.linalg.lstsq(torch.from_numpy(Xi), torch.from_numpy(Y[..., None])).solution[..., 0].numpy()
  return sol[:, 1:]


def rank_ols(Y: np.ndarray, X: np.ndarray) -> np.ndarray:
  """Rank regression per batch: Y (B, n) on rank_cols(X) (B, n, p) (batched_ols). A coefficient on a column without
  spread across the n units (e.g. the class parts' provenance at step 0, all 0) is not estimable: nan, not the
  minimum-norm 0 that would enter a median as an estimate."""
  Xr = rank_cols(X)
  coef = batched_ols(Y, Xr)
  coef[Xr.std(-2) < 0.5] = np.nan                      # ranked columns have sd 1, columns without spread sd 0
  return coef


def x4(ctx: Ctx, own: dict, plast: dict, prov: dict, pbias: dict, own_bound: dict = None) -> dict:
  """F1, F2, F3 and the ladder per direction, stage (and network) scope and phase (module docstring); F1_bias: F1
  with the probe-free capture, the oriented normalized bias push of class k in each unit (bias_pushes). With own_bound
  (the fp16 bounds of O, analyze_block): the (unit, class) cells with |O| within the bound, and F1 recomputed with O_L
  at minus and plus its bound (F1_lo, F1_hi; medians)."""
  run, g, prm = ctx.run, ctx.run.g16, ctx.prm
  off = prm['log_offset']
  res = {}
  stages = sorted({b.split('.')[0] for b in run.blocks})
  for d, (s_, m_) in DIRS.items():
    rows = {'stage': [], 'network': []}
    within = {}
    for j in range(1, len(g)):
      t1 = g[j - 1]
      R = np.concatenate([ctx.R[t1][b][s_] for b in run.blocks])                       # (N, C)
      sd_s = np.concatenate([ctx.sd(t1, b)[s_] for b in run.blocks])
      sd_m = np.concatenate([ctx.sd(t1, b)[m_] for b in run.blocks])
      valid = (sd_s >= ctx.floor(t1, s_)) & (sd_m >= ctx.floor(t1, m_))
      z = (R - R.mean(1, keepdims=True)) / np.maximum(sd_s, 1e-30)[:, None]
      O = np.concatenate([own[b][(t1, g[j], m_)].numpy() for b in run.blocks]) / np.maximum(sd_m, 1e-30)[:, None]
      Pl = np.concatenate([plast[b][(j, m_)].numpy() for b in run.blocks]) / np.maximum(sd_m, 1e-30)[:, None]
      pv = np.concatenate([prov[b][j][s_].numpy() for b in run.blocks])                 # (N, C, 6)
      Ob = np.concatenate([pbias[b][(j, s_)] for b in run.blocks])                      # (N, C)
      Bo = None
      if own_bound is not None and all((t1, g[j], m_) in own_bound[b] for b in run.blocks):
        Bo = np.concatenate([own_bound[b][(t1, g[j], m_)].numpy() for b in run.blocks]) \
            / np.maximum(sd_m, 1e-30)[:, None]
      gs = ctx.gstate.get(t1)
      base = {'kappa': None, 'beta': None, 'mu': None}
      if gs is not None:
        for k in base:
          if all(k in gs['blocks'][b] for b in run.blocks):
            base[k] = np.concatenate([gs['blocks'][b][k][m_].numpy() for b in run.blocks])
      stage_of = np.concatenate([[b.split('.')[0]] * run.U[b] for b in run.blocks])
      scopes = [('stage', st, np.flatnonzero((stage_of == st) & valid)) for st in stages] + \
               [('network', 'all', np.flatnonzero(valid))]
      for scope, name, idx in scopes:
        if len(idx) < 8:
          continue
        zs, Os = z[idx], O[idx]                                                          # (n, C)
        L = zs.argmax(0)
        cls = np.arange(ctx.C)
        F1 = (midrank(Os.T)[cls, L] - 0.5) / len(idx)
        F1b = (midrank(Ob[idx].T)[cls, L] - 0.5) / len(idx)
        F1x = {}
        if Bo is not None:
          Bs = Bo[idx]
          wb = within.setdefault(scope, {'cells': 0, 'within': 0})
          wb['cells'] += int(Os.size)
          wb['within'] += int((np.abs(Os) <= Bs).sum())
          for nm, sgn in (('F1_lo', -1.0), ('F1_hi', 1.0)):
            Oa = Os.copy()
            Oa[L, cls] += sgn * Bs[L, cls]
            F1x[nm] = (midrank(Oa.T)[cls, L] - 0.5) / len(idx)
        Xr = np.stack([zs, zs ** 2] + [pv[idx][:, :, i] for i in range(4)], -1)          # (n, C, 6)
        Yr = rank_cols(Os.T[..., None])[..., 0]
        coef = rank_ols(Yr, np.moveaxis(Xr, 1, 0))                                       # (C, 6); nan at t1 = 0
        Xg = np.stack([zs, zs ** 2] + [pv[idx][:, :, i] for i in (4, 5, 1, 2, 3)], -1)
        coef_gv = rank_ols(Yr, np.moveaxis(Xg, 1, 0))
        pos = np.maximum(Os, 0)
        tot = pos.sum(0)
        piO = np.where(tot > 0, pos[L, cls] / np.maximum(tot, 1e-300), np.nan)
        psi = {'uniform': np.log(piO + off) - np.log(1 / len(idx) + off)}
        for k, arr in (('kappa', base['kappa']), ('beta', base['beta']), ('P', Pl), ('mu', base['mu'])):
          if arr is None:
            continue
          X_ = np.maximum(arr[idx], 0)
          tx = X_.sum(0)
          pix = np.where(tx > 0, X_[L, cls] / np.maximum(tx, 1e-300), np.nan)
          psi[k] = np.log(piO + off) - np.log(pix + off)
        for c in range(ctx.C):
          rows[scope].append({'j': j, 'phase': phase_of(t1), 'stage': name, 'c': c, 'F1': F1[c], 'F1_bias': F1b[c],
                              'F2': float(coef[c, 2]),
                              'F2_all': coef[c].tolist(), 'F2_gate_value': coef_gv[c, 2:4].tolist(),
                              **{k: float(v[c]) for k, v in F1x.items()},
                              **{f'psi_{k}': float(v[c]) for k, v in psi.items()}})
    out = {}
    rng = np.random.default_rng(prm['seed'] + 6)
    for scope, rr in rows.items():
      if not rr:
        continue
      cl = np.array([r['c'] for r in rr])
      key = {}
      for ph in (None, 0, 1, 2):
        sel = [i for i, r in enumerate(rr) if ph is None or r['phase'] == ph]
        if not sel:
          continue
        rec = {}
        for stat in ('F1', 'F1_bias', 'F2', 'psi_uniform', 'psi_kappa', 'psi_P', 'psi_beta', 'psi_mu'):
          v = np.array([rr[i].get(stat, np.nan) for i in sel], float)
          if np.isfinite(v).any():
            rec[stat] = boot_median(v, cl[sel], prm['n_boot'], rng)
        if 'F1_lo' in rr[sel[0]]:
          rec['F1_fp16'] = {k: float(np.nanmedian([rr[i].get(k, np.nan) for i in sel])) for k in ('F1_lo', 'F1_hi')}
        if scope == 'stage':
          for st in stages:
            ss = [i for i in sel if rr[i]['stage'] == st]
            if ss:
              rec[st] = {stat: float(np.nanmedian([rr[i].get(stat, np.nan) for i in ss]))
                         for stat in ('F1', 'F2', 'psi_beta', 'psi_kappa')}
        key['all' if ph is None else f'phase{ph}'] = rec
      out[scope] = key
    if out.get('stage', {}).get('all', {}).get('F1'):
      ci = out['stage']['all']['F1']['ci']
      out['F1_counts_against'] = bool(ci[1] < 0.55)
    out['own_within_bound'] = within
    res[d] = out
  return res


# --------------------------------------------------------------------------------------------------------- X5

def chains(intervals: list, min_len: int) -> list:
  """Maximal runs of consecutive intervals (t1, t2) (sorted, each starting where the last ended) of equal length >=
  min_len, with at least two intervals."""
  out, cur = [], []
  for iv in sorted(intervals):
    if cur and iv[0] == cur[-1][1] and iv[1] - iv[0] == cur[-1][1] - cur[-1][0]:
      cur.append(iv)
    else:
      if len(cur) >= 2:
        out.append(cur)
      cur = [iv]
  if len(cur) >= 2:
    out.append(cur)
  return [c for c in out if c[0][1] - c[0][0] >= min_len]


def loop_regression(Ys: list, Os: list, Zs: list, rng, n_perm: int) -> dict:
  """Rank regression (ranks within each interval) of the own gain in I_{j+1} on the own gain in I_j, the other-derived
  gain and z(t_j), pooled over chains (lists of (rows, J) arrays); null: each row's own-gain series permuted across
  its chain."""
  def ranks(A):
    r = midrank(A.T).T
    return (r - r.mean(0)) / np.maximum(r.std(0), 1e-12)
  rY = [ranks(Y) for Y in Ys]
  rO = [ranks(O) for O in Os]
  rZ = [ranks(Z) for Z in Zs]
  def fit(rYs):
    """OLS of y on (1, own, other, z) by the normal equations, summed over the chains' pairs."""
    XtX, Xty = np.zeros((4, 4)), np.zeros(4)
    for Y, O, Z in zip(rYs, rO, rZ):
      y = Y[:, 1:].ravel()
      X = np.stack([np.ones_like(y), Y[:, :-1].ravel(), O[:, :-1].ravel(), Z[:, :-1].ravel()], 1)
      XtX += X.T @ X
      Xty += X.T @ y
    return np.linalg.solve(XtX, Xty)[1:]
  def permuted(Y):
    J = Y.shape[1]
    if J <= 6:                                             # a random row of the table of all J! orders
      import itertools
      table = np.array(list(itertools.permutations(range(J))))
      idx = table[rng.integers(0, len(table), len(Y))]
    else:
      idx = np.argsort(rng.random(Y.shape), 1)
    return np.take_along_axis(Y, idx, 1)
  obs = fit(rY)
  null = []
  for _ in range(n_perm):
    null.append(fit([permuted(Y) for Y in rY]))
  null = np.array(null)
  return {'coef_own': float(obs[0]), 'coef_other': float(obs[1]), 'coef_z': float(obs[2]),
          'null_own_mean': float(null[:, 0].mean()), 'null_own_ci': np.percentile(null[:, 0], [2.5, 97.5]).tolist(),
          'p_own_greater': float(((null[:, 0] >= obs[0]).sum() + 1) / (n_perm + 1)),
          'null_other_mean': float(null[:, 1].mean()), 'rows': int(sum(len(Y) for Y in Ys))}


def x5(ctx: Ctx, own: dict, oth: dict, own_bound: dict = None) -> dict:
  """The loop regression over G16 and snapshot chains (rows: units above the sd floor on both halves at every t_j),
  the cells whose own gain is within its fp16 bound (own_within_bound; snapshot intervals have no bound), and the
  cumulative concentration of capture against kappa per stage (units below the floor at t1, or for kappa at t2, add
  nothing in that interval)."""
  run, prm = ctx.run, ctx.prm
  have = {(s1, s2) for (s1, s2, h) in next(iter(own.values()))}
  g16iv = [(run.g16[j - 1], run.g16[j]) for j in range(1, len(run.g16))]
  snapiv = list(zip(run.snaps[:-1], run.snaps[1:]))
  fams = {'g16': chains([iv for iv in g16iv if iv in have], prm['x5_min_len']),
          'snap': chains([iv for iv in snapiv if iv in have], prm['x5_min_len'])}
  res = {'chains': {k: [[list(iv) for iv in ch] for ch in v] for k, v in fams.items()}}
  rng = np.random.default_rng(prm['seed'] + 7)
  for d, (s_, m_) in DIRS.items():
    Ys, Os, Zs = [], [], []
    wb = {'cells': 0, 'within': 0, 'no_bound': 0}
    for fam, chs in fams.items():
      for ch in chs:
        cols_y, cols_o, cols_z, cols_b, valid = [], [], [], [], None
        for (t1, t2) in ch:
          if t1 not in ctx.R:
            break
          sd_m = np.concatenate([ctx.sd(t1, b)[m_] for b in run.blocks])
          sd_s = np.concatenate([ctx.sd(t1, b)[s_] for b in run.blocks])
          R = np.concatenate([ctx.R[t1][b][s_] for b in run.blocks])
          ok = (sd_m >= ctx.floor(t1, m_)) & (sd_s >= ctx.floor(t1, s_))
          valid = ok if valid is None else valid & ok
          scale = np.maximum(sd_m, 1e-30)[:, None]
          cols_y.append(np.concatenate([own[b][(t1, t2, m_)].numpy() for b in run.blocks]) / scale)
          cols_o.append(np.concatenate([oth[b][(t1, t2, m_)].numpy() for b in run.blocks]) / scale)
          cols_z.append((R - R.mean(1, keepdims=True)) / np.maximum(sd_s, 1e-30)[:, None])
          have_b = own_bound is not None and all((t1, t2, m_) in own_bound[b] for b in run.blocks)
          cols_b.append(np.concatenate([own_bound[b][(t1, t2, m_)].numpy() for b in run.blocks]) / scale
                        if have_b else None)
        else:
          sel = np.flatnonzero(valid)
          for Y, Bd in zip(cols_y, cols_b):
            if Bd is None:
              wb['no_bound'] += int(Y[sel].size)
            else:
              wb['cells'] += int(Y[sel].size)
              wb['within'] += int((np.abs(Y[sel]) <= Bd[sel]).sum())
          Ys.append(np.stack([c[sel].ravel() for c in cols_y], 1))
          Os.append(np.stack([c[sel].ravel() for c in cols_o], 1))
          Zs.append(np.stack([c[sel].ravel() for c in cols_z], 1))
    if Ys:
      res[d] = loop_regression(Ys, Os, Zs, rng, prm['n_perm'])
    res.setdefault('own_within_bound', {})[d] = wb
    # cumulative concentration of capture against kappa, per stage and G16 point
    traj = {}
    for stage in sorted({b.split('.')[0] for b in run.blocks}):
      blocks = [b for b in run.blocks if b.startswith(stage + '.')]
      cum, rows = 0, []
      for j in range(1, len(run.g16)):
        t1, t2 = run.g16[j - 1], run.g16[j]
        sd_m = np.concatenate([ctx.sd(t1, b)[m_] for b in blocks])
        sd_s = np.concatenate([ctx.sd(t1, b)[s_] for b in blocks])
        O = np.concatenate([own[b][(t1, t2, m_)].numpy() for b in blocks]) / np.maximum(sd_m, 1e-30)[:, None]
        O[(sd_m < ctx.floor(t1, m_)) | (sd_s < ctx.floor(t1, s_))] = 0
        cum = cum + np.maximum(O, 0)
        tot = cum.sum(0)
        H = np.where(tot > 0, ((cum / np.maximum(tot, 1e-300)) ** 2).sum(0), np.nan)
        row = {'step': t2, 'H_capture': float(np.nanmedian(H))}
        gs = ctx.gstate.get(t2)
        if gs is not None:
          kp = np.maximum(np.concatenate([gs['blocks'][b]['kappa'][m_].numpy() for b in blocks]), 0)
          if t2 in ctx.R:
            k_m = np.concatenate([ctx.sd(t2, b)[m_] for b in blocks])
            k_s = np.concatenate([ctx.sd(t2, b)[s_] for b in blocks])
            kp[(k_m < ctx.floor(t2, m_)) | (k_s < ctx.floor(t2, s_))] = 0
          tk = kp.sum(0)
          row['H_kappa'] = float(np.nanmedian(np.where(tk > 0, ((kp / np.maximum(tk, 1e-300)) ** 2).sum(0), np.nan)))
        rows.append(row)
      traj[stage] = rows
    res.setdefault('concentration', {})[d] = traj
  return res


# --------------------------------------------------------------------------------------------------- tests main

def path_table(ctx: Ctx, own: dict, oth: dict) -> dict:
  """G+ check per refined G16 interval: the sum of the sub-intervals' own terms and other-derived terms against the
  direct G16 terms (median |difference| / sd and relative L2), pooled over blocks and halves."""
  out = {}
  subs = {}
  for s1, s2, role, j in ctx.plan:
    if role == 'sub' and j in ctx.refined:
      subs.setdefault(j, []).append((s1, s2))
  g = ctx.run.g16
  for j, ivs in sorted(subs.items()):
    d_own, d_oth, n_own = [], [], []
    for b in ctx.run.blocks:
      for h in (0, 1):
        sd = ctx.sd(g[j - 1], b)[h][:, None]
        direct = own[b][(g[j - 1], g[j], h)].numpy()
        summed = sum(own[b][(a, z, h)].numpy() for a, z in ivs)
        d_own.append(np.abs(summed - direct) / np.maximum(sd, 1e-30))
        n_own.append(np.linalg.norm(summed - direct) / max(np.linalg.norm(direct), 1e-30))
        do = oth[b][(g[j - 1], g[j], h)].numpy()
        so = sum(oth[b][(a, z, h)].numpy() for a, z in ivs)
        d_oth.append(np.linalg.norm(so - do) / max(np.linalg.norm(do), 1e-30))
    out[f'{g[j - 1]}-{g[j]}'] = {'subs': [list(x) for x in sorted(ivs)],
                                 'own_median_abs_diff_sd': float(np.median(np.concatenate([x.ravel() for x in d_own]))),
                                 'own_rel_l2_median': float(np.median(n_own)),
                                 'other_rel_l2_median': float(np.median(d_oth))}
  return out


def carryover_report(ctx: Ctx) -> dict:
  """AdamW runs, G16 intervals that start at a usable snapshot: the part of each class's own writes T_c(c) that its
  momentum at t1 carried in from before t1 (rcdl.interval_split.momentum_carryover; an estimate with D and the decay
  held at their t1 values), as the median over units and classes of |carry| / |T_c(c)| and the ratio of the sums of
  absolute values (measurement half H2)."""
  run, g = ctx.run, ctx.run.g16
  if run.cfg.get('opt', 'adamw') != 'adamw':
    return {}
  lr_of = lr_schedule(run.cfg)
  out = {}
  for j in range(1, len(g)):
    t1, t2 = g[j - 1], g[j]
    if t1 not in run.snaps:
      continue
    A = run.full(t1)['attr']
    ratios, num, den = [], 0.0, 0.0
    for b in run.blocks:
      X = ctx.xhw(t1, t2)['blocks'][b]
      cm = {k: X[k][1].to(ctx.dev, torch.float64) for k in ('Hbar_gate', 'Hbar_value', 'qbar_gate', 'qbar_value')}
      own = own_terms(run.inc(t1, t2, b, bounds=False), cm)
      carry = 0
      for M in MAPS:
        r = A[f'{b}.pw_{M}']
        t = int(r['t'])
        cw = momentum_carryover(r['m'][:ctx.C].to(ctx.dev), r['v'].to(ctx.dev), t, t2, lr_of)
        cbias = momentum_carryover(r['mb'][:ctx.C].to(ctx.dev), r['vb'].to(ctx.dev), t, t2, lr_of)
        carry = carry + torch.einsum('cnd,ncd->nc', cw, cm[f'Hbar_{M}']) + cbias.T * cm[f'qbar_{M}']
      ratios.append((carry.abs() / own.abs().clamp(min=1e-300)).flatten().cpu().numpy())
      num += float(carry.abs().sum())
      den += float(own.abs().sum())
    out[f'{t1}-{t2}'] = {'median_ratio': float(np.median(np.concatenate(ratios))),
                         'ratio_of_sums': num / max(den, 1e-300)}
  return out


def rho_table(run: Run) -> dict:
  """Decay of the earlier parts over each G16 interval, rho = a_t2 / a_t1 (median, min and max over the maps), to be
  read next to every F term (Muon's lr x wd gives a_T of about 4e-6 in E4, against about 0.29 under AdamW)."""
  out = {}
  g = run.g16
  for j in range(1, len(g)):
    A1, A2 = run.attr(g[j - 1]), run.attr(g[j])
    r = [float(A2[k]['a']) / float(A1[k]['a']) for k in A1 if k.endswith(('pw_gate', 'pw_value'))]
    out[f'{g[j - 1]}-{g[j]}'] = {'rho_median': float(np.median(r)), 'rho_min': min(r), 'rho_max': max(r)}
  return out


def interval_checks(ctx: Ctx) -> dict:
  """The largest value of every check of the interval files (completeness before and after fp32 storage, with the
  p*_t2 reference, the per-image identity, the pre-activation identity, the class against the random split), and the
  intervals flagged as short with an fp16 endpoint after step 10,000."""
  out = {'max': {}, 'short_fp16_late': []}
  for (s1, s2) in sorted(ctx.files):
    X = ctx.xhw(s1, s2)
    if X['flags'].get('short_fp16_late'):
      out['short_fp16_late'].append([s1, s2])
    for r in X['blocks'].values():
      for k, v in r['checks'].items():
        out['max'][k] = max(out['max'].get(k, 0.0), float(v))
  return out


def cmd_tests(a, run: Run):
  ctx = Ctx(a, run)
  prm = ctx.prm
  t0 = time.time()
  own, own_bound, oth, plast, prov, pbias = {}, {}, {}, {}, {}, {}
  exits, entries, x3 = [], [], []
  checks = {'complete_rel_max': 0.0, 'view_check': 0.0, 'prov_check': 0.0, 'event_check': 0.0}
  per_block = {}
  for b in run.blocks:
    rec = analyze_block(ctx, b)
    own[b], oth[b], plast[b], prov[b] = rec['own'], rec['oth'], rec['plast'], rec['prov']
    own_bound[b] = rec['own_bound']
    checks['complete_rel_max'] = max(checks['complete_rel_max'], max(rec['complete'].values()))
    checks['view_check'] = max(checks['view_check'], rec['view_check'])
    checks['prov_check'] = max(checks['prov_check'], rec['prov_check'])
    ex_b = [exit_summary(ctx, ac) for ac in rec['accs'] if ac.kind == 'exit']
    en_b = [entry_summary(ctx, ac) for ac in rec['accs'] if ac.kind == 'entry']
    checks['event_check'] = max([checks['event_check']] + [r['check'] for r in ex_b + en_b])
    bp = bias_pushes(ctx, b)
    pbias[b] = bp['oriented']
    x3_b = x3_block(ctx, b, bp)
    exits += ex_b
    entries += en_b
    x3 += x3_b
    per_block[b] = {'complete_rel_max': max(rec['complete'].values())}
    print(json.dumps({'block': b, 'exits': len(ex_b), 'entries': len(en_b),
                      'complete': per_block[b]['complete_rel_max'], 'sec': round(time.time() - t0)}), flush=True)
  checks['n_c_equal'] = ctx.n_c_info['equal']
  res = {'version': 2, 'run': run.dir, 'code_hash': code_hash(), 'events_hash': ctx.events['code_hash'],
         'steps': run.g16, 'snapshots': run.snaps, 'snapshots_dropped_no_full': run.snaps_no_full, 'params': prm,
         'has_random': run.has_random, 'class_counts': ctx.n_c_info,
         'band_source': {'X1_own_band': 'random' if run.has_random else 'C1', 'X3_S1_primary': 'C1',
                         'X3_S1_randband': run.has_random},
         'checks': checks, 'event_counts': ctx.events['counts'], 'pooled': {}, 'blocks': {}, 'primary': {}}
  groups = {'all': run.blocks, 'early': [b for b in run.blocks if b.startswith(('stage0', 'stage1'))],
            'late': [b for b in run.blocks if not b.startswith(('stage0', 'stage1'))]}
  for gname, bl in groups.items():
    if not bl:
      continue
    pool = {}
    for d in DIRS:
      ed = [r for r in exits if r['d'] == d and r['b'] in bl]
      anch = [r for r in ed if r['anchored']]
      turn = [r for r in ed if not r['anchored']]
      gate = [r for r in anch if 'skip' not in r]                  # the validity gate: units above the sd floor
      rep_rate = float(np.mean([r['rep'] for r in gate])) if gate else None
      x1 = {'replication_rate_anchored': rep_rate,
            'replication_rate_anchored_relaxed': float(np.mean([r['rep_relaxed'] for r in gate])) if gate else None,
            'noise_dominated': bool(rep_rate is not None and rep_rate < 0.5),
            # primary population: anchored exits selected on the selection half only
            'anchored_all': x1_population(ctx, anch),
            'turnover': x1_population(ctx, turn),
            'by_phase': {f'phase{ph}': {
              'anchored': share_table([r for r in anch if r['phase'] == ph], 'C1', ctx, fine=False),
              'turnover': share_table([r for r in turn if r['phase'] == ph], 'C1', ctx, fine=False)}
              for ph in range(3)},
            # sensitivity: 'replicated' also selects on the fall of the measurement half, the half every term is
            # measured on (C1 is not so selected)
            'anchored_replicated': x1_population(ctx, [r for r in anch if r['rep']]),
            'turnover_replicated': x1_population(ctx, [r for r in turn if r['rep']])}
      en = [r for r in entries if r['d'] == d and r['b'] in bl]
      xs3 = [r for r in x3 if r['d'] == d and r['b'] in bl]
      x2 = x2_population(ctx, en)
      dropped_ev = sum_info([ctx.events.get('block_counts', {}).get(b, {}).get(d, {}) for b in bl])
      x2['entries_from_step0'] = dropped_ev['entries_from_step0']      # left-censored, left out of each arm
      x2['low_sd_entries'] = dropped_ev['low_sd_entries']
      x1['low_sd_exits'] = dropped_ev['low_sd_exits']
      pool[d] = {'X1': x1, 'X2': x2, 'X3_S1': x3_population(ctx, [r for r in xs3 if r['anchored']])}
    res['pooled'][gname] = pool
  # half swap: the same (unit, class) exit in both directions
  lab = {d: {(r['b'], r['n'], r['c']): r.get('label') for r in exits if r['d'] == d and 'label' in r} for d in DIRS}
  both = set(lab['s0m1']) & set(lab['s1m0'])
  res['pooled']['all']['half_swap'] = {'s0m1': len(lab['s0m1']), 's1m0': len(lab['s1m0']), 'both': len(both),
                                       'same_label': sum(lab['s0m1'][k] == lab['s1m0'][k] for k in both)}
  for b in run.blocks:
    res['blocks'][b] = dict(per_block[b])
    for d in DIRS:
      ed = [r for r in exits if r['d'] == d and r['b'] == b]
      res['blocks'][b][d] = {'exits': len(ed), 'anchored': sum(r['anchored'] for r in ed),
                             'X1_anchored': share_table([r for r in ed if r['anchored']], 'C1', ctx, fine=False),
                             'X1_anchored_replicated': share_table([r for r in ed if r['anchored'] and r['rep']], 'C1',
                                                                   ctx, fine=False),           # sensitivity
                             'labels': {k: sum(r.get('label') == k for r in ed) for k in LABELS}}
  res['X3_S2'] = x3_s2(ctx, own)
  res['X4'] = x4(ctx, own, plast, prov, pbias, own_bound)
  res['X5'] = x5(ctx, own, oth, own_bound)
  res['path_dependence'] = path_table(ctx, own, oth)
  res['carryover'] = carryover_report(ctx)
  res['rho'] = rho_table(run)
  res['interval_checks'] = interval_checks(ctx)
  # the five primaries (H1 -> H2), Holm
  late = res['pooled'].get('late', res['pooled']['all'])['s0m1']
  sh = late['X1']['anchored_all']['excess_vs_C1'].get('shares', {})
  pr = {'X1_own_excess_share': sh.get('own', {}).get('p0'),
        'X2_own_rise_dropped_minus_persistent': res['pooled']['all']['s0m1']['X2'].get('own_diff', {}).get('p0'),
        'X3_S1_withdrawn_minus_C1': late['X3_S1'].get('withdrawn', {}).get('diff', {}).get('p0'),
        'X4_F2_relatives': res['X4'].get('s0m1', {}).get('stage', {}).get('all', {}).get('F2', {}).get('p0'),
        'X5_loop_own': res['X5'].get('s0m1', {}).get('p_own_greater')}
  x2d = res['pooled']['all']['s0m1']['X2'].get('own_diff', {})
  x3d = late['X3_S1'].get('withdrawn', {}).get('diff', {})
  f2 = res['X4'].get('s0m1', {}).get('stage', {}).get('all', {}).get('F2', {})
  x5r = res['X5'].get('s0m1', {})
  res['primary'] = {'estimates': {'X1_own_excess_share': sh.get('own', {}).get('est'),
                                  'X2_own_rise_dropped_minus_persistent': x2d.get('est'),
                                  'X3_S1_withdrawn_minus_C1': x3d.get('est'), 'X4_F2_relatives': f2.get('est'),
                                  'X5_loop_own': x5r.get('coef_own')},
                    'ci': {'X1_own_excess_share': sh.get('own', {}).get('ci'),
                           'X2_own_rise_dropped_minus_persistent': x2d.get('ci'),
                           'X3_S1_withdrawn_minus_C1': x3d.get('ci'), 'X4_F2_relatives': f2.get('ci'),
                           'X5_loop_own': None},
                    'null': {'X1_own_excess_share': 0.0, 'X2_own_rise_dropped_minus_persistent': 0.0,
                             'X3_S1_withdrawn_minus_C1': 0.0, 'X4_F2_relatives': 0.0,
                             'X5_loop_own': x5r.get('null_own_mean')},
                    'p': pr, 'holm': holm(pr)}
  # the 50 units with the most anchored exits (H1 -> H2)
  cnt = {}
  for r in exits:
    if r['d'] == 's0m1' and r['anchored']:
      cnt.setdefault((r['b'], r['n']), []).append(r)
  top = sorted(cnt.items(), key=lambda kv: (-len(kv[1]), block_sort_key(kv[0][0]), kv[0][1]))[:50]
  res['units'] = []
  for (b, n), rr in top:
    s3 = [r for r in x3 if r['d'] == 's0m1' and r['b'] == b and r['n'] == n and r['anchored'] and 'label_frac' in r]
    res['units'].append({'b': b, 'n': n, 'anchored_exits': len(rr), 'replicated': sum(r['rep'] for r in rr),
                         'shares': share_table(rr, 'C1', ctx, fine=False).get('shares'),
                         'labels': {k: sum(r.get('label') == k for r in rr) for k in LABELS},
                         'S1': {k: float(sum(r['label_frac'][k] for r in s3)) for k in S1}})   # leave-one-out
  dump_json(jsonable(res), a.out_json or os.path.join(a.out, 'exits_tests.json'), indent=1)
  print(json.dumps(jsonable(res['primary'])), flush=True)
  print('EXITS_TESTS_DONE', flush=True)


def cmd_compare(a):
  """Replication of the primaries of one run (--other, e.g. E1r) against a reference run (--ref, E1): replicated when
  the estimate has the same direction relative to its null and lies within the reference's 95% CI (X5, which has a
  permutation null and no CI: the same direction and a permutation p below 0.05). No verdict ('replicated': None,
  'refused') when the two runs differ in the G16 steps, the X3 S1 band or the class counts (or a file predates these
  records), unless --allow_mismatch."""
  ref, oth = json.load(open(a.ref)), json.load(open(a.other))
  mism = []
  if ref.get('steps') != oth.get('steps'):
    mism.append('steps')
  band = lambda r: (r.get('band_source') or {}).get('X3_S1_primary')
  if band(ref) is None or band(ref) != band(oth):
    mism.append('X3_S1_band')
  counts = lambda r: (r.get('class_counts') or {}).get('n_c_sha')
  if counts(ref) is None or counts(ref) != counts(oth):
    mism.append('class_counts')
  allow = bool(getattr(a, 'allow_mismatch', False))
  out = {'ref': a.ref, 'other': a.other, 'mismatch': mism, 'allow_mismatch': allow,
         'band_source': {'ref': ref.get('band_source'), 'other': oth.get('band_source')},
         'class_counts': {'ref': ref.get('class_counts'), 'other': oth.get('class_counts')}, 'primaries': {}}
  for k, e1 in ref['primary']['estimates'].items():
    e2 = oth['primary']['estimates'].get(k)
    n1, n2 = ref['primary']['null'].get(k), oth['primary']['null'].get(k)
    ci = ref['primary']['ci'].get(k)
    rec = {'ref': e1, 'other': e2, 'ref_ci': ci}
    if None not in (e1, e2, n1, n2):
      same = np.sign(e1 - n1) == np.sign(e2 - n2)
      inside = ci is not None and ci[0] <= e2 <= ci[1]
      rec['same_direction'] = bool(same)
      if k == 'X5_loop_own':
        p2 = oth['primary']['p'].get(k)
        rec['replicated'] = bool(same and p2 is not None and p2 < 0.05)
      else:
        rec['within_ref_ci'] = bool(inside)
        rec['replicated'] = bool(same and inside)
      if mism and not allow:
        rec['replicated'], rec['refused'] = None, mism
    out['primaries'][k] = rec
  dump_json(jsonable(out), a.out_json, indent=1)
  print(json.dumps(jsonable(out)), flush=True)


# ------------------------------------------------------------------------------------------------------------ CLI

def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('stage', choices=['probe', 'integrity', 'stock', 'events', 'intervals', 'gradstate', 'tests', 'all',
                                    'compare'])
  p.add_argument('--run', default=None, help='run directory (model_*, attr_*, attrR_*, attr_init.pt, config.json)')
  p.add_argument('--out', default=None, help='output directory (default: <run>/exits)')
  p.add_argument('--history', default=None, help='snap_{step}.pt and history.json of experiments.convnext_snapshots')
  p.add_argument('--full', default=None, help='full_{step}.pt (fp32 parts and moments) of convnext_snapshots --full')
  p.add_argument('--probe', default='/marimo/work/probe50', help='probe directory (x.npy, y.npy) or a .pt file')
  p.add_argument('--val', default='/marimo/work/data/imagenet/data', help='validation parquet (probe stage)')
  p.add_argument('--per_class', type=int, default=50)
  p.add_argument('--grid', default='', help='comma-separated G16 steps (default: the trainer checkpoints)')
  p.add_argument('--steps', default='', help='stock / gradstate: comma-separated grid points (default: all of G+)')
  p.add_argument('--family', default='all', choices=['g16', 'gplus', 'snap', 'all'])
  p.add_argument('--batch', type=int, default=250)
  p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
  p.add_argument('--visual', default='docs/data/imagenet_dinov2_classdist.npz')
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--class_counts', default='', help='per-example labels (labels.npy or rand_groups.npy), or per-class '
                 'counts: a file of exactly C entries, any dtype; default: <run>/rand_groups.npy, then '
                 '/dev/shm/imagenet256/labels.npy; none found is an error (toy tests: --params \'{"equal_counts": '
                 'true}\')')
  p.add_argument('--params', default='', help='JSON overrides of the protocol parameters (tests on toy runs only)')
  p.add_argument('--force', action='store_true', help='events: rebuild a frozen events.json')
  p.add_argument('--no_events', action='store_true', help='intervals without events.json (no event pairs; tests only)')
  p.add_argument('--skip_integrity', action='store_true')
  p.add_argument('--no_full', action='store_true', help='drop history snapshots that have no full_{step}.pt (otherwise '
                 'an error: G+, mu, S3, the carry-over and D need them)')
  p.add_argument('--no_random', action='store_true', help='run without the random split even if attrR files exist')
  p.add_argument('--allow_mismatch', action='store_true', help='compare: give verdicts although the runs differ in '
                 'steps, X3 band or class counts')
  p.add_argument('--out_json', default=None)
  p.add_argument('--ref', default=None, help='compare: the reference run\'s exits_tests.json (E1)')
  p.add_argument('--other', default=None, help='compare: the replicating run\'s exits_tests.json (E1r)')
  a = p.parse_args()
  a.prm = defaults() | (json.loads(a.params) if a.params else {})
  if a.stage == 'probe':
    build_probe(a.val, a.probe, a.per_class)
    return
  if a.stage == 'compare':
    if not (a.ref and a.other):
      p.error('compare needs --ref and --other')
    a.out_json = a.out_json or os.path.join(a.out or os.path.dirname(os.path.abspath(a.other)), 'exits_compare.json')
    os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
    cmd_compare(a)
    return
  a.out = a.out or os.path.join(a.run, 'exits')
  os.makedirs(a.out, exist_ok=True)
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  grid = [int(t) for t in a.grid.split(',')] if a.grid else None
  mk = lambda snaps=None: Run(a.run, a.history, a.full, grid=grid, snaps=snaps, device=a.device, no_random=a.no_random)
  run = mk()
  require_full(a, run)
  avail = run.snaps_avail
  if a.stage not in ('integrity', 'all'):
    run = mk(usable_snaps(a, a.run, a.history, a.full, avail))
  stages = ['integrity', 'stock', 'events', 'intervals', 'gradstate', 'tests'] if a.stage == 'all' else [a.stage]
  for st in stages:
    if st == 'integrity':
      cmd_integrity(a, run)                            # always: integrity.json records what was checked and dropped
      run = mk(usable_snaps(a, a.run, a.history, a.full, avail))
    elif st == 'stock':
      cmd_stock(a, run, Probe(a.probe))
    elif st == 'events':
      if not (st == a.stage) and os.path.exists(os.path.join(a.out, 'events.json')):
        continue
      cmd_events(a, run)
    elif st == 'intervals':
      cmd_intervals(a, run, Probe(a.probe))
    elif st == 'gradstate':
      cmd_gradstate(a, run, Probe(a.probe))
    elif st == 'tests':
      cmd_tests(a, run)


if __name__ == '__main__':
  main()
