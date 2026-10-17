"""Stage-1 reversal: the residual stream along a unit's write direction where the unit is active, and how the network
reads that direction (X1); the unit's pole push when its same-direction partners or the stream along its direction are
changed (X2); which part of the downstream network reads the direction (X4).

convnext_bias_paths.py finds that from about step 1,000 on, a stage-1 unit's 20 strongest classes push its response down
(late blocks: up), and convnext_unit_roles.py that most stage-1 units have a partner whose write direction is nearly
parallel (|cos| > 0.9; the 8-dimensional write space is nearly collapsed). Candidate mechanisms (review of 2026-10-16):
  M1 overshoot: same-direction partners already fill the stream along the unit's direction, so more of the unit hurts
  M4 cancellation: the unit's write cancels a useful incoming component of the stream along its direction
  M5 class-specific negative readout of the direction, independent of the stream's level
  M3 SwiGLU curvature: the push comes from the unit's own nonlinearity, not from how its write is read
Notation as in convnext_bias_paths.py: for block b, unit n, image i of class k and position p, u = SiLU(g) v, the
unit's top-20 classes P_n (by class-mean u* = max_p u), and the pole coefficients pa_v = mean_{c in P_n} SiLU(Gm[n, c]),
pa_g = mean_{c in P_n} SiLU'(Gm[n, c]) Vm[n, c].

X1. The block writes w_n = gamma * W_down[:, n] per unit of u_n; e = w_n / |w_n|. At every position:
  s_in    = x_in . e                    the block's input along e
  s_own   = u_n |w_n|                   the unit's own write
  s_part  = sum_{m != n} u_m w_m . e    the other units of the block (partners)
  s_bias  = gamma b_down . e            the block's bias write;  s_out = x_out . e = s_in + s_own + s_part + s_bias
  s_later = x_end . e - s_out           the later blocks of the stage (x_end: the stream at the stage's end; 0 in the
                                        stage's last block);  s_end = s_out + s_later
  s_excl  = s_out - s_own               the stream along e without the unit
  kappa   = SiLU(g) pa_v + SiLU'(g) v pa_g   how much this position's bias gradient moves the pole's response
  delta   = dCE_i / du_n = |w_n| r,  r = e . dCE_i / dx_out   the downstream readout of e (r > 0: more e raises CE)
  push    = -kappa delta                its sum over positions, class-averaged, is convnext_bias_paths' pole[k, n]
r does not involve unit n's own SwiGLU; kappa does, and |kappa| grows with g and v, so a push-based curve along u_n or
along s_out (which contains s_own) moves with |kappa| under every mechanism. The r-based readouts do not.
Roles per unit: top20 (P_n), relatives (WordNet relatives of P_n outside it, with --wordnet), rest (all other classes).
Each quantity is a class value (mean over the class's images), then a mean over the role's classes per unit, then a mean
over live units, unless stated:
  pole_{role}                 class-mean push summed over positions, in units of the unit's mean |pole| over all
                              classes (convnext_bias_paths' ce split, with the relatives left out of 'rest')
  pole_{q}_{role}             the same from the positions with the signs q of kappa and delta (the four add up to
                              pole_{role}): kp_dp (kappa > 0, delta > 0: the readout is against more e; push < 0),
                              kn_dn (kappa < 0, delta < 0: the pole's bias write lowers u_n here, where more would
                              help: the unit's own curvature; push < 0), kp_dn and kn_dp (push > 0)
  push_act_{role}, n_act_{role}   the push from the active positions alone (same units), active positions per image
  {k}_{role}                  for k in s_in, s_own, s_part, s_bias, s_out, s_later, s_end: |kappa|-weighted mean at the
                              active positions (u_n >= 0.5 u_n* in the image, u_n* > 0; with --active pole: u_n >= half
                              the unit's mean u* over P_n), in stream units
  pstar_{k}_{role}            the same at p* alone, over the images with u_n* > 0 (unweighted)
  {k}_rev_{role}              the same over all positions, weighted by the reversal mass max(-push, 0)
  frac_sout_pos_{role}        share of (unit, class) pairs with s_out > 0 (the active-position means above)
  frac_m1_{role}              s_out > 0 and s_excl > s_out / 2 (input, partners, bias supply most of a positive level)
  part_share_{role}           mean of s_part / (|s_in| + |s_own| + |s_part| + |s_bias|)
  frac_sin_neg_{role}         s_in < 0 (s_own > 0 at the active positions: the input opposes the unit)
  frac_own_shrinks_{role}     |s_out| < |s_excl| (the unit reduces the stream along e)
  frac_m4_{role}              s_in < 0 and |s_out| < |s_excl| (M4: incoming component negative, reduced by the unit)
  frac_m4_in_{role}           s_in < 0 and |s_in + s_own| < |s_in| (the unit cancels part of the incoming component)
Bins, for v in sout, sin, u (u_n), sexcl: per unit, deciles of v over all positions of all probe images, each position
weighted by |kappa| (each bin holds a tenth of the unit's derivative mass; --edges positions: unweighted), estimated
from --n_sample random positions per image, shared by all roles:
  push_by_{v}_bin_{role}      class-mean push within each bin, same units as pole (the 10 add up to pole_{role})
  push_per_pos_by_{v}_bin_{role}  push per position in the bin (pooled over the role's images) times the positions per
                              image: the pole push an image would get if all its positions were in that state
  occ_by_{v}_bin_{role}       share of the role's positions in each bin
  r_by_{v}_bin_{role}         |kappa|-weighted mean of r in each bin, times the positions per image (the CE change of an
                              image whose stream along e rose by 1 at every position, if all were in this state)
  slope_by_{v}_{role}, slope_r_by_{v}_{role}   Spearman correlation of the unit-averaged push_per_pos (r) curve with
                              the bin index (bins with positions)
  role_offset_by_{v}          per unit, push_per_pos top20 minus rest in the same bin, weighted by the top20 occupancy
  role_offset_r_by_{v}        the same for r, weighted by the top20 |kappa| mass in the bin
Two-variable bins (sout x sin and sout x u, the same edges, 10 x 10 cells; the grids are in the raw file):
  role_offset_r_by_sout_{b}   top20 minus rest r in the same cell, weighted by the top20 |kappa| mass
  slope_r_by_{b}_within_sout_{role}   Spearman correlation of the unit-averaged r with the b bin inside each s_out row
                              (rows with 3 or more cells), averaged over the rows
Regression, per unit and role over all positions (|kappa|-weighted, classes weighted equally), r ~ r0 + b_excl s_excl +
b_own s_own, times the positions per image as r_by: r0_{role}, b_excl_{role}, b_own_{role} are medians over live units;
sd_own_over_excl_{role}: median of the weighted SD of s_own over that of s_excl (b_own is poorly determined, and open to
confounding through u_n's inputs, when this is small).
Block: write_eff_rank ((sum sigma)^2 / sum sigma^2 over the singular values of the write directions; at most d),
share_cos_hi (share of units with another at |cos| > 0.9), gamma_max_over_median, cos_e_ones (mean |cos| of e with the
all-ones direction, which a LayerNorm reading the stream directly removes: the downsample's; the blocks' own LayerNorm
reads the depthwise convolution's output), rms_sin_all, rms_sout_all (the scale of the s values), units (live),
positions.
Readings: M1: s_out > 0 at the active pole positions mostly not from the unit (frac_m1), r rising with the level on one
curve for both roles (slope_r_by_sout > 0, role_offset_r_by_sout and _sout_u near 0, b_excl ~ b_own > 0), reversal mass
in kp_dp; M4: frac_m4 high; M5: r flat in the level with role_offset_r_* > 0 and r0_top20 > r0_rest; M3: reversal mass
in kn_dn. All of X1 is correlational; X2 intervenes.

X2 (stage-1 blocks by default), on the images of P_n and of 40 other classes (random; not WordNet relatives of P_n with
--wordnet): dCE / du is recomputed by torch.autograd.grad under each config and n's pole push is rebuilt with the
BASELINE pa_v, pa_g and kappa (n's own g, v depend only on the block's input, which no config changes; earlier blocks'
units are left alone for that reason). Partners of n, from its active positions on its top-20 images (as in X1):
c_m = mean u_m times w_m . e (the partner's mean write along e there) and size_m = mean |u_m| |w_m|:
  splus, sminus     units m != n of the block with |cos(w_m, w_n)| > 0.9 and c_m > 0, c_m < 0
  later             units of the stage's later blocks with |cos| > 0.9 and c_m > 0 (they write after x_out)
  ctrl_{set}        per member, the unused unit of the same block with |cos| < 0.5 (live, size > 0) closest in log
                    size (members in a random order, seeded per unit): a control matched in write size
(a unit without active positions on its top-20 images, u_n* <= 0 there, gets no partner sets, only base and beta)
Configs: base; {set}_a{alpha}: the set's units scaled by alpha (--alphas); splus_e_a{alpha}: only S+'s component along
e scaled, x_out -> x_out - (1 - alpha) sum_{m in S+} u_m (w_m . e) e; splus_perp_a{alpha}: only the rest of S+'s writes
(orthogonal to e) scaled, the damage control of splus_e (the two writes add up to S+'s; in a collapsed write space few
units have |cos| < 0.5, so ctrl_splus is often much smaller than S+: n_short_ctrl); beta{b}: the whole stream along e at
x_out scaled, x_out -> x_out + (b - 1)(x_out . e) e (--betas; autograd's delta then carries a factor b, divided out).
Per unit: class-mean push of the top-20 and of the 40 classes over the baseline mean |push| of the 60, and the
class-mean change of CE. Summary per block:
  units, top20_base, rest_base, ce_top20_base, ce_rest_base, s_out_act (mean s_out at the active positions)
  per config K: top20_K, rest_K, dce_top20_K, dce_rest_K, units_K (units with the config: its set not empty),
    top20_base_K (their baseline), share_up_K (share of them whose top-20 push rises against their baseline)
  per set: units_{set}, n_empty_{set}, size_{set} (mean members), ds_e_{set} (mean sum of c_m: the set's share of the
    level along e at the active positions, against s_out_act); n_short_ctrl_{set}: controls smaller than the set
  d_{set}_vs_ctrl_a{alpha}, share_{set}_above_ctrl_a{alpha} (set in splus, sminus, later): per unit, the top-20 push
    under the set minus under its control; mean and share > 0; d_e_vs_perp_a{alpha}, share_e_above_perp_a{alpha}: the
    same for splus_e against splus_perp
Readings: M1: the reversal (top20 < 0) shrinks for splus and splus_e (d_e_vs_perp > 0 and d_splus_vs_ctrl > 0 at
alpha = 0) and for beta < 1 when the level s_out_act is positive, deepens for sminus; M4, M5: flat in beta, sets and
controls alike.

X4 (--x4_blocks, blocks of one stage, e.g. stage1.block0,stage1.block1,stage1.block2): which reader downstream of x_out
reads e against the top classes. Stage 1 here; any stage i works the same way (stage1 -> stage{i}, stage2, stage3 ->
the later stages). x_j: the stream after stage-1 block j (x_b = x_out, the unit's block b; S the stage's last block);
y_k after stage-2 block k, y_0 = D1(x_S) (D1: the stage-2 downsample, LayerNorm2d then 2x2 stride-2 conv); z_l after
stage-3 block l, z_0 = D2(y_K); then the head (mean pool, final LayerNorm, linear) on z_L. g_j = dL/dx_j,
h_k = dL/dy_k, zeta_l = dL/dz_l (one torch.autograd.grad call per loss over all streams). A residual block
x_j = x_{j-1} + f_j(x_{j-1}) gives g_{j-1} - g_j = J_{f_j}^T g_j: the paths whose first block after x_b is f_j (pure
skips before it). With D^T the downsample's vector-Jacobian product at the forward point (linear in its argument),
g_b = sum over the readers R of term_R, exactly:
  stage1.block{j}, j > b    g_{j-1} - g_j
  stage2.block{k}           D1^T (h_{k-1} - h_k)
  stage3.block{l}           D1^T D2^T (zeta_{l-1} - zeta_l)
  head                      D1^T D2^T zeta_L            (skips all the way to the pool)
(a last block without a residual connection, tiny_crush13's, is entered by every path that reaches it: its term is
D1^T D2^T zeta_{L-1}, and head is 0). The later stages' terms do not depend on b (pure skips inside stage 1), so they
are pulled back once per batch for all blocks. A reader's term covers every path that enters it first, including the
paths through later blocks after it. 'head' is the linear path into the classifier: zeta_L is the same at every
position of z_L (the mean pool), so the head term at a position is the classifier's rows (weighted by softmax - onehot
for CE; -W_head[k] for L_true) pulled back through the final LayerNorm and, at that position, D2's and D1's LayerNorm
and conv phase. For L_true its only class dependence is W_head[k], a weight quantity that does not depend on the stream
along e, so it survives matching on s_out (static_head below).
Single paths, for every block reader R: R>head = D^T (J_{f_R}^T c_R), c_R the head's skip-only cotangent at R's output
(the head term there; zeta_L at z_L): the path R, then pure skips to the pool. term_R - term_{R>head} are the paths
from R through later blocks; for the network's last block the two are equal. Cost: one more block backward per block
reader, batch and loss (about one more backward pass) and its pullbacks through the downsamples.
Per position and unit: r_R = e . term_R (sum over the readers: r, X1's readout); groups: stage1_later (the stage-1
readers after b; none for the stage's last block), stage2, stage3 (sums over their blocks), head; single groups:
stage1_later>head, stage2>head, stage3>head (sums of the single paths); total: r itself (e . g_b, from the autograd
call, not the sum); perp: e_perp . g_b, with e_perp one random unit vector orthogonal to e per unit (seeded by --seed
and the block; raw e_perp). Losses (summary keys): ce (L = CE), true (L = -z[k], the true-class term; ce - true is the
logsumexp term); with --x4_sigma also ce_fixed and true_fixed (every LayerNorm's 1 / sigma held fixed in the backward,
convnext_bias_paths.fixed_sigma; ce - ce_fixed is the path through the normalisers' scale). Live units, roles, |kappa|
weights and the s_out bins are X1's (the same edges from the same --n_sample positions per image, --seed, --edges and
--batch), and every quantity is accumulated as X1's r_by bins (class sums of |kappa| r, each class weighted by 1 / its
images, per role, unit and bin), so ce's total is X1's readout (on a GPU up to TF32 rounding: X4's pass runs with
TF32 off, torch.backends.cuda.matmul and cudnn.allow_tf32 restored after, since TF32 rounds each reader's pullback
through a downsample separately and the readers would add up only to about 1e-3; X1 keeps the default). Per loss
(summary[loss]) and R in the readers, single paths, groups, single groups, total and perp:
  r_{R}_by_sout_bin_{role}    as X1's r_by_sout_bin_{role}, from r_R: per unit and bin the |kappa|-weighted mean of r_R
                              times the positions per image, mean over live units
  role_offset_r_{R}_by_sout   as X1's role_offset_r_by_sout: per unit, top20 minus rest in the same s_out bin, weighted
                              by the top20 |kappa| mass in the bin, mean over live units
  role_offset_r_{R}_by_sout_kpos, _kneg   the same from the positions with kappa > 0 and kappa < 0 alone, over the same
                              |kappa| mass (the two add up to role_offset_r_{R}_by_sout). The push is -kappa |w_n| r: a
                              positive offset of r deepens the reversal at kappa > 0 and relieves it at kappa < 0, where
                              a negative one drives it (as in E4 at step 2,000)
  r0_{R}_{role}               per unit the |kappa|-weighted mean of r_R over all positions of the role's images (classes
                              weighted equally) times the positions per image, mean over live units (not X1's
                              regression intercept r0)
The role offset as a vector, per group and total G: O_n = per unit, top20 minus rest of the |kappa|-weighted mean of
term_G (all d components of the stream gradient) in the same s_out bin, weighted as role_offset_r, so e_n . O_n is unit
n's role_offset_r_{G} and v . O_n the offset along any direction v (perp's along e_perp):
  offset_vec_norm_{G}         mean over live units of |O_n|
  offset_vec_cos_e_{G}        mean of cos(e_n, O_n) (a random direction instead of e: 0 +/- 1 / sqrt(d U), U live units)
  offset_vec_e_pos_{G}        share of live units with e_n . O_n > 0
  offset_null_sd_{G}          sd of the unit mean of v_n . O_n over independent random unit v_n orthogonal to e_n:
                              sqrt(sum_n |O_n - (e_n . O_n) e_n|^2 / (d - 1)) / U. perp is one such draw (its sign is
                              random per unit, so its offset is about 0 whatever the readout); this is its null
The head's weight geometry (weights only; none for a last block without skip): c_k = -W_head[k] pulled back along the
head's path with every 1 / sigma left out and the downsample convs averaged over their four 2x2 phases: c = N_f(-W[k]),
then for each later stage t from the last, c <- N_t(c K_t), K_t the conv weight averaged over the phases (out, in),
N(c) = gain * c minus its channel mean. Under fixed sigma, L_true's head term at a position is the same pull-back at
that position's phases over sigma_f sigma_D2 sigma_D1 times the last stage's positions (each phase pair equally often).
q[k, n] = e_n . c_k (q > 0: more e lowers z_k):
  static_head_offset          per unit, mean of q over top20 minus over rest, over q's SD over all classes; mean over
                              live units (static_head_pos: share of live units with the difference > 0)
  spearman_static_{G}         (per loss) Spearman correlation over live units of the unit's static difference (top20
                              minus rest, unscaled) with its role_offset_r_{G} (the groups and total)
  head_offset_at_static_0     (per loss) intercept of the least-squares line of the units' head offsets on their static
                              differences: the head offset left for a unit whose classifier rows do not favour its top
                              classes along e
The offset, kpos, r0 and bin keys are linear in r with weights shared by all R, so the readers add up to the total and
the groups to the same: sum_err_{loss} is the largest |sum over the readers - total| over them, relative to the largest
|value| (asserted < 1e-3; X1 and X2 of the step are written before X4 runs). Also: units, positions, edges_weight,
readers, groups, single, single_groups. Raw (raw['x4'][block]): bin_w (3, U, N_BINS) the |kappa| mass per role, unit
and s_out bin; bin_rw_{loss} {R: (3, U, N_BINS)} the |kappa| r_R sums (roles in ROLES order), bin_rwp_{loss} their
kappa > 0 part; bin_rv_{loss} {G: (3, U, N_BINS, d)} the |kappa| term_G sums; unit_offset_{loss} {R: (U,)} the
per-unit role offsets; static_q (C, U), static_offset (U,); edges_sout, e_perp, live, top, pa_v, pa_g, P.
Readings: the groups' signed offsets add up to the total and can cancel (one group above the total, others negative),
so the reading is the signed contributions, not the largest one. A head-dominated offset with a high
spearman_static_head and head_offset_at_static_0 near 0 describes the classifier's weight geometry (its top-class rows,
pulled back along the skips, point against e), not a readout that depends on the stream state. A block group whose
single path (G>head) carries most of its offset, with a spearman_static_G of the opposite sign, rewrites e into
something the head reads (it partly cancels e's head image); the rest of its offset goes through later blocks. true
against ce: whether more e lowers the true-class logit or raises the competitors' (ce - true); ce_fixed: whether the
normalisers' scale carries it. Specific to e: role_offset_r_total_by_sout well above offset_null_sd_total and
offset_vec_cos_e_total well above its random spread; offset_vec_norm_total far above the offset along e: most of the
class difference of the stream gradient lies outside e.

X1 and X2 start from each block's input, X4 from the stage's input, computed once for all probe images (as
convnext_unit_roles.py does). Raw arrays (X1 class-level and role-level accumulators, bin edges, regression
coefficients; X2 per-unit rows; X4 role-level bins) go to stage1_mech_raw_{step}.pt next to the JSON (keys x1, x2,
x4; the JSON has x4 only with --x4_blocks; the raw file is named by step alone, so a run of X4 alone needs an --out in
its own directory to keep an earlier run's files). --geometry_only reports the block geometry alone (no probe images),
e.g. on replay checkpoints (X3).

  python3 -m experiments.convnext_stage1_mech --run results/e1_tiny_adamw --steps 1000,2000,12500
  python3 -m experiments.convnext_stage1_mech --run results/e1_tiny_adamw --steps 12500 --blocks '' --x2_blocks '' \\
    --x4_blocks stage1.block0,stage1.block1,stage1.block2 --x4_sigma --out results/e1_tiny_adamw/x4/stage1_mech.json
"""

import argparse
import json
import os
import warnings
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
import torch.nn.functional as F

from experiments.convnext_atlas import probe_set
from experiments.convnext_bias_paths import dsilu, fixed_sigma, pole_coefs, spearman
from experiments.convnext_unit_roles import block_inputs, block_out, block_units, write_dir
from rcdl.convnext import build_convnext

DEFAULT_BLOCKS = 'stage0.block2,stage1.block0,stage1.block1,stage1.block2,stage2.block0'
X2_BLOCKS = 'stage1.block0,stage1.block1,stage1.block2'
N_BINS = 10
BIN_VARS = ('sout', 'sin', 'u', 'sexcl')
PAIRS = (('sout', 'sin'), ('sout', 'u'))
PARTS = ('s_in', 's_own', 's_part', 's_bias', 's_out', 's_later')
QUADS = ('kp_dp', 'kn_dn', 'kp_dn', 'kn_dp')
ROLES = ('top20', 'relatives', 'rest')
BIN_T = ('push', 'cnt', 'rw', 'w')              # binned: push, positions, |kappa| r, |kappa| (all class-weighted)
X2_SETS = ('splus', 'sminus', 'later')
X4_LOSSES = ('ce', 'true')


def parse_block(name):
  return tuple(int(t) for t in name.replace('stage', '').split('.block'))


@contextmanager
def no_param_grads(model):
  """Parameters without requires_grad while inside (only the gradient with respect to u is formed)."""
  flags = [p.requires_grad for p in model.parameters()]
  model.requires_grad_(False)
  try:
    yield
  finally:
    for p, f in zip(model.parameters(), flags):
      p.requires_grad_(f)


def block_guv(blk, x):
  """Gate, value and SwiGLU product (B, H, W, U) of a block for its input x (B, C, H, W)."""
  h = blk.norm(blk.dwconv(x).permute(0, 2, 3, 1))
  g, v = blk.pw_gate(h), blk.pw_value(h)
  return g, v, F.silu(g) * v


def stage_rest(model, i, j, x, scales=None):
  """The stream at the end of stage i from block (i, j)'s output x. scales: {j2: (U,) factor on the units of the
  later block j2}."""
  for j2 in range(j + 1, len(model.stages[i])):
    blk = model.stages[i][j2]
    x = block_out(blk, x, block_units(blk, x) * scales[j2]) if scales and j2 in scales else blk(x)
  return x


def tail(model, i, j, x, u, out_fn=None, scales=None, keep=None):
  """Logits from block (i, j)'s input x and its units u (B, H, W, U). out_fn(out, u): transform of the block's output;
  scales: as in stage_rest; keep: a list that receives the stream at the end of stage i."""
  x = block_out(model.stages[i][j], x, u)
  if out_fn is not None:
    x = out_fn(x, u)
  x = stage_rest(model, i, j, x, scales)
  if keep is not None:
    keep.append(x)
  for down, stage in zip(model.downsample[i + 1:], model.stages[i + 1:]):
    x = stage(down(x))
  return model.head(model.norm(x.mean((2, 3))))


@torch.no_grad()
def write_geometry(blk):
  """Write directions of a block: W (U, d) with rows w_n = gamma * W_down[:, n], norms |w_n|, unit directions E,
  M[m, n] = w_m . e_n, and bias_e[n] = (gamma * b_down) . e_n."""
  assert blk.skip, 'X1 needs a residual block'
  U = blk.pw_down.in_features
  W = torch.stack([write_dir(blk, n) for n in range(U)]).float()
  norms = W.norm(dim=1)
  E = W / (norms[:, None] + 1e-30)
  b = blk.pw_down.bias * blk.gamma if blk.gamma is not None else blk.pw_down.bias
  return {'W': W, 'norms': norms, 'E': E, 'M': W @ E.T, 'bias_e': E @ b.float()}


@torch.no_grad()
def geometry_stats(blk, cos_hi=0.9):
  """Block-level write geometry (see the module docstring), from the weights alone."""
  geo = write_geometry(blk)
  sv = torch.linalg.svdvals(geo['W'])
  cos = (geo['E'] @ geo['E'].T).abs()
  cos.fill_diagonal_(0)
  out = {'write_eff_rank': float(sv.sum() ** 2 / (sv ** 2).sum().clamp(min=1e-30)),
         'share_cos_hi': float((cos.max(1).values > cos_hi).float().mean()),
         'cos_e_ones': float((geo['E'].mean(1).abs() * geo['E'].shape[1] ** 0.5).mean())}
  if blk.gamma is not None:
    g = blk.gamma.detach().abs()
    out['gamma_max_over_median'] = float(g.max() / g.median().clamp(min=1e-30))
  return out


@torch.no_grad()
def stream_parts(blk, x, u, geo, x_end=None):
  """s_in, s_own, s_part, s_bias, s_out (B, P, U) along each unit's write direction, and s_later with x_end (the
  stream at the end of the stage; see the module docstring)."""
  E = geo['E']
  uf = u.flatten(1, 2)
  s_in = x.permute(0, 2, 3, 1).flatten(1, 2) @ E.T
  s_own = uf * geo['norms']
  s_part = uf @ geo['M'] - s_own
  s_bias = geo['bias_e'].expand_as(uf)
  s_out = block_out(blk, x, u).permute(0, 2, 3, 1).flatten(1, 2) @ E.T
  out = {'s_in': s_in, 's_own': s_own, 's_part': s_part, 's_bias': s_bias, 's_out': s_out}
  if x_end is not None:
    out['s_later'] = x_end.permute(0, 2, 3, 1).flatten(1, 2) @ E.T - s_out
  return out


def bin_values(sp, u):
  return {'sout': sp['s_out'], 'sin': sp['s_in'], 'u': u.flatten(1, 2), 'sexcl': sp['s_out'] - sp['s_own']}


@torch.no_grad()
def class_stats(model, name, X, y, n_classes, batch):
  """R, G, V (U, C): class means of u* = max_p u and of g, v at p* (as convnext_bias_paths.probe_paths)."""
  i, j = parse_block(name)
  blk = model.stages[i][j]
  U = blk.pw_gate.out_features
  acc = torch.zeros(3, n_classes, U, dtype=torch.float64, device=X.device)
  for s in range(0, len(X), batch):
    g, v, u = (t.flatten(1, 2) for t in block_guv(blk, X[s:s + batch]))
    ustar, pstar = u.max(1)
    yb = y[s:s + batch]
    for k, t in enumerate((ustar, g.gather(1, pstar[:, None])[:, 0], v.gather(1, pstar[:, None])[:, 0])):
      acc[k].index_add_(0, yb, t.double())
  cnt = torch.bincount(y, minlength=n_classes).double()[:, None]
  return tuple((acc[k] / cnt).T.cpu().numpy() for k in range(3))


@torch.no_grad()
def later_live(model, i, j, X, y, n_classes, batch):
  """{j2: (U,) bool} for the later blocks j2 of stage i: class-mean u* not constant (as pole_coefs' live), from block
  (i, j)'s input X."""
  stage = model.stages[i]
  acc = {j2: torch.zeros(n_classes, stage[j2].pw_gate.out_features, dtype=torch.float64, device=X.device)
         for j2 in range(j + 1, len(stage))}
  if not acc:
    return {}
  for s in range(0, len(X), batch):
    x = stage[j](X[s:s + batch])
    for j2 in acc:
      u = block_units(stage[j2], x)
      acc[j2].index_add_(0, y[s:s + batch], u.flatten(1, 2).amax(1).double())
      x = block_out(stage[j2], x, u)
  cnt = torch.bincount(y, minlength=n_classes).double()[:, None]
  out = {}
  for j2, a in acc.items():
    R = (a / cnt).T.cpu().numpy()
    out[j2] = np.ptp(R, 1) > 1e-3 * np.abs(R).max()
  return out


def weighted_edges(vals, w, n_bins=N_BINS):
  """vals, w: (N, U). Per unit the n_bins - 1 inner edges (U, n_bins - 1) of the w-weighted quantiles of vals
  (unweighted for a unit whose weights are all zero)."""
  v, order = vals.T.sort(1)
  cw = torch.gather(w.T.double(), 1, order).cumsum(1)
  N = v.shape[1]
  flat = cw[:, -1] <= 0
  if flat.any():
    cw[flat] = torch.arange(1, N + 1, dtype=torch.float64, device=cw.device)
  q = torch.arange(1, n_bins, dtype=torch.float64, device=cw.device) / n_bins
  k = torch.searchsorted(cw.contiguous(), (q[None] * cw[:, -1:]).contiguous()).clamp(max=N - 1)
  return torch.gather(v, 1, k)


def _kappa(g, v, pa_v, pa_g):
  return F.silu(g) * pa_v + dsilu(g) * v * pa_g


def _nanmean(a, axis=None):
  with warnings.catch_warnings():
    warnings.simplefilter('ignore', RuntimeWarning)
    return np.nanmean(a, axis)


def _div(a, b):
  with np.errstate(invalid='ignore', divide='ignore'):
    return np.where(b != 0, a / np.where(b != 0, b, 1), np.nan)


def roles_of(top, n_classes, rel=None):
  """{role: (C, U) bool}: top20, relatives (only with rel) and rest."""
  U = top.shape[0]
  inT = np.zeros((n_classes, U), bool)
  np.put_along_axis(inT, top.T, True, 0)
  relm = (rel[top].any(1).T & ~inT) if rel is not None else np.zeros_like(inT)
  out = {'top20': inT, 'rest': ~inT & ~relm}
  if rel is not None:
    out['relatives'] = relm
  return out


def _slope(curve):
  ok = np.isfinite(curve)
  return spearman(np.arange(len(curve))[ok], curve[ok]) if ok.sum() > 2 and np.ptp(curve[ok]) > 0 else float('nan')


def _unit_offsets(d, wgt):
  """Per unit the wgt-weighted mean of d over its finite cells (U, ...) -> (U,)."""
  d, wgt = d.reshape(len(d), -1), wgt.reshape(len(wgt), -1)
  wgt = np.where(np.isfinite(d), wgt, 0)
  return _div((wgt * np.nan_to_num(d)).sum(1), wgt.sum(1))


def _offset(d, wgt, live):
  """Per unit the wgt-weighted mean of d over its finite cells (U, ...), then the mean over live units."""
  return float(_nanmean(_unit_offsets(d, wgt)[live]))


@torch.no_grad()
def bin_edges(blk, inputs, geo, pv, pg, n_sample=16, edges='kappa', seed=0):
  """X1's pass 1. inputs: the block's input, batch by batch. Per unit the inner edges {v: (U, N_BINS - 1)} of the bin
  variables, from n_sample positions per image drawn by a generator seeded with seed (one draw per batch), |kappa|-
  weighted (edges 'kappa') or not ('positions'); the sums of s_in^2 and s_out^2 (2, U); the number of positions."""
  U = blk.pw_gate.out_features
  dev = geo['E'].device
  gen = torch.Generator().manual_seed(seed)
  samp = {k: [] for k in BIN_VARS + ('w',)}
  sq = torch.zeros(2, U, dtype=torch.float64, device=dev)
  n_pos = 0
  for xb in inputs:
    g, v, u = block_guv(blk, xb)
    sp = stream_parts(blk, xb, u, geo)
    B, P = sp['s_in'].shape[:2]
    kap = _kappa(g, v, pv, pg).flatten(1, 2)
    idx = torch.randint(P, (B, min(n_sample, P)), generator=gen).to(dev)[..., None].expand(-1, -1, U)
    for k, t in list(bin_values(sp, u).items()) + [('w', kap.abs())]:
      samp[k].append(t.gather(1, idx).flatten(0, 1))
    sq[0] += (sp['s_in'].double() ** 2).sum((0, 1))
    sq[1] += (sp['s_out'].double() ** 2).sum((0, 1))
    n_pos += B * P
  w = torch.cat(samp['w']) if edges == 'kappa' else torch.ones_like(torch.cat(samp['w']))
  return {k: weighted_edges(torch.cat(samp[k]), w).float().contiguous() for k in BIN_VARS}, sq, n_pos


def x1_block(model, name, X, y, rel=None, stats=None, batch=100, n_sample=16, edges='kappa', seed=0, top_n=20,
             active='image'):
  """X1 for one block. X: inputs of the block for all probe images (on the device), y: labels (same device).
  Returns (summary, raw)."""
  i, j = parse_block(name)
  blk = model.stages[i][j]
  dev = X.device
  C = int(y.max()) + 1
  U = blk.pw_gate.out_features
  R, G, V = stats if stats is not None else class_stats(model, name, X, y, C, batch)
  top, pa_v, pa_g, live = pole_coefs(R, G, V, top_n)
  pv, pg = (torch.from_numpy(a).float().to(dev) for a in (pa_v, pa_g))
  geo = write_geometry(blk)
  roles = roles_of(top, C, rel)
  ridx_np = np.full((C, U), 2)
  ridx_np[roles['top20']] = 0
  if 'relatives' in roles:
    ridx_np[roles['relatives']] = 1
  ncls = np.stack([(ridx_np == r).sum(0) for r in range(3)])                                 # (3, U) classes per role
  n_img_t = torch.bincount(y, minlength=C).double()
  thr = torch.from_numpy(0.5 * np.take_along_axis(R, top, 1).mean(1)).float().to(dev)       # --active pole
  def active_mask(uf):
    if active == 'pole':
      return (uf >= thr) & (thr > 0)
    ustar = uf.amax(1, keepdim=True)
    return (uf >= 0.5 * ustar) & (ustar > 0)
  # pass 1: bin edges from a sample of positions, and the RMS of the stream along e
  edg, sq, n_pos = bin_edges(blk, (X[s:s + batch] for s in range(0, len(X), batch)), geo, pv, pg, n_sample, edges,
                             seed)                                                           # (U, N_BINS - 1)
  # pass 2: dCE/du and the per-class sums (C, U[, cells]); bins are scattered per image first (few atomic collisions)
  f64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
  reg_keys = [(a_, b_) for a_ in range(3) for b_ in range(a_, 4)]                   # b_ = 3: the product with r
  cls_keys = (['push', 'push_act', 'n_act', 'den', 'n_star', 'den_rev'] + [f'push_{q}' for q in QUADS]
              + [f'{t}_{k}' for t in ('num', 'pstar', 'rev') for k in PARTS] + [f'reg_{a_}{b_}' for a_, b_ in reg_keys])
  acc = {k: f64(C, U) for k in cls_keys}
  b1 = {(t, k): f64(C, U, N_BINS) for t in BIN_T for k in BIN_VARS}
  b2 = {(t, a_, b_): f64(C, U, N_BINS * N_BINS) for t in BIN_T for a_, b_ in PAIRS}
  with no_param_grads(model):
    for s in range(0, len(X), batch):
      xb, yb = X[s:s + batch], y[s:s + batch]
      with torch.no_grad():
        g, v, u = block_guv(blk, xb)
      ul = u.detach().requires_grad_(True)
      keep = []
      ce = F.cross_entropy(tail(model, i, j, xb, ul, keep=keep).float(), yb, reduction='sum')
      delta, = torch.autograd.grad(ce, [ul])
      with torch.no_grad():
        sp = stream_parts(blk, xb, u, geo, keep[0])
        B, P = sp['s_in'].shape[:2]
        uf = u.flatten(1, 2)
        kap = _kappa(g, v, pv, pg).flatten(1, 2)
        d = delta.flatten(1, 2)
        push = -kap * d                                                                     # (B, P, U)
        r = d / geo['norms'].clamp(min=1e-30)                                               # e . dCE/dx_out
        ka = kap.abs()
        ustar, pstar = uf.max(1)
        star = ustar > 0
        act = active_mask(uf)
        wk = ka * act
        rev = (-push).clamp(min=0)
        add = lambda key, t: acc[key].index_add_(0, yb, t.double())
        add('push', push.sum(1))
        add('push_act', (push * act).sum(1))
        add('n_act', act.sum(1))
        add('den', wk.sum(1))
        add('n_star', star)
        add('den_rev', rev.sum(1))
        for q, qm in zip(QUADS, ((kap > 0) & (d > 0), (kap < 0) & (d < 0), (kap > 0) & (d < 0), (kap < 0) & (d > 0))):
          add(f'push_{q}', (push * qm).sum(1))
        for k in PARTS:
          add(f'num_{k}', (wk * sp[k]).sum(1))
          add(f'pstar_{k}', sp[k].gather(1, pstar[:, None])[:, 0] * star)
          add(f'rev_{k}', (rev * sp[k]).sum(1))
        Z = (torch.ones_like(r), sp['s_out'] - sp['s_own'], sp['s_own'], r)      # 1, s_excl, s_own; and r
        for a_, b_ in reg_keys:
          add(f'reg_{a_}{b_}', (ka * Z[a_] * Z[b_]).sum(1))
        vals = {t: x.double().permute(0, 2, 1).contiguous()                                  # (B, U, P)
                for t, x in (('push', push), ('cnt', torch.ones_like(push)), ('rw', ka * r), ('w', ka))}
        bins = {k: torch.searchsorted(edg[k], t.permute(2, 0, 1).reshape(U, -1).contiguous()).reshape(U, B, P)
                .permute(1, 0, 2).contiguous() for k, t in bin_values(sp, u).items()}       # (B, U, P)
        def scatter(ix, nb, dst):
          for t in BIN_T:
            dst[t].index_add_(0, yb, f64(B, U, nb).scatter_add_(2, ix, vals[t]))
        for k in BIN_VARS:
          scatter(bins[k], N_BINS, {t: b1[t, k] for t in BIN_T})
        for a_, b_ in PAIRS:
          scatter(bins[a_] * N_BINS + bins[b_], N_BINS * N_BINS, {t: b2[t, a_, b_] for t in BIN_T})
  P = n_pos // len(X)
  n_img = n_img_t.cpu().numpy()
  A = {k: t.cpu().numpy() for k, t in acc.items()}
  rw = np.stack([ridx_np == r for r in range(3)]) / np.maximum(n_img_t.cpu().numpy(), 1)[None, :, None]   # (3, C, U)
  by_role = lambda a: np.einsum('rcu,cu...->ru...', rw, a)        # role sums of class means (3, U, ...)
  B1 = {k: by_role(t.cpu().numpy()) for k, t in b1.items()}
  B2 = {k: by_role(t.cpu().numpy()).reshape(3, U, N_BINS, N_BINS) for k, t in b2.items()}
  del b1, b2
  REG = {k: by_role(A.pop(f'reg_{k[0]}{k[1]}')) for k in reg_keys}
  pole = A['push'] / n_img[:, None]
  scale = np.abs(pole).mean(0) + 1e-30
  S = {k: _div(A[f'num_{k}'], A['den']) for k in PARTS}                                    # (C, U), NaN: never active
  Sp = {k: _div(A[f'pstar_{k}'], A['n_star']) for k in PARTS}
  Sr = {k: _div(A[f'rev_{k}'], A['den_rev']) for k in PARTS}
  for D in (S, Sp, Sr):
    D['s_end'] = D['s_out'] + D['s_later']
  raw = {'top': top, 'pa_v': pa_v, 'pa_g': pa_g, 'live': live, 'R': R, 'G': G, 'V': V, 'n_img': n_img, 'P': P,
         'pole': pole, 'roles': ROLES, 'n_classes_role': ncls, 'active': active,
         'edges': {k: e.cpu().numpy() for k, e in edg.items()}, 'edges_weight': edges, **A,
         **{f'bin_{t}_{k}': a for (t, k), a in B1.items()}, **{f'bin2_{t}_{a}_{b}': x for (t, a, b), x in B2.items()}}
  rm = lambda M, mask: float(_nanmean(_nanmean(np.where(mask, M, np.nan), 0)[live]))
  um = lambda M: _nanmean(M[live], 0)                                                       # mean over live units
  out = {'units': int(live.sum()), 'edges_weight': edges, 'active': active, 'positions': P, **geometry_stats(blk),
         'rms_sin_all': float(np.sqrt(sq[0].cpu().numpy() / n_pos)[live].mean()),
         'rms_sout_all': float(np.sqrt(sq[1].cpu().numpy() / n_pos)[live].mean())}
  si, so, sw, sb, sx = (S[k] for k in ('s_in', 's_own', 's_part', 's_bias', 's_out'))
  se = sx - so
  fin = np.isfinite(sx)
  frac = lambda cond: np.where(fin, cond.astype(float), np.nan)
  ppp, occ, rr, rr2 = {}, {}, {}, {}
  for role, mask in roles.items():
    ri = ROLES.index(role)
    nc = ncls[ri][:, None]                                                                  # (U, 1)
    out[f'pole_{role}'] = rm(pole / scale[None], mask)
    for q in QUADS:
      out[f'pole_{q}_{role}'] = rm(A[f'push_{q}'] / n_img[:, None] / scale[None], mask)
    out[f'push_act_{role}'] = rm(A['push_act'] / n_img[:, None] / scale[None], mask)
    out[f'n_act_{role}'] = rm(A['n_act'] / n_img[:, None], mask)
    for k in PARTS + ('s_end',):
      out[f'{k}_{role}'] = rm(S[k], mask)
      out[f'pstar_{k}_{role}'] = rm(Sp[k], mask)
      out[f'{k}_rev_{role}'] = rm(Sr[k], mask)
    with np.errstate(invalid='ignore'):
      out[f'frac_sout_pos_{role}'] = rm(frac(sx > 0), mask)
      out[f'frac_m1_{role}'] = rm(frac((sx > 0) & (se > 0.5 * sx)), mask)
      out[f'part_share_{role}'] = rm(sw / (np.abs(si) + np.abs(so) + np.abs(sw) + np.abs(sb) + 1e-30), mask)
      out[f'frac_sin_neg_{role}'] = rm(frac(si < 0), mask)
      out[f'frac_own_shrinks_{role}'] = rm(frac(np.abs(sx) < np.abs(se)), mask)
      out[f'frac_m4_{role}'] = rm(frac((si < 0) & (np.abs(sx) < np.abs(se))), mask)
      out[f'frac_m4_in_{role}'] = rm(frac((si < 0) & (np.abs(si + so) < np.abs(si))), mask)
    for k in BIN_VARS:
      bp_, bc, brw, bw = (B1[t, k][ri] for t in BIN_T)                                       # (U, N_BINS)
      ppp[role, k] = _div(bp_, bc) * P / scale[:, None]
      occ[role, k] = _div(bc, nc * P)
      rr[role, k] = _div(brw, bw) * P
      out[f'push_by_{k}_bin_{role}'] = [float(x) for x in um(_div(bp_, nc) / scale[:, None])]
      for key, M in ((f'push_per_pos_by_{k}_bin_{role}', ppp[role, k]), (f'occ_by_{k}_bin_{role}', occ[role, k]),
                     (f'r_by_{k}_bin_{role}', rr[role, k])):
        out[key] = [float(x) for x in um(M)]
      out[f'slope_by_{k}_{role}'] = _slope(um(ppp[role, k]))
      out[f'slope_r_by_{k}_{role}'] = _slope(um(rr[role, k]))
    for a_, b_ in PAIRS:
      rr2[role, a_, b_] = _div(B2['rw', a_, b_][ri], B2['w', a_, b_][ri]) * P                 # (U, N_BINS, N_BINS)
      grid = um(rr2[role, a_, b_])
      out[f'slope_r_by_{b_}_within_{a_}_{role}'] = float(_nanmean(np.array([_slope(row) for row in grid])))
    XX = np.zeros((U, 3, 3))
    XR = np.zeros((U, 3))
    for (a_, b_), t in REG.items():
      if b_ == 3:
        XR[:, a_] = t[ri]
      else:
        XX[:, a_, b_] = XX[:, b_, a_] = t[ri]
    beta = np.full((U, 3), np.nan)
    for n in np.flatnonzero(live):
      if XX[n, 0, 0] > 0 and np.linalg.cond(XX[n]) < 1e12:
        beta[n] = np.linalg.solve(XX[n], XR[n]) * P
    raw[f'reg_{role}'] = beta
    for c_, nm in enumerate(('r0', 'b_excl', 'b_own')):
      ok = np.isfinite(beta[:, c_])
      out[f'{nm}_{role}'] = float(np.median(beta[ok, c_])) if ok.any() else float('nan')
    w0 = XX[:, 0, 0]
    with np.errstate(invalid='ignore', divide='ignore'):
      var = [XX[:, a_, a_] / w0 - (XX[:, 0, a_] / w0) ** 2 for a_ in (1, 2)]
      ratio = np.sqrt(np.clip(var[1], 0, None) / var[0])[live]
    ok = np.isfinite(ratio)
    out[f'sd_own_over_excl_{role}'] = float(np.median(ratio[ok])) if ok.any() else float('nan')
  for k in BIN_VARS:                                              # top20 minus rest in the same bin
    out[f'role_offset_by_{k}'] = _offset(ppp['top20', k] - ppp['rest', k], occ['top20', k], live)
    out[f'role_offset_r_by_{k}'] = _offset(rr['top20', k] - rr['rest', k], B1['w', k][0], live)
  for a_, b_ in PAIRS:
    out[f'role_offset_r_by_{a_}_{b_}'] = _offset(rr2['top20', a_, b_] - rr2['rest', a_, b_], B2['w', a_, b_][0], live)
  return out, raw


@torch.no_grad()
def partner_writes(model, i, j, X, n, geos, batch):
  """Unit n's active positions (u_n >= 0.5 u_n* in the image, u_n* > 0) on the images X (inputs of block (i, j)).
  Returns c {j2: (U,)}: every unit's mean write along e_n there (mean u_m times w_m . e_n) for block j and the later
  blocks in geos; size {j2: (U,)}: mean |u_m| |w_m|; the mean s_out = x_out . e_n there; and the number of active
  positions (0: c and size are 0)."""
  stage = model.stages[i]
  e = geos[j]['E'][n]
  su = {j2: 0.0 for j2 in geos}
  sa = {j2: 0.0 for j2 in geos}
  cnt, s_out = 0.0, 0.0
  for s in range(0, len(X), batch):
    x = X[s:s + batch]
    act = None
    for j2 in sorted(geos):
      u = block_units(stage[j2], x)
      uf = u.flatten(1, 2)
      if j2 == j:
        un = uf[..., n]
        mx = un.amax(1, keepdim=True)
        act = ((un >= 0.5 * mx) & (mx > 0)).float()                                       # (B, P)
      su[j2] = su[j2] + (uf * act[..., None]).sum((0, 1))
      sa[j2] = sa[j2] + (uf.abs() * act[..., None]).sum((0, 1))
      x = block_out(stage[j2], x, u)
      if j2 == j:
        s_out += float(((x.permute(0, 2, 3, 1).flatten(1, 2) @ e) * act).sum())
    cnt += float(act.sum())
  n_act, cnt = int(cnt), max(cnt, 1.0)
  c = {j2: (su[j2] / cnt * (geos[j2]['W'] @ e)).cpu().numpy() for j2 in geos}
  size = {j2: (sa[j2] / cnt * geos[j2]['norms']).cpu().numpy() for j2 in geos}
  return c, size, s_out / cnt, n_act


def x2_sets(n, j, cos, c, size, live, rng, cos_hi=0.9, cos_lo=0.5):
  """{set: {j2: unit indices}} for splus, sminus, later and their size-matched controls ctrl_* (see the module
  docstring). cos, c, size, live: {j2: (U,)} for block j and the later blocks."""
  sets = {k: {} for k in X2_SETS}
  for j2 in sorted(cos):
    hi = np.abs(cos[j2]) > cos_hi
    if j2 == j:
      hi &= np.arange(len(hi)) != n
      sets['splus'][j2], sets['sminus'][j2] = np.flatnonzero(hi & (c[j2] > 0)), np.flatnonzero(hi & (c[j2] < 0))
    else:
      sets['later'][j2] = np.flatnonzero(hi & (c[j2] > 0))
  out = {}
  for k in X2_SETS:
    S = {j2: m for j2, m in sets[k].items() if len(m)}
    ctrl = {}
    for j2, mem in S.items():
      cand = (np.abs(cos[j2]) < cos_lo) & live[j2] & (size[j2] > 0)
      if j2 == j:
        cand &= np.arange(len(cand)) != n
      pool, pick = list(np.flatnonzero(cand)), []
      for m in rng.permutation(mem):
        if not pool:
          break
        dist = np.abs(np.log(size[j2][pool]) - np.log(max(size[j2][m], 1e-30)))
        pick.append(pool.pop(int(np.argmin(dist))))
      if pick:
        ctrl[j2] = np.sort(np.array(pick))
    out[k], out[f'ctrl_{k}'] = S, ctrl
  return out


def x2_push(model, name, X, y, by_class, cls, n, configs, pa_v, pa_g, batch=100):
  """Class-mean pole push of unit n on the classes cls under each config: None (baseline); ('units', {j2: indices},
  alpha): those units of block j2 (this block or a later one of the stage) scaled by alpha; ('e_only', indices, alpha):
  only their component along e_n scaled; ('perp_only', indices, alpha): only the rest of their writes (orthogonal to
  e_n) scaled; ('beta', b): the stream along e_n at the block's output scaled by b.
  Returns ({key: push (len(cls),)}, {key: class-mean CE (len(cls),)})."""
  i, j = parse_block(name)
  stage = model.stages[i]
  dev = X.device
  U = stage[j].pw_gate.out_features
  geo = write_geometry(stage[j])
  e = geo['E'][n].view(1, -1, 1, 1)
  idx = np.concatenate([by_class[c] for c in cls])
  pos = torch.from_numpy(np.repeat(np.arange(len(cls)), [len(by_class[c]) for c in cls])).to(dev)
  as_idx = lambda a: torch.as_tensor(np.asarray(a, dtype=np.int64), device=dev)
  plans = {}
  for key, cfg in configs.items():
    sc, later, fn, div = torch.ones(U, device=dev), {}, None, 1.0
    if cfg is None:
      pass
    elif cfg[0] == 'units':
      for j2, members in cfg[1].items():
        t = sc if j2 == j else later.setdefault(j2, torch.ones(stage[j2].pw_gate.out_features, device=dev))
        t[as_idx(members)] = float(cfg[2])
    elif cfg[0] == 'e_only':
      Sx, al = as_idx(cfg[1]), float(cfg[2])
      fn = lambda o, u, Sx=Sx, Mn=geo['M'][Sx, n], al=al: o - (1 - al) * (u[..., Sx] @ Mn)[:, None] * e
    elif cfg[0] == 'perp_only':
      Sx, al = as_idx(cfg[1]), float(cfg[2])
      Wp = geo['W'][Sx] - (geo['W'][Sx] @ e.flatten())[:, None] * e.flatten()[None]      # (|S|, d), orthogonal to e
      fn = lambda o, u, Sx=Sx, Wp=Wp, al=al: o - (1 - al) * (u[..., Sx] @ Wp).permute(0, 3, 1, 2)
    elif cfg[0] == 'beta':
      b = float(cfg[1])
      fn, div = (lambda o, u, b=b: o + (b - 1) * (o * e).sum(1, keepdim=True) * e), b
    else:
      raise ValueError(f'unknown X2 config {cfg!r}')
    plans[key] = (sc, later, fn, div)
  acc = {k: torch.zeros(len(cls), dtype=torch.float64, device=dev) for k in configs}
  acc_ce = {k: torch.zeros(len(cls), dtype=torch.float64, device=dev) for k in configs}
  with no_param_grads(model):
    for s in range(0, len(idx), batch):
      ib = torch.from_numpy(idx[s:s + batch]).to(dev)
      xb, yb, pb = X[ib], y[ib], pos[s:s + batch]
      with torch.no_grad():
        g, v, u = block_guv(stage[j], xb)
        kap = _kappa(g[..., n], v[..., n], float(pa_v), float(pa_g))                      # (B, H, W)
      for key, (sc, later, fn, div) in plans.items():
        ul = u.detach().requires_grad_(True)
        cei = F.cross_entropy(tail(model, i, j, xb, ul * sc, fn, later).float(), yb, reduction='none')
        d, = torch.autograd.grad(cei.sum(), [ul])
        acc[key].index_add_(0, pb, (-(kap * d[..., n] / div).flatten(1).sum(1)).double())
        acc_ce[key].index_add_(0, pb, cei.detach().double())
  cnt = torch.bincount(pos, minlength=len(cls)).double()
  return ({k: (a / cnt).cpu().numpy() for k, a in acc.items()}, {k: (a / cnt).cpu().numpy() for k, a in acc_ce.items()})


def _mean(xs):
  return float(np.mean(xs)) if len(xs) else float('nan')


def x2_block(model, name, X, y, rel=None, stats=None, alphas=(0.0, 0.5), betas=(0.8, 0.9, 1.1), n_top=20, n_other=40,
             batch=100, seed=0, cos_hi=0.9, cos_lo=0.5, units=None, later=True):
  """X2 for one block (see the module docstring). units: optional subset of unit indices; later: also the later
  blocks' partners. Returns (summary, rows)."""
  i, j = parse_block(name)
  stage = model.stages[i]
  y_np = y.cpu().numpy()
  C = int(y_np.max()) + 1
  R, G, V = stats if stats is not None else class_stats(model, name, X, y, C, batch)
  top, pa_v, pa_g, live = pole_coefs(R, G, V, n_top)
  geos = {j2: write_geometry(stage[j2]) for j2 in range(j, len(stage) if later else j + 1)}
  lives = {j: live, **(later_live(model, i, j, X, y, C, batch) if later else {})}
  by_class = [np.flatnonzero(y_np == c) for c in range(C)]
  rows = []
  for n in (np.flatnonzero(live) if units is None else units):
    rng = np.random.default_rng([seed, int(n)])           # per unit: its classes do not depend on the other units
    others = np.setdiff1d(np.arange(C), top[n])
    if rel is not None:
      far = others[~rel[top[n]].any(0)[others]]
      others = far if len(far) >= n_other else others
    oth = rng.choice(others, min(n_other, len(others)), replace=False)
    cls = np.concatenate([top[n], oth])
    ib = torch.from_numpy(np.concatenate([by_class[c] for c in top[n]])).to(X.device)
    c, size, s_act, n_act = partner_writes(model, i, j, X[ib], n, geos, batch)
    cos = {j2: (g['E'] @ geos[j]['E'][n]).cpu().numpy() for j2, g in geos.items()}
    sets = x2_sets(n, j, cos, c, size, lives, np.random.default_rng([seed, int(n), 1]), cos_hi, cos_lo)
    configs = {'base': None}
    for k, S in sets.items():
      if S:
        configs.update({f'{k}_a{al:g}': ('units', S, al) for al in alphas})
    if sets['splus']:
      configs.update({f'splus_{t}_a{al:g}': (f'{t}_only', sets['splus'][j], al)
                      for t in ('e', 'perp') for al in alphas})
    configs.update({f'beta{b:g}': ('beta', b) for b in betas})
    push, ce = x2_push(model, name, X, y, by_class, cls, n, configs, pa_v[n], pa_g[n], batch)
    norm = np.abs(push['base']).mean() + 1e-30
    rec = {'unit': int(n), 'n_act': n_act, 's_out_act': s_act, 'ce_top20_base': float(ce['base'][:n_top].mean()),
           'ce_rest_base': float(ce['base'][n_top:].mean())}
    for k, S in sets.items():
      rec[f'n_{k}'] = int(sum(len(m) for m in S.values()))
      rec[f'ds_e_{k}'] = float(sum(c[j2][m].sum() for j2, m in S.items()))
    for key in configs:
      rec[f'top20_{key}'] = float(push[key][:n_top].mean() / norm)
      rec[f'rest_{key}'] = float(push[key][n_top:].mean() / norm)
      if key != 'base':
        rec[f'dce_top20_{key}'] = float((ce[key] - ce['base'])[:n_top].mean())
        rec[f'dce_rest_{key}'] = float((ce[key] - ce['base'])[n_top:].mean())
    rows.append(rec)
  out = {'units': len(rows), **{k: _mean([r[k] for r in rows]) for k in ('top20_base', 'rest_base', 'ce_top20_base',
                                                                          'ce_rest_base', 's_out_act')}}
  keys = sorted({k[len('dce_top20_'):] for r in rows for k in r if k.startswith('dce_top20_')})
  for key in keys:
    sel = [r for r in rows if f'top20_{key}' in r]
    out[f'units_{key}'] = len(sel)
    for f in ('top20', 'rest', 'dce_top20', 'dce_rest'):
      out[f'{f}_{key}'] = _mean([r[f'{f}_{key}'] for r in sel])
    out[f'top20_base_{key}'] = _mean([r['top20_base'] for r in sel])
    out[f'share_up_{key}'] = _mean([r[f'top20_{key}'] > r['top20_base'] for r in sel])
  for k in X2_SETS + tuple(f'ctrl_{k}' for k in X2_SETS):
    sel = [r for r in rows if r[f'n_{k}'] > 0]
    out[f'units_{k}'], out[f'n_empty_{k}'] = len(sel), len(rows) - len(sel)
    out[f'size_{k}'] = _mean([r[f'n_{k}'] for r in sel])
    out[f'ds_e_{k}'] = _mean([r[f'ds_e_{k}'] for r in sel])
  for k in X2_SETS:
    out[f'n_short_ctrl_{k}'] = int(sum(r[f'n_ctrl_{k}'] < r[f'n_{k}'] for r in rows))
  for al in alphas:
    for name_, a_, b_ in [(f'{k}_vs_ctrl', f'{k}_a{al:g}', f'ctrl_{k}_a{al:g}') for k in X2_SETS] + \
                         [('e_vs_perp', f'splus_e_a{al:g}', f'splus_perp_a{al:g}')]:
      d = [r[f'top20_{a_}'] - r[f'top20_{b_}'] for r in rows if f'top20_{a_}' in r and f'top20_{b_}' in r]
      out[f'd_{name_}_a{al:g}'] = _mean(d)
      out[f'share_{name_.replace("_vs_", "_above_")}_a{al:g}'] = _mean([x > 0 for x in d])
  return out, rows


def _logits_dtype(z):
  return z.to(torch.promote_types(z.dtype, torch.float32))


X4_LOSS = {'ce': lambda z, y: F.cross_entropy(_logits_dtype(z), y, reduction='sum'),
           'true': lambda z, y: -_logits_dtype(z).gather(1, y[:, None]).sum(),          # the true-class term -z[k]
           'lse': lambda z, y: torch.logsumexp(_logits_dtype(z), 1).sum()}              # ce = true + lse


def x4_streams(model, i, j0, x):
  """The streams from block (i, j0)'s output x: [x_j0, ..., x_S] of stage i, per later stage [y_0 (after its
  downsample), y_1, ..., y_K] (after each block), and the logits."""
  xs = [x]
  for blk in model.stages[i][j0 + 1:]:
    xs.append(blk(xs[-1]))
  later, h = [], xs[-1]
  for down, stage in zip(model.downsample[i + 1:], model.stages[i + 1:]):
    ys = [down(h)]
    for blk in stage:
      ys.append(blk(ys[-1]))
    later.append(ys)
    h = ys[-1]
  return xs, later, model.head(model.norm(h.mean((2, 3))))


@contextmanager
def no_tf32():
  """TF32 off while inside (the flags are restored after): X4 pulls each reader's term back through a downsample in its
  own conv backward, and TF32 (cuDNN's default for convolutions) rounds each cotangent separately, so the readers would
  add up to the total only to about 1e-3."""
  flags = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
  torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
  try:
    yield
  finally:
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = flags


def x4_check(model, names):
  """(i, sorted js) for the X4 blocks names, or ValueError: existing blocks, all of one stage i, residual blocks from
  the first of them on (only the network's last block may lack the skip)."""
  try:
    ij = sorted({parse_block(b) for b in names})
  except ValueError:
    raise ValueError(f'X4 blocks are named stage{{i}}.block{{j}}: {names}') from None
  if not ij:
    raise ValueError('X4: no blocks')
  i = ij[0][0]
  if any(t != i for t, _ in ij):
    raise ValueError(f'X4: blocks of one stage: {names}')
  if not 0 <= i < len(model.stages) or any(not 0 <= j < len(model.stages[i]) for _, j in ij):
    raise ValueError(f'X4: no such block in {names} (stage depths {[len(st) for st in model.stages]})')
  last = model.stages[-1][-1]
  if any(not b.skip for b in model.stages[i][ij[0][1]:]) or \
     any(not b.skip and b is not last for st in model.stages[i + 1:] for b in st):
    raise ValueError(f'X4: a block without skip after {ij[0]}')
  return i, [j for _, j in ij]


def x4_terms(model, i, j0, x, y, losses=('ce', 'true'), sigma=False):
  """X4's split of dL/dx for one batch (see the module docstring). x: block (i, j0)'s output (B, d, H, W); losses: keys
  of X4_LOSS (summed over the batch); sigma: forward (and so backward) with every LayerNorm's 1 / sigma held fixed.
  Returns {loss: (g, T, T1)}: g = [dL/dx_j0, ..., dL/dx_S] (one autograd call over all streams), T = {reader: term
  pulled back to x_S} for the readers in the later stages ('stage{t}.block{l}': D^T (zeta_{l-1} - zeta_l)) and 'head';
  T1 = {block: its single path, pulled back to x_S} for the stage's blocks after j0 and the later stages' blocks:
  D^T (J_f^T c), c the head's skip-only cotangent at the block's output (the head term there; zeta_L at z_L)."""
  stage = model.stages[i]
  assert all(b.skip for b in stage[j0:]), 'X4: residual blocks in the unit\'s stage'
  def single(o, x_, c, skip=True):                              # J_f^T c of the block x_ -> o
    gc, = torch.autograd.grad(o, x_, c, retain_graph=True)
    return gc - c if skip else gc
  with torch.enable_grad():
    x = x.detach().requires_grad_(True)
    with fixed_sigma() if sigma else nullcontext():
      xs, later, z = x4_streams(model, i, j0, x)
    flat = xs + [t for ys in later for t in ys]
    out = {}
    for loss in losses:
      grads = torch.autograd.grad(X4_LOSS[loss](z, y), flat, retain_graph=True)
      g, pos = list(grads[:len(xs)]), len(flat)
      T, T1 = ({}, {}) if later else ({'head': g[-1]}, {})
      for t in reversed(range(len(later))):                     # last stage first; then pulled back through D_t
        ys, st = later[t], model.stages[i + 1 + t]
        gy, pos = grads[pos - len(ys):pos], pos - len(ys)
        for l, blk in enumerate(st):                            # only the network's last block may lack the skip
          assert blk.skip or (t == len(later) - 1 and l == len(st) - 1), 'X4: a block without skip inside'
          T[f'stage{i + 1 + t}.block{l}'] = gy[l] - gy[l + 1] if blk.skip else gy[l]
        if t == len(later) - 1:
          T['head'] = gy[-1] if st[-1].skip else torch.zeros_like(gy[-1])
          T1[f'stage{i + 1 + t}.block{len(st) - 1}'] = T[f'stage{i + 1 + t}.block{len(st) - 1}']   # its only path
        for l in range(len(st) - (t == len(later) - 1)):
          T1[f'stage{i + 1 + t}.block{l}'] = single(ys[l + 1], ys[l], T['head'])
        prev = later[t - 1][-1] if t else xs[-1]
        pull = lambda D: {R: torch.autograd.grad(ys[0], prev, v, retain_graph=True)[0] for R, v in D.items()}
        T, T1 = pull(T), pull(T1)
      for j2 in range(j0 + 1, j0 + len(xs)):
        T1[f'stage{i}.block{j2}'] = single(xs[j2 - j0], xs[j2 - 1 - j0], T['head'])
      out[loss] = (g, T, T1)
  return out


def _reader_key(R):
  return (1, 0, 0) if R == 'head' else (0,) + parse_block(R)


def x4_readers(i, j, j0, g, T):
  """{reader: term at x_j} for block (i, j)'s output (j >= j0; g, T from x4_terms with j0): the stage's later blocks
  j2 (g_{j2-1} - g_{j2}: pure skips from x_j), then T, in network order. They add up to g[j - j0]."""
  out = {f'stage{i}.block{j2}': g[j2 - 1 - j0] - g[j2 - j0] for j2 in range(j + 1, j0 + len(g))}
  out.update({R: T[R] for R in sorted(T, key=_reader_key)})
  return out


def x4_single(i, j, T1):
  """{'{block}>head': single path at x_j} for block (i, j)'s output (T1 from x4_terms): the blocks after it, in network
  order."""
  return {f'{R}>head': T1[R] for R in sorted(T1, key=_reader_key) if parse_block(R) > (i, j)}


@torch.no_grad()
def head_static(model, i):
  """(C, d_i) float64: the true-class cotangent -W_head[k] pulled back to stage i's stream along the head's path (pure
  skips), every LayerNorm's 1 / sigma left out and the downsample convs averaged over their 2x2 phases (see the module
  docstring); None without that path (a last block without skip)."""
  if not model.stages[-1][-1].skip:
    return None
  def ln_back(c, ln):                                           # gain, then centering over the channels
    c = c * ln.weight.double() if ln.weight is not None else c
    return c - c.mean(-1, keepdim=True)
  c = ln_back(-model.head.weight.double(), model.norm)
  for t in reversed(range(i + 1, len(model.stages))):
    ln, conv = model.downsample[t]
    c = ln_back(c @ conv.weight.double().mean((2, 3)), ln)
  return c


def _x4_sum_err(d, readers, roles):
  """Largest |sum over the readers - total| over X4's linear keys, relative to the largest |value| (inf: the NaN
  patterns differ)."""
  err, scale = 0.0, 1e-30
  for f in ['role_offset_r_{}_by_sout', 'role_offset_r_{}_by_sout_kpos'] + \
           [p + role for role in roles for p in ('r0_{}_', 'r_{}_by_sout_bin_')]:
    tot = np.asarray(d[f.format('total')], float)
    parts = np.array([d[f.format(R)] for R in readers], float).reshape((len(readers),) + tot.shape)
    if (np.isnan(parts.sum(0)) != np.isnan(tot)).any():
      return float('inf')
    err = max(err, np.nanmax(np.r_[0.0, np.abs(parts.sum(0) - tot).ravel()]))
    scale = max(scale, np.nanmax(np.r_[0.0, np.abs(parts).ravel(), np.abs(tot).ravel()]))
  return float(err / scale)


def _unit_offset_vecs(d, wgt):
  """_unit_offsets for vectors: d (U, N, k), wgt (U, N) -> (U, k), over the cells where d is finite."""
  wgt = np.where(np.isfinite(d).all(-1), wgt, 0)
  return _div(np.einsum('un,unk->uk', wgt, np.nan_to_num(d)), wgt.sum(1)[:, None])


def x4_block(model, names, X, y, rel=None, stats=None, batch=100, n_sample=16, edges='kappa', seed=0, top_n=20,
             sigma=False, tol=1e-3):
  """X4 for the blocks names, all of one stage i, in one pass (the later stages' reader terms are shared). X: the
  stage's input (block (i, 0)'s input) for all probe images (on the device), y: labels (same device); stats: optional
  {name: (R, G, V)} as class_stats. Live units, roles, |kappa| and s_out bins as x1_block with the same batch,
  n_sample, edges, seed and top_n; the backward pass with TF32 off. Returns {name: (summary, raw)}."""
  i, js = x4_check(model, names)
  stage, j0, dev = model.stages[i], js[0], X.device
  C = int(y.max()) + 1
  n_img_t = torch.bincount(y, minlength=C).double()
  losses = X4_LOSSES + (tuple(f'{k}_fixed' for k in X4_LOSSES) if sigma else ())
  inputs = lambda j: (stage[:j](X[s:s + batch]) for s in range(0, len(X), batch))      # block (i, j)'s input
  later = [f'stage{t}.block{l}' for t in range(i + 1, len(model.stages)) for l in range(len(model.stages[t]))]
  f64 = lambda *s: torch.zeros(*s, dtype=torch.float64, device=dev)
  proj = lambda t, E: t.permute(0, 2, 3, 1).flatten(1, 2) @ E.T                            # (B, P, U)
  qs = head_static(model, i)
  S = {}
  with torch.no_grad():
    for j in js:
      name, blk = f'stage{i}.block{j}', stage[j]
      U = blk.pw_gate.out_features
      st = stats[name] if stats and name in stats else class_stats(model, name, torch.cat(list(inputs(j))), y, C,
                                                                     batch)
      top, pa_v, pa_g, live = pole_coefs(*st, top_n)
      pv, pg = (torch.from_numpy(a).float().to(dev) for a in (pa_v, pa_g))
      geo = write_geometry(blk)
      edg, _, n_pos = bin_edges(blk, inputs(j), geo, pv, pg, n_sample, edges, seed)
      roles = roles_of(top, C, rel)
      ridx = np.select([roles['top20'], roles.get('relatives', np.zeros_like(roles['top20']))], [0, 1], 2)
      E = geo['E']
      Z = torch.from_numpy(np.random.default_rng([seed, i, j]).standard_normal(tuple(E.shape))).to(E)
      Z = Z - (Z * E).sum(1, keepdim=True) * E                                              # e_perp: random, _|_ e
      readers = [f'stage{i}.block{j2}' for j2 in range(j + 1, len(stage))] + later + ['head']
      groups = {f'stage{i}_later': [R for R in readers if R.startswith(f'stage{i}.')],
                **{f'stage{t}': [R for R in readers if R.startswith(f'stage{t}.')]
                   for t in range(i + 1, len(model.stages))}, 'head': ['head']}
      single = [f'{R}>head' for R in readers[:-1]]
      S[j] = dict(name=name, blk=blk, top=top, pa_v=pa_v, pa_g=pa_g, live=live, pv=pv, pg=pg, E=E,
                  Ep=Z / Z.norm(dim=1, keepdim=True), edg=edg['sout'], P=n_pos // len(X), roles=roles,
                  ridx=torch.from_numpy(ridx).to(dev), readers=readers, groups=groups, single=single,
                  w=f64(3, U, 2 * N_BINS),                     # bins [N_BINS:]: the positions with kappa > 0
                  rw={k: {R: f64(3, U, 2 * N_BINS) for R in readers + single + ['total', 'perp']} for k in losses},
                  rv=f64(3, U, N_BINS, len(losses) * (len(groups) + 1) * E.shape[1]))
  # one pass: the streams once, the reader terms per loss, then each block's bins (role sums of class means)
  with no_param_grads(model), no_tf32():
    for s in range(0, len(X), batch):
      yb = y[s:s + batch]
      with torch.no_grad():
        xin = [X[s:s + batch]]                                    # xin[j]: block j's input, xin[j + 1]: its output
        for blk in stage[:js[-1] + 1]:
          xin.append(blk(xin[-1]))
      terms = x4_terms(model, i, j0, xin[j0 + 1], yb, X4_LOSSES)
      if sigma:
        terms.update({f'{k}_fixed': t for k, t in x4_terms(model, i, j0, xin[j0 + 1], yb, X4_LOSSES, True).items()})
      with torch.no_grad():
        wimg = 1 / n_img_t[yb]
        for j in js:
          st = S[j]
          g, v, u = block_guv(st['blk'], xin[j])
          kap = _kappa(g, v, st['pv'], st['pg']).flatten(1, 2)
          ka = kap.abs()
          s_out = proj(block_out(st['blk'], xin[j], u), st['E'])
          B, P, U = s_out.shape
          ix = torch.searchsorted(st['edg'], s_out.permute(2, 0, 1).reshape(U, -1).contiguous()).reshape(U, B, P) \
              .permute(1, 0, 2).contiguous()                                                 # (B, U, P)
          ix2 = ix + N_BINS * (kap > 0).permute(0, 2, 1)
          oh = F.one_hot(st['ridx'][yb], 3).double() * wimg[:, None, None]                  # (B, U, 3)
          def add(acc, t):                                                                  # t (B, P, U)
            acc += torch.einsum('bur,bun->run', oh,
                                f64(B, U, 2 * N_BINS).scatter_add_(2, ix2, t.double().permute(0, 2, 1)))
          add(st['w'], ka)
          vecs = []
          for k in losses:
            gl, T, T1 = terms[k]
            rd = x4_readers(i, j, j0, gl, T)
            for R, t in {**rd, **x4_single(i, j, T1), 'total': gl[j - j0]}.items():
              add(st['rw'][k][R], ka * proj(t, st['E']))
            add(st['rw'][k]['perp'], ka * proj(gl[j - j0], st['Ep']))
            vecs += [sum((rd[R] for R in mem), torch.zeros_like(gl[0])) for mem in st['groups'].values()]
            vecs.append(gl[j - j0])
          V = torch.cat([t.permute(0, 2, 3, 1).flatten(1, 2) for t in vecs], 2).double()     # (B, P, K)
          kau = ka.permute(0, 2, 1).double()                                                # (B, U, P)
          for n in range(N_BINS):                                     # |kappa| term_G per bin, all d components
            st['rv'][:, :, n] += torch.einsum('bur,buk->ruk', oh, torch.bmm(kau * (ix == n), V))
  out = {}
  for j in js:
    st = S[j]
    live, P, roles, readers, groups = st['live'], st['P'], st['roles'], st['readers'], st['groups']
    E = st['E'].double().cpu().numpy()
    U, dim = E.shape
    sgroups = {f'{G}>head': [f'{R}>head' for R in mem] for G, mem in groups.items() if G != 'head'}
    w2 = st['w'].cpu().numpy()
    bw = w2[..., :N_BINS] + w2[..., N_BINS:]
    um = lambda M: _nanmean(M[live], 0)
    summ = {'units': int(live.sum()), 'positions': P, 'edges_weight': edges, 'readers': readers, 'groups': groups,
            'single': st['single'], 'single_groups': sgroups}
    raw = {'top': st['top'], 'pa_v': st['pa_v'], 'pa_g': st['pa_g'], 'live': live, 'P': P, 'roles': ROLES,
           'edges_sout': st['edg'].cpu().numpy(), 'e_perp': st['Ep'].cpu().numpy(), 'bin_w': bw}
    stat = None
    if qs is not None:                                          # the classifier's rows along e: weights only
      q = qs[:C].cpu().numpy() @ E.T                                    # (C, U): the probe's classes
      cm = lambda m: (q * m).sum(0) / np.maximum(m.sum(0), 1)
      stat = cm(roles['top20']) - cm(roles['rest'])
      summ['static_head_offset'] = float(_nanmean(_div(stat, q.std(0))[live]))
      summ['static_head_pos'] = float((stat[live] > 0).mean())
      raw['static_q'], raw['static_offset'] = q.astype(np.float32), stat
    rv = st['rv'].cpu().numpy().reshape(3, U, N_BINS, len(losses), len(groups) + 1, dim)
    for li, k in enumerate(losses):
      a2 = {R: a.cpu().numpy() for R, a in st['rw'][k].items()}
      for gname, mem in {**groups, **sgroups}.items():
        a2.setdefault(gname, sum((a2[m] for m in mem), np.zeros_like(w2)))
      brw = {R: a[..., :N_BINS] + a[..., N_BINS:] for R, a in a2.items()}
      raw[f'bin_rw_{k}'], raw[f'bin_rwp_{k}'] = brw, {R: a[..., N_BINS:] for R, a in a2.items()}
      d, uo = {}, {}
      for R, a in brw.items():
        rr = _div(a, bw) * P                                                                # (3, U, N_BINS)
        for role in roles:
          ri = ROLES.index(role)
          d[f'r_{R}_by_sout_bin_{role}'] = [float(x) for x in um(rr[ri])]
          d[f'r0_{R}_{role}'] = float(_nanmean((_div(a[ri].sum(-1), bw[ri].sum(-1)) * P)[live]))
        uo[R] = _unit_offsets(rr[0] - rr[2], bw[0])
        d[f'role_offset_r_{R}_by_sout'] = float(_nanmean(uo[R][live]))
        for q_, h in (('kpos', a2[R][..., N_BINS:]), ('kneg', a2[R][..., :N_BINS])):
          rq = _div(h, bw) * P                                  # the same |kappa| mass below: the two add up
          d[f'role_offset_r_{R}_by_sout_{q_}'] = _offset(rq[0] - rq[2], bw[0], live)
      raw[f'unit_offset_{k}'] = uo
      bv = {}
      for gi, G in enumerate(list(groups) + ['total']):         # the role offset as a vector O_n (U, d)
        bv[G] = rv[:, :, :, li, gi]
        O = _unit_offset_vecs(_div(bv[G][0], bw[0][..., None]) * P - _div(bv[G][2], bw[2][..., None]) * P, bw[0])
        oe, nrm = (O * E).sum(1), np.linalg.norm(O, axis=1)
        ok = live & np.isfinite(oe)
        d[f'offset_vec_norm_{G}'] = float(nrm[ok].mean()) if ok.any() else float('nan')
        d[f'offset_vec_cos_e_{G}'] = float(_div(oe, nrm)[ok].mean()) if ok.any() else float('nan')
        d[f'offset_vec_e_pos_{G}'] = float((oe[ok] > 0).mean()) if ok.any() else float('nan')
        d[f'offset_null_sd_{G}'] = (float(np.sqrt(np.clip(nrm[ok] ** 2 - oe[ok] ** 2, 0, None).sum() / (dim - 1))
                                          / ok.sum()) if ok.any() else float('nan'))
      raw[f'bin_rv_{k}'] = bv
      if stat is not None:
        for G in list(groups) + ['total']:
          ok = live & np.isfinite(uo[G])
          d[f'spearman_static_{G}'] = spearman(stat[ok], uo[G][ok]) if ok.sum() > 2 and np.ptp(uo[G][ok]) > 0 \
              else float('nan')
        ok = live & np.isfinite(uo['head'])
        d['head_offset_at_static_0'] = float(np.linalg.lstsq(np.c_[np.ones(ok.sum()), stat[ok]], uo['head'][ok],
                                                             rcond=None)[0][0]) if ok.sum() > 2 else float('nan')
      summ[k] = d
      summ[f'sum_err_{k}'] = err = _x4_sum_err(d, readers, roles)
      assert err < tol, f'X4 {st["name"]} {k}: the readers do not add up to the total (relative error {err:.2e})'
    out[st['name']] = (summ, raw)
  return out


def main():
  p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument('--run', required=True, help='run directory with config.json and model_{step}.pt')
  p.add_argument('--steps', required=True)
  p.add_argument('--blocks', default=DEFAULT_BLOCKS, help="X1 blocks ('' to skip X1)")
  p.add_argument('--x2_blocks', default=X2_BLOCKS, help="X2 blocks ('' to skip X2)")
  p.add_argument('--alphas', default='0,0.5', help='X2 scales of the partner sets')
  p.add_argument('--betas', default='0.8,0.9,1.1', help='X2 scales of the stream along e at the block output')
  p.add_argument('--no_later', action='store_true', help='X2: leave out the later blocks\' partners')
  p.add_argument('--x4_blocks', default='', help="X4 blocks, all of one stage (e.g. 'stage1.block0,stage1.block1,"
                 "stage1.block2'; '' to skip X4)")
  p.add_argument('--x4_sigma', action='store_true', help='X4: also the pass with every LayerNorm\'s 1 / sigma fixed')
  p.add_argument('--val', default='/marimo/work/data/imagenet/data')
  p.add_argument('--probe_cache', default='/dev/shm/probe_8.pt')
  p.add_argument('--wordnet', default='docs/data/imagenet_wordnet.npz')
  p.add_argument('--edges', default='kappa', choices=['kappa', 'positions'],
                 help='X1 bin edges: |kappa|-weighted deciles of all positions, or unweighted')
  p.add_argument('--active', default='image', choices=['image', 'pole'],
                 help='X1 active positions: u >= half the image\'s u*, or >= half the unit\'s mean pole u*')
  p.add_argument('--n_sample', type=int, default=16, help='X1: positions per image for the bin edges')
  p.add_argument('--batch', type=int, default=50, help='X1 batch (and the block inputs)')
  p.add_argument('--x2_batch', type=int, default=480, help='X2 batch (480 = one unit\'s 60 classes x 8 images)')
  p.add_argument('--seed', type=int, default=0)
  p.add_argument('--geometry_only', action='store_true', help='only the block write geometry (no probe images)')
  p.add_argument('--out', default=None, help='JSON (default: stage1_mech.json, or stage1_geometry.json, in --run); '
                 'raw arrays next to it')
  a = p.parse_args()
  dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  cfg = json.load(open(os.path.join(a.run, 'config.json')))
  model = build_convnext(cfg['size']).to(dev).eval()
  model.requires_grad_(False)
  steps = [int(t) for t in a.steps.split(',')]
  b1 = [b for b in a.blocks.split(',') if b]
  b2 = [b for b in a.x2_blocks.split(',') if b]
  b4 = sorted({b for b in a.x4_blocks.split(',') if b})
  if b4:
    try:                                                # before any work: a bad name fails here, not after X1 and X2
      x4_check(model, b4)
    except ValueError as e:
      p.error(str(e))
    b4.sort(key=parse_block)
  blocks = sorted(set(b1) | set(b2), key=parse_block)
  if a.geometry_only:
    res = {}
    for s in steps:
      model.load_state_dict(torch.load(os.path.join(a.run, f'model_{s:05d}.pt'), map_location=dev))
      res[s] = {b: geometry_stats(model.stages[parse_block(b)[0]][parse_block(b)[1]]) for b in blocks}
      print(s, {b: {k: round(v, 3) for k, v in r.items()} for b, r in res[s].items()}, flush=True)
    json.dump(res, open(a.out or os.path.join(a.run, 'stage1_geometry.json'), 'w'), indent=1)
    print('STAGE1_GEOMETRY_DONE', flush=True)
    return
  xs, y = probe_set(a.val, 8, a.probe_cache)
  y = y.to(dev)
  C = int(y.max()) + 1
  rel = None
  if a.wordnet and os.path.exists(a.wordnet):
    rel = np.load(a.wordnet)['dist'][:C, :C] <= 4
    np.fill_diagonal(rel, False)
  else:
    print(f"no WordNet file at {a.wordnet}: 'rest' includes the relatives", flush=True)
  alphas = tuple(float(t) for t in a.alphas.split(','))
  betas = tuple(float(t) for t in a.betas.split(',') if t)
  out = a.out or os.path.join(a.run, 'stage1_mech.json')
  odir = os.path.dirname(os.path.abspath(out))
  os.makedirs(odir, exist_ok=True)
  res = {}
  def save(s, raw):
    torch.save(raw, os.path.join(odir, f'stage1_mech_raw_{s:05d}.pt'))
    json.dump(res, open(out, 'w'), indent=1)
  for s in steps:
    model.load_state_dict(torch.load(os.path.join(a.run, f'model_{s:05d}.pt'), map_location=dev))
    res[s], raw = {'x1': {}, 'x2': {}}, {'x1': {}, 'x2': {}}
    cache = {}
    for b in blocks:
      i, j = parse_block(b)
      X = block_inputs(model, xs, i, j, a.batch, dev)
      stats = cache[b] = class_stats(model, b, X, y, C, a.batch)
      if b in b1:
        res[s]['x1'][b], raw['x1'][b] = x1_block(model, b, X, y, rel, stats, a.batch, a.n_sample, a.edges, a.seed,
                                                 active=a.active)
        r = res[s]['x1'][b]
        print(s, b, 'X1', {k: round(r[k], 3) for k in (
            'pole_top20', 'pole_kp_dp_top20', 'pole_kn_dn_top20', 's_in_top20', 's_own_top20', 's_part_top20',
            's_out_top20', 's_later_top20', 'frac_m1_top20', 'frac_m4_top20', 'slope_r_by_sout_top20',
            'role_offset_r_by_sout', 'role_offset_r_by_sout_u', 'b_excl_top20', 'b_own_top20', 'write_eff_rank')},
              flush=True)
      if b in b2:
        res[s]['x2'][b], raw['x2'][b] = x2_block(model, b, X, y, rel, stats, alphas, betas, batch=a.x2_batch,
                                                 seed=a.seed, later=not a.no_later)
        r = res[s]['x2'][b]
        print(s, b, 'X2', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
                           if k in ('units', 'top20_base', 'size_splus', 'size_later', 'ds_e_splus', 's_out_act')
                           or k.startswith(('d_', 'top20_beta', 'top20_splus_e', 'top20_splus_perp'))}, flush=True)
      del X
    if b4:
      save(s, raw)                                      # X1 and X2 of this step are kept if X4 fails
      X = block_inputs(model, xs, parse_block(b4[0])[0], 0, a.batch, dev)          # the stage's input
      r4 = x4_block(model, b4, X, y, rel, cache, a.batch, a.n_sample, a.edges, a.seed, sigma=a.x4_sigma)
      del X
      res[s]['x4'], raw['x4'] = {b: r[0] for b, r in r4.items()}, {b: r[1] for b, r in r4.items()}
      for b, (r, _) in r4.items():
        losses = [k for k in X4_LOSSES + tuple(f'{k}_fixed' for k in X4_LOSSES) if k in r]
        rd = lambda v: round(v, 3)
        print(s, b, 'X4 role_offset_r_by_sout', {k: {R: rd(r[k][f'role_offset_r_{R}_by_sout'])
                                                     for R in ['total', *r['groups'], *r['single_groups']]}
                                                 for k in losses},
              'total kpos/kneg', {k: (rd(r[k]['role_offset_r_total_by_sout_kpos']),
                                      rd(r[k]['role_offset_r_total_by_sout_kneg'])) for k in losses},
              'perp / null sd', {k: (rd(r[k]['role_offset_r_perp_by_sout']), rd(r[k]['offset_null_sd_total']))
                                 for k in losses},
              'cos(e, O_total)', {k: rd(r[k]['offset_vec_cos_e_total']) for k in losses},
              'static head', rd(r.get('static_head_offset', float('nan'))),
              {k: rd(r[k].get('spearman_static_head', float('nan'))) for k in losses},
              'sum_err', max(r[f'sum_err_{k}'] for k in losses), flush=True)
    save(s, raw)
  print('STAGE1_MECH_DONE', flush=True)


if __name__ == '__main__':
  main()
