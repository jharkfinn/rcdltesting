# Amendment 2 to predictions_abc.md: fair speed comparison, a real level test, and tighter rules (written before these runs)

**Why.** A design review (workflow `review-looping-abc`; issues Q-1, A-1 to A-5, B-1 to B-4, C-1 to C-8, P-1, S-1, saved in `review_abc.json`) found these problems:
- The tests did not compare the families fairly.
- The step size a acts as a learning rate on the looped block's readout. With Adam, K = 1 at a = 4 equals K = 1 at a = 1 with the readout rate × 4 and its bias × 4, to 3.6e-5 in the logits after 60 steps. So a speed-up measured at one a can come from a readout rate that suits one K better.
- Test A cannot make the looped layer the capacity limit, and its A3 check cannot tell speed from level.
- Several rules pooled the seeds or let one run decide.

**What had been seen when this was written** (results of tests A, B and C so far):
- A: the K = 1 late accuracy of all six widths, both seeds. Cos D = 32 / 64 / 128: 0.995 / 0.994, 0.999 / 0.999, 1.000 / 1.000. GeGLU w = 11 / 21 / 42: 0.929 / 0.907, 0.985 / 0.976, 1.000 / 0.999. No looped A run had been seen.
- C-scale (all 12 runs) and C-lr (Cos σ = 4: all 6; GeGLU: K = 1 at 2e-3 and 4e-3, K = 4 at 2e-3):
  - Cos σ = 4 at K = 1 speeds up 1.22–1.27 with a = 2 and 1.48–1.58 with a = 4.
  - At a learning rate of 4e-3 for all weights, K = 1 does not train (Cos and GeGLU, 0.034).
  - At 2e-3 there is no speed-up for Cos K = 1 (0.90–1.04).
  - Late accuracies of the σ = 8 and GeGLU scale runs: 0.90–0.92.
- C-decomp: Cos σ = 4 at K = 4 and a = 0.25 has a speed-up of 0.97–1.01 over K = 1 (both seeds). The σ = 8 and GeGLU decomposition runs had been pulled but not looked at.
- C-σ: only the σ = 4 K = 1 reruns, which reproduce the width-sweep runs exactly, evaluation by evaluation.
- B: nothing. The B runs were started from commit 26248c3 a few minutes before the review came back.

## 1. The headline test: does looping help a family at its own best readout step?

For each family, K = 1 and K = 4 each get the same grid of initial output scales s = K × a ∈ {1, 2, 4, 8, 16}. That is a ∈ {1, 2, 4, 8, 16} at K = 1 and a ∈ {0.25, 0.5, 1, 2, 4} at K = 4. Families, all at about 607k trained parameters, 10,000 steps, Adam 1e-3, both seeds, train_probe on:
- Cos-Bochner σ = 4, D = 1,536;
- Cos-Bochner σ = 8, D = 1,536;
- ReLU random features, D = 1,536;
- zero-initialized GeGLU 512;
- zero-initialized ReLU MLP 768.

Runs that already exist under the same code are reused. 80 runs are new (`facts_loops_grid.json`). Also new: Cos σ = 4 at D = 6,144 and K = 1, a ∈ {1, 4}, as an unlooped layer with the compute of K = 4 at D = 1,536 (4 runs, reference only).

**Definitions.**
- *Thresholds used:* the early thresholds that every run in the comparison reaches. A run that does not train (late accuracy < 0.1) is never "best". If fewer than 2 thresholds remain, the verdict is "not testable".
- *Best scale for K:* the s with the lowest mean step to the thresholds used, mean over both seeds. If the best s is at the edge of the grid (1 or 16) and that run trained, the grid is extended by one step (s = 0.5 or 32) before the verdict.
- *Pass gain* P_f = (step of best K = 1) / (step of best K = 4), per threshold and per seed.

| | Prediction | Rule |
|---|---|---|
| H1 | The passes add speed for Cos σ = 4 at its best readout step. | P ≥ 1.15 at every threshold used, in both seeds. |
| H2 | This is specific to the narrow cosine. | H1 holds for Cos σ = 4, and in each seed its P (mean over thresholds) is at least 0.10 above P for each of the other four families. |
| H3 | Looped Cos beats the best unlooped MLP. | Best Cos σ = 4 at K = 4 reaches every threshold used at least 1.15 times sooner than best zero-initialized GeGLU 512 at K = 1, in both seeds. |

My expectation: H1 and H2 are refuted, because Cos σ = 4 at K = 4 and a = 0.25 was no faster than K = 1 at a = 1, and K = 1 at a = 4 was as fast as K = 4. For H3 I have no expectation.

## 2. A level test where the looped layer limits capacity

Test A found no capacity-limited Cos width: even D = 32 stores 99.4–99.5% of the 32,768 facts. The facts fit in the attention and embedding weights, which hold 84–93% of the parameters. A1, A2, A4, D3 and D4 are therefore **not testable** for Cos (the selection rule found no width). The A-long runs for GeGLU w = 11 go ahead as pre-registered. Under the stricter rule below they get two extra K = 1 runs of 120,000 steps.

**New level test (L).** N = 32,768 individuals, so 131,072 facts (655,360 bits), with the same model. Cos σ = 4 at D = 256 (279,200 trained parameters), and zero-initialized GeGLU w = 85 (279,284; equal parameters in one FFN layer, 32,896 against 32,938). Adam 1e-3 decayed by a cosine; both seeds; train_probe on.
- K = 1 at a = 1 and a = 4, 60,000 and 120,000 steps.
- K = 4 at a = 0.25 and a = 1, 60,000 steps.
- K = 1 at a = 0 (the last FFN switched off), 60,000 steps.

That makes 28 runs (`facts_loops_level2.json`). Final accuracy is the mean of the last 3 evaluations. Under cosine decay the rate is near 0 there, so it is the level reached.

| | Prediction | Rule |
|---|---|---|
| L0 (precondition) | The looped layer stores a share of the facts. | K = 1 at a = 1 minus K = 1 at a = 0 ≥ 3 points at 60,000 steps, in both seeds. If this fails, L1 and L2 are "not testable" for that family. |
| L-cap (precondition) | The model is at capacity. | K = 1 at a = 1 below 0.97 at 120,000 steps, in both seeds. |
| L1 | The passes raise the level, beyond speed and step size. | Best K = 4 at 60,000 steps (over a = 0.25 and 1) minus best K = 1 at 120,000 steps (over a = 1 and 4) ≥ +1.0 point, in both seeds. A pure speed-up of at most 2 times cannot pass this, because K = 1 gets twice the steps. |
| L2 | The level gain is larger for Cos than for GeGLU. | L1 holds for Cos, and the Cos margin minus the GeGLU margin ≥ +1.0 point, in both seeds. |

Expectation: L1 holds weakly for both families, and L2 is refuted. This is a weak prior. Looping makes the block's output nonlinear in its readout, which can add capacity for any FFN.

**A-long, stricter rule (replaces A3).** For GeGLU w = 11, the 30,000-step K = 4 gain is a level gain only if K = 4 at 30,000 steps exceeds K = 1 at 120,000 steps by ≥ 1 point (final accuracy), in both seeds. A3 as written is kept for the record. It is "not applicable" unless the 30,000-step gain is ≥ +1 point.

## 3. Changes to the rules of tests B and C

- **Seeds.** Every rule has a per-seed form: holds if both seeds pass, refuted if both fail, split otherwise. C1 and C2 are classified per seed. A rule that splits, or whose seed values lie within ±0.5 point (level) or ±0.05 (speed-up) of its threshold, gets a third seed (seed 3), decided on 3 seeds by majority. No other changes after the results.
- **Unreached thresholds** follow the "thresholds used" definition of §1 in every rule (C1, C2, C2b, C3, B3, D1, D2).
- **B1 (gate inactive).** Holds if the gate-off probe changes accuracy at the trained K by less than 1 point, and the largest weight_sd (SD over examples of a candidate's weight) is below 0.02, in both seeds. The score-SD condition is dropped. It measures |q| and not gating by content.
- **B2 (no overshoot with deep supervision)** counts only arms whose loop does work: accuracy at pass K minus accuracy at pass 1 ≥ 2 points (final probe), in both seeds. Arms that fail this are "loop idle" and are not counted.
  - New same-code runs without deep supervision at K = 8, residual and gated, Cos σ = 4 D = 1,536 and GeGLU 512, both seeds (8 runs, in `facts_loops_gate2.json`). The K = 4 runs without deep supervision come from C (residual) and B-rerun (gated).
  - Reported for each arm: the drop with deep supervision against the drop without.
- **B3** is one verdict: all four conditions must hold (Cos residual, Cos gated, GeGLU residual, GeGLU gated). Also reported, with no rule: deep-supervised K = 4 against the same-code K = 4 without deep supervision.
- **B4** is decided at K = 4. It holds if the gate-off change is at least 1 point in absolute value in both seeds. K = 8 is reported separately, with the sign.
- **C1** is classified per seed in this order: peak, else monotone, else other.
- **C4** becomes descriptive only. overlap_prev equals exp(−shift² / 2σ²) to 3 decimals, so it is nearly the kernel by construction, and a Spearman correlation over 6 points with a −0.6 threshold is weak (P = 0.12 under the null). Overlap and input_shift / σ are reported at steps 1,000 and 2,000 and at each run's 20% and 30% points.
- **C2, C2b, D1, D2** are readout-learning-rate effects, and §1 supersedes them as the decision on "looping helps beyond the step size". They are still reported per seed.
- **C3** counts an unstable run (late accuracy < 0.1) as "not best". Its asymmetric grid (3 rates for K = 1, 2 for K = 4) makes it a secondary check. The readout-scale grid of §1 decides.
- **C5** also compares each family's K = 4 descent_cos with its own K = 1 pass-1 value at the 30% point. It is descriptive.
- **Code version.** Every new run is launched from a commit whose `rcdl/facts.py` is byte-identical to that of 2dae75a (sha256 recorded in `code_versions.txt`). Comparisons pair runs of identical training code only.
- **A known limit, with no runs added.** σ and D change both FFN layers, so the K = 1 baseline itself changes with σ (review issue C-6).
