# Predictions for tests A, B and C (written 2026-10-20, before any of these runs)

Follow-ups to REPRODUCTION.md §7.5d ("Where the question stands"). The user asked for all three.

Configs: `experiments/configs/facts_loops_level.json` (A), `facts_loops_gate2.json` (B), `facts_loops_why.json` (C).
Code: `rcdl/facts.py` with `deep_supervision`, `train_probe` (`pass_probe`), the gate log and the gate-off probe.

## Common definitions

- **Task.** The facts task of §7.5d. N = 8,192 individuals with 4 attributes each, so there are 32,768 facts. Each attribute has 32 values (5 bits). Names have two tokens and appear at a random position among 10 distractors. Accuracy is measured on 4,096 fixed evaluation facts, a random eighth of the trained facts.
- **K** is the number of passes of the last FFN block with tied weights. **D** is the number of random features (Cos-Bochner). **w** is the GeGLU width. **σ** sets the kernel width: W ~ N(0, 1/σ²), and the kernel is exp(−‖x − y‖² / (2σ²)). **a** is the step size of each pass: h ← h + a f(LN(h)).
- **Late accuracy:** the mean over the evaluations in the last quarter of training (steps ≥ 0.75 × steps).
- **Gain** (in points): late accuracy of a variant minus late accuracy of its baseline, same seed.
- **Speed-up at threshold t:** the step at which the baseline first reaches accuracy t, divided by the step at which the variant first reaches t. Steps are linearly interpolated between evaluations. A value above 1 means the variant learns faster. "Early" thresholds are t = 0.2, 0.3, 0.4 and 0.5; "high" thresholds are t = 0.7 and 0.9. A threshold that a run does not reach is reported as not reached and left out of the range.
- **The 30% point** of a run: its first evaluation with accuracy ≥ 0.3.
- "Both seeds" means seeds 1 and 2. A prediction holds if its rule is met in both seeds. It is refuted if the rule fails in both seeds, and split otherwise.

## Test A: does looping raise the reachable level when the layer is at capacity?

Test 3 could not answer this, because every model reached 100% once the rate decayed. Here the layers are made small enough that K = 1 stays below 100%.

**Runs** (36). Cos-Bochner σ = 4 at D = 32, 64 and 128. GeGLU at w = 11, 21 and 42, which match the trained parameters of one Cos layer: 386 w + 128 against 128 D + 128, that is 4,374 / 8,234 / 16,340 against 4,224 / 8,320 / 16,512. The models have 221,856–246,432 trained parameters in all. K = 1, 4 and 8, both seeds. Adam 1e-3 decayed to 0 by a cosine over 30,000 steps, with an evaluation every 1,000 steps.

**Selection rule (uses the K = 1 runs only).** A width is *capacity-limited* if its K = 1 late accuracy is below 0.97 in both seeds. A1 to A4 use capacity-limited widths only.

**A-long (8 runs, added after the K = 1 results).** For each family, pick the capacity-limited width whose K = 1 late accuracy (mean of seeds) is closest to 0.85. Run it at K = 1 and 4, both seeds, with the same cosine schedule over 60,000 steps. If a family has no capacity-limited width, it gets no A-long runs.

| | Prediction | Rule |
|---|---|---|
| A1 | Looping raises the level of a capacity-limited Cos layer. | At each capacity-limited Cos width, the K = 4 gain over K = 1 is at least +1.0 point. |
| A2 | Cos gains more than GeGLU at equal trained parameters. | For each matched pair (D, w) where both widths are capacity-limited: Cos gain at K = 4 minus GeGLU gain at K = 4 ≥ +1.0 point. |
| A3 | The gain is a level gain, and the extra training time does not remove it. | In A-long, the K = 4 gain at 60,000 steps is at least half the K = 4 gain at 30,000 steps (same width). If it falls below half, the 30,000-step gain is mostly speed. |
| A4 | More passes raise the level further for Cos. | At each capacity-limited Cos width, the K = 8 late accuracy ≥ the K = 4 late accuracy − 0.5 point. |

My reasons for A1 and A2: looping makes the layer's output nonlinear in its trained readout. For a narrow kernel each pass reads a new feature vector, so a K-pass layer can express more than a rank-D linear readout. GeGLU tied looping lost accuracy in every earlier test.

## Test B: is the gate active, and does deep supervision stop the overshoot?

In Test 2 the gated runs tracked the residual runs closely, and q and β were not logged. In the paper (Li et al. 2026) the gate and the stable fixed point come with deep supervision. Here the loss is read after every pass.

**Runs** (20). Cos-Bochner σ = 4 at D = 1,536, and GeGLU 512. 10,000 steps at Adam 1e-3. Both seeds.
- B-rerun: the gated K = 4 runs of `facts_loops_infiloop.json` again, now with the gate log and the gate-off probe (4 runs).
- B-ds: deep supervision for the residual and the gated loop, at K = 4 and 8 (16 runs).

The K = 1 baselines are the C reruns (`cos_w1536_s4_K1`, `geglu_w512_K1`).

- **Gate-off probe:** accuracy at the trained K with q set to 0. With q = 0 every candidate gets the same score, so the loop becomes a residual loop with a falling convex step.
- **Score spread:** for each candidate, the standard deviation over examples of the clamped score q · RMSNorm(z_t), in nats.
- **Overshoot drop:** accuracy at the trained K minus accuracy at K_test = 2K, in points.

| | Prediction | Rule |
|---|---|---|
| B1 | Without deep supervision the gate stays inactive. | For both arms in B-rerun, at the end of training: the gate-off probe changes accuracy at the trained K by less than 1 point, and the score spread is below 0.5 for every candidate. |
| B2 | Deep supervision removes the overshoot. | For each of the 8 deep-supervised arms (2 loops × 2 FFNs × 2 K), the overshoot drop is at most 2 points. |
| B3 | Cos keeps its speed-up under deep supervision, and GeGLU does not gain one. | At K = 4 with deep supervision, residual and gated: Cos has an early speed-up ≥ 1.2 at every early threshold; GeGLU ≤ 1.05 at every early threshold. |
| B4 | With deep supervision the gate becomes active for Cos. | Cos gated deep-supervised, K = 4 or 8: the gate-off probe changes accuracy at the trained K by at least 1 point. |

Also reported, with no rule: late accuracy at the trained K with and without deep supervision; the gate log (|q| and β over training); and the B-rerun late accuracy against the original Test 2 runs (GPU nondeterminism).

## Test C: why does the narrow cosine gain from looping?

**Runs** (56). D = 1,536 for Cos-Bochner. 10,000 steps, constant Adam rate. Both seeds.
- C-σ: σ = 1.5, 2, 3, 4, 6 and 8 at K = 1 and 4 (24 runs). σ = 4 reruns the width-sweep runs; its late accuracy must agree within 1 point.
- C-scale: K = 1 with a = 2 and a = 4, for Cos σ = 4, Cos σ = 8 and GeGLU 512 (12 runs).
- C-lr: K = 1 at Adam 2e-3 and 4e-3, and K = 4 at 2e-3, for Cos σ = 4 and GeGLU 512 (12 runs).
- C-probe: GeGLU 512 and ReLU random features at D = 1,536, K = 1 and 4 (8 runs).

Every C run records `pass_probe` at every evaluation.

**Why the scale control works.** The Cos readout starts at zero. While it is near zero, the passes barely move the state, so h_k ≈ h_0. Then K passes write about K f(LN(h_0)), which is K times one pass. A K = 1 run with a = K matches that output. Adam moves each weight by about its learning rate per step, so the scale changes how much each step changes the output. Looping that only acts through this scale effect should show the same speed-up as a = K.

| | Prediction | Rule |
|---|---|---|
| C1 | The looping speed-up peaks at an intermediate kernel width. This is the mean-shift picture: each pass is one shift step, and the gain per step is largest at an intermediate bandwidth. | Take the mean early speed-up of K = 4 over K = 1 at each σ (mean over thresholds and seeds). *Peak*: the largest value is at σ = 2, 3 or 4, and both σ = 1.5 and σ = 8 are at least 0.1 below it. *Monotone*: the values do not rise by more than 0.05 from one σ to the next larger σ. *Other*: neither. Prediction: peak. |
| C2 | The output scale explains at least half of the Cos σ = 4 speed-up. | For Cos σ = 4, with S(·) the mean early speed-up over K = 1 at a = 1: S(K = 1, a = 4) − 1 ≥ 0.5 × (S(K = 4) − 1). |
| C2b | The output scale slows GeGLU. | GeGLU 512 at K = 1, a = 4: early speed-up below 1 at every early threshold. |
| C3 | Looping helps Cos σ = 4 beyond a change of the learning rate. | Take K = 4 at the better of Adam 1e-3 and 2e-3, and K = 1 at the best of 1e-3, 2e-3 and 4e-3 ("best" is the lowest mean step to the early thresholds). The K = 4 run reaches every early threshold at least 1.15 times sooner. |
| C4 | Narrow kernels give each pass new features. | At the 30% point of each K = 4 run: the median overlap of consecutive pass features (overlap_prev, mean over passes 2–4) is lower for σ = 4 than for σ = 8. Across the six σ values, the Spearman correlation between this overlap and the mean early speed-up is at most −0.6 (one value per σ, mean of seeds). |
| C5 | Each looped Cos pass moves the state downhill on the loss read before it, more than a GeGLU pass does. | At the 30% point of each K = 4 run, the median descent_cos is above 0 for every pass of Cos σ = 4 and above the GeGLU 512 value (mean over passes). |

C1 and C4 are competing pictures. C1 predicts a peak (mean-shift). C4 predicts a speed-up that falls as the feature overlap rises (decorrelation), and the overlap rises steadily with σ. Both cannot hold well at once, and the sweep decides between them.

## What would change the conclusions of §7.5d

- If A1 holds and A3 holds, looping adds storage to a small Cos layer. That would be a specific advantage beyond speed. If A2 also holds, the advantage is specific to Cos.
- If C2 holds, most of the "special" Cos speed-up is a scale effect that a K = 1 model gets from a = K. If C3 also fails, looping is mostly a learning-rate change for this layer.
- If B2 holds, the overshoot of §7.5d comes from supervising only pass K, and it is not a property of the Cos map.
