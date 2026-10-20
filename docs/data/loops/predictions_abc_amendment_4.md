# Amendment 4 to predictions_abc.md: the late-phase gain of looped Cos, and third seeds (written before these runs)

**What had been seen.**
- Every result of tests B and C and of the readout-scale grid (seeds 1 and 2).
- Test A at 30,000 steps.
- The 120,000-step K = 1 runs of test L.
- The verification workflow on §7.5e (`verify_75e.json`), which found this in the grid data. At each family's best early scale, the best looped Cos reaches 70–90% accuracy sooner than the best unlooped Cos:
  - P at 90% is 1.20–1.30 for both kernels and both seeds;
  - late accuracy at 10,000 steps is higher by 1.3 points (σ = 4) and 2.0–2.7 points (σ = 8);
  - no K = 1 readout scale in the grid reaches that late accuracy;
  - zero-initialized GeGLU goes the other way (P at 90% 0.91–0.94), and the ReLU MLP is near 1.
- H1 was pre-registered on the 20–50% window only, so this finding is exploratory. The runs below test it on new seeds.

## 1. Third seeds required by amendment 2 (rules that split or sit within the margin)

- C2b and D2: the GeGLU speed-ups lie within 0.05 of their thresholds (C2b 0.96 against < 1; D2 0.92–0.93 against ≥ 0.95). Runs: `geglu_w512_K1_s3`, `geglu_w512_a4_K1_s3` and `geglu_w512_a0p25_K4_s3` (`facts_loops_why.json`).
- B2 for GeGLU residual with deep supervision at K = 8: the drop is 2.03 / 1.95 points against ≤ 2, a split. Runs: `geglu_w512_ds_K8_s3`, and `geglu_w512_K8_s3` without deep supervision for reference (`facts_loops_gate2.json`).
- Each of these rules is decided on 3 seeds by majority (2 of 3).

## 2. The late-phase gain on new seeds (LP1–LP3)

**Runs** (seeds 3 and 4, 10,000 steps, Adam 1e-3 constant, D = 1,536, train_probe on; `facts_loops_grid.json`):
- Cos σ = 4 and Cos σ = 8 at K = 1 with a = 8 and 16, and at K = 4 with a = 2 and 4. These are scales 8 and 16, the best scales of seeds 1 and 2 by the early and the late measure.
- Zero-initialized GeGLU 512 at K = 1 with a = 2, and at K = 4 with a = 0.5 (scale 2, its best).

That makes 20 runs. **Best** is chosen per family and K: the scale (8 or 16) with the lowest step to 90%, mean over seeds 3 and 4. **P90** is the step to 90% of the best K = 1 divided by that of the best K = 4, per seed.

| | Prediction | Rule |
|---|---|---|
| LP1 | The passes make Cos reach high accuracy sooner, beyond the readout step. | P90 ≥ 1.15 in both new seeds, for Cos σ = 4 and for Cos σ = 8 (verdict per kernel). |
| LP2 | GeGLU gets no such gain. | Zero-initialized GeGLU: P90 ≤ 1.0 in both new seeds. |
| LP3 | The looped Cos model is more accurate at 10,000 steps. | Late accuracy of the best K = 4 minus the best K = 1 (best by late accuracy over scales 8 and 16) ≥ +1.0 point in both new seeds, per Cos kernel. |

## 3. Decayed rate at D = 1,536 (LP4)

At a constant rate, a higher late accuracy can be faster learning, a higher level, or less oscillation. With the rate decayed to 0, the oscillation goes away. At D = 1,536 every model probably reaches about 100% (§7.5d), so the test uses the step to 95% and 99%.

**Runs** (seeds 3 and 4, 30,000 steps, cosine decay to 0, evaluation every 1,000 steps; `facts_loops_late.json`):
- Cos σ = 4 at K = 1 with a = 8 and 16, and at K = 4 with a = 2 and 4;
- zero-initialized GeGLU 512 at K = 1, a = 2 and K = 4, a = 0.5.

That makes 12 runs.

| | Prediction | Rule |
|---|---|---|
| LP4 | Under a decayed rate looped Cos still reaches 95% sooner. | Best K = 4 over best K = 1 (best = the lower mean step to 95% over seeds 3 and 4) ≥ 1.15 at 95%, in both seeds. Also reported: the step to 99% and the final accuracy (mean of the last 3 evaluations). |

Expectation: LP1, LP2 and LP3 hold, as in seeds 1 and 2. LP4 is uncertain.
