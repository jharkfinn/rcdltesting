# Amendment 1 to predictions_abc.md: separate the output scale from the extra passes (written before these runs)

**What had been seen when this was written.**
- The first 6 C-scale results: Cos σ = 4 at K = 1 with a = 2 and a = 4, and Cos σ = 8 at K = 1 with a = 2, seeds 1 and 2.
- One log line of test A: the two Cos D = 128 K = 1 runs had accuracy 0.9998 and 1.0000 at step 24,000 of 30,000. So D = 128 is probably not capacity-limited.
- No other result of tests A, B or C. Against the earlier K = 1 runs of `facts_loops_width` (the pre-registered baseline is the C rerun, not yet finished), Cos σ = 4 at K = 1 with a = 4 has an early speed-up of 1.48–1.58. That equals the speed-up of K = 4 looping, 1.45–1.54. So prediction C2 may hold in full. If it does, the "looping speed-up" of the narrow cosine would be a step-size effect that needs no extra pass.

**The 2 × 2 design.** Two models have the same function class when one's trained weights can be rescaled to give the other's outputs.
- K = 1 at a = 4 has the same function class as K = 1 at a = 1. The readout W_o and its bias can absorb the factor 4, so only the optimization differs. With Adam each step moves a weight by about the learning rate, so a = 4 makes each step change the block's output about 4 times as much.
- K = 4 at a = 1/4 has the same function class as K = 4 at a = 1, for the same reason. Near initialization (readout near zero) it writes about 4 × 1/4 = 1 times one pass's output, the same as K = 1 at a = 1.

| | output scale near initialization ≈ 1 | ≈ 4 |
|---|---|---|
| one pass (K = 1) | a = 1 (baseline) | a = 4 |
| four passes (K = 4) | a = 1/4 | a = 1 |

Differences along a row measure the passes at a fixed initial scale. Differences along a column measure the scale with the function class fixed.

**New runs.**
- C-decomp: K = 4 at a = 0.25 for Cos σ = 4, Cos σ = 8 and GeGLU 512, seeds 1 and 2, 10,000 steps (6 runs, `facts_loops_why.json`).
- A-scale: K = 1 at a = 4 and K = 4 at a = 0.25 for the six widths of test A, seeds 1 and 2, the 30,000-step cosine schedule (24 runs, `facts_loops_level.json`).

**Predictions.** S(x) is the mean early speed-up of run x over K = 1 at a = 1, same arm and seed (mean over the 20–50% thresholds).

| | Prediction | Rule |
|---|---|---|
| D1 | For Cos σ = 4, the passes add no speed at a fixed initial scale. | K = 4 at a = 0.25: early speed-up ≤ 1.10 at every early threshold, in both seeds. |
| D2 | The GeGLU slowdown under looping is a scale effect too. | GeGLU 512 at K = 4 and a = 0.25: early speed-up ≥ 0.95 at every early threshold, in both seeds. |
| D3 | At a capacity-limited Cos width, the passes raise the level at a fixed initial scale. | Late accuracy at K = 4, a = 0.25 minus K = 1, a = 1 ≥ +1.0 point, and K = 4, a = 1 minus K = 1, a = 4 ≥ +1.0 point, in both seeds, at each capacity-limited Cos width (selection rule of test A). |
| D4 | GeGLU gets no such level gain. | Same two differences < +1.0 point for each capacity-limited GeGLU width. |

D1 and D2 follow from the C-scale result. D3 and D4 restate A1 and A2 with the scale held fixed, which A1 and A2 alone cannot do.
