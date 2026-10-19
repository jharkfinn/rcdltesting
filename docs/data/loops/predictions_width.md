# Width sweep: predictions written before the results

Written 2026-10-19 18:50 UTC, before any width-sweep run.

The user asked to equalize parameters: can looping let a small Cos-Bochner layer match the large one? Config: `experiments/configs/facts_loops_width.json`.
- Cos-Bochner σ = 4, D = 256 ... 8,192 random features, K = 1, 2, 4, 8, seeds 1 and 2.
- Reference: the large layer (D = 16,384) at K = 1, late accuracy 0.982 (round 1).
- Measure: the late accuracy, the mean over the evaluations in the last quarter of training (amendment 2).

Parameters per FFN layer:
- trained: W_o and its bias, 128 D + 128;
- fixed random features: W and b, 129 D.

D = 1,536 has the trained parameters of GeGLU 512: 196,736 against 197,760 per layer, and 606,880 against 608,928 for the whole model.

A rough capacity count:
- The task stores 8,192 × 4 facts of log2(32) = 5 bits, which is 163,840 bits.
- At 2 bits per trained parameter (Allen-Zhu and Li; Ouro 2510.25741), that needs about 82k parameters, fewer than the smallest model here (279k).
- But each fact is seen only about 39 times in 10,000 steps. Capacity per parameter is far lower with few exposures, so learning speed may bind before parameter count does.

| ID | Prediction | Refuted if |
|---|---|---|
| W1 | Late accuracy rises with D at every K. | a smaller D beats a larger D at the same K in both seeds |
| W2 | The looping gain (late accuracy at K = 4 minus at K = 1) is larger at small D than at D = 16,384 (+0.5 points). | the gain at D ≤ 2,048 is not above +0.5 points in both seeds |
| W3 | Looping does not let a layer of D ≤ 2,048 reach the large layer: no K gives late accuracy ≥ 0.982. | some D ≤ 2,048 at some K reaches 0.982 in both seeds |
| W4 | The parameter-matched Cos-Bochner layer (D = 1,536) at its best K beats GeGLU 512 at K = 1 (0.924). | it does not, in both seeds |

K = 8 is exploratory. Ouro reports loss spikes when training with 8 loop steps.
