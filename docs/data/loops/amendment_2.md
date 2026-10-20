# Looping test: amendment 2 (measure of recall)

Written 2026-10-19 18:40 UTC, after the round 1 facts results were seen under both measures. This change is post hoc and not blind. Both measures are reported from now on.

## Why

predictions.md scores L1 with the final-step accuracy: the accuracy at step 10,000 minus the accuracy at K = 1, same seed. The evaluation set is fixed (4,096 facts, fixed distractors). The training rate is constant (Adam 1e-3), and the model oscillates late in training.

- In the last quarter of training (6 evaluations, steps 7,500–10,000), the within-run standard deviation of accuracy is 0.3 to 1.6 points.
- Example: Cos-Bochner σ = 4, K = 1, seed 1 reads 0.983, 0.975, 0.988 and 0.978 at steps 5,000, 7,500, 9,000 and 10,000.

So one evaluation at the last step is a noisy snapshot. Round 1 differences of 1 to 2 points between arms are of the same size.

## Change

- **Primary recall measure:** the mean accuracy over the evaluations in the last quarter of training (late accuracy). The late gain is the late accuracy at K minus the late accuracy at K = 1, same arm and seed.
- **Secondary:** the mean loss over the same window, and the first steps at which accuracy reaches 50% and 90% (speed).
- **The final-step accuracy is still reported** as the pre-registered measure.

## Effect on round 1

| Arm | Final-step gain (K = 2, K = 4) | Late gain (K = 2, K = 4) |
|---|---|---|
| Cos σ = 4 | +1.2, +1.6 points | +0.3, +0.5 points |
| Cos σ = 8 | −0.4, −0.7 points | +0.2, +0.2 points |
| GeGLU | −0.4, −1.2 points | −0.2, −1.3 points |

L1 (narrow Cos-Bochner gains more than GeGLU) holds under both measures in both seeds. The difference between the narrow and the wide kernel is small under the late measure. The σ sweep and the third seed of round 2 decide it.
