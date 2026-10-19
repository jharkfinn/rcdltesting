# Basis tests: predictions written before the results

Written 2026-10-19 19:05 UTC, before any basis-test run. Config: `experiments/configs/facts_loops_basis.json` (facts task, K = 1 and 4, seeds 1 and 2).

Measure: the late gain, which is the late accuracy at K = 4 minus the late accuracy at K = 1, same arm and seed. The late accuracy is the mean over the last quarter of training (amendment 2).

Two explanations for why fixed Cos-Bochner gains from looping and GeGLU does not:

- **Reason 1, fixed basis.**
  - One pass of a fixed random-feature layer, f(x) = W_o φ(x), is linear in its trained parameters W_o. Here φ is the fixed feature map and x is the normalized input.
  - Looping composes the features, φ(LN(h + W_o φ(LN(h)))), with h the residual state. This gives new nonlinear functions without new parameters.
  - Trained-feature MLPs (GeGLU, ReLU MLP) already adapt their features in one pass.
- **Reason 2, cosine features.** Smooth, bounded cosine features with bounded derivatives iterate stably. GeGLU's trained map is expansive.

| Test | Arm | Reason 1 predicts | Reason 2 predicts |
|---|---|---|---|
| T1 | ReLU random features, frozen first layer (`rfrelu`) | gain > 0, near fixed Cos at the same D | gain below fixed Cos (ReLU features are not smooth or bounded) |
| T2 | Cos-Bochner σ = 4 with W and b trained (`cos_trainW`) | gain below fixed Cos at the same D | gain near fixed Cos |
| T3 | trained ReLU MLP (`relu`, widths 768 and 6,144) | gain ≤ 0, like GeGLU | gain ≤ 0 or small |

Decision rules:
- Reason 1 is supported if T1 gains in both seeds at D ≤ 4,096 and T2 gains less than fixed Cos in both seeds.
- Reason 2 is supported if T1 does not gain and T2 gains about as much as fixed Cos.
- Any other pattern is reported as mixed.

Also expected: at D = 16,384 every gain is small, because the large fixed-Cos gain is only +0.5 points.
