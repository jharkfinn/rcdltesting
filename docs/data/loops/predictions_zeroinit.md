# Initialization control: predictions written before the results

Written 2026-10-19 22:10 UTC, before any run. Config: `experiments/configs/facts_loops_zeroinit.json`.
- Arms: GeGLU 512, ReLU MLP 768 and GeGLU 4,096, with the FFN output weight zero-initialized.
- Passes and seeds: K = 1, 2, 4 (GeGLU 4,096: K = 1, 4), seeds 1 and 2.

The question: do looped trained MLPs lose because they start as a random map applied K times? With default initialization, GeGLU and the ReLU MLP write 0.11–0.24 |h| per pass at the start. The Cos-Bochner arms write about 0.015 |h|.

Measures: the speed-up (step at which K = 1 first reaches 20%, 30%, 40%, 50% accuracy, divided by the same step for K = 4; above 1 means looping learns faster) and the late gain (K = 4 minus K = 1, late window).

| ID | Prediction | Supports "initialization explains the MLP loss" if | Supports "reusing one trained map is the problem" if |
|---|---|---|---|
| Z1 | Zero-init GeGLU 512 under looping | speed-up ≥ 1.0 at all four thresholds in both seeds | speed-up < 1.0 at all four thresholds in both seeds (default init: 0.85–0.89) |
| Z2 | Zero-init ReLU MLP 768 | as Z1 | as Z1 (default init: 0.85–0.90) |
| Z3 | Zero-init GeGLU 4,096 | as Z1 | as Z1 (default init: 0.38–0.73) |

- Any pattern in between is reported as partial.
- If zero-init MLPs reach the narrow-Cos speed-up (1.33–1.53), the earlier "Cos gains, MLPs lose" contrast was an initialization artifact.
- If they stay below 1.0, the contrast stands.
