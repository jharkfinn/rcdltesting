# Tests 2 and 3: predictions written before the results

Written 2026-10-19 22:40 UTC, before any run.

## Test 2: InfiLoop gating (Li et al. 2026, arXiv 2610.11570)

Configs: `experiments/configs/facts_loops_infiloop.json` and `bios_loops_infiloop.json`.

The next state is a content-weighted, decayed average of the candidates z_t = h_t−1 + f(LN(h_t−1)) instead of the latest one:
- h_t = N_t / D_t, with N_t = b N_t−1 + e_t z_t and D_t = b D_t−1 + e_t;
- e_t = exp(clamp(q · RMSNorm(z_t), −8, 8));
- q is a learned vector, zero at initialization; b = exp(−softplus(r)) is a learned decay, 0.55 at initialization.

Speed-up below is the step at which the residual K = 1 run of the same arm and seed first reaches 20%, 30%, 40% and 50% accuracy, divided by the same step for the run in question.

| ID | Prediction | Measure | Refuted if |
|---|---|---|---|
| G1 | Gating removes overthinking. | accuracy at K_test = 2 K_train against K_train (probe, facts) | for some arm the drop is more than 2 points in both seeds |
| G2 | Gating rescues GeGLU. | GeGLU 512 gated at K = 4: late gain over residual K = 1, and speed-up | the late gain is below 0 or the speed-up below 1.0 in both seeds |
| G3 | Cos-Bochner keeps its looping advantage under gating. | speed-up of gated Cos σ = 4 (D = 256, 1,536) at K = 4 against gated GeGLU | Cos is not faster than GeGLU in both seeds |
| G4 | Gating stops the K = 4 wording memorization of Cos T = 2 (biographies). | seen − held-out gap, late window | the gap is 2 points or more in both seeds (residual K = 4: 0.144 / 0.113) |
| G5 | Gated Cos T = 1 at K = 4 does not lose held-out accuracy. | held-out, late window, seed 1 | below residual K = 1 (0.655) by more than 2 points |

## Test 3: cosine learning-rate decay at D = 256 (60,000 steps)

Config: `experiments/configs/facts_loops_long_cosine.json`, K = 1, 4, 8, seeds 1 and 2.

| ID | Prediction | Measure | Refuted if |
|---|---|---|---|
| R1 | Decay removes the late oscillation. | standard deviation of accuracy over the last quarter | above 0.5 points for any run |
| R2 | Looping raises the level the small layer reaches (not only speed). | late accuracy at K = 4 minus K = 1 | +0.5 points or less in either seed |
