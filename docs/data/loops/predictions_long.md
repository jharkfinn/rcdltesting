# Long runs: predictions written before the results

Written 2026-10-19 20:55 UTC, before any long run. Config: `experiments/configs/facts_loops_long.json`: 60,000 steps, seed 1. Measure: the late accuracy, the mean over the last quarter of training (steps 45,000–60,000).

At 10,000 steps, Cos-Bochner D = 256 reaches 0.752 (K = 1) and 0.828 (K = 4), and it is still climbing. Two readings:

- **Speed only.** Looping makes the small layer learn faster but does not raise the level it can reach. Given enough steps, K = 1 catches up.
  - Capacity count: the D = 256 model has 279,200 trained parameters. At 2 bits per parameter that is 558k bits, against 164k bits of facts.
- **Level.** Looping raises the reachable level, because it gives a fixed-basis layer composed features without new parameters (reason 1 in `predictions_basis.md`).

| ID | Prediction | Refuted if |
|---|---|---|
| C1 | D = 256: late accuracy at K = 4 minus K = 1 shrinks from 10,000 to 60,000 steps (speed reading). | the gap at 60,000 steps is as large as or larger than at 10,000 (9.3 points, late window) |
| C2 | No D = 256 run reaches the large layer's 60,000-step late accuracy minus 2 points. | K = 4 or K = 8 comes within 2 points |
| C3 | D = 1,536 at K = 4 matches or beats GeGLU 512 at K = 1 (equal trained parameters) at 60,000 steps. | it is more than 1 point below |

K = 8 is exploratory. Ouro reports loss spikes when training with 8 loop steps.
