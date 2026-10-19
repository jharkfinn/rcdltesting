# Looping test data (REPRODUCTION.md §7.5d)

| File | Content |
|---|---|
| `predictions.md` | Predictions L1-L5, written before any Cos-Bochner or biography result |
| `amendment_2.md` | Post hoc change of the recall measure: late accuracy (mean over the last quarter of training) is primary, final-step accuracy is still reported |
| `amendment_1.md` | Protocol amendment before any probe v2 result: L3 measured on the tangent space (the plain radius is at least 1 by construction), dilution control for L2, 3 seeds, controls for L1 |
| `literature.json` | Literature workflow: 18 candidates, 8 papers read end to end (structured notes), synthesis |
| `results/facts_loops/` | Facts round 1, 18 runs (`result.json` per run: config, training log, `loop_probe`) |
| `results/facts_loops_ctrl/` | Facts controls as they finish (untied passes, compute-matched width, GeGLU 4096, N = 16,384) |
| `round1/tables.md`, `round1/summary.json` | Tables from `python -m experiments.loops_analysis --facts results/facts_loops --facts_ctrl results/facts_loops_ctrl --out round1` |

Round 2 results (facts probe v2, biographies) are added under `results/` as they finish.
