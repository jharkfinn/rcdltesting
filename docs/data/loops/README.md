# Looping test data (REPRODUCTION.md §7.5d)

| File | Content |
|---|---|
| `predictions.md` | Predictions L1-L5, written before any Cos-Bochner or biography result |
| `amendment_2.md` | Post hoc change of the recall measure: late accuracy (mean over the last quarter of training) is primary, final-step accuracy is still reported |
| `amendment_1.md` | Protocol amendment before any probe v2 result: L3 measured on the tangent space (the plain radius is at least 1 by construction), dilution control for L2, 3 seeds, controls for L1 |
| `predictions_width.md`, `predictions_basis.md`, `predictions_long.md` | Predictions W1-W4 (width sweep), the T-rule (basis tests) and C1-C3 (60,000-step runs), each written before its runs |
| `predictions_zeroinit.md`, `predictions_gating_lr.md` | Predictions Z1-Z3 (zero-initialized MLP control), G1-G5 (InfiLoop gating) and R1-R2 (learning-rate decay), written before the runs |
| `literature.json` | Literature workflow: 18 candidates, 8 papers read end to end (structured notes), synthesis |
| `papers/2610.11570_infiloop.md` | Notes on Li et al., InfiLoop (read end to end) |
| `verify_interim_2115.json` | Verification workflow on the interim basis and width results (all issues applied) |
| `verify_tests123.json` | Verification workflow on the initialization control, Tests 1-3 and the closing summary (all issues applied) |
| `results/facts_loops/` | Facts round 1, 18 runs (`result.json` per run: config, training log, `loop_probe`) |
| `results/facts_loops_v2/` | Facts round 2 (probe v2: tangent spectrum, dilution control, trajectory to 16 passes): seed 3 of the main arms, plus seed-1 repeats at K = 1 |
| `results/facts_loops_ctrl/` | Facts controls (untied passes, GeGLU 4,096) |
| `results/facts_loops_width/` | Width sweep, Cos-Bochner σ = 4 at D = 256 to 8,192 |
| `results/facts_loops_basis/` | Basis tests: ReLU random features, Cos-Bochner with W trained, trained ReLU MLP |
| `results/facts_loops_long/` | 60,000-step runs at a constant rate (D = 256, D = 1,536, D = 16,384, GeGLU 512) |
| `results/facts_loops_long_cosine/` | 60,000-step runs at D = 256 with the rate decayed to 0 by a cosine |
| `results/facts_loops_zeroinit/` | MLPs with a zero-initialized output weight |
| `results/facts_loops_infiloop/` | InfiLoop gating, facts (with residual K = 8 baselines) |
| `results/bios_loops/`, `results/bios_loops_infiloop/` | Biography runs: residual looping (K = 1, 2, 4) and InfiLoop gating (K = 4) |
| `round1/tables.md`, `round1/summary.json` | Tables from `python -m experiments.loops_analysis --facts results/facts_loops --facts_ctrl results/facts_loops_ctrl --out round1` |
| `interim_2110/` | Interim tables for the basis tests and width sweep |
