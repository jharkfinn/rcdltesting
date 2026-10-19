# Looping test: amendment 1 (before any probe v2 result)

Written 2026-10-19 18:25 UTC. Round 1 facts results were seen; no probe v2 result, no control result and no biography result had been seen.

## Why

The literature review (docs/data/loops/literature.json) found a flaw in the round 1 Jacobian measure.

The step is s(h) = h + a f(LN(h)). LayerNorm ignores scale and a shift along the ones vector: LN(c h) = LN(h) and LN(h + t 1) = LN(h), up to its eps. So J h = h and J 1 = 1, and the spectral radius of J is at least 1 by construction. RMSNorm in the biography model ignores scale only, so J x = x.

Here:
- h is the state before a pass;
- a is the step size;
- f is the FFN;
- J = ds/dh is the step Jacobian;
- 1 is the ones vector.

A radius above 1 therefore says nothing about expansion.

## Changes

1. **L3 is re-stated** on the tangent space. The trivial directions (h and 1 for LayerNorm, x for RMSNorm) are projected out (rcdl.facts.tangent_spectrum).
   - Measures: the median tangent radius and the share of tangent eigenvalues below 1 in modulus.
   - L3 (narrow kernel has the largest per-pass Jacobian) is judged on the tangent radius.
   - The round 1 plain radius stays in the tables, marked "at least 1 by construction".
2. **L2 (clean-up) gets a dilution control.**
   - The same noise vector n is added either before the passes (propagated) or after the clean passes (end-added).
   - Clean-up means: accuracy with propagated noise is higher than with end-added noise, at the trained K.
   - L2 holds only if this margin is larger for Cos-Bochner than for GeGLU, and larger at K = 4 than at K = 1.
   - Round 1 compared noisy accuracy across K only, which mixes processing with dilution.
3. **Seeds:** 3 seeds for the main facts arms and for the biography arms.
4. **New decision rule for L1:** the gain must hold in all 3 seeds and must exceed two things:
   - the untied control (separate weights per pass);
   - the compute-matched wider layer at K = 1.

   If the untied or wider control gains as much, the gain is attributed to extra depth or width, not to reusing one map.
5. **Added exploratory measures, not tests:**
   - the trajectory to 16 passes (accuracy, flips, norm growth, step cosine);
   - test-time step size;
   - the σ sweep 2 / 3 / 4 / 6 / 8;
   - the tangent spectrum at noisy states.
