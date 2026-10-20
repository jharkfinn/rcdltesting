# Amendment 5: a fifth seed for the split late-phase rules (written before these runs)

Seen: all late-phase runs of amendment 4 on seeds 3 and 4 at 10,000 steps (LP1 to LP3), and the third seeds of C2b, D2 and B2. LP4 (decayed rate) had not been seen.

Two rules split:
- LP2: GeGLU P90 = 1.01 / 0.95 against ≤ 1.0.
- LP3 for Cos σ = 4: late gap +0.7 / +1.1 points against ≥ 1.0.

Amendment 2 gives a split rule one more seed. So seed 5 runs the same configurations:
- Cos σ = 4 at K = 1 with a = 8 and 16, and at K = 4 with a = 2 and 4;
- zero-initialized GeGLU 512 at K = 1, a = 2 and K = 4, a = 0.5.

That makes 6 runs (`facts_loops_grid.json`). The best scale stays the one chosen on seeds 3 and 4. LP2 and LP3 (σ = 4) are then decided by majority over seeds 3, 4 and 5. No other rule changes.
