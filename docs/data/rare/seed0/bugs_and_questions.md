# Rare classes, seed 0: suspected bugs, protocol deviations and open questions (2026-10-17)

Sources:
- Code: `experiments/convnext_rare.py` (line numbers at the current HEAD), `experiments/rare_common.py`,
  `experiments/rare_gonogo.py`.
- Protocol: `rare_protocol_v2.md`.
- Outputs: `rare/s0/results/`.

All numbers are seed 0, step 25,020, n=112 primary classes unless stated.

## A. Code that departs from the protocol, or makes a decision wrongly

1. **`convnext_rare.py:1459` and `:1465`: an uninformative noise control is reported as failed.**
   - `controls_ok = noise.passes and relabel.passes`. It never reads `noise_control['informative']`, which is computed
     at `:1364` and is false at seed 0 (fit_T [1,0,0,0,0,0,1,0], median 0).
   - Protocol Sec. 4.3 says an unfit noise control "is reported as uninformative". The note says "positive controls
     failed: primaries are unvalidated".
   - Sec. 7's literal failure condition is also met: noise median PI +0.993 (n=8) against real -0.970 (n=112). So the
     protocol itself is inconsistent here; see question B1.
   - Fix: report three states (pass, fail, uninformative) and decide which section governs.
2. **`convnext_rare.py:1433-1434`, used at `:1440-1442` and `:1456`: `floor_ok` checks only that the floor
   Wilcoxon has p < 0.05, not its sign.**
   - At seed 0, P2's median is -0.032, but its floor test is z = +3.61, p = 3.1e-4: the rare run is deeper than the
     other-set run.
   - Had P2 been Holm-significant, "median lambda_3 difference < 0" would have passed a floor that points the other
     way, and the support rule would have the mirror-image problem.
   - Fix: require sign(z) == sign(median).
3. **`convnext_rare.py:1281`: the P2 reference is only the exact whole-class excision (`P2_whole.lambda3`).**
   - The protocol (Sec. 3, P2) says the reference is "computed both by exact excision and to first order (shares do not
     depend on scale)".
   - The first-order value is in the measure files (`P2_whole_fo`) but `compare` never uses it.
   - Exact whole-class excision is far from linear: exact over first-order stage ratios are 1.31, 1.34, 0.18 and 0.35,
     and lambda_3 is inflated by +0.090 [0.078, 0.113].
   - P2 with the first-order reference is +0.069 [0.045, 0.082] (n=110), against -0.032 with the exact one.
4. **`convnext_rare.py:1276`: P1 is the median over defined classes only (n=23).**
   - The protocol (Sec. 3 and 6) says "median PI over the 112 rare classes" and does not say how 'no route' enters.
   - The alternatives differ a lot: raw PI over all 112 gives -0.970 [-0.996, +0.220], p = 0.14. Counting 'no route'
     as 0 gives 0.0.
   - This is an ambiguity in the protocol that the code resolved in one direction. It should be stated in the
     protocol before seed 1.
5. **Inconsistent samples for PI.** P1 uses the definedness gate. Several other statistics use raw PI, including 'no
   route' classes:
   - the cells (`convnext_rare.py:1354`);
   - the noise-against-real comparison (`:1361-1362`, `:1372`);
   - the rare - C0 contrasts for PI, F, O, lambda_3 and R_own (`:1303-1314`), which apply no `defined`, `P2.defined`
     or `P3.defined` filter;
   - S5 (`:1316`).

   Examples:
   - Cell dense/100 has median PI +0.043 raw against -0.992 on its defined classes only (n=5).
   - rare_minus_C0_PI has n=112, although only 23, 24 and 26 PIs are defined in the three conditions.
   - rare_minus_C0_lambda3 includes classes 203 and 279, which are undefined in the rare run. This has a negligible
     effect: +0.060 with both runs defined, against +0.062.

   With the gate applied, the noise control still fails: -0.9994 (n=3) against -0.99987 (n=23).
6. **`convnext_rare.py:1347`: the universality binomial treats the pseudo rate (kp + 1) / (n + 2) as known, with
   kp = 1.**
   - It ignores the sampling error of the pseudo count and is not a paired test, so it is anti-conservative.
   - An exact paired McNemar test (18 against 1 discordant, 0 with both) gives one-sided p = 3.8e-5, against 1.46e-12.
7. **`convnext_rare.py:1334-1341`: S2 rare - C0 is reported without the same-seed floor.** Protocol Sec. 6 requires
   the floor for rare - C0 contrasts. The only nominal effect, stage 2 at +0.047 (p = 0.007, n=56), equals its floor
   D0 = +0.055. Rare against the other-set run gives Wilcoxon p = 0.84.
8. **`convnext_rare.py:1320-1332`: S1 reports only the carrier version RBpx_K.** It omits the own-effect-weighted
   RBpx_w that protocol Sec. 3 also lists. RBpx_w has median 0.470 (C0 0.485), which is chance, and would temper the
   K-based intercept of 0.606.
9. **`convnext_rare.py:764` (measure): S6 "train fit on T_c" is not computed at 25,020 for real classes.**
   - `top1` there is measured on the probes, which are the validation images V_c for real classes.
   - Fit on T_c exists only in the go/no-go at 12,500 (`rare_gonogo.py:162`), where it is 0 in every rare cell.
   - For noise classes the probes are the training images, so `compare` correctly uses `top1` as fit (`:1363`).
10. **`convnext_rare.py:1439` and `:1448`: `floor_ok` is hard-coded to True for the P1 and universality candidates.**
    - This matches the protocol's letter, which requires a floor only for rare - C0 contrasts.
    - But C0 shows the same P1 pattern (24 defined, all F < 0, median PI -0.998) and a signature excess of its own (11
      against 5). So the decision rule can count against the essay something that does not depend on rarity.

## B. Protocol-level design issues (no code deviation)

1. **Noise control, contradictory rules.** Sec. 4.3 (fit check, otherwise uninformative) and Sec. 7 (noise PI not
   below real means the controls failed) both apply at seed 0, and the text does not say which one wins.
2. **Definedness (`convnext_rare.py:403-413`) compares |F| + |O| with the 95th percentile of |F_j| + |O_j|.**
   - F (ten whole classes) and O (10 or 100 images) are on very different scales: |O| < |F| in 109 of 112 classes, and
     sign(PI) = sign(F) in 110 of 112.
   - The F_j null appears left-skewed (null median below 0 in 104 of 112).
   - Together these make 'defined' equal to F_pct <= 0.05 in the primary sample (23 = 23). None of the 13 classes with
     F_pct >= 0.95 is defined.
   - P1 then tests the sign of F in the lower tail. It is not own against relatives.
3. **Calibration has no criterion (`convnext_rare.py:1377-1381`).** Protocol Sec. 4.3 says only that the C0 whole own
   class "gives a large O". Against what PI needs it fails: |O_whole| is below C0's definedness threshold in 112 of 112
   classes (median threshold 14.9, max |O_whole| 12.0). So P1 cannot detect even a real own-built niche.
4. **The O criterion of the packing signature is loose (`convnext_rare.py:415`, `:420`).** `np.percentile(Oj, 99)` over
   13 values (12 for the pseudo-own) is effectively the maximum, so the nominal 1% is about 7%. Both arms share this,
   but the label overstates how strict the criterion is.
5. **The pseudo-own (`convnext_rare.py:417-420` with `rare_common.py:95-97`) is the first N_own class in sorted order.**
   - So 16 distinct pseudo groups serve 112 classes, and the pseudo-signatures are not independent.
   - The pseudo-own is not exchangeable with Own_m either: K(c) is selected on c's own validation images, and Own_m
     are images of c.
6. **N_own for the noise controls (`rare_common.py:95-97`) is the other 3 noise classes of the set.**
   - They share the same carrier route (8 of 8 carriers in common for every A0 pair).
   - So O is about 0, the O null has 3 values (O_pct in {0, 1/3, 2/3, 1}), and the signature test runs on 3 values.
   - The protocol does not define N_own for controls.
   - Since every noise PI equals sign(F), the noise comparison passes if and only if 5 or more of 8 noise F are
     negative. That has probability 0.36 if F is symmetric, and does not depend on O.
7. **Joint gate-and-value knockout of ten whole classes (`convnext_rare.py:290-300`, `:360-385`; protocol Risk 2).**
   - The joint F differs from F_gate + F_value in 98 of 112 classes (joint below the sum), median F - F_gate - F_value
     = -11.2.
   - Per image, the difference is -s mean[(SiLU(g) - SiLU(g - G)) V] / sigma.
   - This does not explain P1 in the defined set: 13 of 23 defined classes have both single pathways negative. But it
     makes F hard to read as "feeding".
8. **P3's tolerance is degenerate (`convnext_rare.py:1297`).** delta = 0.2 x median(R_flow - R_mis) = 8.6e-5,
   because R_flow is about R_mis (+0.00043). The tolerance therefore reduces to Delta_flow >= 0. This matches the
   protocol as written.
9. **The relabel control has no aggregator.** The protocol says only that "source must exceed label"; `compare` uses
   the median of the differences (`convnext_rare.py:1376`).
   - Per class it fails for 2 of 8 (classes 11 and 113). The sign test gives p = 0.145.
   - The same statistic varies as much between runs with nothing relabelled (median |other - C0| 13.9) as between the
     rare run and C0 (16.5).

## C. Missing outputs and reproducibility

1. The floors floor_instrumentation_C0_minus_E1, floor_seed_E1r_minus_E1 and floor_seed_C1_minus_C0 are null, because
   `compare` was run without `--e1` and `--e1r` and there is no C1 (`convnext_rare.py:1423-1425`). Protocol Sec. 4.4
   requires them.
2. Several pre-registered analyses are absent from the seed-0 outputs:
   - step 12,500 for P1, P3 and S1 to S3, and steps 12,500 and 5,000 for P2 (Sec. 5, Step 6);
   - S4 churn and tau (there is no rare_churn.json);
   - S7;
   - the Udec and Rdec descriptives;
   - S6 fit on T_c at 25,020.
3. The per-draw arrays (rand_both, nown_both) and the per-unit g, v, G_S and V_S at p* are not saved. The only .npz
   files on disk are smoke-test arrays. So the left skew of the F null and the joint-knockout mechanism can only be
   inferred.
4. The bootstrap depends on class order. `rare_recs` (`convnext_rare.py:1267-1271`) fills a dict with A0's rare
   records, then B0's.
   - In that order every number reproduces bit for bit.
   - Sorting by class id moves contrast intervals and p values in the third digit: rare - C0 PI p is 0.954 against
     0.935, and the S5 PI interval is [-0.047, 0] against [-0.051, 0].
   - Sorting by class id before bootstrapping would make the results independent of the order the files are passed in.

## D. Open questions

1. **Validation.** Can the noise control be made informative under this recipe? Only 2 of 8 noise classes reach top-1
   on their 10 images by step 25,020. Options include more images, one texture per label, or O computed against real
   classes' N_own. Each is a post hoc change. Without a working positive control, P1 cannot be validated at all.
2. **Why is the F null left-skewed, and why are large-|F| classes uniformly F < 0?** This holds for 23 of 23 defined
   classes in rare and 24 of 24 in C0. Two candidates:
   - K(c) is selected by specific effect, so relatives (hard negatives) may write inhibitory parts there;
   - gate-opening removals interact with value suppression.

   Saving the F_j draws and g, v, G and V at p* for a sample of classes would decide it. So would a run of P1 with
   carriers selected by raw rather than specific effect.
3. **Should a scale-free variant decide P1?** Candidates are F_pct against O_pct, separate tests of F and O against
   their own nulls, or the pre-registered linear version (F_lin_pct above 0.5 in 80 of 112). Any of them would be a post hoc
   change and must be labelled as one.
4. **Which P2 reference should govern?** The candidates are exact whole class (0.450), first-order whole class (0.353)
   and Own_m in C0 (0.365). The sign of P2 depends on the choice, and the protocol's assumption that shares do not
   depend on scale fails at seed 0.
5. **Why does excising another rare level-10 class's Own_m from stage 3 lower zeta_c of rare class c by about 0.30?**
   In C0 the figure is about 0. Is there a shared stage-3 direction that all AdamW-boosted rare head rows read? This
   null shift drives part of the level-10 lambda_3 change.
6. **Head share against optimizer.** The head share (0.91 rare, 0.56 C0) tracks the AdamW effective-step ratio
   (median 6.1; Spearman 0.76). Only the optional Muon arm can separate "memorization sits near the output" from
   AdamW's per-row normalization.
7. **What does P3 measure?** Does R_own > R_flow reflect where own writes go, or only which units move zeta_c? A flow
   null weighted by |d zeta_c / d rows n| (relevance without own writes) is not computed. C0 gives the same Delta_flow
   either way.
8. **Should rare - C1 decide C2a and C1 at seed 1?** P1's pattern and the signature excess appear equally where the
   classes are common, so a paired rare-against-common comparison may be the right test. In the protocol it is only a
   secondary for PI and absent for universality. Phase 0 was meant to give the PI distribution for common classes,
   and it is not in these outputs.
9. **Do these runs test memorization at all?** Rare classes have top-1 0 on validation and fit_T 0 at 12,500, so the
   tiny model does not memorize them. A wider model (the protocol's fallback) or a fit measure at 25,020 is needed for
   the essay's memorization claims.
10. **Noise logits.** Are the 4 noise labels of a run the top-4 logits on every noise image? This needs per-image
    logits. It is inferred from top-5 = 1.0 for all 8 and exactly one top-1 winner per run.
11. **Stage-0 carriers.** Why do A0 and B0, but not C0, recruit stage-0 units as carriers for many classes? Stage 0
    has 133 carriers in rare against 9 in C0, and A0's stage0.block2 unit 21 is a carrier for 17 of its 56 rare primary classes. Do the noise
    images or the missing rare data cause it, and does it bias P1 (the stage-0 carrier count correlates with the
    threshold, Spearman 0.55)?
12. **Relatedness over training.** Does the relatedness of own writes (R_own - R_mis) grow between steps 12,500 and
    25,020, as the essay's "semantic gravity" strengthening over training implies? Step 12,500 is pre-registered as
    secondary but was not measured.
13. **Seed 1.** A1, B1 and C1 have not been run. Both 'against' candidates (median PI < 0 and the signature excess) are
    pending, and seed 0 already puts their interpretation in doubt.
