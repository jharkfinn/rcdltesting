# Rare classes: protocol amendment 1

Written 2026-10-17, after seed 0 (A0, B0, C0) was analysed and before any seed-1 run. It amends
`rare_protocol_v2.md` item by item against `seed0/bugs_and_questions.md`. Code: `experiments/convnext_rare.py`
(`cmd_compare`, `measure`, `cmd_measure`) and `tests/test_rare_analysis.py`. The commit that contains them freezes
the analysis code for seed 1.

Rules:
- P1, P2, P3 and universality (C1) keep their definitions and decision rules, except where the code departs from the
  protocol text or the text contradicts itself. Each such case names the text that governs.
- Everything else is a labelled secondary report. Secondary reports and flags never add or remove a count.
- No rule moves because of the direction of a seed-0 result. Seed-0 values below show a change's effect, not its
  reason.

Codes: (a) fixed in code, (b) added as a secondary report, (c) left as is.

## A. Code against the protocol

- **A1 (a).** Sec. 4.3 says an unfit noise control "is reported as uninformative"; compare called it failed. The noise
  control now has three states: 'uninformative' when the fit check fails or no noise PI is defined, otherwise 'pass'
  or 'fail' by Sec. 7's comparison. The fit check is "top-1 on T >= 50%", and the protocol does not say how the rates
  of the 8 noise classes combine. The aggregator is the one the analysis code fixed before seed 0 (fdff146), kept as
  the resolution of a detail the protocol leaves unstated: the median over the noise classes of their per-class top-1
  on T must be at least 0.5. Two other readings are possible, pooled top-1 over the noise images (the mean here, since
  each class has 10) and a per-class check that keeps only the fitted classes. They are reported descriptively
  (`fit_T_mean`, `cls_fit_ge_half`) and decide nothing. `decision.controls_state` is given per seed, and the note then
  reads "noise control uninformative ...: the primaries are not validated by it". Seed 0: uninformative under all three
  readings (median 0, mean 0.25; the fitted classes 122 and 617 have no defined PI).
- **A2 (a).** Sec. 6 says rare - C0 contrasts must "exceed" the same-seed floor; a floor test pointing the other way
  does not. A floor now passes only with p < 0.05 and sign(z) = sign(median), for the P2 count and P2 support.
- **A3 (a), a contradiction.** Sec. 3 asks for the C0 reference "computed both by exact excision and to first order";
  compare used only the exact one. The text gives neither computation precedence, and three passages bear on the
  choice. The reference is c's whole own class "(shares do not depend on scale)", the premise that lets the exact and
  first-order computations stand in for each other. P2's own excision is justified as "(small edits)". Risk 2 says
  "Every statistic is local and exact, or first order, against size-matched nulls; whole-network excision of class
  sets is avoided." Exact excision of a whole common class is a large edit, seed 0 shows that the premise fails (exact
  minus first order +0.090 [0.078, 0.113]), and the choice decides P2's sign at seed 0. Resolution: the exact
  whole-class reference stays the primary statistic (Holm, decision) because the analysis code fixed it before seed 0
  (fdff146, `P2_whole.lambda3`); the text does not make it the better choice. The first-order whole class and Own_m
  in C0 (the protocol's sensitivity check) are secondary (`secondary.P2_ref_first_order`, `secondary.P2_ref_own_m`).
  The text never ties the decision to agreement between references, so a count against the essay does not need all
  three negative. Disagreement is a secondary flag on the P2 candidate. Seed 0: -0.032, +0.069 and +0.060 (n = 110
  each).
- **A4 (c), with (b).** "PI is defined when ...; otherwise the class counts as 'no route'" means a no-route class has
  no PI, so the median over the 112 is the median of the PIs that exist, with the no-route count reported. This stays
  primary. Secondary (`secondary.P1_alternatives`): the raw median over all 112, the median with 'no route' as 0, the
  median over classes with F_pct <= 0.025 or >= 0.975, and the counts in each 5% tail of F_pct.
- **A5 (a).** Sec. 3 defines PI, lambda_3 and Delta_flow through their 95th-percentile gates, so they exist only where
  the gates hold, and every statistic built from them uses the same gate. This covers:
  - the cells, noise against real, and the rare - C0 and S5 contrasts (both conditions);
  - the P2 floor;
  - R_own - R_mis, under P3's gate (it enters the support rule with P3), including the noise control's;
  - the C0 references of P2: exact by `P2_whole.defined`, first order by `P2_whole_fo.defined`, Own_m by `P2.defined`,
    with the classes each gate excludes reported (`n_ref_undefined`);
  - the floors C0 - E1, E1r - E1 and C1 - C0, where PI and the whole-class lambda_3 enter only where defined in both
    runs.

  Raw values remain only as labelled `_raw` reports (`noise_control.R_own_minus_mis_raw`; `PI_raw` and
  `P2_whole.lambda3_raw` in the floors). F, O and AUROCs have no gate. Seed 0: all 112 C0 references are defined (the
  one undefined whole-class lambda_3 in C0, class 349, is a noise control), so P2 is unchanged. The noise control's
  R_own - R_mis has 1 gated value (class 915, +0.0023) against 8 raw.
- **A6 (b).** Sec. 6 asks for a "one-sided binomial test"; the code's test is one and stays. Secondary: the exact
  paired McNemar test, signature against pseudo-signature (seed 0: 18 against 1, p = 3.8e-5).
- **A7 (a).** Sec. 6's floor covers every rare - C0 contrast; S2 now has it per stage and head.
- **A8 (a).** Sec. 3 lists RBp^x on K(c) and weighted by own effect; compare adds RBpx_w (intercept at RPp_w = 0.5,
  C0 medians).
- **A9 (a).** S6 lists train fit on T_c. With --train, `measure` computes fit_T (top-1 of the training label on center
  crops of all own training images); compare adds fit_T and auroc_near to the rare - C0 contrasts.
- **A10 (c), a contradiction, with (a) and (b).** Sec. 7: "Counts against the essay. Any one of the following, after
  Holm correction, beyond the floors and replicated in seed 1:", then "median PI < 0", "median lambda_3 difference
  < 0", "the 95% interval of median Delta_flow lies below -delta" and "an excess of packing signatures over the
  pseudo-own count". Sec. 6:
  "Holm correction over P1 to P3 at alpha 0.05" and "Contrasts between rare and C0 must also exceed the same-seed
  floor: a paired Wilcoxon test of D(c) against D0(c), p < 0.05." Sec. 4.4 defines the floors as differences between
  runs: "C0 - E1", "D0(c) = the other set's run - C0", "E1r - E1, and C1 - C0". Read alone, Sec. 7 asks for a floor on
  all four counts. Sec. 6, the only passage that says how a floor is tested, defines the test for rare - C0 contrasts,
  and three of the four counts are not contrasts. The text contradicts itself, and the choice decides whether the two
  pending seed-0 candidates can count. Resolution, on textual grounds: Sec. 7's three qualifiers name Sec. 6's three
  procedures, and each applies with the scope Sec. 6 gives it.
  - Holm is "over P1 to P3". It applies to the P1, P2 and P3 counts but not to universality, whose test Sec. 6 sets
    apart (the one-sided binomial). So "after Holm correction" cannot apply literally to all four counts either.
  - The floor test is for "Contrasts between rare and C0". Of the four counts only P2 (lambda_3(rare) -
    lambda_3^ref(C0)) is one. P1, P3 and universality are statistics of the rare condition alone, as their claims are
    stated in Sec. 0: C2a, C2b and C1 are stated within the rare class, and only C4 is stated against "the same images
    when the class is common". A floor in Sec. 4.4 is a difference between runs, and the protocol defines no test of
    one against a level.
  - Replication applies to all four.

  The analysis code fixed this reading before seed 0 (fdff146: `floor_ok` True for P1, P3 and universality), and it
  is kept. The candidates now say what it means: `floor_ok` is None ("no floor applies") rather than True. "Beyond the
  floors" was pre-registered, so applying it would not be post hoc; what this item settles is what the phrase means.
  One code departure is fixed: the P3 count checked only its interval and skipped Holm. It now also needs P3's
  Holm-adjusted p < 0.05 (Holm over P1 to P3, each tested two-sided against 0, as Sec. 6 sets it). Each candidate
  carries `holm_p` (None for universality, with its test named). Secondary, under each candidate's `secondary`, never
  decisive:
  - `also_in_common`: the same statistic and test give the same result in that seed's common run (C0 or C1): median
    PI < 0 at p < 0.05, a binomial signature excess, or a P3 interval below -delta. Matches are listed in
    `decision.secondary_flags`.
  - The other reading of "beyond the floors": the same statistic against the same-seed run where c is common (Wilcoxon
    of PI and of Delta_flow, McNemar of signatures; `secondary.vs_other_set_run`).
  - The paired McNemar test of signatures, rare against common.

  Seed 0: both pending candidates are flagged as also holding in C0 (median PI -0.998, n = 24; signatures 11 against 5,
  p = 0.034). Against the other-set run: PI p = 0.69 (6 classes defined in both runs), signatures 17 against 10
  (McNemar p = 0.12).
- **A11 (a).** Sec. 6: "Replication: seed 1 must give the same sign, one-sided p < 0.05; otherwise the result is
  reported as 'not replicated'." Sec. 7's support rule does not repeat this, and compare listed support from seed 0
  alone, so support that failed to replicate stayed listed. Each support entry now carries `replication`:
  - 'pending seed 1' without seed 1;
  - 'replicated' when P1 and P2 replicate by Sec. 6's rule and seed 1 meets P3's part of the rule (the interval of
    median Delta_flow not below -delta, with R_own > R_mis);
  - 'not replicated' otherwise.

  P3's part is an interval criterion, not a sign. It therefore replicates the way the P3 count against the essay
  already does: by meeting the same criterion at seed 1. Seed 0: no support, so no effect.

## B. Protocol design

- **B1 (a), with A1.** Sec. 4.3 decides the state: it is the specific rule for this control, and its premise ("if
  memorized") is false when the fit check fails. Sec. 7's failure rule applies to an informative control. An
  uninformative control validates nothing, so the primaries stay unvalidated without having failed a control.
- **B2 (c), with (b).** The definedness rule is as written and coded; changing it after seeing its one-sided action at
  seed 0 would be post hoc. The A4 secondaries show its effect.
- **B3 (c), with (b).** No calibration criterion was pre-registered, and none is added. Descriptive: the fraction of
  classes whose C0 |O_whole| exceeds C0's definedness threshold.
- **B4 (c).** The criterion stays for both arms. Its label is corrected: the interpolated 99th percentile of 13 null
  values is near their maximum, so chance is about 1 in 14 per class.
- **B5 (c).** The pseudo-own is the pre-registered comparison. The rare-against-common McNemar test (A10) addresses its
  non-exchangeability, because in the common run K(c) is also selected on c's images.
- **B6 (c).** The protocol does not define N_own for controls; the code's choice stays so the seeds match. If the fit
  check ever passes, the report must say the comparison reduces to the sign of noise F (`n_F_negative`).
- **B7 (b).** L_S stays the joint knockout. Sec. 3's other P1 versions are now reported (`secondary.P1_versions`), over
  all and defined classes: gate-only F, value-only F, F_lin_pct - 0.5, O_lin_pct - 0.5, F_gate - O_gate, and F for
  absent classes.
- **B8 (c).** As written; delta comes from each seed's own data.
- **B9 (c), with (b).** The median rule stays (the text gives no aggregator). Secondary: a sign test and the same-seed
  floor of the rare - C0 differences (`rare_minus_C0`, aligned with `cls_rare_minus_C0`). A missing relabel control is
  'uninformative', not 'fail'.

## C. Outputs and reproducibility

- **C1 (a), procedure.** Sec. 4.4 requires the floors: compare is passed --e1, --e1r and --c1.
- **C2 (a), procedure.** Each pre-registered analysis missing from the seed-0 outputs, with its status:
  - Run at both seeds with the frozen code:
    - measure at 12,500, and at 5,000 for P2;
    - churn (S4, with E1r's attrR floor);
    - Phase 0 on E1 and E1r;
    - compare on step 12,500 (secondary, C5).
  - Added to `measure` here, all secondary:
    - The sensitivity checks of Sec. 2 and 4.7, with Near10 replaced by Near20 or by the WordNet relatives (distance
      <= 4). They give F, O, PI and its gate (`sens`) and R_own, R_flow, R_mis and Delta_flow (`P3.sens`). Each set is
      drawn, as Near10 is, from the classes common in c's rare run and in C0, and has its own matched draws from Pi(c)
      without the set. The WordNet set is empty for 15 of the 56 iso primaries; its size is 8 to 45 for dense classes
      and 0 to 4 for iso classes. Near20 contains Near10 for all 112 primaries.
    - P1 at mean pooling instead of p* (`sens.mean_pool`, Risk 7): u, g, v and the parts are taken at every position
      of the V_est images and averaged, with sigma from the class-mean pooled u on Q.
    - Near10 from training-image DINOv2 means, behind `measure --visual_train` (a class-distance file in the format of
      `imagenet_dinov2_classdist.npz`). No such file exists: the repository's file is built from the 50 validation
      images per class, and `experiments/class_embed.py` reads validation images only. Unless the file is produced
      before the seed-1 analysis, this check is reported as not run, for that reason.
    - S6's family-correct rate. The protocol gives no formula, so this amendment defines it before seed 1: the
      fraction of probe images whose top-1 class is c or a class at WordNet distance <= 4 from c (the protocol's own
      WordNet-relative threshold). It is reported as `family_top1`, with its rare - C0 contrast and floor.
    - S7, "Selected classes are also compared, rare against C0". S7 now runs in every `measure`, not only with
      --phase0. Each class record carries its C1, PR and N, and compare reports `rare_minus_C0_S7_C1`, `_PR` and `_N`
      with the floor.

    compare summarizes the sensitivity checks under `secondary.sensitivity`, with the same gates (A5).
  - Dropped: Udec and Rdec. Sec. 4.1 names them as descriptive but gives no formula.

  The seed-0 files have none of the added fields; these appear when seed 0 is re-measured (see Seed 1).
- **C3 (b).** `measure` saves each class's per-unit knockouts (Near10, the 1,000 draws, own, N_own) to
  `_arrays.npz`, which shows the F null's shape (D2).
- **C4 (a).** The class is the unit of analysis (Sec. 6), so compare sorts by class id before every bootstrap.
  Seed 0: P2's p moves from 0.107 to 0.098.
- **C5 (a).** Everything outside the primaries is labelled as a secondary report (Rules):
  - Each seed's compare block lists the roles of its keys in `roles`. The primaries are P1, P2, P3 and universality.
    The decision inputs are P3_own_minus_mis, noise_control and relabel_control. Every other key is secondary.
  - This amendment's additions sit under `secondary`: `P1_alternatives`, `P1_versions`, `P2_ref_first_order`,
    `P2_ref_own_m`, `common`, `universality` (the McNemar tests and the common run's test), `vs_other_set_run` and
    `sensitivity`. The primary `universality` block holds only the pre-registered test.
  - Each decision candidate holds its decisive fields (`claim`, `holm_p`, `floor_ok`, `replicated`) and a `secondary`
    dict.
  - compare reads the step of the measure files, which must all share one step. At a step other than 25,020 it writes
    the decision block as `decision_secondary`, with the role "secondary: step ...", and writes no `decision`.

  Every key of the pre-amendment `rare_compare_seed0.json` is still written.

Section D is not decided here. D1 (a new noise design) is not adopted for seed 1: it would be post hoc, and seed 1
uses the same specs. D3, D4 and D8 are covered by the secondaries above.

Seed 0 under the amended compare: Holm p is below 1e-4 for P1 and P3 and 0.098 for P2. There are no counts and no
support. "median PI < 0" and "excess of packing signatures" are pending. Neither has a floor (A10), and both are
flagged as also holding in C0. Controls: uninformative.

## Seed 1

Runs use the same specs (`spec_A.npz`, `spec_B.npz`, `spec_C.npz`, unchanged) with seed 1. C1 is required, because
the common-run flags and the C1 - C0 floor need it:

    python3 -m experiments.convnext_train --size tiny --opt adamw --track all --track_head \
      --spec spec_{A|B|C}.npz --snapshots --resume_every 2500 --seed 1 --out results/rare_{A|B|C}1

Step 4's go/no-go passed at seed 0; at 12,500 it is descriptive.

Analysis, with the frozen code:
- For A1, B1 and C1: the atlas, `flow`, `churn`, and `measure` at 25,020 and 12,500 (P2 also at 5,000). `measure`
  uses --train, --flow and the default draws (--n_draws 1000, --seed 0). It adds --visual_train only if the
  training-image class-distance file exists (C2).
- For E1 and E1r: Phase 0 measures at 25,020 (--phase0), if missing.
- For seed 0: a re-measure that adds fit_T, the nulls, the earlier steps, the sensitivity checks, family_top1 and S7
  per class. These secondaries come from a second compare run. The decision compare keeps the committed seed-0 files,
  and any per-class change in the primaries is reported.

Decision compare:

    python3 -m experiments.convnext_rare compare \
      --rare rare_A0/rare_measure_25020.json rare_B0/rare_measure_25020.json \
      --c0 rare_C0/rare_measure_25020.json \
      --rare1 rare_A1/rare_measure_25020.json rare_B1/rare_measure_25020.json \
      --c1 rare_C1/rare_measure_25020.json \
      --e1 E1/rare_measure_phase0_25020.json --e1r E1r/rare_measure_phase0_25020.json \
      --out rare_compare_seed1.json

With --c1, C1 is seed 1's common reference throughout. E1 and E1r enter only the floors C0 - E1 and E1r - E1. The
same command on the 12,500 files gives the secondary step, written as `decision_secondary` (C5).
