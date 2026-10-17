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
  control now has three states: 'uninformative' when the fit check fails (median top-1 on T below 0.5) or no noise PI
  is defined, otherwise 'pass' or 'fail' by Sec. 7's comparison. `decision.controls_state` is given per seed, and the
  note then reads "noise control uninformative ...: the primaries are not validated by it". Seed 0: uninformative.
- **A2 (a).** Sec. 6 says rare - C0 contrasts must "exceed" the same-seed floor; a floor test pointing the other way
  does not. A floor now passes only with p < 0.05 and sign(z) = sign(median), for the P2 count and P2 support.
- **A3 (a).** Sec. 3 asks for the C0 reference "both by exact excision and to first order"; compare used only the
  exact one. P2 is now reported with the exact whole class (primary statistic, Holm, decision), the first-order whole
  class and Own_m in C0 (the protocol's sensitivity check). Exact governs because P2's rare side is an exact
  excision, so only an exact reference is like for like. The text names one statistic and never ties the decision to
  agreement between references, so a count against the essay does not need all three negative. Disagreement is a
  secondary flag on the P2 candidate. Seed 0: -0.032, +0.069 and +0.060.
- **A4 (c), with (b).** "PI is defined when ...; otherwise the class counts as 'no route'" means a no-route class has
  no PI, so the median over the 112 is the median of the PIs that exist, with the no-route count reported. This stays
  primary. Secondary: the raw median over all 112, the median with 'no route' as 0, the median over classes with
  F_pct <= 0.025 or >= 0.975, and the counts in each 5% tail of F_pct.
- **A5 (a).** PI, lambda_3 and Delta_flow exist only where their gates hold, so every statistic built from them uses
  the same gate: cells, noise against real, rare - C0 and S5 contrasts (both conditions), the P2 floor, and
  R_own - R_mis (P3's gate; it enters the support rule with P3). F, O and AUROCs have no gate.
- **A6 (b).** Sec. 6 asks for "a one-sided binomial test"; the code's test is one and stays. Secondary: the exact
  paired McNemar test, signature against pseudo-signature (seed 0: 18 against 1, p = 3.8e-5).
- **A7 (a).** Sec. 6's floor covers every rare - C0 contrast; S2 now has it per stage and head.
- **A8 (a).** Sec. 3 lists RBp^x on K(c) and weighted by own effect; compare adds RBpx_w (intercept at RPp_w = 0.5,
  C0 medians).
- **A9 (a).** S6 lists train fit on T_c. With --train, `measure` computes fit_T (top-1 of the training label on center
  crops of all own training images); compare adds fit_T and auroc_near to the rare - C0 contrasts.
- **A10 (c), with (b).** The code follows Sec. 6, which requires floors only for rare - C0 contrasts; P1 and
  universality are not contrasts, and a new condition would be a rule added after seeing seed 0. Secondary: each
  candidate carries `also_in_common`, true when the same statistic and test give the same result in that seed's
  common run (C0 or C1): median PI < 0 at p < 0.05, a binomial signature excess, or a P3 interval below -delta.
  Matches are listed in `decision.secondary_flags`. A paired McNemar test of signatures, rare against common, is also
  reported. Seed 0: both pending candidates are flagged (C0 median PI -0.998; C0 signatures 11 against 5, p = 0.034).

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
- **B7 (b).** L_S stays the joint knockout. Sec. 3's other P1 versions are now reported, over all and defined
  classes: gate-only F, value-only F, F_lin_pct - 0.5, O_lin_pct - 0.5, F_gate - O_gate, and F for absent classes.
- **B8 (c).** As written; delta comes from each seed's own data.
- **B9 (c), with (b).** The median rule stays (the text gives no aggregator). Secondary: a sign test and the same-seed
  floor of the rare - C0 differences. A missing relabel control is 'uninformative', not 'fail'.

## C. Outputs and reproducibility

- **C1 (a), procedure.** Sec. 4.4 requires the floors: compare is passed --e1, --e1r and --c1.
- **C2 (a), procedure.** The missing pre-registered analyses run at both seeds: measure at 12,500, and at 5,000 for
  P2; churn (S4, with E1r's attrR floor); Phase 0 with S7 on E1 and E1r. compare on step 12,500 is secondary.
  Udec and Rdec stay out: the protocol gives them no formula.
- **C3 (b).** `measure` saves each class's per-unit knockouts (Near10, the 1,000 draws, own, N_own) to
  `_arrays.npz`, which shows the F null's shape (D2).
- **C4 (a).** The class is the unit of analysis (Sec. 6), so compare sorts by class id before every bootstrap.
  Seed 0: P2's p moves from 0.107 to 0.098.

Section D is not decided here. D1 (a new noise design) is not adopted for seed 1: it would be post hoc, and seed 1
uses the same specs. D3, D4 and D8 are covered by the secondaries above.

Seed 0 under the amended compare: Holm p is below 1e-4 for P1 and P3 and 0.098 for P2. There are no counts and no
support. "median PI < 0" and "excess of packing signatures" are pending, both flagged as also holding in C0.
Controls: uninformative.

## Seed 1

Runs use the same specs (`spec_A.npz`, `spec_B.npz`, `spec_C.npz`, unchanged) with seed 1. C1 is required, because
the common-run flags and the C1 - C0 floor need it:

    python3 -m experiments.convnext_train --size tiny --opt adamw --track all --track_head \
      --spec spec_{A|B|C}.npz --snapshots --resume_every 2500 --seed 1 --out results/rare_{A|B|C}1

Step 4's go/no-go passed at seed 0; at 12,500 it is descriptive.

Analysis, with the frozen code:
- for A1, B1 and C1: the atlas, `flow`, `churn`, and `measure` at 25,020 and 12,500 (P2 also at 5,000). `measure` uses
  --train, --flow and the default draws (--n_draws 1000, --seed 0).
- for E1 and E1r: Phase 0 measures at 25,020 (--phase0), if missing.
- seed 0: a re-measure that adds fit_T, the nulls and the earlier steps. These secondaries come from a second compare
  run. The decision compare keeps the committed seed-0 files, and any per-class change in the primaries is reported.

Decision compare:

    python3 -m experiments.convnext_rare compare \
      --rare rare_A0/rare_measure_25020.json rare_B0/rare_measure_25020.json \
      --c0 rare_C0/rare_measure_25020.json \
      --rare1 rare_A1/rare_measure_25020.json rare_B1/rare_measure_25020.json \
      --c1 rare_C1/rare_measure_25020.json \
      --e1 E1/rare_measure_phase0_25020.json --e1r E1r/rare_measure_phase0_25020.json \
      --out rare_compare_seed1.json

With --c1, C1 is seed 1's common reference throughout. E1 and E1r enter only the floors C0 - E1 and E1r - E1. The
same command on the 12,500 files gives the secondary step.
