# The essay's claims against our evidence (status 2026-10-18)

## Scope and method

This scorecard sets 35 claims from N. Cammarata, "Neural Networks Are Already Clean" (Goodfire 2026) against the evidence
in `NEURON_DEVELOPMENT.md`, `EXACT_HISTORY.md` and `docs/data/`. Our tools are:

- **InceptionV1:** the public network with Google's exact weights and the LRN fixed, on all 50,000 ImageNet validation
  images. We split a unit's pre-activation exactly into per-channel input shares (E0, E0b, E0c) and test every mixed
  unit of mixed4c to mixed5b (E0d).
- **Our tiny ConvNeXt-SwiGLU runs:** 91,828 parameters, 10 epochs, 18.5 to 19.5% top-1, trained with exact per-group
  attribution. Every weight update is split into the writes of each class or group of training images, and the parts
  add up to the measured change to about 1e-8.
- **The essay's public microscopes:** dataset examples only, for its tiny, medium (7.06M parameters) and large (about
  28.6M) ConvNeXt-SwiGLU. Its checkpoints are not served (HTTP 403).
- **Blind audits:** Haiku raters, 2 per item, 40 to 64 items per set, plus a held-out description test. A positive
  control compares units with random directions in the same layer.
- **A toy (E8):** relatedness against anticorrelation, 160 runs.

What we cannot test:

- language models (Qwen3-32B, Kimi K3, GPT-2 XL);
- any retrained InceptionV1;
- SAEs or feature visualization;
- any medium-scale training of our own.

The queued medium runs never started. Both compute sandboxes were reclaimed on 2026-10-18, taking with them T90 (at
about step 50,000), the E6 umbrella queue and the E4 exits run. The private results dataset has been full since
2026-10-17, so uploads fail. Statuses below include those 2026-10-18 notes (`NEURON_DEVELOPMENT.md` Sec. 8,
`EXACT_HISTORY.md` 4.1 to 4.2).

Each claim was mapped to our tests by one reader and checked by a second, skeptical reader, who recomputed the numbers
from `docs/data/`. Where the skeptic changed the verdict, the table and the claim's section use the corrected verdict
and say so. Where the skeptic corrected a number or a status, the corrected value is used.

Verdict scale:

- **supported:** our evidence agrees and the measure can decide.
- **partly supported:** some sub-claims hold and others are untested or fail.
- **not supported:** the evidence that bears on the claim leans against it.
- **measure cannot decide:** we have tests, but they do not separate the essay's reading from the alternative.
- **untested:** no test of ours bears directly on the claim.
- **pending:** a direct test was designed and run, but its result is not available.

## Symbols and terms

- **Unit.** One channel (InceptionV1) or one SwiGLU coordinate u = SiLU(g) v (ConvNeXt), where g is the gate
  pre-activation and v is the value pre-activation. A **pole** is the positive (or negative) tail of a unit's responses.
- **Random direction.** A Gaussian random combination of a layer's units. In InceptionV1 it is either pre-ReLU (before
  the nonlinearity) or post-ReLU (after it, the space the next layer reads). In ConvNeXt it is taken over the SwiGLU
  products.
- **'One job' rate.** The share of blind ratings that call a pole "one job" or "one job with related offshoots". The
  other calls are "two related jobs", "several unrelated" and "not understood". There are 2 ratings per item, so 40
  items give 80 ratings.
- **Description test.** A third agent uses only the rater's description to pick a unit's 8 held-out strong crops from
  16. Chance is 4 of 8, and 6 or more correct has p = 0.066 by chance.
- **Strict count.** Ratings with the 'one job' call and 6 or more correct on the description test.
- **Wilson** gives a 95% score interval. **Fisher p** is a two-sided Fisher exact test. Unless stated otherwise, it
  treats the two ratings of an item as independent, so intervals and p values on ratings are somewhat too narrow.
- **R16.** An automatic class rule (top 16 images, WordNet plus DINOv2, groups of at least 3). It gives the share of
  directions called class-mixed.
- **E0d rule.** A unit is mixed if its top 100 images split into two WordNet cut-7 clusters of at least 15 images each.
  Each mixed unit gets a verdict of shared, separate or undetermined inputs, measured against 32 other units of the same
  branch (the **reader baseline**).
- **Mantel r.** The correlation between class-pair distance (WordNet or DINOv2) and the gap between the two classes'
  entry times into a unit. **% of planted** is r as a fraction of the r a simulated strict branch order would give on
  the same entry times.
- **T2.** The slope of entry time on distance from the unit's final strongest class (outward growth).
- **T5.** The share of entering classes that stay in the pole when their own writes are removed.
- **Exits primaries (X1 to X5), Holm-corrected over five:**
  - **F1:** the percentile of the leader unit's capture of a class's own writes within its stage (0.5 = median unit).
  - **F2 (X4):** the rank-regression coefficient of a unit's capture of class k's own writes on the part of its on-ness
    that k's relatives wrote, given the on-ness z and z^2. n = 56,000 class x interval x stage rows.
  - **X5:** own-gain persistence across 2,500-step intervals, against a within-row permutation null.
- **Rare-class measures (seed 0):**
  - **R_own:** relatedness of the units that carry a class's own writes. **R_flow** and **R_mis** are the same measure
    for the comparison sets, and Delta_flow = R_own - R_flow.
  - **lambda_3:** the stage-3 share of a class's own effect within the unit maps. **head_share:** the share in the
    classifier head.
  - **auroc_far:** AUROC of the class logit against far classes on held-out images.
  - **fit_T:** top-1 on the class's own training images.
  - **P1, F, O, PI:** the feeding test. F is the relatives' effect on the carriers, O is the own effect, and PI is
    their index.
  - **C0:** the run in which all classes are common.
- **E8.** rho is the embedding cosine between related features, m is the bottleneck width and H is the number of
  hidden units. An **excess** is a pole-sharing rate minus the rate with shuffled pole memberships. **DiD** is a
  difference in differences.
- **Runs:**

  | run | what it is | top-1 |
  |---|---|---|
  | E1 | tiny, AdamW, seed 0, 25,020 steps | 18.52% |
  | E1r | seed 1 | 19.51% |
  | e1f_a, e1f_b | clean reruns of E1 after the loader fix | 18.60%, 18.61% |
  | E4 | Muon | 19.07% |
  | E7 | last block crushed from 64 to 13 units, no skip | 12.47% |
  | width 16 | stage-1 residual width 16 instead of 8 | 21.07% |
  | T90 | 90-epoch schedule; only its 10-epoch checkpoint survives | 15.76% at 10 epochs |
  | essay tiny | the essay's 90-epoch tiny checkpoint | 13.8% |

## Scorecard

Verdict counts (35 claims):

| verdict | count | claims |
|---|---|---|
| supported | 1 | C08 |
| partly supported | 9 | C02, C03, C04, C07, C09, C11, C13, C14, C34 |
| not supported | 3 | C10, C27, C29 |
| measure cannot decide | 6 | C01, C06, C20, C22, C23, C25 |
| untested | 15 | C05, C12, C15, C16, C17, C18, C19, C21, C26, C28, C30, C31, C32, C33, C35 |
| pending | 1 | C24 |

The skeptic changed 6 verdicts:

- C08: partly supported -> supported
- C14: measure cannot decide -> partly supported
- C16: pending -> untested
- C27: partly supported -> not supported
- C29: pending -> not supported (weakly against)
- C31: partly supported -> untested

| id | claim (short) | verdict | confidence | key evidence |
|---|---|---|---|---|
| C01 | What remains of polysemanticity is mostly recursive generalization (RG): one principle, not two | measure cannot decide | medium | Raters give 'one job with related offshoots' to random directions as often as to units (mixed5b 48/80 vs 42/80, p = 0.43); growth co-timing is only 5-7% of planted in 6 runs |
| C02 | InceptionV1's late layers were an architecture plus tools problem, not superposition; ConvNeXt is 99-100% legible throughout | partly supported | medium | Last layer 40/40 vs 21/40 neurons 'one job' (p = 2.3e-7), but InceptionV1 units read like random directions (46/80 vs 54/80, p = 0.25) and the medium mid layer (48/80) is below crop-matched mixed4c (69/80, p = 3.0e-4) |
| C03 | Separable steps: 5% -> 50% -> 70% -> 99% | partly supported | medium | Endpoints reproduce (mixed5b 50/80, medium 80/80); 5% and 70% steps untested; plain 'one job' without offshoots on mixed5b is 2/80 |
| C04 | Legible at every scale; less room means coarser, never folded | partly supported | medium | Essay tiny 116/128 and medium 80/80 'one job'; R16 mixed 0.34 vs 0.077; in E1, random directions beat units (61/64 vs 76/128, p = 2.5e-8); large model unaudited |
| C05 | The rapeseed neuron grows locally along facets; ConvNeXt shows less of this | untested | medium | No InceptionV1 training; tiny ConvNeXt shows no outward growth (T2 4.6%, CI -3% to +11%); exit events fail their validity gate (25-27% replicate) |
| C06 | Locality of groupings and of growth separates generalization from superposition | measure cannot decide | medium | E8: anticorrelation packs antipodally (7/8 ReLU cells), never co-positively (all 8 SwiGLU cells negative); F2 changes sign at step 2,500 |
| C07 | Every branch traces to one origin and grows outward in order | partly supported (narrow) | low | Co-timing in 6/6 runs at 5-7% of planted; order lean 7-24% of planted, nominally significant in about 3 of 6; outward growth only in E7 early blocks (11%) |
| C08 | 'Superposition' means arbitrary packing; the Toy Models count sense is set aside | supported (changed) | high for the reading | Stated in four places in the essay; in E8 the antipodal excess is largest with no feature excess (m = H = 32: +0.10, +0.28) |
| C09 | 4e:55 is one V-shape detector; 'polysemantic' is one pattern named by a list of nouns | partly supported | low | Top input is fox faces (4d:493, 0.311); no flame or fireboat channel in the top 6; cars enter through a car chain (4c:447 -> 4d:352); first vehicle image is 56 of 50,000 |
| C10 | The Zoom In car circuit is shared pose; 100+ pose neurons; nothing reads the combination | not supported | medium | 241/295 vehicle-vs-dog input cosine 0.08/0.12 against a reader baseline of 0.18/0.17 (excess CIs exclude 0); their shared input 4c:41 is rated 'several unrelated' in 6/6 ratings |
| C11 | Free feature visualization falls into a dog basin; guided renders show cars | partly supported | low | No feature visualization run; the reader split holds (167: 94/100 vehicles, 417: 98/100); dogs are half of 241/295's natural top 100 (52, 49) |
| C12 | The pose neuron exists in DINOv2 and the large and medium ConvNeXts; only a rough version in the smallest | untested | high | No pose labels and no DINOv2 unit search; random directions read as side-profile cars plus dogs (3 cases) more often than units (1) |
| C13 | Last-layer neurons pair a pattern with its sibling on the negative pole, which acts as a gate | partly supported | low to moderate | Large model (post hoc): 953/1,774 distinct negative poles hold a relative (0.54 vs null 0.021); the sibling is the larger pole in 483/571; 'gate' untested |
| C14 | The negative side is shaped by learning and is almost as informative as the positive side | partly supported (changed) | medium | Mechanism is an identity; negative pole rated 'one job' at least as often as the positive in 6/7 sets (mixed5b 69 vs 50 of 80, p = 0.001); no informativeness test |
| C15 | mixed3b crushed 480 -> 10 gives coarser, not mixed, neurons; an SAE finds no curves | untested | high | No InceptionV1 retrain and no SAE; the tiny analogue E7 has 8/13 units mixed (0.62) vs 0.55 for random directions |
| C16 | Medium last layer crushed to 13 broadens its units; the small model does the same | untested (changed) | high | Medium crush never started; in E7 the units lose E1's advantage over random directions (E1 0.36 vs 0.50; E7 0.62 vs 0.55) |
| C17 | Stream narrowed 4x and 8x: messier, not interleaved; what leaks in is related | untested | high | N4/N8 never ran; at width 16 the write collapse disappears (rank 11.6-11.9 of 16) but the stage-1 reversal persists |
| C18 | SAEs add no building blocks; the rotation search lands back on the neurons | untested | high | No SAE code; rotation run lost; strict audit count favours random directions (E1 45/128 vs 39/64, p = 0.0011) |
| C19 | Weak activations ablate almost for free; the cost sits in early 'dial' layers; flips are near-ties | untested | high | No ablation run; the essay's own data: p95 costs 7.7 pp in mixed5b, and the tiny ConvNeXt loses 4.62 pp at p75 network-wide |
| C20 | Unrelated-looking weak activations trace to related circuits | measure cannot decide | low | One partial trace (E0c): 241 writes 6-40% of its readers' responses on its dog images; the rest is unattributed; no null |
| C21 | Rare classes are consolidated; weak-activation ablation singles out no class; adversarial routes are normal | untested | high | S7 has n = 0; post hoc top-carrier share isolated 0.211 vs dense 0.212 (n = 56 each, p = 0.90); no attack code |
| C22 | Rare items ride related machinery: carriers are related, not quiet units | measure cannot decide | medium | R_own 0.0143 vs chance 0.010, but the same for common classes; own writes lean to busy units (111/112); controls failed |
| C23 | Memorization sits next to the output | measure cannot decide | medium to high | Stage 3 + head holds 0.948 of the rare own effect, but 0.932 for common classes; noise labels have head share >= 0.97 in 8/8, though only 2 fit |
| C24 | Umbrella -> bee in 9 steps, through bee neurons in the last 3 blocks | pending | high | E6 ran about 140 jobs and its outputs were lost 2026-10-18; E1 recognizes 0/50 umbrella images, so the primary arm is jack-o'-lantern -> bee |
| C25 | Packing only in structureless memories; a noise-trained model reuses known textures | measure cannot decide | medium | E8 SwiGLU antipodal excess about 0 at rho 0 (+0.04/-0.03/+0.05/-0.00); noise control fit 2/8 labels and learned them as one shared route |
| C26 | Coarse-to-fine from the first steps, with no memorize-first phase | untested | high | No one- or ten-photo runs; E1 top-1 0.26% at step 100 (n = 10,000, p = 1.7e-5), but same-seed A0 gives 0.13%, not above chance |
| C27 | Circuit racing: the closest unit captures related error first | not supported (changed) | low to moderate | F2 +0.033/+0.039 before step 2,500, -0.135/-0.147 after 10,000 (E1/E1r); leader capture F1 0.62/0.66 is generic |
| C28 | Recognition is distributed first; consolidation follows, for buildability | untested | medium | No network-level readout over training; the essay's own notes: readout 0.855-0.874 at random init, precursor ablation null |
| C29 | Organization strengthens over training and with capability | not supported, weakly against (changed) | low to medium | Stage-3 F2 +0.012 -> -0.154 -> -0.299 (E1); Muon matches final top-1 (19.07%) with fewer selective units (best car 0.889 vs 0.916) |
| C30 | Offshoots are rarer and less causal; no sleeper agents; no ten-job neurons | untested | medium | No core-vs-offshoot causal test; a late unit's 20 most-helped classes overlap its top 20 only 0.07-0.29 |
| C31 | Toy: related inputs grow onto shared neurons; higher frequency gives PCA-like codes | untested (changed) | moderate | E8 uses embedding similarity, not co-firing; ReLU related co-occurring pairs at chance in 4/4 cells; no density sweep |
| C32 | Qwen3-32B: about 90% one job, 7% two related jobs, 2-4% unexplained | untested | high | No language-model work; in vision the lenient 'one job' call goes to 61/64 of E1's random directions |
| C33 | SwiGLU value is broad and gate specific; a double-negative echo tops about 4% of LM neurons | untested | high | No breadth analysis; the double-negative lobe holds 49-57% (E1) of stage-1 unit-class pairs and 0-15% in late blocks |
| C34 | Explanations can be scored falsifiably: 164 won, 15 tied, 4 lost of 183 | partly supported | low | Our blind test can lose (E1 13/64 units score 4 or less of 8) but scores pre-ReLU random directions like units (5.71 vs 5.60 of 8) |
| C35 | mixed5a:719 is a pinched shape plus co-occurring objects, through shared inputs | untested | high | mixed5a never measured; mixed4e has swimwear-plus-snorkel units read through shared inputs (4e:727, cosine excess 0.158) |

## Claim by claim

### C01. What remains of polysemanticity is mostly recursive generalization (central)

> "Our current belief is that it's most of what's left of neuron polysemanticity once the microscope, the architecture
> and scale are accounted for, not a niche form of it."

**Our tests:**

- the audits (pilot, tiny, T90 at 10 epochs, mid layer, InceptionV1 control);
- E0, E0b and E0d;
- E8;
- growth tests v4 on E1, E1r, e1f_a, e1f_b, E4 and E7;
- the E7 crush;
- rare classes, seed 0;
- exits on E1 and E1r.

**Numbers:**

- **Audit control.** InceptionV1 units against pre-ReLU random directions, 'one job' rate:
  - mixed5b 46/80 vs 54/80 (Fisher p = 0.25); mixed4c 63/80 vs 59/80 (p = 0.58).
  - The essay's RG category, 'one job with related offshoots', alone: mixed5b 42/80 vs 48/80 (p = 0.43); mixed4c
    38/80 vs 38/80 (p = 1.0).
  - Our tiny models: E1 units 76/128 vs random 61/64 (p = 2.5e-8); T90 at 10 epochs 90/128 vs 60/64 (p = 0.0002).
- **E0d.** Over the mixed units of mixed4c / 4d / 4e / 5b (n = 236 / 291 / 494 / 660):

  | | 4c | 4d | 4e | 5b |
  |---|---|---|---|---|
  | shared inputs | 23% | 16% | 23% | 1.4% |
  | separate inputs | 0.8% | 1.7% | 0.4% | 2.9% |
  | undetermined | 76% | 82% | 76% | 96% |

- **Growth, late blocks.**
  - Co-timing of near classes: WordNet Mantel r 0.021 to 0.037 (z 4.0 to 7.7) across the 6 runs, 5 to 7% of planted.
  - Outward slope T2: E1 4.6% of planted (95% CI -3% to +11%), E1r 1.9%, E7 1.5%.
  - E1's early order result (33-43% of planted) did not replicate (9%, 8% and 2% in the other runs).
- **Exits (E1 / E1r).** F2 is +0.033 / +0.039 before step 2,500 and -0.135 / -0.147 after step 10,000; stage 3 after
  10,000 gives -0.30 / -0.31.
- **E8.** Unrelated anticorrelated features sit on opposite ReLU poles above chance in 7 of 8 cells. At rho 0.6, 75-83%
  of spillover entries come from unrelated members.
- **Rare classes (n = 112).** R_own 0.0143 against a chance level of 0.010, the same as for common classes; no primary
  is validated.

**Verdict:** measure cannot decide (medium). The skeptic kept the verdict and added which way the evidence leans:

- The static measures (audits, E0d, class-mixing rules) cannot separate RG from arbitrary mixtures.
- The dynamic measures in the tiny model give a weak RG signature: co-timing of near classes, no outward growth, and,
  after early training, relatives' on-ness predicting less capture. The F2 result is replicated across two seeds.
- Nothing positively supports "mostly".

**Caveats:**

- The essay's "mostly" rests on a few neurons watched as they grow and on static audits (Qwen 90/7/2-4%). Our control
  shows that such audits do not separate RG from mixtures.
- The essay authors' own early-training notes say the trajectories "do not yet demonstrate a causal
  generalist-to-specialist consolidation mechanism".
- Our tests are vision only, mostly in a tiny 10-epoch model whose units read worse than random directions. That may
  not be the regime the claim covers.
- The growth tests are pooled, use 16 checkpoints, and measure relatedness with WordNet and DINOv2, both of which miss
  co-occurring facets.
- E0d is blind to a variable built inside the unit's own layer.
- The category tests on the audits are post hoc.

**Next test:** take a run whose units beat random directions in the blind audit, which needs the restarted T90 or a
medium run. For each offshoot cluster, use the exact record to measure whether its entry was written by the unit's core
classes and their relatives or by unrelated classes. Compare against size-matched random-group nulls and a planted-order
null. "Mostly RG" predicts that most offshoots enter through related-core spillover, well above chance.

### C02. InceptionV1's problem was architecture plus tools, not superposition (central)

> "One main result is that superposition was the wrong explanation for InceptionV1. Architectural problems made later
> layers overflow. Weak interpretability tools amplified these problems ... 99-100% of neurons in that last layer are
> easily interpretable and the model is broadly legible the whole way through."

**Our tests:**

- the blind audit pilot (essay medium stage3.block2 vs InceptionV1 mixed5b);
- the InceptionV1 units-vs-random-directions control;
- the tiny and T90 audits;
- the mid-layer audit with a crop-zoom check;
- E0d;
- E7.

**Numbers:**

- **Last layer.**
  - Medium: 80/80 ratings 'one job' (Wilson 0.95-1); 40/40 neurons by both raters, a neuron-level lower bound of 0.91;
    description test 7.45 of 8.
  - mixed5b: 50/80 (0.52-0.72); 21/40 neurons by both raters; description test 5.43.
  - Our renderer with the LRN fixed gives mixed5b 46/80.
  - R16 mixed share: 0.077 (medium) against 0.69 (mixed5b).
- **"Not superposition".** InceptionV1 units read like pre-ReLU random directions (numbers in C01). In mixed5b, E0d
  leaves 96% of the 660 mixed units undetermined (separate 2.9%, shared 1.4%).
- **Same architecture at tiny size.**
  - E1 (10 epochs): 0.59 'one job' (76/128), with random directions at 0.95.
  - The essay's tiny model (90 epochs): 0.91 (116/128).
  - The two differ in length, in the unpublished recipe and possibly in width, so the cause is not decided.
- **Middle layer.**
  - Medium stage2.block4: 48/80 (0.60), 29/80 'not understood'. That ties mixed5b on our protocol (48 vs 50, p = 0.87;
    strict count 24 vs 24).
  - It is below InceptionV1 mixed4c zoomed to the same crop window: 69/80, p = 3.0e-4.
- **The essay's own breakdown.** About 5% -> 50% (tools) -> 70% (regularization) -> 99% (ConvNeXt). On these numbers at
  most about 29 of the 94 points belong to architecture.

**Verdict:** partly supported (medium). The skeptic kept the label and corrected its basis:

- **Supported:** the last-layer contrast. "99-100%" is not confirmed, since 40/40 cannot be told apart from about 93%.
- **Undecidable:** "not superposition".
- **Untested:** the causal role of the residual stream and LayerNorm.
- **In tension:** "broadly legible the whole way through", at the one middle block audited.

**Caveats:**

- The medium recipe (optimizer, epochs) is unpublished. The mapper's "mixup, AdamW" was wrong.
- In E1's last block the units are less class-mixed than random directions (E0d rule 0.36 vs 0.50; R16 0.42 vs 0.66).
  So 'non-privileged' holds only for rater coherence.
- The audits use Haiku raters, 40 neurons per set and dataset examples only, with no feature visualization.
- The medium model gets no random-direction control, because its weights are not served.

**Next test:** train InceptionV1 and the parameter-matched medium ConvNeXt (7.06M) with one recipe, plus two arms:
InceptionV1 with a residual stream and LayerNorm added, and ConvNeXt with them removed. Use 2 or more seeds. Run the same
blind audit with pre- and post-ReLU random directions and the description test, plus E0d.

### C03. Separable steps from 5% to 99% (central)

> "I put it around 5% with tools of that era ... more like 50% on the same network ... Retraining the same architecture
> with more reasonable modern regularization it's more like 70%. In the ConvNeXt of the same size, it's about 99%."

**Our tests:** the pilot, the InceptionV1 control, the tiny, T90 and mid-layer audits, E0d, and the LRN provenance check.

**Numbers:**

- **The 50% step.**
  - mixed5b: 50/80 lenient ratings (0.52-0.72), but only 24/80 on the strict count (0.21-0.41).
  - By neuron: 21/40 with both raters (0.375-0.671), and 7/40 strict with both raters (post hoc).
- **The 99% step.** Medium 40/40 neurons; Fisher on neurons, 40/40 vs 21/40, p = 2.3e-7.
- **Post hoc, from the pilot ratings.** Plain 'one job' without offshoots is 2/80 for mixed5b (0.025; 0.007-0.087)
  against 36/80 for medium. The essay's 5% and 50% can both come out of one tools-free rating, depending only on whether
  offshoots count.
- **Neuron-level controls, which count each item once:**
  - E1 units vs random directions: 32/64 vs 30/32 (p = 1.1e-5).
  - mixed5b units vs pre-ReLU random directions: 20/40 vs 25/40 (p = 0.37).
- The 5% and 70% steps have no test.

**Verdict:** partly supported (medium), unchanged by the skeptic. The endpoints reproduce under blind rating. That the
steps are separable is asserted, not shown.

**Caveats:**

- The essay gives no protocol, n or interval for any of the four numbers.
- The T90 run that would have tested training length was lost before its 30-epoch checkpoint.
- The 70% retrain's recipe and length are not published, so step 3 to 4 is not shown to be recipe-matched. Nor is it
  known to differ.
- Haiku raters are lenient on mixed5b: 6/40 neurons are at or below chance on the description test.

**Next test:** the matched-recipe design of C02. Add the original 2014 mixed5b on the same protocol, with the description
test as the rater-independent measure.

### C04. Legible at every scale; less room means coarser, not folded (central)

> "With less room, the network doesn't fold unrelated things together; its neurons just cover coarser-grained concepts,
> like one car neuron covering many kinds of car."

**Our tests:**

- the pilot (essay medium);
- the tiny audit (the essay's 90-epoch tiny model, E1 units, E1 random directions), with a Sonnet check and R16;
- T90 at 10 epochs, the InceptionV1 control and the mid-layer audit;
- E7;
- the E1 car map.

**Numbers:**

- **Legibility on the essay's checkpoints.**
  - Tiny (all 64 units): 116/128 (0.91; 0.84-0.95), description test 6.83.
  - Medium: 80/80, description test 7.45.
  - 'Several unrelated': 0/128 and 0/80.
  - Narrowing with size (R16 mixed): 0.34 (tiny) vs 0.077 (medium).
- **The essay's tiny vehicle unit is reproduced blind.** Channel 22 reads as "boxy land vehicles" against "watercraft and
  aircraft".
- **Coarse or folded: the measures cannot decide.**
  - Random directions read as well as units (InceptionV1) or better (E1 0.95 vs 0.59; T90 0.94 vs 0.70).
  - R16 calls E1's random directions the most class-mixed set (0.66), yet raters find them the most coherent.
- **E7:** 8/13 units mixed (0.62; 0.36-0.82), against 0.55 for random dense directions and 0.61 for random pairs.

**Verdict:** partly supported (medium), unchanged by the skeptic.

- Supported: tiny and medium are legible, and tiny-to-medium narrowing is shown by R16.
- Cannot decide: "never folds unrelated things".
- Untested: the large model and the "specific cars" differentiation.

**Caveats:**

- In the tiny audit the raters had no class labels and no activation values: the cells file was reported missing in
  about 301 of 320 Haiku ratings. The tiny-vs-medium contrast (p = 0.004) therefore compares two protocols.
- The tiny model's kappa is 0.27; 0.88 is raw agreement.
- Random directions do get 'several unrelated' outside E1: InceptionV1 pre-ReLU 16/80 (mixed5b) and 6/80 (mixed4c).
- At the same size, our 10-epoch recipe gives units below random directions. Legibility at 92k parameters depends on
  recipe and length, not on scale alone.

**Next test:** on all three essay microscopes and on matched-recipe runs of our own, with random directions at each
size, measure each unit's class spread and whether a tiny unit's classes nest into related medium and large units.
Coarse units nest into related units; folded units split into unrelated ones.

### C05. The rapeseed neuron grows along facets; ConvNeXt shows less of it (central)

> "When we retrained the model and watched a neuron form, it started on yellow flowers ... Then it grew along each facet
> of those photos ... This is true of every neuron I looked at. The improved architecture shows the same pattern, but
> much less of it."

**Our tests:** growth tests v4 on six runs, exits on E1 and E1r, E8 and E0d (static). We trained no InceptionV1.

**Numbers:**

- The essay's evidence is n = 1 neuron per model (InceptionV1 338, ConvNeXt 399), chosen as "the least legible I can
  find".
- **Tiny ConvNeXt.**
  - Late co-timing replicates in 6/6 runs (5-7% of planted).
  - The visual nearest-earlier order statistic leans toward order in all 5 runs measured (7-23% of planted). It is
    significant only in E1 (p = 0.01), with E7 borderline.
  - No outward growth: T2 from -2.0% to 4.6% of planted.
- **Exits.**
  - The events fail the pre-registered validity gate in both seeds: 0.27 (E1) and 0.25 (E1r) of anchored exits
    replicate, against 0.5 required.
  - The anchored share of exits rises from 20% before step 2,500 to 66% after step 10,000. That resembles "drops
    offshoots, keeps its core", but part of it is mechanical.
- **E8.** 36-76% of pole entries come by spillover.

**Verdict:** untested (medium; higher for the untested part). The skeptic kept it, with this wording: untested for
InceptionV1 and for the InceptionV1-vs-ConvNeXt contrast; the ConvNeXt half was probed only indirectly, in a tiny model.

**Caveats:**

- "Every neuron I looked at" gives no count.
- "Facet" via co-occurrence makes almost any drift count as local.
- T2 anchors on the final strongest class, so it cannot see a seed that is lost or an offshoot that is later dropped,
  which is exactly the rapeseed case.
- Our runs are tiny (92k parameters).

**Next test:** train InceptionV1 and the medium ConvNeXt from scratch on the essay's epoch grid (0.7 to 12) with exact
attribution and 2 or more seeds each. Measure per unit: seed retention, the outward slope from the early seed cluster
using a co-occurrence distance, and the rate of dropped offshoots with half-split replication.

### C06. Locality separates generalization from superposition (methodological)

> "These have opposite predictions: generalization should be local, superposition should be non-local, since it would
> want very different things to be near each other."

**Our tests:** E8, E0b and E0d, growth tests (T2, T4), E7, rare-class P3, exits F2, and the audits.

**Numbers:**

- **E8 toy, unrelated anticorrelated pairs.**
  - In ReLU they land on opposite poles: excess +0.10 / -0.01 / +0.07 / +0.10 at rho 0 (SE 0.03).
  - They do not share a positive pole: the co-positive excess runs from -0.175 to +0.076 over 16 cells, and all 8
    SwiGLU cells are negative.
- **E8 toy, related pairs.** Relatedness puts them on a shared positive pole: DiD up to +0.20 (SwiGLU, SE 0.04) and
  +0.24 (ReLU, SE 0.03). Yet 75-83% of spillover growth comes from unrelated members.
- **E0d.** Shared inputs occur mostly for visually near pairs (near 36 / 26 / 38 / 3%, far 10 / 6 / 9 / 0% in mixed4c to
  5b).
- **'Local' depends on the distance used.** E1's random directions are the most WordNet-mixed set (R16 0.66) and the most
  visually coherent.
- **Audit 'several unrelated', units vs pre-ReLU random directions.** mixed5b 13/80 vs 16/80 (p = 0.68); mixed4c 4/80 vs
  6/80 (p = 0.75). The essay's medium model gets 0/80 against 11/80 for mixed5b (p = 0.0007), but has no random control.
- **Exits F2.** Positive before step 2,500 and negative after it, in both seeds.

**Verdict:** measure cannot decide (medium), unchanged. The skeptic sharpened it: the "opposite predictions" hold only
when superposition means co-positive packing, the essay's own definition. In the toy, packing goes antipodal (ReLU) or is
small (SwiGLU).

**Caveats:**

- Correction: the negative pole is passed on in SwiGLU, since u = SiLU(g) v is signed. Only ReLU cuts it.
- Sibling classes are both related and anticorrelated (Sec. 1 gap 2).
- The essay's LM test (179 of 183 neurons silent on distant texts) is outside our scope.

**Next test:** calibrate the measure first. Plant co-positive packing of far-apart classes in a crushed tiny model, for
example by merging targets, and score pole locality with a pre-fixed WordNet and DINOv2 metric on both poles. Score the
planted model, E1 and E7 units, random directions and the medium model. The measure counts only if it flags the planted
packing.

### C07. Branches trace to one origin and grow out in order (central)

> "If you visualize a branch well, everything on it should trace back to the same local origin ... and it should grow
> out from there over training in a particular order."

**Our tests:** T1 to T5 on six runs, E8, E0d, exits, and the audit category tallies.

**Numbers:**

- **Co-timing (late, WordNet).** E1 r = 0.029 (z = 5.7), E1r 0.025, e1f_a 0.021, e1f_b 0.037, E4 0.025, E7 0.027.
- **Order beyond co-timing** (visual nearest-earlier, late, fraction of planted). E1 13-15% (p = 0.01), E1r 8-9%,
  e1f_a 23%, e1f_b 7%, E7 14%, E4 24%.
- **Outward growth.**
  - T2 late: -2.0% to 5.2% of planted.
  - The only positive result: E7 early blocks, WordNet, 11% (4-16%).
  - E4 early visual T2 is significantly inward (-7.7%, t = -2.79).
- **T5.** No strict head start: 33% of entering classes survive without their own writes, against 52% without one random
  class's writes (12/12 late blocks).
- **Audit.** Raters give 'core with related offshoots' to units and to random directions alike (mixed5b 42 vs 48 of 80;
  mixed4c 38 vs 38).

**Verdict:** partly supported (low), unchanged by the skeptic but to be read narrowly. Only the non-specific part
(co-timing) replicates robustly. The order specific to the claim is a small lean, and outward growth is not detected.

**Caveats:**

- The essay's own cases are untested: InceptionV1 neuron 338, the 'iy' LM neuron and the Multimodal Neurons cases.
- Class-level tests cannot see within-class facets.
- Planted orders reuse noisy entry times.
- The E8 spillover range is 36-76% across cells, not 48-76%.

**Next test:** retrain InceptionV1 with several seeds and dense early checkpoints. Run per-unit order tests anchored on
each unit's earliest stable member, with co-occurrence distances. Compare against a planted order with matched
entry-time noise.

### C08. 'Superposition' means arbitrary packing (methodological)

> "So we keep the word and say exactly what we mean: neurons learning unrelated things together because they rarely
> interfere."

**Our tests:** a reading of the essay (Sec. 1 gap 1), E8, E0d, the audits, E7, rare classes, and the stage-1 residual
geometry.

**Numbers:**

- **The essay states the narrowing four times:**
  - "even a fully monosemantic network would satisfy [the count sense], because its neurons compose";
  - "When I say superposition in this essay, I'm referring to this version";
  - the Note on Superposition;
  - the Note on Features and Definitions.
- **E8 separates the count sense from packing.**
  - Unrelated co-positive excess is at or below chance in 15 of 16 cells.
  - The antipodal excess is largest with no feature excess (m = 32, H = 32: +0.10 at rho 0 and +0.28 at rho 0.6), and a
    4x bottleneck (m = 8) does not raise it.
  - So the driver is anticorrelation, the essay's "rarely interfere", not the feature count.
- **E0d.** 'Separate' inputs (packing beyond what the images impose) are rare: 0.4-2.9% of mixed units, though 76-96%
  are undetermined.
- **Raters disagree on 'several unrelated'.** Haiku gives it to 0/128 E1 units; Sonnet to 4/20.

**Verdict:** supported (changed by the skeptic from "partly supported"). Confidence is high for the reading and medium for
its use. C08 is a claim about what the essay means, and the essay says it outright. The mapper had argued that E8 merges
the two senses, but E8 separates them.

**Caveats:**

- The subtitle and headline ("wasn't superposition") read broader than this definition.
- Whether arbitrary packing is actually absent is a separate empirical claim, which C01 and C06 leave undecided.
- The stage-1 write collapse (effective rank 1.7-6.4 of 8; 48-94% of units with a near-parallel partner) is a width-8
  property. At width 16 it disappears (rank 11.6-11.9 of 16; partners 3-8%), so it does not bear on packing.

**Next test (for the empirical half):** apply the E8 DiD design to E7's 13 crushed units on two seeds. Test whether
co-pole class pairs are related beyond random-direction and class-shuffled nulls, stratified by visual co-occurrence.

### C09. 4e:55 is one V-shape detector (central)

> "I looked at its circuit connections: V-shaped flames, V-shaped water jets from a fireboat, V-shaped fox faces. I
> looked at its negative pole: an O, round where the V is pointed."

**Our tests:** E0 on all 50,000 validation images (profile, and linearized mixed4d inputs), E0d, and the audit control.

**Numbers:**

- **Top tail.**
  - The top 50 images contain 0 cars; the first vehicle image is number 56 of 50,000.
  - Class means (n = 50 images per class): the top 9 are pointed-faced animals (papillon 243.3, Persian cat 241.5, red
    fox 232.0, ...).
  - 10 vehicle classes, interleaved with animals, hold 1-based ranks 11-36 (beach wagon 189.2 down to minibus 143.6).
    They lie between the 90th (110.9) and 99th (244.9) percentiles of per-image maxima, so the car response is real at
    class level.
- **Inputs (linearized, expanded weights).**
  - The strongest is 4d:493, a fox-face channel (0.311, 2.4 times the next).
  - No flame or fireboat channel is among the 6 strongest.
  - The vehicle channel 4d:352 (0.102) is the second-strongest reader of the car neuron 4c:447 (0.517, correlation
    0.72), so cars enter through a car chain.
  - A round-object channel, 4d:334 (+0.080), feeds the unit positively.
- **Negative pole.**
  - 34 dogs in the 50 most negative images.
  - Round muzzles and objects fit the "O" (bull mastiff -124.7, pug -107.0, golf ball -82.0), but black-and-tan dogs and
    birds rank as high.
  - The most negative inputs are 4d:345 (spaniels, -0.072), 4d:456 (rail and bus, -0.067), and two bird channels,
    4d:349 and 4d:381.
- **E0d.** 4e:55 is not mixed under our rule. Its related split (foxes vs cats) is read through shared inputs (cosine
  excess +0.145 [0.129, 0.182]).

**Verdict:** partly supported (low), unchanged by the skeptic. Fox faces are confirmed; flames and fireboats are not
among the top 6 inputs; the "O" fits only loosely. The V itself was never tested, and the general noun-list claim cannot
be decided (audit control).

**Caveats:**

- Correction: the most negative input is 4d:345, not 4d:456.
- A car-specific input chain does not rule out a V built inside the layer, which the essay's footnote allows.
- A ReLU follows every branch, so the negative pole sends nothing downstream.

**Next test:**

- Split 4e:55's response exactly into mixed4d shares on its vehicle images (from image 56 on) and on its cat and fox
  images, against the 32-reader baseline.
- Add within-class tuning for pointed vs round fronts (sports cars vs vans; Persian vs tabby).
- Reading: shared channels with shape tuning support the V; car-only channels without tuning support OR(cat, car); car-only
  channels with tuning still allow a V built inside the layer.

### C10. The Zoom In car circuit is shared pose (central)

> "Cars and dogs were being brought together because of shared pose. InceptionV1 has more than a hundred of these
> generalized pose neurons ... nothing directly reads out the combination."

**Our tests:** E0 (4c:447 and its mixed4d readers), E0b (exact path split), E0c (mixed4e readout), E0d, and the audit
ratings.

**Numbers:**

- **Car readers.**
  - 4d:167: 94 vehicles and 3 dogs in its top 100. 447 writes 88% of its vehicle response, and zeroing 447 lowers that
    response by 73%.
  - 4d:417: 98 and 0. 447 writes 54%; zeroing lowers it by 60%.
  - Four of 447's eight strongest mixed4d readers are vehicle units.
- **Mixed readers 4d:241 and 4d:295** (46/52 and 49/49 vehicles/dogs in the top 100):
  - Cosine of the mean input-share vectors, vehicle vs dog images: 0.08 and 0.12. Random halves of one group give 0.999
    / 0.987 and 0.997 / 0.993.
  - Reader baseline median: 0.18 and 0.17. Excess -0.10 and -0.06, with 95% CIs that exclude 0.
  - 447 writes 0.76 / 0 and 0.45 / 0 of the response on vehicle / dog images.
  - 295's montages show side-on cars and frontal terriers.
  - Their one shared input, 4c:41, is rated 'several unrelated' in 6 of 6 blind ratings across 3 renders.
- **Readout (E0c).** 241 writes 8% of its top mixed4e reader (4e:591, a vehicle unit). 295 writes at most 14% of
  4e:543.
- **Pose readings in the audits.** Random directions twice got a side-profile cars-plus-dogs reading from both raters.
  No sampled unit did.

**Verdict:** not supported (medium), unchanged by the skeptic. By sub-claim:

- the two car neurons: supported;
- shared pose at 241/295: not supported;
- "more than a hundred pose neurons": untested;
- "each a clean detector": cannot decide;
- "later layers keep building on them": not supported at the one layer followed;
- "nothing reads the combination": partly supported.

**Caveats:**

- Pose was never labelled.
- E0b cannot see a pose variable built inside mixed4d from class-specific channels.
- 447's pure dog reader, 4d:139, was never split.
- The audit unit called "4c:3" in the mapping is channel 4c:41.

**Next test:** label viewpoint (side, front, three-quarter, rear) on every validation image of the 32 vehicle and 118 dog
classes. Measure the pre-ReLU tuning of 241, 295, 139 and every mixed4c-5b unit against random directions, and count
units with matched cross-class tuning. Extend E0c to mixed5b and the logits.

### C11. Free feature visualization falls into a dog basin (supporting)

> "If you even lightly bias it toward the natural paths the top images take, most of them are cars, and some are both
> dogs and cars, because of shared pose."

**Our tests:** E0, E0b and E0c on natural images. No feature visualization exists in the repository.

**Numbers:**

- 4c:447 has 49 vehicles in its top 50 (32-class list) and 39 dogs in its 50 most negative images. Blind ratings call it a
  vehicle unit in 6 of 6 (held-out 7 or 8 of 8).
- For 241 and 295, dogs are half the natural top 100 (52 and 49), at the same mean response as vehicles (727 vs 710; 549
  vs 542). They are not a reward hack.
- Dog routes into the vehicle readers exist: 167's 3 dog images reach a mean response of 522, against 568 for its
  vehicles, through dog channels with 0 from 447. This makes the basin plausible but does not test it.
- The essay's four readers are not the four strongest. 4d:139 (weight 0.39; 39 dogs and 0 cars in its top 50) was never
  analysed.

**Verdict:** partly supported (low), unchanged. The partition of the readers holds on natural images. Pose is not
supported, and every feature-visualization claim (basin, no recovery, 64-step guidance) is untested.

**Caveats:**

- The essay's figure is "visualized through the car", which favours cars by construction.
- The essay's InceptionV1 microscope uses Lucent weights, whose LRN differs from Google's graph by about 2 times the
  activation scale.

**Next test:** render 167, 241, 295, 417 and about 30 low-dog late units, free and guided, with 10 or more seeds, on both
weight sets. Score every render blind with an independent classifier (dog vs vehicle, plus viewpoint), and include a
random-route guidance control.

### C12. The pose neuron appears in DINOv2 and the essay's ConvNeXts (supporting)

> "The same neuron shows up in DINOv2 and in our large and medium ConvNeXts: dogs and vehicles grouped by the way they
> face the camera."

**Our tests:** none targeted. Incidental evidence comes from the audits of the essay's tiny and medium microscopes and
from the InceptionV1 control.

**Numbers:**

- No rater description puts vehicles and dogs together on the positive pole of any sampled essay-model unit (0/64 tiny
  last block, 0/40 medium last block).
- Random directions read as side-profile vehicles plus dogs against frontal dog faces in 3 cases (two pre-ReLU, one
  post-ReLU, held-out up to 7 of 8). Units did so once, from 1 of 2 raters.
- Pose units do exist in InceptionV1. 4c:376 is a side-profile animal unit (4/4 'one job', held-out 7 to 8 of 8), but it
  covers animals only.

**Verdict:** untested (high), unchanged.

**Caveats:**

- Our medium audits sample 3.9% and 7.8% of a block.
- A car-plus-dog side-profile reading is expected from the representation space alone, so finding one does not show a
  privileged neuron.

**Next test:** build a viewpoint-labelled probe of cars and dogs. Score every unit of DINOv2 and of our medium runs (or
the essay's microscope crops) for matched car and dog pose tuning, against random directions in the same layer.

### C13. A neuron's negative pole is a sibling, acting as a gate (supporting)

> "If we look at the last layer, we see high-level patterns and what they're being distinguished from: taxis from
> ambulances, broccoli from cauliflower, and spoonbills from flamingos."

**Our tests:** a post hoc count of the modal positive class against the modal negative-pole class from the microscope crop
labels (not in the docs), E8, E0 and the audits.

**Numbers:**

- **Essay large model, stage3.block2 (2,048 units).**
  - 953 of the 1,774 units whose negative pole is a different class have a relative there: 0.54 (Wilson 0.51-0.56),
    against a shuffled null of 0.021. On strong poles it is 571 of 1,157.
  - Counting strict WordNet siblings only: 231 of 1,774 (0.13), against a null of 0.002.
  - The essay's three pairs exist as single units (230 cab/ambulance, 829 broccoli/cauliflower, 1440
    spoonbill/flamingo), each with 16/16 top crops of one class and 4/4 negative crops of the sibling.
- **The rate rises with size.** Tiny 1/64, medium 0.39 (all 1,024 units), large 0.54. Our E1 and T90 are at floor, and so
  is InceptionV1 mixed5b (0/40).
- **The microscope statistics lean against a gate.**
  - The sibling is the larger pole in 483 of 571 strong sibling units.
  - The SwiGLU gate opens more on the negative sibling (median 2.4 times), with the value's sign separating the two.
  - That fits a signed contrast code at least as well as a gate.
- **E8.** Anticorrelation alone puts features on opposite ReLU poles (7/8 cells). Relatedness pulls siblings off
  opposite poles.

**Verdict:** partly supported (low to moderate). The skeptic kept the label but changed the split behind it: the
descriptive half is supported on the large model; the "gate rather than opposite" half is untested and leans against.

**Caveats:**

- The count is ad hoc and post hoc, from crop labels.
- No random-direction control is possible on the essay's models.
- Dataset examples cannot separate a gate, a contrast code and antipodal packing (Sec. 1 gap 2).

**Next test:** on our medium run (or the large model if its weights become available), split each sibling unit's negative
output into the drop in the positive class's logit and the rise in the sibling's logit, with a clamp of u < 0 as a check,
against units with unrelated negatives and against random directions.

### C14. The negative side is learned and almost as informative (supporting)

> "The negative side of neurons matters almost as much as positives for understanding them, even in ReLU networks where
> the negative side is cut off."

**Our tests:** the dual-form identity, E8, E0, every audit's negative-pole calls, and stage-1 lobe counts.

**Numbers:**

- **Mechanism.**
  - The exact identity c_{j,t}(q) = -eta_t delta_{j,t} <h_t(x_t), h_T(q)>, summed over steps t and added to the decayed
    initial weights, gives unit j's response z_j(q) on any input q, of either sign. Here eta_t is the learning rate,
    delta_{j,t} the unit's backpropagated error and h the layer input.
  - In E8, ReLU negative poles carry learned antipodal structure (+0.10 to +0.28 in 7 of 8 cells).
- **Coherence, negative vs positive pole, same units.**
  - mixed5b: 69 vs 50 of 80 (p = 0.001); with our renderer 65 vs 46 (p = 0.002).
  - mixed4c: 74 vs 59 on whole crops (p = 0.003).
  - Medium stage2.block4: 58 vs 48.
  - Essay tiny: 118 vs 116 of 128. E1: 85 vs 76. T90: 91 vs 90.
  - Only the medium last layer is lower: 60 vs 80 of 80 (p = 5e-7). There, 9/40 units have a strongest negative crop at
    0.09 of the positive or below, but 23/40 have a negative pole larger than the positive.
- **Control.** Negative-pole calls do not separate units from pre-ReLU random directions (p = 0.84 and 0.45).

**Verdict:** partly supported (changed by the skeptic from "measure cannot decide"; medium). The mechanism half holds.
The informativeness half is undecided: coherence leans toward the essay, but no test measures what the negative side adds
to understanding a neuron.

**Caveats:**

- Raters see 4 negative crops against 24 positive ones.
- In InceptionV1 a ReLU negative pole sends exactly 0 downstream (447 on dog images), so it is information about the
  weights, not a signal.
- In SwiGLU the claim's premise ("never gets gradient on its own") is false.

**Next test:** have raters write descriptions with and without the negative side, for units and matched random
directions. A blind agent then predicts held-out negative crops and pairwise mid-range orderings. Also split E8's ReLU
negative poles into their initialization and training parts.

### C15. mixed3b crushed to 10 channels gives coarser, not mixed, neurons (central)

> "The neurons stay interpretable and generalized, but coarser: one 'curves & loops' neuron instead of ten curve
> detectors."

**Our tests:** none direct. E7 is a ConvNeXt analogue. There is no mixed3b data and no SAE code anywhere.

**Numbers:**

- E7 (last block 64 -> 13 units, no skip; top-1 12.47% vs E1's 18.52%):
  - mixed share 0.62 (8/13; Wilson 0.36-0.82) against 0.55 for random dense directions and 0.61 for random pairs;
  - E1's last block: 0.36 / 0.50 / 0.36.
- Class purity cannot tell coarse from packed: the essay's tiny units are R16-mixed 0.34 yet rated 0.91 'one job'.
- The essay's crush costs 11.73 points of top-1 (58.35% vs 70.08%), which the section does not discuss.

**Verdict:** untested (high), unchanged.

**Caveats:**

- Curves and textures are dense features. Toy Models also predicts coarse, PCA-like codes for them (Sec. 1 gap 3).
- An SAE can only find what the crushed layer still encodes, and the essay's negative SAE result has no positive
  control.
- A unit's top crops sample the regime where even packed units look monosemantic.

**Next test:** retrain InceptionV1 crushed and uncrushed under one recipe. On the crushed layer, run three checks:

- a blind audit against 10-dimensional pre- and post-ReLU random directions;
- synthetic curve tuning;
- an SAE with a planted-superposition positive control.

### C16. The medium last layer crushed to 13 units broadens (central)

> "So we can take our medium size ConvNext, crush the last layer, stop it from routing around the residual stream, and
> see what it learns. We see the same pattern of it broadening."

**Our tests:** E7, the tiny audit and R16, the InceptionV1 control, and the essay's own crush figure. The medium crush,
medium_crush13, never ran.

**Numbers:**

- **E7.**
  - Crushed units are as class-mixed as random directions: 0.62 (8/13) vs 0.55.
  - E1's uncrushed units were less mixed than random on both rules (E0d 0.36 vs 0.50; R16 0.42 vs 0.66), so the crush
    removed that advantage.
  - Fisher, E7 units vs E1 units: p = 0.12.
- **The small-model half.** The essay's tiny checkpoint reads coarse but one job (0.91; R16 0.34 vs 0.077 for medium),
  with no random-direction control.
- **The essay's crush figure.**
  - 5 of 26 poles name two unrelated themes (0.19; 0.09-0.38).
  - 4 of 13 positive poles are a single class.
  - Its captions say deletion hits unrelated classes and that "No single shared visual mechanism has yet been
    established".

**Verdict:** untested (changed by the skeptic from "pending"; high). The medium queue waited for E6 on sandbox G, which
was reclaimed on 2026-10-18, so no medium crush is running or queued.

**Caveats:**

- E7 is one seed, a 5x crush (64 -> 13) rather than 79x (1,024 -> 13), and was trained before the loader fix.
- The E0d rule calls a broadened unit mixed more often, so it cannot tell coarse from mixed.

**Next test:** run medium_crush13 with 3 seeds (39 units, 78 poles). Let N be the share of poles with a split that is
far in both WordNet and DINOv2, and N_rot the same share for 64 random 13-dimensional directions per seed.

- The essay is supported if N <= 0.20 and N/N_rot <= 0.6.
- Packing is indicated if N >= 0.35 and N/N_rot >= 0.85.

Add a blind audit against random directions.

### C17. A narrowed residual stream is messier but not interleaved (supporting)

> "Accuracy fell from 72% to 60% and 39%, but compared with a normal-width model the crushed stream's map looks quite
> similar, just a bit messier. Nothing looks cleverly interleaved."

**Our tests:** none direct. The N4 and N8 runs are queued but never ran. Indirect evidence comes from the stage-1
geometry, the width-16 test, X4, E7, E8 and the audits.

**Numbers:**

- **Stage 1 at width 8 (E1, E1r, E4).** 48-94% of units have a partner at |cos| > 0.9; the effective rank of the write
  directions is 1.7-6.4 of 8; the writes run against the incoming stream component (s_in -0.26 to -0.51).
- **Width 16.** Rank 11.6-11.9 of 16 and partner share 0.03-0.08, but the cancellation stays in the reversed blocks
  (s_in -0.28 / -0.30).
- **X4.** In 12/12 stage-1 blocks across 4 runs, the sign of the weights-only classifier term matches whether the block
  is reversed (Spearman 0.77-0.98 against the head reader, unit by unit).
- **E8.** 75-83% of spillover through the bottleneck comes from unrelated members.

**Verdict:** untested (high), unchanged.

**Caveats:**

- Narrowing the stream also cuts parameters 4 to 8 times (7.06M to about 1.7M and 0.86M), so width and size are
  confounded.
- The essay's dark-bird example cannot separate coarsening from packing.
- 8x narrower than our inferred widths gives 48 dimensions, not the essay's 46.

**Next test:** run M10, N4 and N8. For each narrowed unit, measure how related its newly entering classes are, against
random directions and a planted-packing control. Add a quantitative stream-map metric. Units need a matching method,
since units cannot be matched across runs.

### C18. SAEs add no building blocks; the rotation search lands back on the neurons (supporting)

> "On the last layer, 88% of its activity is near-copies of single neurons ... if you search the rotations of a layer
> ... you land back on the neurons, in every block."

**Our tests:** there is no SAE code. `experiments/basis_rotation.py` implements the rotation search with a
rotated-activation control, but its run on molab-D was lost and no output exists. Indirect evidence comes from the audits,
R16, E7 and the rare classes.

**Numbers:**

- **Audits, strict count, units vs random SwiGLU-product directions.** E1 45/128 vs 39/64 (p = 0.0011); T90 48/128 vs
  40/64 (p = 0.0013). Random directions win.
- **InceptionV1.** Post-ReLU directions do worse than units (strict 17 vs 32, p = 0.016; 29 vs 49, p = 0.0025).
- **Class-purity rules lean the other way.** R16: E1 units 0.42 vs random 0.66 (p about 0.05).
- **Rare classes** are learned as directions (auroc_far 0.828 at 10 images and 0.953 at 100) with top-1 0, so a top-1
  ablation would not register their loss.

**Verdict:** untested (high), unchanged. The rotation part is "no result available", not "pending".

**Caveats:**

- With an elementwise nonlinearity the neuron basis is privileged, so landing on the neurons under a concentration
  objective is expected under superposition too.
- The essay gives no SAE size, sparsity or 'near-copy' threshold.

**Next test:** train a TopK SAE on our medium last block, with the same SAE on rotated activations as a positive control.
Ablate unmatched atoms and score top-1, logit KL and per-class AUROC. Rerun the rotation search per block against its
rotated control.

### C19. Weak activations ablate almost for free (supporting)

> "What we see is that you can ablate almost all the weak activations of the network for almost no cost."

**Our tests:** none. InceptionV1 top-1 was never scored. Context comes from rare classes, the stage-1 ablations, X4 and
the essay's own figure data.

**Numbers:**

- **The essay's own data, InceptionV1.**
  - mixed5b at the p75 cut: 69.47% vs 70.08%. At p95 it costs 7.7 points; at p99, 33.6 points.
  - Early layers cost most at p75: conv2d0 -22.4, conv2d1 -20.9, conv2d2 -11.2 points.
- **The essay's own data, ConvNeXt (residual bypass intact).**
  - The costliest block is stage0.block0: medium -13.4 points at p99, large -3.5. Stage 3 costs about 0.
  - Network-wide at p75: the tiny-sized model loses 4.62 points (13.73% to 9.10%), medium 1.02, large 0.55.
- **Rare classes.** Their median top-1 is 0 while auroc_far is 0.83 to 0.95, so top-1 cannot see them.

**Verdict:** untested (high), unchanged.

**Caveats:**

- The near-tie claim rests on 4 curated examples, with a mass-matched control (removing the strongest 7% takes the medium
  model from 72% to 4%) but no margin-matched null.
- Toy Models puts packed features in the lower activations, so "removable" does not mean "not superposed".

**Next test:** a per-layer percentile ablation (p25 to p99) on InceptionV1 and E1, scored by top-1, CE and KL,
rare-class loss and flipped-image margins against margin-matched random flips, with a random-basis control.

### C20. Unrelated-looking weak activations trace to related circuits (supporting)

> "Of the probably hundreds I've followed by hand, almost all turned out to be related, coming through a reasonable
> circuit; it just wasn't visually obvious until I traced it."

**Our tests:** E0, E0b, E0c, E0d, the audits and the bias teaching signal.

**Numbers:**

- **E0c, the one partial weak-activation trace.** On 241's and 295's top dog images, 16 mixed4e readers sit at their
  58th to 97th percentile. The mixed4d unit writes 6-40% of their response (median about 20%); the rest is unattributed.
- **4e:55.** Its vehicle class means lie between the 90th and 99th percentiles of its maxima: a weak-band case settled by
  reading, not tracing.
- **E0d's related-pair positive control** is barely above distant pairs (mixed4c 22.7% vs 22.9% shared), so the test has
  little sensitivity.
- **Late ConvNeXt blocks.** Relatives outside a unit's top 20 push its bias up by +0.15 (E1), against +0.69 for the top
  20.

**Verdict:** measure cannot decide (low), unchanged. Strictly it is untested for weak activations and for language models.

**Caveats:**

- The essay concedes: "We don't yet have a semantic test for weak activations that we trust".
- Through a dense network some path can be followed to almost anything.

**Next test:** a pre-registered, blinded trace of far-class activations in the 90th to 99th percentile band. Use units,
random directions (negative control) and E7's crushed units (positive control), and score route overlap against the
reader baseline.

### C21. Rare classes are consolidated; adversarial routes are normal (supporting)

> "Rare classes are often more consolidated ... When I removed weak activations in the last stage, no class stood out.
> The adversarial examples we traced used normal semantic circuit routes."

**Our tests:** rare classes, seed 0. The S7 consolidation measure is coded but has n = 0. There is no attack code and no
weak-activation ablation.

**Numbers:**

- Post hoc top-carrier share of K_effect (the summed specific effect of a class's 8 carrier units):
  - isolated 0.211 vs dense 0.212 (n = 56 each, p = 0.90);
  - rare 0.196 vs the same classes in C0 0.211 (p about 0.003).
- ImageNet classes have 732 to 1,300 training images each, so Triceratops is isolated, not rare.
- The essay's figure gives only aggregate numbers for the medium model's stage-3 blocks at p99 (-0.65 / -0.36 / +0.06
  points).

**Verdict:** untested (high), unchanged.

**Caveats:**

- An attack that raises the jaguar logit must raise units with large weights to jaguar, so recruiting jaguar units is
  nearly guaranteed.
- Rare head share (0.912 vs 0.563) is confounded with AdamW's per-row step size (median ratio 6.1).

**Next test:** run S7 (C1 and participation ratio of each class's late-unit response) on InceptionV1 mixed5b from existing
statistics, comparing isolated classes with dog classes per class, against random rotations of the layer.

### C22. Rare items ride related machinery (central)

> "The prediction is testable: the neurons and heads that carry a rare fact should be related to what the fact is about,
> not unrelated ones chosen because they rarely collide."

**Our tests:** rare classes, seed 0 (runs A0, B0 and C0; 112 primary classes at 10 or 100 images), plus T4, T5, exits,
E7 and E8.

**Numbers:**

- **Carriers are related.** R_own 0.0143, R_flow 0.0098, R_mis 0.0093, chance 0.010; Delta_flow +0.0043 [0.0036,
  0.0051] (n = 106).
- **But the same holds for common classes.** C0 gives +0.0039 (n = 112); rare minus C0 is +0.0005 (p = 0.30).
- **The quiet-unit prediction fails.** Spearman(w/f, q) has median -0.15 and is negative in 111/112 classes, so own
  writes lean to busy units.
- **Feeding (P1) cannot decide.** PI follows the sign of F in 110/112 classes, and all 23 defined classes have F < 0.
- **Rarity-specific differences, all post hoc, lean toward more shared machinery.**
  - Stage-0 carriers: 133 rare against 9 in C0.
  - One unit carries 17 of 56 rare classes.
  - The stage-3 null shift is +0.30 in 56/56 rare classes, against -0.004 in C0.
  - Packing signatures: 18 rare against 11 in C0 (McNemar p = 0.21).

**Verdict:** measure cannot decide (medium), unchanged. This matches the pre-registered decision: no support and no count
against.

**Caveats:**

- Rare classes were not memorized (top-1 0; fit_T 0 at step 12,500), so the runs test weakly learned classes, not rare
  facts.
- The controls were uninformative.
- Seed 0 only; seed 1 was blocked when T90's sandbox was lost.
- Shelter (S4) was not run.

**Next test:** rerun the paired rare-vs-common design with items that actually fit (50% or more top-1 on their own
images), a noise positive control that fits and has no relatives, a scale-free P1, and S4 shelter.

### C23. Memorization sits next to the output (supporting)

> "Across all experiments in language and vision, memorization tends to want to sit as close to the output as possible."

**Our tests:** rare classes, seed 0 (lambda_3, head_share, S2), the noise and relabel controls, the biography model
(language), and E1 T4.

**Numbers:**

- **Within the unit maps.**
  - lambda_3 is 0.426 [0.406, 0.445] (n = 110).
  - Stage shares 0.036 / 0.024 / 0.497 / 0.426.
- **With the head counted, the concentration is not specific to rare classes.** Stage 3 plus head holds 0.948 of the rare
  own effect (110/110 classes above 0.8), against 0.932 for the whole common class and 0.726 for the same images in C0.
- **Head concentration does not track fitting.** All 8 noise labels have head share 0.974 to 0.99996; only 2 fit.
- **Per-image specificity rises toward the output identically** in rare and C0 classes (level 10, head 2.40 vs 2.45).
- **Language.** With the table read off, the biography model answers 25.5% of old facts under AdamW and 88.4% under
  Muon, so where lookups sit depends on the optimizer.

**Verdict:** measure cannot decide (medium to high), unchanged. Nothing of ours locates memorized items, and late
concentration is also what ordinary learning shows.

**Caveats:**

- Head share tracks AdamW's per-row step (Spearman 0.76).
- E6 was lost, not running.
- Maini et al. (2023) report memorization spread over many layers.

**Next test:** rerun E6's localization on photos whose new label is actually fit, with a correctly labelled control photo
at the same dose, in AdamW, frozen-value and Muon arms.

### C24. Umbrella -> bee in nine steps (supporting)

> "In nine training steps, the top guess went from umbrella to maypole to bee. The umbrella neurons faded and existing
> bee neurons came on, nearly all of it in the last three blocks."

**Our tests:** E6 (fine-tunes from E1 with exact attribution; pre-registered swap analysis), plus rare-class lambda_3 and
head share, and the audits.

**Numbers:**

- E6 ran about 140 jobs on molab-G, none failed. Outputs written after the 2026-10-17 storage limit were lost when the
  sandbox was reclaimed, and nothing was analysed.
- E1 recognizes 0 of 50 umbrella validation images, so the primary arm is jack-o'-lantern -> bee (90% top-1).
- The essay's evidence is 1 photo and 6 hand-picked units. The final bee probability is 52.8%.
- Rare-class head share is at least 0.78 in all 110 classes, so a swap that includes the head could say 'bee' without
  stage 3.

**Verdict:** pending (high), unchanged by the skeptic. Note that "pending" here means a rerun is needed, not that a run is
in progress. By the standard applied to C16 it is close to untested.

**Caveats:**

- Even a rerun tests only a tiny-model analogue.
- Four of the six essay units are read on their negative pole, where the pilot gives 'one job' in only 59 to 60 of 80
  ratings.
- The bee units were nearly silent before learning (#752 at 0x, #797 at 0.11x of a typical bee response).

**Next test:** rerun and analyse E6. Then on a medium model that recognizes umbrellas, swap in stage 3 alone, the head
alone and both, with carriers vs matched non-carriers and random directions, over 5 photos and 3 seeds.

### C25. Packing only in structureless memories (speculative)

> "The reuse argument also says where I'd look for packing: memories that really are ends in themselves, related to
> nothing else. That's the toy's world, and there it does pack."

**Our tests:** E8, the rare-run noise control, E0d, E7 and rare-class C1.

**Numbers:**

- **The toy packs**, in ReLU at rho 0 (3 of 4 cells). In SwiGLU the antipodal excess is about 0: +0.04 / -0.03 / +0.05 /
  -0.00.
- **"Only".** At rho 0.6 the unrelated antipodal packing persists (ReLU +0.14 to +0.28), but every E8 feature is a
  reconstruction target.
- **The noise control** used smooth low-frequency noise, trained from scratch.
  - 2 of 8 labels fit.
  - The 4 labels of a run share one carrier route (8/8 carriers in common in A0).
  - Their own effect sits in the head.
- **Post hoc.** One B0 unit that carries all 4 noise labels is also the third carrier of 'wool'. This is one unit at the
  base rate of carrier sharing, so it is an anecdote.

**Verdict:** measure cannot decide (medium), unchanged.

**Caveats:**

- The essay says only "where I'd look", and it did not fully audit its noise model.
- The noise control "cannot test memorization" as built.
- Correction: the co-positive excess of unrelated co-occurring pairs is 2.7 to 5.5 SE above 0 in 5 of 8 cells, though
  tiny.

**Next test:** fine-tune a trained model on high-frequency noise with random labels to at least 50% fit, with a matched
relabelled-photo arm. Measure packing signatures and the share of own effect in fine-texture units against a matched
null, with a Muon arm.

### C26. Coarse-to-fine from the first steps (supporting)

> "But in most natural data we've looked at, coarse-to-fine wins right away. The network doesn't memorize first and
> generalize later."

**Our tests:** none targeted. Context comes from E1's early validation curve, E1r's between-class share, rare classes and
the noise-control timing.

**Numbers:**

- **E1 top-1** (n = 10,000): 0.08% at step 0, 0.26% at step 100 (Wilson 0.18-0.38; p = 1.7e-5 against 0.1%), 1.47% at
  step 1,000.
- **Same-seed runs at step 100.** A0 gives 0.13% (p = 0.21, not above chance) and B0 0.23%. All three runs are above
  chance by step 200.
- **Noise labels** fit only after 5 epochs, while real classes are above chance by step 100 to 200. This is an
  Arpit-like order on n = 2 labels.
- **The essay's ten-photo result** (about 20% vs chance 10%) is matched by color matching and nearly by nearest-pixel
  lookup, which is pure memorization.

**Verdict:** untested (high), unchanged.

**Caveats:**

- In the first epoch no image repeats, so memorizing first cannot be told apart from generalizing.
- E1r's between-class share would be high under random labels too.

**Next test:** replicate the ten-photo experiment with 10 or more seeds. Log training-photo fit and held-out top-1 at
every step from 0 to 300, with color-NN, pixel-NN and shuffled-label baselines.

### C27. Circuit racing (speculative)

> "A region of the network that already has useful parts and readers can learn the next related thing faster ... Offshoots
> look like the same race, a bit of error picked up by the neuron that was already closest."

**Our tests:** exits (X1 to X5) on E1 and E1r, T5, T2, T3, the bias pushes, stage 1, E8 and rare-class P1.

**Numbers:**

- **Generic signals.** These fit, but any gradient account predicts them:
  - leader capture F1, pooled 0.62 (E1) and 0.66 (E1r); stage 3 0.83 and 0.85;
  - X5, 0.77 vs a null of 0.59 and 0.84 vs 0.66.
- **The specific test (F2).** It reverses and grows against the claim: +0.033 / +0.039 before step 2,500, -0.089 /
  -0.096 at steps 2,500 to 10,000, and -0.135 / -0.147 after 10,000. Stage 3 after 10,000 gives -0.30 / -0.31.
- **Stage-1 leaders** capture less than the median unit (F1 0.39 / 0.45).
- **X3 starvation:** +0.012 (p = 0.48) and +0.006 (p = 0.72).

**Verdict:** not supported, leaning against (changed by the skeptic from "partly supported"; low to moderate). The essay
guesses the head start strengthens over training. In two seeds it reverses after step 2,500.

**Caveats:**

- Two AdamW seeds of a tiny model.
- The gating baseline kappa uses no optimizer preconditioner (D = 1).
- The F2 columns are nearly collinear.
- The 'readers' half is untested, because its statistic (psi_beta = 1.3e-7) looks broken.
- T5 does not test the essay's milder head start.

**Next test:** from a step-5,000 snapshot with optimizer state, run paired replays that zero the loss weight of about 100
target classes' visual relatives, against a matched random set. Measure leader capture and target learning speed against
the replay-vs-replay floor.

### C28. Distributed recognition first, consolidation later (speculative)

> "A network trained on cars, with car specialists, recognizes cars fine before its car neurons have formed ... It seems
> to consolidate so the specialists underneath have a sheltered place to build."

**Our tests:** the E1 and E4 car maps, the essay's two tiny runs, the essay authors' notes, exits and the audits.

**Numbers:**

- **Best single-unit car AUROC.**
  - E1: 0.72 at step 0, 0.83 at step 1,000 (top-1 1.5%), 0.92 at the end.
  - The essay's runs: 0.77 / 0.74 at step 0 and 0.71 / 0.69 at step 1,000.
- **The essay authors' notes.**
  - Their multi-neuron readout is already 0.855 to 0.874 at random initialization.
  - The ambulance coordinate is upstream of the car coordinate in all three runs.
  - The precursor ablation was within matched controls.
- **Our car map shows no generalist-first order.** E1's best ambulance-vs-cars unit is 0.871 at step 1,000, above the
  best car unit (0.833).
- **Capture concentration in stage 3 falls over training** (E1 0.0175 to 0.0148; E1r 0.019 to 0.0135), while
  on-ness concentration rises.

**Verdict:** untested (medium), unchanged. The indirect within-layer evidence, now replicated in E1r, leans against parts
(b) and (c).

**Caveats:**

- No network-level car readout was measured over training.
- E1's early timeline comes from the run with corrupted early batches.

**Next test:** from a step-1,000 snapshot of a clean rerun, run a control replay and a replay with the 10 car classes
merged into one label. Measure network-level car AUROC (logits and a probe) and unit consolidation at every checkpoint.

### C29. Organization strengthens over training and with capability (speculative)

> "Our guess is that this gets stronger over training and with more capable models, because nearly everything that makes
> training better also encourages more organized niches."

**Our tests:** exits F2 by phase, F1, the stage-1 time course, E4 (Muon), the audits, the width-16 run, E7 and E8.

**Numbers:**

- **Over training.**
  - Stage-3 F2 goes +0.012 -> -0.154 -> -0.299 (E1) and +0.023 -> -0.150 -> -0.312 (E1r).
  - Leader capture F1 rises: network 0.669 -> 0.894 in both seeds. This is a generic signal.
  - Stage-1 units' top classes go from helping them (+0.12 at step 200) to pushing them down (-0.42 at step 20,000, E1).
- **Optimizer.**
  - Muon leads only early (top-1 3.0% vs 1.5-1.7% at step 1,000).
  - It ends at 19.07%, between the AdamW seeds, with fewer selective units (best car 0.889 vs 0.916) and 58 dead units
    (26% of stage 0).
- **Accuracy and legibility come apart.** The essay's tiny model reads 0.91 at 13.8% top-1; E1 reads 0.59 at 18.5%.
- **E8 SwiGLU.** Doubling H lowers antipodal packing in 4/4 comparisons; ReLU is mixed.

**Verdict:** not supported (changed by the skeptic from "pending"; low to medium). Weakly against for "over training" at
tiny scale; mixed and untested for "with more capable models". T90 was lost, so no direct test is running.

**Caveats:**

- Single seeds for E4, E7 and width 16.
- E4 is not a pure optimizer swap.
- "Organized niches" is undefined in the essay.

**Next test:** restart T90 to 90 epochs. At 10, 30, 60 and 90 epochs, audit the gap between units and random directions
next to top-1. Add an optimizer arm with 2 or more seeds that ends clearly above the AdamW seed spread.

### C30. Offshoots are rare and unimportant; no sleeper agents (supporting)

> "First, the sub-branches seem both rarer and much less causally important than a neuron's main job. They don't look
> like sleeper agents that fire rarely but matter a lot when they do."

**Our tests:** exits, stage-1 ablation profiles, E0, E0b, E0c, the audits, E7 and rare classes.

**Numbers:**

- **Top activations do not equal function.**
  - Stage-1 units hurt their top 20 classes: ablation changes their CE by -3.2e-3 / -1.3e-3 / -2.6e-3 in E1 / E1r / E4,
    against +6.3e-4 / +5.6e-5 / +4.5e-4 for random classes.
  - A late unit's 20 most-helped classes overlap its top 20 only 0.07 to 0.29.
- **"Rarer".** 4e:55's vehicles are a minority (ranks 11-36), but 241 and 295 split evenly.
- **'Several unrelated' depends on the rater.** Haiku gives it to 0/128 E1 units and Sonnet to 4/20. E1 has 52/128
  'not understood'.
- **"Pressure".** The F2 reversal and the downward bias pushes from about 960 classes (-0.045 each) fit it. Starvation
  (X3) is null. X1 and X2 fail the validity gate.

**Verdict:** untested (medium), unchanged.

**Caveats:**

- Rare classes were learned as directions (auroc_far up to 0.953), so sleeper agents could arise there. They were not
  measured.
- The ablation profiles cover 5 blocks with 8 images per class, so per-firing costs need new computation.

**Next test:** fix each late unit's core (classes it both activates on and helps) and its offshoots on one half of the
validation images. On the other half, measure per-class firing frequency and per-firing ablation cost, against random
class sets and random directions.

### C31. The toy grows related inputs onto shared neurons (supporting)

> "When we put correlated inputs and hidden neurons in one model, related inputs grew onto the neurons. Making inputs
> fire more often pushed the toy toward what the paper calls 'PCA-like' solutions."

**Our tests:** E8 only, which is a cousin of the essay's toy. In E8, relatedness is an embedding cosine of 0.6, not
co-firing, and density is fixed. No PCA metric is computed.

**Numbers:**

- 13 of 16 co-positive DiDs are positive, but only 4 of 16 t-intervals (n = 10 seeds) exclude 0.
- ReLU related co-occurring pairs are at chance in all 4 cells (+0.049, +0.075, -0.009, +0.002).
- The relatives' median share of a feature's pole response is negative in ReLU (-0.03 to -0.12) and positive in SwiGLU
  (up to +0.15).
- Sec. 6 itself: relatedness is "one driver of growth here, and it is not the main one".

**Verdict:** untested (changed by the skeptic from "partly supported"; moderate). E8's evidence is weak and confined to
SwiGLU, and the frequency half has no test.

**Caveats:**

- A PCA-like collapse at high density is Toy Models' own result, as the essay concedes.
- The essay's figure measures the bottleneck, not the neurons.

**Next test:** rebuild the essay's toy (20 inputs in 4 co-firing groups, a 5-dimensional bottleneck, a hidden layer),
sweep density, and add uncorrelated and no-hidden-layer controls.

### C32. Qwen3-32B: about 90% one job (central)

> "In preliminary audits of Qwen3-32B, we estimate that roughly 90% of neuron poles have one main job, often with small
> related offshoots; another ~7% have two related jobs; and only ~2-4% are not yet understood."

**Our tests:** none on any language model. The vision audits bear on the method only.

**Numbers:**

- With the essay's five-way call, units do not beat pre-ReLU random directions in InceptionV1 (p = 0.25, 0.58), and lose
  to random directions in E1 (61/64 vs 76/128) and T90 (60/64 vs 90/128).
- 'Two related jobs' is rare: about 1.7% of unit ratings and 2.5% of random-direction ratings, pooled.
- The essay's own GPT-2 XL re-audit (about 70% roughly one thing, against under 2% in 2023) suggests that most of the jump
  comes from the method, not the model.

**Verdict:** untested (high), unchanged.

**Caveats:**

- The essay's richer agent pipeline may separate units from random directions where ours does not.
- The essay mentions 1,000 random neuron papers but ties no n, model or protocol to the 90/7/2-4 split.
- Its Kimi K3 statement rests on informal inspection.

**Next test:** a blind Qwen3 audit with matched random directions (pre- and post-activation) and a rotated basis, using
the essay's pipeline or our five-way call plus held-out tests. Score GPT-2 XL with the same pipeline.

### C33. SwiGLU value is broad, gate specific; the double-negative echo (supporting)

> "The value tends to learn broader things, and gate more specific things. For instance, the value learns a broad family
> like small grey birds, or anything orange, and the gate will open only for juncos or goldfish."

**Our tests:** none of breadth. Context comes from the stage-1 sign-lobe counts and the rare-class gate-only and
value-only F.

**Numbers:**

- The double-negative lobe (g < 0, v < 0) holds 49-57% (E1) and 24-53% (E1r) of stage-1 unit-class pairs at step 20,000,
  against 0-15% in late blocks.
- Negative-lobe units are 96-99% sign-pure in their top 20, which shows consistency, not that the regime is noise or an
  echo.
- Rare gate-only F (+1.61) and value-only F (+6.06) describe carrier pathways, not tuning breadth, and are scaled
  differently.

**Verdict:** untested (high), unchanged.

**Caveats:**

- The essay's public microscopes already expose a per-unit double-negative-at-max fraction (doubleNegativeAtMaxFraction)
  for the medium and large models. We have not pulled it.
- The essay's 4% has no stated threshold or denominator.

**Next test:** this needs no training. On the E1, E1r and E4 atlases, compare each unit's gate and value breadth (classes
above half max, WordNet and DINOv2 spread, entropy) with a swapped-label null. Pull the microscope double-negative
statistic.

### C34. Explanations can be scored falsifiably (methodological)

> "Among the 183 whose explanations describe their positive side, the new examples fired their neurons strongly 76% of the
> time; the near misses, 11%. The explanation won for 164, tied for 15 and lost for 4."

**Our tests:** the held-out description test in every audit, the InceptionV1 control, E0 (4e:55 thresholds) and E8.

**Numbers:**

- **The essay's counts.** Won 164/183 = 0.896 (Wilson 0.844-0.933).
- **Our blind test can lose.**
  - E1: 13 of 64 units score 4 or less of 8.
  - mixed5b pilot: 6 of 40.
  - The essay's medium model: 7.45 of 8, with no unit at 4 or less.
- **It scores random directions like units, or better.**
  - Pre-ReLU random directions match units: 5.71 vs 5.60 (mixed5b), 6.03 vs 6.03 (mixed4c).
  - Post-ReLU directions do worse (strict 17 vs 32, p = 0.016).
  - In E1 and T90, random directions do better.
- **Base rate.** For 4e:55, 30% of the maximum (143.5 of 478.25) lies between the 90th and 99th percentiles, so 1-10% of
  all images clear it.

**Verdict:** partly supported (low), unchanged. The method is supported by analogy. All the Qwen numbers are untested.

**Caveats:**

- The distant arm counts only strong positive firing. E8 places packing on the negative pole (ReLU) or below chance on
  the positive pole (SwiGLU), so the arm has little power against it.

**Next test:** run the essay's protocol unchanged on matched random directions in Qwen3 MLP layers (a smaller Qwen3 if
needed), with a planted-packing positive control, and report the distant arm's base rate.

### C35. mixed5a:719 is generalization with semantic residue (supporting)

> "One InceptionV1 neuron collects a pinched-center shape: bow ties, butterflies, swimsuit tops. But it also picks up
> things that go with them ... since they share neurons in earlier layers."

**Our tests:** none on mixed5a; E0 never collected that layer. We have E0d on mixed4e (which feeds it) and mixed5b, the
E0b analogy and E8.

**Numbers:**

- 719 sits in mixed5a's pool branch (channels 704-831), which reads max-pooled mixed4e.
- In mixed4e, 23% (18-29%) of 494 mixed units read their two groups through shared inputs.
- **Post hoc.**
  - mixed4e has swimwear-plus-co-occurring-object units called shared: 4e:727 (snorkel and pool table vs bikini, cosine
    excess 0.158 [0.119, 0.282]), 4e:790 and 4e:231.
  - 4e:727's linearized weight into 719 is negative (-0.056).

**Verdict:** untested (high), unchanged.

**Caveats:**

- By the essay's definition, co-occurring objects count as related, so this case cannot separate the hypotheses.
- Shoes may appear in bow-tie photos, which is a spatial confound.
- DINOv2 'near' absorbs co-occurrence.

**Next test:** extend E0 and E0d to mixed5a. Split 719's response exactly into inputs on shape images and on shoe-only or
bag-only images (heat-map checked), against the pool-branch reader baseline. Add a co-occurrence relatedness score.

## What we learned that the essay does not say

1. **The audit control.** Reading dataset examples, the essay's own method, does not show a privileged basis.
   - InceptionV1 units and pre-ReLU random directions read alike: mixed5b 46 vs 54 of 80, p = 0.25; mixed4c 63 vs 59,
     p = 0.58.
   - In our tiny ConvNeXt, random directions over the SwiGLU products read better than units: E1 61/64 vs 76/128,
     p = 2.5e-8; T90 at 10 epochs 60/64 vs 90/128, p = 0.0002.
   - Units win only against post-ReLU directions in InceptionV1, and only on the strict count.
   - So a high 'one job' share (99%, 90%) measures how coherent the top tails of a layer's directions are, not that the
     neurons are clean. None of the essay's legibility numbers comes with this control.
2. **Recipe, not only architecture.** The same ConvNeXt-SwiGLU reads 0.59 'one job' after our 10 epochs and 0.91 in the
   essay's 90-epoch tiny checkpoint. The cause (length, recipe or width) is not decided. The essay's clean tiny model is
   the least accurate (13.8% top-1). The essay's medium mid layer (48/80) reads no better than InceptionV1 mixed5b
   (50/80).
3. **The stage-1 reversal and X4.**
   - In stage 1 of every tiny run, a unit's 20 strongest classes push its response down (pole push -0.28 to -0.82), and
     ablating the unit lowers their loss.
   - X4 traces this to the classifier: the weights-only term (the rows of the classifier for the unit's top classes,
     carried back through the final norm against the unit's write direction) has the reversal's sign in 12 of 12 stage-1
     blocks over 4 runs (Spearman 0.77-0.98 with the head reader).
   - Widening stage 1 to 16 removes the write-space collapse but not the reversal.
   - So a low-level direction is read through the skips and undone by the stage's own units on the very classes it
     signals. That is the kind of stream accounting the essay says it does not see. It also shows that top activations
     can describe what a unit sees while its output hurts those classes.
4. **First in line reverses.** The relatives' head start (X4 F2) is positive only before step 2,500 and grows negative
   afterwards, to -0.30 in stage 3, in two seeds and both half-directions. Exit events fail their own validity gate
   (25-27% replicate against 50% required), so "which pressure removes offshoots" stays open.
5. **The rare-class construction issue.** As built, the rare-class runs do not test rare facts or memorization:
   - Classes with 10 or 100 images were never fit (top-1 0; fit_T 0).
   - The noise control used smooth low-frequency noise. Its 4 labels were learned as one shared route, only 2 of 8 fit,
     and their null (the other 3 noise labels) subtracts that shared route by construction.
   - The head share (0.912 vs 0.563) tracks AdamW's per-row step ratio (median 6.1, Spearman 0.76).
   - P1's index reduces to the sign of F (110/112), and its definedness gate would reject a real own niche (|O| below
     threshold in 112/112 C0 classes).
   - A valid design needs fitted items, a fitting positive control with no relatives, and a Muon arm.
6. **What the toy says about packing.** In E8, anticorrelation packs unrelated features antipodally in ReLU units, not
   onto a shared positive pole, and SwiGLU packs little either way. The excess does not need more features than
   dimensions: it is largest at m = H = 32. Relatedness pulls siblings off opposite poles, so taxi/ambulance-style poles
   need another cause. A positive-pole locality test (C06) or a distant-text arm (C34) cannot see the packing the toy
   produces.
7. **The car circuit, exactly.** The two readers the essay calls pose neurons read cars and dogs through different
   inputs: overlap below the reader baseline, with 447 writing 0 on dog images. The one shared input is rated 'several
   unrelated' in every blind rating. 4e:55's car response arrives through a car chain (4c:447 -> 4d:352).
8. **Sibling negative poles grow with scale (post hoc).** The share of last-layer units with a relative on the negative
   pole rises from 1/64 (tiny) to 0.39 (medium) to 0.54 (large), against a null of about 0.02. The sibling is usually the
   larger pole and the SwiGLU gate opens more for it, which looks like a signed contrast code more than a gate.
9. **Provenance and process.**
   - Lucent's InceptionV1, which the essay's microscope uses, differs from Google's graph by about 2 times the
     activation scale after the first LRN.
   - The essay's tiny widths are not determined by its published parameter count.
   - E1 had a loader race in its early batches: clean reruns diverge between steps 100 and 200, and E1's early order
     result did not replicate.
   - The tiny audit's raters worked without class labels, unlike the other audits.

## Open experiments, ordered by how much they would change the picture

Blockers for anything that trains:

- both sandboxes were reclaimed on 2026-10-18;
- the results dataset has been at its storage limit since 2026-10-17, so uploads fail;
- the fp32 history snapshots were lost with molab-F, which limits replays.

The experiments:

1. **Units against random directions over training, at a scale where units might become privileged.** Restart T90 to 90
   epochs and run the medium M10 run with random directions in its microscope. Audit the units-minus-random gap at 10,
   30, 60 and 90 epochs next to top-1. Decides whether our "units read worse than random" result is a property of
   undertrained tiny models or of the method. Bears on C01, C04, C18, C29 and the reading of C02, C03 and C32.
2. **The random-direction control on the essay's language-model method.** Run the neuron-paper pipeline and the
   explanation test (positives, near misses, distant texts) on matched random directions in Qwen3 SwiGLU layers (a
   smaller Qwen3 if needed). Score GPT-2 XL with the same pipeline. Decides whether 90% and 164/183 say anything about
   neurons. Bears on C32, C34 and C01.
3. **A matched-recipe architecture swap.** Train InceptionV1 and the 7.06M ConvNeXt under one recipe, plus residual-stream
   and LayerNorm add and remove arms, with 2 or more seeds. Run the same blind audit with pre- and post-ReLU random
   directions, and E0d. Bears on C02 and C03, the essay's headline causal claim.
4. **Offshoot provenance and sleeper agents in a run whose units beat random directions.** Use the exact record to
   measure whether offshoot entries are written by related core classes or unrelated ones, against size-matched and
   planted-order nulls. Measure per-firing ablation cost of core against offshoot classes. Bears on C01, C05, C07 and C30.
5. **Medium crush and narrowed stream** (C13 with 3 seeds, N4, N8), scored by the pre-registered non-local rule against
   random 13-dimensional directions, with a planted-packing positive control for the locality measure. Bears on C16,
   C17, C06 and indirectly C15.
6. **Viewpoint-labelled tuning tests (no training).** On InceptionV1 241, 295, 139 and all mixed4c-5b units, DINOv2, and
   the essay's microscopes, against random directions. Add the exact input split and within-class shape tuning for
   4e:55. Bears on C10, C11, C12 and C09.
7. **Rotation search rerun and a TopK SAE with a rotated-activation positive control** on our medium last block. Ablate
   unmatched atoms and score with KL and per-class AUROC. Bears on C18.
8. **Weak-activation percentile ablation** on InceptionV1 and E1 with CE/KL, rare-class loss, a margin-matched flip null
   and a random-basis control. Bears on C19 and C21.
9. **E6 rerun and the memorization design of `EXACT_HISTORY.md` 4.4** with items that actually fit, control photos and a
   Muon arm. Bears on C23, C24, C22 and C25.
10. **Causal replays:** a head-start replay with relatives' loss weight zeroed (C27) and a merged-car-label replay (C28).
    Both need snapshots with optimizer state.
11. **Cheap analyses of existing data:**
    - gate vs value breadth on the E1, E1r and E4 atlases, plus pulling the essay microscopes' double-negative statistic
      (C33);
    - E0 and E0d extended to mixed5a for unit 719 (C35);
    - S7 consolidation on InceptionV1 mixed5b with a rotated-basis control (C21);
    - the sibling logit split with a clamp of u < 0 (C13);
    - the held-out negative-side test (C14).
12. **Small studies:** rebuild the essay's co-firing toy with a density sweep (C31); replicate the ten-photo experiment with
    seeds and color and pixel baselines (C26); blinded weak-activation tracing with crushed-unit and random-direction
    controls (C20).
