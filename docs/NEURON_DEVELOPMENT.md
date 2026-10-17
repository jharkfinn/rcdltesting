# Neuron development through training, with exact training-data provenance

Working notes for the follow-up to the RCDL reproduction (docs/REPRODUCTION.md): use the dual form of trained layers
(RCDL Sec. 2) to follow, at the level of single neurons and single groups of training examples, how the neurons of small
image models form. The target is the image-model evidence in N. Cammarata, "Neural Networks Are Already Clean"
(Goodfire, 2026; already-clean.goodfire.com, read in full 2026-10-16).

## 1. What the essay claims about small image models, and what is published

Claims: polysemantic neurons in InceptionV1 came from bad tools (reward-hacked feature visualization), a bad architecture
(no residual stream or LayerNorm) and "recursive generalization" (a neuron grows outward along related branches), not
from superposition; a parameter-matched ConvNeXt has about 99% interpretable last-layer neurons; crushing a layer
(InceptionV1 mixed3b 480 -> 10, ConvNeXt last layer 1,024 -> 13, residual stream 4 and 8 times narrower) makes neurons
coarser, not mixed; SAEs add no new building blocks; an "umbrella = bee" label is learned in 9 steps through the existing
bee route; neurons grow outward over training (InceptionV1 neuron 338: rapeseed fields -> tennis balls, green apples ->
shaggy dogs, broccoli, epochs 0.7 to 12).

Published: no code; the trained checkpoints are named in the microscope metadata but not served (HTTP 403). Public:
the microscopes' metadata and figure data, a training-map manifest (architecture sizes, recipe, probe image ids,
per-checkpoint scores) and the authors' own early-training notes (`assets/early-training/RESULTS.md`), which are more
cautious than the essay: the trajectories "do not yet demonstrate a causal generalist-to-specialist consolidation
mechanism", and the test of an early precursor unit found effects "within the range of matched controls".

Gaps found in reading (details in the conversation log of 2026-10-16):
1. Superposition is redefined as "arbitrary packing of unrelated things"; the Toy Models sense (more features than
   dimensions) is set aside, so "no superposition" is a narrower claim than it reads.
2. The local versus non-local test does not separate the two views in natural data: Toy Models predicts that
   anticorrelated features share a dimension as antipodal pairs, and sibling classes (taxi / ambulance) are both
   anticorrelated (one label per image) and related. Related pairs on two poles fit both views.
3. The mixed3b crush uses dense features (curves, textures), for which Toy Models also predicts coarse, PCA-like codes.
4. Growth is shown on one or two neurons, qualitatively; the essay says a whole-model test "would not be very hard" but
   does not run one.
5. The weak-activation ablation measures top-1 accuracy only.

## 2. Provenance

| Item | Source | Status |
|---|---|---|
| InceptionV1 weights | Google, download.tensorflow.org/models/inception5h.zip (sha256 d13569f6...), the graph Lucid uses | used as the reference |
| PyTorch weights | ProGamerGov/pytorch-old-tensorflow-models inception5h.pth (sha256 44f1f1e0...), loaded by Lucent | **equal bit for bit** to Google's graph, 116 tensors (`rcdl/inception_provenance.py`); the 6 auxiliary training-head layers are not ported |
| PyTorch model | Lucent 0.1.8, vendored in `rcdl/third_party/lucent_inceptionv1` | **upstream differs from Google's graph**: its two local response normalizations use window 9 and bias 1 and pass alpha unscaled, while the graph has radius 5 (window 11), bias 2, alpha 1e-4 and torch divides alpha by the window; every activation after the first LRN then differs from TensorFlow by about 2 times its scale. With the graph's values, all named layers (conv2d0, mixed3a-5b, logits) match TensorFlow running Google's graph to float32 rounding (max relative difference 1.2e-6) |
| Preprocessing | Lucid InceptionV1: RGB, values x * 255 - 117 | as Lucid |
| ImageNet validation (the essay's microscopes use all 50,000 center crops, 224 from 256) | ILSVRC/imagenet-1k on Hugging Face (gated) | the user's account must accept the dataset terms |
| Probe images of the essay's training map | manifest.json image ids and scores | downloaded |
| ConvNeXt-SwiGLU (tiny 91,828 / wider 345,552 / medium 7,057,377 / large about 28.6M parameters; SwiGLU units per block 64, 64 / 64 / 128 / 192, 128 / 256 / 512 / 1,024 and 256 / 512 / 1,024 / 2,048; 18 blocks; 224 px; 10 epochs, 25,020 updates, mixup on images and labels) | inferred from the published metadata (`rcdl/convnext.py`) | widths match the published counts only for medium (399 parameters off, classifier excluded) and wider (24 off); tiny's widths are not determined; optimizer, schedule and other augmentation are not published |

## 3. Method: exact attribution of every update to groups of training examples

`rcdl/attribution.py`. For a linear map z = W h + b (a 1x1 convolution or a channels-last Linear), every update is built
from per-example gradients delta_i h_i^T. SGD with momentum, AdamW given its second moment and Muon given its
Newton-Schulz polynomial are linear in those gradients, so with one optimizer state per group of examples

  W_T = a_T W_0 + sum_k C_T^(k),   a_T = prod_t (1 - lr_t * weight_decay),

where C_T^(k) is the part of W_T written by group k; a neuron's pre-activation on any input h splits the same way.
For Muon, NS(U) = P U / ||U|| with P a polynomial in X X^T built from the total momentum U, so group k's part is
P U^(k) / ||U|| (the iteration runs in fp32 here; torch runs it in bf16). Tests (`tests/test_attribution.py`): the
tracked AdamW and SGD updates equal torch's to 1e-11, Muon follows torch's rule, the parts add up to 1e-12, and with one
group per example plain SGD gives the paper's linear-attention identity. A toy trained with and without the tracker is
bit-identical.

## 4. Experiments

| # | Experiment | Claim tested | Status |
|---|---|---|---|
| E0 | Static checks on the public InceptionV1 (V neuron 4e:55, the car / dog pose circuit 4c:447, every mixed unit of mixed4c-5b) | the essay's re-readings of canonical neurons | done (Sec. 5) |
| E1 | Writer atlas: ConvNeXt-SwiGLU tiny and medium, per-class attribution of the gate and value maps of late blocks, dense early checkpoints | instrument | code reviewed and ready (`experiments/convnext_train.py`, `experiments/convnext_atlas.py`, `experiments/convnext_growth.py`, `experiments/molab_e1.sh`); tiny AdamW run done, 18.5% top-1 (Sec. 7) |
| E2 | Order of growth: when each class enters a neuron's writer set, and its distance to the classes already there | recursive generalization | tiny AdamW and Muon done (Sec. 7): weak co-timing of near classes, no outward direction |
| E3 | Head start: did the neuron already fire on a class before that class wrote into it? | circuit racing | tiny AdamW and Muon done (Sec. 7): strict head start not seen; the essay's milder one not tested |
| E4 | Muon against AdamW against SGD | optimizer as an intervention on circuit racing | Muon done (Sec. 7, E4); SGD not run |
| E5 | RCDL head (growing table) as a memory store | do offshoots come from captured residual error? | |
| E6 | Umbrella = bee with exact attribution, parametric against RCDL head | off-ramp claim | |
| E7 | Crush where packing is predicted (long-tail classes, 13-unit last layer) | rare facts ride related circuits | 13-unit crush done (Sec. 7, E7); long-tail classes as the rare-class experiment (`EXACT_HISTORY.md` 4.3), running |
| E8 | Toy: relatedness against anticorrelation as the cause of shared neurons | gap 2 above | done, 160 runs (Sec. 6) |

## 5. E0 results: the canonical InceptionV1 neurons on all 50,000 validation images

Model and images as in Sec. 2 (exact weights, LRN fixed, Lucid preprocessing, 256 -> 224 center crops). Per image and
channel: spatial maximum and minimum of the pre-ReLU response (`experiments/inception_e0.py collect`). "Top 50" means
the 50 images with the largest maximum (or smallest minimum). Channel numbers are Lucid's concatenation order.
Results: `docs/data/e0/` (JSON). The per-image statistics (`val_stats.npz`, sha256 3b301c01...) and the montages
are in the private Hugging Face dataset `joshhf/rcdl-molab-results`, folder `neuron-dev/e0`.

**4e:55 (the "cat and car" neuron; the essay: a V-shape detector with an O on its negative pole).**
- Top 50 images: cats and foxes (tabby 7, red fox 6, kit fox 5, tiger cat 5, Persian 5, Pomeranian 4), no vehicles.
- Class means (`inception_e0 profile`): the top 9 classes are pointed-faced animals (papillon, Persian, red fox,
  Pomeranian, tabby, tiger cat, Japanese spaniel, Arctic fox, kit fox); then 10 vehicle classes fill ranks 10 to 35 of
  1,000 (beach wagon 10, convertible 14, pickup 18, minivan 20, limousine 21, ambulance 25, sports car 26, cab 31,
  police van 32, minibus 35). The first vehicle image is number 56 of 50,000. Both readings have support: the car
  response is real at the class level, and it is weaker than the response to pointed animal faces.
- Negative pole: broad, round-muzzled dogs (bull mastiff, pug, boxer, French bulldog) and round objects (soccer and golf
  balls, a frog in a hand, a snail shell, a hermit crab, a spider body), with the response centred on the muzzle or the
  round object (heat montage). This fits the essay's "O". Black-and-tan dogs (coonhound, Doberman, Rottweiler, Gordon
  setter) also rank high, so colour pattern is a second candidate.
- In InceptionV1 a ReLU follows every branch convolution, so the negative pole of a channel sends nothing downstream.
  The O is a property of the weights, not a signal that a later layer can read.
- Strongest linearized inputs from mixed4d (expanded weights through the bottlenecks): 4d:493 (foxes, papillon,
  Pomeranian; weight 0.31), 4d:511 (insects), 4d:61, 4d:352 (vehicles; 0.10).

**The car circuit (Zoom In's "superposition" example; the essay: two car neurons and two "generalized pose neurons").**
The essay's figure shows four mixed4d readers of the car neuron 4c:447: 4d:167, 241, 295 and 417.
- 4c:447: top 50 has 28 vehicles (ambulance 11, minibus 7, minivan 7) and no dogs; its negative pole has 39 dogs of 50.
  As above, the negative pole cannot reach mixed4d: on dog images 447 writes exactly 0 into every reader.
- 4d:167 and 4d:417: vehicle neurons (94 and 98 vehicles in their top 100, 3 and 0 dogs). 4c:447 writes 88% and 54% of
  their response on vehicle images; zeroing 447 lowers it by 73% and 60%. This agrees with the essay.
- 4d:241 and 4d:295: mixed (46 vehicles and 52 dogs; 49 and 49). Class means: 295's top classes interleave
  wire-haired terriers with station wagons, minivans and pickups; 241 interleaves vans with shaggy dogs (Bouvier,
  otterhound, Irish water spaniel).

E0b (`experiments/inception_paths.py`) asks what drives the two halves of 241 and 295. mixed4d is piecewise linear in
the mixed4c output x for one image, so the pre-ReLU response at the strongest position splits exactly as

  z = sum_c s_c + r,   s_c = sum_{y,x} x[c,y,x] * dz/dx[c,y,x],

where c runs over the 512 mixed4c channels, (y, x) over positions, s_c is channel c's share and r is the part written
by the biases (0.2% to 0.8% of z here). Check: scaling x by 1.001 changes z by exactly 0.001 * sum_c s_c (float64,
real weights). A shared pose detector would be driven by the same mixed4c channels on its car and its dog images; a
neuron that packs a car feature with a dog feature would be driven by car channels on cars and by dog channels on dogs.

| | 4d:241 | 4d:295 |
|---|---|---|
| cosine of the mean share vectors, vehicle images against dog images | 0.08 | 0.12 |
| the same between two random halves of the vehicle images / of the dog images | 0.999 / 0.987 | 0.997 / 0.993 |
| share of z from 4c:447 on vehicle images / on dog images | 0.76 / 0 | 0.45 / 0 |
| drop of z when 4c:447 is zeroed, vehicle images / dog images | 79% / 0% | 48% / 0% |
| share of the dog z written by the top 10 vehicle channels | 0.14 | 0.29 |
| top inputs on dog images (their own top 50 images) | 4c:460 (50 of 50 dogs: Westie, Norwich, Norfolk, Tibetan terriers), 4c:219 (sighthounds), 4c:506 (Great Pyrenees, kuvasz) | 4c:460 (37%), 4c:452 (poodles, wire-haired fox terriers; 48 of 50 dogs), 4c:66 (terriers) |

So on natural images, 241 and 295 are each driven by a vehicle detector (447) on vehicles and by dog-breed detectors
on dogs, with almost orthogonal input patterns. One mixed4c channel, 4c:41 (20 dogs and 18 vehicles in its top 50:
beach wagon, pickup, Irish and Yorkshire terriers), feeds both halves (11% and 8% of z in 295, 5% and 6% in 241), and
4c:66 (terriers) also writes into both halves of 295 (4% of the vehicle z, 14% of the dog z). These are the only shared
inputs we found, and they carry a small part of z. Heat montages agree: in 295 the vehicle response
sits on wheels and the lower side of side-on cars, and the dog response on the faces of terriers, mostly seen from the
front, so the top images do not share a pose.

Reading (revised after E0d, below). The low overlap of 241 and 295 is mostly what any reader of these images gets:
32 other units of the same mixed4d branch, read at the same positions on the same images, give a median overlap of
0.18 and 0.17 (10th to 90th percentile 0.07 to 0.31). 241 sits below 88% of them and 295 below 62%, so neither reads
the two groups through unusually separate inputs. Neither reads them through a shared input either: the excess over
the baseline is negative, -0.10 and -0.06, with 95% intervals that exclude 0. So the exact attribution gives no support
to a pose feature shared at mixed4c, and it does not show packing beyond what the images themselves impose. An earlier
version of this section read the low overlap as Zoom In's packing; without the baseline that reading was not justified.
Separate inputs alone also cannot exclude one variable built in mixed4d from class-specific channels (last limit).

| overlap against the reader baseline | tuning on a shared attribute such as pose | reading |
|---|---|---|
| above (and near the class reference) | any | a feature shared at the previous layer |
| below | matched across the two groups | one variable built in this layer (the essay survives) |
| below | unmatched or flat | packing |
| below | not tested | undecided |

Limits:
- The reader baseline was added in E0d (see the revised reading above): 32 other units of the same branch, on the
  same images, give a median overlap of 0.18 (241) and 0.17 (295).
- The cosine of mean share vectors depends on how concentrated the inputs are; 4c:447 holds about 60% of 241's
  positive vehicle mass. E0d adds a shared-mass measure and draws conclusions only where both agree.
- Only the 100 strongest images per unit are used; a pose code could live in the weaker range.
- Pose was not labelled; the claim about shared pose rests on the montages.
- The essay's guided ("natural path") feature visualization is not published, so its figure cannot be repeated.
- A pose unit could, in principle, be built as an OR of class-specific pose detectors. Then its inputs would differ by
  class, but its car and dog images would still share a pose. The montages of 295 show side-on cars and frontal
  terriers, so that variant is not supported either.

E0c (`inception_paths readout`, `docs/data/e0/readout.json`): who in mixed4e reads 241 and 295. Zoom In's account needs
a later unit that reads the car half back out; the essay says later layers "keep building on" the pose units. For the 8
strongest mixed4e readers of each (expanded weights), the exact share of the reader's response that 241 or 295 writes:
- 241's strongest reader, 4e:591, is a vehicle unit (45 vehicles in its top 50: jeep, fire engine, garbage truck); 241
  writes 8% of its response on its own top vehicle images. The next five readers are dog units (Bedlington, Afghan,
  Leonberg, Appenzeller); on their top dog images 241 writes 1% to 7%, and on their (weaker) vehicle images 7% to 19%.
- 295's readers are mostly not vehicle or dog units: 4e:543 (tabby, tiger cat, pickup; 27 vehicles), then units for
  sports, swimwear, meat loaf, bottles, an Afghan hound unit and an entertainment-center unit. 295 writes at most 14% of
  the vehicle response of 543.
- On 241's top dog images the vehicle unit 4e:591 reaches the 92nd percentile of its responses; 241 writes 16% of that
  and other inputs the rest.
So the car half of 241 is read back out by one vehicle unit, at a small share, and the car half of 295 hardly at all.
Both units feed many unrelated mixed4e units at shares of 1% to 30%. Neither "stored, then recovered" nor "a pose
feature that later layers build on" describes these two readouts well; they look like intermediate features with many
small readers. Limits: readers are ranked by linearized weights, which ignore the ReLUs; only 8 readers per unit.

**E0d: every mixed unit of mixed4c, 4d, 4e and 5b** (`experiments/inception_mixed.py`, reviewed before the run;
`docs/data/e0/mixed_v2_cut7.json.gz`). A unit is mixed when its 100 strongest validation images fall into two WordNet
clusters (average linkage, cut 7) of at least 15 images each. For both groups the exact input shares from the previous
mixed layer are compared (cosine of the mean share vectors, and a shared-mass measure that one dominant channel does
not sway), against the same numbers for 32 other units of the same branch read at this unit's positions on the same
images (the reader baseline). A unit is 'shared' when both measures exceed the baseline (bootstrap interval above 0,
above 95% of the readers, at least halfway to the sampling ceiling), 'separate' when both fall below it (interval below
0, below 95% of the readers), else undetermined. Visual similarity of the two groups: DINOv2 ViT-S/14 embeddings,
independent of InceptionV1; 'visually far' = below the median. Intervals: bootstrap over cluster pairs.

| layer | mixed units (cluster pairs) | shared | separate | shared, visually far / near | excess over baseline, median (95% CI) | overlap below the class reference |
|---|---|---|---|---|---|---|
| mixed4c | 236 (80) | 23% (15-33%) | 0.8% (0-2%) | 10% / 36% | +0.021 (0.016, 0.025) | 83% |
| mixed4d | 291 (102) | 16% (10-22%) | 1.7% (0.4-3.4%) | 6% / 26% | +0.022 (0.018, 0.030) | 84% |
| mixed4e | 494 (145) | 23% (18-29%) | 0.4% (0-1.1%) | 9% / 38% | +0.050 (0.042, 0.064) | 86% |
| mixed5b | 660 (125) | 1.4% (0-3.9%) | 2.9% (1.6-4.1%) | 0% / 3% | +0.014 (0.009, 0.018) | 96% |

With the cut at 9 (fewer, coarser clusters) the same pattern holds: shared 15 to 19% in mixed4c-4e (5 to 7% for visually
far pairs), 0.6% in mixed5b; separate 0 to 4% everywhere. Related pairs (two sub-clusters inside one cluster, cut 6):
shared 23 to 36% in mixed4c-4e, 5% in mixed5b, separate 0%. By branch, 1x1 units share most (36 to 40% in mixed4c-4e).
The class reference is the overlap between two halves of the classes of one group; most mixed units' two groups overlap
less than that.

Reading:
- Packing beyond what the images impose is rare: at most 4% of mixed units read their two groups through more
  separate inputs than an arbitrary reader of their branch.
- Shared inputs are found for a minority: about a fifth of mixed units in mixed4c-4e, mostly where the two groups look
  alike (the positive control, where both readings expect sharing). Where the groups look different, 5 to 10% share.
  In mixed5b, mixed units almost never read their groups through shared inputs.
- Most mixed units (75 to 85% in mixed4c-4e, 95% in mixed5b) are undetermined: they read each group through the inputs
  that group activates, as any reader of the branch does.
- For the essay: the claim that mixed units come from features shared by their classes holds for a minority of
  units, and mostly for visually similar classes. For visually distant classes, the inputs do not show it. The test is
  blind to a shared variable built inside the unit's own layer from group-specific channels (E0b's last limit), so
  'undetermined' units can still be pose-like or packed; this needs tuning tests (for example on viewpoint labels).
- Limits: the top 100 images only; WordNet clusters define the groups; 32 readers give a rank resolution of 1/32.

## 6. E8 results: relatedness against anticorrelation in a toy, ReLU and SwiGLU, 10 seeds

Setup (`experiments/toy_packing.py`): 8 families of 4 features; related = embedding cosine rho within a family;
exclusive families (members never co-occur), co-occurring families, and exclusion pairs of families; 1 or 2 active
features per sample; x -> Linear(64, m) -> hidden layer of H units (ReLU, or SwiGLU SiLU(W_g z) * (W_v z)) ->
Linear(H, 32) + ReLU, reconstructing the feature values; AdamW 3e-3, 6,000 steps, batch 256; the hidden layer tracked
with `rcdl.attribution` (one group per active-feature set). Cells: rho in {0, 0.6}, m in {8, 32}, H in {16, 32}; 10
seeds each, 160 runs. A unit's poles are the features whose response (the pre-activation for ReLU, the SwiGLU product
for SwiGLU) is above 0.3 times its largest absolute response (positive pole) or below -0.3 times it (negative pole).
Summaries: `experiments/toy_summary.py` (`docs/data/toy_packing_summary.json`) and `experiments/toy_growth.py`.

Pole sharing. For each pair type t, the excess is the share of pairs of type t that share a positive pole (copositive)
or sit on opposite poles of one unit (antipodal), minus the share expected when pole memberships are shuffled across
features. Mean over seeds, with the standard error:

| | ReLU, unrelated anticorrelated, excess antipodal | SwiGLU, same |
|---|---|---|
| rho 0, (m, H) = (8, 16) / (8, 32) / (32, 16) / (32, 32) | +0.10 / -0.01 / +0.07 / +0.10 (SE 0.03) | +0.04 / -0.03 / +0.05 / -0.00 (SE 0.02 to 0.03) |
| rho 0.6, same cells | +0.14 / +0.20 / +0.11 / +0.28 (SE 0.02 to 0.05) | +0.07 / +0.02 / +0.07 / +0.02 (SE 0.01 to 0.04) |

- Anticorrelation alone (unrelated features that never co-occur) puts features on opposite poles of ReLU units above
  chance in 7 of 8 cells, as Toy Models predicts. In SwiGLU units the effect is small (at most +0.07).
- The effect of relatedness is a difference in differences, because the two anticorrelated pair types already differ
  at rho 0 (the exclusion structure differs):
  DiD = [ex_related_anti(0.6) - ex_unrelated_anti(0.6)] - [ex_related_anti(0) - ex_unrelated_anti(0)], seeds paired.
  For antipodality: ReLU -0.19, -0.38, -0.06, -0.24 (SE 0.05 to 0.07); SwiGLU -0.12, -0.14, -0.08, -0.09 (SE 0.02 to
  0.09). Relatedness moves related, anticorrelated pairs away from opposite poles.
- Relatedness moves related pairs onto a shared positive pole: DiD of the copositive excess, related co-occurring
  against unrelated co-occurring pairs, up to +0.20 (SwiGLU, m 8, H 32; SE 0.04); related anticorrelated against
  unrelated anticorrelated, up to +0.24 (ReLU, m 8, H 32; SE 0.03).
- At rho 0.6, related anticorrelated pairs (the toy's "sibling classes") are on opposite poles at or below chance
  (ReLU -0.12 to +0.04, SwiGLU -0.13 to -0.05). In this toy, a sibling pair on two poles of one unit is not what
  relatedness produces. The essay's taxi / ambulance poles need another cause, for example the need to tell the two
  apart, which a reconstruction toy with 1 or 2 active features does not stress.

Growth (from the exact parts; for a unit h and a pole feature j, z = I + sum_w W_w, with W_w the part written by the
samples of writer feature w and I the decayed initial weight's part; a pair sample's part is split equally):

| cell | own / relatives / others, median share of z | spillover entries | of which mainly from a relative (chance) |
|---|---|---|---|
| ReLU rho 0 (4 cells) | -0.18 to 0.07 / about -0.05 / 0.76 to 1.10 | 48% to 68% | at chance (8 / 14.9, 25 / 21.8, 41 / 35.6, 149 / 134.6) |
| ReLU rho 0.6 | -0.12 to 0.29 / -0.12 to -0.03 / 0.53 to 0.95 | 55% to 76% | 35 / 19.6, 40 / 20.4, 78 / 35.9, 283 / 116.2 |
| SwiGLU rho 0 | 0.12 to 0.54 / about -0.03 / 0.25 to 0.68 | 36% to 58% | at chance (10 / 11.5, 10 / 13.7, 26 / 25.3, 61 / 49.5) |
| SwiGLU rho 0.6 | 0.13 to 0.53 / -0.03 to 0.15 / 0.16 to 0.59 | 44% to 72% | 27 / 16.3, 47 / 28.9, 75 / 32.2, 133 / 53.1 |

A spillover entry is a feature that enters a unit's final pole (first checkpoint above 0.3 of the unit's largest
response) with more than half of its pre-activation written by members already in the pole.
- Most features enter a pole by spillover in every cell, so growth by spillover is the normal case in this toy.
- With related inputs, the earlier member that wrote most of the entry is a relative 1.6 to 2.5 times as often as
  chance. At rho 0 the rate is at chance, which checks the method. This shows that relatives write more onto each
  other (similar inputs), not that growth depends on order: the null does not control for static similarity (Sec. 7,
  T4: any account predicts that similar inputs write into each other's responses).
- Most spillover still comes from unrelated members (75% to 83% at rho 0.6), through the shared m-dimensional
  bottleneck. Relatedness is one driver of growth here, and it is not the main one.
- In ReLU units a feature's own samples write little of its pole response (median own share -0.18 to 0.29). In SwiGLU
  units they write more (0.12 to 0.54).

Limits: a reconstruction toy with a trained bottleneck and 32 features. It isolates the two causes; it does not tell
which one dominates in an ImageNet model. E1 to E3 measure the same quantities on ConvNeXt-SwiGLU, with WordNet 3.0
distances between ImageNet classes (`rcdl/imagenet_hierarchy.py`, `docs/data/imagenet_wordnet.npz`) as relatedness.

## 7. E1: the tiny run, and how it compares with the essay's own tiny runs

Run `e1_tiny_adamw` (molab sandbox, 2026-10-16): ConvNeXt-SwiGLU tiny (92,228 parameters without the classifier), the
essay's recipe (all 1,281,167 training images, batch 512, 25,020 updates, mixup on images and labels), AdamW 2e-3 with
weight decay 0.05 and 1,250 warm-up steps then cosine; every pw_gate and pw_value map of all 18 blocks tracked with one
group per class (mixup's two classes as soft groups). Top-1 validation accuracy: 0.08% at step 0, 1.5% at 1,000, 9.9% at
5,000, 15.7% at 12,500, 18.5% at 25,020 (all 50,000 images; earlier steps on the first 10,000). The atlas (exact parts
of every unit's gate and value at its strongest position, 8 probe images per class) sums to the model's own
pre-activations within 1e-5 at all 16 checkpoints. Files: `docs/data/e1/` (log, config, growth summary, car map).

Comparison with the essay's runs. The essay's training map publishes, for its two fresh tiny runs (seed 0 and 1, the
same 1,152 units and the same 16 checkpoint steps), an AUROC per unit and checkpoint for car (ten car classes against all
other sampled classes), ambulance and cab (one car class against the other nine), in four pooling modes
(`assets/training-map/<run>/timeline.bin`). `experiments/convnext_carmap.py` computes the same for our run on a probe
of the same design (all 50 validation images of each car class, 4 of every other class; the essay's own image ids do
not follow the Hugging Face or the file-name order, so its exact images could not be used). Population numbers
(units cannot be matched across runs):

| step | best car AUROC, mean pooling: ours / essay seed 0 / seed 1 | best cab AUROC: ours / seed 0 / seed 1 |
|---|---|---|
| 0 | 0.72 / 0.77 / 0.74 | 0.71 / 0.72 / 0.71 |
| 1,000 | 0.83 / 0.71 / 0.69 | 0.77 / 0.69 / 0.66 |
| 5,000 | 0.87 / 0.83 / 0.86 | 0.78 / 0.75 / 0.76 |
| 12,500 | 0.90 / 0.87 / 0.88 | 0.79 / 0.80 / 0.80 |
| 25,020 | 0.92 / 0.84 / 0.86 | 0.79 / 0.76 / 0.80 |

Car selectivity develops on the same schedule and to about the same level in all three runs (ours a little higher for
car; the essay's optimizer is not published). No unit of the essay's runs reaches car AUROC 0.9 in mean pooling; 0.3%
of ours do at the end. Our run is therefore a fair stand-in for the essay's tiny models in what follows.

### E2 and E3 on the E1 run: order of growth and head start (tests v4, 2026-10-16)

Code: `experiments/convnext_growth_tests.py` (v4). Results: `docs/data/e1/growth_tests_v4.json.gz` (WordNet distance)
and `growth_tests_v4_visual.json.gz` (visual distance); the E4 (Muon) run in `docs/data/e4/`. Two adversarial checks
(six agents each; every number recomputed from the stored results) shaped v3 and v4; their findings are in
`docs/data/e1/review_e2e3_v2_findings.json` and `review_e2e3_v3_findings.json`. Earlier result files are kept.

Notation. For unit n and a class c of its final positive pole (the 20 strongest classes at the end, response above 0.3
of the largest; units whose response is nearly the same for every class are skipped as dead: none in E1, 58 in E4):
- t_c: the stable entry checkpoint, the first checkpoint from which c stays among the unit's 20 strongest classes to
  the end, as an index on the grid 0, 50, 100, 200, 350, 500, 1,000, 2,000, 2,500, 5,000, 7,500, 10,000, 12,500,
  17,500, 20,000, 25,020 (intervals of 50 to 5,020 steps). Entries at the last checkpoint are censored.
- R_c(t): the class mean of the unit's response at its strongest position (8 probe images per class).
- W_k: the part of training class k in the class mean of c's value pre-activation, multiplied by the sign of that value
  at the end. For 51 to 60% of the stage-1 pole classes the value is negative and the response runs through a negative
  gate, so the writes that build it are the negative ones.
- d(c, c'): WordNet 3.0 distance ('relatives': d <= 4; median 10 per class, mean 14.8, 49 classes have none), or the
  symmetric neighbour rank of DINOv2 class means (`docs/data/imagenet_dinov2_classdist.npz`; 'visual relatives': rank
  at most 10, 1 to 16 per class). The essay's growth runs along visual facets (yellow flowers, then green objects, then
  dogs), so both are used. Neither captures facets that co-occur in photos (dogs in flower photos).
- Early blocks: stages 0 and 1 (6 blocks); late blocks: stages 2 and 3 (12 blocks). All statistics below are medians
  or means over (unit, class) pairs pooled within a block, then over blocks as stated.
- Mixup (alpha 0.8) splits each example's gradient between its two classes, so a class keeps about 0.69 of its own
  label signal and the rest goes to random partners; own and relatives' parts are both about 0.65 of what a label
  split would give.

What the atlas shows at the end (T4, value pre-activation only). A pole class's value is the small remainder of large
writes that cancel: sum_k |W_k| / |vbar_c| is 302 to 477 (median per block) in early blocks and 405 to 575 in late
blocks; independent zero-mean writes of the same sizes would give about 30 to 40. (Class parts sum to the integrated
update, so they cancel whenever classes pull in opposite directions; how much of each part is class structure and how
much SGD noise is measured by the random-group split of run E1r, pending.) Medians over pairs, per block:
- own part's percentile among the other 999 classes' parts (chance 0.5): 0.48 to 0.60 in stage 0, 0.30 to 0.41 in
  stage 1 (the own writes oppose the class's value more often than a typical class's do), 0.52 to 0.81 in stage 2
  (0.59 and 0.52 in blocks 0 and 3) and 0.83 to 0.92 in stage 3. Pole classes are selected for a large response, so
  part of this is selection; the statement is descriptive.
- relatives: their share of the positive mass over the share of as many random classes is 0.73 to 0.92 in early
  blocks and 0.93 to 1.92 in late blocks (median 1.25; about 2 only in stage 2 block 8 and stage 3); visual relatives
  0.68 to 0.99 and 0.99 to 2.47 (median 1.55). The sign-based measure agrees: a relative's part ranks at 0.47 to 0.54
  among the non-relatives' parts in early blocks and 0.51 to 0.62 in late blocks (visual 0.42 to 0.55, 0.51 to 0.68).
- So in stage 3 and most of stage 2 the own class and its relatives write more into a pole class's value than other
  classes do; in stage 1 they write less; stage 0 and stage 2 blocks 0 and 3 are near chance. Every account predicts
  that similar inputs write into each other's responses (class k's AdamW part on probe class c is about the sum over
  steps of lr_t m_t^(k) D_t h_c), so this is not evidence for any order of growth.

Order of entry (T1).
- Withdrawn: the earlier reading that this run 'shows no sign that classes join them in branch order'. It rested on the
  nearest-earlier distance (from an entering class to its nearest pole member with an earlier entry checkpoint) and on
  the share of entering classes whose nearest earlier member is a relative. With WordNet distance a planted branch
  order moves both the other way: the distance rises (z = +7.1 to +12.5 per block) and the share falls (0.14 against
  0.23, pooled). The planted order is the observed checkpoints of the unit, sorted and reassigned along a tree grown from the
  strongest class (each next class the one nearest to any class placed). Tied checkpoints cause the reversal, above
  all the large tie at the last checkpoint, which a planted order fills with the pole's outliers. Co-timing alone
  (related classes tied at one checkpoint, clusters in random order) raises the WordNet statistic about three times
  more than a planted order does, so the WordNet nearest-earlier statistic says nothing about order. With visual
  ranks a planted order lowers it and co-timing raises it, so the visual statistic is the one order statistic that
  co-timing cannot produce: late blocks 13% of the planted shift (15% against the tercile shuffle on both sides; z =
  -2.2, one-sided p = 0.01), early blocks 33% (43%; p = 0.002), carried by the three stage-0 blocks (stage 1: +0.26, -0.20, -0.41). The earlier-
  relative share is reversed by ties under both distances.
- Co-timing (Mantel statistic): per unit, the correlation over class pairs of |t_c - t_c'| with d(c, c'), averaged
  over units. It is symmetric in time: it measures whether near classes enter at similar checkpoints, which a branch
  order and relatives entering together both give. Ties lower it but do not reverse it.

| Mantel r (fraction of planted) | WordNet, early | WordNet, late | visual, early | visual, late |
|---|---|---|---|---|
| observed | 0.010 (0.02) | 0.029 (0.06) | 0.015 (0.06) | 0.014 (0.05) |
| null: shuffle within strength terciles | 0.003; z = 1.3, p = 0.09 | 0.006; z = 5.7, p = 0.002 | 0.004; z = 2.6, p = 0.008 | 0.004; z = 3.4, p = 0.004 |
| after the network-wide class clock | 0.014; z = 1.7, p = 0.04 | 0.033; z = 5.8, p = 0.002 | 0.017; z = 2.8, p = 0.006 | 0.015; z = 3.3, p = 0.002 |
| after the stage class clock | 0.013; z = 1.5, p = 0.06 | 0.035; z = 6.0, p = 0.002 | 0.017; z = 2.3, p = 0.016 | 0.016; z = 3.4, p = 0.002 |

  (z and p against the tercile shuffle; p from 500 draws, so 0.002 is the floor. The planted Mantel r is 0.36 to 0.38
  with WordNet, 0.19 to 0.20 with visual ranks.) Controls. Final strength: the tercile shuffle takes 0.2 to 0.3 of the
  observed r; the Mantel of |log R_c - log R_c'| with d is -0.03 to +0.05 (visual -0.04 to
  +0.02); this tests one route of the strength confound only. Class clock: t_c minus the class's mean entry checkpoint
  over the other units of the network, or of the same stage, where it is a pole class (relatives that become learnable
  at the same time everywhere, or within a stage, give co-timing with no spread inside a unit). Its null permutes the
  entry checkpoints in every unit at once and recomputes the clocks, so it carries the noise of the clock estimate.
  The late co-timing survives both clocks.
- Reading: in late blocks, near classes enter a unit at slightly more similar checkpoints than chance, not explained by
  final strength (one route) or by a network-wide or stage-wide class clock, at about 5 to 6% of a planted branch
  order; the one order-specific statistic (visual nearest-earlier) leans toward order at 13 to 15% of the planted shift
  (p = 0.01). In early blocks the effect is weak (WordNet) or of the same size (visual).

Direction (T2). Within units, with log R_c(T) as a covariate: the slope of the checkpoint at which R_c reaches half its
final value on the distance from the unit's strongest final class, among final pole classes, against a planted outward
order (the observed half-rise checkpoints reassigned within the unit in order of distance). Late blocks: 4.6% of the
planted slope with the first crossing (t = 1.7 over blocks, 95% interval -3% to +11%) and 3.1% with the stable
crossing (t = 1.4); visual ranks 3.3% and 1.4%. Early blocks: 5.7% (WordNet, t = 1.4) and -3.8% (visual). No outward
direction is detected; the bound is about a tenth of a strict outward order. Not visible to this test: seeds that do
not stay strongest (the essay's InceptionV1 rapeseed neuron) and offshoots that grow and are then dropped.

Relatives' writes over time (T3). For each pole class whose oriented own part and summed relatives' part are positive
at the end (2,381 of 7,640 pole classes in early blocks, 7,448 of 15,358 in late blocks), the checkpoint at which each
reaches half its final value, by the first crossing and by the stable crossing (the first checkpoint from which it stays
above). Own minus relatives (positive: relatives first), late blocks: +0.19 checkpoint (first; interval above 0 in 9 of
12 blocks) and +0.10 (stable; 1 of 12); early +0.14 and +0.04; visual relatives late +0.06 and +0.05. Against as many
non-relative classes matched one to one on the final oriented part, relatives minus matched is +0.32 (first) and +0.16
(stable) in late blocks, +0.57 and +0.21 in early blocks; visual late +0.19 and +0.05. A sum of parts crosses half its
final value at a time set by its own drift-to-noise ratio, and the matched null equalizes only the final value, so the
two crossing rules still disagree. Reading: relatives' writes may reach half their final value slightly before the
class's own writes, by at most about a fifth of a checkpoint interval in the direct comparison; this is not robust
across rules and is not established.

Head start (T5). Approximate response: SiLU(G_c) V_c from class means of gate and value, strongest position held fixed;
the block input (which the same classes wrote in earlier blocks) is held fixed too. Removing the writes of a class set
changes the response of every stored pole class (40 per unit), so the bar (the 20th strongest response among the
other classes) is recomputed with them; the other 960 classes keep their response, so the bar is only partly
recomputed. A class survives if it stays at or above the bar. The approximation keeps 62 to 81% of entering classes
above the bar with no knockout. Every class writes into every unit from the first step, and one class's part is often
a large fraction of a pole class's value, so removing even one random class's writes drops many classes; that is the
yardstick. Late blocks, at the stable entry checkpoint:
- without its own writes 33% of entering classes survive; without one random non-relative class's writes 52% (own
  lower in 12 of 12 blocks, by 3 to 31 points). Without its relatives' writes 37%; without as many random classes' writes
  50% (relatives lower in 11 of 12). The same in E4 and with visual relatives. So at entry a class's own writes, and its
  relatives', hold up its response more than other classes' writes do.
- entering against incumbents at the same checkpoint, beyond the random baseline: own minus one random class +0.091
  (t = 5.6 over blocks, 11 of 12 positive), relatives minus random +0.058 (t = 6.1, 11 of 12). Entering classes depend
  less on their own writes than incumbents do: own writes keep growing after entry relative to the rest of the response.
- In early blocks own writes count no more than one random class's (0.44 against 0.41), in line with T4.
- Reading: a strict head start, in which a class already sits in the unit's top 20 before its own writes count for more
  than any other class's, is not seen at this checkpoint resolution. The essay's head start is milder: a unit that is
  already partly on for a class is first in line to take its error, so the class's own writes then do the lifting. T5
  is consistent with that and does not test it. 49 classes (4.9%) have no WordNet relative; their relatives' and random
  knockouts are empty (about 2 points on both rows).

Result for E2 and E3 on this run. Late-block units are groups of classes with similar inputs, and in stage 3 and most of
stage 2 their own and related classes write more into them than other classes do (T4). Near classes enter a unit at
slightly more similar times than chance, specifically within the unit, at 5 to 6% of what a strict branch order gives;
the one statistic that separates order from co-timing leans toward order at 13 to 15%; no outward direction is
detected (bound about a tenth of a strict order). Relatives' writes do not clearly lead own writes. At entry a class's
own writes hold up its response more than other classes' writes do, so a strict head start is not seen; the essay's
milder head start is not tested. Each class's net value is the small remainder of large cancelling writes.

How this bears on the essay. The essay shows recursive generalization qualitatively on one or two neurons, chosen as
the messiest or least legible ones, in InceptionV1, and says the improved architecture 'shows the same pattern, but
much less of it'. A pooled whole-model test on a tiny ConvNeXt cannot separate a weak effect in every unit from a strong
effect in a few, and a weak effect here is partly what the essay itself predicts. What this run does show is that the
essay's narrower prediction ('everything on it should trace back to the same local origin ... and it should grow out
from there over training in a particular order') holds here only weakly: co-timing of near classes, little order, and
no detectable outward growth. The essay authors' own early-training notes are cautious in the same direction, on a
different test (zeroing a candidate precursor unit against matched controls). Open: the seed-1 replication and the
random-group control (run E1r), a denser checkpoint grid around entry for T3 and T5, facet co-occurrence distances, a
per-unit analysis of the strongest-growing units, and the medium model.

### E4: Muon against AdamW from the same initialization

Run `e4_tiny_muon` (`docs/data/e4/`): the E1 recipe and seed (the same initial weights and data order; the step-0
numbers below are identical), with Muon (lr 2e-2, momentum 0.95, Nesterov, weight decay 0.05) on every 2-D weight
matrix except the classifier (torch.optim.Muon for the down maps; the attribution's exact Muon, fp32 Newton-Schulz
without TF32, for the tracked gate and value maps) and AdamW (lr 2e-3) on the convolution kernels (stem, downsampling,
depthwise), biases, norms, layer scales and the classifier, as in E1; the tracked biases use AdamW at 2e-3. The run was lost at about step 8,000 with its first sandbox and resumed from the step-6,000 state on a
second one: the data order is rebuilt exactly, the random crops of the resumed epoch are not.

| step | 1,000 | 2,000 | 5,000 | 10,000 | 17,500 | 25,020 (all 50,000 images) |
|---|---|---|---|---|---|---|
| top-1, AdamW (E1) | 1.5% | 4.5% | 9.9% | 13.5% | 17.3% | 18.52% |
| top-1, Muon (E4) | 3.0% | 6.7% | 10.4% | 13.5% | 16.7% | 19.07% |
| top-1, AdamW seed 1 (E1r) | 1.7% | 5.2% | 11.2% | 14.4% | 17.7% | 19.51% |

- Muon learns faster in the first 2,000 steps: its lead at 1,000 and 2,000 steps (3.0% and 6.7% against 1.5-1.7% and
  4.5-5.2%) is larger than the difference between the two AdamW seeds. From 5,000 steps on, Muon lies inside or below
  the AdamW seed range, and its final 19.07% lies between the two AdamW seeds (18.52% and 19.51%), so the final
  difference is not resolved. (E1r also differs from E1 in its seed and in a resume after a lost sandbox; the
  intermediate values use the first 10,000 validation images.)
- Dead units: 58 of 1,152 under Muon (50 in stage 0, 26% of its 192 units; 8 in stages 2 and 3), none under AdamW. A
  dead unit responds almost the same to every class (spread below 0.1% of its largest response).
- Car units: the best car AUROC (mean pooling) is 0.889 at the end against 0.916 for AdamW and 0.840 / 0.856 for the
  essay's two runs; Muon is ahead at 1,000 and 5,000 steps (0.854, 0.894) and behind from 12,500 on; no Muon unit
  reaches 0.9 (0.3% of AdamW units do). Cab and ambulance are about equal.
- Cancellation is stronger: sum_k |W_k| / |vbar| is 579 to 2,312 in early blocks and 631 to 962 in late blocks (AdamW
  302 to 477 and 405 to 575). In stage 1 the own part's percentile falls to 0.13 to 0.62.
- Growth tests (v4), late blocks: co-timing r = 0.025 (7% of planted; z = 5.8 against the tercile shuffle), surviving
  the network clock (0.032, z = 6.7) and the stage clock (0.023, z = 4.4); visual 0.010 (5%). Early blocks: WordNet
  co-timing is much stronger than under AdamW, r = 0.066 (15% of planted, z = 7.6; network clock 0.060, z = 6.5),
  while visual co-timing is about the same (0.012). No outward direction (T2). Head start (T5) as under AdamW: own
  writes hold up an entering class more than one random class's do (survival 0.33 against 0.51, 12 of 12 late blocks);
  entering classes depend less on their own writes than incumbents (+0.098 beyond the one-class baseline, t = 7.2).
- Reading: an optimizer that equalizes the step size over all singular directions of each update did not remove the
  weak co-timing of near classes, and it increased it in early blocks, by WordNet but not by visual distance. It left a
  quarter of the stage-0 units dead, and its car units are slightly less selective. One Muon seed; the AdamW
  seed-to-seed spread from E1r is about 1 point of final accuracy, and E1r's spread on the other measures is pending.
  SGD (the third arm of E4) is not run.

### E1r: a second AdamW seed, and the class split against random groups

Run `e1r_tiny_adamw_s1`: the E1 recipe with seed 1 (other initial weights, data order and crops), with the same updates
also split over 1,000 random groups of class size (`--rand_groups`: the training labels shuffled across examples, a
shadow attribution that does not change training). The sandbox was lost at about step 23,000 and the run resumed from
its last resume state; the data order is rebuilt exactly, the crops of the resumed epoch are not. Top-1 accuracy: 1.7%,
5.2%, 11.2%, 14.4%, 17.7% and 19.51% at 1,000, 2,000, 5,000, 10,000, 17,500 and 25,020 steps (E1: 18.52%).

Class structure against noise. With E_class the summed squared size of the 1,000 class parts of a unit and E_rand the
same for the 1,000 random groups, the between-class share f = 1 - E_rand / E_class (median over units, value map;
range over the 18 blocks) is 0.87 to 0.91 at step 100, 0.97 to 0.99 at step 1,000, 0.994 to 0.998 at step 5,000 and
0.997 to 0.999 at the end. A class's own part on its own probe images, in units of the random groups' spread, has a
block median of 4 to 39 in 11 of 12 late blocks (-2.4 in stage 2 block 0), 4 to 16 in stage 0 and -4 to -2 in
stage 1. The class parts cancel far more than random groups do: sum_k |W_k| / |sum_k W_k| (block medians at the end)
is 286 to 539 for classes and 14 to 23 for random groups. So the per-class parts are class structure, not SGD noise,
and the large cancellation is between classes.

Growth tests v4 on E1r (`growth_tests_v4*.json`), against E1:

| statistic | E1 (seed 0) | E1r (seed 1) |
|---|---|---|
| Mantel r, WordNet, late (z against tercile shuffle) | 0.029 (5.7) | 0.025 (6.4) |
| Mantel r, WordNet, early | 0.010 (1.3) | 0.016 (2.9) |
| Mantel r, visual, late | 0.014 (3.4) | 0.016 (4.8) |
| Mantel r, visual, early | 0.015 (2.6) | 0.008 (2.1) |
| after the network clock, WordNet late | 0.033 (5.8) | 0.028 (6.1) |
| after the stage clock, WordNet late | 0.035 (6.0) | 0.027 (6.0) |
| visual nearest-earlier, late: fraction of planted (z) | 13-15% (-2.2) | 8-9% (-1.5) |
| visual nearest-earlier, early | 33-43% (p = 0.002) | 9-10% (-0.9) |
| T2 outward slope, late, first crossing (95% interval) | 4.6% (-3% to +11%) | 1.9% (-2% to +7%) |
| T3 own minus relatives, late, first / stable | +0.19 / +0.10 | +0.20 / +0.17 |
| T5 survival without own writes against one random class, late | 0.33 against 0.52 (12 of 12) | 0.36 against 0.53 (11 of 12) |
| T5 entering minus incumbents, own minus one random class | +0.091 (t = 5.6) | +0.099 (t = 6.5) |
| T4 gross / net, late blocks | 405 to 575 | 390 to 539 |

(For the visual nearest-earlier statistic a planted order lowers it, so negative z leans toward order.)

Reading: the co-timing of near classes in late blocks, its survival of both class clocks, the absence of an outward
direction, and the head start at entry (T5) replicate across seeds. The order-specific statistic does not: E1's strong
early result (33 to 43% of a planted order, carried by stage 0) is 9% in E1r and not significant, and the late lean
toward order falls from 13-15% (p = 0.01) to 8-9% (not significant). Across the two seeds the evidence for a branch
order beyond co-timing is weak. The final accuracy differs by 1 point between seeds.

The per-class teaching signal on each unit (bias parts; `experiments/convnext_bias_signal.py`, `docs/EXACT_HISTORY.md`
Sec. 1), over the checkpoint intervals from step 1,000 on, value map, pushes oriented so that positive raises the
unit's response on its pole and given in units of the mean push size. Late blocks (median over blocks and intervals,
range): a unit's 20 strongest classes push its bias up, 0.69 (-0.16 to 1.62) in E1 and 0.71 (-0.31 to 1.93) in E1r;
their WordNet relatives outside the top 20 push it up less, 0.15 and 0.17; each of the about 960 other classes pushes it
down, -0.045 and -0.050. In stage 1 the top classes push the other way in both seeds (-0.34 and -0.39). In stage 0 the
seeds differ (median 0.00 in E1, 0.46 in E1r). In E1r, each class pushes 5 to 20 times harder (median 14) than a
random group of the same size, and the class pushes cancel to a net of about 0.1% of their gross sum (block medians
800 to 1,000 for gross / net; random groups 56 to 75, about 1.5%). A class's strongest unit (largest response z-score)
takes 0.8 to 1.9 times a uniform share of that class's pushes over the block's 64 units. Reading: each unit's bias sits
at a balance of strong, class-specific opposing pushes, upward from its pole and the pole's relatives and downward from
the rest; a tug of war that continues through training. These are descriptive; the interval-split tests in
`docs/EXACT_HISTORY.md` ask what the pushes do to individual classes.

### E1f: two clean reruns of E1 (the loader race and run-to-run spread)

E1 was trained before the GPU-decode race in the training loader was fixed (`docs/EXACT_HISTORY.md` Sec. 5). Runs
`e1f_a` and `e1f_b` repeat E1 exactly (flags, seed, data order) with the fixed loader, saving models at E1's checkpoints
and history steps. Both were interrupted by lost sandboxes and resumed from `resume.pt`; their first analyses saw only
the checkpoints present after the resume (e1f_a from 19,000, e1f_b from 2,000) and were redone after restoring every
file from the HF dataset. The growth tests below use E1's 16 checkpoint steps, so they are comparable with E1 and E1r
(on all 24 saved steps the same order holds).

Weight distance d(X, Y) = ||theta_X - theta_Y|| / sqrt(||theta_X|| ||theta_Y||) over all parameters
(`experiments/convnext_rerun_compare.py`):

| step | 100 | 200 | 350 | 1,000 | 2,000 | 5,000 | 12,500 | 23,000 |
|---|---|---|---|---|---|---|---|---|
| d(E1, e1f_a) | 0 | 0.043 | 0.244 | 0.734 | 0.955 | 1.068 | 1.055 | 1.015 |
| d(e1f_a, e1f_b) | 0 | 0.0008 | 0.020 | 0.180 | 0.500 | 0.776 | 0.862 | 0.848 |

- E1 is bit-identical to the clean reruns up to step 100 and leaves them between steps 100 and 200 by 50 times more
  than the reruns differ from each other; afterwards it stays further away (1.01 against 0.85). This fits corrupted
  batches early in E1. The two clean reruns also separate (GPU non-determinism, amplified by training) and end far
  apart in weight space, with the same accuracy: top-1 18.52% (E1), 18.60% (e1f_a), 18.61% (e1f_b).

Growth tests v4 on E1's 16 steps:

| statistic | E1 | E1r (seed 1) | e1f_a | e1f_b |
|---|---|---|---|---|
| WordNet co-timing, late blocks: Mantel r (z against the tercile shuffle) | 0.029 (5.7) | 0.025 (6.4) | 0.021 (4.0) | 0.037 (7.7) |
| the same after the network clock | 0.033 (5.8) | 0.028 (6.1) | 0.021 (4.2) | 0.042 (8.1) |
| WordNet co-timing, early blocks | 0.010 (1.3) | 0.016 (2.9) | 0.015 (2.6) | 0.016 (2.0) |
| visual co-timing, late blocks | 0.014 (3.4) | 0.016 (4.8) | 0.011 (3.3) | 0.014 (4.2) |
| visual nearest-earlier, early blocks: fraction of a planted order | 0.33 | 0.09 | 0.08 | 0.02 |
| visual nearest-earlier, late blocks | 0.13 | 0.08 | 0.23 | 0.07 |
| T2 outward slope, late blocks, fraction of a planted order | 0.046 | 0.019 | 0.027 | -0.020 |

- The co-timing of near classes in late blocks holds in all four runs (z 4.0 to 7.7), with a run-to-run spread (r 0.021
  to 0.037 at the same seed) about as large as the spread between seeds.
- E1's strong early order result (33 to 43% of a planted order, the only order-specific statistic that leaned strongly
  toward order) appears in neither clean rerun of the same seed (8%, 2%) nor in E1r (9%). It is not a property of the
  recipe; whether the corrupted early batches or chance produced it is not decided, and it should not be cited.
- No outward direction in any run. The conclusions of the E2/E3 section stand, without E1's early order statistic.

### Stage 1: a unit's strongest classes teach it down, and its output hurts them

The bias signal above has one exception to the late-block pattern. In stage 1 (three blocks, residual width 8), a
unit's 20 strongest classes write its bias in the direction that lowers its response on them. This section tests that
exception in three runs: E1 and E1r (AdamW, seeds 0 and 1) and E4 (Muon, seed 0). Scripts:
`experiments/convnext_bias_self.py` (exact writes), `convnext_bias_paths.py` (probe gradients, splits, ablation),
`convnext_unit_roles.py` (per-class ablation profiles and write directions). An adversarial review (three reviewers and
a synthesis) checked the scripts and the claims below; its corrections are included.

**1. The exact writes, with each class's own derivative.** In stage 1, 49 to 57% (E1) and 24 to 53% (E1r) of the
(unit, pole class) pairs at step 20,000 lie in the negative lobe of the gate (g < 0 and v < 0, so u = SiLU(g) v > 0),
against 0 to 15% in the late blocks. A single sign per unit is therefore not the sign of du/dv for every class.
`convnext_bias_self.py` uses the local derivative at each class's mean pre-activations instead. For unit n, classes k
and c, and an interval (t1, t2]:

  pole[k, n] = dv[k, n] mean_{c in P_n} SiLU(Gm[n, c]) + dg[k, n] mean_{c in P_n} SiLU'(Gm[n, c]) Vm[n, c]

- dv[k, n], dg[k, n]: class k's exact value and gate bias writes in the interval (no bias decay, so the increments of
  the bias parts are exactly the writes);
- P_n: the unit's 20 strongest classes at t1, by z-score of the class-mean response R (the mean over 8 validation
  images of the class of u* = max over positions of u);
- Gm[n, c], Vm[n, c]: class c's mean gate and value pre-activations at the strongest position, at t1;
- SiLU'(g) = sigma(g) (1 + g (1 - sigma(g))), with sigma the logistic function.

pole[k, n] is the first-order change of the mean response of P_n caused by class k's writes. Values are given in units
of the mean |pole| over the unit's 1,000 classes (a per-unit scale; `convnext_bias_signal.py` uses one scale per block,
so its numbers are on a different scale). Medians over the intervals from step 1,000:

| block | E1 top 20: all / negative-lobe units / positive units | E1r top 20: all / negative-lobe / positive |
|---|---|---|
| stage 1 block 0 | -0.71 / -0.61 / -0.76 | -0.49 / -0.34 / -0.66 |
| stage 1 block 1 | -0.51 / -0.28 / -0.82 | -0.66 / -0.57 / -0.70 |
| stage 1 block 2 | -0.54 / -0.36 / -0.69 | -0.45 / -0.21 / -0.62 |
| late blocks other than stage 2 blocks 0 and 3 (range) | +0.47 to +1.92 | +0.38 to +2.26 |

(A unit is 'negative-lobe' if most of its top 20 classes have g < 0 and v < 0, and 'positive' if most have g > 0 and
v > 0. After step 2,500 units are nearly pure: 96 to 99% of a negative-lobe unit's top 20 classes are in its lobe
(block means over intervals; single intervals down to 91%), and 97 to 100% of a positive unit's (E1, steps 20,000 and
25,020).)

- Not an artefact of the orientation. With the derivative the reversal is stronger than with one sign per unit. A
  reviewer recomputed the derivatives per image on E1 (instead of at class means): the values agree within 0.06 and
  no sign changes.
- Not caused by the negative lobe. It is stronger in positive units than in negative-lobe units, in every block and
  both seeds. In E1, block 2 flips at a negative-lobe share of 0.07; the share rises after the flip, and almost all
  units that change type in the flip intervals go from positive to negative lobe, consistent with the downward pushes
  moving units into the negative lobe. In E1r, block 1 reverses while its negative-lobe share stays at 0.16 to 0.28,
  and in block 2 the share doubled (0.18 to 0.36) in the interval before the flip, so there the order is not resolved.
- Not selection (a reviewer's checks on E1). Choosing the top 20 at t1, at t2, at the end or at the previous
  checkpoint gives similar values (-0.41 to -0.78, the same sign in every block)
  (the probe images that define the top 20 are validation images, independent of the training writes). The whole
  ranking is reversed: in stage 1 a unit's 20 weakest classes push its pole response up (+0.20 to +0.73), and a random
  set of 20 classes gives about 0. 70 to 88% of live units are negative in each interval after step 2,000.
- In E1 the WordNet relatives of the top 20 also push down (-0.17, -0.11, -0.07); in E1r only in block 1 (-0.00,
  -0.12, +0.01). The remaining classes push up a little (+0.03 to +0.04 per class in E1). The pushes nearly cancel
  (net about 0.1% of gross), so the top classes' sign is the finding and the rest's sign follows from the balance.
- A class's own writes lower its own response too (-0.61 to -0.89 in stage 1, same formula with c = k).
- Weights as well as the bias (a reviewer's computation on E1 only; script in the review folder): with each pole
  class's mean input at its strongest position recovered from the atlas, the full writes of the top 20 classes
  (weights and bias) give -0.24 to -0.93 from interval 2,000-2,500 on, about 80% of it through the weights. The bias
  is not a special lever.
- Stage 0 differs by seed (E1: +0.20, -0.12, -0.18; E1r: -0.31, +0.61, +0.42). Stage 2 blocks 0 and 3 are small in E1
  (-0.05, -0.16), but in E1r stage 2 block 0 (width 24) is reversed (-0.35; -0.35 to -0.46 per interval from step
  7,500) and block 3 is +0.07, so the reversal is not confined to the 8-wide stages.

Time course (stage-1 values per interval start). The flip is staggered by block: in E1, block 2 is negative from (500,
1,000], block 1 from (1,000, 2,000], block 0 from (2,000, 2,500] (+0.18 before). Before that, stage 1 is built the
late-block way (+0.09 to +0.53 in the intervals starting at steps 200 and 350). Block 2 flips before the warm-up ends
at step 1,250, so the end of warm-up is not the trigger for that block (in E1r all three blocks flip in (1,000,
2,000], which contains step 1,250).

**2. Where the signal comes from: probe gradients.** The exact writes sum the teaching signal over every position of
every training image. `convnext_bias_paths.py` recomputes it on the probe set (8 clean validation images per class,
hard labels) at a checkpoint and splits it. For an image of class k, with z the logits and L = logsumexp(z) - z[k],
delta[p, n] = dL/du[p, n] is the teaching signal of unit n at position p. The bias gradients are
B_v = sum_p delta SiLU(g) and B_g = sum_p delta SiLU'(g) v (class means), and the probe pole effect is
-(B_v mean_{c in P_n} SiLU(Gm) + B_g mean_{c in P_n} SiLU'(Gm) Vm). A test checks B_v and B_g against autograd's bias
gradients. Splits, each adding up to the whole:

- positions: the strongest position p*, the 'on' positions (u >= 0.5 u*, p* included; about 75 to 245 per image
  for the top classes in stage 1 from step 2,000 in E1 and E1r), and the rest;
- logit terms: the logsumexp(z) term plus the true-class term -z[k]. (In the JSON files of this round the key 'own'
  holds the gradient of +z[k]; the true-class term is -own.)
- LayerNorm scale: every LayerNorm with 1 / sigma held fixed (forward values unchanged), and the rest, which is the
  path through the normalisers' scale.

The probe reproduces the training writes. Per unit, the Spearman correlation over the 1,000 classes between the probe
effect at a checkpoint and the exact writes of the next interval is 0.71 to 0.76 (E1 and E1r) and 0.53 to 0.59 (E4;
its block 1, with near-zero writes, 0.14 to 0.18) in stage 1 from step 2,000 (E4: from 5,000), and 0.66 to 0.76 in
the late blocks. A reviewer also ran, on E1 with 200 classes, a probe that copies training
(mixup pairs with Beta(0.8, 0.8) weights, each image's gradient credited to both classes): same pattern, correlation
0.50 to 0.78, and the term from the partner's label is small (|x| <= 0.10). Mixup is not the cause.

Stage-1 mean of the probe effect on the top 20 classes:

| step | E1 | E1r | E4 (Muon) |
|---|---|---|---|
| 200 | +0.12 | +0.20 | |
| 350 | +0.10 | +0.19 | |
| 1,000 | +0.05 | +0.00 | -0.10 |
| 2,000 | -0.33 | -0.45 | -0.13 |
| 5,000 | -0.30 | -0.40 | -0.42 |
| 12,500 | -0.41 | -0.33 | |
| 20,000 | -0.42 | -0.41 | -0.38 |

Late blocks other than stage 2 blocks 0 and 3: +0.76 to +1.11 (E1, E1r) and +0.85 to +1.06 (E4) from step 1,000.

Splits at step 20,000, stage-1 mean (E1 / E1r / E4):
- positions: p* 0.00 / 0.00 / 0.00 (one position of 784); on -0.19 / -0.15 / -0.13; rest -0.23 / -0.25 / -0.26. Both
  regions push down. The reversal is not a background effect and not an effect of the position where the response
  is read.
- logit terms: true-class term -0.52 / -0.70 / -0.51; logsumexp term +0.10 / +0.29 / +0.13. Raising the unit lowers the
  true class's logit; the competing classes' term works against the reversal. In late blocks the true-class term is
  +1.11 / +1.09 / +1.48.
- LayerNorm scale: fixed scale -0.70 / -0.66 / -0.89; scale path +0.29 / +0.25 / +0.51. The normalisers work against
  the reversal; it sits in the path at fixed scale. In late blocks the scale path is negative (-0.43 / -0.39 / -0.72):
  there a strong unit inflates sigma and pays for it.

**3. What the unit does to its top classes: ablation.** Setting one unit's output to zero at every position
(`convnext_bias_paths.py --ablate`: the unit's 20 strongest classes and 20 random others; `convnext_unit_roles.py`: all
1,000 classes), cross-entropy change at step 20,000:

| | E1 | E1r | E4 |
|---|---|---|---|
| stage 1, top 20 classes (mean over units, then over the three blocks) | -3.2e-3 | -1.3e-3 | -2.6e-3 |
| stage 1, 20 random classes | +6.3e-4 | +5.6e-5 | +4.5e-4 |
| stage 1, share of units whose removal helps the top 20 more than the rest (blocks 0 / 1 / 2) | 0.69 / 0.67 / 0.60 | 0.57 / 0.57 / 0.39 | 0.81 / 0.68 / 0.60 |
| late blocks (stage 2 block 5, stage 3 block 1), same share | 0.17 / 0.16 | 0.34 / 0.14 | 0.24 / 0.13 |
| late blocks other than stage 2 blocks 0 and 3, top 20 classes | +1.9e-2 | +2.2e-2 | +3.1e-2 |

- On average, removing a stage-1 unit lowers the loss of the classes that drive it and raises the loss of others (E1r:
  +5.6e-5, about 0). Per unit, removal lowers the top-20 loss in 53 to 72% of units in E1, 50 to 75% in E4 and 28 to
  56% in E1r (step 20,000; E1r block 2: 28%). In late blocks the unit helps its top classes. At step 1,000 most E1
  stage-1 units still help their top classes (share helping 0.71; E1r 0.42, where the probe signal is already about
  0).
- Not overshoot of the single unit: the cross-entropy of the top classes is nearly linear in a scale factor alpha on
  the unit's output (alpha = 0 against the linear prediction: E1 -3.2e-3 against -4.1e-3, E1r -1.3e-3 against -2.0e-3,
  E4 -2.6e-3 against -2.9e-3), and the share of units that help their top classes but would help them more at smaller
  scale is 0.02 to 0.05 (run means at step 20,000).
- Whom stage-1 units help (the 20 classes whose loss rises most when the unit is removed): they overlap the unit's
  top 20 near chance (0.009 to 0.066; chance 0.02) and are no more often WordNet relatives of the top 20 than random
  classes (0.13 to 0.26 against 0.14 to 0.18). Late units help their top classes (overlap 0.07 to 0.29) and their
  relatives (0.24 to 0.40 against 0.15 to 0.20; visual relatives 0.21 to 0.47 against 0.11 to 0.15).
- Write directions: a unit writes into the residual stream along w_n = gamma * W_down[:, n]. In stage 1, 48 to 94% of
  units (step 20,000) have another unit with |cos| > 0.9 (late blocks 0 to 27%). The review measured on E1 an
  effective rank of the stage-1 write directions of 1.5 to 2.6 (of 8) up to step 2,000 and 2.3 to 4.7 later, with the
  layer scale gamma concentrated on one or two channels. At the late step of part 4 the rank is 2.8 to 4.7 in E1, 4.3
  to 6.4 in E1r and 1.7 to 5.8 in E4, and the largest |gamma| is 82 to 89 times the median in E1 blocks 0 and 1 but
  1.1 to 3.3 times in E1 block 2, E1r and E4 blocks 0 and 2 (E4 block 1: 13): the strong collapse is E1's. Removing
  the whole stream component along a stage-1 unit's write direction costs its top 20 and 20 other classes 0.5 to 1.6
  nats on average (late blocks 0.07 to 0.32; no random-direction control was run): the direction is costly to remove,
  and the unit is a small part of it.

**4. Mechanism tests** (`experiments/convnext_stage1_mech.py`; E1 and E1r at steps 1,000, 2,000 and 12,500, E4 at
1,000, 2,000 and 20,000; probe set as in part 2). The review ranked four candidate mechanisms: M1, the unit and its
same-direction partners in the collapsed write space together write too much along that direction on the pole
classes (the reversal tracks how much same-direction partners also respond on the unit's pole: Spearman -0.65 in E1,
by tercile +0.11, -0.61, -1.56); M4, the unit's write cancels a useful incoming stream component; M5, the rest of the
network reads the unit's direction against its top classes whatever the stream state; M3, the curvature of the SwiGLU
product. Notation: e = w_n / |w_n| is the unit's write direction; at a position, s_in is the block's input along e,
s_part the other units' writes along e, s_own = u_n |w_n| the unit's own write, s_out the block's output along e;
kappa = SiLU(g) pa_v + SiLU'(g) v pa_g is the unit's local derivative toward its pole (pa_v, pa_g as in part 2), and
delta = dL/du_n the teaching signal, so the push on the pole is -kappa delta, summed over positions.

- M3 is excluded at the late step. Splitting the push by the signs of kappa and delta, the reversal comes from
  positions with kappa > 0 (raising the unit raises its pole response), net over both signs of delta (E1 at step
  12,500: -0.50, -0.44, -0.35 for blocks 0 to 2; the delta > 0 part alone, -10.1, -12.3, -7.8, is almost cancelled by
  the delta < 0 part); positions with kappa < 0 add -0.03 to +0.03 in every run and block at the late step, and the
  r-based class offset below persists at matched s_out and u_n. Earlier this does not hold everywhere: at step 2,000
  in E4 the kappa < 0 positions carry the reversal (block 0: -0.20 of -0.24; block 2: -0.29, against +0.11 from
  kappa > 0); elsewhere at steps 1,000 and 2,000 they add -0.08 to +0.07.
- Stream state where the unit is active on its top classes (u >= 0.5 u*; E1 and E1r at 12,500, E4 at 20,000): the
  incoming component along e is negative for 67 to 83% of the (unit, top class) pairs (|kappa|-weighted means over the
  active positions; mean s_in -0.26 to -0.51; E4 block 1, below, aside), the block's other units write the other way
  (s_part +0.10 to +0.31), the unit's own write is small (s_own +0.01 to +0.04), and the output stays negative (s_out
  -0.02 to -0.26). The unit's write reduces the magnitude of the stream along e for 56 to 75% of these pairs. This is
  the geometry of M4, but the other classes show it too at their active positions, slightly weaker (s_in -0.08 to
  -0.44, s_part +0.09 to +0.26, s_in < 0 for 62 to 76% of pairs), and the active positions carry only 20 to 58% of the
  late reversal.
- Scaling the whole stream along e (beta; delta rescaled accordingly): beta = 0.8 deepens the reversal and beta = 1.1
  relieves it in every reversed block at the late step (E1 at 12,500: -0.54 / -0.37 / -0.31 for beta 0.8 / 1 / 1.1 in
  block 0; E1r -0.39 / -0.27 / -0.22; E4 at 20,000 -1.03 / -0.77 / -0.68). At step 2,000 it is flat in E1 block 1 and
  E4 block 2 and goes the other way in E4 block 0. Beta acts on the whole image. In loss terms beta = 1 is close to
  the optimum: at the late step beta = 0.8 raises the cross-entropy of the top 20 and of the other classes in every
  block, and beta = 1.1 raises it for the other classes in all 9 block-steps and for the top 20 in 5 of 9. So the
  reading that the downstream network wants more of the (negative) component along e holds for the push on the unit's
  pole, not for the loss.
- Partners (X2): setting to zero the units of the same block whose write direction has |cos| > 0.9 with w_n and whose
  mean write along e at the unit's active positions has the sign of the unit's own write, and comparing with a random
  set of units with |cos| < 0.5 matched in write size member by member: removing the partners raises the top-20 push
  in all 27 block-steps (by +0.00 to +0.57 against the control; the share of units raised more than by the control is
  0.57 to 0.92). 21 of the 27 have a reversal to relieve, and in 5 the change is 0.006 or less. Removing only the
  partners' writes along e has the same effect; removing only their writes orthogonal to e does nothing (that part is
  at most 44% of each partner's write at |cos| > 0.9). At steps 1,000 and 2,000 the collapsed blocks have too few
  units with |cos| < 0.5, so the control is often smaller than the partner set (E1 block 1 at 2,000: 10 against 23
  units); the comparison along e against orthogonal to e does not depend on it. The effect tracks how much the removed
  partners write along e at the active positions (Spearman +0.90 over the 27 block-steps), not the effective rank
  (-0.22): it is largest in E1 at step 2,000 (block 1 from -0.17 to +0.40, rank 1.9, partner sets of 23 units) and at
  12,500 (+0.21 to +0.23 in blocks 0 and 1, rank 2.8 to 2.9, sets of 8 to 10 units), and +0.00 to +0.12 in the other
  block-steps of rank below 2. In E1r and E4 it is +0.00 to +0.15, with partner sets of 2 to 10 units at the late step
  (E4 block 1: 19).
- Class specificity (M5): comparing the unit's top-20 classes with the other classes at the same stream state along e
  (shared quantile bins of s_out), the top classes still get a negative push (E1 at 12,500: -0.37 to -0.47; E1r -0.28
  to -0.41; E4 at 20,000 block 0 -1.02, block 2 -0.22), while the other classes get -0.07 to +0.10 on average in the
  same bins. The push contains |kappa|, which is larger on the top classes under any mechanism; the readout r, which
  does not, gives the same answer at the late step: r is higher for the top classes in the same s_out bin (+0.12 to
  +0.28 in E1 and E1r; E4 block 0 +1.06, block 2 +0.18) and in the same cell of s_out and u_n (+0.11 to +0.26; E4
  +1.09, +0.21). So the late reversal is not explained by where the stream is along e alone; on the top classes'
  images the rest of the network reads the unit's direction against them, while on the other classes' images r is
  negative in the lowest s_out bins (more e lowers their loss there) and rises with the level. At step 2,000 the r
  offset is present in E1 and E1r block 0 (+0.23, +0.31) but about 0 in E1 blocks 1 and 2, E1r block 2 and E4 blocks 0
  and 2 (-0.10 to +0.03), where the push offset is -0.05 to -0.27.
- E4 stage 1 block 1 at step 20,000 is an exception: its writes are near zero (s_part, s_own about 0) and it shows no
  reversal (+0.03).
- The bullets above describe the late state. At step 2,000, where the reversal forms, E1 blocks 1 and 2 and E1r
  block 2 look partly like M1: the stream along e at the active positions is positive (s_out +0.08 to +0.10; positive
  for 58 to 63% of pairs), the partners' writes outweigh the input (E1 block 1: s_part +0.72 against s_in -0.66),
  partner removal has large effects (+0.14 to +0.57), and the r offset is about 0. But beta = 0.8 does not relieve the
  reversal there as M1 predicts (flat or deeper), and removing opposite-direction partners deepens it only in E1r
  block 2.

**Reading.** In this model, from about step 2,000 on (in E4's ablation from 5,000), a stage-1 unit's strongest classes
teach it down and, on average, are hurt by its output (per unit in 28 to 75% of units), in three runs with two
optimizers (E4 shares E1's initialization, and its biases and layer scales are trained by AdamW as in E1). Its
top-activating classes describe what it responds to, not whom it serves. The effect is not an artefact of orientation,
the negative SiLU lobe, selection, mixup, the read-out position, the normalisers' scale or the competing classes (the
selection, mixup and per-image checks on E1 only), nor of the SwiGLU curvature at the late step; the probe gradients
involve no optimizer. Two findings, which are not additive parts: (i) at a fixed checkpoint, removing the
same-direction partners' writes along e, not orthogonal to it, relieves the reversal in proportion to how much they
write along e; the effect is largest at step 2,000 in E1, where the picture is partly that of M1. (ii) At the late
step, at the same stream state along e, the rest of the network reads the unit's direction against the unit's own top
classes, while on other classes' images, where the stream along e is lowest, it reads the same write as helpful. Part
(ii) is a statement about the downstream readout, not yet an explanation of it. Write-space collapse is neither
necessary nor sufficient: the strongest late reversal (E4 block 0, -0.91) is in a block of effective rank 5.8 with a
nearly flat gamma, E1r stage 2 block 0 (rank 18.6) is reversed, and E1r stage 0 block 2, collapsed at step 2,000 (rank
3.4; 98% of units with a partner at |cos| > 0.9), is not. A replay of E1 from its step-1,000 snapshot with the stage-1
layer scale frozen (`--freeze_gamma 1`; control: the same replay unfrozen; both to step 3,000) tests the E1-type
collapse only. Freezing keeps the largest |gamma| at 9 / 64 / 4 times the median (control: up to 29 / 264 / 4), but the
write directions stay collapsed (effective rank 2.2 to 2.7 / 1.5 to 1.8 / 2.0 to 2.8, against 1.9 to 2.8 / 1.5 to 1.9
/ 2.0 to 3.9), and the flip happens anyway and is not delayed: exact writes of the top 20 in stage 1 block 0 per
interval +0.31, -0.08, -1.08, -1.35 frozen against +0.30, +0.14, -0.76, -0.94 control (blocks 1 and 2 reversed in both
from the first interval); validation accuracy is the same (4.5% and 5.4% at steps 2,000 and 2,500). So the further
concentration of the layer scale after step 1,000 is not needed for the flip; the write-space collapse itself was
already in place at step 1,000 and is not removed by freezing.

Caveat on scope: the residual width 8 in stages 0 and 1 is this repository's choice for the tiny size (Sec. 2: the
essay does not publish widths, and its tiny parameter count does not fix them: 8 / 8 / 24 / 28 is 400 parameters off,
while for example 5 / 16 / 22 / 28 and 3 / 14 / 21 / 35 match it within one). Whether the same happens at the essay's
medium size (widths inferred in Sec. 2 as 48 / 96 / 192 / 384) is untested, as is a regime of higher accuracy (these
runs reach 18.5 to 19.5% top-1).

**Width test (run `e1w16_tiny_s1w16`, 2026-10-17).** The E1 recipe and seed with stage 1's residual width doubled to 16
(`--size tiny_s1w16`: widths 8 / 16 / 24 / 28; 99,180 parameters without the classifier; top-1 21.07% against E1's
18.52%), all maps tracked, analysed with the same scripts (`docs/data/` holds no copy yet; files on the HF results
dataset under `neuron-dev/e1w16_tiny_s1w16/`).

| | stage 0 block 0 | stage 1 block 0 | stage 1 block 1 | stage 1 block 2 |
|---|---|---|---|---|
| exact writes of the top 20, median from step 1,000 (E1) | -0.80 (+0.20) | -0.49 (-0.71) | -0.58 (-0.51) | +0.82 (-0.54) |
| probe pole effect at 20,000: all / true-class term / LayerNorm at fixed sigma / scale path | -0.68 / -1.06 / -1.24 / +0.56 | -0.41 / -0.62 / -0.45 / +0.03 | -0.47 / -0.80 / -0.98 / +0.51 | +0.61 / +0.93 / +1.32 / -0.71 |
| ablation at 20,000, CE change of the top 20 / of 20 others | -7.0e-3 / +2.6e-3 | -2.4e-3 / +7.1e-4 | -2.3e-3 / +2.7e-3 | +8.8e-3 / +1.4e-3 |
| write space at 20,000: effective rank of 16 (8 in stage 0) / share of units with a partner at abs(cos) > 0.9 | | 11.6 / 0.03 | 11.9 / 0.08 | 12.0 / 0.06 |
| stream along e where the unit is active on its top classes: s_in / s_out / share with the M4 geometry | | -0.28 / +0.01 / 0.58 | -0.30 / -0.02 / 0.55 | +0.01 / +1.00 / 0.05 |
| readout offset, top 20 minus rest at matched s_out (r-based) | | +0.16 | +0.23 | -0.71 |

- The reversal survives the doubled width in two of the three stage-1 blocks, with E1's signature: it forms between
  steps 1,000 and 2,000, sits in the true-class term and in the LayerNorm path at fixed sigma, and removing a unit
  lowers the loss of its own top classes. It also appears in stage 0 block 0 (still 8 wide; E1 +0.20, E1r -0.31).
- It does not need a collapsed write space: at the late step the reversed blocks have effective rank 11.6 to 11.9 of 16
  and almost no near-parallel partners. This removes M1 (overshoot by same-direction partners) as a necessary cause.
- Within this one model, the blocks separate cleanly by the stream state where their units are active on their top
  classes. In the reversed blocks the incoming component along e is negative and the unit's write cancels it to about 0
  (55 to 58% of (unit, class) pairs show the M4 geometry), and the rest of the network reads e against the top
  classes (+0.16, +0.23). In the non-reversed block 2 there is no incoming component, the output is positive, M4 holds
  for 5% of pairs and the readout favours the top classes (-0.71). This is the M4 picture (a unit's write cancels a
  component that arrives along its own direction), now without collapse; it is a correlation across three blocks of
  one model, and the causal test (X4 and a replay that removes the incoming component) is open.
- Bearing on the essay's clean residual stream: at twice the width, units still write against a component that
  arrives along their direction on their own top classes and leave it near zero. Whether that is interference being
  undone or ordinary computation is still not decided here; the essay's medium stage 1 is 96 wide.

**X4: which part of the network reads e against the top classes (E1, step 12,500).** Part (ii) of the Reading says that
at a matched stream state the readout r = e . dCE/dx_out is higher on a stage-1 unit's top classes than on the rest.
X4 (`experiments/convnext_stage1_mech.py --x4_blocks`, reviewed by three lenses; all six findings fixed) splits r
exactly. With x_j the stream after stage-1 block j (the unit's block is b), g_j = dL/dx_j, D1 and D2 the stage-2 and
stage-3 downsamples (LayerNorm and a 2x2 stride-2 convolution) and h, zeta the stage-2 and stage-3 stream gradients,
the residual recursion g_{j-1} = g_j + J_{f_j}^T g_j unrolls to

  g_b = sum_{j>b} (g_{j-1} - g_j) + D1^T [ sum_k (h_{k-1} - h_k) + D2^T ( sum_l (zeta_{l-1} - zeta_l) + zeta_L ) ]

where each term collects the paths whose first block after x_b is that block (pure skips before it), and the last term
(the head) is the path through every skip to the pooled final norm and the classifier W_head. r_R = e . term_R, so the
readers add up to r (checked to 1e-8; finite differences per reader in float64). Each reader's top-20-minus-rest
offset at matched s_out uses X1's bins and weights, so the offsets add up to X1's. Three additions from the review: the
single path of each block (that block, then pure skips to the head); a weights-only head term q[k, n] = e_n .
(-W_head[k] pulled back through the final norm, D2 and D1 at fixed 1/sigma, averaged over the 2x2 phases), whose
top-20-minus-rest value needs no data; and, for specificity, the d-dimensional class-difference vector O_n of the
whole stream gradient (the same bins and weights), against which the offset along e is e_n . O_n.

| stage 1 | block 0 | block 1 | block 2 |
|---|---|---|---|
| offset along e, CE (X1's role_offset_r_by_sout) | +0.283 | +0.189 | +0.207 |
| first readers: head / stage-1 later blocks / stage 2 / stage 3 | +0.549 / -0.104 / -0.218 / +0.056 | +0.149 / -0.016 / +0.023 / +0.032 | +0.269 / 0 / -0.074 / +0.012 |
| weights-only head term (top 20 minus rest, in sd over classes) | +0.79 | +0.34 | +0.50 |
| Spearman over units, weights-only term against the head reader's offset | 0.98 | 0.86 | 0.89 |
| head offset left where the weights-only term is 0 | +0.016 | -0.007 | +0.008 |
| cos(e_n, O_n), mean over units (null sd of the offset along a random direction) | 0.30 (0.036) | 0.19 (0.043) | 0.18 (0.049) |
| share of units with e_n . O_n > 0 | 0.77 | 0.69 | 0.76 |

- The readout against the top classes comes mostly from the head path, and the head path's class dependence is the
  classifier's weight geometry: the rows of W_head for a unit's top classes, carried back through the final norm and
  the two downsamples, point against the unit's write direction. This term needs no images and predicts the head
  reader's offset unit by unit (Spearman 0.86 to 0.98); what is left after it is about 0.
- The later stage-1 blocks and stage 2 enter with the opposite sign (block 0: -0.10 and -0.22). Their single paths to
  the head (-0.10, -0.08) and their per-unit correlation with the weights-only term (Spearman -0.68, -0.81) show that
  they mostly cancel part of e's image on the skip path, which the classifier then reads; they do not read the class
  themselves.
- e is specific: the stream gradient's class difference has cosine 0.18 to 0.30 with e (random directions about 0), and
  e points with it in 69 to 77% of units. (Under the fixed-sigma true-class readout this cosine is about 0 in blocks 0
  and 1, so the specificity comes with the full loss and the normalisers' scale.)
- Together with the stream geometry of the width test (in reversed blocks the incoming stream along e is negative on
  the unit's top classes, and the unit's write cancels it), this gives one account of the reversal: in this model's
  stage 1, a low component along e is evidence for the unit's top classes that the classifier reads directly through
  the skip path (W_head of those classes points against e), and the unit's own output removes that evidence on exactly
  those images; so raising the unit lowers their logits, and their gradients push it down. Why training builds a
  classifier that reads a stage-1 direction in this way is the next question.

X4 on the other runs (CPU, the same 8,000-image probe; `docs/data/x4/`), CE readout:

| run, step, stage-1 block | exact writes of the top 20 (Sec. 1) | offset along e | head reader | weights-only head term | Spearman (weights-only, head) | cos(e_n, O_n) |
|---|---|---|---|---|---|---|
| E1r 12,500, block 0 / 1 / 2 | reversed / reversed / reversed | +0.13 / +0.18 / +0.12 | +0.40 / +0.22 / +0.18 | +0.60 / +0.30 / +0.33 | 0.92 / 0.94 / 0.95 | 0.17 / 0.14 / 0.13 |
| E4 20,000, block 0 / 1 / 2 | reversed / none (near-zero writes) / reversed | +1.06 / -0.05 / +0.18 | +1.14 / -0.07 / +0.24 | +1.20 / -0.06 / +0.20 | 0.87 / 0.92 / 0.91 | 0.53 / -0.04 / 0.15 |
| width 16, 20,000, block 0 / 1 / 2 | reversed / reversed / normal (+0.82) | +0.16 / +0.23 / -0.71 | +0.24 / +0.07 / -0.76 | +0.28 / +0.14 / -0.95 | 0.92 / 0.94 / 0.77 | 0.11 / 0.10 / -0.38 |

- In all 12 stage-1 blocks of the four runs, the sign of the weights-only term (the classifier rows of a unit's top
  classes, carried back through the final norm and the two downsamples, against the unit's write direction) matches
  whether the block is reversed: positive in the 9 reversed blocks, about 0 in E4's block 1, which shows no reversal,
  and clearly negative in the width-16 run's normal block 2, where the readout favours the top classes. The term needs
  no images and predicts the head reader unit by unit (Spearman 0.77 to 0.98).
- In the width-16 run's block 1 the head reader is small (+0.07) and stage 2 carries more of the offset (+0.12); the
  weights-only term still has the reversal's sign.
- So whether a stage-1 unit's strongest classes teach it down is set, in these runs, by how the final classifier reads
  the unit's direction through the skip path. This moves the open question from stage 1 to the classifier: why do the
  rows of W_head for the classes that most excite a stage-1 unit point against that unit's write direction? Note for the essay's clean-stream picture: a low-level
  stage-1 component that the classifier reads directly through the skips, and that the stage's own units undo on the
  very classes it signals, is the kind of stream accounting the essay says it does not see.

Bearing on the essay. (a) Its account of growth gives a unit already on for a class the first claim on that class's
error (a head start). The late blocks are consistent with it (their top classes push the unit up; E3 did not test the
essay's milder head start); in stage 1 after about step 2,000 the strongest classes act as a brake and the bulk of the
classes holds the unit's response up. Whether a stage-1 unit still has the first claim on the error of new, related
classes is not measured here. (b) Its picture of a clean residual stream ("we don't see the stream accumulating
interference only to undo it later") is not tested at this width. In this model's stage 1 many units share few write
directions (effective rank 1.7 to 6.4 of 8 at the late step, 1.5 to 2.6 in E1 up to step 2,000; 48 to 94% of units
have a partner at |cos| > 0.9), and where a unit is active the block's writes along its direction run against the
incoming component; whether that is interference being undone or ordinary computation is not decided here. The essay's
own test narrowed the medium model's stream four- and eight-fold (to 46 dimensions in the last block) and found that
what leaks in is 'the same kind of thing'. (The cancellation of class pushes to about 0.1% of their gross holds in
every block, late ones included, and is a balance of training gradients, so it is not evidence here.) (c) The essay's
view of neurons as control flow ('less about what it sees than about what it does to the flow coming in') fits this
finding: in stage 1 the top-activating classes describe what a unit sees, while on average its output hurts them, so
reading such a unit's function from its top activations would mislead.

### E7: the last layer crushed to 13 units

Run `e7_tiny_crush13`: the E1 recipe and seed with the last block's SwiGLU narrowed to 13 units and its skip connection
removed (`--size tiny_crush13`), so that everything the head reads passes through 13 units: 1,000 classes on 13 units,
where packing is forced. Top-1 accuracy 12.47% (E1: 18.52%).

Mixed units (the E0d rule: the top-100 validation images of a direction form two WordNet clusters at cut 7 with at
least 15 images each; Sec. 5), share of mixed directions in the last three blocks:

| block | E1 units / random dense / random pairs | E7 units / random dense / random pairs |
|---|---|---|
| stage 3 block 0 (64 units) | 0.41 / 0.38 / 0.45 | 0.48 / 0.45 / 0.42 |
| stage 3 block 1 (64 units) | 0.42 / 0.44 / 0.61 | 0.44 / 0.48 / 0.50 |
| stage 3 block 2 (E1 64, E7 13 units) | 0.36 / 0.50 / 0.36 | 0.62 (8 of 13) / 0.55 / 0.61 |

In the crushed block, units, random dense directions and random pairs of units are mixed about equally often (55 to
62%); the units are not a privileged basis there, as in E1's late blocks. (13 units give a share with a wide interval.)

Growth tests v4 on E7, late blocks: co-timing r = 0.027 (5% of planted, z = 5.2), surviving the network clock (z = 4.9)
and the stage clock (z = 5.3); visual co-timing 0.021 (7%, z = 5.1); visual nearest-earlier 14% of planted (z = -1.9);
no outward direction (T2 1.5%, interval -5% to +6%); head start (T5) as in E1 and E1r: survival without own writes 0.35
against 0.49 for one random class (11 of 12 blocks), entering minus incumbents +0.088 (t = 6.0). Early blocks: WordNet
T2 is 11% of a planted outward order (t = 2.9, interval 4% to 16%), the only outward-order signal in any run so far
(visual 1%, not significant); it is one seed and one test among many.

Reading: crushing the last layer lowers accuracy by 6 points and makes its few units as mixed as random directions,
without changing the growth statistics of the other blocks. Whether the crushed units mix related classes (packing by
relatedness, which the essay expects) or unrelated ones is the question of the rare-class experiment
(`docs/EXACT_HISTORY.md` Sec. 4.3).

### Rare classes, seed 0: own routes, related carriers and an uninformative noise control (2026-10-17)

Question. The essay says that a rare class does not get packed into unrelated units but rides machinery it is related
to: "A rare fact barely moves the loss, so it can't support a niche of its own. What it can do is ride existing
machinery as a small route between circuits it's related to [...] Those circuits feed it, and shelter it from unrelated
gradients. The prediction is testable: the neurons and heads that carry a rare fact should be related to what the fact
is about, not unrelated ones chosen because they rarely collide" (appendix 'Does Scale Help or Hurt Superposition?').
It adds that this held "in every experiment we've run" ('Do Weak Activations Matter?') and that "memorization tends to
want to sit as close to the output as possible" ('Rare Things and Memorization'). The pre-registered protocol (v2,
`docs/EXACT_HISTORY.md` Sec. 4.3) turns these into four claims: C2a (feeding), C2b (carriers), C1 (universality) and
C4 (location). Three parts of the quotes are not tested here. "Shelter" would be measured by S4, which was not run. The
essay's heads are out of scope. And C4 borrows the phrase "nearly all of it in the last three blocks", which the essay
uses for the umbrella-to-bee relabel in its 7.06M-parameter model, not for rare classes.

Setup. There are three tiny ConvNeXt-SwiGLU runs, all with the E1 recipe (AdamW, 25,020 steps, seed 0, E1's data
order). Each run tracks the 36 unit maps and the head under a class split and a fine-group split.
- Runs: A0 makes set A rare, B0 makes set B rare, and C0 keeps every class common.
- Classes: each set has 64 selected classes, 32 dense and 32 isolated. Per stratum, 4 classes get 0 training images, 14
  get 10 and 14 get 100. Each set also has 4 relabel and 4 noise controls.
- Conditions: a class is rare in its own set's run and common in C0 and in the other set's run.
- Primary sample: the 112 selected classes at levels 10 and 100, each measured in its rare run at step 25,020.
- Code: `experiments/convnext_rare.py` (subcommand `measure`, and `compare` for the statistics),
  `experiments/rare_common.py` and `experiments/rare_gonogo.py`.

Four review lenses (primaries, secondaries, controls, the essay) recomputed the numbers from the per-class files, and a
skeptic checked each claim. Below are only the claims that held, or their corrections. Seed 1 (A1, B1, C1) has not been
run. The floors that need E1 and E1r were not passed to `compare`. So nothing here is replicated.

Symbols. Everything is at step 25,020, on V_c, the class's 50 validation images.
- Group sets:
  - Near10(c): the 10 nearest classes to c by DINOv2 class-mean distance. All ten are common in every run.
  - Own_m(c): c's own m training images, m = 10 or 100.
  - N_own(c): Own_m of the 13 other classes with the same set, level and stratum. They are rare in the same run and
    unrelated to c.
  - Matched random10: 10 classes matched to Near10 on image count and part norm, with 1,000 draws.
- K(c): c's 8 carrier units, the units with the largest specific mean-ablation effect on zeta_c, c's centered logit.
  They are chosen without attribution. sigma_n: unit n's response scale (std over classes of its class-mean response).
- L_S(n): the exact local knockout of group set S in unit n. It is the drop in the oriented response u = SiLU(g) v
  when S's summed parts are removed from both the gate g and the value v, at fixed block inputs, divided by sigma_n.
- F(c): median over n in K(c) of L_Near10(n) minus the median L over matched random10 draws, in sigma_n units.
  F > 0 means relatives build c's carriers; F < 0 means they hold c's carriers down. F_pct is F's percentile among
  the 1,000 draws.
- O(c): the same for Own_m against N_own. O > 0 means c's own images build its carriers.
- PI(c) = (F - O) / (|F| + |O|), between -1 and 1. PI > 0 means relatives write more of the route than c's own images.
  PI is 'defined' when |F| + |O| is above the 95th percentile of the same sum with Near10 replaced by a random10 draw
  and Own_m by an N_own group. Otherwise the class has 'no route'.
- Packing signature: O above the 99th percentile of its N_own null and F at or below its null median. A
  pseudo-signature is the same test with Own_m replaced by an N_own class.
- lambda_3(c): the stage-3 share of the own effect.
  - X^s is the drop in mean zeta_c when Own_m's parts are excised exactly from the unit maps of stage s, minus the
    N_own median.
  - lambda_3 = X^3+ / sum_s X^s+, where X+ = max(0, X).
  - Stage 3 is the last 3 of 18 blocks, 192 of 1,152 units. The head is reported separately (head_share).
- R_own(c): the relatedness of units, averaged with own-write weights.
  - rel(n, c) is the fraction of unit n's 20 strongest classes that lie in Near10(c); chance is 0.010.
  - w_n is the first-order effect of Own_m's parts in unit n on zeta_c, minus the N_own median, floored at 0.
  - R_own = sum_n w_n rel / sum_n w_n.
- R_flow: the same average weighted by c's own gradient flow over training. It shows where plain gradient flow would
  put c's writes.
- R_mis: R_own computed with the Near10 sets of 20 unrelated selected classes. It shows the relatedness expected by
  chance.
- Delta_flow = R_own - R_flow. The protocol's tolerance is delta = 0.2 x median(R_flow - R_mis).
- auroc_far: the AUROC of zeta_c on V_c against probe images of far classes (visual rank > 200 and WordNet distance
  > 6). auroc_near is the same against the Near10 classes.
- D0: the same-seed floor for rare - C0 contrasts, defined as the other set's run minus C0 (c is common in both).

**Learning.** Validation top-1 on 50,000 images is 17.85% for A0, 18.12% for B0 and 18.44% for C0 (E1: 18.52%). Both
rare runs passed the pre-registered go/no-go at step 12,500:
- the median auroc_far of the level-100 rare classes was 0.947 in A0 and 0.943 in B0 (n=28 each);
- E_Own_m was above its N_own 95th percentile in 28 of 28 classes.

At step 25,020 the rare classes are told apart from far classes but are almost never predicted. The table splits by
level, which is post hoc. The pooled retention, a pre-registered secondary, is rare - C0 auroc_far -0.044
[-0.069, -0.033] (n=112), well beyond its floor (Wilcoxon z = -9.16).

| level (n=56 each) | auroc_far, rare / C0 | rare - C0, paired median [95% CI] | D0 | top-1 on V_c, rare / C0 (medians) |
|---|---|---|---|---|
| 10 | 0.828 / 0.972 | -0.126 [-0.178, -0.100] | +0.002 | 0 / 0.10 |
| 100 | 0.953 / 0.973 | -0.014 [-0.021, -0.011] | +0.0004 | 0 / 0.16 |

- Median top-5 is 0 for the rare classes at both levels.
- No level-10 class gets a single top-1 hit; the maximum at level 100 is 0.30.
- fit_T is the top-1 rate of the training label on centre crops of the class's own training images. At step 12,500
  it was 0 in every rare cell. It is not computed at 25,020.

So these runs test classes that are learned as directions but not memorized as predictions.

**Primaries (pre-registered; seed 0 only).**

| primary | statistic | n | median [95% CI] | p (Holm) | essay predicts | packing predicts |
|---|---|---|---|---|---|---|
| P1, feeding (C2a) | PI, defined classes only | 23 of 112 (89 no route) | -0.99987 [-1.0, -0.99945] | < 1e-4 (< 1e-4) | PI > 0, F > 0, O smaller than F | PI < 0, F about 0 |
| P2, depth (C4) | lambda_3(rare) - lambda_3(C0, whole class, exact) | 110 | -0.032 [-0.053, +0.005] | 0.107 (0.107) | > 0, with lambda_3(rare) > 0.5 | <= 0 |
| P3, carriers (C2b) | Delta_flow | 106 (6 excluded) | +0.0043 [+0.0036, +0.0051] | < 1e-4 (< 1e-4) | >= -delta (delta = 8.6e-5), R_own > R_mis | < 0 |
| C1, universality | signatures against pseudo-signatures | 112 | 18 against 1 | 1.5e-12 (binomial) | no excess | excess |

The compare stage's decision:
- support for the essay: none;
- counts against it: none;
- pending seed 1: two candidates, "median PI < 0" and "excess of packing signatures";
- controls_pass: false, with the note "positive controls failed: primaries are unvalidated".

All 23 defined PIs are negative, so the bootstrap cannot produce any other result for P1. Recomputing from the three
measure files in the compare stage's class order (A0's rare records, then B0's) reproduces every number in the compare
file bit for bit, intervals and p values included. A different class order moves the bootstrap intervals and p values
of the contrasts slightly: rare - C0 PI gives p = 0.954 against 0.935.

**P1: the negative median PI is the sign of F in the large-F tail, and it is not about rarity.**
- PI is close to the sign of F because the two terms differ in scale: Near10 is ten whole common classes, while Own_m
  is 10 or 100 images.
  - Over the 112 classes, |O| < |F| in 109. The median |O|/|F| is 0.015: 0.0046 at level 10 and 0.068 at level 100.
  - sign(PI) = sign(F) in 110 of 112.
  - Even c's whole class in C0 gives an O_whole (median 1.99, n=112) below |F| in 74 of 112.
  - So the essay's condition |O| < |F| is close to guaranteed by the measure and tells us little.
- In all 23 defined classes, F is large and negative and O is small.
  - F runs from -157.2 to -10.9 (median -25.9); F_pct is at most 0.039 (median 0.004).
  - |O| is at most 0.49 (median 0.027), and |O|/|F| at most 0.034.
  - With O set to 0, all 23 stay defined at PI = -1. For the 9 classes with F <= 0 <= O, PI is exactly -1 whatever O
    is. In the other 14, O raises PI by at most 0.016.
  - So the median of -0.99987 means this: where F is extreme, removing the relatives' parts raises c's carrier
    response more than removing random classes' parts does. It says nothing about own writes. It fits neither
    prediction: the essay predicts F > 0, and packing predicts F about 0.
- In these data, 'defined' works as a one-sided test.
  - The rule treats both signs of F alike. Yet in the primary sample the 23 defined classes are exactly the 23 with
    F_pct <= 0.05.
  - None of the 13 classes with F_pct >= 0.95 is defined; their F is 0.28 to 0.91 of the threshold. For example, class
    877 has F = +20.0, above all 1,000 draws, against a threshold of 27.4.
  - The likely cause is a left-skewed F null: the null median is below 0 in 104 of 112 classes (median -0.37). The
    draws are not saved, so this is inferred.
  - Positive F can pass where the threshold is low: class 911 in the other-set run (F = +11.4, threshold 9.2) and noise
    class 366 (F = +13.0, threshold 7.2).
- The excess of negative extremes is in the data; it is not an artefact.
  - 23 classes lie in the bottom 5% of their null against 5.6 expected (binomial p = 7e-9); 13 lie in the top 5%
    (p = 0.004).
  - Post hoc symmetric gates keep the median negative but less certain:
    - F_pct <= 0.025 or >= 0.975: n=26, 19 negative, median -0.999 [-1, -0.988], p = 0.008;
    - 5% and 95%: n=36, median -0.997 [-1.0, +0.71], p = 0.06.
  - Over all 112 classes with no gate, the median raw PI is -0.970 [-0.996, +0.220] (p = 0.14). Counting 'no route'
    as 0 gives a median of 0.0.
  - The protocol's statistic is "median PI over the 112 rare classes", and it does not say how 'no route' enters.
    Dropping those classes (`convnext_rare.py:1276`) is the code's choice, and it is what makes P1 significant.
- 'No route' means F sits inside its null; it does not mean the carriers are weak.
  - All 112 classes have 8 carriers with a positive specific effect (minimum 0.039 centered logits).
  - The mean carrier effect is 0.180 in defined classes and 0.157 in undefined ones (Mann-Whitney p = 0.15, n = 23 and
    89).
  - In undefined classes, |F| + |O| is a median 3.60 against a threshold of 15.0, and F_pct has median 0.63.
  - Definedness is set mostly by |F| (25.9 against 3.38, AUROC 0.94), only modestly by the threshold (10.5 against
    15.0, AUROC 0.68), and not by how well the class is learned (auroc_far 0.885 against 0.930, p = 0.62).
- The same pattern holds where the classes are common.
  - C0: 24 of 112 defined, all with F < 0, median PI -0.998.
  - Other-set run: 26 defined, 25 with F < 0, median -0.997.
  - Classes defined with negative PI, rare against C0: 18 against 19 discordant (McNemar p = 1).
  - The pre-registered secondary PI(rare) - PI(C0) has median -0.00016 [-0.0166, +0.0003], p = 0.935, n=112. For F it
    is -0.46 [-2.79, +2.28]; for O, -0.001 (p = 0.94).
  - That secondary's floor test (rare against the other-set run, Wilcoxon, n=103) gives z = -2.14, p = 0.033,
    uncorrected. The differences it tests cluster at -2, 0 and +2.
  - Definedness is not a stable property of a class. The rare and C0 defined sets share 5 classes (4.9 expected by
    chance), and per-class F does not carry over between runs (Spearman -0.006, rare against C0).
- The single-pathway and linear versions point the other way over all 112 classes. The protocol reports them next to
  P1; reading them as tests is post hoc.

  | version | rare, median [95% CI] | C0 |
  |---|---|---|
  | gate-only F | +1.61 [0.91, 2.30] | +1.73 |
  | value-only F | +6.06 [2.24, 9.71] | +5.51 |
  | linear, F_lin_pct - 0.5 | +0.298 [0.19, 0.38] | +0.176 |
  | joint F | -0.71 [-2.84, +0.51] | -0.38 |

  - The protocol's "Near10 opens the gate on c's images more than own does" holds over the 112 (F_gate - O_gate +1.43
    [0.83, 2.25]). It holds just as much in C0 (+1.43), and set size favours it.
  - These versions do not explain P1. Among the 23 defined classes, the median gate-only F is -1.81, the median
    value-only F -5.83 and the median F_lin_pct 0.49.
  - 13 of the 23 have both single pathways negative, and only 6 have both positive. Dropping those 6 leaves the median
    PI at -0.9995 (n=17).
  - In the 13 with both pathways negative, the median F_gate + F_value is -21.4 and the median joint F is -35.9. The
    joint knockout deepens a negative sum; it does not flip a sign. A SwiGLU double-negative artefact therefore does
    not account for P1.

**Universality (C1).** By the pre-registered test, 18 of 112 classes carry a packing signature against 1 with a
pseudo-signature (one-sided binomial p = 1.46e-12). Three caveats make this weak evidence of packing that is specific
to rare classes.
- The test treats the pseudo rate (1 + 1) / (112 + 2) as known (`convnext_rare.py:1347`). An exact paired McNemar test
  on the same classes (18 against 1 discordant, none with both) gives one-sided p = 3.8e-5.
- The O criterion is looser than it reads.
  - The interpolated 99th percentile of 13 null values is effectively their maximum. So the criterion means "O above
    all 13 N_own classes", which happens by chance about 1 time in 14, not 1 in 100.
  - 48 of 112 classes meet it, 42 of them 'no route'.
  - The pseudo-own is not exchangeable with Own_m: carriers are chosen on c's own validation images, and Own_m are
    images of c.
  - The pseudo-own is the lowest-index N_own class of its cell, so only 16 distinct pseudo groups serve the 112
    classes.
- The excess also appears where the classes are common.
  - C0: 11 signatures against 5 pseudo-signatures (p = 0.034).
  - Other-set run: 11 against 2 (p = 0.0002).
  - Rare against C0 on the same classes: 18 against 11 (McNemar p = 0.21, post hoc).
  - A likely reading is that own images carry their own label on c's carriers, which an N_own null cannot match.

**P2: the pre-registered reference shows no deepening, and the sign depends on the reference.**
- The pre-registered result is -0.032 [-0.053, +0.005], p = 0.107, n=110 (2 classes undefined).
  - The median lambda_3(rare) is 0.426 (bootstrap 95% CI [0.406, 0.445], post hoc), and 28 of 110 classes (25%) are
    above 0.5.
  - So within the unit maps, the essay's "nearly all of it in the last three blocks" (lambda_3 > 0.5) fails, mostly
    at level 100: 0.395 there, with 3 of 56 above 0.5, against 0.484 at level 10, with 25 of 54.
  - Median stage shares for stages 0 to 3 are 0.036, 0.024, 0.497 and 0.426.
  - Depth still matters: stage 3's share is above its unit fraction (1/6) in 110 of 110 classes, and stages 2 and 3
    together hold 0.93.
- The head holds most of the rare own effect: the median head_share is 0.912 (n=110), against 0.563 for the same
  images in C0 (n=112).
  - The protocol makes the head secondary because AdamW enlarges a rare head row's steps. The rare/C0 ratio of the
    head row's effective step has median 6.1: 11.2 at level 10 and 3.4 at level 100.
  - That ratio tracks the head share across classes (Spearman 0.76, n=110).
  - These runs cannot tell "memorization sits at the output" apart from AdamW's per-row normalization. The optional
    Muon arm can.
- The choice of reference decides the sign.
  - The pre-registered sensitivity check, with C0's Own_m (the same images) as reference, gives +0.062
    [+0.037, +0.077], n=112.
  - In C0, Own_m and the whole class have the same profile to first order: lambda_3 is 0.353 for both, r = 0.985.
    Exact Own_m excision agrees (0.365).
  - Only exact excision of the whole common class moves lambda_3 up, to 0.450: exact minus first order is +0.090
    [0.078, 0.113].
  - That excision is far from linear. Its exact over first-order stage effects are 1.31, 1.34, 0.18 and 0.35 for
    stages 0 to 3. This breaks the protocol's assumption that "shares do not depend on scale", in the large-excision
    regime its Risk 2 warns about.
  - The protocol says to compute the whole-class reference "both by exact excision and to first order", but `compare`
    computes only the exact one (`convnext_rare.py:1281`). With the first-order reference, P2 is +0.069
    [+0.045, +0.082], n=110 (computed post hoc). That nearly matches the Own_m contrast, as it must, given the
    first-order identity above.
- The positive contrast is a level-10 effect (a post hoc split).
  - Rare - C0 on Own_m: +0.135 [0.083, 0.151] at level 10, with 48 of 56 positive, beyond the floor (Wilcoxon of
    D - D0, z = +5.16, p = 2.5e-7).
  - At level 100 it is +0.006 [-0.009, +0.037] (z = -0.95, p = 0.34).
  - The floor D0 is itself positive: +0.027 [0.005, 0.036], n=112.
  - The exact P2's negative median comes from level 100, -0.067 [-0.090, -0.038] (n=56), and that does not exceed its
    own floor (p = 0.34).
- At level 10, part of the gain comes from the null (post hoc).
  - In the rare run, excising another rare level-10 class's Own_m from stage 3 lowers zeta_c by a median 0.30
    (quartiles 0.26 to 0.34, positive for all 56 classes). In C0 the figure is -0.004.
  - Raw stage shares would give lambda_3 = 0.583 against 0.346 in C0. But the raw stage-3 gain is matched by the
    stage-3 null gain (+0.309 against +0.310).
  - The null-subtracted gain of +0.135 comes from stage 2. Its null rises by 0.127 while its raw own effect falls by
    only 0.039.
  - Why unrelated rare classes' images lower c's logit through stage 3 is an open question.
- The P2 floor statistic reduces to lambda_3(rare) - lambda_3(other-set run), whatever the reference.
  - It gives z = +3.61, p = 3.1e-4, n=110, which points the opposite way from P2's negative median.
  - The decision code checks only its p value (`convnext_rare.py:1433`). Had P2 been Holm-significant, a negative P2
    would have passed a floor that contradicts it.

**P3: own writes sit in slightly related units, as they do for common classes.**
- Medians over n=112:
  - R_own is 0.0143, R_flow 0.0098 and R_mis 0.0093, against a chance level of 0.010.
  - Delta_flow is +0.0043 [0.0036, 0.0051] (n=106), and R_own - R_mis is +0.0046 [0.0033, 0.0069].
  - R_flow is close to R_mis (+0.00043), so delta is 8.6e-5 and the tolerance reduces to Delta_flow >= 0.
  - P3 meets the essay's condition. In absolute terms the effect is small: own-weighted relatedness is about 1.4 times
    chance.
- The result does not depend on rarity.
  - C0 gives the same Delta_flow (+0.0039, n=112).
  - Rare - C0 R_own is +0.0005 [-0.0004, +0.0017], p = 0.30 (floor p = 0.22).
- Own weight is not enriched in quiet units.
  - q_n is the unit's AdamW effective step (mean 1/(sqrt(v_hat) + eps) over its rows).
  - The pre-registered covariate Spearman(w_n / f_n, q_n) has a median of -0.15 and is negative in 111 of 112 classes.
  - So the packing prediction fails. Own writes, relative to flow, lean towards busy units.

**Secondaries.**
- S1 (builders beyond pole membership): the intercept at RPp = 0.5 is 0.606 [0.550, 0.661] on K(c), n=112. The protocol
  also lists an own-effect-weighted version, which `compare` omits; it is at chance (RBpx_w median 0.470, C0 0.485).
- S2 (memorization per image, level 10): the only nominal effect is stage 2, +0.047 [0.020, 0.084], p = 0.007, n=56. It
  equals its same-seed floor (D0 +0.055; rare against the other-set run, Wilcoxon p = 0.84). `compare` reports S2
  without the floor.
- S3 (own share): rare - C0 is 0.0000, p = 0.79 (floor p = 0.49).
- S5 (rare against the other-set run, where c is common): PI -0.015 [-0.051, 0], lambda_3 +0.047 [0.016, 0.064],
  auroc_far -0.046 [-0.069, -0.035].
- S6 (retention, post hoc by level): auroc_near loses -0.112 at level 10 and -0.031 at level 100 (paired medians),
  against -0.126 and -0.014 for auroc_far. The essay's secondary prediction (near retention below far retention) holds
  at level 100 only.

**Controls.**
- Noise, fit check. fit_T is [1, 0, 0, 0, 0, 0, 1, 0] (median 0), and top-5 is 1.0 for all 8.
  - Protocol Sec. 4 says such a control is "reported as uninformative".
  - `compare` computes informative = false but never reads it (`convnext_rare.py:1459`). It writes "positive controls
    failed: primaries are unvalidated".
  - Read literally, the protocol contradicts itself here. Sec. 7's failure condition ("noise PI not below the real
    classes'") is also met: the noise median PI is +0.993 (n=8) against -0.970 (n=112). On defined classes only it is
    -0.9994 (n=3) against -0.99987 (n=23).
  - Neither section exempts an uninformative control. Either way, the primaries are not validated.
- Noise, design. As built, this control cannot test memorization.
  - Every noise PI is close to +1 or -1 (8 of 8 have |PI| >= 0.987) and equals sign(F).
  - So the comparison passes when 5 or more of the 8 noise classes have F < 0. If noise F is symmetric about 0, which
    is what the essay expects, that has probability 0.36. O never enters.
- Noise, the two fitted classes are no evidence of memorization per image.
  - In each run one noise label wins top-1 on its 10 images (122 in A0, 617 in B0), and the other three get 0.
  - The 4 noise labels of a run share one carrier route: all 8 carriers in common for every A0 pair, 6 to 8 in B0.
    When the same labels are common, they share 0 to 2.
  - Net of the other noise classes, their own effects are near 0 in the unit maps and large in the head (X_head 26.7
    to 34.0, head_share >= 0.97). So the labels are told apart only in the head.
  - Both fitted classes are 'no route': |F| + |O| is 0.81 against a threshold of 9.31, and 6.12 against 9.19.
  - For a noise class, N_own is the other 3 noise classes, so its O null has only 3 values.
  - Per-image logits are not saved, so we cannot tell whether the winning label also takes the other labels' images.
- Relabel. The statistic is F with the source class's Near10 minus F with the label's Near10.
  - Median +7.84, 6 of 8 positive (n=8). That passes the code's median rule; the protocol gives no aggregator.
  - It is weak: sign test p = 0.145, and the bootstrap interval of the median is [-0.43, 42.99].
  - No relabel class fit its label: top-1 and top-5 are 0 on the source class's validation images, and fit_T was 0 of
    10 at step 12,500.
  - All 8 still learned from it: auroc_far is a median 0.764 against 0.529 in C0, higher in 8 of 8.
  - The same statistic where nothing is relabelled has median +0.86 in C0 and +4.33 in the other-set run (up to
    48.5).
  - Rare - C0 is positive in 7 of 8, but no larger than the run-to-run spread (median |other - C0| 13.9 against
    |rare - C0| 16.5). So the control does not separate from its floor.
- Calibration. C0's whole own class gives O_whole with median 1.99 (n=112), against a rare O of 0.039.
  - The protocol gives no pass criterion.
  - Against what PI needs, it fails: |O_whole| is below C0's definedness threshold (median 14.9) in 112 of 112 classes.
  - So a real own-built niche would not count as 'defined' under P1's rule.

**Bearing on the essay.**
- C2a (feeding): neither supported nor refuted.
  - P1's median PI of -1 is the sign of F among classes with an extreme F. It is the same in common classes, and it is
    unvalidated.
  - It does not fit the packing account either, which predicts F about 0 and a large O.
  - Over all classes the single-pathway and linear versions lean towards feeding, but they are reported, not tested.
- C2b (carriers): consistent with the essay, though not specific to rare classes.
  - Own writes sit in units related to c a little more than gradient flow or chance would put them (R_own 0.0143
    against R_flow 0.0098 and chance 0.010), and not in quiet units.
  - The same holds for common classes, so it says nothing particular about rare facts.
- C4 (location): within the unit maps, the rare own effect is not "nearly all in the last three blocks" (lambda_3
  0.426). Whether it sits deeper than for common classes depends on the reference. The head holds 91% of it, which
  fits "as close to the output as possible", but that is confounded with AdamW's larger steps for a rare head row.
- C1 (universality): the pre-registered test gives an excess of signatures at p about 1e-12. But the excess also
  appears for common classes and rests on a null that is not exchangeable with own. So it is not evidence of packing
  specific to rare classes.
- This tiny model does not memorize rare classes as predictions (top-1 0, fit_T 0). The runs test where the own writes
  of a weakly learned rare class go, not memorization in the essay's sense.

**Limits.**
- Seed 0 only.
- No primary is validated: the noise control is uninformative, and the relabel control does not exceed its floor.
- Floors: the instrumentation floor (C0 - E1) and the seed floors were not computed, because E1 and E1r were not
  passed to `compare`. P1's floor_ok is hard-coded to true.
- Not in the outputs:
  - steps 12,500 (P1, P3, S1 to S3) and 5,000 (P2);
  - S4 (shelter and churn) and S7 (consolidation);
  - the relative and random decoys.
- Two readings are inferred, not observed: the left skew of the F null and the joint-knockout mechanism. The F null
  draws and the per-unit g, v, G and V at p* are not saved.
- Most of the readings beyond the primaries, universality and the pre-registered secondaries are post hoc and
  uncorrected: the level splits, the symmetric gates, the first-order reference, the null decomposition and the
  per-class readings of the controls.

**What seed 1 must show.** Pre-registered: a primary counts only if seed 1 gives the same sign at one-sided p < 0.05,
after Holm correction and, for P2, beyond the same-seed floor. The further conditions below are not pre-registered.
Seed 0 shows they are needed before a replication means what the protocol intends.
- P1: a median PI < 0 at seed 1 would meet the decision rule, but it would only be about rarity if PI(rare) - PI(C1)
  (or PI(C0)) is also negative beyond the same-seed floor. Report alongside it:
  - the F_pct tail counts and the symmetric-gate medians;
  - the median over all 112 classes and the median with 'no route' counted as 0;
  - the gate-only, value-only and linear versions on the defined set.
- Universality: the excess over the pseudo-own must also exceed the common-condition excess on the same classes (rare
  against C1, paired), tested with McNemar.
- P2:
  - Report the first-order whole-class reference and the Own_m reference next to the exact one. A count against the
    essay should need all three to be negative.
  - Support needs lambda_3(rare) > 0.5, which seed 0 is far from (0.426).
  - The level-10 deepening (+0.135, beyond the floor) should replicate, and the stage-3 null shift (0.30) needs an
    explanation.
- P3: Delta_flow >= -delta should replicate. Rare - C1 R_own decides whether the effect has anything to do with rarity.
- Controls:
  - An informative noise control needs top-1 on at least 50% of its training images, which this recipe did not reach
    (2 of 8 classes at 25,020). Without it the primaries stay unvalidated, so the noise dose or design must change.
  - The relabel control should be scored against C1 and the same-seed floor.

Data: `docs/data/rare/seed0/` (compare output, per-class measures, go/no-go, logs) and the list of suspected bugs and
protocol deviations, `docs/data/rare/seed0/bugs_and_questions.md`.

### Blind audit of the essay's own last-layer neurons (pilot, Haiku raters, 2026-10-17)

The essay puts the share of easily interpretable last-layer neurons at about 50% for InceptionV1 mixed5b "with better
tools" and about 99% for its parameter-matched ConvNeXt-SwiGLU ('medium', 7,057,377 parameters). Both of its
microscopes are public, with an example sheet per neuron and the label, value and rank of every crop. The medium
microscope is of a 90-epoch checkpoint (`epoch_090.pt`), not of the 10-epoch recipe of the training map that E1 copies;
a 90-epoch tiny microscope (91,828 parameters) and a ConvNeXt-T-sized one are also public. The InceptionV1 microscope
uses Lucent's weights, whose two LRN layers differ from Google's graph (Sec. 2).

Protocol (`experiments/microscope_audit.py`; data in `docs/data/audit_pilot/`): 40 random neurons of the medium model's
last block (stage3.block2, 1,024 units) and 40 of mixed5b (1,024), seed 0. Each sheet is cut into its crops and rebuilt
in one layout (128 px crops: 12 of the top 16, 4 near the 99.9th percentile, 8 near the 99th, the 4 most negative), with
each crop's label and activation relative to the strongest crop, so the rater cannot tell the model from the format.
Two independent raters per neuron (Claude Haiku agents) follow the essay's own method (describe the positive pole
without a list of nouns first; abstract shared properties count; related offshoots count; weak activations count less;
the negative pole is judged on its own; no vague umbrella) and choose one of: one job, one job with related offshoots,
two related jobs, several unrelated, not understood. Description test: 8 strong crops of each neuron are held out (top-16
ranks 3, 7, 11, 15 and half of the 99.9% crops); a third, blind agent sees only the rater's description and must pick
those 8 from 16 crops, the other 8 being strong crops of 8 other sampled neurons of the same model. Chance is 4 of 8;
6 or more correct happens by chance with probability 0.066 (hypergeometric).

| | medium ConvNeXt (essay ~99%) | InceptionV1 mixed5b (essay ~50%) |
|---|---|---|
| ratings 'one job' or 'one job with related offshoots' | 80 of 80 (Wilson 95%: 0.95 to 1) | 50 of 80 (0.52 to 0.72) |
| neurons rated so by both raters / by either | 40 / 40 of 40 | 21 / 29 of 40 |
| rater agreement on that call (Cohen's kappa) | 1.0 (kappa undefined: no variation) | 0.80 (0.58) |
| description test, mean correct of 8 (neuron minimum) | 7.45 (6.0) | 5.43 (3.0) |
| ratings with the call above and 6 or more correct | 80 of 80 | 24 of 80 |
| rater confidence (1 to 5), mode | 4 | 3 |
| negative pole rated one job | 59 of 80 (20 'none or weak') | 44 of 80 (pre-ReLU: it does not reach the next layer) |

Reading: on the essay's own neurons, blind raters reproduce its contrast. Every sampled medium neuron is read as one job
(often with offshoots) by both raters, and every description finds at least 6 of the 8 held-out crops; mixed5b is read
as one job about 60% of the time, and its descriptions find the held-out crops much less well. With 40 of 40 neurons,
the medium model's true rate is at least 0.91 (Wilson lower bound), which agrees with 99% but cannot tell it from about
93%. Limits: (1) the raters are Haiku; on mixed5b they call some neurons one job whose descriptions find no more held-out
crops than chance (for example a long-necked water-bird neuron, 3 of 8), so their calls are lenient there, and the test
column is the more objective one; (2) the description test rewards specificity, not unity: a neuron read as two unrelated
jobs (shaggy fur and spoked cart wheels) can score 7.5 of 8 when both are named; (3) the medium last-layer units are
largely class or class-group detectors (wolves, revolvers, tabby fur), as expected one block before the classifier of a
well-trained model; the essay's claim that the network is "broadly legible the whole way through" needs mid layers;
(4) dataset examples only (no feature visualizations, which are in a different style per microscope and would break
the blinding); (5) the two microscopes draw crops from different image pools (mixed5b: the first 50,000 training
images, with negatives from all 1.28M; medium: the 50,000 validation images); (6) 40 neurons per model.

### Blind audit at tiny scale: the essay's 90-epoch tiny model, E1's units and E1's random directions (2026-10-17)

The essay says its ConvNeXt is "broadly legible at every scale" (92k to 28.6M parameters), and that with less room
neurons become coarser, not mixed. Its tiny microscope is public (91,828 parameters, 90 epochs, top-1 13.8% on the
microscope page). E1 (10 epochs, our recipe, top-1 18.5%) gives the comparison at the same size, and random directions
of E1's last block give the control that the essay's models do not allow (their weights are not public).
`experiments/convnext_microscope.py` renders E1's stage3.block2 in the essay's own sheet format (the same buckets: ranks
1-16, 5-12, 47-54, 497-504 by each image's maximum of the post-SwiGLU product, and the 4 lowest minima; 192 px crops
centred on the strongest position, moved inside the image; 96 px cells), for the 64 units and for 32 random dense
directions over the 64 units (seeded Gaussian, unit norm; response sum_n d_n u_n). The audit protocol is that of the
pilot above (Haiku raters, two per neuron, blind common layout, held-out description test against distractors from the
same set). Data in `docs/data/audit_tiny/`.

| | essay tiny, 90 epochs (64 units) | E1 units, 10 epochs (64) | E1 random directions (32) |
|---|---|---|---|
| ratings 'one job' (with or without related offshoots) | 116 of 128 (0.91; 0.84 to 0.95) | 76 of 128 (0.59; 0.51 to 0.67) | 61 of 64 (0.95; 0.87 to 0.98) |
| neurons rated so by both raters | 54 of 64 (0.84) | 32 of 64 (0.50) | 30 of 32 (0.94) |
| description test, mean correct of 8 (share of neurons at 4 or less) | 6.83 (0.03) | 5.35 (0.20) | 5.83 (0.00) |
| ratings with the call and 6 or more correct | 105 of 128 | 45 of 128 | 39 of 64 |
| rater agreement (kappa) | 0.88 (0.27) | 0.81 (0.61) | 0.97 (0.65) |

- The essay's tiny last-layer units are read as one job about as often as its medium ones are in the pilot, and their
  descriptions find the held-out crops nearly as well (6.83 against 7.45 of 8). At its own checkpoint, the essay's claim
  that the tiny model is legible holds under this protocol, although that model is the least accurate of the three
  (13.8% top-1).
- In E1, units are read as one job less often than random directions over the same units (76 of 128 against 61 of 64
  ratings; Fisher exact p = 2.5e-8), and their descriptions find fewer held-out crops (5.35 against 5.83). The random
  directions read as visual properties that cut across classes (white or cream bodies against grass, spindly jointed
  limbs, saturated green); the units read more often as "dogs and other furry animals" or as nothing. So in E1's last
  block the unit basis is not privileged, which agrees with the automatic mixed-unit rule (E1 and E7 above) and goes
  further: the units are less coherent than random directions.
- The two tiny models differ in training length (90 against 10 epochs) and in an unpublished recipe (the essay's reaches
  a lower accuracy in nine times the steps, so its regularization or augmentation is probably stronger), and possibly in
  their widths (Sec. 2: the tiny widths are not determined by the published count). Which of these makes the essay's
  units clean is not decided here. A run of our recipe for 90 epochs, rated at 10, 30, 60 and 90 epochs, would test the
  training-length part.
- Limits: Haiku raters (lenient on InceptionV1 in the pilot); "one job" here includes coherent visual properties, which
  is why random directions can score high: the rating measures coherence, and the comparison against random directions
  measures whether the basis is privileged; the essay's models allow only the first.
- Rater check (Sonnet, one rating and one description test per item, 20 random items of each set, same panels;
  `docs/data/audit_tiny/ratings_tiny_sonnet.json`): one job for 19 of 20 essay tiny units, 15 of 20 E1 units (4 called
  several unrelated) and 19 of 20 E1 random directions (Haiku on the same items: 36 of 40, 21 of 40, 37 of 40); held-out
  crops found 7.05, 5.30 and 5.80 of 8 (Haiku: 6.83, 5.08, 5.88). Sonnet is more lenient on E1's units than Haiku, so the
  gap in the category call shrinks (0.75 against 0.95, not significant on 20 items), while the description test, which
  does not depend on the rater's leniency, gives the same order with both raters: essay tiny units, then E1's random
  directions, then E1's units.
- The automatic class rule on the same sets (the medium plan's R16: a direction is mixed if its top 16 images split into
  two WordNet groups at cut 7, each of at least 3 images, and also into two visual groups by DINOv2 class distance):
  essay tiny units 0.34, E1 units 0.42, E1 random directions 0.66 (references: the essay's medium last block 0.077,
  InceptionV1 mixed5b 0.69). By classes, E1's random directions are the most mixed set, while raters and the description
  test find them more coherent than E1's units: they cut across classes along visual properties (a white body on grass is
  egrets, hens and poodles). At 64 units for 1,000 classes the essay's tiny units also span several class groups (0.34)
  and are still read as one job, which is the essay's "coarser, not mixed". The two kinds of measure answer different
  questions (class purity against visual coherence), and only the second separates the essay's tiny model from E1.

### Blind audit of a middle layer: the essay's medium stage2.block4 against InceptionV1 mixed4c (2026-10-17)

The essay says its medium ConvNeXt is "broadly legible the whole way through". Same protocol as the pilot (Haiku raters,
two per neuron, common blind layout, held-out description test), 40 random neurons of the medium model's stage2.block4
(the middle of its 9-block third stage, 512 units, 14 x 14 grid) and 40 of InceptionV1 mixed4c (512 units, 14 x 14
grid), seed 1. Data in `docs/data/audit_mid/`.

| | medium stage2.block4 | InceptionV1 mixed4c |
|---|---|---|
| ratings 'one job' (with or without related offshoots) | 48 of 80 (0.60; 0.49 to 0.70) | 59 of 80 (0.74; 0.63 to 0.82) |
| neurons rated so by both raters | 22 of 40 | 27 of 40 |
| 'not understood' ratings | 29 of 80 | 16 of 80 |
| description test, mean correct of 8 (share of neurons at 4 or less) | 5.49 (0.12) | 6.12 (0.07) |
| ratings with the call and 6 or more correct | 24 of 80 | 45 of 80 |
| rater agreement, kappa | 0.79 | 0.68 |

- In this middle layer the essay's ConvNeXt is not more legible than InceptionV1 under this protocol; by both the rating
  and the description test it is somewhat less. The essay's comparison with InceptionV1 (5%, 50%, 70%, 99%) is about the
  last layer, and there the contrast holds (pilot above); InceptionV1's middle layers are the ones the Circuits work
  found readable, so this does not contradict the essay's last-layer numbers, but it does not support "legible the whole
  way through" as a difference between the two models.
- Limits, beyond those of the pilot: the two microscopes crop differently in middle layers. The medium sheet's stage-2
  crops are 144 px windows around the strongest position (the essay's focus crop), so a rater sees a part or texture,
  while the InceptionV1 sheet's mixed4c crops show close to the whole image, with the object's context. A unit that
  responds to a texture or part can be harder to name from a zoomed window. One block per model; Haiku raters; dataset
  examples only (the essay's microscope also offers feature visualizations and attribution, which the essay counts as
  part of the better tools).

Crop-scale check (zoomed InceptionV1 mixed4c). Each positive InceptionV1 cell was cut down to the medium sheet's stage-2
window (144 / 224 of the cell's side), centred on the argmax of the microscope's own 24 x 24 heat map for that cell and
moved inside the cell; the negative cells were left whole (`microscope_audit.py`, source `iv1_4c_zoom`). Same 40
channels, same item numbers and distractor draws (compose with seed 1 gives identical medium items, checked), new
Haiku ratings (2 per neuron) for the 40 zoomed items only; the medium ratings are those above.

| | medium stage2.block4 | InceptionV1 mixed4c, whole | InceptionV1 mixed4c, zoomed |
|---|---|---|---|
| ratings 'one job' (with or without offshoots) | 48 of 80 (0.60) | 59 of 80 (0.74) | 69 of 80 (0.86; 0.77 to 0.92) |
| neurons rated so by both raters | 22 of 40 | 27 of 40 | 32 of 40 |
| 'not understood' ratings | 29 of 80 | 16 of 80 | 7 of 80 |
| description test, mean correct of 8 | 5.49 | 6.12 | 6.36 |
| ratings with the call and 6 or more correct | 24 of 80 | 45 of 80 | 55 of 80 |
| kappa | 0.79 | 0.68 | 0.48 |

- The zoom raised InceptionV1's score. Paired by channel and rater, 14 ratings changed to 'one job' and 4 changed away
  (two-sided sign test p = 0.031); the description test rose by 0.24 crops (28 up, 17 down, p = 0.14).
- With the crops matched, the gap to the medium model's stage2.block4 grew: 69 of 80 against 48 of 80 (Fisher exact
  p = 3.0e-4); strict count 55 against 24 (p = 1.6e-6). Crop scale does not explain the gap. A part or texture in a
  window was easier to name than a whole scene, so the earlier InceptionV1 number was low.
- Limits still open: the two layers sit at different relative depths (mixed4c is the 5th of 9 Inception blocks; stage2.block4
  is the 11th of 18 ConvNeXt blocks in the medium model, counting its 3-3-9-3 stages); the window centre comes from
  the heat map for InceptionV1 and from the grid position for ConvNeXt; the zoomed raters agree less (kappa 0.48).

### Positive control for the audit: InceptionV1 units against random directions (2026-10-17)

Question: can the audit (same protocol: Haiku raters, two per item, 'one job' call, held-out description test) tell a
model's units from random directions in the same layer? If it cannot, a share of 'one job' neurons says nothing about
whether the units, rather than the space, carry the meaning, and E1's result (random directions 0.95 against units 0.59)
is in part a property of the method.

Setup (`experiments/inception_microscope.py`; data `docs/data/audit_iv1r/`). One renderer for every item, in the format
of `convnext_microscope.py`. Let a(x) be the concatenated pre-ReLU outputs of a layer's four branch convolutions (C
channels, Lucid order). Items per layer: 40 units a_n (mixed5b: the pilot's 40 channels; mixed4c: the mid audit's 40),
40 pre-ReLU random directions v_k = sum_n d_kn a_n and 40 post-ReLU random directions w_k = sum_n d_kn relu(a_n), with d_k
a seeded Gaussian vector over the C channels, unit norm, the same d_k for both kinds. Crops: whole 224-px images for
mixed5b, 144-px windows around the strongest position for mixed4c. 240 items, blind and shuffled together, 480 ratings.
Check of the renderer: each unit's top 16 images equal the top 16 of E0's independent statistics (16 of 16 for every
mixed5b unit, 15.98 of 16 on average for mixed4c).

| | units | random, pre-ReLU | random, post-ReLU |
|---|---|---|---|
| mixed5b, 'one job' ratings | 46 of 80 (0.57; 0.47 to 0.68) | 54 of 80 (0.68; 0.57 to 0.77) | 45 of 80 (0.56; 0.45 to 0.67) |
| mixed5b, both raters | 20 of 40 | 25 of 40 | 18 of 40 |
| mixed5b, description test (mean of 8) | 5.71 | 5.60 | 5.08 |
| mixed5b, 'one job' and 6 or more correct | 32 of 80 | 34 of 80 | 17 of 80 |
| mixed4c, 'one job' ratings | 63 of 80 (0.79; 0.69 to 0.86) | 59 of 80 (0.74; 0.63 to 0.82) | 59 of 80 (0.74; 0.63 to 0.82) |
| mixed4c, both raters | 29 of 40 | 28 of 40 | 29 of 40 |
| mixed4c, description test | 6.03 | 6.03 | 5.34 |
| mixed4c, 'one job' and 6 or more correct | 49 of 80 | 46 of 80 | 29 of 80 |
| kappa (5b; 4c) | 0.69; 0.63 | 0.77; 0.81 | 0.55; 0.94 |

- Units and pre-ReLU random directions score alike in both layers (Fisher exact, units against pre-ReLU directions:
  'one job' p = 0.25 and 0.58; strict count p = 0.87 and 0.75). The audit does not see a privileged basis here.
- Post-ReLU random directions get the 'one job' call as often as units, but their descriptions find fewer held-out
  crops: strict count 17 against 32 (mixed5b, p = 0.016) and 29 against 49 (mixed4c, p = 0.0025). This is the space the
  next layer reads, and the only place where the units do better.
- The random directions are not disguised units. The median overlap of a direction's top 16 images with the top 16 of
  any single unit (all 1,024 or 512 channels, from E0's statistics) is 2 of 16; with any unit's top 50 it is 3 (mixed5b)
  and 5 (mixed4c). One direction of 40 in mixed5b and 7 (pre) or 8 (post) of 40 in mixed4c share 8 or more of their top
  16 with one unit's top 50; all of those got the 'one job' call from both raters, the other directions in about two
  thirds of the ratings (0.67 and 0.55 in mixed5b, 0.68 and 0.67 in mixed4c).
- Our renderer agrees with the essay's sheets on the units: mixed5b 0.57 here against 0.62 in the pilot (same 40
  channels); mixed4c 0.79 against 0.74 (whole images) and 0.86 (zoomed).

Reading against earlier work (each paper read in full, quotes and numbers checked by a second reader):
- Szegedy et al. (ICLR 2014, arXiv 1312.6199, Sec. 3) state that "there is no distinction between individual high level
  units and random linear combinations of high level units". Their evidence is the authors' visual reading of 4 units
  and 4 random directions of AlexNet on ImageNet and the same on MNIST, with no count, blinding or score; the layer, the
  side of the ReLU and the distribution of v are not given. In their printed AlexNet figures, 2 of the 4 random
  directions each show 2 of the 5 images of one printed unit (pixel-identical tiles). The table above is a blind,
  quantified version of their test on another model, with a held-out check, and it agrees for pre-ReLU directions.
- Bau et al. (Network Dissection, CVPR 2017, arXiv 1704.05796, Sec. 3.2, Fig. 4) find the opposite with another metric:
  a random rotation of the 256 conv5 units of an AlexNet trained on Places205 (the number of draws is not given) cuts
  the number of unique detectors from 79 to 16. Their statistic counts the distinct labelled concepts (Broden, IoU > 0.04 at each unit's top 0.5% of
  activations) matched anywhere in the layer, with a fixed vocabulary and pixel masks; ours rates each direction's top
  tail with an open vocabulary. Both can hold: rotated directions can each have a coherent top tail while matching fewer
  distinct labelled concepts, or concepts outside the vocabulary.
- Fong and Vedaldi (Net2Vec, CVPR 2018, arXiv 1801.03454) find that learned combinations of filters match or beat the
  best single filter on 76 to 92% of concepts for segmentation and 85 to 90% for classification (AlexNet conv1 to conv5);
  they draw no random directions.
- Elhage et al. (Toy Models of Superposition, 2022) define a privileged basis (an elementwise nonlinearity makes the
  coordinates special) and add that "having a privileged basis doesn't guarantee that features will be basis-aligned".

What this means for the essay. The essay's headline numbers (about 50% of InceptionV1 mixed5b and about 99% of the
medium ConvNeXt's last layer easy to interpret) come from reading dataset examples, as here. Under this protocol a high
share does not show that the units are clean as units: InceptionV1's random pre-ReLU directions read as well as its
units, and E1's random directions read better than its units. The share measures how coherent the top tails of a
layer's directions are. The essay's comparison between models (which our pilot reproduces, 40 of 40 against 50 of 80)
stays a statement about units, but the step from it to "the neurons are clean" needs a control the essay does not
report. For the medium model the control needs its weights, which are not served; our own medium runs carry random
directions in their microscopes.

Limits: Haiku raters only; 40 items per set; the random directions are Gaussian over channels whose scales differ
(branch outputs differ in size), so a few large channels weigh more; dataset examples only (no feature visualisation);
the 'one job' call is generous for any direction with a coherent top tail, which is the point tested, and the strict
count with the held-out test is the sharper measure.

## 8. Status of E1 and infrastructure notes (2026-10-16)

- Pipeline: `experiments/molab_e1.sh <run> [trainer args]` downloads the validation set, packs the training set into RAM
  (`rcdl.imagenet.prepare_train`), restores `resume.pt` from the results dataset if one exists, and trains.
  `experiments/hf_sync.py` copies results (checkpoints, attribution parts, resume state) to the private Hugging Face
  dataset every 10 minutes. The trainer writes `resume.pt` every 1,000 steps and at every checkpoint, so a lost
  sandbox costs at most 1,000 steps plus the data packing.
- Review (three reviewers, each finding checked by a second agent), all fixed and tested (92 tests pass):
  - under Muon the tracked biases had a 10 times larger learning rate than the other biases (`bias_lr_scale`);
  - TF32 broke the exact sum of the Muon parts to about 1e-3 per step (matmuls of the Muon parts now run in fp32);
  - fp16 checkpoints flushed 7% of the per-class parts to zero at step 50 (now fp16 times a power-of-two scale, plus
    the fp32 remainder, so `W = a W_0 + sum_k C_k + R` holds exactly as stored);
  - a loader error hung the trainer; the run did not save its last step for step counts off the checkpoint list;
    the logged validation count was wrong;
  - the mixup weights forced a host-device sync every step; data packing used 3 of 18 workers; the
    RandomResizedCrop fallback differed from torchvision's; JPEG decode and crops ran in series.
- The uniform weight decay of the parts is deferred to one scale s per map (the stored parts are C / s, flushed when
  read or when s < 1e-6), which saves a full pass over C each step (about 6 to 10 minutes per medium run).
- Open for the medium model: stack the maps of a stage into one state (36 small updates per step become 4).
- molab: five sandboxes ended (HTTP 410, or a fresh container behind the same URL) after 15 to 60 minutes, while the
  one whose notebook stayed open in the user's browser ran for more than 3 hours. The memory limit is 160 GiB (it counts
  /dev/shm), far above what the runs use. The pattern fits a sandbox that lives only while a browser client holds its
  notebook session: a sandbox opened by a script that then closed its browser was recycled within about 10 minutes.
  Sandboxes are now held open by a long-lived cloud browser tab, which also reads the pairing URL and token from inside
  that tab; runs resume from `resume.pt` on Hugging Face when a sandbox is lost anyway.
