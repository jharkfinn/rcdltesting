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

### Stage 1: a unit's strongest classes teach it down, and its output hurts them

The bias signal above has one exception to the late-block pattern. In stage 1 (three blocks, residual width 8), a
unit's 20 strongest classes write its bias in the direction that lowers its response on them. This section tests that
exception in three runs: E1 and E1r (AdamW, seeds 0 and 1) and E4 (Muon, seed 0). Scripts:
`experiments/convnext_bias_self.py` (exact writes), `convnext_bias_paths.py` (probe gradients, splits, ablation),
`convnext_unit_roles.py` (per-class ablation profiles and write directions). An adversarial review (three reviewers and
a synthesis) checked the scripts and the claims below; its corrections are included.

**1. The exact writes, with each class's own derivative.** In stage 1, 41 to 53% of the (unit, pole class) pairs at
the end lie in the negative lobe of the gate (g < 0 and v < 0, so u = SiLU(g) v > 0), against 0 to 14% in the late
blocks. A single sign per unit is therefore not the sign of du/dv for every class. `convnext_bias_self.py` uses the
local derivative at each class's mean pre-activations instead. For unit n, classes k and c, and an interval (t1, t2]:

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
| other late blocks (range) | +0.47 to +1.92 | +0.38 to +2.26 |

(A unit is 'negative-lobe' if most of its top 20 classes have g < 0 and v < 0, and 'positive' if most have g > 0 and
v > 0. After step 2,500 units are nearly pure: 97 to 99% of a unit's top 20 classes are in its lobe.)

- Not an artefact of the orientation. With the derivative the reversal is stronger than with one sign per unit. A
  reviewer recomputed the derivatives per image on E1 (instead of at class means): the values agree within 0.06 and
  no sign changes.
- Not caused by the negative lobe. It is stronger in positive units than in negative-lobe units, in every block and
  both seeds. In E1, block 2 flips at a negative-lobe share of 0.07; the share rises after the flip, and almost all
  units that change type in the flip intervals go from positive to negative lobe. So the downward pushes move units
  into the negative lobe, not the reverse.
- Not selection. Choosing the top 20 at t1, at t2, at the end or at the previous checkpoint gives the same values
  (the probe images that define the top 20 are validation images, independent of the training writes). The whole
  ranking is reversed: in stage 1 a unit's 20 weakest classes push its pole response up (+0.20 to +0.73), and a random
  set of 20 classes gives about 0. 70 to 88% of live units are negative in each interval after step 2,000.
- The WordNet relatives of the top 20 also push down (-0.17, -0.11, -0.07 in E1). The remaining classes push up a
  little (+0.03 to +0.04 per class in E1). The pushes nearly cancel (net about 0.1% of gross), so the top classes' sign is the
  finding and the rest's sign follows from the balance.
- A class's own writes lower its own response too (-0.61 to -0.89 in stage 1, same formula with c = k).
- Weights as well as the bias (a reviewer's computation on E1 only; script in the review folder): with each pole
  class's mean input at its strongest position recovered from the atlas, the full writes of the top 20 classes
  (weights and bias) give -0.38 to -0.93 from interval 2,000-2,500 on, about 80% of it through the weights. The bias
  is not a special lever.
- Stage 0 differs by seed (E1: +0.20, -0.12, -0.18; E1r: -0.31, +0.61, +0.42). Stage 2 blocks 0 and 3 are near zero.

Time course (stage-1 values per interval start). The flip is staggered by block: in E1, block 2 is negative from
(500, 1,000], block 1 from (1,000, 2,000], block 0 from (2,000, 2,500] (+0.18 before). Before that, stage 1 is built
the late-block way (+0.09 to +0.53 in the intervals starting at steps 200 and 350). Block 2 flips before the warm-up ends at step
1,250, so the end of warm-up is not the trigger.

**2. Where the signal comes from: probe gradients.** The exact writes sum the teaching signal over every position of
every training image. `convnext_bias_paths.py` recomputes it on the probe set (8 clean validation images per class,
hard labels) at a checkpoint and splits it. For an image of class k, with z the logits and L = logsumexp(z) - z[k],
delta[p, n] = dL/du[p, n] is the teaching signal of unit n at position p. The bias gradients are
B_v = sum_p delta SiLU(g) and B_g = sum_p delta SiLU'(g) v (class means), and the probe pole effect is
-(B_v mean_{c in P_n} SiLU(Gm) + B_g mean_{c in P_n} SiLU'(Gm) Vm). A test checks B_v and B_g against autograd's bias
gradients. Splits, each adding up to the whole:

- positions: the strongest position p*, the 'on' positions (u >= 0.5 u*, about 75 to 150 per image for the top
  classes in stage 1), and the rest;
- logit terms: the logsumexp(z) term plus the true-class term -z[k]. (In the JSON files of this round the key 'own'
  holds the gradient of +z[k]; the true-class term is -own.)
- LayerNorm scale: every LayerNorm with 1 / sigma held fixed (forward values unchanged), and the rest, which is the
  path through the normalisers' scale.

The probe reproduces the training writes. Per unit, the Spearman correlation over the 1,000 classes between the probe
effect at a checkpoint and the exact writes of the next interval is 0.71 to 0.76 (E1 and E1r) and 0.53 to 0.59 (E4)
in stage 1 from step 2,000 (E4: from 5,000), and 0.66 to 0.76 in the late blocks. A reviewer also ran a probe that copies training
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

Late blocks other than stage 2 blocks 0 and 3: +0.76 to +1.11 (E1, E1r) and +0.85 to +1.06 (E4) after step 1,000.

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
| stage 1, top 20 classes (mean over units and blocks) | -3.2e-3 | -1.3e-3 | -2.6e-3 |
| stage 1, 20 random classes | +6.3e-4 | +5.6e-5 | +4.5e-4 |
| stage 1, share of units whose removal helps the top 20 more than the rest (blocks 0 / 1 / 2) | 0.69 / 0.67 / 0.60 | 0.57 / 0.57 / 0.39 | 0.81 / 0.68 / 0.60 |
| late blocks (stage 2 block 5, stage 3 block 1), same share | 0.17 / 0.16 | 0.34 / 0.14 | 0.24 / 0.13 |
| late blocks other than stage 2 blocks 0 and 3, top 20 classes | +1.9e-2 | +2.2e-2 | +3.1e-2 |

- Removing a stage-1 unit lowers the loss of the classes that drive it and raises the loss of others. In late
  blocks the unit helps its top classes. At step 1,000 most E1 stage-1 units still help their top classes (share helping 0.71; E1r 0.42, where
  the probe signal is already about 0).
- Not overshoot of the single unit: the cross-entropy of the top classes is nearly linear in a scale factor alpha on
  the unit's output (alpha = 0: -3.2e-3; linear prediction -4.1e-3), and the share of units that help their top classes
  but would help them more at smaller scale is 0.01 to 0.05.
- Whom stage-1 units help (the 20 classes whose loss rises most when the unit is removed): they overlap the unit's
  top 20 at chance (0.009 to 0.066; chance 0.02) and are no more often WordNet relatives of the top 20 than random
  classes (0.13 to 0.26 against 0.14 to 0.18). Late units help their top classes (overlap 0.07 to 0.29) and their
  relatives (0.24 to 0.40 against 0.15 to 0.20; visual relatives 0.21 to 0.47 against 0.11 to 0.15).
- Write directions: a unit writes into the residual stream along w_n = gamma * W_down[:, n]. In stage 1, 48 to 94% of
  units (step 20,000) have another unit with |cos| > 0.9 (late blocks 0 to 27%). The review measured an effective
  rank of the stage-1 write directions of 1.5 to 4.7 (of 8), with the layer scale gamma concentrated on one or two
  channels. Removing the whole stream component along a stage-1 unit's write direction costs every class 0.5 to 1.6
  nats: the direction is essential, and the unit is a small part of it.

**Reading.** In this model, from about step 1,000 on, a stage-1 unit's strongest classes teach it down and are hurt by
its output, in three runs with two optimizers. Its top-activating classes describe what it responds to, not whom it
serves. The effect is not an artefact of orientation, the negative SiLU lobe, selection, mixup, the read-out position,
the normalisers' scale, the competing classes, or the optimizer. Leading mechanisms (review ranking): M1 overshoot from
same-direction partners in a collapsed write space (the reversal tracks how much same-direction partners also respond
on the unit's pole: Spearman -0.65 in E1, by tercile +0.11, -0.61, -1.56); M4 the unit's write cancels a useful
incoming stream component; M5 a class-specific negative readout. Tests that decide between them (the stream state at
the unit's active positions, zeroing a unit's same-direction partners, and a replay from step 1,000 with the stage-1
layer scale frozen) are in progress.

Caveat on scope: the residual width 8 in stages 0 and 1 is this repository's choice for the tiny size (Sec. 2: the
essay does not publish widths). Whether the same happens at the essay's medium size (widths 48 / 96 / 192 / 384) is
untested.

Bearing on the essay. (a) Its account of growth gives a unit already on for a class the first claim on that class's
error (a head start). That holds in late blocks; in stage 1 after step 1,000 the strongest classes act as a brake and
the bulk of the classes holds the unit's response up. (b) Its picture of a clean residual stream ("we don't see the
stream accumulating interference only to undo it later") does not hold in this model's 8-wide stages: dozens of units
of both signs share one to three write directions, and the class pushes on a unit's bias cancel to about 0.1% of
their gross. (c) Its reading of early units from their top activations: those describe input selectivity, which this
finding does not contradict, but in stage 1 the top classes are the ones the unit's output hurts.

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
