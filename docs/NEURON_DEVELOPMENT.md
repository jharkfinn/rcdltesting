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
| E0 | Static checks on the public InceptionV1 (V neuron 4e:55, the car / dog pose circuit 4c:447, mixed5b) | the essay's re-readings of canonical neurons | done for 4e:55 and the car circuit (Sec. 5); mixed5b open |
| E1 | Writer atlas: ConvNeXt-SwiGLU tiny and medium, per-class attribution of the gate and value maps of late blocks, dense early checkpoints | instrument | code reviewed and ready (`experiments/convnext_train.py`, `experiments/convnext_atlas.py`, `experiments/convnext_growth.py`, `experiments/molab_e1.sh`); the first run was lost with its sandbox (Sec. 7) |
| E2 | Order of growth: when each class enters a neuron's writer set, and its distance to the classes already there | recursive generalization | |
| E3 | Head start: did the neuron already fire on a class before that class wrote into it? | circuit racing | |
| E4 | Muon against AdamW against SGD | optimizer as an intervention on circuit racing | |
| E5 | RCDL head (growing table) as a memory store | do offshoots come from captured residual error? | |
| E6 | Umbrella = bee with exact attribution, parametric against RCDL head | off-ramp claim | |
| E7 | Crush where packing is predicted (long-tail classes, 13-unit last layer) | rare facts ride related circuits | |
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
beach wagon, pickup, Irish and Yorkshire terriers), feeds both halves (11% and 8% of z in 295, 5% and 6% in 241). It is
the only shared input we found, and it carries a small part of z. Heat montages agree: in 295 the vehicle response
sits on wheels and the lower side of side-on cars, and the dog response on the faces of terriers, mostly seen from the
front, so the top images do not share a pose.

Reading. For these two units, the exact input attribution matches what Zoom In described: a car feature merged into
neurons that also detect dogs. It does not match a single detector of a pose shared across cars and dogs.

Limits:
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
  chance. At rho 0 the rate is at chance, which checks the method.
- Most spillover still comes from unrelated members (75% to 83% at rho 0.6), through the shared m-dimensional
  bottleneck. Relatedness is one driver of growth here, and it is not the main one.
- In ReLU units a feature's own samples write little of its pole response (median own share -0.18 to 0.29). In SwiGLU
  units they write more (0.12 to 0.54).

Limits: a reconstruction toy with a trained bottleneck and 32 features. It isolates the two causes; it does not tell
which one dominates in an ImageNet model. E1 to E3 measure the same quantities on ConvNeXt-SwiGLU, with WordNet 3.0
distances between ImageNet classes (`rcdl/imagenet_hierarchy.py`, `docs/data/imagenet_wordnet.npz`) as relatedness.

## 7. Status of E1 and infrastructure notes (2026-10-16)

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
- Open for the medium model: keep C undecayed (C / a) to save a full pass per step, and stack the maps of a stage
  into one state (36 small updates per step become 4).
- molab: both sandboxes ended with HTTP 410, after about 15 and 45 minutes, the first while writing to local disk
  and the second just after training had started with 52 GB in /dev/shm. The cause is not known (a memory limit
  that counts /dev/shm, a lifetime limit, or the end of the browser session are candidates).
