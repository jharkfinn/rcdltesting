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
| E0 | Static checks on the public InceptionV1 (V neuron 4e:55, the car / dog pose circuit 4c:447, mixed5b) | the essay's re-readings of canonical neurons | model verified; needs ImageNet validation images |
| E1 | Writer atlas: ConvNeXt-SwiGLU tiny and medium, per-class attribution of the gate and value maps of late blocks, dense early checkpoints | instrument | code ready (`rcdl/convnext.py`, `rcdl/attribution.py`); needs a GPU and ImageNet |
| E2 | Order of growth: when each class enters a neuron's writer set, and its distance to the classes already there | recursive generalization | |
| E3 | Head start: did the neuron already fire on a class before that class wrote into it? | circuit racing | |
| E4 | Muon against AdamW against SGD | optimizer as an intervention on circuit racing | |
| E5 | RCDL head (growing table) as a memory store | do offshoots come from captured residual error? | |
| E6 | Umbrella = bee with exact attribution, parametric against RCDL head | off-ramp claim | |
| E7 | Crush where packing is predicted (long-tail classes, 13-unit last layer) | rare facts ride related circuits | |
| E8 | Toy: relatedness against anticorrelation as the cause of shared neurons | gap 2 above | running (`experiments/toy_packing.py`) |

## 5. E8 results so far (preliminary)

Setup: 8 families of 4 features; related = embedding cosine rho within a family; exclusive families (members never
co-occur), co-occurring families, and exclusion pairs of families; 1 or 2 active features per sample; x -> Linear(64, m)
-> Linear(m, H) + ReLU -> Linear(H, 32) + ReLU; AdamW 3e-3, 6,000 steps, batch 256. A first sweep was invalid (output
bias started at 0, 14 of 32 output ReLUs never turned on); with the bias at 0.1 the loss falls to 3e-4 at m = H = 32.

With 3 seeds per cell (pole threshold 0.3 of the neuron's largest response; null = pole memberships shuffled):
- Related features share a positive pole above chance when capacity is tight (for example 0.39 against 0.21 for
  related, co-occurring pairs at rho 0.6, m 32, H 8).
- Opposite poles of a neuron go above chance to unrelated, anticorrelated pairs (0.67 against 0.48 at rho 0.6, m 8,
  H 32; 0.54 against 0.28 at m 32, H 32) and below chance to related, anticorrelated pairs (0.37 against 0.47; 0.29
  against 0.30).
- Confound: at rho 0 the two anticorrelated pair types already differ in some cells, because of how the exclusion
  structure is built; the effect of relatedness is the change of that difference from rho 0 to rho 0.6.
- Open: 7 more seeds for the key cells (running), the growth order from the attribution logs, and a SwiGLU hidden layer
  (the essay's neuron type).
