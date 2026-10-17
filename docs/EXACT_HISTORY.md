# Questions that need the exact training history

This note reviews a proposal for questions about the essay ("Neural Networks Are Already Clean", Cammarata, Goodfire
2026) that the exact dual-form record can answer and that ordinary checkpoints cannot. It checks the proposal's sources
against the papers, read end to end. It then gives the experiments, the controls and the status. The method is in
`NEURON_DEVELOPMENT.md` Sec. 3; the earlier results are in its Sec. 5 to 7.

## 1. The contribution formula, and the optimizers we use

The proposal starts from plain SGD (no momentum, no weight decay). For a linear map z = W h + b, the part of unit j's
pre-activation on a query image q written at step t is

  c_{j,t}(q) = -eta_t * delta_{j,t} * < h_t(x_t), h_T(q) >

- j: an output unit (one row of W).
- t: a training step. T: the step at which we read the network.
- x_t: the training example of step t (for a batch, sum the expression over the examples and positions).
- eta_t: the learning rate at step t.
- delta_{j,t}: the derivative of the loss of x_t with respect to z_j at step t (the teaching signal).
- h_t(x_t): the map's input for x_t at step t.
- h_T(q): the map's input for q at step T.
- < a, b >: the dot product.

The sum of c_{j,t}(q) over t, plus < W_0[j], h_T(q) > (W_0 is the initial weight matrix), is z_j(q) at step T. The bias
follows the same rule with the input 1.

Where this holds in our runs:

| Optimizer | Is the overlap < h_t(x_t), h_T(q) > the same for every unit j? | Exact per-example or per-group parts? |
|---|---|---|
| SGD | yes | yes (the formula above) |
| SGD with momentum beta | yes; the coefficient of step t becomes a sum of eta_s (1 - beta) beta^(s - t) over later steps s | yes |
| decoupled weight decay lambda | yes; older writes are multiplied by a_T / a_t, with a_t = product over s <= t of (1 - eta_s lambda) | yes |
| AdamW (E1, E1r) | no: the update of W[j, i] is divided by sqrt(v[j, i]) + eps, with v the second moment of that weight | yes, given v (our tracker) |
| Muon (E4) | no: the update is P U / norm(U), with P a polynomial in U U^T and U the momentum matrix, so P mixes the rows | yes, given P (our tracker) |

So under AdamW the overlap is a weighted dot product < h_t(x_t), D_{j,t} h_T(q) >, with D_{j,t} the diagonal matrix of
1 / (sqrt(v[j, i]) + eps) over the inputs i. The weights differ between units. Under Muon, a unit's update also carries
the teaching signals of the other units of the same map, through P. The proposal's point 2 ("within a dense layer, all
neurons share the same input-overlap term; differences between their updates enter through delta") is therefore true for
SGD with or without momentum and weight decay, and false for the two optimizers of our runs. Our exact parts do not need
that factorization: the tracker keeps one momentum per group, and with the shared v (AdamW) or the shared P (Muon), the
group parts add up to the weights (Sec. 3 of `NEURON_DEVELOPMENT.md`).

The bias parts give a direct record of the teaching signal per class. The bias input is 1, so class k's bias part of
unit j is

  Cb_T^(k)[j] = - sum_t eta_t * mhat_t^(k)[j] / (sqrt(vb_t[j]) + eps)

- mhat_t^(k)[j]: the bias-corrected momentum of class k's bias gradients for unit j at step t (a positively weighted sum
  of the earlier steps' class-k gradients sum delta_{j,s} over class-k examples).
- vb_t[j]: the shared second moment of unit j's bias gradient.

The factor 1 / (sqrt(vb_t[j]) + eps) is positive and the same for all classes. So within one unit, the sign of
Cb_T^(k)[j] and the order of its size over the classes k give the sign and the order of class k's summed teaching signal,
reweighted over time by positive factors. In E4 the tracked biases use AdamW, so this holds for the Muon run too. The
increments between two checkpoints give the same readout for one interval.

SwiGLU units have two maps. With u = SiLU(g) * v (g the gate pre-activation, v the value pre-activation) and dL/du the
signal arriving from the readers:

- the gate's teaching signal is dL/du * SiLU'(g) * v;
- the value's teaching signal is dL/du * SiLU(g).

The proposal's "for a ReLU neuron, that term includes its gate" becomes: each map's signal is scaled by the other map's
current value. This includes the region where g and v are both negative, which the essay names as a source of noise.

## 2. The sources, checked

Each source was read end to end: Irie, Csordas and Schmidhuber 2022 (ICML, PMLR v162; "The Dual Form of Neural Networks
Revisited"), Schlegel et al. 2026 (arXiv 2610.03858, the RCDL paper, main text and Appendices A to H), Pezeshki et al.
2021 (NeurIPS; arXiv 2011.09468, "Gradient Starvation", main text and Appendices A to C), and the essay (with its figure
data).

| Proposal's statement | Source | Verdict |
|---|---|---|
| "Irie and colleagues explored this connection for interpretation in 2022" | Irie et al. 2022 | Supported. They train with vanilla SGD (MLPs with two 800-unit hidden layers on MNIST and Fashion-MNIST, single-task, multi-task and continual; one-layer LSTM language models) and read the attention over all stored training inputs. The attention weight is the input overlap only. At layer 2, the class with the largest summed attention is the model's output for 84.7% of the correctly classified test images and for 60.1% of the others. They report that weighting the attention by the error norm showed no consistent trend. |
| "Adding these contributions to the initial-weight contribution reconstructs the neuron's value before its nonlinearity" | RCDL paper, Eq. 1 | Supported, for a linear map under gradient descent. The paper derives growing layers for SGD; it states that Adam and Muon are not derived in its framework. |
| "The article predicts that a neuron's branches share a local origin and develop in a specific order" | essay, "Watching Neurons Develop" and recursive generalization | Partly supported. The essay observes that "the movement is local: a neuron drifts to neighboring things", from one retrained InceptionV1, and describes a recursive order (a branch, then branches of the branch: "iy", "diy", "do it yourself"). The mechanism ("a bit of error picked up by the neuron that was already closest") is labelled a guess. |
| "existing parts and downstream readers let a circuit acquire a new case before competing circuits develop" | essay, "More on Circuit Ecology" | Supported, as the essay's guess: "A region of the network that already has useful parts and readers can learn the next related thing faster than anything starting fresh, so it takes the error signal before a competitor can grow." |
| "The article leaves open the mechanism that keeps most offshoots small" | essay, "Recursive Generalization" | Supported: "We suspect some pressure keeps branches from overgrowing. We don't know what it is." |
| "rare facts use established circuits, which protect them from unrelated gradients" | essay, "Rare Things and Memorization" and "semantic gravity" | Supported, as the essay's account: the umbrella's off-ramp "survives only because the umbrella and bee circuits already exist. Those circuits feed it, and shelter it from unrelated gradients." The essay calls the prediction testable and does not test the sheltering. |
| "a distinct neuron becomes useful when other computations need to build on its judgment" | essay, "More on Circuit Ecology" | Supported, as a suspicion: "consolidation into a distinct neuron becomes especially useful when other computations need to build on it". |
| "Existing work on gradient starvation establishes related effects under specified assumptions" | Pezeshki et al. 2021 | Supported, with these assumptions. The network is linearized (neural tangent kernel regime), the task is binary, the loss is cross-entropy with a ridge penalty, and two features couple through a small parameter delta. The result is about the optimum, not about timing: at the optimum, the response z_2 to feature 2 falls when the strength s_1 of feature 1 grows (dz_2* / d(s_1^2) < 0). Their "features" are singular directions of the tangent-feature matrix, not units. Their dual variables alpha_i are the per-example error weights (alpha_i = sigmoid(-y_i yhat_i), with y_i the label in {-1, +1} and yhat_i the logit), and the optimum is a weighted sum of training examples. This is a representer form of the optimum, not the trajectory identity of Irie et al. The mechanism (examples that one feature already fits stop supplying gradient to other features) is what our record measures per unit and per class. Their remedy, an L2 penalty on the logits ("spectral decoupling"), gives a ready intervention. The essay itself uses the term ("circuit racing, gradient starvation and ecology"). |
| "The dual identity also applies to SGD-trained models that develop superposition" (cited to Irie et al. 2022) | Irie et al. 2022 | The identity part is supported. Irie et al. have no superposition or toy-model experiments, so the citation does not support the suggestion of a superposition control; that suggestion is the proposal's own. |
| "Replacing it with an RBF kernel changes the model's learning dynamics" | RCDL paper | Supported. Growing RBF and weighted-softmax layers store each input as a key and add a value from the functional gradient; the kernel is local, and the training curves differ (training loss falls much faster, Appendix G.1). |

Two of the proposal's general statements need a condition:

- "Detailed gradient records contain the same information." For AdamW and Muon this needs the optimizer states (v, or
  the polynomial P) as well as the gradients.
- "Removing a training example changes later updates and representations, so causal claims require training replay."
  This is correct, and Sec. 3 below gives the replay tool and its noise floor.

## 3. Replays: the tool and its noise floor

`rcdl/attribution.py` (`snapshot`, `load_snapshot`) and `experiments/convnext_train.py` (`--snapshots`, `--init_from`,
`--zero_loss_classes`, `--extra_ckpts`, `--batch_crops`). A snapshot holds the model, the optimizer states, the tracked
layers' summed momentum and second moments, and the RNG states. A replay starts a new attribution with W_0 = the
snapshot weights and one extra group ("history") that carries the old momentum. Unit tests: continuing from a snapshot
matches the uninterrupted run to 1e-12 and the parts add up (AdamW, SGD, Muon).

Smoke test on molab (tiny model, 120 steps, snapshot at step 60, max absolute weight difference at step 120):

| comparison | difference |
|---|---|
| two control replays from the same snapshot | 1.6e-5 (GPU nondeterminism) |
| the original run against a control replay | 7.2e-3 (the crops of the partial epoch differed) |
| a control replay against a replay with zero loss weight for 100 classes | 8.6e-3 |

Other checks: the replay's W_0 equals the snapshot weights; the replay's parts add up (largest relative residual
1.1e-7); the history group is not zero.

So a causal contrast must compare two replays from the same snapshot. With `--batch_crops` (added after the smoke test)
the crops are seeded per batch, so a control replay also repeats the original run up to GPU nondeterminism; the original
run must use the same setting. A test checks that an epoch begun at any batch repeats the uninterrupted epoch.

## 4. Four experiments that need the exact history

Each was designed by one agent and attacked by a second (designs, critiques and revised protocols are kept with the
session; the revised protocols are the ones in use). For each: the essay's claim, what only the exact record measures,
and the main corrections from the critique.

### 4.1 Exits, starvation and first in line (no new training)

Claim: offshoots are "a bit of error picked up by the neuron that was already closest" (More on Circuit Ecology);
a related input finds the unit "first in line"; established routes "get there first and starve the packed circuit of
gradient"; "some pressure keeps branches from overgrowing. We don't know what it is" (Recursive Generalization).

What only the record gives: the exact split of every change of a unit's response over an interval into each class's
new writes, the decay of its earlier writes, input drift and a position switch; the per-class teaching signal (bias
parts) as a time-integrated starvation record; E1r's random-group split as the noise null.

E1 facts that shaped it: a unit's top-20 classes turn over heavily (75 to 85% after step 5,000); 16,330 established
permanent drops; 4,603 drops are anchored on a stable core (the essay's "stays on the fields" case).

Corrections from the critique: events and primary splits on the 16 checkpoints only (history snapshots refine timing
and give the second moments); comparison sets fixed before the window (no selection on the outcome); every null
matched in size (number of training examples); cross-fitting (select on 25 validation images per class, measure on the
other 25); no term dropped for fp16 rounding (bounds carried as intervals); the remainder R computed at every grid
point (it is up to 8e-5 of the weights, not 0); trajectory checks before a snapshot is used (E1's step-25,020 history
version is bit-identical to its final checkpoint). X1 (why a class leaves) is exploratory: no outcome of it refutes
the essay, because the essay says the pressure is unknown.

Status: implementation and adversarial review running; runs on E1, E1r and E4 data need no training.

### 4.2 Umbrella = bee (short fine-tunes from E1)

Claim: a nonsense label is learned as "an off-ramp between two routes the network already has", and the existing
circuits "feed it, and shelter it from unrelated gradients" (Rare Things and Memorization; semantic gravity).

What only the record gives: during fine-tuning with continued ordinary training, every weight change splits exactly
into the photo's writes, each class's writes and the earlier momentum, so "feed" (the target class's writes raise the
taught photo's margin more than an untaught photo's) and "shelter" (unrelated classes erode it less than they erode a
memory with no source route) become measurable.

Design (v2, after the critique): branch every run from E1's own final weights and optimizer state after a 300-step
replay-only warm-up; AdamW with the second moment frozen at the branch (otherwise the photo inflates the shared
denominator and fakes "feed"); untracked parameters frozen, so the parts cover the whole change; 8 shadow splits of
the replay slots give 64 size-matched pseudo-groups per run, the same in taught and untaught runs; feed and shelter are
the primaries, umbrella coupling is reported two-sided (the essay does not give its sign).

Deviation, decided before any fine-tune: E1 recognizes no umbrellas (0 of 50 validation images, 10 of 1,300 training
images top-1; its guesses are pinwheel, balloon, maypole). The pre-registered fallbacks apply: the primary arm uses the
best-recognized class unrelated to bee as the source (jack-o'-lantern, 90% top-1; photos with p_source 0.37 to 0.54,
close to the essay's 44%); umbrella with its 10 recognized training images is a secondary, weak-route arm.

Status: the fine-tune script passed its GPU smoke test (interval buckets add up to the weight change to 8e-7; the frozen
parameters do not move; the photo's group is non-zero only in taught runs). Analysis and arm queue being implemented.

### 4.3 Rare classes ride related machinery (four to six training runs)

Claim: "Rare things ... have grown on related circuits in every experiment we've run"; "the neurons ... that carry a
rare fact should be related to what the fact is about, not unrelated ones chosen because they rarely collide";
memorization sits close to the output.

What only the record gives: in the units that carry a rare class, the exact parts written by its relatives' examples
against those written by its own few examples (fine-group shadow split, every own image its own group at level 10);
where the own writes act, against where gradient flow alone would put them.

Design (v2): 128 classes in two crossover sets (dense and isolated strata; 0, 10 or 100 images); runs A0 and B0 (one
set rare each) and an all-common run C0 (the critique found that 104 of the 128 classes have a relative in the other
set, so the crossover's "common" condition was not common); the same images split the same way in every run; noise
controls (10 smooth-noise images labelled to a dropped class) as a positive control that must show a packing
signature; size-matched nulls throughout.

Status (2026-10-17): seed 0 done (A0, B0, C0; both rare runs passed the go/no-go). Results in
`NEURON_DEVELOPMENT.md` ("Rare classes, seed 0"): no primary is validated (the noise control did not fit, the relabel
control does not exceed its floor); P1's median PI of -1 reflects the sign of F where F is extreme, because O is
about 70 times smaller than F by construction, and the same holds for common classes; the rare classes are not
memorized as predictions (top-1 0). The suspected bugs and protocol issues are listed in
`docs/data/rare/seed0/bugs_and_questions.md`; an amendment for seed 1 is to be fixed before any seed-1 measure is read.
Seed-1 runs follow T90 on the same sandbox.

### 4.4 Where memorization sits (one training run with injected items; planned)

Claim: "memorization tends to want to sit as close to the output as possible ... the late layers did most of the
work"; noise memorization is "partly semantic". Maini et al. (2023) found the opposite (memorization spread over a few
neurons in many layers) with approximate gradient accounting. The exact per-item writes decide this in one run: each
memorized item's own writes, removed exactly, with clean items at the same dose as the control.

Corrections from the critique: score the item-specific effect (the item's own writes on the item minus the same writes
on held-out inputs of its type), since its writes to the head bias raise its label on every input; replace knockouts of
whole class parts (they cancel 300 to 600 times, so removing one breaks the network) with first-order route profiles;
validate the knockout against replays from step 5,000 (not 20,000, after which only 1.4% of the learning-rate mass
remains); make the fine-tune arm (teaching a trained model, as the essay did) part of the core; track the final
LayerNorm and the layer scales so that coverage is complete; set the exposure doses from a pilot at the real rates.

Status: designed; needs the most new code (a per-term split of mixup, convolution and affine tracking, injected
items). Next after 4.1 and 4.2.

## 5. An infrastructure finding

The GPU JPEG decoder writes on its own CUDA stream. Memory it receives from PyTorch's caching allocator can still be
in use by kernels queued on the default stream, so the decode can overwrite live tensors. This caused about 120 of
50,000 non-finite validation images earlier (fixed then by synchronizing after the decode) and, in a second form,
NaN weights after two steps of the umbrella warm-up (found today: the replay's synchronous decode ran while the
optimizer's kernels were still queued). Fix: the replay decode synchronizes before and after; the training loader's
decode thread now allocates from its own CUDA stream and marks the images for the consumer's stream; batches are
bit-identical to before and the speed is unchanged. The earlier training runs (E1, E1r, E4, E7) never produced a
non-finite value, and their weights satisfy W = a W_0 + sum_k C_k to about 1e-5 at every checkpoint, which a corrupted
weight would break; corrupted activations or gradients cannot be ruled out by that check.
