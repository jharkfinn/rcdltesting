# Reproduction of "Retrieval-Centric Deep Learning" (RCDL)

Sources reproduced:

- Paper: Schlegel et al., *Retrieval-Centric Deep Learning in Growing Nonparametric Neural Networks*, arXiv 2610.03858 (NeurIPS 2026). Read end to end from the LaTeX source.
- Code: `rcdl_tutorial_torch.ipynb` and `rcdl_tutorial_JAX.ipynb` in this repository (single-pass CIFAR-10).
- X article by J. von Oswald, 9 Oct 2026 (x.com/oswaldjoh/status/2108669918883230177). It also describes an unpublished language experiment (synthetic biographies on a Gemma 3 backbone).

Status: PENDING SECTIONS ARE MARKED. Numbers come from `results/` (not committed) and `experiments/summarize.py`.

## 1. How to run

| What | Command | Hardware | Time |
|---|---|---|---|
| Tests (paper identities, port vs notebook) | `python3 -m pytest -q tests` | CPU | 5-30 s |
| One single-pass run | `python3 -m experiments.run experiments/configs/repro_1epoch.json --only gws --seeds 1` | CPU (4 cores) | about 10 min |
| All single-pass runs, 5 seeds, plus ablations | `bash experiments/gpu_batch.sh results/gpu` | 1 GPU (22 GB) | about 1-2 h on an L4 |
| Kernel baselines + 50-epoch L=0 | `bash experiments/a100_batch.sh results/a100` | 1 GPU (40 GB) | about 1 h on an A100 |
| Table of results | `python3 -m experiments.summarize results/gpu` | any | seconds |

Data: `--source binary` reads the official CIFAR-10 tarball (the notebook's order). `--source parquet` reads the Hugging Face mirror `uoft-cs/cifar10`. Both hold the same 50,000 training images with the same labels, in a different order (the mirror starts at official image 30,000). The test sets are identical.

Colab: the GPU runs use the official Colab CLI (`google-colab-cli`, `colab new --gpu L4`, `colab upload`, `colab exec`). The jobs start in the background on the VM, so a dropped connection does not stop them.

## 2. The port is exact

`rcdl/` ports the notebook's PyTorch code. `tests/test_layers.py` executes the notebook's own cells and compares outputs and stored tables for all six model kinds (GWS, gauge, RBF, hybrid, parametric softmax, ResNet-MLP with Adam and Muon). In the notebook's own sample order, the port reproduces the notebook's stored learning curve:

| Samples | Notebook test loss / acc | Port test loss / acc |
|---|---|---|
| 0 | 6.6935 / 0.1000 | 6.6935 / 0.1000 |
| 4,000 | 1.7883 / 0.3886 | 1.7883 / 0.3890 |
| 8,000 | 1.5734 / 0.4636 | 1.5734 / 0.4636 |
| 12,000 | 1.4602 / 0.4870 | 1.4603 / 0.4872 |
| 16,000 | 1.4038 / 0.5077 | 1.4039 / 0.5077 |
| 50,000 | 1.1631 / 0.5918 | PENDING |

CPU and GPU runs of the port give the same final accuracy and loss to 4 decimals.

## 3. Mathematical claims checked numerically (`tests/test_identities.py`)

All pass in float64.

| Claim (paper section) | Check |
|---|---|
| A linear layer trained by SGD inside a deep net equals linear attention over its stored inputs and scaled output gradients (Sec. 2, Eq. 1). | Exact to 1e-12. |
| For a positive definite kernel, SGD on the implicit feature-space weight equals appending (k_t = x_t, v_t = -η g_t) (Sec. 3.1, App. A). | Explicit polynomial features, exact. |
| A Gaussian-mixture weight function induces exactly the discrete weighted softmax, for any σ_k, σ_v (Sec. 3.2). | 1-D quadrature, error < 1e-6. |
| Functional gradient descent equals appending one tuple (App. C.3). | The loss slope matches the predicted rate. See the note below. |
| Gauge invariance (Prop. C.2) and locality of the gauge-fixed update (Thm. C.3). | Exact. |
| Momentum and decoupled weight decay have a closed form per stored row (App. D.3). | Equals primal SGD-momentum-decay to 1e-6. |
| The mesa layer over (input, gradient-corrected output) pairs is the FOOF optimizer (Sec. 4, App. E.2 "Algorithm A"). | The closed-form regularized least-squares solution equals the recursion W_T = W_{T-1} - η g_T x_T^T Φ_T^{-1} to 1e-10. |
| The Mesa (Algorithm B) and Delta updates as implemented in `rcdl.train`. | One step equals the defining formula exactly. |

Note on the functional gradient. With notation q_t (query), s_t (prediction), g_t (loss gradient with respect to the prediction), w_i and k_i (stored weights and keys), σ_k^2 and σ_v^2 (key and value variances of the base Gaussian measure), η (step size), the exact functional-gradient step appends

- key k_new = σ_k^2 q_t,
- weight w_new = η <s_t, g_t> / D_t,
- weighted value ṽ_new = -η σ_v^2 g_t / D_t,

where D_t = Σ_i w_i exp(<k_i, q_t>) is the weighted softmax denominator at q_t. The loss then falls at the rate e^{σ_k^2 ||q_t||^2} (σ_v^2 ||g_t||^2 + <g_t, s_t>^2) / D_t^2 per unit of η. The test confirms this. The experiments do not use this rule. They drop the 1/D_t factor (App. E.3 says so) and replace the fixed ratio σ_v^2 between the value step and the weight step by two free learning rates β and η. So the "principled" rule is one member of the family that the code implements, not the rule that produced the reported numbers.

## 4. Results: single-pass CIFAR-10 (batch 16, no augmentation)

Port of the notebook configurations, Hugging Face sample order, seeds 1-5, Colab L4. Accuracy is on the full 10,000-image test set after one pass over the 50,000 training images.

| Model | Seeds | Test accuracy (mean ± std) | Per seed | Notebook (1 seed) | Paper |
|---|---|---|---|---|---|
| Growing Weighted Softmax (GWS) | 5 | **59.41 ± 0.22** | 59.18, 59.64, 59.36, 59.64, 59.24 | 59.18 | 59.4 ± 0.4 |
| Growing RBF | 5 | **59.10 ± 0.62** | 58.42, 59.51, 58.46, 59.75, 59.34 | 58.42 | 58.8 ± 0.2 |
| Gauge-invariant GWS | 5 | **58.62 ± 0.44** | 58.48, 58.98, 58.01, 59.11, 58.50 | 58.33 | 58.4 ± 0.4 |
| ResNet-MLP + Muon | 1 | 57.04 | 57.04 | 57.2 ± 0.4 (tuned) | ~57.5 (plot) |
| ResNet-MLP + Adam | 5 | **49.48 ± 0.57** | 49.98, 48.50, 49.55, 49.72, 49.63 | 48.2 ± 1.0 (tuned) | ~50 (plot) |
| PENDING: AdamW, SGD, RMSprop, Mesa, Delta, parametric softmax, hybrid | | | | | |

The first L4 session was evicted when a third GPU session was opened (the account allows two at a time). The rows above come from its logged per-run outputs. Reruns that rebuild the raw files are queued.

**Verdict:** the single-pass claims of Fig. 2 (right) and App. F.2 reproduce. GWS and RBF are the best methods. Muon is about 2 points behind, and Adam about 10 points behind.

## 5. Ablations

PENDING.

## 6. Classical baselines (single pass over 50,000 training images, exact solvers, Colab A100)

| Method | Best on test (as paper) | Held-out selection | Paper (Fig. 2 right, plot) |
|---|---|---|---|
| kNN (euclidean, inverse-distance votes, k = 5) | 35.64 | 36.08 | ~35.5 |
| Kernel ridge regression, RBF kernel, median-heuristic bandwidth ("SVM" in the paper) | 58.55 | 58.12 | ~58.5 |
| Kernel ridge regression, the GWS first-layer kernel exp(<q, q'> / (T √d)) on RMSNorm'd pixels | 58.42 (T = 10, λ = 10) | 58.02 | not in paper |
| Nadaraya-Watson vote with that kernel (pure retrieval, no learning) | 44.58 (T = 1) | 44.14 | not in paper |

Both classical points of the paper reproduce. Context they give: **single-pass GWS (59.41%) is 0.9 points above exact kernel ridge regression on raw pixels (58.55%).** Most of the single-pass advantage of the growing nets over parametric MLPs is therefore the advantage of a kernel machine on CIFAR pixels. The depth and the online learning rule add about 1 point. The pure retrieval vote is 15 points lower, so the learned values (not retrieval alone) carry the result.

## 7. Multi-epoch runs (augmentation, batch 64)

### 7.1 Only the ratio β/η matters (5-epoch sweep, L = 0, T = 10)

Test accuracy after 5 epochs (seed 1):

| η \ β | 1 | 3 | 10 | 30 |
|---|---|---|---|---|
| 0 | 46.25 (train loss 38) | 47.68 (113) | 47.71 (384) | 47.25 (1162) |
| 0.003 | 52.51 | 54.85 | **55.94** | 55.49 |
| 0.03 | 44.45 | 48.16 | 52.51 | 54.85 |
| 0.3 | 36.42 | 40.40 | 44.45 | 48.16 |

Settings with the same β/η give the same accuracy and the same training loss to 4 digits (for example (0.003, 1) and (0.03, 10)). Reason: with ṽ the stored weighted values and w the stored weights, the output Σ e^z ṽ / Σ w e^z does not change when all new ṽ and w are scaled by the same factor c. Only the B initial rows (weight 1/B each) break this symmetry, and their share of the denominator is soon negligible. So η is a gauge (scale) choice, not a learning rate, and the layer has two effective hyperparameters: T and β/η.

Consequences:
- The best ratio is about 3,000 at T = 10.
- The paper's 50-epoch grid (η in [0.3, 3], β in [20, 60]) gives ratios of only 7 to 200. In the notebook's convention these all fall in the poor region; a run with η = 1, β = 30 for 50 epochs stalls at 44.6%. So the paper's grid must use another convention (for example a batch-mean gradient in the weight rule), which it does not state.
- η = 0 (no denominator learning) is not stable over several epochs: the outputs and the loss grow without bound (train loss 38 to 1,162), although the argmax still gives about 47%.

### 7.2 PENDING: 50-epoch L = 0 temperature sweep, linearization, depth at 10 epochs, MLP baselines, parametric softmax, CIFAR-100

## 8. Discrepancies between paper, notebooks and article

### 8.1 Method as derived vs method as run

1. **The density factor is dropped.** The functional-gradient step scales the new weight and value by 1/D_t (section 3 above). The experiments drop it (App. E.3, "for numerical stability"). Every row then gets the same step size, wherever the existing rows are dense or sparse.
2. **Two free step sizes replace one.** The derivation fixes the ratio of the value step to the weight step (σ_v^2). The code uses independent β and η. The best single-pass setting has η = 0.003 and β = 3. With η that small, the denominator hardly grows (the 16 initial rows have weight 1/16 each), so the layer is close to an unnormalized exponential-kernel layer. The ablation `gws_eta0` (η = 0) tests this.
3. **An undocumented clip.** The code clips <s_t, ḡ_t> to [-1, 1] before the max(0, ·) clip, with ḡ_t the batch-mean gradient. This equals clipping <s_t, g_t> to [-B, B], with B the batch size. The paper's Algorithm 2 does not mention it (the notebook does). The ablation `gws_noclip` tests it.
4. **Decay and momentum reach only the values.** Paper App. D says weight decay and momentum scale both the stored weights w and the weighted values ṽ. The notebooks scale only ṽ (numerator), not w (denominator). The port reproduces the notebook by default (`decay_scope='values'`) and offers the paper's version (`'values_and_weights'`).
5. **Momentum and decay are combined approximately.** The notebook multiplies the separate decay factor and momentum factor. This equals the paper's closed form (App. D.3) only if one of them is zero. The port has both (`age_rule='notebook' | 'exact'`).

### 8.2 Bugs in the notebooks (active only with momentum or weight decay > 0)

6. **Ages use the current batch size.** The age of each stored row is computed from the size of the current batch. At evaluation the torch notebook uses batch 500 instead of 16, so the momentum and decay factors at test time are wrong (test: `test_momentum_matches_notebook_only_at_train_batch_size`). The JAX notebook evaluates with the training batch size, so it does not have this bug.
7. **Ages wrap around.** Ages are computed modulo the table size. When the table is exactly full (the 1-epoch configurations end in this state), the oldest batch gets momentum age 0 and coefficient 0, so its values are erased (test: `test_notebook_mask_zeroes_oldest_rows_when_table_is_exactly_full`). Both notebooks have it.

The port fixes both by storing the write step of each row.

### 8.3 Inconsistencies in the paper

8. **Teacher depth.** The main text says the teacher has 1 hidden layer. App. E.1.2 says L = 2 residual blocks.
9. **Teacher-student scale.** Table "Nonparametric GWS teacher-student" lists "Epochs 200000" (these are steps). App. E.11 says 200 epochs add 10^7 rows. 200,000 steps × batch 64 = 12.8 × 10^6. The parametric baseline has 10,240,000 = 160,000 × 64 rows.
10. **Learning-rate ranges look swapped.** Teacher-student: β ∈ [0.1, 1], η ∈ [10, 100]. CIFAR: η ∈ [0.3, 3], β ∈ [20, 60].
11. **Initialization.** Algorithm 2 says K_0 ~ N(0, 1/√d I). The text and the code use standard normal rows rescaled by RMSNorm. The hybrid layer's text says D_0 has zero keys and values. The code uses the same random D_0 as GWS.
12. **RBF table.** It calls the readout a "Weighted RBF-kernel layer" (RBF rows have no weight) and lists a momentum sweep that Algorithm 5 does not have.
13. **Scale convention of the 50-epoch grid.** Its ranges (η ∈ [0.3, 3], β ∈ [20, 60]) do not contain the single-pass optimum in the notebook's convention (η = 0.003, β = 3; App. F.2). In the notebook's convention, η = 1 makes a 50-epoch L = 0 model stall at about 45% test accuracy (section 7). The paper does not say whether its grid uses per-sample or batch-mean gradients, or what values it selected.
14. **Hyperparameters were selected on the test set.** The paper says so implicitly (App. E.1: "we select those hyperparameters that maximize/minimize the metric displayed"). The X article says so outright ("chosen by overfitting the test set"). There is no validation split. Section 6 reports held-out-selected numbers for the baselines.

### 8.4 Code quality notes (no effect on results)

15. The torch notebook backpropagates into the input images (`requires_grad_` on the batch) only to keep the graph alive. That costs a full backward pass through the first retrieval layer. The port skips it and gets identical rows (test: `test_skipping_first_layer_input_grad_keeps_updates`).
16. Every forward pass scans the whole pre-allocated table to build the valid-row mask. The port caches the mask.
17. The JAX notebook adds `bias - stop_gradient(bias)` to each RCDL output. That parameter has no effect on the forward pass, but the optimizer still updates it.
18. The hybrid layer runs Adam over the whole pre-allocated table (4.3 GB of parameters at L = 3), not only the active rows. This needs about 18 GB of memory for a model that uses under half of the table.
19. **Scale of the initial rows.** D_0 stores RMS-normalized random vectors as the *weighted* values ṽ, with weight w = 1/B. Its effective values are therefore ṽ / w = B times the intended unit scale. At initialization the readout logits are about B times too large: the notebook's initial CIFAR test loss is 6.69 instead of ln 10 ≈ 2.30, and the teacher-student initial MSE is about 140 against a target variance of 0.0044. The effect fades as real rows accumulate, so it has little influence on final numbers.

## 9. Extensions

Each direction lists what it tests and the questions it answers. The infrastructure in this repo supports A, B, C, D and F directly. E needs a new language-model harness.

### A. What does the single-pass result measure?
- Compare GWS with exact kernel ridge regression using the same kernel as its first layer (section 6), and with the Nadaraya-Watson vote (no learning).
- Ablate the denominator (η = 0), depth (L = 0, 1, 2) and temperature.
- Measure feature learning: kernel alignment or CKA between each layer's stored keys and the labels. The notebook's analysis already shows weak class purity of the retrieved rows (28% at layer 0, 36% at the readout, chance = 10%).
- Questions: How much of the 2-point margin over Muon and the margin over parametric nets is the kernel machine itself? Does depth add more than a better kernel? Is the "weighted softmax" doing anything that an unnormalized exponential kernel does not?

### B. From the theory to better learning rules
- B1. **Exact functional gradient.** Keep the 1/D_t factor, computed in log space, and tie β = η σ_v^2. Does the rule the paper derives beat the heuristic it runs?
- B2. **A second-order RCDL layer.** Sec. 4 maps the mesa layer to the FOOF optimizer for linear layers. The kernel analogue of the mesa layer is kernel recursive least squares (Engel, Mannor and Meir, 2004). Its approximate-linear-dependence test adds a row only when the new key is not well represented by the stored keys. That is a principled version of the hybrid layer's distance threshold τ. This answers the paper's open question, how to derive advanced optimizers for RCDL, and it gives sublinear growth from the same principle.
- B3. **Positive weights without clipping.** Update w multiplicatively (exponentiated gradient, i.e. mirror descent on the mixture weights). Is the paper's gauge trick a special case?

### C. Memory and compute
- C1. **Sparse retrieval.** The notebook shows that test-time top-K retrieval loses accuracy fast: 37.5% with the top 1,000 of 50,016 rows, against 59.2% dense. Train with top-K retrieval so that the model adapts, then use an approximate nearest-neighbour index.
- C2. **Budget methods from online kernel learning:** NORMA truncation (Kivinen, Smola and Williamson, 2004), the Projectron (Orabona et al., 2008), budgeted SGD with merging (Wang, Crammer and Vucetic, 2012). Plot accuracy against stored rows for these and for the paper's hybrid layer.
- C3. **Distill the table into a parametric layer** (k-means keys plus least-squares values) and measure the loss.

### D. Continual learning (the article's "unforget" claim)
- Class-incremental split CIFAR-10 (5 tasks of 2 classes), single pass: GWS and RBF against MLP+Adam, with and without replay. This is cheap on an L4.
- Questions: Do new rows overwhelm old ones through the shared denominator? Can a test-time temperature change recover old classes, as in the "silent memory" analogy (paper Sec. 6)?

### E. Language: the biography experiment from the article (unpublished)
- Rebuild the synthetic biography task of Zucchet et al. (arXiv 2503.21676, built on Allen-Zhu and Li, arXiv 2309.14316) with a 2-layer transformer. Replace the feed-forward layers by a growing RBF layer, a wide random-Fourier-feature ("Cos Bochner") layer, or GeGLU. Measure the plateau length against population size, then forgetting under fine-tuning.
- Question: the article reports that a wide random-feature layer trained by SGD approaches the growing RBF layer. If that holds, the gain comes from the kernel's locality, not from growth or retrieval. This is the cleanest test of what "retrieval-centric" adds.

### F. Evaluation hygiene
- Select hyperparameters on a validation split for every method. The paper selected on the test set.
- Compare at matched inference compute: a GWS layer costs O(N d) per query and grows with the data.
- Strengthen the parametric softmax baseline. Its keys start with length about 1, while the queries have length √d ≈ 55, so its attention starts almost uniform (logits near 0.006). Initialize its keys from data or rescale them, and retune it.
