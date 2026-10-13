# Reproduction of "Retrieval-Centric Deep Learning" (RCDL)

Sources reproduced:

- Paper: Schlegel et al., *Retrieval-Centric Deep Learning in Growing Nonparametric Neural Networks*, arXiv 2610.03858 (NeurIPS 2026). Read end to end from the LaTeX source.
- Code: `rcdl_tutorial_torch.ipynb` and `rcdl_tutorial_JAX.ipynb` in this repository (single-pass CIFAR-10).
- X article by J. von Oswald, 9 Oct 2026 (x.com/oswaldjoh/status/2108669918883230177). It also describes an unpublished language experiment (synthetic biographies on a Gemma 3 backbone).

Status: PENDING SECTIONS ARE MARKED. Numbers come from `results/` (not committed) and `experiments/summarize.py`. Figures: `python -m experiments.plots` writes `docs/figures/`.

## 0. Claims and verdicts

Budget: about 12 GPU-hours on Colab (L4 and A100-40GB) plus a 4-core CPU. "Reproduced" means the same number within the stated spread, or the same qualitative result where the paper gives only a plot.

| # | Claim (source) | Verdict | Evidence |
|---|---|---|---|
| 1 | A linear layer trained by SGD is linear attention over its training inputs and gradients (Sec. 2) | Reproduced (exact) | §3 |
| 2 | Growing a PDK layer by (x_t, -η g_t) is SGD in the RKHS (Sec. 3.1, App. A) | Reproduced (exact) | §3 |
| 3 | A Gaussian-mixture weight function induces the discrete weighted softmax (Sec. 3.2) | Reproduced (exact) | §3 |
| 4 | Functional gradient descent equals appending one tuple (App. C) | Reproduced (exact), but the experiments run a different rule (no 1/D_t; free β, η) | §3, §8.1 |
| 5 | Gauge invariance and locality of the gauge-fixed update (App. C.6) | Reproduced (exact) | §3 |
| 6 | Closed form of momentum and weight decay per stored row (App. D) | Reproduced (exact); the notebooks implement an approximation and two bugs | §3, §8.2 |
| 7 | The mesa layer over the dual form is the FOOF optimizer (Sec. 4) | Reproduced (exact) | §3 |
| 8 | Single pass on CIFAR-10: GWS 59.4 ± 0.4, RBF 58.8 ± 0.2, gauge 58.4 ± 0.4; all above Muon, Adam, SGD, Mesa, Delta (Fig. 2 right, App. F.2) | Reproduced | §4 |
| 9 | Kernel ridge regression (~58.5) and kNN (~35.5) on pixels (Fig. 2 right) | Reproduced; GWS is only 0.9 points above kernel ridge regression | §6 |
| 10 | Teacher-student: GWS far below the parametric softmax; MLP student best (Fig. 3) | PENDING | §7 |
| 11 | 50 epochs: GWS and RBF beat Adam/AdamW/RMSprop/SGD, below Muon; Mesa best on CIFAR-100 (Fig. 2 left) | PENDING (L = 0 only; L = 2 needs > 150 GB) | §7 |
| 12 | Training loss falls much faster for GWS and RBF (Fig. 2 middle) | PENDING | §7 |
| 13 | Depth helps up to 2 hidden layers; temperature effect is non-monotonic (Fig. 4 left) | PENDING | §7 |
| 14 | Linearizing exp after training causes a large drop (Fig. 4 right) | PENDING | §7 |
| 15 | The parametric softmax overfits with a widening gap; GWS does not (Fig. 5) | PENDING (reduced scale) | §7 |
| 16 | ViT with GWS FFNs beats the parametric ViT after one epoch: 54.4 vs 48.4 (Sec. 5) | PENDING | §5 |
| 17 | The hybrid layer keeps accuracy with sublinear growth (Fig. 6) | PENDING | §5 |
| 18 | DeltaNet decay improves a 10-layer MLP over 2,000 epochs: 74% vs 67% (App. E.3.1) | Not attempted: about 2 × 5 A100-hours | - |
| 19 | Article: plateau grows with N; growing RBF and wide Cos-Bochner shorten it; GeGLU has a lower ceiling | PENDING (minimal version) | §7 |
| 20 | Article: parametric FFNs forget under fine-tuning; RCDL layers "unforget" without replay | PENDING (minimal version) | §7 |
| 21 | Article: RCDL matches but does not beat strong baselines, at much higher cost | PENDING | - |

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
| 50,000 | 1.1631 / 0.5918 | 1.1631 / 0.5918 |

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
| ResNet-MLP + Muon | 2 | **57.41 ± 0.52** | 57.04, 57.78 | 57.2 ± 0.4 (tuned) | ~57.5 (plot) |
| ResNet-MLP + Adam | 5 | **49.48 ± 0.57** | 49.98, 48.50, 49.55, 49.72, 49.63 | 48.2 ± 1.0 (tuned) | ~50 (plot) |
| ResNet-MLP + AdamW (notebook setting) | 1 | 49.98 | 49.98 | - | ~50 (plot) |
| ResNet-MLP (L = 4) + Mesa/FOOF, best of 6 (lr 1e-4, λ = 0.01, γ = 0.99) | 1 | 52.04 | grid: 47.42-52.04 | - | ~49 (plot) |
| ResNet-MLP (L = 4) + Delta, best of 9 (lr 3e-3, λ = 1e-5) | 1 | 46.58 | grid: 34.83-46.58 | - | ~47 (plot) |
| ResNet-MLP + RMSprop | 2 | 48.12 ± 1.44 | 47.10, 49.13 | - | below Adam (plot) |
| ResNet-MLP + SGD (momentum 0.85) | 2 | 44.26 ± 1.56 | 45.36, 43.15 | - | below Adam (plot) |
| Parametric softmax, 5,000 rows per layer | 2 | 41.77 ± 0.09 | 41.83, 41.70 | 42.02 | - |
| Parametric softmax, 50,016 rows per layer | PENDING | | | 40.60 | - |

The paper does not state the Mesa and Delta settings for this plot. The grids above lie inside its 50-epoch ranges (`experiments/configs/repro_1epoch_optimizers.json`), with selection on the test set as in the paper. The best Mesa setting is at the low edge of the learning-rate grid, so a lower rate can be better still.

The first L4 session was evicted when a third GPU session was opened (the account allows two at a time). The rows above come from its logged per-run outputs. Reruns that rebuild the raw files are queued.

**Verdict:** the single-pass claims of Fig. 2 (right) and App. F.2 reproduce. GWS and RBF are the best methods. Muon is about 2 points behind, Mesa about 7 points behind, and Adam and Delta about 10 points behind.

## 5. Single-pass ablations, hybrid layer and ViT

### 5.1 Ablations of the best single-pass GWS setting (T = 6, β = 3, η = 0.003, L = 2, batch 16; seeds 1-2; Colab L4)

| Variant | What it tests | Test acc, seeds 1 / 2 | Online train loss (last 100 batches) |
|---|---|---|---|
| Default (section 4) | reference | 59.18 / 59.64 (5 seeds: 59.41 ± 0.22) | 1.20 / 1.19 |
| η = 0 | learned denominator off: new rows get w = 0 | 28.34 / 25.73 | 81 / 88 |
| No clip of <s, ḡ> | the undocumented clip | 59.07 / 59.57 | 1.20 / 1.19 |
| η = 0.1 (β/η = 30 instead of 1,000) | step-size ratio | 53.45 / 54.03 | 1.41 / 1.40 |
| Growing RBF at T = 6 | same temperature as GWS, no normalization | 56.02 / 56.28 | 1.34 / 1.31 |
| L = 0 (readout only) | depth | 53.54 / 53.55 | 1.42 / 1.41 |
| L = 1 | depth | 58.25 / 58.67 | 1.25 / 1.23 |

Findings:
- **Depth adds 6 points** in one pass (L = 0: 53.5%, L = 1: 58.5%, L = 2: 59.4%). The L = 0 model is an online kernel machine with almost the kernel of section 6, and it stays 5 points below the exact kernel ridge solution (58.55%). So depth mainly compensates for the online, one-pass solver; the margin over the exact kernel machine is about 1 point.
- **The clip does nothing** here; the β/η ratio matters (5.7 points between 1,000 and 30).
- **η = 0 diverges**, but not because normalization is needed: the tuned unnormalized RBF layer reaches 59.1% (section 4). At equal temperature (T = 6) the RBF layer is 3.3 points below GWS; at its own best temperature (T = 10) it is level with GWS.

### 5.2 Hybrid layer: accuracy against stored rows (paper Fig. 6)

Notebook setting for every τ (L = 3, T = 5, β = 30, η = 1, Adam 1e-3, batch 16), seed 1, Colab A100, about 4 minutes per run. Capacity = stored rows summed over the 4 layers / (4 × 50,016). The paper tuned T ∈ {1, 5, 10}, β ∈ {10, 30, 60} and the Adam rate ∈ {3e-4, 1e-3, 3e-3} for each τ (27 settings per point, about 12 A100-hours in all); this sweep uses one setting, so its points are lower bounds.

| τ | Capacity (%) | Test acc (%) | Paper (read off Fig. 6): capacity / acc |
|---|---|---|---|
| 35 | 89.6 | 49.47 | ~88 / ~58.6 |
| 40 | 81.8 | 56.83 | ~79 / ~58.5 |
| 45 | 67.1 | 56.68 | ~65 / ~58.0 |
| 50 | 46.7 | 56.22 | ~45 / ~56.5 |
| 55 | 21.8 | 54.20 | ~26 / ~54.5 |
| 60 | 5.1 | 49.05 | ~5 / ~50.0 |
| 65 | 0.8 | 39.80 | ~1 / ~41.0 |
| 70 | 0.2 | 34.23 | ~0 / ~34.5 |

The capacities match the paper within 4 points at every τ, so the growth rule is the paper's. Accuracy matches within 1.2 points for τ ≥ 50 and is 1.3-1.7 points lower at τ = 40-45, where the paper's per-τ tuning matters most. τ = 35 is an outlier (49.5%, training loss 1.49 against 1.26 at τ = 40); a second seed of τ = 35 and τ = 50 is queued. The notebook's own hybrid run (τ = 50, seed 42, official sample order) gives 55.12%. **Verdict: reproduced** for the trade-off curve: at 22% of the rows the layer keeps 54.2%, 5 points below full single-pass GWS.


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

Direct check (`experiments/inspect_gws.py`, single-pass notebook GWS, seed 1, 59.18%): over 1,000 test queries, the 16 initial rows hold a median 0.20% of the denominator in layer 0, 0.25% in layer 1 and 0.03% in the readout (90th percentiles 0.73%, 0.47%, 0.07%). The same check shows that the clip max(0, ·) sets w = 0 for 47%, 43% and 65% of the stored rows of these three layers. Those rows add to the numerator only. So most readout rows act as unnormalized exponential-kernel rows, and the learned normalization rests on the other third.

The same symmetry holds in the teacher-student sweep (section 7.3): the three settings with β/η = 0.01 give relative test MSE 0.187, 0.186 and 0.186.

Consequences:
- The best ratio is about 3,000 at T = 10.
- The paper's 50-epoch grid (η in [0.3, 3], β in [20, 60]) gives ratios of only 7 to 200. In the notebook's convention these all fall in the poor region; a run with η = 1, β = 30 for 50 epochs stalls at 44.6%. So the paper's grid must use another convention (for example a batch-mean gradient in the weight rule), which it does not state.
- η = 0 (no denominator learning) is not stable over several epochs: the outputs and the loss grow without bound (train loss 38 to 1,162), although the argmax still gives about 47%.

### 7.2 50 epochs at L = 0 (no hidden layer), linearization

Setup: one GWS readout layer over RMSNorm'd pixels (d = 3072), batch 64, augmentation (App. E.1.1), 50 epochs, so 2,499,264 stored rows (30 GB of keys in fp32). η = 0.003 and β = 10 (best ratio of section 7.1). Colab A100, 22.6 min.

| T | Test acc (10,000 images) | Train acc (2,000-image subset) | Test acc with exp(z) → 1 + z after training | Paper (Fig. 4, read off plot) |
|---|---|---|---|---|
| 10 | **61.86** | 82.3 | **15.91** | ~64 (L = 0); large drop when linearized |
| 3 | PENDING | | | ~59 |
| 30 | PENDING | | | ~61 |

Test accuracy on the 2,000-image subset over training: 56.1 (epoch 5), 60.2 (epoch 20), 62.1 (epoch 35), 61.7 (epoch 45). The test loss falls the whole time (1.276 to 1.123), so the 20-point train-test gap does not come with a rising test loss.

Linearization: the paper's Fig. 4 (right) replaces exp(z) by its first-order expansion 1 + z after training. The accuracy falls from 61.86% to 15.91%, near chance. This reproduces the claim that the nonlinearity of the kernel carries the model. With z = <q, k>/(T√d) and T = 10, |z| is at most √d/T ≈ 5.5, far outside the range where 1 + z approximates e^z.

### 7.4 50-epoch baselines, depth, CIFAR-100 (paper Fig. 2 left, Fig. 4 left)

ResNet-MLP baselines (L = 2, width 3,072, GELU, batch 64, augmentation, 50 epochs; Colab A100). The paper's grids: L ∈ {2, 4}, learning rate 1e-4 to 1e-2, batch {16, 64}.

| Model | Test acc (%) | Online train loss (last 100 batches) | Minutes | Paper (Fig. 2 left, read off plot) |
|---|---|---|---|---|
| MLP + Muon, lr 1e-3 | **69.51** | 0.59 | 9.7 | ~69.5 |
| MLP + Muon, lr 3e-3 | PENDING | | | |
| MLP + Adam, lr 3e-4 | 64.72 | 0.85 | 4.5 | ~65 |
| MLP + Adam, lr 1e-4 | 64.37 | 0.87 | 4.5 | ~65 |
| MLP + SGD, lr 1e-2 | PENDING | | | ~61 |
| MLP + Mesa, lr 3e-4 | PENDING | | | ~67.8 |
| GWS, L = 0, T = 10 (section 7.2) | 61.86 | 0.81 | 22.6 | ~64 (L = 0 curve of Fig. 4) |

Depth at 10 epochs (GWS, T = 10, η = 0.003, β = 10, batch 64, augmentation). L = 2 at 50 epochs would store 2.5M rows in each of 3 layers (keys and values of width 3,072 in the two hidden layers, 61 GB each in fp32; 31 GB in the readout), about 154 GB, so depth is compared at 10 epochs:

| Model | Test acc (%) | exp(z) → 1 + z after training | Minutes (A100) |
|---|---|---|---|
| GWS, L = 0 | 58.55 | 20.93 | 1.4 |
| GWS, L = 1 | 63.83 | 24.00 | 3.9 |
| GWS, L = 2 | **65.05** | 23.69 | 7.0 |
| MLP + Adam / Muon, L = 0 and L = 2, 10 epochs | PENDING | | |

Findings so far:
- The Muon and Adam baselines reproduce the paper's values (69.5% and about 65%).
- GWS at L = 0 after 50 epochs (61.9%) is below Adam (64.7%). GWS at L = 2 after only 10 epochs (65.1%) is already level with Adam after 50 epochs. The paper's claim (GWS above Adam, SGD and RMSprop, below Muon) refers to deeper GWS models at 50 epochs, which this budget cannot hold in memory.
- The linearization collapse holds at every depth (21-24% at 10 epochs).

### 7.3 Teacher-student regression (paper Fig. 3, App. E.1.2)

Setup: inputs x on the unit sphere in R^100; target y = f(x), with f a random residual ReLU MLP (width 100, L = 2 blocks, LeCun-normal weights). The target variance is Var(y) = 0.0044. Every step draws a fresh batch, so there is no overfitting. Relative MSE = test MSE / Var(y); 1 means "predicts the mean".

Sweep at 10,000 steps (one seed; grids of the paper's App. tables; Colab A100):

| Student | Grid | Best setting | Test MSE | Relative MSE |
|---|---|---|---|---|
| GWS (one layer on RMSNorm(x), batch 64) | β ∈ {0.1, 0.3, 1}, η ∈ {10, 30, 100}, T ∈ {0.1, 1, 3, 10}, momentum ∈ {0.6, 0.9} (72) | β = 1, η = 30, T = 1, momentum 0.9 | 4.2e-4 | 0.095 |
| MLP (residual ReLU, width 1,024, SGD) | lr ∈ {0.003 ... 0.3}, momentum ∈ {0, 0.9}, batch ∈ {64, 256} (20) | lr 0.3, momentum 0.9, batch 256 | 5.3e-4 | 0.120 |
| MLP, batch 64 (same samples per step as GWS) | as above | lr 0.1, momentum 0.9 | 6.4e-4 | 0.145 |

Relative MSE of GWS at 10,000 steps, momentum 0.9, by β/η and T ("div" = diverged):

| β/η | (β, η) | T = 0.1 | T = 1 | T = 3 | T = 10 |
|---|---|---|---|---|---|
| 0.001 | (0.1, 100) | div | 0.583 | 0.815 | 0.953 |
| 0.0033 | (0.1, 30) | div | 0.302 | 0.551 | 0.855 |
| 0.01 | (0.1, 10) | div | 0.187 | 0.320 | 0.650 |
| 0.01 | (0.3, 30) | div | 0.186 | 0.320 | 0.649 |
| 0.01 | (1, 100) | div | 0.186 | 0.320 | 0.649 |
| 0.033 | (1, 30) | div | **0.095** | 0.240 | 0.369 |
| 0.1 | (1, 10) | div | 0.102 | 0.164 | 0.287 |

Findings:
- Only β/η matters here too, so the paper's 3 × 3 grid over (β, η) holds 5 distinct settings.
- T = 0.1 diverges for every setting.
- With momentum 0.6 and at T = 3, the best ratio is the grid's largest (0.1), so the grid edge limits those rows.
- At 10,000 steps GWS is ahead of the best MLP. The paper's plot shows the MLP ahead at 200,000 steps (about 1.7e-4 against 2.6e-4). The 200,000-step runs decide this (PENDING).
- The paper's absolute values are not comparable with ours: it does not state the teacher's output scale, and our Var(y) = 0.0044 is already close to its parametric-softmax value (4e-3).

GPU note: one GWS step at 200,000 steps scans 12.8 million rows for only 64 queries. PyTorch's fused attention runs this on a single GPU block (50 times slower than the dense path on an A100), and the dense path needs about 90 A100-minutes. `rcdl/splitk_attention.py` adds a split-key Triton kernel (each program takes a chunk of rows; the chunk maxima are merged afterwards). It matches the dense path to fp32 rounding (`tests/test_splitk.py`) and is 2.3 times (tf32x3) to 6.6 times (tf32) faster. End-to-end check on the best GWS setting, 10,000 steps on the L4: test MSE 4.2175794e-4 (dense), 4.2175796e-4 (tf32x3), 4.2197e-4 (tf32, 0.05% higher). The 200,000-step run uses tf32.

## 8. Discrepancies between paper, notebooks and article

### 8.1 Method as derived vs method as run

1. **The density factor is dropped.** The functional-gradient step scales the new weight and value by 1/D_t (section 3 above). The experiments drop it (App. E.3, "for numerical stability"). Every row then gets the same step size, wherever the existing rows are dense or sparse.
2. **Two free step sizes replace one.** The derivation fixes the ratio of the value step to the weight step (σ_v^2). The code uses independent β and η. Only their ratio matters (section 7.1), so the layer has one effective step-size parameter, β/η, and the paper's grids over both waste most of their points. With η = 0 (all new rows get w = 0, so only the 16 initial rows normalize) the outputs grow without bound (training loss near 90) and single-pass accuracy falls from 59.4% to 27% (section 5). The unnormalized RBF layer, with its own step size, reaches 59.1%. So in the GWS rule the learned denominator sets the scale of the step; normalization as such is not what gives the accuracy.
3. **An undocumented clip.** The code clips <s_t, ḡ_t> to [-1, 1] before the max(0, ·) clip, with ḡ_t the batch-mean gradient. This equals clipping <s_t, g_t> to [-B, B], with B the batch size. The paper's Algorithm 2 does not mention it (the notebook does). It has no measurable effect in the single-pass setting: without the clip, seeds 1 and 2 give 59.07% and 59.57% (section 5).
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

Each direction states what it tests, the evidence from this reproduction that motivates it, and the questions it answers. Cost estimates are for the Colab GPUs used here.

### A. What does the single-pass result measure?
- Evidence: exact kernel ridge regression on raw pixels reaches 58.55%, only 0.9 points below single-pass GWS (section 6). The Nadaraya-Watson vote with the GWS kernel reaches 44.6%.
- Experiments: kernel ridge regression with the GWS layer-0 kernel on the features of a trained GWS layer 0 (does depth learn a better kernel?); kernel alignment or CKA between each layer's stored keys and the labels; the class purity of the rows each query retrieves.
- Questions: Is the margin over parametric MLPs a property of kernel machines on CIFAR pixels, or of the growing rule? Does a second GWS layer compute a better kernel, or only add capacity? Why do 43-65% of the stored rows end with w = 0, and what do these numerator-only rows contribute?

### B. From the theory to better learning rules
- B1. **Exact functional gradient.** Keep the 1/D_t factor (in log space) and tie β = η σ_v^2. Evidence: only β/η matters in the rule the paper runs (sections 7.1 and 7.3), so the derived rule removes one free parameter at no cost. Question: does the rule the paper derives beat the heuristic it runs, and does it remove the need to tune β/η per temperature?
- B2. **A second-order RCDL layer.** Sec. 4 maps the mesa layer to the FOOF optimizer. The kernel analogue of the mesa layer is kernel recursive least squares (Engel, Mannor and Meir, 2004). Its approximate-linear-dependence test adds a row only when the new key is not well represented by the stored keys, a principled form of the hybrid layer's distance threshold τ. Questions: does it give sublinear growth and better accuracy from one principle? Is the hybrid layer's need for per-τ tuning (section 5) a symptom of the missing second-order term?
- B3. **Positive weights without clipping.** Update w multiplicatively (exponentiated gradient, i.e. mirror descent on the mixture weights). Question: does it keep the rows that max(0, ·) now zeroes, and does that matter?

### C. Memory and compute
- C1. **Sparse retrieval.** The notebook shows that test-time top-K retrieval loses accuracy fast (37.5% with the top 1,000 of 50,016 rows, 59.2% dense). Train with top-K retrieval so that the model adapts, then serve with an approximate nearest-neighbour index.
- C2. **Kernels for the few-query regime.** The split-key kernel (section 7.3) shows that the cost profile of RCDL (few queries, millions of rows) does not fit the attention kernels of deep-learning libraries. The backward pass of the fused attention stalled the GWS-ViT run at 2M rows (section 5). Question: what does a full forward and backward split-key implementation change in the cost comparison of claim 21?
- C3. **Budget methods from online kernel learning:** NORMA truncation (Kivinen, Smola and Williamson, 2004), the Projectron (Orabona et al., 2008), budgeted SGD with merging (Wang, Crammer and Vucetic, 2012). Plot accuracy against stored rows for these and for the paper's hybrid layer.
- C4. **Distill the table into a parametric layer** (k-means keys plus least-squares values) and measure the loss.

### D. Continual learning (the article's "unforget" claim)
- Class-incremental split CIFAR-10 (5 tasks of 2 classes), single pass: GWS and RBF against MLP+Adam, with and without replay. Cheap on an L4 (one GWS pass takes 44 s).
- Questions: Do new rows overwhelm old ones through the shared denominator? Can a test-time temperature change recover old classes, as in the "silent memory" analogy (paper Sec. 6)? In the fact task (section 7.5), does the growing layer keep old facts because their rows stay unchanged, or does it lose them to the new rows' share of the softmax?

### E. Language: the biography experiment
- Section 7.5 shows that the plateau needs rare individuals, multi-token names and varying positions; the two-token task had none. Next steps: the full Zucchet et al. biographies (templates, 6 attributes, 64k individuals); the attention patching test (does the growing layer change the plateau, or only the speed after the circuit forms?); the article's claim that wide random-feature layers approach the growing RBF layer.
- Question: if a wide Cos-Bochner layer trained by SGD matches the growing layer, the benefit comes from the kernel's locality, not from growth or retrieval. This is the cleanest test of what "retrieval-centric" adds.

### F. Evaluation hygiene
- Select hyperparameters on a validation split for every method. The paper selected on the test set.
- Compare at matched inference compute and memory (section 7.6 gives the factors).
- Strengthen the parametric softmax baseline. Its keys start with length about 1, while the queries have length √d ≈ 55, so its attention starts almost uniform (logits near 0.006). Initialize its keys from data or rescale them, and retune it.
