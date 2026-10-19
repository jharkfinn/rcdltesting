# Looping test: predictions written before the results

Written 2026-10-19 17:02 UTC. At that time 6 of the 18 facts runs had finished (all GeGLU), and no Cos-Bochner and no biography result had been seen.

Setup: the last FFN block runs K times with the same weights, h <- h + a f(LN(h)), a = 1, trained at that K.

Kernel view of Cos-Bochner: f(x) = W_o sqrt(2/D) cos(W x + b) approximates a kernel smoother sum_j k(x, x_j) v_j. Here x_j are training states, v_j are stored outputs, and k is a Gaussian kernel of width sigma (facts) or 2 T sqrt(d) (biographies).

| ID | Prediction | Measure | Refuted if |
|---|---|---|---|
| L1 | Looping helps the narrow Cos-Bochner kernel (sigma 4) more than GeGLU on the facts task. | final accuracy at K = 2, 4 minus K = 1, same seed | the Cos gain is not larger than the GeGLU gain in both seeds |
| L2 | Looping makes Cos-Bochner more robust to state noise (clean-up toward a stored state), more than GeGLU. | accuracy at noise 0.25 minus final accuracy, K = 4 against K = 1 | the Cos change is not better than GeGLU's |
| L3 | The narrow kernel has the larger per-pass Jacobian (sharper kernel, larger Lipschitz constant). | median spectral norm of J_k, sigma 4 against sigma 8 and GeGLU | sigma 4 is not the largest |
| L4 | Biographies: looping widens the seen / held-out gap of Cos-Bochner at T = 1 (it amplifies memorization of wording). | seen minus held-out accuracy, K = 2, 4 against K = 1 | the gap does not grow in either seed |
| L5 | Biographies: Cos-Bochner at T = 2 stays wording-invariant under looping. | seen minus held-out < 2 points at every K | the gap exceeds 2 points |

Already seen for GeGLU (facts, final accuracy, seeds 1 / 2): K = 1 0.944 / 0.931; K = 2 0.938 / 0.929; K = 4 0.929 / 0.923. Looping slightly hurts GeGLU here.
