# Amendment 3 to predictions_abc.md: the C5 size comparison is confounded (written before C5 is read)

The code review (workflow `review-looping-abc`, issue descent_cos_radial_dilution) found this. descent_cos = cos(h_k − h_k−1, −g_k−1) uses the full step. The readout LayerNorm ignores the scale of the state and a shift along the ones vector. So the gradient g is perpendicular to h_k−1 (measured |cos(g, h)| ≤ 1.4e-5), and any radial or mean part of the step lowers |descent_cos| without changing the loss read.

The size of that dilution differs between the FFNs. The tangent share |P u| / |u| was 0.87–1.00 for Cos and 0.71–0.85 for GeGLU at passes 3–4. Here u is the step and P removes the h and ones directions. In a CPU check the Cos-against-GeGLU order flipped once the radial part was removed (step 500: raw 0.338 against 0.322; tangent only 0.361 against 0.366).

**Change.** C5 keeps only its sign part: the median descent_cos is above 0 for every pass of Cos σ = 4 at K = 4, in both seeds. The comparison with GeGLU becomes descriptive and is reported with this caveat. The runs do not log the tangent share, and the models at the 30% point were not saved, so it cannot be corrected after the fact.

Other code-review findings, none of which affects a configured run:
- the gate-off probe would crash for an untied gated model;
- acc_by_alpha uses absolute steps, so for the a = 2 and a = 4 runs its keys are not multiples of the trained a;
- the probe medians are lower medians;
- two deep-supervision tests check shapes and F.cross_entropy instead of the training loss line.

They will be fixed after the runs. Changing rcdl/facts.py now would break the identical-code pairing of amendment 2.
