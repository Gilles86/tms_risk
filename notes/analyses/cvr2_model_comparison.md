# cvR² model comparison for nPRF encoding model

## Question

Three nPRF encoding-model variants are fit to the TMS dataset
(`encoding_model2.model-{0,1,2}.smoothed`):

| Model | Parametrisation                                         | What it lets vary across sessions |
|------:|---------------------------------------------------------|-----------------------------------|
| m0    | Pooled across sessions — one set of (μ, σ, amp, base)   | nothing                           |
| m1    | Amplitude varies per session, (μ, σ, baseline) pooled   | amplitude                         |
| m2    | Full session interaction on μ, σ, amplitude, baseline   | everything                        |

m1 is what every downstream decoding/Fisher/mc_decode script uses (it's
the paper's Fig 2 model). But we have *never* checked whether m1 is
actually better than m0 in cross-validated R² — or whether m2 adds
anything over m1. That's the question.

## Approach (user-specified, 2026-05-27)

1. **Null model = "predict the per-voxel training-set mean".** cvR² of
   the null model on a held-out fold is approximately zero (or
   slightly negative due to train-vs-test mean drift). It's the
   per-voxel baseline that any encoding model has to beat to be
   informative.

2. **Filter to non-noise voxels.** A voxel is "non-noise" if **any** of
   m0 / m1 / m2 beats the null on cvR². I.e. the encoding model
   carries some signal in that voxel; voxels where all three models
   underperform the null are dropped.

3. **Per-subject win counts on the non-noise pool.** For each
   non-noise voxel, identify which of m0 / m1 / m2 has the highest
   cvR². For each subject, compute the proportion of non-noise voxels
   where each model is the winner.

4. **Group-level summary.** Average win proportions across subjects.
   If m1 wins everywhere uniformly, the paper's choice is correct and
   m2's extra session-interaction parameters don't earn their keep on
   held-out data. If m2 wins materially, the per-session
   (μ, σ, baseline) shifts matter and the current encoding model is
   underfit — which is a candidate explanation for the
   decoder-collapse pattern in the spherical-Ω expected-uncertainty
   figure.

## Plot

Stacked-or-swarm figure with one bar per model (m0, m1, m2):

- y-axis: proportion of non-noise voxels won (0 to 1).
- x-axis (or hue): model variant.
- One swarm point per subject; bar = group mean ± SEM.
- Sub-panel by ROI (NPC12r, V1 / V2 for sanity, whole brain).

Also worth: a stacked-bar version per subject (one column per subject,
stacked m0/m1/m2 contributions summing to 1) to show subject-level
variability.

## Practical bits

- Use the per-fold cvR² (`run-X_desc-cvr2.optim_space-T1w_pars.nii.gz`)
  averaged across runs, OR the global cvR²
  (`sub-XX_desc-cvr2.optim_space-T1w_pars.nii.gz`). Both are on disk
  for all 35 subjects and all 3 model variants.
- The null cvR² is approximately 0; using `cvR² > 0` (any model) as
  the non-noise threshold is a clean default. A stricter threshold
  (e.g., cvR² > 0.02) selects more confidently signal-carrying voxels.
- Subject set: same 35-subject intersection used elsewhere
  (tms_keys.yml ∩ encoding_model2 on disk).

## Why this matters for the decoding work

The decoder-collapse diagnostic showed median per-subject log-slope of
decoded vs true ≈ 0.05 — most voxels carry near-zero signal once
projected through m1. If m2 wins on a substantial fraction of
non-noise voxels, the right downstream move is to refit the residual
covariance + redo monte_carlo_decode against m2 — possibly fixing the
collapse.

If m1 wins overwhelmingly, the encoding model isn't the bottleneck
and we should look at voxel selection (FDR over top-N) or ROI choice
instead.
