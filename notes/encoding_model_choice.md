# m1 vs m2 as the nPRF encoding model

Written 2026-08-03. Context read first: `notes/PROVENANCE.md`, `notes/figure_plan_briefing.md`.

**Verdict: m1.** It beats m0 decisively out of sample (+0.0244 cvR², 34/35 subjects) and
m2 is *worse* than m1 out of sample in the target ROI (−0.0055, p = 0.010, only 9/35
subjects favour m2). Per-session amplitude earns its keep; per-session tuning does not.

## Disclosure on analysis order

Parts (a)–(c) below were computed and this file saved **before** (d) was appended, as
instructed — with one exception I have to state plainly: the brain–behaviour
correlation in (d) had **already been computed earlier in the same session**, in
response to an earlier question in the conversation, before the instruction to defer it
arrived. So for that specific number the ordering was not achievable and I am not
claiming it. Everything in (a)–(c) is independent of it, and (a)–(c) alone determine
the verdict.

## What the three models are

`modeling/fit_regression_nprf.py::get_model`, a braincoder `RegressionGaussianPRF` with
`0 + C(session)` regressors on the listed parameters. Sessions 2 and 3 are the two TMS
sessions, so "per session" *is* the IPS-vs-vertex contrast.

| | amplitude | mu | sd | baseline | mu/sd grid points |
|---|---|---|---|---|---|
| m0 | pooled | pooled | pooled | pooled | 50 |
| **m1** | **per session** | pooled | pooled | pooled | 50 |
| m2 | per session | per session | per session | per session | **10** (`mus[::5]`) |

Verified on disk: under m1, max abs(IPS − vertex) is exactly **0.000** for mu, sd and
baseline (`prf_voxel_table.tsv` construction check). m1's Δamplitude is a gain change
at fixed tuning; m2's is estimated jointly with three other per-session parameters.

## Method and its gate

The pipeline's cross-validation (`fit_regression_nprf_cv.py`) is leave-one-run-out with
run N held out from **both** sessions at once, and it stores a single pooled cvR² per
fold. Per-session cvR² therefore had to be recomputed. Predictions were rebuilt
analytically, exactly as braincoder computes them
(`models/prf_1d.py::_basis_predictions_without_amplitude`, `utils/math.py::norm`,
`utils/stats.py::get_rsq`):

    pred = amplitude * exp(-0.5 * (x - mu)**2 / sd**2) + baseline,   x = log(n1)
    R2   = 1 - sum(resid**2) / sum((data - data.mean(0))**2)

**Gate:** before any cross-validated number is written, the in-sample R² rebuilt this
way must reproduce the stored `desc-r2` map. All **105 gates (35 subjects × 3 models)
passed**: min r = 0.99999999999, max abs difference = 9.5e-07.

**Caveat that applies to every cvR² number below.** `fit_regression_nprf_cv.py` runs
`max_n_iterations=10` in both gradient stages, where the main fit
(`fit_regression_nprf.py`) runs `10000`. The CV parameters are therefore far less
converged than the published parameter maps, and a model with more free parameters
(m2) has more to lose from that. This is a property of the existing pipeline, not of my
re-analysis, but it means the m1-vs-m2 CV gap is a lower bound on m2's disadvantage at
best and confounded at worst. Flagged rather than corrected.

## (a) Held-out prediction

Per subject × session, mean cvR² over voxels; the two sessions are averaged within
subject before the paired test, so n = 35. CIs are bootstrap over subjects (10k).
`notes/data/encoding_cvr2_by_session.tsv`.

**Mask = NPC12r (the target ROI):**

| contrast | Δ cvR² | 95% CI | t(34) | p | subjects favouring |
|---|---|---|---|---|---|
| m1 − m0 | **+0.02441** | [+0.01742, +0.03225] | +6.39 | <0.0001 | **34/35 → m1** |
| **m2 − m1** | **−0.00550** | [−0.00938, −0.00150] | −2.71 | **0.0104** | **9/35 → m2** |
| m2 − m0 | +0.01892 | [+0.00977, +0.02855] | +3.83 | 0.0005 | 26/35 → m2 |

By arm: m2 − m1 = −0.0066 (t = −2.55, p = 0.015) for IPS, −0.0044 (t = −1.08, p = 0.29)
for vertex. Fraction of NPC12r voxels with cvR² > 0: m0 0.075, m1 **0.092**, m2 0.090.

**Whole-brain mask:** the *mean* cvR² is unusable — it is dominated by
near-zero-variance voxels and runs to −1e12. Do not quote it. The interpretable summary
is the fraction of voxels with cvR² > 0: m0 0.0305, m1 **0.0351**, m2 0.0346 — the same
ordering as in the ROI.

This is consistent with, and stronger than, the earlier per-voxel win-count analysis
(`cvr2_model_comparison_intersect.tsv`), which had m2 nominally ahead of m1 in NPC12r
by +0.025 win-share at p = 0.73, i.e. no difference. On mean held-out R², m2 is
significantly *worse*.

## (b) Range restriction on the cTBS amplitude change

Between-subject SD of the per-subject Δamplitude (IPS − vertex).
`notes/data/encoding_param_shifts.tsv`.

| voxel set | statistic | m1 | m2 | SD ratio m1/m2 |
|---|---|---|---|---|
| all NPC12r | mean over voxels | mean −0.0911, SD 0.2887, t = −1.87, p = 0.071 | mean −0.0366, SD 0.2422, p = 0.38 | 1.19 (F p = 0.31) |
| all NPC12r | median over voxels | mean −0.0456, SD 0.1839 | mean −0.0158, SD 0.1204 | 1.53 (F p = 0.016) |
| NPC12r ∩ m0 cvR²>0 (n=25) | mean over voxels | **mean −0.1431, SD 0.2192, t(24) = −3.26, p = 0.0033** | mean −0.0829, **SD 1.1674**, p = 0.73 | **0.19 (F p < 0.0001)** |
| NPC12r ∩ m0 cvR²>0 (n=25) | median over voxels | **mean −0.0870, SD 0.1344, t(24) = −3.24, p = 0.0035** | mean −0.1828, SD 1.0933, p = 0.41 | 0.12 (F p < 0.0001) |

**The premise of the range-restriction worry is not supported.** m1's spread is not
"much smaller": on the full ROI it is slightly *larger* than m2's (ratio 1.19–1.53), and
on the model-neutral signal-voxel set m2's spread is **5–8× larger** than m1's — that
extra spread is noise, not signal, because m2's mean is not distinguishable from zero
while m1's is (p = 0.003).

The voxel set matters and I chose it to be model-neutral: NPC12r voxels with cvR² > 0
under **m0**, the model that is neither of the two being compared. Selecting on m1's or
m2's own cvR² would bias the comparison.

Agreement between the two models' estimate of the same quantity: r = +0.201 (p = 0.25)
on all NPC12r, r = +0.233 (p = 0.26) on the neutral set; sign agreement 20/35 and 14/25.
**They do not measure the same thing.**

## (c) Amplitude / dispersion trade-off in m2

Per subject, voxelwise correlation between the session-to-session change in amplitude
and the change in each other parameter, then tested across subjects.
`notes/data/encoding_amp_sd_tradeoff.tsv`. n = 14 subjects — the script required ≥20
voxels in the model-neutral set, which only 14 subjects have; this is a real limitation
of the neutral selection, not of the data.

For m0 and m1 the correlation is **undefined**: Δsd is identically zero by construction.

| m2, within subject | mean r | t(13) | p |
|---|---|---|---|
| Δamplitude vs **Δsd** | **+0.378** [+0.173, +0.535] | +3.92 | 0.0018 |
| Δamplitude vs **Δbaseline** | **−0.500** | −5.72 | 7.1e-05 |
| Δamplitude vs Δmu | −0.400 | −4.08 | 0.0013 |

m2's "amplitude change" is entangled with all three of its other per-session
parameters, most strongly with baseline (−0.50, the classic gain/offset trade-off) and
then with dispersion (+0.38, negative in only 1 of 14 subjects). So yes — a substantial
part of what m2 calls an amplitude change is a width and offset change.

## (d) Localisation-slope correlation (computed last; see the disclosure above)

Per-subject cTBS amplitude change against the behavioural noise-localisation slope
Δν_perceptual(7 CHF) − Δν_perceptual(28 CHF), n = 35.

| amplitude measure | `flexible1` | `flexible1nf` | `flexible2nf` |
|---|---|---|---|
| m2 | r = −0.332, p = .051 | **r = −0.432, p = .0096** | r = −0.332, p = .051 |
| **m1 (canonical)** | −0.250, p = .15 | **−0.083, p = .64** | −0.227, p = .19 |

The m2 correlation does not survive its own confounds: partialling out Δbaseline gives
r = −0.322 (p = 0.064); partialling out Δbaseline and Δmu gives r = −0.285 (p = 0.108).
The reverse partial is clean (Δbaseline given Δamplitude: r = +0.087, p = 0.63), so
amplitude is the better of the two predictors — it just is not significant once
disentangled, and it is absent under the model (a)–(c) select.

**This correlation should not be reported.** Choosing m2 for it, while the rest of the
paper runs on m1, would be selecting the encoding model by which one yields the result.

## The third variant (mu fixed, amplitude and dispersion free) — NOT RUN

The requested m3 — the model matching "cTBS reduces gain without moving tuning
preference" — requires new braincoder/TensorFlow fits: 35 subjects × (1 main fit + 6 CV
folds). It is not computable from stored derivatives. **Not attempted**, because the
GPU boxes offered (`sciencecloud_gpu…gpu4`) have neither `tensorflow` nor `braincoder`
installed and their 22 GB derivatives tree contains only `cogmodels.*` behavioural
traces — no `encoding_model2.*`, no `glm_stim1.*`, no `ips_masks`. The natural home is
sciencecluster (`tms_risk_cuda`, `/shares/zne.uzh/gdehol/ds-tmsrisk`), where the
existing m0/m1/m2 fits were produced and `modeling/slurm_jobs/` already has the
wrappers. Adding it needs a `model_label == 3` branch in
`fit_regression_nprf.py::get_model` and `get_grid`, and a matching `fixed_pars` branch
in both fit scripts.

## Scripts and paths

| Output | Script | Reads |
|---|---|---|
| `notes/data/encoding_cv_gate.tsv`, `encoding_cvr2_by_session.tsv`, `encoding_param_shifts.tsv`, `encoding_amp_sd_tradeoff.tsv` | `tms_risk/modeling/scripts/extract_encoding_model_cv.py` | `derivatives/encoding_model2.model-{0,1,2}.smoothed[.cv]/`, `derivatives/glm_stim1.denoise.smoothed/`, `derivatives/ips_masks/` |
| `notes/data/prf_voxel_table.tsv` | `tms_risk/modeling/scripts/extract_prf_voxel_table.py` | same parameter maps, no GLM betas |
