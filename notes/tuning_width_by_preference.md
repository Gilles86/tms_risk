# Tuning width as a function of preferred numerosity, in this dataset

Written 2026-08-03. Context read first: `notes/PROVENANCE.md`,
`notes/figure_plan_briefing.md`. All numbers are statistics of *these* 35 subjects, not
citations.

**Headline: the answer flips with units, and the paper must say which it means.** In
*absolute* (linear numerosity) units, low-preference nPRFs are narrower — trivially, a
scale effect. In *relative* (log) units, low-preference nPRFs are **wider**, i.e. less
sharply tuned, and that runs opposite to the "sharper tuning at small numerosities"
reading. What *is* solidly true is that the tuning **mass is concentrated at small
numerosities**.

## Units, stated once

The paradigm is x = log(n1), so under the canonical model m1:

- `mu` is **log** preferred numerosity; preferred numerosity = exp(mu).
- `sd` is a tuning width **in log units** — a dimensionless Weber-like coefficient, not
  a numerosity. Constant `sd` across `mu` means scale-invariant (Weber) tuning.
- Approximate width in linear numerosity units ≈ pref_n × sd.

Under m1 `mu` and `sd` are shared across sessions by construction (verified: max abs
IPS − vertex = 0.000), so tuning preference and width are single per-voxel numbers.
This is why the analysis uses m1 — see `notes/encoding_model_choice.md`.

## Mask and threshold

Voxels are **cvR² > 0 under m1** (out-of-sample, from
`encoding_model2.model-1.smoothed.cv/sub-XX/sub-XX_desc-cvr2...`), within each ROI,
after dropping degenerate fits (pref_n outside 1–300, sd outside 0.01–20). Threshold
sensitivity is reported in (b). Three masks: **NPC12r** (the numerosity ROI),
**NPCr2cm-cluster** (the stimulation site), and the union of all numerosity ROIs
(NPC12r, NPCr, NPCl, NF1, NTO, and the 2 cm variants).

## (a) Dispersion regressed on preferred numerosity

Pooled slopes are reported because they were requested, but they pool voxels across
subjects and are **pseudoreplicated** — the per-subject version is the inferential one
(slope fitted within each subject, then tested across the 35). CIs are bootstrap over
subjects.

| mask | n voxels | `sd` ~ **linear** pref_n, per-subject slope | `sd` ~ **log** pref_n, per-subject slope |
|---|---|---|---|
| NPC12r | 3632 | +0.00019 [−0.0167, +0.0150], t(34) = +0.02, p = 0.98 | **−0.245 [−0.481, −0.023], t(34) = −2.05, p = 0.049** |
| **NPCr2cm-cluster** | 2317 | −0.0167 [−0.0505, +0.0062], t(33) = −1.10, p = 0.28 | **−0.369 [−0.615, −0.137], t(33) = −3.05, p = 0.0045** |
| all numerosity ROIs | 16388 | **+0.0111 [+0.0077, +0.0144], t(34) = +6.31, p = 3.4e-07** | −0.080 [−0.213, +0.047], t(34) = −1.18, p = 0.24 |

Pooled (descriptive) slopes for reference: NPC12r +0.0116 linear (r = +0.161) and
+0.0853 log (r = +0.041); all-ROI +0.0106 linear (r = +0.166) and +0.0481 log
(r = +0.024). Note the pooled log slope is **positive** while the per-subject mean is
**negative** in every mask — a Simpson's-paradox reversal, and the reason the pooled
number must not be quoted.

Correlation form, per subject, Spearman(log pref_n, `sd`):

| mask | mean rho | t | p | sign |
|---|---|---|---|---|
| NPC12r | −0.117 | t(34) = −2.72 | 0.010 | positive in 11/35 |
| NPCr2cm-cluster | −0.157 | t(33) = −3.77 | 0.00064 | positive in 10/34 |
| all numerosity ROIs | −0.032 | t(34) = −0.98 | 0.33 | positive in 17/35 |

And the linear-width version, Spearman(pref_n, pref_n × `sd`): mean rho **+0.43**
(NPC12r), +0.40 (2 cm), +0.54 (all ROIs) — strongly positive, i.e. absolute width grows
with preferred numerosity.

**Reading.** Log-space width *decreases* with log preferred numerosity, so relative
tuning precision is **better at large** numerosities, not at small ones. Absolute width
grows with preferred numerosity, so linear tuning is sharper at small numerosities.
Both are true; they are the same fact in two units. Since `sd` is the parameter the
model actually fits and the noise function in the behavioural model is compared on a
log/ratio axis, the log-space statement is the one that is *not* a scale artifact.

## (b) Distribution of preferred numerosity

| mask | threshold | n voxels | median | IQR |
|---|---|---|---|---|
| NPC12r | cvR² > 0 | 3632 | 9.67 | [7.10, 15.22] |
| NPC12r | cvR² > 0.02 | 2120 | 9.70 | [7.33, 14.38] |
| NPC12r | cvR² > 0.05 | 1096 | 9.70 | [7.38, 13.40] |
| NPC12r | cvR² > 0.10 | 483 | 9.28 | [6.43, 12.52] |
| NPCr2cm-cluster | cvR² > 0 | 2317 | 9.41 | [6.96, 14.83] |
| NPCr2cm-cluster | cvR² > 0.05 | 690 | 9.33 | [7.21, 12.68] |
| NPCr2cm-cluster | cvR² > 0.10 | 288 | 8.69 | **[5.15, 11.73]** |

**Presented numerosities: IQR [13, 20, 30] — reproduces the paper's [13, 30] exactly**
(`get_all_behavior`, n1 and n2 identical).

### FINAL ANSWER (2026-08-03): the IQR is [6.0, 10.5]. The published [6, 10] stands.

Recipe: old tree `encoding_model.denoise.smoothed`, ROI `NPCr2cm-cluster`, notebook
cell-7 mask (cvR² > 0 in *either* arm), preferred numerosity within the plotted 0–30
range → **IQR [6.00, 10.48], median 8.45**. Against a presented IQR of **[13, 30]**,
median 20.

Sensitivity — the median is stable everywhere, only the upper quartile moves:

| variant | IQR | median |
|---|---|---|
| **notebook mask, pref < 30 (plotted range)** | **[6.00, 10.48]** | 8.45 |
| row-wise cvR² > 0, no range restriction | [6.00, 10.45] | 8.42 |
| row-wise cvR² > 0, pref < 30 | [5.84, 9.85] | 8.21 |
| notebook mask, no range restriction | [6.13, 12.65] | 8.89 |

Three of the four round to [6, 10]; only including voxels above the plotted axis pushes
the upper quartile to 12.7. **The earlier decision to report the model-1 value
[7.10, 15.22] is superseded** — that comes from `encoding_model2.model-1`, which is not
the source the paper uses and cannot produce four of the five statistics in the same
paragraph (`mu`, `sd`, `r2`, `cvr2` are session-invariant there).

### How the published IQR was located

Initially flagged as not reproducing, because the tables above use the **current**
canonical tree `encoding_model2.model-1`. It reproduces exactly from the **older
log-space tree** `encoding_model.denoise.smoothed` — the source CLAUDE.md/PROVENANCE
identify for the paper's Fig-2 parameter numbers, which commit `ba58cb1` stopped using
but which is still on local disk (46 subject dirs, 520 NIfTIs, genuine per-session
`mu`).

**Recipe: `NPCr2cm-cluster` (the 2 cm stimulation cluster), cvR² > 0 from
`encoding_model.cv.denoise.smoothed`, exp(mu) pooled across voxels and sessions
(2089 voxel-sessions, 35 subjects) → IQR [6.00, 10.45], median 8.42.** That is the
published [6, 10].

Nearby variants, same tree, for sensitivity:

| ROI | threshold | pooled IQR | median | n |
|---|---|---|---|---|
| **NPCr2cm-cluster** | **cvR² > 0** | **[6.00, 10.45]** | **8.42** | 2089 |
| NPCr2cm-cluster | cvR² > 0, 1 < pref_n < 50 | [6.30, 10.21] | 8.56 | 1789 |
| NPCr2cm-cluster | r² > 0.05 | [7.82, 19.87] | 9.94 | 6209 |
| NPCr2cm-cluster | none | [8.00, 30.28] | 12.14 | 22044 |
| NPC12r | cvR² > 0 | [6.18, 11.14] | 8.67 | 3195 |

Two things follow, both worth stating in the paper's provenance:

1. **The threshold does the work.** Unthresholded, the same voxels give [8.00, 30.28] —
   an upper quartile three times larger. The [6, 10] claim depends on the cvR² > 0
   selection, which should be stated in the caption.
2. **The number changes if Fig 2B is regenerated with current code.** Under
   `encoding_model2.model-1` the same quantity is **[7.10, 15.22]** (NPC12r) or
   [6.96, 14.83] (2 cm cluster). So [6, 10] is correct *as published* but is not
   reproducible from the tree every current script reads. Either keep the old tree as
   the documented source for that sentence, or re-quote the model-1 value.

Per-subject medians (NPC12r): mean 12.24, range [6.68, 35.15], SD 6.82. Subjects at the
extremes: **sub-47 (35.15) and sub-35 (30.52)** sit far above the group; sub-62 (6.73)
and sub-63 (6.68) far below. sub-47 and sub-35 are worth inspecting — their nPRFs prefer
numerosities near the top of the presented range, unlike everyone else.

## (c) Coverage / resolution across the numerosity grid

**What was computed, exactly:** for each point g on a 25-point geometric grid from 7 to
112, the density-weighted precision

    D(g) = sum over voxels of  (1 / sd_v) * exp(-0.5 * ((log g - mu_v) / sd_v)**2)

normalised to sum to 1 within subject, then averaged across the 35 subjects. It is
tuning density weighted by inverse width. **This is not Fisher information** — it uses
no noise model and no amplitude, so it says where tuning *sits* and how sharp it is,
not how well the population supports discrimination.

| numerosity | 7.0 | 11.1 | 17.6 | 28.0 | 44.4 | 70.6 | 112.0 |
|---|---|---|---|---|---|---|---|
| normalised D | 0.1080 | 0.0612 | 0.0399 | 0.0260 | 0.0223 | 0.0208 | 0.0147 |
| SEM | 0.0158 | 0.0046 | 0.0030 | 0.0018 | 0.0022 | 0.0028 | 0.0022 |

Monotonically decreasing, by a factor of **7.3× from 7 to 112**. Mass at n ≤ 20 is
**0.662** vs **0.338** above, t(34) = +5.24, **p = 8.4e-06**.

**This is the one solid, dataset-derived statistic backing "the code is concentrated at
small numerosities."** It is a statement about where tuning sits, and it holds strongly.

### Why the proper Fisher-information / decoding version is not here

The requested simulation-based version (Prat-Carrabin et al. approach) was attempted
from the existing `derivatives/monte_carlo_decode.denoise.spherical/` output, which is
already exactly that computation — `monte_carlo_decode.py` fits Ω with `ResidualFitter`
(`method='t'`, `spherical=True`, i.e. diagonal Ω), simulates 1000 patterns per stimulus
from t(f(s), Ω, dof), and decodes with `get_expected_uncertainty` over
`stim_grid = np.arange(7, 112)` with a flat prior. **The point estimate is the posterior
mean** (braincoder `base.py`, column `E`), which answers the "check and match it"
question. Coverage is complete for NPC12r: 35 subjects × 2 sessions.

**It cannot answer the question as it stands, because the decoder is collapsed.**
Mean decoded value, vertex sessions, averaged over subjects (nvoxels-100):

| true s | 7 | 28 | 56 | 111 |
|---|---|---|---|---|
| mean decoded | 59.06 | 59.06 | 59.89 | 59.96 |
| bias | +31.06 | +31.06 | +3.89 | −51.04 |

The decoder returns approximately the grid centre (~59–62) regardless of the true
stimulus, so E(s) is essentially |60 − s| and any log-log slope fitted to it measures
the bounded grid, not the neural code. The fitted slopes bear this out — they are
+0.11 (nvoxels-500), −0.12 (ses1cvr2), i.e. near zero or negative, versus the Weber
prediction of 1.0 and the behavioural 0.45; both differ from 0.45 at p < 1e-4, but the
comparison is meaningless given the collapse. This is the documented "decoder collapse"
in CLAUDE.md (median log-slope of decoded vs true ≈ 0.05).

Consistently, the cTBS contrast ΔE(s) = E_IPS − E_vertex has a bootstrap CI spanning
zero at every stimulus and every voxel selection (e.g. nvoxels-100: +0.54 [−2.87, +4.08]
at s = 7; −0.07 [−0.36, +0.20] at s = 28).

Also note `var_E` is the variance of the point estimate **across simulations**, not the
mean posterior SD — so the requested "mean posterior SD as a second precision measure"
is not in these files and needs a rerun.

Extending the evaluation grid (part ii) will not rescue this: if the likelihood is
near-flat, widening the grid just moves the centre the posterior mean collapses to. The
prerequisite is a decoder that carries information at all — which is a voxel-selection
and noise-model problem, not a grid problem.

## (d) Does the relation hold within IPS specifically?

Yes, and it is **strongest there**. The log-space relation is significant in the
stimulation ROI (NPCr2cm-cluster: slope −0.369, p = 0.0045; rho −0.157, p = 0.00064),
weaker but present in NPC12r (−0.245, p = 0.049; rho −0.117, p = 0.010), and **absent**
when all numerosity ROIs are pooled (−0.080, p = 0.24; rho −0.032, p = 0.33). So it is a
property of parietal numerosity-tuned cortex at the stimulation site, not of
numerosity-tuned cortex generally.

## Scripts and paths

| Output | Script | Reads |
|---|---|---|
| `notes/data/prf_voxel_table.tsv` | `tms_risk/modeling/scripts/extract_prf_voxel_table.py` | `derivatives/encoding_model2.model-{0,1,2}.smoothed[.cv]/`, `derivatives/ips_masks/` |
| (c) decoding section | analysis over `derivatives/monte_carlo_decode.denoise.spherical/` | produced by `tms_risk/modeling/monte_carlo_decode.py` |
