# The cTBS amplitude effect, and why voxel selection was hiding it

Written 2026-08-03. Companion to `notes/encoding_model_choice.md` (which settles m1 vs
m2) and `notes/tuning_width_by_preference.md` (which settles the preferred-numerosity
IQR). Figure: `tms_risk/modeling/scripts/plot_figure2.py`.

## Bottom line

**The effect is there. Thresholding on cvR² was hiding it.** In the anatomically defined
2 cm stimulation cluster with **no functional selection at all**:

    NPCr2cm-cluster, all 11022 voxels, per-subject mean, paired across 35 subjects
    Delta amplitude (IPS - vertex) = -0.1470
    t(34) = -2.757,  p_two = 0.0093,  p_one = 0.0047,  Wilcoxon p = 0.0112
    negative in 23/35 subjects

## It is not a lucky cell — the sign is essentially universal

72 combinations of ROI × selection × aggregation were computed
(`scratchpad/amp_sensitivity.py`). **The difference is negative in 71 of 72**, median
−0.0757. What varies is only whether it clears significance: 11/72 cells reach
p₂ < .05, 17/72 reach p₁ < .05. Every cell that does is a cell with a weak threshold or
none, and the ranking is systematic — the top of the table is dominated by
`selection = none`.

## The anatomical gradient is the real argument

No functional threshold, per-subject mean, paired across subjects. This is a specificity
result no choice of threshold can manufacture:

| ROI | Δ amplitude | t(34) | p₂ | Wilcoxon |
|---|---|---|---|---|
| **NPCr2cm-cluster** (the stimulation site) | **−0.147** | **−2.757** | **0.0093** | 0.0112 |
| NPCr | −0.092 | −2.093 | 0.0439 | 0.0495 |
| NPC12r | −0.091 | −1.867 | 0.0706 | 0.0840 |
| NF1 | −0.059 | −1.633 | 0.1116 | 0.0557 |
| NPCl (contralateral) | −0.036 | −0.850 | 0.4012 | 0.6918 |
| NTO | −0.034 | −0.824 | 0.4159 | 0.7038 |

Largest at the targeted tissue, absent contralaterally and in NTO.

## Why a cvR² threshold costs power here

**Session-1 and session-2/3 cvR² are uncorrelated: r = +0.011.**

Session 1 is the baseline session, before any TMS, so a threshold on session-1 cvR² is
the only selection genuinely independent of the IPS-vs-vertex contrast. A genuine
session-1 cvR² exists **only in the older tree**,
`encoding_model.cv.denoise.smoothed/sub-XX/ses-1/` (238 files, 34 subjects).
`encoding_model2` fits sessions 2 and 3 only and its cvR² has no session axis, so the
`ses1cvr2` selection in `monte_carlo_decode.py` (`n_voxels == 0`) is a **no-op** for that
tree — it returns the pooled value. Worth fixing or renaming.

That near-zero correlation means the voxels that fit well at baseline are not the voxels
that fit well later, so **any cvR² threshold is close to an arbitrary 10–20% subsample**.
It discards 80% of the data and buys nothing. (It is also the same finding, from another
angle, as the decoder collapse in `notes/tuning_width_by_preference.md` §c: if
cross-validated fit does not replicate across sessions, most voxels carry no stable
signal.)

Selecting on session-1 cvR², the effect keeps its size but loses significance to the
smaller voxel count — consistent with a power cost rather than a biased selection:

| selection | ROI | Δ | t | p₂ |
|---|---|---|---|---|
| none | NPCr2cm-cluster | −0.137 | −2.542 (df 33) | 0.0159 |
| session-1 cvR² > 0 | NPCr2cm-cluster | −0.134 | −1.642 | 0.1101 |
| session-1 cvR² > 0.02 | NPCr2cm-cluster | −0.150 | −1.534 | 0.1358 |

(34 subjects here, not 35 — one lacks session-1 CV output.)

**A hypothesis that turned out to be WRONG, recorded so it is not re-run:** I expected
pooled cvR² to select *against* effect-carrying voxels, on the reasoning that a large
IPS-vertex difference degrades a fit pooled over both sessions. The opposite holds —
per-subject Spearman(cvR², |Δamplitude|) = **+0.22, p = 1e-8**, and every cvR² quartile
shows the effect (p = 0.003–0.033). The threshold is not biased against the effect; it
just throws away power.

## Consequence for the figure

`plot_figure2.py` therefore uses **two deliberately different voxel sets**, which needs a
caption sentence:

- **Panel a (amplitude)**: the whole anatomical ROI, no threshold. Amplitude is defined
  for every voxel.
- **Panel b (preferred numerosity)**: cvR² > 0 plus a degenerate-fit drop. Preferred
  numerosity is only interpretable where the model fits — otherwise ~25% of voxels sit
  on the mu grid floor (`pref_n ≤ 1`; the grid is `log(linspace(5, 80))`, and gradient
  descent runs unbounded after the grid stage, so 34.6% land below 5 and 5.9% above 80).

## What the figure replicates, and the one deviation

`plot_figure2.py` reproduces cell 17 of `modeling/notebooks/analyze_encoding_model.ipynb`
(saved there as `derivatives/figures/amplitude_vs_preferred_numerosity.pdf`): linear
x-axis with `xlim(0, 30)`, amplitude binned by `np.arange(0, 50, 5)` midpoints, preferred
numerosity as a grey step histogram with unit bins `np.arange(1, 100)`, the stimulus
distribution as a dashed black KDE over `n1`, GridSpec 5:3, legends in both panels.

**One deliberate deviation.** The original passed voxel-level rows straight to
`sns.lineplot`, so its band is a bootstrap over ~10⁴ voxels rather than over 35 subjects
— pseudoreplication, roughly an order of magnitude too narrow. Here the median is taken
within subject × bin first and the band is ±1 SEM across subjects. `--voxel_level_ci`
reproduces the original band as published.

## RESOLVED: the published paragraph reproduces exactly

An earlier version of this note said the published Fig-2B numbers could not be
reproduced. **That was my masking error, not a problem with the paper.** All five
statistics in the Figure-2 paragraph reproduce to 3–4 decimals via
`modeling/scripts/reproduce_figure2_stats.py`:

| # | statistic | published | reproduced |
|---|---|---|---|
| 1 | amplitude | 1.3015 → 1.0416, t(34) = 1.9924, p₁ = .027 | 1.3015 → 1.0416, t = −1.9928, p₁ = .0272 |
| 2 | preferred numerosity | 14.75 → 17.79, t(34) = 1.0069, p₂ = .32 | 14.75 → 17.79, t = +1.0088, p₂ = .3202 |
| 3 | dispersion | 0.7733 → 0.9083, t(34) = 1.2780, p₂ = .21 | 0.7733 → 0.9083, t = +1.2756, p₂ = .2108 |
| 4 | explained variance | 0.0681 → 0.0493, t(34) = 2.0644, p₁ = .023 | 0.0681 → 0.0493, t = −2.0646, p₁ = .0233 |
| 5 | proportion cvR² > 0 | 0.1113 → 0.0752, t(34) = 1.9893, p₁ = .027 | 0.1112 → 0.0752, t = −1.9880, p₁ = .0275 |

**The step I had wrong is the mask.** Notebook cell 7 is

    cvr2 = pars.droplevel('session')['cvr2'].unstack('stimulation_condition')
    mask = (cvr2 > 0.0).any(axis=1)

i.e. keep a voxel if cvR² > 0 in **either** arm, which keeps the pair intact. I had been
thresholding row-wise (keep a voxel-session where *its own* cvR² > 0), which drops
different voxels from the two arms and destroys the pairing. None of the numbers come
out that way. Statistic [5] uses **no** mask at all — it is a property of every voxel in
the ROI — which is why it was the one that reproduced before I found the bug.

Descriptives are the **median across subjects of the per-subject mean over voxels**; the
test is on the means. And the manuscript's natural-space 14.8 / 17.8 is specifically the
median of the per-subject mean of **exp(mu)** — taking the mean in log space and then
exponentiating gives 10.10 / 11.28 instead, with an identical t and p.

**So the paragraph is sound as written**, with the two wording issues already logged in
`notes/v8_stats_check.md` §5 standing: for both `mu` and `sd` the larger value is IPS,
not vertex, so "from 0.9 to 0.77" inverts the direction in a paragraph whose convention
is vertex → parietal.

**But note what it depends on.** All five come from the OLD log-space tree. Commit
`ba58cb1` switched `analyze_encoding_model.ipynb` to
`get_prf_parameters(model_label=1)`, under which `mu`, `sd`, `r2` and `cvr2` are
session-invariant **by construction** — so four of the five cannot be computed there at
all, and the specificity argument ("amplitude but not tuning") would be vacuous rather
than supported. The old tree is the source of record for this paragraph and must not be
pruned.
