# Flexible PMC parameter estimates — text for the Results

Source: `tms_risk/behavior/scripts/extract_pmc_parameters.py` → `notes/data/pmcpars_*.flexible2.tsv`
Figure: `tms_risk/behavior/scripts/plot_pmc_parameters.py` → `notes/figures/pmc_parameters.pdf`
Model: `flexible2` (5 splines, TMS on both noise terms), evaluated against `bauer@ecc6454`.
All intervals are 95% credible intervals on the group-level posterior.

---

## Draft paragraph

The fitted priors were narrow and centred well below the payoffs they describe
(Supplementary Fig. S2a). For safe options the prior mean was 10.91 CHF
[8.24, 13.62] with a standard deviation of 2.83 CHF [2.13, 3.62], against an
average presented safe payoff of 15.82 CHF; for risky options the prior mean was
18.77 CHF [14.94, 23.02] with a standard deviation of 3.29 CHF [2.51, 4.13],
against an average presented risky payoff of 36.15 CHF. Because both prior means
lie below the bulk of the payoff distribution, Bayesian regression toward the prior
underestimates most payoffs, and does so more strongly the larger the payoff — for
safe options the perceived value exceeded the objective value only for the two
smallest payoffs (7 and 10 CHF) and fell below it for 14, 20 and 28 CHF, crossing
over almost exactly at the fitted prior mean.

Representational noise increased steeply with magnitude for both terms
(Supplementary Fig. S2b), from roughly 0.5 CHF at the smallest payoffs to about
5 CHF at the largest, so the model does not reduce to scalar variability. Parietal
cTBS increased *perceptual* noise, and did so only over the lower part of the
presented range (Supplementary Fig. S2c): +0.44 CHF [+0.03, +0.87] at a payoff of
7 CHF, +0.45 CHF at 10 CHF, +0.45 CHF at 14 CHF, +0.40 CHF at 20 CHF and +0.29 CHF
at 28 CHF, with credible intervals excluding zero throughout that range and no
credible effect for payoffs above roughly 30 CHF. The *memory* noise function showed
no credible effect of cTBS at any payoff (Supplementary Fig. S2c, dashed). Of the ten
stimulation contrasts on individual spline coefficients only one — the first
perceptual spline — individually excluded zero (−1.11 [−2.34, −0.02];
Supplementary Fig. S2d), which is expected given that neighbouring B-spline
coefficients are strongly correlated: the noise *function* is identified far better
than any single coefficient, which is why the curve in panel c is the interpretable
quantity.

---

## Numbers

| Parameter | Group posterior (CHF) | For reference |
|---|---|---|
| μ safe | 10.91 [8.24, 13.62] | mean presented safe payoff 15.82 |
| σ safe | 2.83 [2.13, 3.62] | |
| μ risky | 18.77 [14.94, 23.02] | mean presented risky payoff 36.15 |
| σ risky | 3.29 [2.51, 4.13] | |

cTBS effect on representational noise, IPS − vertex (CHF); `*` = 95% CrI excludes 0:

| Payoff | 7 | 10 | 14 | 20 | 28 | 56 | 112 |
|---|---|---|---|---|---|---|---|
| Perceptual | +0.44\* | +0.45\* | +0.45\* | +0.40\* | +0.29\* | −0.14 | −0.20 |
| Memory | −0.08 | −0.06 | −0.03 | +0.00 | +0.02 | −0.03 | −0.33 |

Spline coefficients: 10 of 20 have 95% CrIs excluding zero (all ten are intercepts,
i.e. the noise functions themselves are well identified; only one of the ten
stimulation contrasts is individually credible).

---

## Two caveats to settle before this goes in

**1. The panel titles of the published Fig. 4B do not describe what was fitted.**
Under `ecc6454`, `_get_trialwise_evidence_sd` built the first option's noise as
`softplus(Σ mem_i·b_i^mem + Σ mem_i·b_i^perc)` — the *memory* coefficients applied to
both spline bases — so the perceptual noise function entered only the
second-presented option, not "both options". The first option's noise is likewise not
`ν_perceptual + ν_memory`. The curves plotted here are the two fitted spline
functions as such, which is well defined either way; but any sentence describing
`ν_perceptual` as shared across options, or `ν_memory` as an increment on top of it,
does not match the code. Either refit (in progress) or restate.

**2. "TMS affects both perception and working memory" is a model-selection statement,
not a parameter statement.** That model wins on ELPD because both regressors improve
prediction, but only the perceptual function shows a credible cTBS effect in its own
posterior. Worth phrasing as "the best-fitting model allowed cTBS to affect both
noise terms; the credible effect was on perceptual noise for small payoffs".
