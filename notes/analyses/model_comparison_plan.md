# Where the model comparison goes in v12

## What v12 currently says, and what has to change

l. 80 of the draft:

> "…compared them qualitatively, using targeted posterior predictive checks…
> and quantitatively, using leave-one-out expected log-predictive density
> (ELPD…), which also confirmed the power law was superior to Weber function
> in these models (ΔELPD=35.6, dSE 8.8; See Table XXXX)."

Two problems.

**1. It implies ELPD picked the placement. It did not.** Referenced to the
reported model (`log-power-n1n2`), the three placements the sentence names are
inside each other's noise:

| model | ΔELPD | dSE |
|---|---|---|
| Both options (reported) | — | — |
| Perceptual + memory | −3.6 | 8.2 |
| Perceptual only | −4.9 | 8.7 |
| **Second option only** | **−27.0** | 6.7 |
| **First option only** | **−24.0** | 6.6 |
| **Memory only** | **−44.7** | 10.0 |
| **No cTBS effect** | **−100.3** | 13.0 |

So ELPD **does** decide four things and **not** the fifth:

* cTBS moves the noise function at all — 100 nats, 7.7 dSE
* it is not the memory stage — 45 nats, 4.5 dSE
* it is not confined to one presentation position — 24 and 27 nats, 3.6–4.0 dSE
* a payoff-dependent noise function is needed — Weber 34 nats, 3.6 dSE
* **which channel carries it — unresolved**, the top three within 0.6 dSE

The reported model is chosen on the grounds the draft already gives (most
flexible, orthogonal stimulation parameters, tests order-specificity within the
model), not on ELPD. The sentence should say that.

**2. The Weber number is stale.** ΔELPD 35.6 / dSE 8.8 is a raw-choice-rule
figure. Under the consistent rule it is **34.1 / dSE 9.6** against the reported
model, and −112.4 / 14.6 for Weber with no cTBS term.

## Proposed placement

**Main text — replace l. 80 with two sentences** and cite the supplement:

> We fitted power-law PMCM models in which cTBS could change the noise on the
> first-presented option, the second-presented option, or both, together with
> models placing the effect on a perceptual or memory stage, models holding the
> noise function to Weber's law, and a model with no cTBS effect at all
> (Supplementary Figure S3, Supplementary Table S1). Leave-one-out ELPD
> (Vehtari et al., 2017) establishes that cTBS changed the noise function at
> all (ΔELPD = 100.3, dSE 13.0 against a model without a stimulation
> parameter), that a payoff-dependent noise function is required (34.1, dSE 9.6
> against Weber's law), that the effect is not confined to the memory stage
> (44.7, dSE 10.0) and that restricting it to a single presentation position
> fits worse in either direction (24.0 and 27.0, dSE ≈ 6.6). It does not
> distinguish among the remaining placements, which fall within 0.6 dSE of one
> another; we therefore report the most flexible of them, which is also the
> only one that tests order-specificity within the model rather than by
> assumption.

Then keep the existing sentences about orthogonality and convergence unchanged.

**Supplementary Figure S3 — the ELPD ladder.** Two panels, already built:
`notes/figures/supp_elpd_ladder.pdf`. Panel a is placement (where the effect
acts), panel b is shape (what form the noise function takes). Each bar is the
PAIRED difference against the reported model with the standard error of that
difference; rungs that miss the convergence gate carry their own r̂ and ESS
rather than a verdict, and only genuinely unusable fits are greyed.

**Supplementary Figure S4 — the spline ladder.** `supp_elpd_splines.pdf`. Two
anchors against three and five, with a no-cTBS row as its own control. It
answers a different question from S3 — resolution rather than shape — and
mixing three- and five-anchor channels into a shape comparison invites the
reader to read the ranking as flexibility.

**Supplementary Table S1** stays where it is but is regenerated
(`make_supp_table1.py`): converged models only, ELPD, ΔELPD with dSE, effective
parameters, r̂ and ESS, and an explicit list of the models excluded for
non-convergence with their diagnostics.

## One thing to decide

The reported model does not currently pass the convergence gate at PRIOR_SPEC
defaults (r̂ 1.12, ESS 42) — the draft's "r̂ ≤ 1.002, ESS ≥ 1727" is a
raw-choice-rule number and must be replaced. Either report it with a tightened
noise prior (τ_noise, sensitivity in the supplement) or report the fit at
default priors with its diagnostics stated. Until that is settled, S3's
"Both options" rung is greyed, which reads oddly for the reported model. The
caption should say the comparison is at PRIOR_SPEC defaults and that the
reported fit uses the prior given in Methods.
