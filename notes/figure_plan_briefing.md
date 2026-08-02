# What the reanalysis established — briefing for the figure plan

Written 2026-08-02. Everything below is from the refits, not the preprint. Numbers are
group-level posteriors unless stated. Source data for all of it is in `notes/data/*.tsv`;
provenance in `notes/PROVENANCE.md`.

## 0. Why the preprint's model results cannot be reported as they stand

The published `flexible2` fit (bauer@`ecc6454`) built the first option's noise from the
**memory spline coefficients twice**, so the perceptual noise function never entered the
first-presented option at all. The Methods describe ν₁ = ν_perceptual + ν_memory. So the
published "perceptual vs memory" decomposition is not what the labels say — read that
fit as a relabelled family-1 (per-position) fit. All results below come from refits with
the intended composition, ν₁ = softplus(η_mem + η_perc), ν₂ = softplus(η_perc).

## 1. Which model wins, and by how much

Sixteen-cell comparison, two noise families × four cTBS loci, all refit. ELPD (LOO):

| model | ELPD | cost vs best |
|---|---|---|
| **Family 2, cTBS on perceptual noise only** | **−4157.7** | — |
| Family 2, cTBS on both terms | −4159.7 | +2.0 ± 3.9 |
| Family 1, cTBS on both options | −4184.6 | +27.0 ± 10.7 |
| Family 2, memory noise only | −4217.5 | +59.9 ± 12.2 |
| Family 1, first-presented option only | −4222.0 | +64.3 ± 13.4 |
| Family 1, second-presented option only | −4227.5 | +69.9 ± 13.1 |
| Family 1 null | −4271.8 | +114.1 ± 15.8 |
| Family 2 null | −4273.2 | +115.6 ± 14.4 |

Two things matter here. (a) Dropping the **memory** term costs nothing (2.0 ± 3.9) while
dropping the **perceptual** term costs 59.9 ± 12.2 — the effect is on shared perceptual
noise. (b) Models that contain an explicit **presentation-position** parameter fit
*worse* (+64, +70) than one that has none. That is the quantitative answer to reviewers
who suspect the order effect is fitted rather than emergent.

Weber (log-space, scalar-invariant) baselines were also refit. In the Weber model the
cTBS effect on perceptual noise is **−0.001, i.e. nothing** — it can only place the
effect on memory. A magnitude-proportional perceptual perturbation does not fit.

## 2. The shape of the noise function

- Log-log slope of the perceptual noise function is **0.45**, not the 1.0 Weber's law
  predicts: ν ∝ √n. Poisson-like, as a numerosity-tuned population code would give.
- Memory noise is essentially **flat** (log-log slope 0.05, ~0.7 CHF).
- cTBS adds a roughly **constant absolute** amount, +0.15 to +0.26 CHF, with the 95%
  credible interval excluding zero at **88 of 120 grid points**, spanning 7 to ~80 CHF.
- In *proportional* terms that constant injection is **15.2% at 7 CHF and 4.1% at 112**,
  because baseline noise grows with magnitude.

**This is the resolution of an apparent contradiction in the paper.** The psychophysical
(probit) analysis found reduced consistency for low-stake trials specifically; the PMC
noise curve looks flat in CHF. Both are right: the probit slope lives on log(ratio), and
a constant absolute injection *is* localized on that scale. The disagreement is a units
artifact. Any figure making the "specific to small magnitudes" claim should show the
**relative** (%) or log-log version, not the absolute one.

Resolution limit: the noise spline supports **one interior knot**. df = 6, df = 9 and
degree-2/df-5 all fail to converge (r̂ ≥ 1.24, ESS ≤ 12), including in the 2024 fits. So
group-level magnitude-localization is not estimable; the localization evidence is the
per-subject brain–behaviour correlation (r = −0.43 with Δ nPRF amplitude, m2).

## 3. The mechanism, and why the effect is order-specific

cTBS raises noise → a Bayesian observer with noisier evidence shrinks percepts harder
toward its prior → options lose perceived value. The channel decomposition is decisive:

| channel | Δ P(risky), risky second | risky first |
|---|---|---|
| bias (prior attraction) only | **+0.074** | +0.014 |
| noise (added randomness) only | −0.003 | −0.003 |
| full model | +0.072 | +0.011 |

So the effect travels through **prior attraction, not psychometric flattening**. The
preprint's Fig-5 sentence "noise contributing the bulk of the effect" is the one line
that must change. The paper's headline claim survives — noise is still the *cause*; it
just acts via bias rather than via randomness.

The order-specificity then follows from three facts, none of which is a fitted order
parameter:

1. ν₁ is built from both noise components, ν₂ from the perceptual one alone, so the same
   cTBS perturbation raises the first option's noise **1.43×** more (+0.224 vs +0.157 CHF
   over the safe range).
2. Therefore the **safe** option loses 0.459 CHF of perceived value when presented first
   vs 0.343 when second, while the **risky** option is indifferent to position
   (−0.313 vs −0.320).
3. Choices track the gap between the two, which is **4.5× larger** when the risky option
   is second (−0.139 vs −0.031 CHF) — giving ΔP(risky) of +0.071 vs +0.009.

Observed data agree: mean ΔP(chose risky) is **+0.053 risky-second vs +0.006
risky-first**, peaking at +0.096 ± 0.041 at the smallest safe payoff.

## 4. A caveat that must be stated, not hidden

The fitted priors sit **below the entire payoff range**: safe_prior_mu = 3.57 CHF
(CrI −3.70 to 5.55) against payoffs of 7–112, and prior SD ~1.2. That compresses an
objective 28 CHF into a perceived 9 CHF, which is not credible as a subjective value.
The prior is doing the work of a compressive value function — it is how this
architecture produces risk aversion at all.

This is **not** an artifact of our priors: `--constrain` centres `*_prior_mu` on the
empirical payoff mean with σ = 10 (bauer's own default is σ = 25, i.e. weaker), and the
posteriors land 1.1–2.7 SD *below* that centre. The likelihood drives them down against
a resisting prior. A comparison fit with the prior pinned to the objective payoff
distribution is running; note that under an objective prior ~60% of trials sit *below*
the prior mean, so cTBS would push most percepts *up* — the opposite direction — which
may mean that model simply cannot produce the observed effect.

Do not quote the percept-compression numbers as perceived values.

## 5. A bug that was caught late, and the gate that now prevents it

`decision_space.<label>.tsv` stored `norm.cdf((EV2−EV1)/s)` = P(choose the **second**
option) under the name `p_vertex`, plotted as "P(chose risky)". On risky-first trials the
second option is the safe one, so that column and the `effect` derived from it were
sign-flipped across half the design. Fixed at source and re-extracted.
`validate_source_data.py` now asserts the invariants (including "P(chose risky) must rise
with the payoff ratio, because it does in the observed data") and gates every figure.

## 6. What figures currently exist

In `notes/figures/paper/`: Figure 2 panels (imaging, unchanged), `ppc_fig3a.flexible2nf`
(posterior predictive), `noise_winner.flexible2nf_perception` (noise functions of the
best model, 3 panels), `fig5.flexible2nf` (decision space, 4 columns × 2 order rows),
`why_risky_second.flexible2nf` (the order mechanism, 4 panels),
`percept_distortion.flexible2nf` (perceived value by option × order × stimulation).

Supporting material in `notes/figures/{noise,mechanism,ppc,decision_space,percepts}/`,
including per-model PPCs for all 14 fits and a `noise_variants` panel comparing the
fitted noise contrast across all eight flexible variants.
