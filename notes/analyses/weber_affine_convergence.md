# `log-weber+affine-n1n2`: why it would not converge, and the fix

**2026-09-03.** The published-candidate fit (`.pathfinder`) failed its
convergence gate: max r̂ 1.29, min ESS 20, **0 divergences**. Resolved. The
trace to use is **`model-log-weber+affine-n1n2.pathfinder.spm0.4_trace.netcdf`**
(8/8 chains, r̂ 1.002, min ESS 3807).

## Diagnosis: a prior-location ridge, not a sampler problem

Leave-one-chain-out isolated it immediately — a single rogue chain, every time:

| Init | Rogue chains | Worst safe-prior μ | max r̂ |
|---|---|---|---|
| pathfinder | 1/8 | 33.5 CHF | 1.29 |
| mapjitter | **0/8** | — | 1.002 |
| priorjitter | 1/8 | 15.6 CHF | 1.19 |
| default (`jitter+adapt_diag`) | 2–3/8 | 31.1 CHF | 1.41 |

~4–5 of 32 chains, landing at *different points* (11 → 15 → 31 → 33 CHF): a flat
ridge, not a discrete second mode. **`mapjitter` passing 8/8 was luck.** The
ridge trades prior location and width against second-option noise through
`w = σ²/(σ² + ν²)` — push both priors up and out, pay for it with more ν₂, and
the likelihood barely notices.

It is reachable because `PRIOR_SPEC` v1-2026-08-28 puts `sigma_intercept = 1.0`
on `*_prior_mu`, which **in log space is a factor of e**. A safe-option prior
centred at 31 CHF, when no safe payoff in the experiment exceeds 28, sits inside
one SD. (Note this is the prior on the group MEAN. The group SDs are HalfNormal
throughout -- bauer's default since 0.3.0, `core.py` `group_sd_dist` -- so the
neck is not a heavy-tail problem; the legacy argument name
`cauchy_sigma_intercept` is misleading on that point.)

## Fix: close the ridge, don't re-roll the dice

New flag `--sigma_prior_mu` on `fit_anchor.py`, stamped into
`tms_risk_prior_spec` (`v1-2026-08-28+spm0.4`) **and** the filename, so no trace
can silently mix prior specs. Dose–response over 40 chains:

| σ(prior_μ) | Init | Rogue | Worst excursion | max r̂ |
|---|---|---|---|---|
| 1.0 | default | 2–3/8 | 31.1 CHF | 1.41 |
| 0.6 | default | 2/8 | 30.2 CHF | 1.31 |
| 0.4 | default | 1/8 (mild) | 13.1 CHF | 1.12 |
| **0.4** | **pathfinder** | **0/8** | — | **1.002** |

Monotone, which confirms the prior width is the cause. σ = 0.6 buys almost
nothing; 0.4 is the working value.

## The tightening does not steer the fit

Across the whole range the main mode is invariant, and so is prediction:

| σ(prior_μ) | Safe prior μ | Risky prior μ | ν(2nd) @ 7 CHF | ELPD |
|---|---|---|---|---|
| 1.0 (mapjitter) | 10.9 CHF | 19.9 CHF | 0.128 | −4169.4 |
| 0.6 | 11.2 CHF | 20.2 CHF | 0.128 | — |
| 0.4 | 11.5 CHF | 20.6 CHF | 0.128 | −4169.1 |

ΔELPD between σ = 1.0 and σ = 0.4 is **+0.3 ± 0.2** — nothing. So the prior width
changes *whether the ridge is sampled*, not *where the answer sits*.

## Verdict on the model — unchanged, and now sayable

Paired pointwise LOO, `mapjitter` trace at the standard prior spec (positive =
weber+affine better); the spm0.4 numbers differ from these by < 0.5 throughout:

| vs | ΔELPD | dSE | ratio |
|---|---|---|---|
| `log-power-n1n2` (reported model) | **−14.9** | 5.2 | −2.85 |
| `log-power-n1n2psd` (best overall) | −45.2 | 9.1 | −4.98 |
| `log-weber-n1n2` (pure Weber) | **+20.7** | 6.5 | **+3.19** |

`p_loo` 197.6, **0% of points with k > 0.7**. It is a well-behaved intermediate
rung: clearly better than pure Weber, clearly worse than the power law. That is
a much stronger supplementary sentence than "it did not converge".

## Which trace to quote in the ELPD table — and why not the spm0.4 one

ELPD across models with different prior specs is not apples-to-apples, so a
matched `log-power-n1n2.pathfinder.spm0.4` was fitted (job 5485118). **It does
not converge**: max r̂ 1.151, min ESS 34, and unlike weber+affine there is no
single rogue chain (dropping any one leaves r̂ ≥ 1.09) — chains 3, 6 and 7 are
all mildly displaced. σ = 0.4 sits far enough from `log-power-n1n2`'s preferred
safe-prior μ (2.37 ≈ 10.7 CHF, against a prior centre of 2.64) that prior and
likelihood pull against each other and the geometry gets harder.

**So σ = 0.4 is not a universally safe setting. It fixes `weber+affine` and
breaks `log-power-n1n2`.** It must stay an opt-in per-model flag; do not promote
it into `PRIOR_SPEC`.

Resolution: **quote `log-weber+affine-n1n2.mapjitter` in the ELPD table.** It
carries the standard prior spec, so it is directly comparable to every other
model in the ladder, and the spm0.4 fit is the corroboration that its mode is
the right one rather than a lucky artefact — the two agree to **+0.3 ± 0.2
ELPD** and ~5% on the prior means. The `mapjitter` run was a fluke *sampling
success*, not a fluke *answer*.

For the record, the reported model is unaffected: `log-power-n1n2` at the
standard prior spec converges cleanly (max r̂ 1.002, min ESS 1727, 4 chains all
agreeing), so **Figure 5 needs no revision**.

## Reproduce

```bash
# on sciencecluster
sbatch tms_risk/behavior/slurm_jobs/refit_weber_affine.sh        # 4-init diagnosis
sbatch tms_risk/behavior/slurm_jobs/refit_weber_affine_tight.sh  # the fix + sensitivity
```
