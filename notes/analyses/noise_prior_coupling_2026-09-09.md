# Does the noise increase predict the safe-prior shift, across participants?

**Proposed mechanism (Gilles, 2026-09-09):** report the safe prior's μ shift
*after* the n1n2 result, and speculate that it follows from the noise increase —
a prior has to be learned from the same representations it later corrects, so if
cTBS makes small payoffs noisier, the prior over small payoffs (in this design,
the prior over SAFE options: mean 15.8 CHF against 36.2 for risky) is estimated
from worse evidence and ends up less well calibrated.

That is a genuinely falsifiable version of a story that would otherwise be
decoration, because it predicts a **coupling across participants**: whoever's
noise rose most should show the biggest prior shift.

## Verdict: the coupling cannot be tested on these fits. Not "null" — untestable.

`behavior/scripts/anchor_noise_prior_coupling.py`, on
`log-power-percpmu.mapjitter.klw` (n = 35):

| noise | prior | r | 95% CrI | P(r>0) | r_within |
|---|---|---:|---|---:|---:|
| perc @7 | safe_prior_mu | −0.010 | [−0.34, +0.33] | 0.478 | +0.001 |
| perc @112 | safe_prior_mu | +0.019 | [−0.31, +0.35] | 0.546 | −0.031 |
| perc @7 | risky_prior_mu | +0.041 | [−0.27, +0.34] | 0.603 | +0.000 |
| perc @112 | risky_prior_mu | −0.027 | [−0.34, +0.28] | 0.434 | +0.227 |

`r_within` ≈ 0 confirms the test itself is clean: the two contrasts do **not**
trade off within a participant, so a real coupling would not have been masked by
the fit's own geometry. But that is not why r is zero.

**The reason is that the per-participant safe-prior shift has no usable
individual variation.** From `anchor_subject_reliability.py` on the same trace:

| parameter | sd_within | sd_between (observed) | reliability_draw | ceiling |
|---|---:|---:|---:|---:|
| perc_power_sd112 | 0.288 | 0.296 | 0.534 | **0.731** |
| perc_power_sd7 | 0.280 | 0.219 | 0.405 | **0.636** |
| risky_prior_mu | 0.218 | 0.166 | 0.385 | **0.621** |
| **safe_prior_mu** | **0.163** | **0.031** | **0.043** | **0.208** |

The between-participant spread in the safe-prior shift is **five times smaller
than the within-participant posterior width**. The model says essentially the
same thing about every participant. The attenuation ceiling on any correlation
involving it is 0.208 × 0.636 ≈ **0.13** — so even a perfect underlying coupling
could not have produced an observable r above about 0.13, and the credible
interval above is four times wider than that.

## What this means for the paper

* **The group-level shift is what is identified, and only that.** In `percpmu`
  the safe prior's mean falls 14.4% under IPS with P(< 0) = 0.948 and a CrI that
  crosses zero. It is a suggestive group effect, not a credible one, and it has
  no individual-differences backing.
* **Do not write the coupling as a supporting result, and do not write it as a
  tested-and-null result either.** The honest statement is that the
  per-participant prior shift is too weakly identified to test the prediction —
  which is a statement about the design (120 choices per session pin the prior
  mean only weakly), not about the mechanism.
* **The neural data do not corroborate it either.** `decoded_bias_by_payoff.py`,
  within payoff level, on the CV decoding tree (35 participants, 8335 trials):
  safe +1.48 CHF, risky +1.19, difference +0.29 (t = 0.39). A weak *upward*
  shift, the same in both roles — where a downward safe-prior shift predicts
  the safe percept to move *down*.
* So this stays **supplementary and explicitly speculative**, phrased as a
  possibility the present data cannot adjudicate, as Gilles proposed.

## Still to do

`log-power-n1n2pmu` and `log-power-n1n2pmusd` are fitting (job 5711934) at the
ladder's τ_noise 0.15 and sampler, for two reasons:

1. **Consistency.** A prior shift reported after n1n2 must be estimated on top
   of n1n2. `percpmu` sits in the shared perc/mem family, so quoting its prior
   shift beside an n1n2 noise effect takes the two halves of the story from two
   different models.
2. **ELPD.** On the ladder's prior and sampler it also answers whether the prior
   shift buys any predictive accuracy over n1n2 alone.

Rerun both scripts on those traces when they land. If `n1n2pmu` gives the
safe-prior shift more per-participant reliability than `percpmu` does, the
coupling test becomes possible; on present evidence, expect it not to.

---

# Convergence audit of the prior-shift family (2026-09-09, after the PPC)

The PPC comparison first quoted `n1n2pmu` at ΔP +0.029 / ppp 0.06 and ELPD
−4144. **Those numbers come from traces that fail the gate and are retracted.**

| trace | r̂ | ESS | div | passes |
|---|---:|---:|---:|:--:|
| log-power-percpmu.mapjitter.klw | 1.000 | 11016 | 4 | **yes** |
| log-power-percpmusd.mapjitter.klw | 1.000 | 7818 | 7 | **yes** |
| log-power-percpsd.pathfinder.klw | 1.000 | 3224 | 1 | **yes** |
| log-power-percmempsd.pathfinder.klw | 1.000 | 3145 | 27 | yes |
| log-power-spmusd.pathfinder.klw | 1.000 | 3934 | 2 | **yes** |
| log-power-spsd.pathfinder.klw | 1.000 | 4176 | 4 | **yes** |
| log-power-pmusd.klw | 1.000 | 1779 | 26 | yes |
| log-power-psd.klw | 1.000 | 972 | **147** | yes, but |
| log-power-n1n2pmu.mapjitter.klw | 1.060 | 90 | 3 | **no** |
| log-power-n1n2pmu.klw | 1.180 | 15 | 4 | **no** |
| log-power-n1n2pmu.pathfinder.klw | 1.210 | 26 | 1 | **no** |
| log-power-n1n2psd.klw | 1.200 | 14 | 10 | **no** |
| log-power-n1n2psd.pathfinder.klw | 1.130 | 38 | 2 | **no** |
| log-power-pmu.klw | 1.520 | 7 | 6 | **no** |

**The split is perfectly clean along the family, not the placement.** Every
model in the *shared* perc/mem family samples; every *position*-indexed
(`n1n2*`) model with prior parameters fails, at every initialisation tried
(default, mapjitter, pathfinder). The worst parameter is the same one every
time — `log_n1_power_sd7_sd[Intercept]`, the between-participant SD of the
first-presented option's noise at the low anchor.

That is the documented geometry of the independent family: σ_n1 = σ_perc +
σ_mem makes σ_n1 > σ_n2 true by construction in the shared family, and nothing
enforces it in the independent one. Plain `n1n2` already needs the noise-anchor
prior to sample (r̂ 1.12 / ESS 42 at defaults); `n1n2pmu` puts two more
parameters into the same badly conditioned posterior.

## What this leaves standing

* **`percpmu` is the only converged model that both carries a prior-mean shift
  and passes the gate.** Its PPC is therefore the one that can be quoted:
  ΔP on risky-second +0.024 predicted against +0.053 observed, ppp 0.040 — a
  substantial improvement on the noise-only models (n1n2 +0.013 ppp 0.005; perc
  +0.006 ppp 0.000) but still only just inside the interval, and its
  `order_contrast` sits exactly on the line at ppp 0.050.
* **No converged position-indexed prior-shift model exists yet.** Job 5711934
  fits `n1n2pmu` and `n1n2pmusd` at the ladder's τ_noise 0.15 — the same prior
  that rescues plain `n1n2`. This is the only route to a like-for-like
  comparison, and it is not redundant with the traces above.
* **Do not quote any ELPD from the failing rows**, which removes `n1n2psd`
  (−4128.3) and `n1n2pmu` (−4144.0) from the ladder. `pmusd` (−4126.2) and
  `psd` converge but carry 26 and 147 divergences on 4 chains; refit before use.
