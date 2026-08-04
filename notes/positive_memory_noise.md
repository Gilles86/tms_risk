# Should memory noise be constrained positive?

**No. The data reject it by 55.7 nats (5.3 x dSE), and the paper's result is unchanged
either way.**

## The question

bauer composes the first-presented option's noise as

    nu_1 = softplus(eta_memory + eta_perceptual),    nu_2 = softplus(eta_perceptual)

The softplus wraps the *sum*, so nothing stops `nu_1 < nu_2` -- i.e. a *negative* memory
contribution, the first-presented option being remembered more precisely than the second
one is seen. The Methods describe memory noise as something added on top of perception,
which reads as a non-negative quantity, so the natural fix is

    nu_1 = nu_2 + softplus(eta_memory)

implemented as `--memory_composition additive` in `fit_pmc_noisefix.py`
(bauer branch `feat/additive-memory-noise`). Both were fitted on the GPU nodes,
same data, same 5-df spline basis, same regressors.

## Result

| Model | composition | ELPD (LOO) | diff | dSE | p_loo | r&#770; | ESS | div |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cTBS on perceptual noise only | default | **-4157.7** | 0.0 | - | 247.6 | 1.000 | 745 | 86 |
| cTBS on perceptual + memory | default | -4159.7 | -2.0 | 3.9 | 266.3 | 1.010 | 353 | 28 |
| cTBS on perceptual + memory | **additive** | -4211.3 | -53.6 | 10.7 | 183.6 | 1.000 | 1095 | 26 |
| cTBS on perceptual noise only | **additive** | -4213.3 | -55.7 | 10.4 | 181.6 | 1.000 | 949 | 16 |

Source: `/data/table_additive.md` on `sciencecloud_gpu`, produced by
`loo_table.py` over `cogmodels.overnight` (default) and `cogmodels.additive`.

Constraining the memory contribution to be non-negative costs **55.7 nats at a dSE of
10.4** on the best model. That is not "comparable" under any reading -- it is a larger
penalty than confining the cTBS effect to one presentation position (52-70 nats), and
half the penalty of dropping the cTBS effect altogether (116 nats).

The effective number of parameters falls from 247.6 to 181.6. The constraint is not a
cosmetic reparameterisation; it removes a region of parameter space the data are using.

## Why this settles the "is the negative memory contribution credible?" worry

It is the strongest evidence available that it is real. Under the default composition
the memory contribution at 7 CHF is **-0.180 CHF, CrI [-0.384, -0.043]**, negative over
the lowest quarter of the payoff range, crossing zero at 11.5 CHF. Under the constraint
it is pinned to **+0.031 CHF, CrI [+0.004, +0.106]** -- flat against the boundary, which
is what a posterior does when it wants to go somewhere it is not allowed to go. The 55.7
nats are the price of that wall.

So the reading stands: **at small payoffs the first-presented option is encoded *more*
precisely than the second-presented one.** Small numbers are easier to encode when there
is not yet a second number competing for the same representation; the memory decay that
should hurt the first option is outweighed by that encoding advantage, and only above
~11.5 CHF does the usual memory cost dominate.

## The paper's headline result does not depend on this at all

cTBS effect on the perceptual noise function, IPS - vertex, in CHF:

| payoff | default | additive (constrained) |
|---:|---|---|
| 7 CHF | +0.191 [+0.041, +0.419] | +0.318 [+0.080, +0.625] |
| 28 CHF | +0.195 [+0.038, +0.376] | +0.263 [+0.066, +0.468] |
| 112 CHF | +0.221 [-0.195, +0.629] | +0.307 [-0.160, +0.755] |

Same sign, same roughly-constant-in-absolute-terms shape, same credibility pattern
(excludes zero at small and mid payoffs, includes it at the top of the range). If
anything the constrained fit gives a *larger* effect. Nothing in the magnitude-specificity
argument turns on the composition.

## Decision

Report the **default** composition, which is also what every existing trace and figure
uses. Mention the constrained variant as a robustness check: the sign of the memory
contribution is a finding, not an artefact of an unconstrained parameterisation, and
forbidding it costs 55.7 nats without changing the cTBS result.

Do **not** describe memory noise as necessarily additive in the Methods -- write the
composition as bauer implements it, `nu_1 = softplus(eta_mem + eta_perc)`, and say
explicitly that this leaves the sign of `nu_1 - nu_2` free.

## Does family 1 show it too? Yes — and that is the strongest version of the check

Added 2026-08-03. The worry about `nu_1 < nu_2` is that it might be an artefact of family
2's composition, `nu_1 = softplus(eta_mem + eta_perc)`. Family 1 settles it: it fits the
two positions as **independent spline functions**, with no memory/perceptual
decomposition at all, so it has no composition to produce the result.

`extract_pmc_parameters.py` previously emitted `memory_contribution` only for family 2
(the block was gated `if family == 2`). A family-1 branch was added, computing
`nu_1 - nu_2` on the same draws. Run on `sciencecloud_gpu4` against
`cogmodels.overnight/model-flexible1_noisefix.head_trace.netcdf`.

Posterior probability that **nu_1 < nu_2**:

| payoff | fam 1 vertex | fam 1 IPS | fam 2 vertex | fam 2 IPS |
|---|---|---|---|---|
| 7 | **0.997** | **0.983** | **0.998** | **0.952** |
| 10 | **0.982** | 0.852 | **0.990** | 0.732 |
| 14 | 0.270 | 0.003 | 0.123 | 0.002 |
| 20 | 0.074 | 0.000 | 0.003 | 0.000 |
| 28 | 0.075 | 0.000 | 0.005 | 0.000 |
| 56 | 0.120 | 0.405 | 0.067 | 0.318 |

Family 1, vertex, `nu_1 - nu_2` in CHF: **−0.358 [−0.685, −0.091]** at 7, −0.222 at 8.8,
−0.106 at 10.5, crossing zero at **~12.6 CHF**, then +0.08 from 17 upward. Family 2 gives
−0.275 [−0.568, −0.075] at 7 and crosses at **~12.3 CHF**. The two families agree on the
sign, the magnitude and the crossing point to within a fraction of a CHF, having
parameterised the problem completely differently.

**This is only visible on the draws.** In family 1 the marginal 95% CrIs of nu_1 and nu_2
overlap at *every* payoff (e.g. at 7 CHF, nu_1 = 1.132 [0.907, 1.400] vs
nu_2 = 1.490 [1.185, 1.862]). The two curves share subject-level and spline structure, so
their draws are strongly correlated and the difference is far better determined than
either marginal — the case ERROR_BARS.md rule 2 exists for. Anyone reading the sign off
the marginal intervals will conclude, wrongly, that there is nothing here.

Three caveats worth stating with it:

1. **It is a low-payoff phenomenon only.** By 14–28 CHF the sign is firmly reversed —
   the ordinary memory cost, with P(nu_1 > nu_2) ≈ 0.93 (family 1) and > 0.99
   (family 2). Above ~56 CHF the CrIs are too wide to say anything either way.
2. **cTBS strengthens the reversal.** Under IPS the crossing happens earlier and the
   high-payoff nu_1 > nu_2 is more certain (P(nu_1 < nu_2) = 0.000 at 20–28), which
   follows from cTBS raising nu_1 more than nu_2 (nu_1 carries both components).
3. **It is a property of the refits, not of the published traces.** Published
   `flexible1` (bauer `ecc6454`) has nu_1 = 1.32 vs nu_2 = 0.42 at 7 CHF — the *opposite*
   ordering. That is the same `b66c806` choice-rule/prior divergence documented in
   `notes/pmc_refit_results.md`, so this finding stands or falls with reporting the HEAD
   side.

## Did cTBS shift the PRIOR instead of the noise? (2026-08-03)

The obvious alternative to the paper's account: since the model produces its
risk-attitude shift through prior attraction, maybe cTBS moved the observer's **prior**
rather than adding noise. `fit_pmc_noisefix.py` defines three variants that put the cTBS
regressor on the prior — `_prior` (`risky_prior_mu`, `safe_prior_mu`), `_priorsd` (the
prior SDs) and `_perception_prior` (perceptual noise *and* the prior means). Two were
fitted on 2026-08-02 but never entered Table 1.

LOO over the full family-2 set including them, all on the same 8335 trials
(`loo_table.py` on `sciencecloud_gpu3`, traces in `cogmodels.overnight`, bauer
`e05f73a`) → `notes/data/table1_with_prior_variants.{tsv,md}`:

| model | ELPD | Δ vs best | dSE | r̂ | ESS | div |
|---|---|---|---|---|---|---|
| **perceptual noise + prior means** | **−4154.5** | 0 | — | 1.010 | 420 | 70 |
| perceptual noise only | −4157.7 | 3.2 | 2.6 | 1.000 | 745 | 86 |
| perceptual + memory noise ⚠ | −4159.7 | 5.2 | 4.7 | 1.010 | 353 | 28 |
| **prior means only (no noise effect)** | **−4179.2** | 24.7 | 7.4 | 1.000 | 779 | **0** |
| memory noise only ⚠ | −4217.5 | 63.0 | 12.4 | 1.010 | 386 | 739 |
| null ⚠ | −4273.2 | 118.7 | 14.6 | 1.020 | 109 | 664 |

**Three readings, and the first two support the paper.**

1. **A pure prior shift is a much worse account than a pure noise effect.** Prior-only
   sits **21.5 nats below** perception-only. It is nonetheless far better than the null
   (94 nats), so a prior shift does capture a lot — it is a serious alternative, not a
   straw man, and it deserves to be in Table 1 rather than omitted.
2. **Once the noise effect is in the model, adding a prior shift buys nothing credible**:
   3.2 nats at dSE 2.6, i.e. ~1.2 SE. The combined model is nominally best but is not
   distinguishable from perception-only.
3. **Caveat on the headline.** `perception_prior` being *nominally* the best model means
   "cTBS on perceptual noise only" is no longer the top row of Table 1. The honest
   statement is that the top three models are within ~5 nats of each other and all
   contain a perceptual-noise term, while every model lacking one is ≥ 24 nats worse.

Note also that prior-only is the **best-behaved fit in the family** — 0 divergences,
ESS 779, r̂ 1.000 — so its poorer ELPD is not a sampling artefact.

**Gap:** `_priorsd` (cTBS on the prior *width* rather than its mean) was never fitted.
A prior that widens under cTBS is not the same hypothesis as one that moves, and it is
arguably the closer competitor to a noise account. Worth running before Table 1 is final.

### How big a prior shift would it take? The parameters say the model is degenerate

`extract_prior_shift.py` → `notes/data/prior_shift.priorshift.tsv`,
`prior_percept_shift.priorshift.tsv`; figure `notes/figures/prior_shift.pdf`
(`plot_prior_shift.py`). Three things, and together they are a stronger argument against
the prior account than the 21.5-nat ELPD gap:

**1. The fitted priors are not plausible and barely identified.** In CHF, against
payoffs of 7–112:

| prior | vertex | IPS | 95% CrI (IPS) |
|---|---|---|---|
| safe | 83.9 | 71.0 | [1.4, 306] |
| **risky** | **13 995** | **13 393** | **[223, 69 082]** |

The risky prior mean runs off to ~13 000 CHF with an interval spanning three orders of
magnitude. It samples cleanly (0 divergences, r̂ 1.000, ESS 779) because it barely
enters the likelihood: with prior SD ≈ 1.6 log units the shrinkage weight w is large,
the percepts stay near-veridical, and the prior mean is then free to drift.

**2. Neither shift is credible.** IPS − vertex is −0.229 log units for the risky prior
(P(shift < 0) = 0.71) and −0.280 for the safe prior (P = 0.83). Both CrIs straddle zero.

**3. The decisive point: the two accounts predict qualitatively different percept
shifts.** Both must ultimately move the percept. A prior shift moves it by
(1 − w)·Δμ_prior, and (1 − w) grows steeply with payoff as ν grows — so the prior
account requires a percept shift that **grows with magnitude**: −0.50 CHF at 7 CHF,
−4.9 at 28, **−23.6 at 112** (risky, first-presented; safe is −0.71 / −6.3 / −29.3).
The noise account's fitted percept shifts are a few tenths of a CHF and roughly flat
(−0.14 to −0.6 across safe payoffs 7–28, `pmc_percepts_by_order.flexible2nf.tsv`).

So the prior account does not just fit worse — to produce the observed effect at all it
needs percept distortions one to two orders of magnitude larger than the noise account,
concentrated at exactly the large payoffs where the behavioural effect is weakest.

## Why the Weber model does not find it

Added 2026-08-03. Two independent reasons, both structural. The Weber model did not look
and fail to find it — it cannot represent it.

**1. In the Weber family-2 model, nu_1 < nu_2 is forbidden by construction.**
`bauer/models/risky_choice.py`, the `shared_perceptual_noise` branch:

```python
free_parameters['perceptual_noise_sd'] = {'mu_intercept': -1., 'transform': 'softplus'}
free_parameters['memory_noise_sd']     = {'mu_intercept': -1., 'transform': 'softplus'}
...
model_inputs['n1_evidence_sd'] = perceptual_sd + memory_sd     # both already softplus'd
model_inputs['n2_evidence_sd'] = perceptual_sd
```

Both terms are softplus-transformed **before** the sum, so
`nu_1 - nu_2 = softplus(eta_mem) > 0` identically. Measured on
`model-weber2_noisefix.head_trace.netcdf` (group Intercept, averaged over subjects):
perceptual = 0.1725 [0.1601, 0.1856], memory = **0.0610 [0.0443, 0.0801]** log-units, and
**P(nu_1 - nu_2 < 0) = 0.0000 — by construction, not by evidence.**

This is exactly the `additive` composition tested above, which costs **55.7 nats** when
imposed on the Flexible model. The Weber model has that constraint hard-wired. Contrast
the *Flexible* family 2, where the softplus wraps the **sum**,
`nu_1 = softplus(eta_mem + eta_perc)`, leaving the sign free — which is why only the
Flexible model can report on it at all.

**2. Even the Weber family-1 model has no magnitude axis to put a crossover on.**
The `independent` branch fits `n1_evidence_sd` and `n2_evidence_sd` as two separate
softplus scalars, so `nu_1 < nu_2` *is* representable there. But they are **constants in
log space** — one number per subject and condition, no dependence on magnitude. So
`nu_1 - nu_2` is a single number that applies at every payoff and **cannot cross zero at
~12.6 CHF**. The model must commit to one sign over the whole range, and
**76% of option presentations sit above the crossover** (only 23.9% fall below 12.6 CHF),
so the estimate is pulled to the positive side — the ordinary memory cost — and the
low-payoff reversal is averaged away. (Structural, from the code: only `weber2_*` traces
were on `sciencecloud_gpu4`, so the family-1 Weber fit was not checked empirically.)

**Same root cause as the other Weber failure.** `notes/figure_plan_briefing.md` records
that in the Weber model the cTBS effect on perceptual noise is −0.001, i.e. nothing, and
that it can only place the effect on memory. That is this same limitation seen from
another angle: with noise constant in log space there is no magnitude-varying noise
function, so any magnitude-localised perturbation has nowhere to go.

The model comparison agrees that this rigidity costs it: the best Weber variant is
−4192.4 against the best Flexible −4157.7, i.e. **34.8 nats worse (dSE 17.3)**
(`notes/data/table1_all16.md`).

## Caveats

- `additive` changes the prior geometry as well as imposing the constraint, so the 55.7
  nats are attributable to "the constraint and the prior it implies", not to the
  constraint in isolation. The magnitude and the boundary-hugging posterior both point at
  the constraint doing the work.
- PSIS-LOO raised Pareto-k > 0.7 warnings on 5 of the 7 traces, as it does throughout
  this trial-level hierarchical family. The differences here are far larger than the
  wobble that introduces.
- The default fits have more divergences (86 vs 16 on the winner). The constrained fits
  genuinely sample better -- they are simply fitting a worse model. Convergence quality
  is not evidence for a model; it lost on the quantity that measures fit.

Reproduce with:

    python -m tms_risk.behavior.scripts.fit_pmc_noisefix flexible2 --suffix _perception \
        --memory_composition additive --backend numpyro
    python -m tms_risk.behavior.scripts.loo_table --trace_dir <default> <additive> \
        --pattern 'flexible2_noisefix*' --out_stem table_additive
