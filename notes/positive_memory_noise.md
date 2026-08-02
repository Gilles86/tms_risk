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
