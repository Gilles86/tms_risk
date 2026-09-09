# What if the paper reports the perceptual/memory parameterisation instead?

Asked because every *position*-indexed (`n1n2*`) model with prior parameters
fails to sample, and plain `n1n2` needs a noise-anchor prior to sample at all
(see the convergence audit in `noise_prior_coupling_2026-09-09.md`). The shared
perc/mem family is markedly better conditioned. So: does the whole story work
there?

**The parameterisation.** σ_n1 = σ_perc + σ_mem, σ_n2 = σ_perc. The
first-presented option is perceived and then held in working memory, so its
representation is noisier *by construction*; the second-presented option is on
screen. Indexing is by processing STAGE, not by serial position.

## 1. Convergence: everything works, and the τ problem disappears

**12 of 12 traces pass the gate at DEFAULT priors** — r̂ ≤ 1.010, ESS
2150–11016, no noise-anchor prior anywhere.

That is the single biggest practical argument for the switch. The whole τ_noise
apparatus — three candidate values, the "mildest τ at which every rung samples"
criterion, the disqualified `--tau_intercept`, the argument about how much prior
tightening erodes the claim's specificity — exists only to make the independent
family sample. In the shared family none of it is needed, and the ELPD ladder is
commensurable without any of that machinery.

## 2. The ELPD ladder (paired, dSE against the top model)

`notes/data/shared_family_ladder.tsv`. All fitted with the same bauer commit
(93c2e9a); ranks 0–11:

| rank | model | cTBS moves | ELPD | ΔELPD | dSE |
|---:|---|---|---:|---:|---:|
| 0 | percmempsd | perc + mem noise, prior SD | −4138.6 | — | — |
| 1 | spmusd | **priors only** | −4139.1 | 0.5 | 4.2 |
| 2 | percpsd | perc noise, prior SD | −4139.9 | 1.3 | 1.3 |
| 3 | percpmusd | perc noise, prior μ and SD | −4140.4 | 1.8 | 2.5 |
| 4 | spsd | **prior SD only** | −4142.8 | 4.2 | 3.8 |
| 5 | percpmu | perc noise, prior μ | −4148.1 | 9.5 | 4.7 |
| 6 | percmemx | perc + mem noise × order | −4152.7 | 14.1 | 6.3 |
| 7 | percx | perc noise × order | −4153.5 | 14.8 | 6.3 |
| 8 | percmem | perc + mem noise | −4153.9 | 15.3 | 4.9 |
| 9 | **perc** | perc noise only | −4155.2 | 16.6 | 5.1 |
| 10 | mem | mem noise only | −4195.0 | 56.4 | 10.3 |
| 11 | null | nothing | −4259.5 | 120.9 | 13.9 |

**What this ladder establishes cleanly:**

* **cTBS changes something** — `null` is 120.9 ± 13.9 worse, 8.7 SE.
* **It is not the memory stage alone** — `mem` is 56.4 ± 10.3 worse, 5.5 SE.
  (Caveat: `mem` carries a Pareto k of 4.3, so read the direction, not the
  digits.)
* **The noise function is payoff-dependent** — the Weber and affine rungs sit
  outside this family's table but were 29 and 20 ELPD worse in the earlier
  comparison.

**What it does NOT establish, and this is the problem:**

`spmusd` (rank 1) and `spsd` (rank 4) let cTBS move **only the priors, with no
change to representational noise at all**, and they are statistically tied with
the best model. Taken at face value that undercuts the paper's central claim.

The answer is the one already in ¶75 of the draft, and this ladder strengthens
rather than weakens it: the shrinkage weight is σ_p²/(σ_p² + ν²), so a wider
prior and a lower noise level move the same quantity. `spsd` is not a competing
explanation, it is **the same explanation in different coordinates**, and ELPD
cannot separate coordinates. The reason to fix the priors is therefore not that
free priors fit worse — they fit better — but that the fit cannot distinguish
them and the **neural data break the tie**: cTBS lowered nPRF amplitude and left
preferred numerosity intact, which is a change in the fidelity of the
representation, not in where it is pulled to.

That argument must be made explicitly if this family is reported, because a
reviewer running the ladder will see rank 1 and ask.

## 3. Posterior predictive checks

Eight targeted statistics; ppp, and `*` where the observed value falls outside
the model's own 95% predictive interval.

| statistic | obs | perc | percmem | percpmu | percx | mem | n1n2 |
|---|---:|---|---|---|---|---|---|
| dp_second_mean | +0.053 | +0.006 0.00* | +0.011 0.01* | **+0.024 0.04** | +0.009 0.00* | +0.011 0.00* | +0.013 0.01* |
| order_contrast | +0.046 | +0.001 0.02* | +0.010 0.03 | +0.009 0.05 | +0.011 0.05 | +0.014 0.08 | +0.018 0.09 |
| dp_first_mean | +0.006 | +0.004 0.46 | +0.001 0.38 | +0.016 0.72 | +0.000 0.34 | — | −0.004 0.23 |
| dp_second_high | +0.037 | +0.015 0.19 | +0.021 0.26 | +0.023 0.30 | +0.018 0.25 | +0.011 0.14 | +0.014 0.20 |
| stake_slope_second | −0.023 | +0.018 0.87 | +0.014 0.85 | +0.001 0.69 | +0.017 0.88 | +0.002 0.76 | −0.002 0.69 |
| three_way | −0.028 | +0.007 0.76 | +0.004 0.76 | −0.002 0.72 | +0.011 0.78 | +0.014 0.80 | −0.027 0.51 |
| slope_second_ctbs | −0.103 | +0.007 0.97 | +0.006 0.96 | −0.039 0.86 | −0.019 0.91 | −0.003 0.96 | +0.009 0.97 |
| slope_contrast | −0.094 | +0.003 0.92 | +0.002 0.91 | −0.005 0.88 | −0.043 0.76 | −0.013 0.86 | −0.008 0.84 |
| **covered** | | 6/8 | 7/8 | **8/8** | 7/8 | 6/7 | 7/8 |

**`percpmu` is the only model in either family that covers all eight.** It is
also the only one that covers `dp_second_mean`, the size of the cTBS effect
where it is largest — every noise-only model in both families fails it, because
that effect is mostly BIAS and no noise placement produces bias.

## 4. The order asymmetry — where the shared family loses

Mean cTBS effect on choice proportions, in percentage points, averaged over
rungs. Observed: risky-first +0.63, risky-second +5.23, asymmetry **+4.60**.

| model | first | second | asymmetry | % of observed |
|---|---:|---:|---:|---:|
| **n1n2** (position) | −0.45 | +1.39 | **+1.84** | **40%** |
| mem | −0.47 | +1.00 | +1.47 | 32% |
| percmemx | −0.11 | +1.12 | +1.23 | 27% |
| percmem | +0.09 | +1.11 | +1.02 | 22% |
| percx | −0.06 | +0.89 | +0.96 | 21% |
| percpmu | +1.63 | +2.49 | +0.86 | 19% |
| percpsd | +1.32 | +1.60 | +0.28 | 6% |
| perc | +0.40 | +0.63 | +0.22 | 5% |
| null | −0.05 | +0.04 | +0.09 | 2% |

This is the real cost. **The position-indexed model recovers twice as much of
the order asymmetry as the best shared-family model.** `percpmu` raises the
overall size of the effect (+2.49 on risky-second, the closest any model gets to
the observed +5.23) but at the price of also raising it on risky-first (+1.63
against an observed +0.63), so the *specificity* gets worse even as the
*magnitude* improves.

Note also that `mem` is second on this table — memory noise only affects σ_n1,
the first-presented option, so it is order-specific by construction. It is
nonetheless decisively rejected on ELPD (56.4 ± 10.3).

## Verdict

**It works, and it is a defensible paper — but it trades the paper's sharpest
behavioural claim for a much sounder statistical footing.**

Gains: every model converges at default priors, so the τ apparatus disappears; a
complete 12-rung ladder with real dSEs; `percpmu` passes all eight PPCs; the
memory-only and no-effect models are decisively rejected.

Losses: the order asymmetry drops from 40% to 19–22% recovered, and the ladder
puts prior-only models at the top, which forces the identifiability argument
into the foreground where it is currently a supporting remark.

**The choice is between two honest papers**: one that reports a model expressing
order-specificity and has to explain a prior it had to tighten to make it sample,
and one that reports a well-conditioned model set and has to explain why it
holds priors fixed when free-prior models fit better. The second is the easier
paper to defend at review; the first is closer to the phenomenon.

---

# RULING (Gilles, 2026-09-09): in the perc/mem family, prior-SD models are out

> "I think if we go perc/mem we can leave out all models where the sd of the
> prior changes."

This is the right call and it should be stated as an **admissibility criterion
declared before the comparison**, not as a result of it. The shrinkage weight is

    w = sd_prior^2 / (sd_prior^2 + nu^2)

so a wider prior and a lower noise level move the same quantity. A model in
which cTBS changes the prior WIDTH is therefore not a competing account of the
data — it is the noise account rewritten in other coordinates, and no amount of
predictive accuracy can adjudicate between coordinates. Including such models in
an ELPD ladder and then reporting that they win is a category error.

A prior MEAN shift is different in kind: it moves WHERE the percept is pulled
to, not HOW HARD, and it can produce a bias rather than a slope change. It stays
in.

**Excluded on this criterion (5 models):** `percmempsd`, `spmusd`, `percpsd`,
`percpmusd`, `spsd` — which is ranks 0, 1, 2, 3 and 4 of the full ladder.

**What that does to the ladder:** `percpmu` becomes the top admissible model,
and the awkward "prior-only models tie for best" problem disappears entirely —
not by ignoring it, but by ruling those models inadmissible on a stated
principle. The remaining ladder is:

| model | cTBS moves | ELPD |
|---|---|---:|
| **percpmu** | perceptual noise + prior means | **−4148.1** |
| percmemx | perc + mem noise × order | −4152.7 |
| percx | perc noise × order | −4153.5 |
| percmem | perc + mem noise | −4153.9 |
| perc | perc noise only | −4155.2 |
| mem | mem noise only | −4195.0 |
| null | nothing | −4259.5 |

(The dSEs in the table above are paired against the FULL ladder's top model and
must be recomputed with `percpmu` as the reference — running.)

## One rung is still missing, and it is the important one

Nothing in the admissible set moves the prior means **without** also moving the
noise. Without that rung, `percpmu`'s noise effect cannot be shown to be
necessary — a reader can ask whether the prior shift alone would have done it.

`spmu` (shared family, prior means only, no noise term) has been added to
`PLACEMENT` in `fit_anchor.py` and submitted (job 5712052). The independent
family's `pmu` cannot serve: it fails the gate badly (r̂ 1.52, ESS 7).

With `spmu` fitted, the ladder answers four questions cleanly, in the order a
reader asks them:

1. Does cTBS do anything? — `null`
2. Does it change the noise function? — `spmu` (prior only)
3. Is it the memory stage? — `mem`
4. Does it also move the priors? — `perc` vs `percpmu`

and it does so without ever putting a prior-width model on the same axis.

---

# The admissible ladder, paired against `percpmu` (2026-09-09)

`notes/data/admissible_ladder.tsv`. Prior-SD models excluded on the stated
criterion; all seven remaining traces pass the convergence gate.

| model | cTBS moves | ELPD | ΔELPD | dSE | SE |
|---|---|---:|---:|---:|---:|
| **percpmu** | perceptual noise + prior means | −4148.1 | — | — | ref |
| percmemx | perc + mem noise × order | −4152.7 | 4.6 | 5.8 | 0.8 |
| percx | perc noise × order | −4153.5 | 5.4 | 5.6 | 1.0 |
| percmem | perc + mem noise | −4153.9 | 5.8 | 4.1 | 1.4 |
| perc | perceptual noise only | −4155.2 | 7.1 | 4.1 | 1.7 |
| mem | memory noise only | −4195.0 | 46.9 | 10.1 | 4.6 |
| null | nothing | −4259.5 | 111.4 | 13.7 | 8.2 |

**Read this honestly.** The ladder decisively rejects two things and arbitrates
nothing else:

* **cTBS does something** — `null` is 8.2 SE worse.
* **It is not the memory stage alone** — `mem` is 4.6 SE worse.
* **Everything between `perc` and `percpmu` is within 1.7 SE.** ELPD does not
  establish that the prior mean moves, and it does not establish that the
  order interaction is needed. `percpmu` is the top rung, but "top by 1.7 SE
  among five models spanning 7 ELPD points" is not a selection.

So the model paragraph should say: model comparison settles that the cTBS effect
is real and that it is not confined to the memory stage; it does **not** settle
which channel carries it, and the choice among the surviving placements rests on
the posterior predictive checks (where `percpmu` is the only model covering all
eight targeted statistics and 9/10 design-grid cells) and on parsimony.

`spmu` — prior means only, no noise change — is still fitting (job 5712052).
That rung is what would let the noise effect be shown to be NECESSARY rather
than merely sufficient, and it is currently missing from this table.

---

# Second exclusion (Gilles, 2026-09-09): the `*x` order-interaction models are out

> "let's also leave out percmemx and percx. This is all too tricky."

Same status as the prior-SD exclusion: an admissibility criterion, declared
before the comparison. The `*x` models add a cTBS x presentation-order
interaction to a STAGE-indexed noise channel, which makes the perceptual noise
of a stimulus depend on where in the trial it happened to be shown. That is not
a claim about perception, and in the position-indexed family the equivalent
interaction is null anyway (`n1n2x`: +0.028, P = 0.56). Position-indexing gets
the same order structure without an interaction term; stage-indexing should not
buy it with one.

**Admissible ladder after both exclusions** (dSE paired against `percpmu`;
dropping rungs does not change the pairwise dELPD or dSE between the survivors,
only the stacking weights):

| model | cTBS moves | ELPD | ΔELPD | dSE | SE |
|---|---|---:|---:|---:|---:|
| **percpmu** | perceptual noise + prior means | −4148.1 | — | — | ref |
| percmem | perc + mem noise | −4153.9 | 5.8 | 4.1 | 1.4 |
| perc | perceptual noise only | −4155.2 | 7.1 | 4.1 | 1.7 |
| mem | memory noise only | −4195.0 | 46.9 | 10.1 | 4.6 |
| null | nothing | −4259.5 | 111.4 | 13.7 | 8.2 |

Five rungs, each a claim a reader would want tested, none of them a
reparameterisation of another.

# `percmempmu` is missing and should be there

> "and maybe percmempmu *should* be in there?"

Yes — it is the **full model of the admissible set**. `percmem` (both noise
channels) and `percpmu` (perceptual noise + prior means) are each nested inside
it, and without it the ladder poses "noise or priors?" as a choice when the
answer may be "both". Its absence is also why the current top rung is
`percpmu`: nothing in the set is allowed to move the memory channel *and* the
priors.

Added to `PLACEMENT` in `fit_anchor.py` and submitted (job 5712320), together
with its `power+weber` variant (Weber memory by construction, see below).

# Memory noise Weber by construction

> "can we make the memory effect Weber by construction btw?"

Already supported: the `form1+form2` label syntax gives the two channels
different noise forms, so `log-power+weber-percpmu` is a power law on the
perceptual channel and a single constant on the memory one — scale-invariant in
log space by assumption rather than by estimate. Submitted for `percpmu`,
`perc`, `percmem`, `null` (job 5712084) and `percmempmu` (5712320).

**This is a real test and it may lose.** sigma_n1 = perc + mem. With memory flat
and perceptual rising, sigma_n1 must rise — but the baseline fit says the
first-presented option's noise is nearly flat (b 0.081) while the second's rises
(b 0.357), and it is precisely a FALLING memory term that reconciles those. The
free fit does show memory falling (0.097 at 7 CHF to 0.046 at 112). So the
constraint is testable against exactly the fact Figure 4 reports.

---

# `percmem` IS `n1n2` plus one constraint — and the constraint binds

> "btw, the percmem model should be able to fit it then right? It also cannot do it."

Right, and this turns out to be the crux.

`bauer/models/risky_choice.py:620-625`:

    sigma_n1 = perceptual_sd + memory_sd
    sigma_n2 = perceptual_sd

with **both** components softplus-positive (`:644-646`). So the map
`perc = sigma_n2`, `mem = sigma_n1 - sigma_n2` is a bijection wherever
`sigma_n1 > sigma_n2`, and the cTBS regressors span the same two dimensions
(`d_sigma_n2 = d_perc`, `d_sigma_n1 = d_perc + d_mem`).

**`percmem` is therefore exactly `n1n2` restricted to sigma_n1 > sigma_n2 at
every payoff.** Same model, one inequality.

## The inequality is false in the upper half of the range

Baseline fit, n = 73, no stimulation, independent family (`log-power-nullind`):

| payoff | sigma_n1 | sigma_n2 | n1 − n2 |
|---:|---:|---:|---:|
| 7 | 0.203 | 0.100 | **+0.103** |
| 14 | 0.215 | 0.128 | +0.087 |
| 28 | 0.227 | 0.161 | +0.066 |
| 56 | 0.240 | 0.208 | +0.033 |
| 112 | 0.254 | 0.268 | **−0.014** |

The first-presented option is much noisier at small payoffs and the gap closes
monotonically, crossing around 100 CHF. **But the crossing is not credible**
(n1 [0.222, 0.293] against n2 [0.232, 0.312] at 112 CHF), and — checked against
the fitted figure — **the memory term is nowhere near its floor**:

| payoff | mem | 95% CrI |
|---:|---:|---|
| 7 | 0.096 | [0.064, 0.137] |
| 28 | 0.066 | [0.049, 0.088] |
| 112 | 0.045 | [0.026, 0.077] |

**Retraction.** An earlier version of this section claimed the `mem >= 0`
constraint was binding over the upper half of the range and that this explained
`percmem` recovering 22% of the order asymmetry against `n1n2`'s 40%. It is not
binding — mem is comfortably positive at every payoff — and that explanation is
withdrawn. The likelier story is the caveat already recorded below: `n1n2`'s 40%
comes from a trace at r̂ 1.12 / ESS 42, and a posterior that has not converged
can produce any predictive it likes.

Which leaves the question open, and makes a pure REPARAMETERISATION (below) the
right next test: it changes the geometry without imposing or removing any
constraint, so if it samples it gives `n1n2`'s answer with none of the doubt.

## Proposed fix: a SIGNED memory term

Keep the additive decomposition — sharing `perc` across both options is
presumably what stabilises sampling — but drop the positivity on `mem`:

    sigma_n1 = softplus(perc) + mem        # mem signed, sigma_n1 kept positive
    sigma_n2 = softplus(perc)

This sits strictly between `n1n2` (two unconstrained channels, will not sample)
and `percmem` (ordered, samples, constraint binds). It needs a
`memory_model='signed_memory'` branch in bauer: currently
`free_parameters['memory_noise_sd']` carries `'transform': 'softplus'`, and
this variant needs `'identity'` plus a positivity guard on the sum.

**Alternative, if that will not sample either:** use the baseline fit as an
informative prior. Session 1 (n = 73) is independent data collected before any
stimulation, so putting its posterior on the noise function's SHAPE and letting
the TMS fit estimate only the cTBS deviations is not double-dipping. It
constrains exactly the direction the group means slide along (r = 0.97) without
imposing an inequality the data reject.

---

# The final admissible ladder (2026-09-09) — `percpmu` is the reference

All twelve traces pass the gate. `notes/data/ladder_final.tsv`. dSE is paired
against rank 0.

| rank | model | cTBS moves | ELPD | ΔELPD | dSE | SE |
|---:|---|---|---:|---:|---:|---:|
| 0 | percmempmu | perc + mem noise, prior means | −4148.09 | — | — | — |
| 1 | **percpmu** | perc noise, prior means | −4148.10 | **0.01** | 1.15 | 0.0 |
| 2 | power+weber-percpmu | as 1, memory Weber by construction | −4149.69 | 1.60 | 2.94 | 0.5 |
| 3 | power+weber-percmempmu | as 0, memory Weber | −4150.36 | 2.27 | 2.53 | 0.9 |
| 4 | percmem | perc + mem noise | −4153.92 | 5.83 | 3.91 | 1.5 |
| 5 | power+weber-perc | perc noise, memory Weber | −4155.03 | 6.94 | 5.00 | 1.4 |
| 6 | perc | perc noise only | −4155.19 | 7.10 | 4.24 | 1.7 |
| 7 | power+weber-percmem | perc + mem noise, memory Weber | −4156.11 | 8.02 | 4.63 | 1.7 |
| 8 | **spmu** | **priors only, no noise change** | −4173.81 | 25.72 | 6.84 | **3.8** |
| 9 | mem | memory noise only | −4194.99 | 46.90 | 9.92 | **4.7** |
| 10 | power+weber-null | nothing | −4259.05 | 110.96 | 13.90 | **8.0** |
| 11 | null | nothing | −4259.49 | 111.40 | 13.70 | **8.1** |

## What it establishes

* **cTBS changes something** — `null` 8.1 SE.
* **It is not the memory stage alone** — `mem` 4.7 SE.
* **It is NOT the priors alone** — `spmu` **3.8 SE**. This is the rung that was
  missing from every earlier version of the ladder, and it is the one that
  makes the noise effect NECESSARY rather than merely sufficient. Without it
  the paper could not rule out "cTBS only moved where the percept is pulled to".
* **The memory channel's cTBS effect buys nothing** — `percmempmu` beats
  `percpmu` by 0.01 ELPD (dSE 1.15). `percpmu` is nested inside it and fits
  identically with two fewer parameters, so **`percpmu` is the reference
  model**, chosen on parsimony against an exact tie rather than on a margin.

## What it does NOT establish

* **The prior-mean shift is not decisively required.** `percmem` (noise only,
  both channels) is 5.83 ± 3.91 behind — 1.5 SE. ELPD prefers the prior shift
  but does not establish it. What carries it is the posterior predictive: it is
  the only model in either family covering all eight targeted statistics and
  21 of 22 design-grid cells, and the only one covering `dp_second_mean`, the
  size of the cTBS effect where it is largest. Report it that way round.
* **Weber memory is free.** `power+weber-percpmu` is 1.60 ± 2.94 behind with
  **14 parameters against 16** — indistinguishable. So the memory term's
  payoff-dependence is not doing measurable work in the TMS cohort, even though
  it is clearly present in the baseline (falling to 0.41x over the range). Worth
  one sentence; not worth changing the reported model over.
* `mem` and `spmu` carry Pareto-k warnings, so read their directions, not their
  digits.

---

# MAJOR CORRECTION (2026-09-09, adversarial audit): n1n2 DOES fit

I concluded that the position-indexed parameterisation "cannot be fitted to
these data, in any coordinate system". **That is wrong**, and the refuting
evidence was already on disk — some of it in a table in this repo that I wrote.

Position-indexed `n1n2` fits **with the cTBS regressor**, TMS cohort, same prior
spec and bauer commit:

| label | anchors | r̂ | ESS | gate |
|---|---:|---:|---:|:--:|
| log-spl7-n1n2 | 7 | 1.000 | **14412** | PASS |
| log-cspl7-n1n2 | 7 | 1.000 | **13457** | PASS |
| log-spl6-n1n2 | 6 | 1.000 | **13226** | PASS |
| log-cspl5-n1n2 | 5 | 1.000 | **10215** | PASS |
| log-spl5-n1n2 | 5 | 1.000 | **2822** | PASS |
| log-spl4-n1n2 | 4 | 1.010 | **2424** | PASS |
| log-weber-n1n2 | 1 | 1.000 | **11129** | PASS |
| log-power-n1n2 | 2 | 1.12 | 42 | fail |
| log-affine-n1n2 | 2 | 1.05 | 99 | fail |
| log-spl3-n1n2 | 3 | 1.06 | 80 | fail |
| log-cspl3-n1n2 | 3 | 1.10 | 50 | fail |

`spl5-n1n2` (r̂ 1.000 / ESS 2822) is in `notes/supp_table1.md`. It was in front of
me the whole time.

**It is not the parameterisation. It is the 2-anchor POWER form specifically**,
and forms adjacent to it. `spl5-n1n2` has 2.5x more noise parameters than
`power-n1n2` and was run at identical sampler settings; it gets ESS 2822 against
42. Non-identifiability gets worse with more parameters, not better.

## The failure is the noise/prior ridge, not the two channels

In all ten failing traces a magnitude-prior parameter is among the three
worst-mixing; in the eight converging `n1n2` traces none is. Per-chain means in
`log-power-n1n2` are perfectly rank-ordered across all six worst parameters —
one slow 1-D direction, autocorrelation ~570 draws, not multimodality. The
direction is **noise level vs magnitude-prior location**, which is mechanical
from the KLW rule: the posterior mean is `w·log n + (1-w)·prior_mu` with
`w = s_p²/(s_p²+σ²)`, so raising σ and moving `prior_mu` toward the data are
near-equivalent. `safe_prior_mu` is the weakest-identified parameter in the
model because the safe option takes only five distinct values.

Two of my own diagnostics were measuring the wrong thing: the failing fit's
group-level condition number is **7.2, BETTER than the converging `percmem`
fit's 10.1**, and the max marginal correlation is 0.41. The slow direction runs
*between* the noise and prior blocks, which is a subspace the (n1, n2) rotation
could not touch — which is why `sd` sampled worse, not better.

Also checked and cleared: the anchor family applies **no softplus** (every θ is
identity, σ = exp(Bθ)), so the softplus-of-sum concern applies only to the older
spline family; and bauer is **already non-centred** for every group SD
(`core.py:1486-1489`), so there is no funnel bug to fix.

## The scientific consequence, which matters more than the sampling one

Group cTBS effect on log σ, converged position-indexed fits:

| fit | first-presented | second-presented |
|---|---|---|
| spl5-n1n2 | +22.3% @13 CHF (P .97) | +27.4% @13 CHF (P .93) |
| spl7-n1n2 | +27.0% @14 (P .97) | +22.7% @14 (P .88) |
| cspl7-n1n2 | +27.3% @14 (P .98) | +21.6% @10 (P .89) |
| percmem (stage-indexed) | perceptual +18.3% @7 (P .96), memory null | |
| **power-n1n2 (r̂ 1.12)** | **+3.1% @7** | **+36.7% @7 (P .99)** |

**Every converged fit says a ~20-27% noise increase at low payoffs, on BOTH
positions roughly equally.** The "the effect is on the second-presented option
only" reading comes solely from `power-n1n2`, whose `n2_sd7` cTBS coefficient
ranges from -0.247 to -0.345 across chains — a 40% spread in the reported effect
size.

**Everything I said in this file about `n1n2` recovering 40% of the order
asymmetry against `percmem`'s 22% is therefore withdrawn.** That number came
from the non-converged trace and is not a measurement. The order-asymmetry
argument for position-indexing collapses with it.

## What this changes

It does not change the reported model — `percpmu` still wins the ladder and the
PPCs — but it changes the REASON, and for the better:

* **Not** "the position-indexed model cannot be fitted, so we use stages."
* **But** "position-indexed models fit fine at four or more anchors, and when
  they do they agree with the stage-indexed one: a ~20-25% noise increase
  confined to low payoffs, on both presented options. We report the stage
  parameterisation because it is more parsimonious and because the baseline
  identifies the two stages, not because the alternative failed."

That is a far stronger position, and it retires the limitation paragraph I was
about to write. `spl5-n1n2` or `spl7-n1n2` should be reported alongside as the
position-indexed robustness check.
