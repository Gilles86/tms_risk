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
