# What the reanalysis changed, and what the paper now has to say

Written 2026-08-03. A briefing for someone who has read the preprint (v8) but not the
last week of work. Everything here is fact with a source; the numbered sources are
files in this repo. Nothing below is speculation, and the open decisions are flagged
as such at the end.

Full detail: `notes/pmc_refit_results.md` (the model audit),
`notes/figure_plan_briefing.md` (what the refits established),
`notes/positive_memory_noise.md` (the memory-sign robustness test),
`notes/v8_stats_check.md` (the line-by-line audit of the v8 Results),
`notes/PROVENANCE.md` (which script produces which number).

---

## Bottom line

The paper's headline claim survives intact: **cTBS over numerosity-tuned parietal
cortex raises representational noise, and that noise change causes the risk-attitude
shift.** What changed is (a) *which* noise — it is now decisively the **shared
perceptual** component, not memory; (b) *how* the noise reaches choice — through
**prior attraction / bias**, not through psychometric flattening; and (c) the
**group-level magnitude-localisation** claim, which no longer holds at the group
level and now rests on a per-subject brain–behaviour correlation.

Four bugs were found. Three are in `bauer` (the modelling library) and predate the
preprint; one was ours, in the figure pipeline, and was caught before anything was
published.

---

## 1. The bugs

### 1.1 The published `flexible2` fit did not implement its own Methods

At bauer `ecc6454` (the commit the published traces were fitted with),
`_get_trialwise_evidence_sd` built the `shared_perceptual_noise` branch as

```python
spline_pars2 = pt.stack([parameters[l1] for l1 in labels1], axis=1)   # labels1, twice
```

i.e. it used the **memory** spline coefficients for *both* terms of the
first-presented option. So ν₁ = softplus(η_mem·B₁ + η_mem·B₂): perceptual noise never
entered the first-presented option, and "family 2" collapsed into two decoupled
per-position curves. The Methods describe ν₁ = ν_perceptual + ν_memory.

**Consequence:** the published perceptual-vs-memory decomposition is not what its
labels say. The published `flexible2` trace should be read as a *relabelled family-1
(per-position) fit*. Family 1 (`independent`) is unaffected. Fixed in bauer `b66c806`.

### 1.2 The choice rule changed meaning, silently

`ecc6454` computed the decision variable's SD from the **raw** evidence SD,
`diff_sd = sqrt(ν₁² + ν₂²)`, while shrinking the *mean* toward the prior — internally
inconsistent. `b66c806` propagates the evidence SD through the posterior mean:
`post_sd²/ν · p = ν·w·p`, with `w = σ²/(σ²+ν²)`. **HEAD is correct here** (it is KLW's
own algebra; see `notes/klw_variance_analysis.md`).

**Consequence:** ν does not denote the same quantity on the two sides of that commit,
so published and refit parameters are not directly comparable. This is *not* just a
reparameterisation — see §1.3.

### 1.3 The regression path ignored the declared priors on every softplus parameter

At `ecc6454`, `RegressionModel.build_hierarchical_nodes` set the Intercept's `mu`/
`sigma` only for the `identity` and `logistic` transforms; the `softplus` branch was
missing. So `risky_prior_sd` and `safe_prior_sd` silently got `Normal(0, 1)` on the
untransformed scale regardless of what `get_free_parameters` declared. HEAD added the
branch, which turns those into wildly diffuse, badly centred priors
(`Normal(33, 25)`, `Normal(22, 0.5)` on the softplus scale) — which is why the
unconstrained HEAD refits would not converge and needed `--constrain`.

**Consequence:** the published and refit fits land in different modes of a weakly
identified direction. Implied shrinkage weight and percepts for a safe option, 7 → 28 CHF:

| | w | perceived / objective |
|---|---|---|
| published `flexible1` (ecc6454) | 0.98 → 0.81 | 101% → 88% (near-veridical) |
| refit `flexible1nf` (HEAD) | 0.34 → 0.18 | 80% → 32% (heavily compressed) |

Both fit the choices about equally well, because choice depends on the *ratio* of two
percepts and both options compress toward their own priors.

Important: the priors **are** identified in this design, and the flat direction people
worry about is a degenerate special case we are not in. Scaling both shrinkage weights
by a common λ and sliding the prior means to compensate is exactly zero-cost only when
a single constant ν is shared by both options. Profiled on the real 8335-trial design:

| | λ = 0.5 | λ = 0.7 |
|---|---|---|
| ν constant **and** ν₁ = ν₂ | **0.0** nats | **0.0** |
| ν constant, ν₁ ≠ ν₂ | 932 | 343 |
| ν ∝ magnitude, ν₁ ≠ ν₂ (the real model) | 237 | 79 |

±5% in w costs 2.0 nats, ±10% costs 8 — a clean quadratic, not a ridge. **Asymmetric
noise between the two presentation positions is what identifies the priors.**

### 1.4 A reconstruction bug that affects every published noise curve and the r = −0.379

`get_sd_curve` rebuilds the spline basis from whatever x-grid it is handed, and patsy
places interior knots at *quantiles of that grid*. With `degree=3, df=5,
include_intercept=True` there is exactly one interior knot. The **fit** used the
paradigm, knot at **20 CHF**; `figure4.ipynb` and `neurobehavioral_correlates.ipynb`
passed `np.arange(7, 50)`, knot at **28**; `get_sd_curve`'s default `linspace(7, 112)`
gives **59.5**.

**Consequence:** correcting the anchoring moves the *published* fit's credible
perceptual window from 7–29 to **7–13 CHF** and its localisation test from 0.79 to
**0.95** — i.e. with the knots right, the published fits reproduce the preprint's
stated "approximately 7–14" range exactly. It also changes the brain–behaviour
correlation: ecc6454-anchored gives **r = −0.3792** (matching the paper's saved
−0.379173), HEAD's `get_sd_curve` gives **r = −0.3167** on identical inputs.

### 1.5 Our own bug, caught and gated

`decision_space.<label>.tsv` stored `norm.cdf((EV2−EV1)/s)` — P(choose the **second**
option) — under the name `p_vertex` and plotted it as P(chose risky). On risky-first
trials the second option is the safe one, so that column and the `effect` derived from
it were sign-flipped across half the design. Fixed at source, re-extracted, and
`validate_source_data.py` now asserts the invariants and gates every figure. No
published figure was ever affected.

---

## 2. What the results now say

All from refits with the intended composition, ν₁ = softplus(η_mem + η_perc),
ν₂ = softplus(η_perc). Traces in `derivatives/cogmodels.overnight/`, bauer `e05f73a`,
4 chains × 5000 tune + 5000 draws, numpyro/JAX on the T4 nodes, constrained priors.

### 2.1 Model comparison (16 cells: two noise families × four cTBS loci, all refit)

| model | ELPD (LOO) | cost vs best |
|---|---|---|
| **Family 2, cTBS on perceptual noise only** | **−4157.7** | — |
| Family 2, cTBS on both terms | −4159.7 | +2.0 ± 3.9 |
| Family 1, cTBS on both options | −4184.6 | +27.0 ± 10.7 |
| Family 2, memory noise only | −4217.5 | +59.9 ± 12.2 |
| Family 1, first-presented option only | −4222.0 | +64.3 ± 13.4 |
| Family 1, second-presented option only | −4227.5 | +69.9 ± 13.1 |
| Family 1 null | −4271.8 | +114.1 ± 15.8 |
| Family 2 null | −4273.2 | +115.6 ± 14.4 |

Two things matter. (a) **Dropping the memory term costs nothing** (2.0 ± 3.9) while
**dropping the perceptual term costs 59.9 ± 12.2** (4.9 SE) — the effect is on shared
perceptual noise, and the perception-only model is not merely adequate but the *best*
model in the family. (b) Models carrying an explicit **presentation-position**
parameter fit *worse* (+64, +70) than one that has none — the quantitative answer to a
reviewer who suspects the order effect was fitted rather than emergent.

Preprint Table 1 for comparison: full −4167.4, drop-one −26.2 / −26.3, null −80.2. The
*structure* replicates; the magnitudes are larger and the winner has moved.

Weber (log-space, scalar-invariant) baselines were refit too. In the Weber model the
cTBS effect on perceptual noise is **−0.001, i.e. nothing** — a
magnitude-proportional perceptual perturbation does not fit these data.

Convergence: the whole nested family now converges (r̂ ≤ 1.010, ESS ≥ 418). The
published `flexible1_null` had **r̂ = 2.12 and 5165 divergences**. The one exception is
the family-2 null (r̂ = 1.020, ESS 115, 664 divergences) — treat its ELPD as
indicative; the family-1 null *did* converge and gives the same verdict.

### 2.2 The shape of the noise function

- Perceptual noise log-log slope **0.45**, not the 1.0 Weber's law predicts:
  ν ∝ √n. Poisson-like, as a numerosity-tuned population code would give.
- Memory noise is essentially **flat** (log-log slope 0.05, ≈ 0.7 CHF).
- cTBS adds a roughly **constant absolute** amount: +0.191 [+0.041, +0.419] at 7 CHF,
  +0.195 [+0.038, +0.376] at 28, +0.221 [−0.195, +0.629] at 112. The 95% CrI excludes
  zero at **88 of 120 grid points**, spanning 7 to ~80 CHF.
- In **proportional** terms that constant injection is **15.2% at 7 CHF and 4.1% at
  112**, because baseline noise grows with magnitude.

**This resolves an apparent contradiction in the paper.** The psychophysical (probit)
analysis finds reduced consistency specifically for low-stake trials; the PMC noise
curve looks flat in CHF. Both are right — the probit lives on log(ratio), and a
constant absolute injection *is* localised on that scale. It is a units artifact. Any
figure or sentence making the "specific to small magnitudes" claim must use the
**relative (%) or log-log** version, never the absolute one.

### 2.3 The mechanism: bias, not flattening

Channel decomposition of the model's ΔP(chose risky) (`flexible2nf`):

| channel | risky second | risky first |
|---|---|---|
| bias (prior attraction) only | **+0.074** | +0.014 |
| noise (added randomness) only | −0.003 | −0.003 |
| full model | +0.072 | +0.011 |

The effect travels through **prior attraction, not psychometric flattening**. Noise is
still the cause; it acts via bias. Prior attraction is therefore *necessary* to the
account: without a prior, more noise can only flatten the psychometric curve, and the
data show a shift.

Model-free confirmation, no model needed: flattening must pull every choice proportion
toward 0.5, so wherever the baseline proportion is above 0.5 it must push it **down**.
Observed ΔP is **positive** in those bins.

### 2.4 Why the effect is order-specific — and it is not a fitted parameter

Three facts, none of them an order parameter:

1. ν₁ is built from both noise components and ν₂ from the perceptual one alone, so the
   same cTBS perturbation raises the first option's noise **1.43×** more (+0.224 vs
   +0.157 CHF over the safe range).
2. So the **safe** option loses 0.459 CHF of perceived value when presented first vs
   0.343 when second, while the **risky** option is nearly indifferent to position
   (−0.313 vs −0.320).
3. Choice tracks the gap between the two, which is **4.5× larger** when the risky
   option is second (−0.139 vs −0.031 CHF) → ΔP(risky) of +0.071 vs +0.009.

Observed data agree: mean ΔP(chose risky) is **+0.053 risky-second vs +0.006
risky-first**, peaking at +0.096 ± 0.041 at the smallest safe payoff.

(Reminder for whoever writes this up: `risky_first = True` means the **risky option
came first**, `utils/data.py:267`. The v8 manuscript has these labels swapped in two
correlations — §3.2 below.)

### 2.5 Magnitude localisation: no longer a group-level claim

P[Δν_perceptual(7 CHF) > Δν_perceptual(28 CHF)], on draws:

| fit | absolute (CHF) | relative (fraction of ν) |
|---|---|---|
| published `flexible1` (ecc6454) | **0.95** | **0.97** |
| published `flexible2` (ecc6454) | **0.96** | **0.98** |
| refit `flexible1nf` (HEAD) | 0.43 | 0.76 |
| refit `flexible2nf` (HEAD) | 0.35 | 0.73 |

The published fits are credibly localised on both scales; **the refits are not**. The
mechanical reason: the refit puts ν(7 CHF) = 1.49 where the published fit puts 0.42,
so the same absolute Δ is a much smaller fraction — which traces straight back to the
prior-scale difference of §1.3.

And it cannot be rescued by a more flexible spline. **One interior knot is the
resolution ceiling of this design**: df = 6, df = 9 and degree-2/df-5 all fail to
converge (r̂ ≥ 1.24, ESS ≤ 12), *including in the 2024 fits*, so this is not the
patch, the choice rule or the degree.

The proposed replacement was **individual-specific localisation tracking nPRF
amplitude** — the localisation slope Δν_perc(7) − Δν_perc(28) per subject, against
that subject's cTBS amplitude change:

| amplitude measure | `flexible1` | `flexible1nf` | `flexible2nf` |
|---|---|---|---|
| **m2** (full per-session model) | r = −0.33, p = .051 | **r = −0.43, p = .010** | r = −0.33, p = .051 |
| m1 (canonical), signal-voxel mean | −0.25, p = .15 | −0.08, p = .64 | −0.23, p = .19 |

**As of 2026-08-03 this result should be treated as not safe to publish.** See §7 — it
exists only under m2, m2's Δamplitude does not measure the same thing as m1's
(r = +0.086 between them), and it does not survive controlling for m2's own nuisance
parameters. The m1/m2 fork is resolved in §7 in favour of m1, under which the
correlation is r = −0.08.

### 2.6 The memory contribution is negative at small payoffs, and that is real

Under the default composition ν₁ − ν₂ at 7 CHF is **−0.180 CHF, CrI [−0.384, −0.043]**,
crossing zero at 11.5 CHF: the first-presented option is encoded *more* precisely than
the second-presented one at small payoffs. Constraining it non-negative
(`ν₁ = ν₂ + softplus(η_mem)`) costs **55.7 nats at dSE 10.4** and pins the estimate to
+0.031 [+0.004, +0.106] — flat against the boundary. The cTBS result is unchanged
either way (if anything the constrained fit gives a *larger* effect).

**Decision taken: report the default composition**, and describe the composition in
the Methods as bauer implements it — `ν₁ = softplus(η_mem + η_perc)`, saying explicitly
that this leaves the sign of ν₁ − ν₂ free. Mention the constrained fit as a robustness
check.

### 2.7 A caveat that must be stated, not hidden

The fitted priors sit **below the entire payoff range**: `safe_prior_mu` = 3.57 CHF
(CrI −3.70 to 5.55) against payoffs of 7–112, prior SD ≈ 1.2. That compresses an
objective 28 CHF into a perceived 9 CHF, which is not credible as a subjective value —
the prior is doing the work of a compressive value function, which is how this
architecture produces risk aversion at all.

It is **not** an artifact of our priors: `--constrain` centres `*_prior_mu` on the
empirical payoff mean with σ = 10 (bauer's own default is σ = 25, weaker), and the
posteriors land 1.1–2.7 SD *below* that centre. The likelihood drives them down against
a resisting prior. A pinned-to-objective-prior comparison fit exists
(`objprior_perception`); note that under an objective prior ~60% of trials sit below
the prior mean, so cTBS would push most percepts *up* — the opposite direction.

**Do not quote the percept-compression numbers as perceived values.**

---

## 3. What has to change in the manuscript

### 3.1 From the model reanalysis

1. **The perceptual/memory decomposition must be re-stated from the refits.** The
   published `flexible2` trace does not implement its Methods (§1.1). Check which trace
   the preprint's Fig 4B/4C was drawn from before reusing either panel.
2. **Table 1 is replaced** by the 16-cell refit table (§2.1). The new headline is
   "cTBS acts on shared perceptual noise; dropping the memory term costs nothing".
3. **The Fig-5 sentence "noise contributing the bulk of the effect" is the one line
   that must change** (§2.3). Noise remains the cause; it acts through prior
   attraction, not through added randomness.
4. **The "specific to small magnitudes" claim must be re-scoped.** At the group level
   the cTBS injection is constant in absolute CHF and only magnitude-specific in
   *relative* terms (§2.2); the strong localisation claim now rests on the per-subject
   brain–behaviour correlation (§2.5), with its two caveats stated.
5. **Add the position-parameter result** (§2.1b) — it pre-empts the obvious reviewer
   objection that the order effect was fitted rather than emergent.
6. **Add the noise exponent 0.45** (√n, Poisson-like) and the Weber-baseline failure
   (§2.2) — this is a positive result about the code, not just a control.
7. **State the prior caveat explicitly** (§2.7) and remove any quoted perceived values.
8. **Methods**: write the noise composition as bauer implements it and note the free
   sign of ν₁ − ν₂ (§2.6).

### 3.2 From the v8 stats audit (independent of the bugs, all verified)

All 12 reported statistics were re-run and reproduce. Three fixes, none of which
changes a conclusion:

1. **`p = 0.004` for r = 0.76 is impossible** at N = 35 (t(33) = 6.65, p ≈ 1.5 × 10⁻⁷).
   It comes from a *different analysis* — the MAP point-estimate version of the same
   correlation (r = 0.4716, p = 0.004234). Report the Bayesian pair: *r*(33) = 0.76,
   *p* < 0.001.
2. **The trial-order labels in the two split correlations are swapped.** Per the code,
   risky-**first** gives r = −0.5944 and risky-**second** gives r = −0.3285; the
   manuscript attaches each to the other condition, and the summary sentence three
   lines later ("the effect only occurs when safe options are presented first")
   inverts with it.
3. **`r(34)` should be `r(33)`** in all five correlations (N = 35 ⇒ df = N − 2). The
   t-test and ANOVA dfs of 34 are correct.

Plus, in Fig 2's paragraph: the preferred-numerosity pair (17.8 / 14.8) and the
dispersion pair (0.9083 / 0.7733) are **correct numbers assigned to the wrong
conditions**. IPS = 17.79 and 0.9083, vertex = 14.75 and 0.7733; in a paragraph whose
convention is vertex → parietal, both should read the other way round (i.e. cTBS went
with a non-significant *increase* in both). Also note the paper pairs a natural-space
descriptive with a log-space test — defensible, but the Methods should say so.

Tails: items 1–4 come from a cell that runs two-sided and the paper halves it. Only the
cvR²-proportion test is natively one-sided. Defensible given directional hypotheses,
but the code as written does not compute those one-sided values.

### 3.3 Supplement

**S1.1–S1.3's right column is a ratio (IPS / vertex), not a difference**, despite being
titled "IPS − Vertex" (colour scales 0.90–1.10 / 0.95–1.05, centred on 1). Only S1.4 is
a genuine subtraction. The regenerated figures label this correctly; the S1.1 and S1.3
captions still need the fix.

---

## 4. What did NOT change

- cTBS raises representational noise in numerosity-tuned parietal cortex. Decisive:
  115.6 ± 14.4 nats against the null (8.0 SE).
- The effect is order-specific, and larger when the risky option comes second.
- The imaging results (Fig 2) are untouched — nPRF amplitude, explained variance,
  decoding accuracy and the decoding × order interaction all reproduce exactly.
- "Perceptual, not memory" is **version-independent**: all four fits (both bauer
  versions, both families) put the effect on shared perceptual noise with P ≈ 0.92–0.96
  over 7–14 CHF, and none finds a credible memory effect. This was the preprint's
  claim and it holds.
- The family-1 ↔ family-2 reparameterisation is verified: within each bauer version the
  two families land on the same perceptual effect to ~0.04 CHF.

---

## 5. Current plan for the figures

House rules now enforced across every figure: **IPS = `#d62728` red, vertex =
`#2ca02c` green, and a *difference* gets its own ink (black/mako), never one of the
condition colours.** Error-bar conventions are one rule with three cases, written down
in `notes/ERROR_BARS.md`. All figures rebuild from `notes/data/*.tsv` alone — no trace,
no bauer, no GPU — and `validate_source_data.py` gates the source data first.
`notes/figures/` is output and untracked; `notes/data/` is the source of record and is
in git.

| Figure | Plan | Script | Status |
|---|---|---|---|
| **Fig 2** | Imaging. Unchanged. | `tms_risk/notebooks/figure2.ipynb` | done |
| **Fig 3** | The effect argued **without the cognitive model** — and the probit *is* that argument: it assumes nothing beyond "there is a psychophysical curve". Laid out as the published Figure 3: block A the psychometric functions, block B the two probit parameters, presentation order as the row variable throughout. Block B draws the **paired difference posterior** (IPS − vertex) rather than the published figure's mirrored per-condition marginals — see below. | `plot_fig3_probit.py` | **decided** |
| **Fig 4** | Which model the data prefer, and what its noise function looks like: (A) the posterior predictive check run against **both** noise families side by side — Weber PMC and Flexible PMC, split by presentation order and stake, i.e. the v8 Fig-4A panel; (B) the winner's perceptual noise on log-log axes vs a slope-1 Weber reference, with ν₁ dashed alongside so the memory contribution reads at its own (small) scale; (C) the cTBS increase as **% of baseline** with its CrI; (D) all 16 models on one ELPD axis with dSE. The memory contribution no longer gets a panel or a credible-interval bar of its own — a tenth of a franc against a noise level of one to six does not carry a panel. | `plot_fig4_model.py` | drafted |
| **Fig 5** | Where in the decision space cTBS changes behaviour: distortion (a) × leverage (b) → effect (c), checked against data (d), two presentation-order rows sharing one colour scale per column. A 1D alternative on the ratio axis also exists. | `plot_fig5.py` / `plot_fig5_1d.py` | drafted, 1D alternative **open** |
| **New mechanism figure** | How two compressed payoff curves produce a shifted *ratio* — the percept → ratio → choice step. **Four prototypes, pick one.** | `proto_ratio_chain.py`, `proto_ratio_waterfall.py`, `proto_percept_nomogram.py`, `proto_shrink_weight.py` | **open choice** |

**Why Fig 3B departs from the published panel.** The published Figure 3B mirrored the
two per-condition posteriors against each other (vertex up, IPS down) in the
parameters' own units. Those marginals are strongly correlated across posterior draws,
so how much they overlap says almost nothing about the difference: for RNP with the
risky option second the two marginals overlap over most of their range, while the
paired difference is [+0.023, +0.089] and excludes zero at p < 0.001. The panel and
the p-value beside it were telling the reader opposite things. Drawing the paired
(chain, draw) difference instead makes the plotted density, the credible interval and
the p-value one and the same object, and puts zero on the axis. What is lost is the
parameters' absolute scale and the published panel's reference marks (0.55 for
risk-neutral, the low/high-consistency anchors); the anchors are kept as end-labels on
the Δ axis, and `plot_fig3_probit.py` prints the per-condition posterior means for the
caption.

**Fig 3 split by stake (`fig3_probit_stake`).** Added 2026-08-03 as a variant, not a
replacement. Stake = (n_safe + n_risky)/2 split at each participant's own median, the
repo's existing convention; four independent hierarchical probits, one per (order ×
stake) cell, ~2080 trials and all 35 subjects each, all converged (r̂ = 1.000, min ESS
1602). `analyze_probit_by_stake.py` fits, `plot_fig3_probit_stake.py` draws.

| | Risky first | | Risky second | |
|---|---|---|---|---|
| | Low | High | Low | High |
| Δ slope | −0.01 (p .48) | +0.28 (p .14) | **−0.40** (p .059) | +0.08 (p .38) |
| Δ RNP | +0.007 (p .36) | +0.003 (p .44) | **+0.053** (p .006) | **+0.047** (p .047) |

**This does not support magnitude localisation, and it is a fourth independent route
to that conclusion** (cf. §7.4's three). The risk-attitude shift on risky-second
trials is essentially the same size at high stakes as at low (+0.047 vs +0.053) — if
cTBS were adding noise only to the small numerosities the nPRF population prefers, the
high-stake cell should be near zero and it is not. The consistency loss *is* confined
to the low-stake cell, which is the one thing here that points the predicted way, but
halving the trials per cell widens its interval to include zero.

Two caveats before this is quoted. (a) These are four separate fits, so the table is
descriptive; the actual claim "the effect differs by stake" is a stake × stimulation ×
order interaction and has not been tested in a single model. (b) Each cell has half
the data, so every interval here is ~√2 wider than the pooled one in Fig 3 — a cell
going from p = 0.012 to p = 0.059 is what losing half the trials looks like, not
evidence of absence.

The four mechanism prototypes, so the choice can be made on the merits:

- **`ratio_chain`** — the decision variable is a ratio, and a ratio is a *difference* on
  a log axis, so three panels read left to right as one exact arithmetic statement:
  `dlog(ratio) = dlog(risky) − dlog(safe)`. The order effect becomes a purely visual
  fact (curves coincide in the top row, separate in the bottom).
- **`ratio_waterfall`** — the same identity as two opposing contributions that mostly
  cancel: the safe option losing value *raises* the ratio, the risky option losing value
  *lowers* it, so the choice effect is a residual. One pair of bars and a net dot per
  stake, with the observed effect alongside for checking.
- **`percept_nomogram`** — the compression drawn as a *mapping*: objective → perceived →
  ratio, three vertical scales per stake, tie-line length = compression. Shows directly
  that all tie-lines point down and the safe one lengthens more under cTBS.
- **`shrink_weight`** — the most mechanistic: percept = w·n + (1−w)·μ, so
  d(percept) = dw · (n − μ) is knob × lever. Separates *why* the safe option loses more
  at low stakes (the knob turns further) from what happens at high stakes (the risky
  option's lever is longer, and only the knob keeps the net sign).

Supporting / supplementary material already generated: per-model PPCs for all 14 fits
(`ppc/`, including the by-safe-payoff and stake-tercile breakdowns with Vincentized
bins), a `noise_variants` panel comparing the fitted noise contrast across all eight
flexible variants, `prior_comparison` (free vs pinned prior), `spline_ladder` (the
resolution-ceiling evidence of §2.5), `percept_distortion`, `why_risky_second`,
`decision_space`, and the S1 perceptual heatmaps. Sorted by plot type with
`organize_figures.py --apply`; `notes/figures/paper/` holds whichever variant the
manuscript currently uses.

---

## 6. Open decisions (for the discussion this document is for)

1. ~~Fig 3: model-free signature argument, or the probit refit in the preprint's
   layout?~~ — **resolved 2026-08-03: the probit.** A probit assumes nothing beyond
   "there is a psychophysical curve", so it already *is* the model-free argument; the
   flattening-vs-shift signature panel (`plot_fig3_modelfree.py`) was a second,
   harder-to-read route to the same conclusion and is retired. The script stays in the
   tree but nothing in the paper should point at it.
2. **Which of the four mechanism prototypes becomes the new figure** — and whether it is
   main text or supplement.
3. **Fig 5: heatmaps or the 1D ratio-axis version.** The heatmaps show *where* in the
   design the effect lives; the 1D version is far easier to read and puts the
   indifference crossings and leverage peaks on the same axis.
4. ~~m1 vs m2~~ — **resolved in §7: use m1.** What remains open is the consequence:
   the brain–behaviour correlation does not survive under m1, so the per-subject
   localisation result should come out of the paper unless something else supports it.
5. **How hard to state the localisation claim** — and per §7.4, whether to state it at
   all. Three independent routes (group noise curve, per-voxel neural test, model-free
   behavioural interaction) now all fail to find magnitude localisation. What survives
   is a *targeting* fact and a *relative-scale* statement, not an empirical
   localisation result.
6. **Whether to report the pinned-objective-prior fit** as a robustness check for §2.7,
   given it may be structurally unable to produce the observed effect.

---

## 7. m1 vs m2, and whether "cTBS mostly affects low numerosities" can be backed

Added 2026-08-03. All numbers below were computed from `notes/data/*.tsv` and the local
BIDS behavioural data; scripts are in the session scratchpad and should be promoted to
`tms_risk/modeling/scripts/` if any of this goes in the paper.

### 7.1 What the two models actually are

`modeling/fit_regression_nprf.py::get_model`, a `RegressionGaussianPRF` with
`0 + C(session)` regressors on:

| | amplitude | mu | sd | baseline | mu/sd grid |
|---|---|---|---|---|---|
| **m0** | pooled | pooled | pooled | pooled | 50 points |
| **m1** | **per session** | pooled | pooled | pooled | 50 points |
| **m2** | per session | per session | per session | per session | **10 points** (`mus[::5]`) |

Sessions 2 and 3 are the two TMS sessions, so "per session" *is* the IPS-vs-vertex
contrast. Verified in `m1_tms_param_shifts.tsv`: under m1, max |IPS − vertex| is
exactly **0.000** for mu, sd and baseline, and 1.060 for amplitude. m1's Δamplitude is
therefore a **gain change at fixed tuning**; m2's is a gain change estimated jointly
with three other per-session parameters on the same data.

### 7.2 m2 does not earn its extra parameters in the target ROI

cvR² win proportions on the non-noise voxel pool (a voxel counts if any of m0/m1/m2
beats the per-voxel training mean; only 4–11% of voxels qualify), group mean over
subjects, `cvr2_model_comparison_intersect.tsv`:

| ROI | m0 | m1 | m2 | m2 − m1, paired |
|---|---|---|---|---|
| **NPC12r** (the target) | 0.186 | 0.395 | 0.420 | +0.025, t = 0.35, **p = 0.73** |
| NPCr | 0.188 | 0.388 | 0.424 | +0.035, p = 0.61 |
| NF1 | 0.150 | 0.363 | 0.487 | +0.123, p = 0.096 |
| NTO | 0.192 | 0.357 | 0.450 | +0.093, p = 0.16 |
| NPCl | 0.162 | 0.321 | 0.517 | +0.196, **p = 0.011** |

m2 is nominally ahead everywhere, but **in the ROI the paper is about, the gap is
0.025 and completely non-significant.** Both m1 and m2 clearly beat m0, so
per-session amplitude *is* worth modelling — the question is only whether per-session
*tuning* is, and in NPC12r it is not.

### 7.3 The decisive fact: m1 and m2 do not measure the same quantity

Per-subject cTBS amplitude change (IPS − vertex), the two models against each other,
n = 35:

- Pearson **r = +0.086** (p = 0.62), Spearman ρ = +0.022
- **Sign agreement 15/35** — below the 17.5 expected by chance

These are not two noisy estimates of one number. And within m2, Δamplitude is strongly
entangled with the nuisance parameters it now shares the fit with:

| | correlation with m2 Δamplitude |
|---|---|
| Δbaseline | **r = −0.582, p = 0.0002** |
| Δmu | r = −0.333, p = 0.051 |
| Δsd | r = +0.219, p = 0.21 |

That −0.58 is the textbook amplitude/baseline trade-off: a session whose fitted
baseline drops gets a compensating amplitude rise. m2's "amplitude change" is partly a
re-partitioning between gain and offset, which has no theoretical status in the TMS
hypothesis.

**The brain–behaviour correlation inherits this.** For `flexible1nf`, the correlation
with the noise-localisation slope is r = −0.432 (p = 0.0096) for m2's Δamplitude, but:

| | r | p |
|---|---|---|
| m2 Δamplitude, raw | −0.432 | 0.0096 |
| …partialling out Δbaseline | −0.322 | 0.064 |
| …partialling out Δbaseline and Δmu | −0.285 | 0.108 |
| m2 Δbaseline, partialling out Δamplitude | +0.087 | 0.63 |
| **m1 Δamplitude (canonical model)** | **−0.083** | **0.64** |

Amplitude beats baseline as the predictor of the two (the reverse partial is clean),
but the effect does not survive controlling for the parameter it is confounded with,
and it is absent under the canonical model.

### 7.4 Recommendation

**Use m1** for the amplitude/gain change, for decoding, Fisher and mc_decode, and for
anything brain–behaviour. It is the paper's canonical model, m2 buys nothing
cross-validated in NPC12r, and m1's Δamplitude is the interpretable quantity — a gain
change at fixed tuning — where m2's is entangled with baseline at r = −0.58. m2 also
grids mu/sd 5× more coarsely.

**Use m2 only where m1 structurally cannot answer the question**: any per-session test
of `mu` or `sd`. Under m1 these are session-invariant by construction and return
`t = NaN` (the CLAUDE.md trap #1). Report such tests as secondary, and note the coarser
grid.

**Consequence:** the per-subject brain–behaviour correlation (§2.5) does not survive
this choice and should come out of the paper. Reporting m2 there while the rest of the
paper runs on m1 would be a model choice made by which one gives the result.

### 7.5 Can "cTBS mostly impacts lower numerosities" be backed? Three routes, all negative

**Route 1 — neural.** Do voxels tuned to *low* numerosities lose more amplitude?
Per-subject voxelwise Spearman(log preferred numerosity, Δamplitude), tested across
subjects (the within-subject test; pooling voxels across subjects is
pseudoreplication). From `m1_amplitude_by_pref_n.tsv`, 2738 voxels surviving a
degenerate-fit clean, 31 subjects with ≥ 20 voxels:

    mean rho = -0.040,  t(30) = -0.75,  p = 0.46,  11/31 subjects in the predicted direction

A null, if anything in the wrong direction. Pooled quintiles *do* show the predicted
pattern (median Δamp −0.035 / −0.064 / −0.087 / −0.024 / **+0.062** from lowest to
highest preferred numerosity), but that pattern is between-subject, not within — which
is exactly the trap the within-subject test exists to catch. Caveat: this voxel
aggregation and cleaning are my own choices, and 31.5% of raw rows had an amplitude
within 1e-6 of zero.

**Route 2 — behavioural, model-free, and this is the important one.** The quintile
descriptive looks dramatic: on risky-second trials the cTBS effect on P(chose risky)
is **+0.160 ± 0.045 (p = 0.0013)** in the smallest risky-payoff bin (7–17) and
+0.029 to +0.046 (all n.s.) in the other four. But "significant here, not there" is
not a test of a difference. The proper within-subject interaction:

| test (risky-second trials, n = 35) | effect | t(34) | p |
|---|---|---|---|
| median split on **n_risky**, low − high | +0.067 | 1.88 | 0.069 |
| median split on **n_safe**, low − high | +0.017 | 0.47 | 0.64 |
| per-subject slope of the effect on log(n_risky) | −0.050 | −1.26 | 0.22 |
| median split on **log(risky/safe) ratio**, low − high | +0.058 | 2.07 | **0.048** |

n_risky and ratio are confounded (r = +0.589) — a small risky payoff *forces* a small
ratio. n_safe is the clean magnitude axis: five fixed levels, each spanning the same
ratio range, and log(n_safe) is orthogonal to log(ratio) (**r = +0.024**). Testing each
while holding the other fixed:

| | low | high | difference | p |
|---|---|---|---|---|
| **Ratio**, within each subject × n_safe level | +0.078 | +0.023 | +0.055 | 0.076 (Wilcoxon 0.053) |
| **Magnitude**, within each ratio tercile | +0.042 | +0.038 | **+0.004** | **0.92** |

And in a per-subject logistic regression carrying both interactions:
`ips × log(ratio)` β = −0.49 (p = 0.24), `ips × log(n_safe)` β = +0.17 (p = 0.65).

**On the clean magnitude axis the effect is flat: +0.042 vs +0.038.** The apparent
low-magnitude localisation in the n_risky bins is the ratio confound. What is
(marginally) localised is the *ratio* — i.e. proximity to indifference — which is the
leverage story of Figure 5, not a magnitude story.

**Route 3 — the model.** Already in §2.5: group-level P[Δν(7) > Δν(28)] = 0.35–0.43 on
the absolute scale, 0.73–0.76 on the relative scale. Not credible either way, and the
spline cannot be refined (§2.5).

### 7.6 What can still honestly be said

1. **The targeting fact, which is real and reportable.** The stimulated voxels are
   overwhelmingly tuned to the low end of the payoff range: median preferred numerosity
   **9.2**, 25th–75th percentile **7.0–13.9**, 95th percentile 31.9, against payoffs
   spanning 7–112. This motivates the low-magnitude hypothesis and is a statement about
   *where the stimulation landed*, not a claim that the behavioural test confirmed it.
2. **The relative-scale statement.** A constant absolute noise injection against a
   baseline that grows as √n *is* proportionally larger at small magnitudes (15.2% at
   7 CHF vs 4.1% at 112). True, but it is close to a definitional consequence of
   "constant absolute + growing baseline" rather than an independent finding — say it
   as a description of the fitted noise function, not as evidence of localisation.
3. **The ratio/leverage statement**, which the data do marginally support (p = 0.048
   unadjusted, p = 0.076 holding magnitude fixed) and which the model independently
   predicts: the effect appears where choices sit near indifference.

**Recommendation: drop "cTBS mostly affects low numerosities" as a group-level
empirical claim.** Keep it as the targeting rationale (7.6.1), and let the behavioural
localisation claim be about proximity to indifference instead. Three independent
routes now fail to support the magnitude version, and the one analysis that did
support it (§2.5's r = −0.43) is the one §7.4 rules out.
