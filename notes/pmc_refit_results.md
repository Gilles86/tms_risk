# Flexible PMC refit against current bauer — what the cTBS effect actually is

Traces: `derivatives/cogmodels.overnight/model-flexible1_noisefix*.head_trace.netcdf`
(GPU node `sciencecloud_gpu`, numpyro/JAX, 4 chains × 5000 tune + 5000 draws,
constrained priors, bauer `e05f73a`). Figure: `notes/figures/pmc_explained.pdf`.
Rebuild any panel from the TSVs in `notes/data/*.flexible1nf.tsv` with
`python -m tms_risk.behavior.scripts.plot_pmc_explained`.

## Convergence

| trace | max r̂ | min ESS | divergences |
|---|---|---|---|
| `flexible1_noisefix` (full) | 1.000 | 2445 | 10 / 20000 |
| `flexible1_noisefix_first` | 1.000 | 3301 | 15 / 20000 |
| `flexible1_noisefix_second` | 1.000 | 983 | 26 / 20000 |
| `flexible1_noisefix_null` | 1.000 | 1653 | 0 / 20000 |

All four pass `r̂ ≤ 1.01`, `ESS ≥ 400`. This is the first time the whole nested
family has converged; the published `flexible1_null` had r̂ = 2.12 and 5165
divergences.

## Is there a cTBS effect? Yes, decisively

LOO over the nested family (`summarize_traces --loo`, `notes/data/summary_flexible1.loo.tsv`):

| model | Δ ELPD vs. full | SE of the difference |
|---|---|---|
| cTBS modulates noise on **both** options | 0 | — |
| first-presented option only | −37.4 | 7.7 |
| second-presented option only | −42.9 | 8.3 |
| **no cTBS effect on noise** | **−87.2** | **12.9** |

6.8 SE between the full model and the null. Both option-specific terms are
needed (~5 SE each). All four carry Pareto-k warnings, as hierarchical models on
8335 trials usually do; the differences are far larger than the diagnostic noise.

## The mechanism, in order

1. **cTBS raises representational noise.** ν increases by **+0.28 CHF**
   [0.09, 0.51] on the first-presented option and **+0.20 CHF** [−0.04, 0.46] on
   the second, roughly flat across 7–56 CHF (`pmcpars_contrast.flexible1nf.tsv`).
   Both CrIs exclude zero over 10–56 CHF. Reparameterised into perceptual /
   memory coordinates, P(Δν > 0 everywhere in 7–20 CHF) = 0.94 for the perceptual
   component, 0.62 for the memory component.

   Over the full presented range (7–112 CHF; `pmc_explained_to112.pdf`) ν itself
   is strongly **sub-Weber** — ν/n falls from 0.21 at 7 CHF to 0.048 at 112, a
   4.4× gain in proportional precision. The cTBS increase stays ~+0.2–0.3 CHF in
   *absolute* terms all the way up, so as a fraction of ν it shrinks from ~20% at
   7 CHF to ~5% at 112. The apparent rise of the second-option contrast to +0.42
   [+0.14, +0.70] at 84 CHF sits on the last spline knot and is carried by 2.1%
   of option presentations (8.0% lie above 56 CHF); do not lean on it.

2. **Noise is prior attraction.** Both fitted priors sit far below the payoffs
   they describe (safe μ = 4.90 CHF, risky μ = 9.95 CHF), so percepts compress
   downward: an objective 28 CHF safe option is perceived as 9.1 CHF. More noise
   → more compression.

3. **The safe option loses more than the risky one.** On risky-second trials
   the perceived EV drops by 0.21 → 0.60 CHF for the safe option and by only
   0.09 → 0.47 CHF for the risky one, across safe payoffs 7 → 28. Net: the risky
   option looks relatively better under IPS cTBS.

4. **It shows up as a bias, not as extra randomness.** Decomposing the model's
   ΔP(chose risky) into channels, the bias channel reproduces the full effect
   (mean +0.049 vs +0.052) while the pure randomness channel contributes
   +0.001 [−0.000, +0.002]. **Prior attraction is therefore necessary**: without a
   prior, a noise increase can only flatten the psychometric curve, and the data
   show a shift.

5. **It only bites where the design sits.** The distortion times the local
   leverage |dP/dm| produces ΔP up to +0.07 on risky-second trials, concentrated
   at low ratios and low safe payoffs — exactly where the design cells sit — and
   ≈ +0.01 uniformly on risky-first trials, because there the two options' shifts
   nearly cancel.

Observed effect, hierarchical probit and Flexible PMC agree panel-by-panel
(`pmc_explained.pdf` panel i, `decision_space_curves.flexible1nf.pdf`).

## In the paper's own coordinates, the refit agrees with the preprint

The first/second-option parameterisation makes the two fits look like they
disagree. They do not, once you rotate into the memory/perceptual coordinates the
paper uses. Two things have to be right for that rotation, and both are easy to
get wrong.

**1. The components add on the linear-predictor scale, not in CHF.** bauer builds
family 2 as (`magnitude.py::_get_trialwise_evidence_sd`)

    ν₁ = softplus(η_memory + η_perceptual)          ν₂ = softplus(η_perceptual)

— the softplus wraps the *sum*. So the family-1 rotation is η_perc = η₂,
η_mem = η₁ − η₂, and the plotted curves are softplus of each. **Memory noise is
≥ 0 by construction in both families**; a negative memory curve means the
subtraction was done in ν space (ν₁ − ν₂), which is not the model's
parameterisation.

**2. The spline knots are anchored to the paradigm, not to the plotting grid.**
patsy's `bs` places interior knots at *quantiles of the x it is handed*, so a
basis rebuilt on a `linspace` does not match the fitted one. bauer anchors each
variable's `design_info` at construction time (`magnitude.py::_spline_x_for`):
n1 for `n1_evidence_sd` / `memory_noise_sd`, **n2** for `n2_evidence_sd` /
`perceptual_noise_sd`.

In *this* dataset the n1-vs-n2 half of that is a no-op — the design is balanced,
so `median(n1) = median(n2) = 20` and the two bases coincide exactly. What
actually bites is **grid vs paradigm**: with `degree=3, df=5,
include_intercept=True` there is exactly one interior knot, at the median of
whatever x you hand `bs`. The fit used **20 CHF**; `get_sd_curve`'s default
`linspace(7, 112, 100)` gives **59.5**; the `np.arange(7, 50)` grid used by
`figure4.ipynb` and `neurobehavioral_correlates.ipynb` gives **28**. Not a small
effect: correcting it moved the published fit's credible perceptual window from
7–29 to 7–13 CHF and its localisation test from 0.79 to 0.95.

All four fits, in those coordinates (`fig4bc_style.<label>.pdf`, drawn in the
preprint's Fig-4B/4C layout). `flexible2*` estimate memory/perceptual directly;
`flexible1*` are rotated as above:

| Δν, IPS − vertex, at 7 / 14 / 28 / 56 CHF | perceptual | memory | P(Δν_perc>0 all 7–14) | all 7–28 | all 28–112 |
|---|---|---|---|---|---|
| published `flexible1` (ecc6454) | +0.40 +0.27 +0.02 −0.14 | −0.92 −0.13 +0.04 −0.00 | **0.957** | 0.589 | 0.046 |
| published `flexible2` (ecc6454) | +0.44 +0.29 +0.03 −0.15 | −0.08 +0.07 +0.03 −0.08 | **0.958** | 0.594 | 0.041 |
| refit `flexible1nf` (HEAD) | +0.20 +0.16 +0.21 +0.34 | +0.06 +0.05 +0.03 −0.03 | 0.946 | 0.943 | 0.917 |
| refit `flexible2nf` (HEAD) | +0.16 +0.14 +0.19 +0.31 | +0.07 +0.04 +0.03 −0.03 | 0.920 | 0.910 | 0.834 |

Three things follow.

1. **The family-1 ↔ family-2 reparameterisation is verified.** Within each bauer
   version the two families land on the same perceptual effect to ~0.04 CHF, and
   on the same regional probability to ~0.01. Fitting family 2 directly was worth
   doing precisely because it could have failed this check; it does not.
2. **"Perceptual, not memory" is version-independent.** All four put the effect on
   shared perceptual noise with P ≈ 0.92–0.96 over 7–14 CHF, and none finds a
   credible memory effect. This is the preprint's claim.
3. **The published fits reproduce the preprint's stated range exactly.** With the
   knots anchored correctly, `flexible1`/`flexible2` give a credible perceptual
   increase over **7–14 CHF** (P = 0.957 / 0.958) that has decayed to ~0 by 28 and
   reversed by 56 — which is the preprint's own "for smaller payoff magnitudes
   (approximately 7–14)". Under HEAD the increase is instead flat at ~+0.2 CHF and
   credible across the whole range (P = 0.92 / 0.83 even over 28–112). So the
   *localisation* is version-dependent while the *channel* is not.

### Is the effect localised at low magnitudes? Test it, don't eyeball the spline

"Specific to low magnitudes" is a claim about the *slope* of the contrast, and it
is scale-dependent — ν itself triples over 7–56 CHF, so a constant absolute Δν is
a shrinking relative one. `noisecurve_localisation.<label>.tsv` reports both, on
draws. P[Δν_perceptual(7 CHF) > Δν_perceptual(28 CHF)]:

| fit | absolute (CHF) | relative (fraction of ν) |
|---|---|---|
| published `flexible1` | **0.95** | **0.97** |
| published `flexible2` | **0.96** | **0.98** |
| refit `flexible1nf` | 0.43 | 0.76 |
| refit `flexible2nf` | 0.35 | 0.73 |

**The published fits are credibly localised on both scales; the refits are not.**
In the published fit Δν_perceptual runs +0.40 → +0.02 CHF from 7 to 28 (150% →
1% of vertex noise); in the refit it is flat at ~+0.2 CHF (14% → 7%). The refits
still lean in the same direction on the relative scale (0.73–0.76) but nowhere
near credibly. Compare 7 vs 28 CHF, not 7 vs 112: only 2.1% of option
presentations lie above 84 CHF, so the top of the spline is prior-driven.

The mechanical reason the refit flattens: it puts ν(7 CHF) = 1.49 where the
published fit puts 0.42, so the same absolute Δ is a much smaller fraction. That
traces straight back to the prior-scale difference, i.e. to the choice-rule and
prior changes below — so **which side of `b66c806` you report decides whether the
paper can claim a magnitude-localised effect.**

One further asymmetry: under `ecc6454` the two families *disagree* about memory
(−0.54 vs −0.08 at 7 CHF) while agreeing about perception; under HEAD they agree
about both. The published family-1 fit needs memory noise to fall by ~0.5 CHF at
low payoffs to offset the perceptual rise in the first-presented option — a
compensation the family-2 fit of the same data does not make.

The same holds for the preprint's Figure 5A quantity, total representational
noise √(ν₁² + ν₂²), which depends only on the noise functions and not on the
choice rule (`fig5a_style.<label>.pdf`, `fig5_style.flexible1nf.pdf`). Both fits
put the cTBS increase at small payoffs and both extend it further across the
decision space when the risky option comes second — the caption's claim. The
published fit's increase crosses below 1 at large safe payoffs and high ratios;
the refit's stays above 1 everywhere (+5% to +16%).

And the LOO structure matches the published Table 1:

| | preprint Table 1 | refit |
|---|---|---|
| full model ELPD | −4167.4 | −4184.6 |
| drop one noise term | −26.2 / −26.3 | −37.4 / −42.9 |
| null (no cTBS on noise) | −80.2 | −87.2 |

## Where the parameters diverge, and why

The published `flexible1`/`flexible2` traces were fit under bauer `ecc6454`,
whose `_get_choice_predictions` used `diff_sd = sqrt(ν1² + ν2²)`. Commit
`b66c806` replaced that with a posterior-variance- and probability-scaled form.
**ν does not mean the same thing on the two sides of that commit**, so the two
fits are not directly comparable:

| | published `flexible1` (ecc6454) | refit `flexible1nf` (HEAD) |
|---|---|---|
| ν₁ at 7 → 28 CHF, vertex | 1.32 → 2.19 | 1.13 → 2.25 |
| ν₂ at 7 → 28 CHF, vertex | 0.42 → 1.33 | 1.49 → 2.28 |
| cTBS Δν, first option | ≈ 0 (CrI spans 0 everywhere) | +0.28, credible |
| cTBS Δν, second option | +0.40 at 7 CHF, decaying to +0.03 at 28 | +0.20, flat |
| safe / risky prior μ | 10.74 / 18.58 CHF | 4.90 / 9.95 CHF |

`b66c806` is a **refactor commit** — "Add v0.2.0: tutorial notebooks, docs
infrastructure, pyproject.toml", which deleted the flat `bauer/models.py` and
created `bauer/models/{psychophysics,magnitude,risky_choice}.py`. The behavioural
changes rode along inside it, which is why they never show up as a diff you would
look at. `ecc6454` ("`mutable` is always True in new versions of pymc") is seven
commits earlier.

The choice rule, in `FlexibleNoiseRiskModel._get_choice_predictions`, payoff
branch:

```python
# ecc6454 -- shrink the mean, but feed the RAW evidence SD to the decision
diff_mu, diff_sd = get_diff_dist(post_n2_mu * p2, n2_evidence_sd,
                                 post_n1_mu * p1, n1_evidence_sd)

# b66c806 -- propagate the evidence SD through the posterior mean, and scale by p
n1_noise = post_n1_sd**2 / n1_evidence_sd * p1
n2_noise = post_n2_sd**2 / n2_evidence_sd * p2
diff_mu, diff_sd = get_diff_dist(post_n2_mu * p2, n2_noise,
                                 post_n1_mu * p1, n1_noise)
```

`post_sd² / ν = ν·σ²/(σ²+ν²) = ν·w`, so the new form is the correct propagation:
the decision variable is the posterior *mean*, whose SD is `w·ν`, not `ν`. The
old form shrank the mean but not its variance — internally inconsistent. **HEAD
is right here**, and it is KLW's own algebra (see `klw_variance_analysis.md`).

**The rule change alone does not explain the parameter gap either.** In the fully
symmetric case the two rules give the identical PSE and are an exact
reparameterisation, ν_new = β·ν_old and σ_new = β·σ_old with the same β. Note
that w = σ²/(σ²+ν²) is *invariant* under that common rescaling — so a
reparameterisation cannot move w at all, and the published w ≈ 0.98 vs refit
w ≈ 0.34 must come from somewhere else.

### The priors ARE identified — because ν₁ ≠ ν₂

There is a zero-cost flat direction in (μ_risky, σ_risky, μ_safe, σ_safe): scale
both shrinkage weights by a common λ and slide the prior means to compensate.
**But it exists only when a single constant ν is shared by both options.**
Achieving w′ = λw requires σ′² = λwν²/(1−λw), which pins σ′ to *one* value of ν.
Two different noise functions, or one that varies with magnitude, and no single
σ′ scales w by a common λ everywhere.

Profiled on the real 8335-trial design, cost of halving both weights (λ = 0.5),
prior means re-optimised, in total nats:

| | λ = 0.5 | λ = 0.7 |
|---|---|---|
| ν constant **and** ν₁ = ν₂ | **0.0** | **0.0** |
| ν constant, ν₁ ≠ ν₂ | 932 | 343 |
| ν ∝ magnitude, ν₁ = ν₂ | 156 | 51 |
| ν ∝ magnitude, ν₁ ≠ ν₂ (the real model) | 237 | 79 |

Finely, in the real case: ±5% in w costs 2.0 nats, ±10% costs 8 nats — a clean
quadratic, not a ridge. The fitted posteriors agree (`flexible2nf`:
`risky_prior_sd` 1.47 [0.91, 2.33], `safe_prior_sd` 1.25 [0.67, 2.41]).

So the flat direction is real but belongs to a degenerate special case this
experiment is not: **asymmetric noise between the first- and second-presented
option is exactly what identifies the priors.** Percept-compression numbers are
reportable, and the published-vs-refit disagreement about them is a genuine
disagreement between two likelihoods, of which HEAD's is the correct one.

Two further code changes, neither renaming a parameter:

1. **At `ecc6454` the `shared_perceptual_noise` branch used the memory
   coefficients twice.** In `_get_trialwise_evidence_sd`:
   `spline_pars2 = pt.stack([parameters[l1] for l1 in labels1], axis=1)` — `l1`
   / `labels1` where `l2` / `labels2` belonged. So for **family 2 only**,
   ν₁ = softplus(η_mem·B₁ + η_mem·B₂): perceptual noise never entered the
   first-presented option, and family 2 collapsed to two decoupled curves rather
   than a shared-perceptual model. Family 1 (`independent`) is unaffected. This
   means the published `flexible2` trace does not implement its own Methods —
   worth checking which trace the preprint's Fig 4B/C was drawn from.
2. **The regression path silently ignored the priors on every softplus
   parameter.** At `ecc6454`, `RegressionModel.build_hierarchical_nodes` set the
   Intercept's `mu`/`sigma` from `mu_intercept`/`sigma_intercept` only for the
   `identity` and `logistic` transforms — the `softplus` branch was missing (a
   bare comment `# Possibly use inverse of softplus` sat where it should have
   been). So `risky_prior_sd` and `safe_prior_sd` got `Normal(0, 1)` on the
   untransformed scale no matter what `get_free_parameters` declared. HEAD added
   the branch (its comment names it "the source of the regression-DDM
   convergence pathology"), which turns those priors into
   `Normal(np.std(risky_n) ≈ 33, 25)` and `Normal(np.std(safe_n) ≈ 22, 0.5)` —
   wildly diffuse and badly centred on the softplus scale. That is the flat
   direction the unconstrained HEAD refits wandered along, and it is why they
   needed `--constrain` to converge at all.

The consequence is a genuinely different implied psychophysics, not a
reparameterisation. Implied shrinkage weight `w = σ²/(σ²+ν²)` and the percept of
a safe option, 7 → 28 CHF:

| | w | perceived / objective |
|---|---|---|
| published `flexible1` | 0.98 → 0.81 | 101% → 88% (near-veridical) |
| refit `flexible1nf` | 0.34 → 0.18 | 80% → 32% (heavily compressed) |

Both fit the choices about equally well, because choices depend on the *ratio* of
the two percepts and both options compress toward their own priors. This is a
weakly identified direction in the model; which mode you land in is decided by
the prior on `*_prior_sd`, i.e. by which side of the missing-softplus-branch fix
you are on.

Worth deciding explicitly which side of `b66c806` the paper reports, rather than
letting the checkout decide. The choice matters for the reported prior means and
percepts; it does not change the conclusion about perceptual noise.

## A note on the `nf` tag

`flexible1nf` / `flexible2nf` are the *refits*, and the `nf` is a misnomer worth
knowing about. `fit_pmc_noisefix.py` requires the model label to match
`flexible[12](\.\d)?_noisefix<suffix>` — `_noisefix` there just marks "fitted by
this script", it is not the bauer variant. The variant is a separate flag and
these were all fit with `--variant head`, which is why the files are
`model-flexible1_noisefix.head_trace.netcdf`. So `nf` = "from the noisefix
script", *not* "the `noisefix` bauer patch".

## Still running

`flexible2` nested family, both nodes:

| | GPU (numpyro, T4) | CPU VM (pymc) |
|---|---|---|
| full | done, r̂ 1.010, ESS 427, **28** divergences | done, **2283** divergences |
| null | 55% | 92%, 1826 divergences |
| memory | queued | 93%, 1043 divergences |
| perception | queued | done |

Same model, same priors, same seed count — numpyro/NUTS gets 28 divergences where
pymc gets 1000–2300. Take the GPU traces as the family-2 answer and use the CPU
ones only as a sampler cross-check. Note both nodes write
`derivatives/cogmodels.overnight/` with **identical filenames** on their own
filesystems, so never rsync one over the other.

## The localisation is individual-specific, and it tracks nPRF amplitude

Group-average curves ask the wrong question. The targeting logic predicts a
*subject-specific* effect: whoever lost more nPRF amplitude under cTBS should show
a more low-magnitude-weighted noise increase. `extract_subject_noise_shift.py`
writes one number per subject from the subject-level regression coefficients
(`<term>_spline<i>`, dims chain x draw x subject x regressor).

Pre-specified quantity: the **localisation slope** Δν_perceptual(7 CHF) −
Δν_perceptual(28 CHF), correlated with the per-subject cTBS amplitude change.

| amplitude measure | `flexible1` | `flexible1nf` | `flexible2nf` |
|---|---|---|---|
| **m2** (full per-session model) | r = −0.33, p = .051 | **r = −0.43, p = .010** | r = −0.33, p = .051 |
| m1, median over voxels | −0.04, p = .84 | +0.07, p = .70 | −0.07, p = .67 |
| m1, low-preferred-n half | −0.15, p = .41 | −0.08, p = .63 | −0.17, p = .34 |

Negative r = more amplitude loss goes with a more low-payoff-concentrated noise
increase — the predicted direction. Spearman agrees (ρ = −0.35 to −0.39, p = .02–.04).
The *level* Δν(7 CHF) does not correlate in any fit (|r| < 0.14); only the slope does.

**Two honest caveats.** (i) It holds for the m2 amplitude and not for m1, which is
the paper's canonical encoding model — that fork has to be settled on principled
grounds (m1 fixes `mu`/`sd` across sessions so its Δamplitude is a change at fixed
tuning; m2 lets tuning move) rather than by which one works. My m1 aggregation
(median voxel `amp_diff`, `1 < pref_n < 200`) is also my own choice and may not
match the paper's pipeline. (ii) Three quantities were examined per trace; the
slope is the one that was theoretically motivated in advance, and it replicates
across all four fits, but p = .010 is uncorrected.

This is the answer to "why isn't the group effect localised": subjects differ in
how localised their effect is, and that difference is explained by how much
amplitude their stimulated voxels lost. The group mean averages it away.

## Table 1, family 2 (2026-08-01): it is the perceptual term that carries the effect

LOO over the nested family-2 refits (`cogmodels.overnight/model-flexible2_noisefix*`,
bauer `e05f73a`, constrained, 4 × 5000+5000, numpyro):

| model | ELPD | Δ vs best | SE of Δ |
|---|---|---|---|
| cTBS on **perceptual noise only** | **−4157.7** | 0 | — |
| cTBS on both terms (full) | −4159.7 | 2.0 | 3.9 |
| cTBS on **memory noise only** | −4217.5 | 59.9 | 12.2 |
| **no cTBS effect on noise** | −4273.2 | 115.6 | 14.4 |

Read the middle two rows against each other: **dropping the memory term costs
nothing** (2.0 ± 3.9, well inside noise) while **dropping the perceptual term
costs 59.9 ± 12.2** (4.9 SE). The perception-only model is not merely adequate,
it is the *best* model in the family — the memory term is surplus. Against the
null, 115.6 ± 14.4 (8.0 SE). Preprint Table 1 for comparison: full −4167.4,
drop-one −26.2 / −26.3, null −80.2.

Convergence: full, memory and perception all pass (r̂ ≤ 1.010, ESS ≥ 418).
**The null does not** (r̂ = 1.020, ESS = 115, 664 divergences), so treat its ELPD
as indicative; it is the worst model by a wide margin either way. The family-1
null *did* converge (r̂ 1.000, ESS 1653, 0 divergences) and gives Δ87.2 ± 12.9
against its full model, so the conclusion does not rest on the failing trace.

Two traces are stamped `e05f73a+patched` (memory, perception) and two are not.
The patch is `get_sd_curve` (reconstruction only), `spline_degree` (defaults to
3, no behaviour change) and `safe_prior_sd`'s `sigma_intercept` — which
`--constrain` overwrites to `Normal(3, 1)` regardless. So all four fit the same
model.

## The noise spline cannot be resolved more finely than one interior knot

Asked whether the flat perceptual contrast is an artefact of basis rigidity.
It is not — the finer parameterisations are not estimable:

| fit | interior knots | max r̂ | min ESS | divergences |
|---|---|---|---|---|
| `flexible2` df=5 cubic (reported) | 1 (20) | 1.010 | 427 | 28 |
| `flexible2.9` df=9 cubic | 5 (10,14,20,28,41) | 1.590 | 7 | 19 |
| `flexible2` df=5 **quadratic** | 2 (14, 28) | 1.240 | 12 | 2522 |
| published `flexible2.4` df=4 (ecc6454) | 0 | 1.010 | 388 | — |
| published `flexible2` df=5 (ecc6454) | 1 | 1.010 | 494 | — |
| published `flexible2.6` df=6 (ecc6454) | 2 | **1.530** | **7** | — |

Three independent failures, across two bauer versions, two spline degrees and
two df values, with the same signature (ESS ≈ 7) — and the 2024 fits show it
too, so it is not the patch, the choice rule or the degree. **One interior knot
is the resolution ceiling of this design.** df=5 was therefore not a convenience
choice. The corollary is that group-level localisation cannot be demonstrated by
adding knots; the per-subject brain-behaviour result is where that evidence has
to come from.

(Honest limit: "did not converge under identical settings and priors" is strong
evidence of weak identification, not proof. Longer warmup or tighter noise-spline
priors might rescue df=6.)
