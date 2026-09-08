# Plan to submission — v12

Written 2026-09-08. Ordered by what blocks what, not by size.

## The one open scientific decision

**Which model the paper reports.** Under the KLW-consistent choice rule
(now the only rule, see CLAUDE.md) the candidates are:

| Model | Δν @ 7 CHF | P(Δν > 0) | converged | rank stability of the reported ν |
|---|---|---|---|---|
| `log-power-n1n2.klw` | +0.047 | 0.991 | **no** (r̂ 1.12, ESS 42) | 0.20 (ν₂@7) |
| `log-power-perc.klw` | +0.031 | 0.971 | yes | **0.50** (ν_perc@7) |
| `log-power-percmem.klw` | +0.030 | 0.964 | yes | 0.48 (ν_perc@7) |
| `log-spl5-n1n2.klw` | +0.025 | 0.819 | yes | — |
| `log-power-mem.klw` | +0.019 | 0.891 | yes | — |

Eleven arms are currently trying to make `n1n2.klw` mix (below). The decision
rule: **if any arm passes r̂ ≤ 1.01 / ESS ≥ 400, report `n1n2.klw`; otherwise
report `perc.klw`** and give `n1n2.klw` no space beyond a line in the
supplement saying the position-indexed variant does not sample.

`perc.klw` is not a fallback to apologise for. It is the more parsimonious
claim (one channel, not two), it is the only candidate whose per-participant
estimates are reliable enough to correlate with anything, and its noise
channel is the perceptual one — which is what the nPRF result is about.

### The eleven arms (all `log-power-n1n2`, all KLW)

| Job | Arms | Lever |
|---|---|---|
| 5655915 | 4 | tighter group SDs: τ 0.15/0.10 × target_accept 0.97/0.99, one also σ_prior_mu 0.4 |
| 5655993 | 3 | `--prior_estimate fix_prior_sd` (8 free params → 6), also for `perc` and `nullind` |
| 5656026 | 4 | soft-pinned prior width: `--sigma_prior_sd` 0.15/0.25, one with τ 0.15, one with σ_prior_mu 0.4 |

All three attack the same ridge — the shrinkage weight is
sd²/(sd²+ν²), so a free prior width trades off directly against the noise
anchors — from three directions: shrink the group SD, remove the parameter,
or narrow its prior. Whichever wins, the setting is stamped into the trace and
the filename.

Check with `summarize_traces.py` on `cogmodels.anchor`; ~6–14 h from 09:50.

### It is the whole position-indexed family, not just n1n2

Convergence of the KLW `log-power-*` set (25/40 pass):

| Model | r̂ | ESS | |
|---|---|---|---|
| `nullind` | 1.000 | 1419 | passes |
| `n1` | 1.020 | 247 | fails |
| `n2` | 1.050 | 109 | fails |
| `n1n2` | 1.120 | 42 | fails |
| `perc` / `mem` / `percmem` / `null` | 1.000 | 1765–5179 | all pass |
| **`n1n2x`** (cTBS × order) | **1.010** | **956** | **passes** |

`nullind` mixes and every model with a cTBS regressor on a position-indexed
channel does not, so it is the regressor-on-a-position that breaks the sampler.
Fixing `n1n2` alone would therefore still leave the nested ladder without its
middle rungs — **job 5656293** refits `n1`, `n2` (and `nullind`, `perc` for a
like-for-like prior spec) under both structural levers, so the ladder is ready
the moment a winner is known.

### What `n1n2x` actually says (checked, not assumed)

`n1n2x` is a strict SUPERSET of `n1n2`: identical everywhere except that the
four noise anchors carry `stimulation_condition*risky_first` (4 regressor
columns) instead of `stimulation_condition` (2). The bigger model samples
(r̂ 1.010 / ESS 956); the smaller one does not (1.120 / 42).

The tempting story — that a position-indexed channel has to compromise across
orders and the interaction relieves that tension — is **wrong**, and the trace
says so. Group-level cTBS contrasts (log units, IPS − vertex):

| | risky SECOND | risky FIRST | difference |
|---|---|---|---|
| ν₂ @ 7 | **+0.321 [+0.060, +0.574]**, P = 0.992 | +0.289 [−0.058, +0.682], P = 0.947 | +0.028 [−0.340, +0.378], P = 0.56 |
| ν₁ @ 7 | +0.023 | −0.022 | +0.044, P = 0.64 |
| ν₂ @ 112 | −0.086 | +0.109 | −0.195, P = 0.18 |
| ν₁ @ 112 | +0.136 | −0.035 | +0.172, P = 0.86 |

The interaction is null on every anchor, and so is the `risky_first` main
effect on baseline noise (−0.02 to −0.21, every CrI crossing 0). Given the
freedom to make the cTBS effect order-specific, the model declines to use it.
**Why the larger model mixes and the smaller one does not is unexplained** —
it is not the init (`n1n2.pathfinder.klw` is r̂ 1.15).

Two things this is genuinely worth:

1. **It corroborates the headline from a model that converges.** `n1n2x` puts
   Δν₂ @ 7 CHF at +0.321 [+0.060, +0.574], P = 0.992; the non-converged `n1n2`
   puts it at +0.315 [+0.052, +0.564], P = 0.991. Near-identical. So the effect
   is not an artefact of `n1n2`'s bad sampling — which is the main worry about
   reporting it.
2. **It is evidence against an order-specific noise mechanism**, which is worth
   one sentence given how much the model-free result leans on risky-second
   trials. The behavioural asymmetry is not because cTBS raises noise more when
   the risky option comes second; it is because raising the noise on whichever
   option came second has an asymmetric behavioural consequence.

LOO for the whole `*x` family is extracting (job 5656327) so the ladder can
include it.

## Blocked on that decision

1. **Figure 5** — rerun `plot_fig4_big.py` for the chosen label.
2. **ELPD ladder + Supp Table 1** — `plot_elpd_ladder.py`, `make_supp_table1.py`.
   Both now read a KLW-only `notes/data/loo_anchor/`, so they no longer need a
   filter; they DO need every KLW trace to have a LOO TSV (several are missing).
3. **Supp PPC gallery / psychometric** — same, KLW-only.
4. **Results text** — the Δν numbers, the credible range in payoff, the
   one-sided P(Δν > 0).

## Not blocked — do now

5. **Methods, spline count.** v11 says "spline 3 to 9 free parameters". Only
   spl3, spl5 and spl7 were fitted, and the basis is piecewise-linear over the
   anchors, not a B-spline. Fix both.
6. **Methods, sampler.** Stamp the reported model's actual settings
   (`trace.posterior.attrs`), not the generic ones.
7. **Methods, choice equation.** See TODO 1 below — the text now matches the
   code, but only after the switch; say the decision SD is
   sqrt((w₁ν₁)² + (w₂ν₂)²) with w = σ_p²/(σ_p²+ν²).
8. **Correlation paragraph.** The attenuation ceiling is 0.71 for the
   perceptual channel at 7 CHF, not 0.42. Rewrite (below).
9. **n = 73 vs 75/78.** Already located; put the sentence in Methods §
   Participants.

## The three TODOs from the writing chat

**TODO 1 — the choice equation.** Confirmed against the code, and it has now
changed. `bauer/core.py:169-197`: the numerator is always the difference of
POSTERIOR MEANS. The denominator used to be the raw evidence SDs
(`sqrt(ν₁²+ν₂²)`), which is the inconsistency the chat spotted. Since
2026-09-08 the default is `consistent_choice_noise = True`, giving

    sd_k = w_k · ν_k,   w_k = σ_p,k² / (σ_p,k² + ν_k²)      bayes.py:26-47
    diff_sd = sqrt(sd₁² + sd₂²)                              core.py:190-194

so numerator and denominator are shrunk by the same w. **The code changed, so
the text should state the consistent form.** Note this is `w`, the Bayesian
shrinkage weight — not a `β_k`; there is no separate scaling coefficient in
this model.

**TODO 2 — the payoff means.** Verified on the 8335 trials / 35 participants
that enter the reported fits (`get_data(..., 'lfx2-bs3-m2-dp-bm')`):

| | mean | median | range |
|---|---|---|---|
| Risky payoff | **36.15** CHF | 30 | 7–112 |
| Safe payoff | **15.82** CHF | 14 | 7–28 |

v11's 36.2 / 15.8 still hold. (The model's priors are centred on the mean of
*log* payoff, which is 30.2 and 14.1 CHF — different numbers, different
quantity; don't let them cross over into the same sentence.)

**TODO 3 — the reliability bound is now committed.**
`behavior/scripts/anchor_subject_reliability.py --trace_dir ...` writes
`notes/data/subject_reliability.<label>.tsv`. The σ_group = 0.14 /
σ_within = 0.33 in the draft came from the raw-rule `n1n2` fit and are
superseded. Under KLW:

| Model · parameter | σ_within | σ_between | rank stability | ceiling |
|---|---|---|---|---|
| `perc` · ν_perc @ 7 | 0.29 | 0.39 | **0.50** | **0.71** |
| `perc` · ν_perc @ 112 | 0.26 | 0.45 | 0.57 | 0.76 |
| `percmem` · ν_mem @ 7 | 0.44 | 0.38 | 0.13 | 0.36 |
| `n1n2` · ν₂ @ 7 | 0.35 | 0.35 | 0.20 | 0.44 |
| `n1n2` · ν₁ @ 112 | 0.29 | 0.42 | 0.53 | 0.73 |

The script reports three estimators and the variance-subtraction one is
biased — see the CLAUDE.md convention. The headline: **the perceptual channel
is reliable enough to correlate (ceiling 0.71); the memory channel is not
(0.36).** That is itself a result worth one sentence.

## Stop: the r = 0.31 has the wrong sign

The writing chat wants the "nominally largest correlation, r = 0.31 on ν at
7 CHF" reported. It should not be. Both quantities are IPS − vertex:

* `d_amp` = amplitude(IPS) − amplitude(vertex), so **negative = amplitude lost**
  (`modeling/scripts/extract_brain_behavior_table.py:77`)
* Δν = IPS − vertex on log noise, so **positive = noise increased**
  (`behavior/scripts/extract_anchor_subject_params.py:44`)

The hypothesis therefore predicts **r < 0**. The model-free result obeys it
(Δamp × Δconsistency on risky-second trials = +0.53: lose amplitude, lose
consistency). The model parameters do not:

| Pair | r | sign |
|---|---|---|
| Δamp × Δconsistency (risky second) | +0.53 | as predicted |
| Δν_perc @ 112 × Δconsistency | −0.35 | as predicted |
| Δν_perc @ 112 × Δamp | −0.19 | as predicted, weak |
| Δν_perc @ 7 × Δconsistency | −0.01 | nothing |
| **Δν_perc @ 7 × Δamp** | **+0.31** | **opposite** |

And it is opposite in every mask, including the controls that the model-free
analysis showed the effect falls off in: left parietal +0.27, occipito-temporal
+0.26, frontal +0.11 — flat with distance from the coil, which is what a
non-specific artefact looks like, not a causal link.

**Report nothing from the model-parameter correlations.** One sentence: the
model-free consistency link (Fig. 3) does not reappear in the model's
per-participant noise parameters, and the per-participant reliability of those
parameters — ceiling 0.71 for the perceptual channel — is high enough that low
reliability is not the explanation. Better to say that than to report a
wrong-signed r at p = 0.07.

`behavior/scripts/anchor_brain_behavior_posterior.py` computes these the right
way: the correlation is recomputed **per posterior draw**, so the interval
contains the per-participant measurement error and no bootstrap is involved
(the draft's "bootstrap 95% CI [0.37, 0.67]" on the model-free correlation is
the last maximum-likelihood interval left in the paper — replace it).
