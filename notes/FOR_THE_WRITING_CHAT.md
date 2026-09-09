# Instructions for the writing chat — v11 → v12

Everything below is measured, with the script that produced it named. Where a
number in v11 is superseded, the old value is given so you can find it in the
draft.

---

## 1. The single biggest change: the choice rule

**v11's model comparison is not internally comparable and has been refitted.**

bauer's historical decision rule compared two posterior means — each already
shrunk toward the prior by `w = σ_p²/(σ_p² + ν²)` — but normalised by the *raw*
evidence SDs. The prior width therefore changed the psychometric slope as a pure
artefact of the normalisation, so ν did not denote the same quantity in two
models with different priors.

Every model is now fitted with the consistent rule:

> ν̂_k = w_k · ν_k with w_k = σ_p,k² / (σ_p,k² + ν_k²), and the two options are
> compared with decision SD √(ν̂₁² + ν̂₂²).

**Methods, TODO 1:** write this form. It is `w`, the Bayesian shrinkage weight —
there is no separate scaling coefficient `β_k`. Code: `bauer/core.py:169-197`,
`bauer/utils/bayes.py:26-47`.

## 2. The reported model changed: `perc`, not `n1n2`

v11 reported a model with the cTBS effect on the first- and second-*presented*
options (`n1n2`). **Under the consistent rule that model does not converge**, and
cannot be made to: r̂ 1.12 / ESS 42, against a gate of r̂ ≤ 1.01 / ESS ≥ 400,
across seven different remedies (tighter group SDs, three prior widths, pinned
prior SDs, and combinations).

The reported model is now **`log-power-perc`** — one *perceptual* noise channel
carrying the cTBS effect, plus a memory channel shared across conditions.
r̂ 1.000, ESS 5179.

Three independent reasons this is not a retreat:

* **ELPD cannot tell the candidates apart.** `n1n2x`, `percmemx`, `percx`,
  `percmem` and `perc` all sit within 1.2 dSE of each other, so the choice falls
  to convergence, parsimony (6 parameters) and reliability, all of which favour
  `perc`.
* **A converged superset agrees.** `n1n2x` (same model plus a cTBS × order
  interaction) does converge and puts the effect in the same place.
* **A converged flexible version agrees, more weakly.** `spl5-n1n2` (five
  anchors) converges and gives P(Δν > 0) = 0.82.

## 3. Numbers to use

`log-power-perc.mapjitter.klw`, group level, IPS − vertex:

| Quantity | Value |
|---|---|
| Δν, perceptual channel @ 7 CHF | **+20%**, P(Δν > 0) = **0.971** (one-sided p = 0.029) |
| Δν, perceptual channel @ 112 CHF | −8%, P(Δν > 0) = 0.09 (not credible either way) |
| Credible range | the effect is credible up to ≈ 14 CHF and gone by 56 |
| Memory channel | shared across conditions in this model |

**Trap:** `subject_params.*.tsv`'s `GROUP` row is the mean over participants of
the per-participant contrast, which is tighter than the group-level parameter
(P = 0.997 vs 0.971). **Report the group-level parameter** — that is what the
figure's panel g shows.

Model comparison, paired dELPD against the best converged model:

| Claim | Evidence |
|---|---|
| cTBS moves the noise function at all | `nullind` is **7.8 dSE** worse |
| It is not the memory channel alone | `mem` is **4.6 dSE** worse |
| Which channel carries it is not resolvable | top five models within **1.2 dSE** |

## 4. Delete the model-parameter brain–behaviour correlation

v11 reports (or was going to report) a correlation between the cTBS change in
nPRF amplitude and the model's per-participant noise contrast — "r = 0.31,
p = 0.073 on ν₁ at 7 CHF".

**It has the wrong sign.** Both measures are IPS − vertex, so `d_amp` negative
means amplitude *lost* and Δν positive means noise *gained*; the hypothesis
predicts r < 0. The model-free result obeys it (Δamp × Δconsistency on
risky-second trials = +0.53). Δν at 7 CHF does not, and it is equally wrong-way
in every control mask — left parietal +0.27, occipito-temporal +0.26, frontal
+0.11 — i.e. flat with distance from the coil, the signature of an artefact.

Low reliability is *not* the excuse: the attenuation ceiling for that parameter
is 0.71 (see §5).

**Replace with one sentence:** the model-free link between amplitude loss and
consistency loss does not reappear in the model's per-participant noise
parameters. Keep the model-free result (Fig. 3), which is unaffected.

**Also in that paragraph:** "bootstrap 95% CI [0.37, 0.67]" is the last
maximum-likelihood interval in the paper and violates the project's own rule.
`behavior/scripts/anchor_brain_behavior_posterior.py` recomputes the correlation
**once per posterior draw**, so its interval already contains the
per-participant measurement error. Use that.

## 5. Reliability (TODO 3) — the bound in the draft is superseded

σ_group = 0.14 / σ_within = 0.33 came from the old raw-rule fit. Under the
consistent rule, and with the estimator that is actually correct:

| Parameter | rank stability | attenuation ceiling |
|---|---|---|
| perceptual ν @ 7 CHF | 0.50 | **0.71** |
| perceptual ν @ 112 CHF | 0.57 | 0.76 |
| memory ν @ 7 CHF | 0.13 | 0.36 |

The classical variance correction — subtracting within-participant variance from
the spread of the posterior means — **double-counts shrinkage** and returns 0.00
for most of these (it returns 0.00 on a synthetic case whose true reliability is
0.5). The correlation between two independent posterior draws of the whole
participant vector cannot go negative and estimates reliability directly.
Script: `behavior/scripts/anchor_subject_reliability.py`.

Worth one sentence: **the perceptual channel is reliable enough per participant
to correlate with an external measure; the memory channel is not.**

## 6. TODO 2 — the payoff means still hold

On the 8335 trials / 35 participants in the reported fits: risky mean
**36.15 CHF**, safe **15.82 CHF**. v11's 36.2 / 15.8 are correct.

Do **not** mix these with the priors' centres, which are means of *log* payoff
(30.2 and 14.1 CHF) — different quantity, same sentence is a trap.

## 7. Participants — v11 has an arithmetic slip

v11: *"Of the 35 selected for the follow-up sessions, the same two outliers were
excluded, leaving 35 analysed."* Thirty-seven were selected.

> All analyses of the baseline session therefore rest on 75 participants, reduced
> to 73 after excluding two behavioural outliers (see Exclusion criteria).
> Thirty-seven of these were selected for the follow-up sessions; the same two
> outliers fall in this group, leaving 35 analysed.

Exclusion criteria — the exclusion is **behavioural, not technical**:

> Two participants were excluded on the basis of their choice behaviour alone.
> Across all sessions they chose the risky option on 11.7% and 13.7% of trials
> (z = −2.79 and −2.64 relative to the group), the only two beyond 2.5 SD and
> well separated from the next participant (31.4%, z = −1.40). Both were
> consistently risk-avoidant in every session, including the pre-stimulation
> baseline, so the exclusion is unrelated to stimulation; with so few risky
> choices their psychometric functions do not constrain an indifference point.
> No participant was excluded for technical reasons relating to the TMS.

Sources: `tms_risk/data/all_subjects.yml` (75), `tms_keys.yml` (37),
`utils/data.py:36`, `behavior/notebooks/archive/outliers.ipynb`.

## 8. Two other Methods corrections

* v11 says the spline noise function had "3 to 9 free parameters". Only 3, 5 and
  7 were fitted, and the basis is **piecewise-linear through anchor payoffs**,
  not a B-spline. The parameters are the noise SD's own values at those payoffs.
* The sampler settings quoted should be the reported model's actual stamp
  (`trace.posterior.attrs`), not generic ones.

## 9. Prior shifts stay out — and now there is a figure saying why

v11 dropped models in which cTBS shifts the magnitude prior. That decision is
right, and `percpsd` (perceptual noise **plus** both prior SDs free to shift)
shows why: **neither** the noise effect **nor** the prior shift is credible
(p = 0.39 risky, p = 0.22 safe). Because the shrinkage weight is
σ_p²/(σ_p² + ν²), widening the prior and lowering the noise move the same
quantity — it is an identifiability failure, not a competing explanation.
Supplementary figure S6.

## 10. Model fit — what to say, and what not to

The posterior predictive panels look scattered. They are not misfitting; they
are plotted at a resolution where sampling noise dominates. Audited:

* Median |residual| divided by the observed point's **own standard error** is
  **0.97** (`ppc_anchor.rung`) and **0.73** (`stake3rung`). A ratio of 1.0 means
  the residuals are entirely explained by sampling noise in the observed
  proportions. Each point is a subject-averaged proportion over 12-20 trials per
  participant.
* At full ladder resolution (48 rungs, ranking every trial within participant x
  safe payoff) the residual trend is **+0.0005 per rung**, and the same computed
  WITHIN participant and then averaged, so it is not an aggregation artefact.
* Per participant: indifference point SD **0.312 observed vs 0.319 model**
  (the model spreads participants correctly); psychometric slope **0.810
  observed vs 0.756 model**, r = **0.92**.
* Cell-level coverage of the 95% predictive band: **95%**, with r = 0.92 between
  observed and predicted choice proportions across 420 participant-cells.

**One sentence for limitations:** the model's psychometric function is about 7%
too flat (slope 0.756 against 0.810), which is why predictive coverage runs
79-88% rather than 95% on the ladder-rung tables.

**One sentence that must be in Results, not buried:** the model reproduces the
PATTERN of the cTBS effect but underpredicts its SIZE. On the targeted
posterior predictive check for the mean cTBS effect on risky-second trials the
observed value is +0.053 against a predictive interval of [-0.014, +0.044],
posterior predictive p = 0.005 -- roughly a four-fold underprediction. Six of
the seven targeted statistics are covered, including the order contrast
(p = 0.09) and the three-way stake x order x stimulation interaction (p = 0.51).
Do not claim the model reproduces the magnitude of the behavioural effect.

## 11. What is still open — read this before writing §2, §3 or Figure 5

**Which model is reported.** `log-power-n1n2` is the scientifically right model
— it is the only one that reproduces the order asymmetry the paper is about —
and under the consistent choice rule it does not yet sample. Status:

| Route | r̂ / ESS | verdict |
|---|---|---|
| default prior | 1.120 / 42 | fails |
| level/slope + sum-to-zero coding | **1.040 / 142** | fails; best legitimate result |
| `--tau_intercept 0.10` | 1.010 / 1214 | **converges but disqualified** — it shrinks the magnitude prior's between-subject SD by 56%, against the code's own note that it must not be shrunk, and every predictive metric degrades |
| `--tau_noise` (noise anchors only) | running | the principled attempt |

Gate is r̂ ≤ 1.01 and ESS ≥ 400. The reparameterisations lifted ESS 3.4-fold
without touching a single prior, which confirms the problem is geometry rather
than the model — but 142 is still short.

**So:** everything in §1 and §4–§10 is settled and can be written now. §2 (which
model), §3 (the Δν numbers) and Figure 5's numbers are provisional. If
`--tau_noise` converges they stand as written with new values; if it does not,
the choice is between reporting `n1n2` with its diagnostics stated openly and
falling back to `log-power-perc`, which converges cleanly but cannot express the
order asymmetry (§2).

---

# Section-by-section revision plan for TMS_paper_v11

Line numbers are from `notes/paper/TMS_paper_v11.txt`. Work top to bottom; only
the sections listed need touching.

## Abstract (l. 14)

One change. Whatever it currently claims about the cognitive model, the claim
that survives is **where in payoff space the noise moves**, not the size of the
behavioural effect the model reproduces. If the abstract says the model
"explains" or "accounts for" the choice effect, weaken to "localises": cTBS
increased representational noise on the second-presented option at small
payoffs. See §10 — the model underpredicts the effect size four-fold.

## Results § Experimental approach (l. 242)

No change.

## Results § the stake-size / choice section (from l. 634)

* The model-comparison paragraph gets the new ELPD numbers (§3 and
  `notes/supp_table1.md`). Two claims are decisive and should be stated as
  such: cTBS moves the noise function at all (`nullind` 7.8 dSE worse) and it
  is not the memory channel alone (`mem` 4.6 dSE worse). The claim that is NOT
  supported is which channel carries it — the top five models are within
  1.2 dSE, so say the comparison does not resolve it and that the reported
  model is chosen on convergence and parsimony.
* Delete any sentence implying the model set was compared under one choice rule
  in v11; it was not (§1). The refit is the reason the numbers changed.
* Add the sensitivity sentence for the reported model's prior (§11).

## Results § Linking Neural and Behavioral cTBS Effects (l. 788–822)

The heaviest edit in the paper.

* **Keep** the model-free result: Δ nPRF amplitude × Δ choice consistency on
  risky-second trials, and the site-specificity that follows it. Unaffected.
* **Replace** `bootstrap 95% CI [0.37, 0.67]` (l. 810) with the per-draw
  posterior interval — §4. It is the last maximum-likelihood interval in the
  manuscript.
* **Delete** the model-parameter correlation entirely and replace with one
  sentence: the model-free link does not reappear in the model's
  per-participant noise parameters. Do **not** report r = 0.31; it has the
  wrong sign and is not site-specific (§4).
* **Add** one sentence on why that is not a reliability failure: the perceptual
  channel's attenuation ceiling is 0.71, the memory channel's 0.36 (§5).

## Discussion (l. 823)

* The limitation sentence about the psychometric function being ~7% too flat
  (§10).
* The sentence about underpredicting the effect size (§10). Better volunteered
  than found by a reviewer.
* If the Discussion currently leans on individual differences in the model
  parameters, cut that — §4 and the per-participant PPC say the model's
  predicted contrasts are compressed (SD 0.09 against 0.20 observed).

## Methods § Participants (l. 1048)

The 75 / 73 / 37 / 35 paragraph and the exclusion criteria, both written out in
full in §7. v11 has an arithmetic slip here.

## Methods § Cognitive computational modeling (l. 1227)

* The choice equation in its consistent form — §1. This is TODO 1 and the code
  changed, so the equation in the text must change with it.
* The payoff means stay as they are (§6), but check they are not in the same
  sentence as the priors' centres.

## Methods § The flexible PMC model (l. 1374)

* "spline 3 to 9 free parameters" → only 3, 5 and 7 were fitted.
* The basis is **piecewise-linear through anchor payoffs**, not a B-spline, and
  the free parameters are the noise SD's own values at those payoffs. §8.

## Methods § Model estimation (l. 1423)

* The sampler settings must be the reported model's actual stamp from
  `trace.posterior.attrs`, not generic ones.
* State the convergence criterion (r̂ ≤ 1.01 and ESS ≥ 400 on group-level
  parameters) and that models failing it are excluded from the comparison and
  listed in Supp Table 1.
* State the prior on the between-subject SDs and the sensitivity (§11).

## Methods § Different priors (l. 1330)

Add the one-sentence justification for excluding prior-shift models, with the
supplementary figure — §9. Currently the exclusion is asserted; now there is
evidence for it.

## Figure and table captions

* Figure 5: name the estimand. Panels c and g are population-level; e and f are
  the posterior of the mean over the 35 sampled participants, which is
  narrower. A reader comparing them will otherwise see a contradiction.
* Figure 5 h/i: "slope of a linear-probability fit within each cell,
  participants pooled; the same statistic is applied to the observed and to
  each posterior draw's simulated choices, so absolute values are attenuated
  equally on both sides."
* Supp Table 1: the caption already states the convergence criterion and lists
  the excluded models. Keep that — it is the part a reviewer will look for.

## Do not touch

Figures 1–4 and their text. The nPRF, decoding and model-free psychophysics
results are unchanged by any of this.

---

# How we "know" it is perceptual — what was actually tested

This is the paragraph most likely to be over-claimed, so here is exactly what
the evidence supports. All ΔELPD are PAIRED against the reported model
(`log-power-perc`), with the standard error of the paired difference.

## Tested, and decisive

| Claim | Evidence |
|---|---|
| cTBS moves the noise function at all | no-cTBS model **−95.5 ± 14.1** (6.8 SE) |
| It is **not** the memory stage | memory-only **−39.8 ± 9.5** (4.2 SE) |
| … and directly: when both channels are free, only the perceptual one moves | in `percmem`, perceptual **+20%, P = 0.992**; memory **+2%, P = 0.55** |
| It is not confined to one presentation position | first-presented only **−19.2 ± 10.2**; second-presented only **−22.1 ± 10.5** |

The memory dissociation is the strongest part of the claim and it is supported
two independent ways: a model comparison and a null posterior on the memory
term inside the model that contains both.

## NOT resolved — do not claim it

| Comparison | ΔELPD |
|---|---|
| perceptual vs perceptual + memory | +1.3 ± 1.5 |
| perceptual vs both options' noise free | +4.9 ± 8.7 |

The perceptual model and the free-both-options model are predictively
indistinguishable. They are also nearly the same claim: with
ν₁ = perceptual + memory and ν₂ = perceptual, raising the perceptual channel
raises BOTH options and raises the second-presented one proportionally more.
The perceptual placement is the *constrained* version that the data do not
reject, not a winner over the free one.

## Suggested wording

> To ask at which stage the stimulation acted, we compared models placing the
> cTBS effect on the perceptual encoding shared by both options, on the memory
> trace that only the first-presented option must carry, or on both. The
> perceptual placement was strongly preferred over the memory placement
> (ΔELPD = 39.8, dSE 9.5), and when both channels were free to change, only the
> perceptual one did (+20%, P(Δν > 0) = 0.992; memory +2%, P = 0.55).
> Restricting the effect to a single presentation position fit worse in either
> direction (first-presented ΔELPD = 19.2; second-presented 22.1). The
> comparison does not distinguish the perceptual placement from one in which
> both options' noise is free to change independently (ΔELPD = 4.9, dSE 8.7);
> we report the perceptual model as the more constrained account the data do
> not reject, and note that because σ₁ = σ_perceptual + σ_memory while
> σ₂ = σ_perceptual, a change in perceptual noise already predicts a larger
> proportional increase for the second-presented option.

## The order asymmetry: `n1n2` gets a third of it, `perc` gets none

**On the choice-proportion scale** — the original PPC, and the one a reader
looks at — averaged over safe payoffs, in percentage points:

| | risky first | risky second | asymmetry |
|---|---|---|---|
| **Observed** | +0.55 | +5.29 | **+4.74** |
| `n1n2` (default prior) | −0.46 | +1.41 | **+1.88** (40%) |
| `n1n2` (τ_noise 0.10) | −0.25 | +1.31 | +1.55 (33%) |
| `perc` | +0.37 | +0.62 | **+0.25** (5%) |

So `n1n2` **does** reproduce the asymmetry qualitatively: essentially nothing
when the risky option comes first, a positive effect when it comes second. It
is about three times too small, not absent. `perc` produces almost none.

This is the honest headline for the model paragraph, and it is a better one
than the slope-scale figure below suggests on its own.

## The same thing on the slope scale, where it looks worse

Measured on the psychometric slope contrast (IPS − vertex), averaged over stake
terciles:

| | risky first | risky second | second − first |
|---|---|---|---|
| **Observed** | −0.017 | −0.123 | **−0.106** |
| `perc` model | −0.053 | −0.045 | **+0.008** |
| `n1n2` model | −0.031 | −0.047 | −0.016 |

**`perc` predicts no order asymmetry at all**, and `n1n2` predicts about 15% of
the observed one. The composition argument — that ν₁ = perceptual + memory and
ν₂ = perceptual should make the second-presented option suffer more — does not
survive measurement: the decision SD mixes both options, so the proportional
difference does not reach the slope.

Neither model is *rejected* on this: the posterior predictive interval on the
order contrast is wide (perc: observed −0.094 against [−0.128, +0.146],
p = 0.917; n1n2 [−0.158, +0.130], p = 0.840). But not-rejected is not the same
as predicted, and the order asymmetry is the paper's central behavioural fact.

**Write it as a limitation**: the model localises where in payoff space the
representation degrades, and does not by itself generate the dependence on
presentation order, which remains a descriptive feature of the data.

### No model in the family does all three things

Searched exhaustively. Ranked by how much of the observed order asymmetry in
the psychometric slope (−0.094) each model reproduces:

| model | converges | ΔELPD vs `perc` | order asymmetry | mechanism |
|---|---|---|---|---|
| `spl5-n1` | ✓ 1.010 / 855 | **−41 ± 13** | 57% | position |
| `percx` | ✓ 1.010 / 2150 | +2 ± 4 | 45% | **none — see below** |
| `spl3-n1` | ✓ 1.000 / 1808 | −11 ± 11 | 36% | position |
| `n1n2` | **✗ 1.120 / 42** | +5 ± 9 | 15% | position |
| `perc` | ✓ 1.000 / 5179 | reference | **0%** | stage |

There is no model that converges, is mechanistically interpretable, is
ELPD-competitive **and** produces the order asymmetry. The models that produce
it either buy it with a covariate (`percx`) or pay 11–41 nats for it (`n1`
family, which also inverts the narrative — it places the effect on the
FIRST-presented option).

**That is a result, not a failed search.** No placement of a payoff-dependent
noise change inside this observer generates the dependence on presentation
order. Say so.

**And it does not damage the paper**, because the order asymmetry is already
established model-free in Figure 3 — including the brain–behaviour link at
r = 0.53, which is the strongest single result in the manuscript. Figure 5
answers a different question: at which STAGE the representation degrades, and
WHERE IN PAYOFF SPACE. It does not need to re-derive Figure 3 to do that, and
claiming it does would be the overreach.

### What the `percx` interaction actually is

σ_perceptual at 7 CHF, all four cells:

| | vertex | IPS | cTBS effect |
|---|---|---|---|
| risky **second** | 0.161 | 0.206 | **+28.5%**, P = 0.993 |
| risky **first** | 0.168 | 0.188 | +12.0%, P = 0.842 |

Baseline is the same in both trial types (main effect of order −8.8%,
P = 0.19), so **the interaction lives entirely in the cTBS effect**: it is
12.7% larger on risky-second trials, P = 0.105, 95% CrI [−29%, +8%].

**What it is mechanistically: a scalar multiplier on the whole cTBS effect,
indexed by trial type.** σ_perceptual is shared — ν₁ = σ_perc + σ_mem and
ν₂ = σ_perc — so the interaction raises the noise on BOTH options equally
within a trial, just by more on risky-second trials. It does not say the risky
option is affected more, or the second-presented one; it says the effect is
bigger in one condition.

**Why that produces the slope asymmetry:** the decision SD is
√((w₁ν₁)² + (w₂ν₂)²). A larger σ_perc raises both terms, so the psychometric
curve flattens more wherever σ_perc rises more — which the interaction has
simply declared to be risky-second trials. The asymmetry is arithmetic from the
regressor, not a consequence of anything in the observer.

**Contrast with `n1n2`, which does contain a mechanism.** There cTBS raises ν₂,
the noise on whichever option came second: the safe option on risky-first
trials, the risky option on risky-second trials. *One* parameter change,
different behavioural consequences by order, because the risky and safe options
sit at different points on a payoff-dependent noise function and receive
different prior pull. That is an explanation — it is just too weak, delivering
15% of the observed asymmetry.

### Why the `*x` models are not the answer, despite fitting it better

`percx` and `n1n2x` do reproduce about 45% of the order asymmetry, converge
cleanly and cost nothing in ELPD. They should still not be reported, because of
what they are: `REGX = 'stimulation_condition*risky_first'` puts presentation
order on the noise channel **as a trial-level covariate**.

For `percx` that is incoherent as a mechanism. The perceptual channel is
*shared by both options within a trial*, so the model says the encoding noise
for BOTH options takes one value on risky-first trials and another on
risky-second trials, and that cTBS moves that shared value differently in the
two trial types. There is no perceptual process that could do that: the
observer cannot set a single encoding noise for both magnitudes according to
which one happens to be risky.

They fit the order effect better because they are handed it — the thing to be
explained enters as a regressor. `n1_evidence_sd` and `n2_evidence_sd` index
POSITION, which is structural (the second option must be compared against a
memory of the first), and `perceptual`/`memory` index STAGE, which is
structural too. `risky_first` on a shared channel indexes neither.

Keep them in the supplementary table as fitted alternatives; do not build the
account on one.

**Do not write** "we localised the effect to perceptual encoding" as though it
beat the position-indexed alternative. It did not; it was not rejected by it,
and it is preferred on parsimony and on sampling behaviour. Those are honest
reasons and they should be given as such.
