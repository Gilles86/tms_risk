# How to edit the paper — the current plan

Written 2026-09-02, superseding everything in
`notes/archive/2026-09-02_superseded/`. One document, current as of today.
Companion: `manuscript_style_card.md` (how to write it so it sounds like you).

Source text: `notes/paper/TMS_paper_v10.docx`, extracted to
`notes/paper/TMS_paper_v10.txt` for searching.

---

## 0. The one-paragraph summary of what changed

The cognitive model is refit under a new parameterisation whose free parameters
ARE the noise SD at named payoffs. Three things follow:

1. **Weber's law is wrong for these data**, and demonstrably so in the baseline
   session before anyone was stimulated. That is a new result and it gets a new
   figure.
2. **The cTBS effect sits on the second-presented option's noise**, not on a
   shared perceptual component. The perceptual/memory decomposition is dropped.
3. **The model reproduces the consistency drop but underpredicts the
   risk-attitude shift** by 2-4x. That has to be stated, not buried.

---

## 1. Figures

| # | content | file |
|---|---|---|
| 1, 2 | paradigm, nPRF targeting, decoding | unchanged |
| 3 | psychophysics: what cTBS did to choices | `notes/figures/fig3_probit.pdf` (rebuilt) |
| **4** | **NEW — noise grows with magnitude; Weber fails** | `notes/figures/fig4_weber.pdf` |
| **5** | the cTBS effect on that noise, and its consequences | `notes/figures/fig5_log-power-n1n2.pdf` — **see the open decision in §6 on `psd`** |
| S1 | PPC for every candidate model | `notes/figures/supp_ppc_gallery.pdf` |
| S2 | model comparison + the discriminating predictive check | `notes/figures/supp_model_comparison.pdf` |

**The old Figure 5 is retired.** Its content is either in the new Figure 5
(panels e-g) or dropped.

**Also promote out of the supplement into Figure 3**: the stake-split
psychophysics (currently Supplementary Fig 1). It is what motivates Figure 4.

---

## 2. The narrative order, which is what makes this hang together

1. **Fig 2** already shows the stimulated populations prefer numerosities below
   the presented range (IQR [6,10] vs [13,30]). So a cTBS effect *should* be
   magnitude-specific. This is a prediction, stated before any test.
2. **Fig 3** confirms it model-free: cTBS reduces consistency on low-stake
   trials, not high-stake.
3. **That raises a question the earlier papers never asked**: does the
   observer's noise depend on magnitude at all? Both previous applications of
   this model assumed scalar invariance.
4. **Fig 4** answers it on baseline data, no stimulation, n = 73.
5. **Fig 5** shows what cTBS did to that noise function.

The point of this ordering: the change of model class is forced by a
pre-stimulation result, so it cannot read as post hoc.

---

## 3. Edits, in manuscript order

Format below: the v10 text to find, what is wrong, and what to say instead. The
drafts are starting points, not replacements — rewrite in your own voice.

### 3.1 Abstract

**Find:** *"A computational cognitive model attributed both effects to a single
mechanism: cTBS raised the noise on the perceptual representation of payoffs, in
relative terms most strongly for smaller magnitudes. Because noisier
representations are drawn more toward prior expectations, this underestimated
the smaller safe options more than the larger risky ones, shifting choices
toward the risky option."*

**Wrong:** "perceptual representation" is a claim about the perceptual/memory
decomposition, which is dropped. The safe-option chain is the old mechanism.

**Say instead:** the effect is on the representation of the *second-presented*
option, at small magnitudes. Keep "in relative terms most strongly for smaller
magnitudes". Re-derive the final clause once §3.5 is settled.

### 3.2 Introduction

**Find:** *"Behaviorally, choices became less consistent and more risk-seeking,
but only when the smaller, safe payoff was presented first."* and the sentences
after it about compression being uneven.

**Wrong:** the preview carries the same dead chain as the Abstract.

**Say instead:** keep the first clause (it is still true — safe first = risky
second). Replace the compression sentences with the second-option account.

**Add, at the end of the modelling preview:** one sentence flagging that these
data reject scalar invariance, since it is now a headline result rather than a
methodological aside.

### 3.3 Results — new paragraph, before the modelling section

This is the bridge, and it is currently missing. Draft:

> Our earlier applications of this model (de Hollander et al., 2024a;
> Barretto-García et al., 2023) assumed scalar invariance — noise proportional
> to magnitude — which is the standard idealisation and which we did not test.
> Two observations, neither of which depends on the cognitive model, indicate
> that it should be relaxed here. First, the psychophysics violates it: in the
> baseline session, before any stimulation, choice consistency declined by 37%
> from low- to high-stake trials when the risky option was presented second
> (p < 0.0001; Fig. 4a-c), and the decline was larger for the second-presented
> option than the first (p = 0.026). Second, the cortical populations we
> targeted are tuned to numerosities well below the presented range (IQR [6, 10]
> versus [13, 30]; Fig. 2B), so any cTBS effect on their representations should
> be magnitude-specific rather than uniform. We therefore extended the model to
> estimate the relationship between magnitude and noise, rather than assuming
> it.

**Replaces** v10's *"we refined the model further, following the recommended
procedures to better account for the observed effects (Gelman et al., 2013; Lee
and Wagenmakers, 2014)"*, which motivates the change by appeal to method rather
than to evidence.

### 3.4 Results — section title and the model-comparison paragraph

**Find the section title:** *"A flexible noise model localizes the cTBS effect
to shared perceptual noise"*. It names a conclusion that changed.

**Find:** *"we introduced a 5-parameter B-spline function that maps magnitudes
to noisiness in natural space"*.
**Say instead:** the anchor parameterisation in log space, two free parameters
per channel which ARE the noise SD at 7 and 112 CHF.

**Every number in the following paragraph changes:**

| v10 | replace with |
|---|---|
| "best flexible beat best Weber by 34.8 nats (dSE 17.3)" | Weber costs **18 nats** in the chosen placement; **12 nats** at baseline (n = 73) |
| "beat the null by 114.1 nats (dSE 15.8)" | **122 nats** vs `nullind`; say which null |
| "adding memory noise costs nothing (2.0, dSE 3.9), removing perceptual costs 59.9 (dSE 12.2)" | **delete** — the perc/mem split is gone |
| "restricted to one presentation position fit substantially worse (64.3, 69.9)" | **delete** — the primary model *is* restricted to one position |
| "around 12% at 7 CHF against roughly half that over most of the range" | **+29.7% at 7 CHF**, crossing zero near 51 CHF, credible only over **7–11.5 CHF** |

**DELETE OUTRIGHT:** *"indicating that the order specificity emerges from the
model rather than being fitted directly"*. The primary model confines the effect
to one presentation position by construction. This is a real loss of an argument
and cannot be finessed.

### 3.5 Results — the Figure 5 walkthrough section

**Find:** the whole section *"Understanding the influence of cTBS on
neurocognitive representations"*, and in particular *"When the risky option was
presented second, the safe option lost more perceived value than the risky
option at every safe payoff"*.

**Wrong:** that chain follows from the perceptual/memory model. Under the
n2 placement the effect is on whichever option came second.

**Say instead:** compress to two paragraphs built on Fig 5f-g (the two
ingredients of the decision variable: perceived ratio and decision noise). The
decision-space maps and the leverage argument go to the supplement or are cut.

**Find:** *"The model reproduces the observed (lack of) change in the proportion
of risky choices in both conditions across all stake sizes (Fig. 5E)."*
**Wrong:** it does not. Replace with the honest statement in §3.7.

### 3.6 Results — brain-behaviour section

**Verified, keep as is:** r(33) = 0.53, bootstrap CI [0.37, 0.67], p = 0.001,
Spearman ρ = 0.59. Reproduces exactly (mask `NPCr2cm-cluster`, selection
`cvr2pos`, measure `d_amp_median`).

**Add one sentence:** the model's own per-subject noise shift does NOT track the
amplitude loss (r = +0.07, Spearman +0.16, n = 35). Omitting this is selective
reporting of the same test at two levels. It also does not undermine the
result — the model-free consistency measure is the one the section is about.

### 3.7 Results — state the shortfall

New, short paragraph at the end of the modelling results.

**Frame it on CHOICES, not on RNP.** The risk-neutral probability is a parameter
of a probit fitted to the choices — a descriptive summary, not an observable.
Writing "the model underpredicts the shift in risk attitude" promotes a
two-stage derived quantity to the thing being explained. The observable is the
choice proportion. Keep the probit decomposition, but demoted to a DIAGNOSTIC:
it earns its place because the raw-proportion check passes (the predictive bands
are wide) while the decomposition is sensitive enough to detect the shortfall
and localise it. Draft:

> The model reproduces the direction, the order-specificity and the stake-
> specificity of the cTBS effect on choice, and about half its magnitude: the
> predicted increase in risky choices when the risky option was presented second
> was 0.025, against an observed 0.053. Decomposing the choices into the two
> parameters of a psychometric function locates the shortfall: the model
> captures the reduction in choice consistency (predicted −0.48 against an
> observed −0.70) but not the accompanying shift in the indifference point
> (+0.021 against +0.049), and no model variant we fitted reproduced the latter.

On the mechanism panel (`log-power-n1n2`, risky second): the two channels CROSS
at around 17 CHF. Decision noise dominates at small payoffs (**+10.7% at 7 CHF**
against +2.5% for the perceived ratio) and the perceived-ratio shift dominates
at large ones (**+8.1% against −2.1% at 28 CHF**). Risky first crosses earlier,
near 14 CHF (+9.8% vs −1.8% at 7 CHF; +6.1% vs +1.0% at 28 CHF).

Two earlier versions of these numbers were wrong, both from aggregation rather
than from the model:

* computing the panel from group-level parameters instead of averaging over
  participants understated the bias channel by up to 2.9x;
* computing it from each participant's MEDIAN parameters instead of integrating
  over posterior draws understated the risky-second effect by 21% and flipped
  the sign of the risky-first one.

The panel is now built by `extract_anchor_mechanism.py`, which integrates over
draws on the trials actually presented and averages across participants with the
mean. Scored against the model's own simulated posterior predictive it reaches
r = 0.991 (the plug-in version reached 0.977). Full comparison:
`notes/analyses/aggregation_check.md`. **Never average these quantities over a
uniform ratio grid** — the ladder is per-subject calibrated, and a uniform grid
flips the sign of the risky-first effect.

### 3.8 Discussion

**Find:** *"representational noise grows sublinearly with payoff magnitude, and
that cTBS raised it by an approximately constant absolute amount"*.
**Wrong on both halves.** In natural space the fitted noise grows **supra**-
linearly (ν ∝ x^1.28 for the second-presented option), and the cTBS effect is
directly magnitude-specific rather than a constant absolute increment.

**Find:** *"this reduced the perceived value of the (smaller) safe option more
than that of the risky one"*. Re-derive under the n2 placement.

**Add:** the Weber result deserves a paragraph of its own in the Discussion.
It is a finding about magnitude representation in 73 people that the earlier
papers assumed away.

### 3.9 Methods

* Replace the B-spline / natural-space description with the anchor
  parameterisation: forms, placements, what the free parameters mean, the log
  link.
* State the prior spec (`v1-2026-08-28`), the sampler, and the **bauer commit**
  (`93c2e9a`) the traces are stamped with.
* Say which choice rule was used (raw-evidence vs KLW-consistent `diff_sd`).
* Describe the baseline Weber analysis: probit with subject dummies, session 1,
  73 participants, bootstrapped over participants.

---

## 4. Numbers you will need, all verified

**Weber violation, baseline session, n = 73, no stimulation**

| order | slope low | high | drop | |
|---|---|---|---|---|
| Risky second | 2.62 | 1.65 | **−37%** | p < 0.0001 |
| Risky first | 2.60 | 2.03 | −22% | |
| order x stake | | | | p = 0.026 |

Replicates across subgroups (never stimulated −40%, stimulated later −35% for
risky second). Fitted noise over 7–112 CHF: first-presented **1.25x**,
second-presented **2.69x**. Six noise forms agree; Weber costs 12 nats.

**cTBS effect, primary model `log-power-n2psd`**

Noise on the second-presented option **+29.7% at 7 CHF**, credible over
7–11.5 CHF, crossing zero near 51 CHF. Prior means unchanged (risky +0.7 CHF
[−2.3, +3.7]; safe −0.6 [−2.7, +1.3]). Prior widths not credibly changed
(risky −0.007 [−0.062, +0.042]; safe −0.050 [−0.164, +0.035]).

**Model comparison** (ELPD vs the primary, ±1 SE of the paired difference)

| model | Δ |
|---|---|
| + noise on 1st option | +3 ± 5 (within noise) |
| Weber noise | −18 |
| Prior width only | −5 ± 4 |
| Noise only, no prior term | −53 |
| No cTBS effect | −122 ± 14 |

**Predictive checks** (cTBS effect on the psychometric slope, risky second, low
stake; observed −0.70): primary model −0.48, ppp 0.145 (passes); prior-width-only
−0.28, ppp 0.025 (**fails**). This, not ELPD, is what justifies the noise term.

---

## 5. The three things a reviewer will attack

1. **"Model comparison does not support your mechanism."** True: the
   prior-width-only model is 5 ± 4 nats behind. The answer is the predictive
   check, which it fails and ours passes. State the stance once, up front:
   *ELPD ranks, predictive checks adjudicate.*
2. **"Your model's noise parameter does not correlate with your neural
   measure."** True (r = +0.07). Report it; the brain-behaviour claim rests on
   the model-free consistency measure, which does correlate (r = 0.53).
3. **"You fitted the effect to one presentation position and then reported that
   it is order-specific."** True of the primary model. The defence is that the
   free model (`n1n2`) puts the credible effect on the second option
   (p = 0.980 vs 0.726) and that the baseline noise functions independently
   identify the second option as the magnitude-dependent, perceptual channel.

---

## 5b. Why not the noise-only model (`log-power-n2`)

Asked and answered 2026-09-03. It is the model with only interpretable
parameters, and its mechanism panels are tidier (the option that was not second
is pinned at exactly zero). But it is not viable as the primary model:

* **53 nats worse** than `n2psd` -- not within noise, unlike the +3 for `n1n2psd`.
* **It barely produces the behavioural effect.** In the choice-proportion PPC its
  IPS and vertex curves lie on top of each other for risky-second trials, where
  `n2psd`'s separate across the whole range and match the observed points. Its
  posterior predictive p-value for the mean cTBS effect is 1.000.
* Its mechanism panels are tidy *because* it has less machinery, not because the
  data prefer it.

Keep it as a supplementary robustness check: the SHAPE of the noise function,
and the direction of the cTBS effect on it, are the same with and without the
prior-width term. Figure: `notes/figures/fig5_log-power-n2.pdf`.

## 5c. What counts as an observable (2026-09-03)

A principle that has to govern how the model comparison is written, because
several of today's confusions came from violating it.

**The risk-neutral probability and the psychometric slope are parameters of a
probit fitted in log-ratio space. They are not latent quantities the brain has.**
They mean what they are usually taken to mean only if the true choice function
IS a probit in log(risky/safe). It is not: in the PMC model nu depends on payoff,
so both the weight w and the decision SD vary along the ladder, and the choice
function is only approximately probit (measured: up to 3.7 percentage points and
a 14% slope error, concentrated at risky-second / high payoff).

Two consequences.

**1. The slope/RNP decomposition is not identified across conditions.** If cTBS
changes the SHAPE of the psychometric function -- not merely its location and
scale -- then describing the change as "so much slope, so much indifference
point" depends on the family you fit. A different family splits the same curve
difference differently. So "cTBS shifted risk attitude by X" is a statement about
a probit, not about the participant.

**2. The only condition-free observable is the choice proportion** for the
choice problems actually presented. Everything else is a summary computed
through a model, and every such summary must be applied IDENTICALLY to observed
and simulated data before the two are compared (`probit_ppc_ml.py` does this by
simulating choices and refitting the same ML probit; the closed-form derivation
does NOT and was discarded for that reason).

### How to write it
* **Lead with choices.** "The model reproduces the pattern of the cTBS effect on
  choice and about half its magnitude (predicted +0.025 against an observed
  +0.053 for risky-second trials)."
* **Then use the probit as a DIAGNOSTIC, clearly labelled as one.** "Summarising
  both observed and simulated choices with the same probit locates the
  shortfall: the model captures the reduction in slope but not the accompanying
  shift in the fitted indifference point."
* **Do not write** "the model underpredicts the shift in risk attitude", or
  "cTBS made participants more risk-seeking by 0.049" as though the RNP were
  measured. Say what was measured -- choice proportions -- and name the probit
  when the probit is doing the work.
* Figure 3 keeps slope and RNP: they are applied identically to both conditions
  there, so they are a fair descriptive summary of the DATA. The caption should
  say "parameters of a probit fitted to the choices", not "risk attitude and
  choice consistency" unqualified.

### The related finding this explains
The model's indifference CONTOUR barely moves between conditions (IPS - vertex,
in ratio units: -0.005 to +0.005 for risky-second across safe payoffs). What the
model produces is flattening, not a preference shift. Yet the fitted-probit
decomposition attributes the effect largely to the intercept -- because changing
w changes the SLOPE of the decision variable in log-ratio, which a probit reads
partly as an intercept change. Same curve, two descriptions. This is exactly the
non-identification above, and it is why the choice proportions are the thing to
report.

## 6. Open decisions

* **Multiplicity.** The noise term rests on one predictive check out of eight.
  Pre-specify it as primary, or report all eight with a multiplicity note.
* **How hard to push the Weber result.** It is a standalone finding in 73
  participants. Motivating paragraph, or its own contribution?
* **Panel f of Figure 4** uses an analytic slope, not simulate-and-refit, so its
  absolute values are not on the same footing as panel c. Caption it, or refit.
* **Which model Figure 5 reports — `n1n2` or a `psd` variant. Unresolved, and
  the biggest open item.** Every model with a cTBS effect on prior WIDTH beats
  the currently-plotted `log-power-n1n2`, decisively:

  | vs `log-power-n1n2` | ΔELPD | dSE | ratio |
  |---|---|---|---|
  | `log-power-n1n2psd` | +30.3 | 6.9 | +4.42 |
  | `log-power-n1psd` | +30.0 | 7.3 | +4.10 |
  | `log-power-n2psd` | +27.5 | 8.3 | +3.32 |
  | `log-power-pmusd` | +24.5 | 9.5 | +2.59 |

  Against that: the prior-width effect has no credibly signed direction, and the
  30 nats buy **nothing** on the checks the paper is about — all four psd
  variants and `n1n2` give the same observed statistics and posterior predictive
  p-values to within 0.05, and all of them under-produce the slope flattening
  (`slope_second_ctbs` ppp 0.93-0.99). So the gap comes from the bulk
  likelihood, not from the cTBS signature.

  Whichever way it goes, the text must say it explicitly — a reviewer running
  the comparison sees this in the first table. Mechanism tables and PPCs now
  exist for all four psd variants, so Figure 5 can be rebuilt for any of them in
  seconds (`plot_fig4_big.py --model_label <label>`).
