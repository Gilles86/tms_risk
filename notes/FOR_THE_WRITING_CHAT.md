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

## 2. The reported model is `log-power-percpmu` — DECIDED 2026-09-10

**Supersedes two earlier versions of this file**, which named `log-power-perc`
and then `log-power-n1n2`. Write **`percpmu`**.

⟦MODEL_NAME⟧ is a power-law noise function in the **stage-indexed**
parameterisation — σ(second-presented) = σ_perceptual, σ(first-presented) =
σ_perceptual + σ_memory — with the cTBS effect free on the perceptual channel
and on the two magnitude-prior means.

### Why the stage parameterisation, stated correctly

The earlier draft of this file said the position-indexed model "cannot be
fitted". **That was wrong and must not appear in the paper.** An adversarial
audit found six converged position-indexed fits already on disk, with the cTBS
regressor: `spl4/5/6/7-n1n2` and `cspl5/7-n1n2`, r̂ ≤ 1.010, ESS 2 424–14 412.
Only the 2-anchor forms (power, affine) and the 3-anchor splines fail, because
their lowest anchor sits at 7 CHF where it is ~80% loaded by SAFE presentations
and therefore trades against `safe_prior_mu`, the weakest-identified parameter
in the model. Move the low anchor to 20 CHF and it samples at ESS 9 088.

**And when the position-indexed models converge, they AGREE with the
stage-indexed one:** a ~20–27% noise increase confined to low payoffs, present
on both presented options roughly equally. The "the effect is on the
second-presented option only" reading came solely from an r̂ 1.12 trace whose
coefficient varied 40% across chains. **Any claim of order-specific noise is
withdrawn.**

So the reason to report the stage parameterisation is parsimony and the fact
that the baseline identifies the two stages directly — not that the alternative
failed. Report `spl5-n1n2` or `spl7-n1n2` as the position-indexed robustness
check.

### One Methods sentence on why the parameterisation matters

Gilles wants this touched on briefly. It must NOT say the stage parameterisation
is "necessary for convergence" — `spl5-n1n2` and `spl7-n1n2` converge at ESS
2 822 and 14 412 and a reviewer refitting would find them. Accurate wording,
about two sentences, in Methods (Model estimation), not Results:

> At the two-parameter power form, the position-indexed parameterisation
> (separate noise functions for the first- and second-presented option) does not
> sample reliably: its low anchor falls at the bottom of the payoff range, where
> presentations are almost all safe options, so it trades against the mean of
> the safe magnitude prior — the least well-identified parameter in the model,
> since the safe option takes only five distinct values. The stage-indexed
> parameterisation shares one perceptual component between the two options and
> is not subject to this trade-off; with four or more spline anchors the
> position-indexed models sample cleanly and give the same answer (Supp. Fig. X).

Why this is the better sentence: it names the cause, it is checkable, and it
turns a sampling annoyance into a statement about what the design can and cannot
identify — which is a real limitation worth reporting.

## 3. Numbers to use

All tokens are in `notes/PLACEHOLDERS.md`. Keep the ⟦…⟧ brackets intact.

`⟦MODEL_NAME⟧`, group level, IPS − vertex:

| Quantity | Token |
|---|---|
| Δν, perceptual channel @ 7 CHF | ⟦DNU7⟧, 95% CrI ⟦DNU7_CRI⟧, P(Δν>0) = ⟦DNU7_P⟧ |
| Δν @ 112 CHF | ⟦DNU_HIGH⟧ — not credible either way |
| Credible range in payoff | up to ⟦CRED_UPPER⟧ |
| Memory channel | shared across conditions in this model |
| Safe prior mean | ⟦SAFE_PMU⟧, P(<0) = ⟦SAFE_PMU_P⟧ |
| Risky prior mean | ⟦RISKY_PMU⟧ — null |

The sentence to write: **cTBS raises the representational noise of the
perceptual stage, and only at small payoffs.** Both restrictions are
informative — a global increase and a payoff-flat increase are each ruled out by
the same posterior.

### The ELPD ladder (Supp. Table 1, Supp. Fig. 2)

Paired ΔELPD against ⟦MODEL_NAME⟧:

| Claim | Token |
|---|---|
| cTBS changed something | ⟦ELPD_NULL⟧ (⟦ELPD_NULL_DSE⟧) worse — ⟦SE_NULL⟧ SE |
| It is not the memory stage alone | ⟦ELPD_MEM⟧ (⟦ELPD_MEM_DSE⟧) — ⟦SE_MEM⟧ SE |
| **It is not the priors alone** | ⟦ELPD_SPMU⟧ (⟦ELPD_SPMU_DSE⟧) — ⟦SE_SPMU⟧ SE |
| Freeing the memory channel too buys nothing | ⟦ELPD_PERCMEMPMU⟧ (⟦…_DSE⟧) |

The `spmu` rung — cTBS on the priors only, no noise change — is what makes the
noise effect **necessary** rather than merely sufficient. Lead with it.

**Do NOT claim the prior-mean shift is established by model comparison.**
`percmem` (noise only) is ⟦ELPD_PERCMEM⟧ behind, ⟦SE_PERCMEM⟧ SE — ELPD prefers
the prior shift but does not establish it. What carries it is the posterior
predictive: ⟦MODEL_NAME⟧ is the only model covering all eight targeted
statistics and ⟦GRID_COV⟧ of the 22 design-grid cells. Report it that way round.

**Weber memory (`power+weber`) is a supplement, not the reported model.**
Constraining the memory channel flat costs ⟦ELPD_PW⟧ (⟦ELPD_PW_DSE⟧) — nothing
— with two fewer parameters, and changes no conclusion. It is not adopted
because Figure 4 identifies the memory term's magnitude-dependence directly on
the baseline (falling to ⟦MEM_RATIO⟧ of its low-payoff value), and the reported
model should not contradict its own Figure 4. Say exactly that.

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

## 9. Prior shifts are IN, but only the means — REVERSED 2026-09-10

Earlier versions of this file said prior shifts stay out. That applied to prior
WIDTHS and still does; it does not apply to prior MEANS, and the reported model
now has them.

**Widths stay out, on a stated principle.** The shrinkage weight is
σ_p²/(σ_p² + ν²), so a wider prior and a lower noise level move the same
quantity. A model in which cTBS changes the prior WIDTH is not a competing
account — it is the noise account in different coordinates, and no amount of
predictive accuracy adjudicates between coordinates. Declare this as an
admissibility criterion BEFORE the comparison, not as a result of it. (Say so
plainly: several width models score at or above the reported one, and they are
excluded on this principle rather than on their scores.)

**Means are in.** A prior MEAN shift moves WHERE the percept is pulled to, not
HOW HARD, and it is the one thing in the family that produces a bias rather than
a slope change. The observed risky-second effect is mostly bias, which no
noise-only model reproduces: adding the prior means takes the predicted effect
from ⟦PPC_FAIL_NOISEONLY⟧ to ⟦PPC_FAIL⟧ against an observed ⟦PPC_OBS_SECOND⟧.

**The `*x` order-interaction models are also excluded**, on the same footing: they
make the perceptual noise of a stimulus depend on where in the trial it was
shown, which is not a claim about perception.

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

## 11. What is still open

**Settled since the last version:** which model is reported (`percpmu`), that
the position-indexed family DOES fit, that the noise effect is necessary
(`spmu`), that Figure 4 moves to perceptual/memory coordinates, and that the
prior-mean shift rests on the predictive checks rather than on ELPD.

**Still pending, all tokenised, none changing a sign or a conclusion:**

* The final ELPD numbers. A single LOO pass over the placement ladder plus the
  noise-form sweep is running; every ⟦ELPD_*⟧ token comes from it.
* ⟦BB_INTERVAL⟧ — the brain–behaviour posterior interval. Its hierarchical fit
  did not converge. **Leave the brackets and write no number.**
* The smooth-spline supplement (does the payoff-localised noise increase survive
  a flexible noise function?). All 23 spline fits converged; only the figure is
  outstanding.
* `percpmuso` — whether the prior shift is larger for participants who received
  IPS first. Exploratory; if it lands it is one supplementary sentence, and if
  it does not, nothing in the paper changes.

---

# Audit of `notes/paper/TMS_paper_v12_draft.docx` (as of 2026-09-09 15:59)

Paragraph numbers are lines in `notes/paper/TMS_paper_v12_draft.txt`, produced
by

    python -c "import docx;d=docx.Document('TMS_paper_v12_draft.docx');
    open('TMS_paper_v12_draft.txt','w').write(chr(10).join(p.text for p in d.paragraphs))"

Re-run it after any edit; the equations are OMML objects and drop out of the
text conversion, so anything that looks like a gap (¶133, ¶135, ¶149, ¶169…)
is an equation, not a missing sentence.

**v12 is much further along than the v11-keyed plan this file used to carry.**
It already reports the position-indexed model, already has the Weber-violation
section, already refuses the per-participant brain–behaviour correlation, and
already frames the PPC p-values correctly. What follows is only what is left.

## A. One error, not a pending number — fix this first

**¶80: "The model converged without any adjustment to sampler or priors
(r̂ ≤ 1.002, ESS ≥ 1727)."** That pair of numbers comes from
`notes/analyses/weber_affine_convergence.md`, which recorded a fit under the
**raw (pre-KLW) choice rule**. Under the consistent rule that the paper now
describes, the same model at default priors gives **r̂ 1.12 / ESS 42**. The
sentence as written is not merely stale, it asserts the opposite of the truth
and would not survive a reviewer running the code.

Replace with ⟦RHAT⟧ / ⟦ESS⟧ and, if the final fit needs the noise-anchor prior,
one sentence saying so. Do not write "without any adjustment" unless the final
sweep earns it.

## B. Numbers to tokenise (all in `notes/PLACEHOLDERS.md`)

| ¶ | In the draft now | Replace with |
|---|---|---|
| 80 | ΔELPD = 35.6, dSE 8.8 (power vs Weber) | ⟦ELPD_WEBER⟧ (⟦ELPD_WEBER_DSE⟧) — the draft's pair is a raw-choice-rule number |
| 80 | "See Table XXXX" | Supplementary Table 1, and cite Fig. S2 |
| 80 | "all \|r\| ≤ 0.07 between the two options" | recompute on the final trace |
| 80 | 0.029 log units, [0.002, 0.057], p = 0.020 | ⟦DNU7⟧, ⟦DNU7_CRI⟧, ⟦DNU7_P⟧ |
| 80 | "33 of 35 participants" | ⟦N_INCREASE⟧ |
| 80 | "p = 0.27 and p = 0.23", "p = 0.18" | ⟦DNU_FIRST_P1⟧ / ⟦DNU_FIRST_P2⟧, ⟦DNU_SECOND_HIGH_P⟧ |
| 81 | 10.7%, 2.5%, 8.1%, −2.1%, 17 CHF, 14 CHF, −1.8% | all re-derived from the final mechanism extraction |
| 82 | "the model has [N] free parameters per participant" | count it on the final model |
| 82 | "r = [0.92 or 0.93]", "[95 or 97]%" | ⟦PPC_R⟧, ⟦PPC_COVERAGE⟧ |
| 82 | +0.053 against [−0.014, +0.044], p = 0.005 | ⟦PPC_FAIL⟧ |
| 82 | order contrast p = 0.09 / slope p = 0.84 / three-way p = 0.51 | ⟦PPC_ORDER⟧, ⟦PPC_SLOPE_ORDER⟧, ⟦PPC_THREEWAY⟧ |
| 85 | "95% posterior interval [x, y]" | ⟦BB_INTERVAL⟧ — **still blocked**, leave the brackets |
| 78 | Fig. 5 caption: "panels e and f show the posterior of the mean over the 35 sampled participants, which is narrower" | **stale** — those panels now draw the same population-level interval panel c does. Delete the clause; nothing in the figure is shaded on a different footing any more. |
| 74 | "3, 5, or 7 anchor payoffs" | ⟦SPLINE_ORDERS⟧ once the spline ladder lands |

¶82 also says "eight targeted posterior predictive checks … Seven of the eight";
keep it consistent with ⟦PPC_N_PASS⟧ / ⟦PPC_N_TOTAL⟧ rather than hard-coding.

## C. Methods gaps — whole sentences that are simply absent

§ *Model estimation* (¶177–190) describes the offset parameterisation and the
random-effects structure well, and then stops. It contains **no sampler
settings, no convergence criterion, no prior specification and no statement of
how ELPD was computed.** A methods reviewer will ask for all four. Add, at the
end of ¶190:

* the sampler: ⟦SAMPLER⟧;
* the gate: ⟦CONV_GATE⟧, and that every model reported met it;
* the priors, including ⟦TAU_PRIOR⟧ if the final fit uses it, and that it
  applies to the noise anchors only;
* LOO-CV via Pareto-smoothed importance sampling (Vehtari et al. 2017), that
  ΔELPD is paired with the SE of the paired difference, and that all
  compared models were fitted under one prior and one sampler configuration.

§ *The flexible PMC model* (¶164–176) documents **only the spline model**, with
"1st 3rd degree spline, with bounds between 7 and 112" at ¶169. The main text
reports the **power law** (¶74, ¶80) and uses the anchor models as a robustness
check. Methods must define the power law that Results reports — the
two-parameter form, what b = 0 means, and that the anchor models are the
flexible alternative — otherwise the reported model has no Methods entry at all.

§ *The PMC model* (¶141–143): this is where the choice rule lives. Check the
equation objects carry the **KLW-consistent** form (§1 of this file): the
decision variable is the noisy posterior mean, so its SD is w·ν with
w = σ_p²/(σ_p² + ν²), and the comparison is normalised by
√((w₁ν₁)² + (w₂ν₂)²). ¶75 already states the shrinkage weight as
σ_p²/(σ_p² + ν²), so the text is consistent; verify the display equations are
too, since they survive the .docx but not the .txt conversion.

Nowhere does the paper state that **every** model in it — including the probit
— is hierarchical Bayesian with partial pooling and that no maximum-likelihood
estimator or bootstrap CI is used anywhere. One sentence, in § *Cognitive
computational modeling* (¶122), and the reliability and error-bar questions
answer themselves.

## D. Prose that should change, beyond the numbers

**¶80, the model-choice justification.** As written the case is orthogonality
plus flexibility. That is true but it is the weaker half. Add the reason from
§2: the position-indexed model is **the only member of the family that can
express the order-specificity the paper is about**, and ELPD deliberately does
not adjudicate between placements (⟦ELPD_UNRESOLVED⟧ dSE). Saying "model
comparison establishes X, Y and Z but not W, and here is why we chose within W
on other grounds" is stronger than implying ELPD settled it.

**¶82, the underprediction.** "an underprediction of roughly fourfold" reads as
a confession. It is on the sharpest of eight statistics, and on the scale a
reader actually reads — choice proportions — the model recovers the *shape* of
the order dependence and ⟦ASYM_FRACTION⟧ of its size (⟦ASYM_MODEL⟧ against an
observed ⟦ASYM_OBS⟧). Keep the honesty; add the shape result immediately after,
so the paragraph ends on what the model does rather than on what it misses. The
existing explanation (partial pooling shrinks contrasts by construction, and
none of the eight statistics is in the likelihood) is correct and well put —
keep it verbatim.

**¶81, the decomposition paragraph.** The percentages quoted for the risky
and safe options (+8.1%, −1.8%, etc.) come from traces whose 95% credible
interval covers zero at EVERY safe payoff — e.g. the risky option at the
largest safe payoff is +4.0% [−1.5, +9.6]. They are a decomposition of the
model's arithmetic, not effects. Quote them as "the model attributes X to
…", never as findings, and do not attach a p to them. The two SOLID traces
(perceived ratio, decision SD) are the ones that enter choice and are the
ones to lead with.

**Title of the section at ¶71** — "localizes the cTBS effect to the foreground
option" — "foreground" appears nowhere else in the paper and is not defined.
Use the second-presented option, or define the term at first use.

## E. Already right — do not touch

* **¶75, the prior-fixing justification.** Exactly the §9 argument, including
  the identifiability point and the stress-study contrast. Leave it.
* **¶86, the refusal to correlate per-participant model parameters.** Exactly
  §4, with the right reasons in the right order.
* **¶82's framing** of a posterior predictive p as a warning sign rather than a
  discovery. Rare and correct.
* **¶78, the Figure 5 caption.** Correctly separates the population-level
  parameter (panels c, g) from the posterior of the mean over 35 participants
  (e, f), and states that the same statistic is applied to observed and
  simulated choices so both are attenuated equally. That sentence is what makes
  panels h and i legitimate; keep it.
* **¶62–68, the Weber-violation section**, and ¶69–70 on stake specificity.
* **¶103–114, participants and exclusions.** The arithmetic that was wrong in
  v11 is right here: 78 → 37 invited → 35 analysed.

# How we "know" it is perceptual — what was actually tested

This is the paragraph most likely to be over-claimed, so here is exactly what
the evidence supports. All ΔELPD are PAIRED against the reported model
(⟦MODEL_NAME⟧ = `log-power-n1n2`, §2), with the standard error of the paired
difference. NOTE: the numbers in the tables below were computed against
`log-power-perc`, the earlier candidate; the *directions* and the qualitative
conclusions carry over unchanged, but every value is re-derived on the final
one-prior sweep and is tokenised in `notes/PLACEHOLDERS.md`. Write the argument,
not the digits.

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
> we therefore report the model in which both presented options' noise is free,
> which is the account that can express the order dependence, and state that the
> comparison does not adjudicate between the two.

## The order asymmetry — WITHDRAWN 2026-09-10

Earlier versions of this file reported that `n1n2` recovers ~40% of the observed
order asymmetry against `percmem`'s 22%, and built an argument for
position-indexing on it. **That number came from a trace at r̂ 1.12 / ESS 42 and
is not a measurement.** Every converged position-indexed fit puts the cTBS
effect on both presented options roughly equally.

What survives, and what the paper should say: the cTBS effect on choices is
concentrated on risky-second trials; the model underpredicts its SIZE; and no
model in the admissible family generates the order dependence from a
position-specific noise change, because there is no evidence for one. The
asymmetry remains a descriptive feature of the data that the model reproduces in
shape and underpredicts in magnitude.

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
