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

## 10. What is still open

`n1n2` under the consistent rule. Three jobs are running; if any converges the
reported model may switch back and §2–3 change. **Do not write §2 as final until
told.** Everything else above is settled.
