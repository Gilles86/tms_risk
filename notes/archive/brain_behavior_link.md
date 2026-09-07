# Does the neural cTBS effect predict the behavioural one?

Written 2026-08-03. A systematic search for a per-subject brain–behaviour link, using
**only the canonical m1 encoding model** and the **latest PMC refits**
(`cogmodels.overnight`, bauer `e05f73a`), per the resolution in
`notes/encoding_model_choice.md` §7.4.

**Headline.** There is one, and it is not the one the earlier analyses looked for.
Subjects whose right-parietal nPRF **gain** dropped most after cTBS were also the
subjects whose **choice consistency** dropped most — but only on trials where the
**risky option was presented second** (i.e. the safe option came first), which is
exactly the cell where the group-level behavioural effect lives.

    r(33) = +0.53, p = .0010   (Spearman rho = +0.59, p = .0002)
    bootstrap 95% CI [+0.37, +0.67];  leave-one-subject-out range [+0.50, +0.56]

It is **site-specific** (significantly weaker in left parietal, frontal and
occipito-temporal cortex), **order-specific** (r = −0.04 for risky-first trials,
Williams p = .0082), and it survives the counterbalancing control. A separate,
statistically independent **within-subject trial-by-trial** test points the same way.

The measures the previous analyses used — the PMC per-subject Δν and ΔP(chose risky) —
are **not** the ones that carry it, and §5 explains why.

---

## 1. What was searched

Per-subject IPS − vertex contrasts throughout, n = 35.

**Neural, all from m1** (`amplitude` is m1's only per-session regressor, so Δamplitude
is a gain change at fixed tuning and cannot be confounded with Δbaseline the way m2's
is — `mu`, `sd` and `baseline` are identical across arms by construction):

| family | measures |
|---|---|
| nPRF gain | mean / median Δamplitude, relative and log-ratio versions, in NPC12r and the NPCr2cm-cluster stimulation site, under three voxel selections (cvR² > 0, top-100 by cvR², all non-degenerate) |
| gain by preferred numerosity | Δamplitude in voxels preferring ≤ 14 vs > 14, and a **tuning-weighted profile** Δamp(n) = Σ_v w_v(n)Δamp_v / Σ_v w_v(n) with w_v(n) the voxel's Gaussian tuning at log n, for n = 7, 10, 14, 20, 28, plus its 7-minus-28 slope |
| decoding | m1-based trial-wise decoder: accuracy corr(E, n1) averaged over runs, mean absolute log error, mean posterior width — overall and split at the median presented numerosity |

**Behavioural:**

| family | measures |
|---|---|
| model-free | ΔP(chose risky); psychometric slope ("consistency") and indifference point of a penalised logistic on log(risky/safe) — each computed overall, for risky-first, for risky-second, and as the risky-second-minus-risky-first difference; low- vs high-`n_safe` splits |
| model-based | per-subject Δν_perceptual at 7/10/14/20/28 CHF and the localisation slope, from `subject_noise_shift.flexible2nf.tsv` and `.flexible1nf.tsv` |

Scripts: `tms_risk/modeling/scripts/extract_brain_behavior_table.py` (builds
`notes/data/bb_*.tsv`), `tms_risk/behavior/scripts/analyze_brain_behavior_link.py`
(the correlation search), `check_brain_behavior_robustness.py` (§3),
`trialwise_decoding_choice_link.py` (§4).

---

## 2. The result

### 2.1 The correlation

Gain change in the stimulation site (median Δamplitude over cvR² > 0 voxels in
NPCr2cm-cluster) against Δ psychometric slope on risky-second trials:

| | r | p |
|---|---|---|
| **Pearson** | **+0.533** | **.0010** |
| Spearman | +0.592 | .0002 |
| Theil–Sen slope | +4.08 [+1.94, +6.92] | — |
| voxel-count weighted | +0.536 | — |
| subjects with ≥ 20 voxels (n = 26) | +0.505 | .0085 |
| subjects with ≥ 50 voxels (n = 16) | +0.549 | .028 |

Positive r with Δ = IPS − vertex on both axes means: **the more gain a subject lost,
the more consistency they lost.**

It is not an artifact of the unbounded logistic slope. Replacing the slope with bounded
alternatives on the same trials: Spearman(chose_risky, log ratio) gives r = +0.477
(p = .0037), point-biserial r = +0.470 (p = .0044). Two fit-quality measures
(classification accuracy, log-loss) show nothing (r = +0.08 / −0.10), so what tracks
the gain change is the *steepness of the payoff-ratio dependence*, not how well any
model fits.

### 2.2 Site specificity

Median Δamplitude in each ROI against the same behavioural measure (cvR² > 0
selection), and a Williams test against the stimulation site:

| ROI | r | p | vs stimulation site |
|---|---|---|---|
| **NPCr2cm-cluster** (stimulated) | **+0.533** | .0010 | — |
| NPC12r (right numerosity ROI) | +0.480 | .0035 | p = .33 |
| NPCl (**left** parietal, contralateral) | +0.211 | .22 | **p = .020** |
| NF1 (frontal) | +0.172 | .32 | **p = .031** |
| NTO (occipito-temporal) | +0.018 | .92 | **p = .0075** |

Graded by distance from the coil, and gone in the two ROIs furthest from it.

### 2.3 Order specificity

| behavioural measure | r with Δ gain | p |
|---|---|---|
| Δ consistency, **risky second** | **+0.533** | .0010 |
| Δ consistency, risky first | −0.037 | .83 |
| Δ consistency, risky-second **minus** risky-first | +0.394 | .019 |

Williams test of the two against each other: **t(32) = +2.82, p = .0082**. Partialling
the risky-first version out of the risky-second one leaves r = +0.540 (p = .0010);
adding ΔP(chose risky) as a second covariate leaves r = +0.540 (p = .0012).

This matters because the order asymmetry was **not** chosen to make the correlation
work — it is the paper's own established signature, on *both* behavioural measures. From
the hierarchical `probit_order` model (`notes/data/localnoise_group_posterior.tsv`):

| group posterior, IPS − vertex | risky second (safe first) | risky first |
|---|---|---|
| psychometric **slope** (consistency) | **−0.375, CrI [−0.695, −0.047]**, P(<0) = .988 | −0.025, CrI [−0.361, +0.310] |
| risk-neutral probability | **+0.055, CrI [+0.023, +0.089]**, P(>0) = .999 | +0.007, CrI [−0.026, +0.039] |

So the group-level consistency reduction is itself credible and confined to safe-first
trials, and restricting the brain–behaviour analysis to those trials follows from a
result that predates it. Note the per-session logistic slopes used for the correlation
are a much noisier estimator of the same thing and do *not* reach significance at the
group level on their own (−0.35, t(34) = −1.01, p = .32) — the hierarchical model is
what establishes the group effect. Model-free ΔP(chose risky) agrees: +0.053
(t(34) = 2.29, p = .028) safe-first vs +0.006 (p = .82) risky-first.

### 2.4 Stability

Across 2 masks × 3 voxel selections × 3 amplitude summaries (18 combinations),
`notes/data/bb_link_stability.tsv`:

- risky-second: **18/18 positive**, 17/18 at p < .05, range +0.27 to +0.53
- risky-first: 9/18 positive, **0/18** at p < .05, range −0.16 to +0.16

---

## 3. Everything that could have produced it spuriously

**Counterbalancing / session order.** 18 subjects had IPS in session 2, 17 in session 3.
Both Δ measures differ between those groups in the same direction (Δconsistency
risky-second: +0.38 vs −1.12, p = .027; Δ gain: +0.005 vs −0.132, p = .11), which would
inflate the correlation. It does not explain it: partialling out the group leaves
**r = +0.482, p = .0039**, and the correlation holds within each group separately
(+0.52, n = 18, p = .026; +0.47, n = 17, p = .059).

**A nuisance parameter soaking up session differences.** Under m1 there is none —
`mu`, `sd` and `baseline` are pooled across sessions, so Δamplitude is the model's only
per-session degree of freedom. (Under m2 this check would be essential: its Δamplitude
correlates −0.58 with Δbaseline.)

**A slope/intercept artifact.** Δconsistency(risky-second) is uncorrelated with both
ΔP(chose risky) on the same trials (r = −0.05) and Δindifference (r = +0.15).
Partialling both out of the brain–behaviour correlation leaves **r = +0.522, p = .0019**.

**Voxel count.** Uncorrelated with the gain change (r = +0.04) and only weakly with the
behavioural measure (r = −0.24, p = .17); weighting by it changes nothing (§2.1).

**Influence.** No single subject: leave-one-out r ranges +0.50 to +0.56, and the
bootstrap never changes sign in 10 000 resamples.

**Multiplicity — the one that is not fully resolved.** Two numbers, and the honest
answer is between them:

- Over a **focused a-priori family** (2 canonical neural × 3 behavioural read-outs × 2
  orders = 12 pairs, max-|r| permutation over subject labels): **p_FWER = .012**.
- Over the **whole exploratory grid** (500 pairs, same permutation): **p_FWER = .246**.
  Across that grid 26/500 pairs reach p < .05, against 25 expected by chance — i.e. the
  grid as a whole is exactly what a null would look like, with one unusually large
  correlation in it.

The focused family was written down after the grid had been looked at, so its FWER is
optimistic; the grid's is conservative because most of its 500 pairs are near-duplicates
of each other. The result should be reported with the specificity tests (§2.2, §2.3) as
its evidence, not with a single p-value.

**Reliability, the remaining caveat.** The per-subject Δconsistency difference score is
close to unmeasurable. Split-half over trials gives half–half r = −0.06 (Spearman–Brown
→ ≈ 0); a parametric bootstrap says binomial noise alone would generate *more*
between-subject variance (9.7) than is observed (4.1). Both estimates are biased toward
zero (they refit slopes that are themselves noise-inflated) and both are imprecise at
n = 35, but the honest reading is that **an observed r of 0.53 is larger than the
measurement precision comfortably supports**, so the point estimate is very likely
inflated by the winner's curse even if the effect is real. The bootstrap lower bound
(+0.37) is the number to plan around, not +0.53.

---

## 4. An independent, within-subject test

Across-subject difference scores can only ever be as clean as the session pairing. The
trial-level version is immune to it: within a session, are the trials on which the
first option's numerosity was decoded more precisely also the trials on which the choice
was more consistent? Decoding quality is residualised on log(n1) first, since decoder
error grows with distance from the grid centre.

Per-subject logistic `chose_risky ~ log(ratio) * decoding_quality`, then a t-test on the
coefficients across the 35 subjects (stimulation site, quality = −mean|log E − log n1|):

| coefficient | meaning | risky second | risky first |
|---|---|---|---|
| `lr × q` | better decoding → **steeper** psychometric | **+0.42, t(34) = +2.59, p = .014** | +0.52, t = +2.48, p = .018 |
| `q` | better decoding → chose risky | −0.28, t = −2.26, p = .031 | −0.45, t = −2.10, p = .043 |

So the coupling is there trial by trial, and in the same direction: a better parietal
representation of the first option goes with a more payoff-driven, less risk-seeking
choice. Using posterior width instead of decoding error it is weaker (p = .075,
Wilcoxon .046 for risky-second) and absent for risky-first.

Two honest caveats. (a) Unlike the across-subject result this one is **not**
order-specific, so it may reflect a general trial-to-trial engagement factor (attentive
trials → better decoding *and* more consistent choices) rather than the specific
mechanism. (b) The cTBS × coupling interaction is only marginal (Δ`lr × q` = +1.12,
t(34) = +1.91, p = .064).

---

## 5. Why the earlier brain–behaviour analyses came out null

They correlated the neural change with the **PMC per-subject Δν** or with **ΔP(chose
risky)**. Under m1 both are null against the gain change (r = +0.09 and −0.02). The
reason is visible in the behavioural data alone:

| | Δ P(chose risky) | Δ consistency (risky second) |
|---|---|---|
| Δν_perceptual(7), flexible2nf | **r = +0.73**, p < .0001 | −0.15, p = .39 |
| Δν_perceptual(14), flexible2nf | **r = +0.80**, p < .0001 | +0.19, p = .28 |
| Δν_perceptual(7), flexible1nf | **r = +0.68**, p < .0001 | −0.02, p = .93 |

The model's per-subject noise increase is very nearly a re-expression of that subject's
**risk-attitude shift**, which is what §2.3 of the handoff already implied: in this
architecture noise reaches choice through prior attraction (bias), not through
psychometric flattening. So Δν indexes the *bias* channel.

Choice consistency indexes the other channel — added randomness — and **that** is the
one the parietal gain change tracks. The two behavioural channels are essentially
orthogonal here (Δconsistency vs ΔP(chose risky) on risky-second trials: r = −0.05).

This is a substantive result, not a technicality: the neural gain loss predicts the
*noise* channel of the behavioural effect, while the model's ν parameter has absorbed
the *bias* channel.

---

## 6. What did **not** show a link

- **Decoding accuracy** (the paper's Fig-2 measure) across subjects: Δcorr(E, n1)
  against Δconsistency r = +0.15, against ΔP(chose risky) r = −0.07, against Δν
  r = −0.15 to −0.24 (p ≈ .08–.19, the strongest of a null set). The only decoding cell
  reaching p < .05 in the focused family was Δaccuracy × Δindifference(risky-second),
  r = +0.43, p = .011, **p_FWER = .11** — suggestive, not established. Placebo contrasts
  built from session 1 are null too, but so is the real contrast, so that test is
  uninformative here. Splitting decoding by presented numerosity does not help: the best
  such cell in the whole grid is Δ posterior width (low minus high n1) ×
  Δindifference(risky-second), r = −0.40, p = .019, p_FWER = .90.
- **Magnitude specificity of the gain change.** The tuning-weighted profile gives
  r = +0.45 (n = 7), +0.52 (14), +0.40 (28) against Δconsistency, but those measures
  correlate > .9 with each other and the 7-minus-28 localisation slope is flat
  (r = +0.19, p = .27). Splitting voxels by preferred numerosity (≤ 14 vs > 14), neither
  band survives the other (partial r = +0.21 and +0.23). **There is no evidence that the
  link is carried by voxels tuned to a particular part of the payoff range** — which is
  the fourth independent route to fail on that question (`notes/reanalysis_handoff.md`
  §7.5 lists the other three).

  Restricting the amplitude measure to the band where the **group** gain loss is
  concentrated makes things strictly worse, so the all-voxel average is the right
  estimator. Group Δamp by preferred numerosity (per-subject medians, then across
  subjects) peaks at 7–10 in the stimulation site (−0.193, t = −1.60) and 7–14 in NPC12r
  (−0.081/−0.083), and turns positive above 20 in the stimulation site. Against
  Δconsistency(risky-second):

  | amplitude measure | NPCr2cm-cluster | NPC12r |
  |---|---|---|
  | **all voxels (used above)** | **+0.533**, p = .0010 | **+0.480**, p = .0035 |
  | voxels in the peak-group-loss band, leave-one-subject-out selected | +0.197, p = .26 | +0.356, p = .036 |
  | bands weighted by group loss, leave-one-subject-out | +0.276, p = .11 | +0.319, p = .062 |
  | fixed band, preferring < 7 | +0.304 (n = 32) | +0.441 (n = 34) |
  | fixed band, preferring 7–10 | +0.147 (n = 32) | +0.118 (n = 34) |
  | fixed band, preferring 10–14 | +0.493 (n = 28) | +0.221 (n = 31) |
  | fixed band, preferring 14–20 | +0.203 (n = 29) | +0.309 (n = 30) |

  Two things follow. (a) The band that carries the **group-mean** gain loss (7–10) is
  *not* the band that carries the **individual differences** (r = +0.15 / +0.12 there) —
  the two questions have different answers. (b) The best fixed band disagrees between
  masks (10–14 in the stimulation site, < 7 in NPC12r) and each band-restricted estimate
  loses voxels and subjects, so the gain is noise. Averaging over the whole population is
  both the better estimator and the honest one.
- **Composite disruption indices** (z-averaged gain loss + decoding loss vs z-averaged
  risk-seeking + consistency loss): r = −0.01. Averaging in the null measures destroys
  it, as it should.

---

## 7. Recommendation

Reportable, with its caveats stated:

> Individual differences in the neural effect of cTBS predicted individual differences
> in the behavioural effect: subjects with a larger reduction in nPRF response gain at
> the stimulation site showed a larger reduction in choice consistency, specifically on
> the trials where the safe option was presented first (r(33) = .53, p = .001). The
> relationship was absent in left parietal, frontal and occipito-temporal cortex, and
> absent for the opposite presentation order.

State alongside it: (a) the effect size is likely inflated — plan around the bootstrap
lower bound of ≈ .37; (b) the family-wise-corrected p depends on how the search family
is drawn (.012 focused / .25 exhaustive); (c) the link is to the *consistency* channel,
not to the PMC ν or to ΔP(chose risky). Do **not** claim magnitude localisation from it.

The trial-level analysis (§4) is worth a supplementary paragraph as convergent evidence
that does not depend on session pairing.

## Files

| Output | Script |
|---|---|
| `notes/data/bb_neural.tsv`, `bb_decoding*.tsv`, `bb_behavior.tsv` | `modeling/scripts/extract_brain_behavior_table.py` |
| `notes/data/bb_link_{master,grid,focused,robustness,composite}.tsv` | `behavior/scripts/analyze_brain_behavior_link.py` |
| `notes/data/bb_link_stability.tsv` | `behavior/scripts/check_brain_behavior_robustness.py` |
| `notes/data/bb_trialwise_*.tsv` | `behavior/scripts/trialwise_decoding_choice_link.py` |
| `notes/figures/brain_behavior_link.pdf` | `behavior/scripts/plot_brain_behavior_link.py` |
