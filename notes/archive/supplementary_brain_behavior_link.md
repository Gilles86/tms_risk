# Supplementary Note — robustness of the neural–behavioural correlation

Draft supplementary text for the correlation between the cTBS-induced change in nPRF
response gain and the cTBS-induced change in choice consistency. Numbers verified
2026-08-03; provenance in `notes/brain_behavior_link.md` and `notes/PROVENANCE.md`.

---

## Supplementary Note S<N>. Robustness of the relationship between the neural and behavioural cTBS effects

**Measures.** The neural measure is the per-participant difference (parietal − vertex)
in the median nPRF response gain across voxels of the individually defined stimulation
site with cross-validated R² > 0. Gain was taken from the encoding model in which
response amplitude is the only session-specific parameter, so that preferred numerosity,
tuning width and baseline are shared across sessions by construction and the contrast
isolates a change in response gain at fixed tuning. The behavioural measure is the
per-participant difference (parietal − vertex) in the slope of a logistic psychometric
function relating choice to log(risky/safe payoff ratio), estimated separately for each
session. Because the group-level effect of cTBS on choice consistency is itself confined
to trials on which the safe option was presented first (hierarchical probit, change in
psychometric slope −0.375, 95% CrI [−0.695, −0.047], *P*(< 0) = .99, versus −0.025,
CrI [−0.361, +0.310] when the risky option was presented first; risk-neutral probability
+0.055, CrI [+0.023, +0.089] versus +0.007, CrI [−0.026, +0.039]), the psychometric
slope was estimated on those trials, with the opposite presentation order retained
throughout as a control. The subset was therefore fixed by a result that precedes this
analysis. All analyses use the 35 participants with usable nPRF fits in both stimulation
sessions.

**Primary result.** The two differences were positively related, *r*(33) = .53,
*p* = .001 (Spearman ρ = .59, *p* < .001; bootstrap 95% CI [.37, .67]; Theil–Sen slope
4.08 [1.94, 6.92]).

**Specificity to the stimulated tissue.** Repeating the analysis with gain measured in
progressively more distant cortex reduced the correlation monotonically. Comparisons of
dependent correlations (Williams's test) against the stimulation site are given in the
last column.

| Region | *r* | *p* | vs. stimulation site |
|---|---|---|---|
| Stimulation site | .53 | .001 | — |
| Right parietal numerosity ROI | .48 | .004 | *p* = .33 |
| Left parietal numerosity ROI | .21 | .22 | *p* = .020 |
| Frontal numerosity ROI | .17 | .32 | *p* = .031 |
| Occipito-temporal numerosity ROI | .02 | .92 | *p* = .008 |

The right parietal ROI largely contains the stimulation site and is therefore not an
independent step in this progression; the three remaining regions each differ
significantly from the stimulation site.

**Specificity to presentation order.** The correlation was .53 when the safe option was
presented first and −.04 when the risky option was presented first (difference between
dependent correlations, *t*(32) = 2.82, *p* = .008). Regressing the risky-first
difference score out of the safe-first one left the correlation unchanged, *r* = .54,
*p* = .001, and the correlation with the order-difference score (safe-first minus
risky-first) was itself significant, *r* = .39, *p* = .019. Pooled across both
presentation orders the correlation was *r* = .28, *p* = .10.

**Construction of the neural measure.** We repeated the analysis for every combination
of two masks (the stimulation site and the right parietal numerosity ROI), three voxel
selections (cross-validated R² > 0; each participant's 100 best-fitting voxels, which
equates voxel counts across participants; and all voxels with non-degenerate fits) and
three summary statistics (median, mean, and gain change expressed relative to mean gain
across the two sessions). The correlation was positive in all 18 combinations and
significant in 17 (range .27 to .53). The same 18 analyses applied to the risky-first
control yielded 9 positive correlations and none significant (range −.16 to .16).
Restricting to participants with at least 20 (*n* = 26) or 50 (*n* = 16) surviving
voxels gave *r* = .51 and *r* = .55; weighting participants by voxel count gave
*r* = .54. Voxel count was unrelated to the neural difference score (*r* = .04).

**Construction of the behavioural measure.** Replacing the psychometric slope with two
bounded alternatives computed on the same trials — the rank correlation between choice
and log payoff ratio, and its point-biserial equivalent — gave *r* = .48 (*p* = .004)
and *r* = .47 (*p* = .004). Two measures of psychometric *fit* rather than steepness
(classification accuracy and log-loss under each participant's own fitted function) were
unrelated to the neural measure (*r* = .08 and −.10), indicating that what tracks the
gain change is the steepness of the dependence of choice on payoff ratio, not overall
model fit.

**Influential observations.** Leaving each participant out in turn gave correlations
between .50 and .56, and none of 10,000 bootstrap resamples changed the sign.

**Session order.** Eighteen participants received parietal cTBS in the first
stimulation session and 17 in the second. Both difference scores differed between these
groups in the same direction, which could in principle inflate their correlation.
Partialling out counterbalancing group left *r* = .48, *p* = .004, and the correlation
held within each group separately (*r* = .52, *n* = 18, *p* = .026; *r* = .47, *n* = 17,
*p* = .059).

**Independence from the change in choice proportions.** The change in psychometric slope
was uncorrelated with the change in the proportion of risky choices on the same trials
(*r* = −.05) and with the change in the indifference point (*r* = .15). Partialling out
the change in risky-choice proportion left the neural–behavioural correlation at
*r* = .53, *p* = .001; partialling out both choice-level measures left *r* = .52,
*p* = .002.

**Preferred numerosity of the affected voxels.** The group-level gain reduction was
largest in voxels preferring numerosities of roughly 7–14. Restricting the neural
measure to that band did not strengthen the relationship: selecting the
largest-group-reduction band by leave-one-participant-out gave *r* = .20 (*p* = .26) at
the stimulation site and *r* = .36 (*p* = .036) in the right parietal ROI, and weighting
bands by their group-level reduction gave *r* = .28 and *r* = .32. The relationship
therefore reflects a population-wide change in response gain rather than a change
confined to voxels tuned to a particular part of the payoff range.

**Multiple comparisons.** The correlation reported here was identified within a larger
set of candidate neural and behavioural difference measures, and its significance is
stated relative to two families. Within an a-priori family of 12 tests (two canonical
neural measures — gain change and decoding accuracy — crossed with three behavioural
read-outs and the two presentation orders), the family-wise error rate controlled by a
maximum-|*r*| permutation over participant labels gives *p* = .012. Across the full
exploratory set of 500 pairs the same procedure gives *p* = .25; in that full set 26 of
500 pairs reached *p* < .05, against 25 expected under the null. The evidence for this
relationship therefore rests on its anatomical and order specificity and its robustness
to measurement choices rather than on its nominal *p* value alone.

**Reliability of the behavioural difference score.** Split-half estimates of the
reliability of the change in psychometric slope are close to zero, and a parametric
bootstrap indicates that binomial choice noise alone can account for the observed
between-participant variance. Both estimates are biased downwards, but they imply that
the point estimate of *r* = .53 is likely inflated and that the lower bound of its
bootstrap interval (.37) is the more appropriate basis for judging effect size.

**Convergent within-participant evidence.** Because the analysis above compares two
sessions, we also tested the relationship within sessions, where session pairing plays
no role. For each participant we fitted a logistic model in which the psychometric slope
was allowed to vary with the trial-by-trial accuracy of the decoded numerosity of the
first-presented option (residualised on the presented numerosity). Trials on which the
first option was decoded more accurately showed steeper psychometric functions
(β = 0.42, *t*(34) = 2.59, *p* = .014 for safe-first trials; β = 0.52, *t*(34) = 2.48,
*p* = .018 for risky-first trials) and fewer risky choices (β = −0.28, *p* = .031 and
β = −0.45, *p* = .043). Unlike the between-participant relationship this within-session
coupling was not specific to presentation order, and may therefore partly reflect
trial-to-trial fluctuations in engagement common to both measures.
