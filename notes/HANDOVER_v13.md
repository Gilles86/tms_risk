# Revision handover for TMS_paper_v13_draft_REDLINE.docx

Everything Claude chat needs to produce v14. **No token is left unresolved** —
§2 gives a value for every one of the 24 in the draft. §1 lists the figures to
upload. §3–§6 are the text changes.

---

## 1. Figures to upload

**Nine files.** All are in `/Users/gdehol/git/tms_risk/notes/figures/`. Upload
the **PDF** of each (vector, editable text); the `.png` alongside is only for
quick viewing.

### Main figures

| # | File | Change since v13 |
|---|---|---|
| 1 | *(hand-made schematic — unchanged, keep the one in the .docx)* | — |
| 2 | `figure2_new.pdf` | **replace.** Font scale brought into line with the rest of the paper (8 / 8.5 / 9 / 8 pt) and panel letters to 9 pt. Note the filename — `figure2.pdf` is the older render, do not use it. |
| 3 | `fig3_probit.pdf` | **replace.** Same three panels, restyled: square IPS markers are now circles, width 6.24″ → 7.25″ to match the other figures, titles at regular weight, panel letters 9 pt. A `sns.set_context('paper')` call was silently overriding the figure's own settings and printing it ~13% heavier than intended; removed. |
| 4 | `fig4_stage.pdf` | **replace.** Panel d now shows **11** flexible noise forms instead of 2 — the earlier render predated the baseline spline fits and silently drew only what existed. Weber's reference line and "High stake" changed from red to orange, so red means cTBS-to-IPS everywhere in the paper. |
| 5 | `FIG5_percpmu.pdf` | **replace.** Font scale raised one step (7.5 / 8 / 8.5 / 7 — it cannot carry the full common scale, see below); "Memo"/"Perc" spelled out; "vertex" capitalised in three axis labels. |
| **6** | `fig6_brain_behavior.pdf` | **NEW.** See §5. |

### Supplementary figures

| # | File | Change |
|---|---|---|
| S1 | `SUPP_S1_probit_by_stake.pdf` | **replace.** Difference densities replaced by Figure 3's paired-point grammar; refitted with random intercepts only, matching the published model. |
| S2 | `SUPP_S2_model_comparison.pdf` | **NEW** |
| S3 | `SUPP_S3_ppc_design_grid.pdf` | **NEW** |
| S4 | `SUPP_S4_noise_flexibility.pdf` | **NEW** |

**Do not upload** `SUPP_Sx_noise_flexibility_ctbs.pdf` — reviewer-response
material, deliberately unnumbered.

---

## 2. Token values — substitute all of these

| Token | Value |
|---|---|
| ⟦DNU7⟧ | 0.029 log units |
| ⟦DNU7_CRI⟧ | [−0.003, +0.061] |
| ⟦DNU7_P⟧ | 0.963 |
| ⟦CRED_UPPER⟧ | 19 CHF |
| ⟦DNU_HIGH⟧ | +0.001 log units [−0.058, +0.054], P(Δν > 0) = 0.52 |
| ⟦SAFE_PMU⟧ | 14% (−0.154 log units, 10.7 → 9.2 CHF) |
| ⟦SAFE_PMU_P⟧ | 0.944 |
| ⟦RISKY_PMU⟧ | +0.036 log units [−0.134, +0.205], P(Δμ < 0) = 0.34 |
| ⟦PERC_RATIO⟧ | 2.25 |
| ⟦PERC_RATIO_CRI⟧ | 0.160 [0.132, 0.192] → 0.361 [0.312, 0.419] log units |
| ⟦SPLINE_ORDERS_BASELINE⟧ | 3, 4, 5, 6 and 7 anchors |
| ⟦SPLINE_ORDERS_CTBS⟧ | 3, 4, 5, 6 and 7 anchors |
| ⟦PPC_N_TOTAL⟧ | 34 |
| ⟦PPC_N_PASS⟧ | 32 |
| ⟦PPC_ORDER⟧ | ppp = 0.05 |
| ⟦PPC_SLOPE_ORDER⟧ | ppp = 0.88 |
| ⟦PPC_THREEWAY⟧ | ppp = 0.72 |
| ⟦PPC_FAIL⟧ | +0.024 [−0.004, +0.056] |
| ⟦PPC_FRACTION⟧ | 0.46 |
| ⟦PPC_R⟧ | 0.91 |
| ⟦PPC_CELLS⟧ | 420 |
| ⟦PPC_COVERAGE⟧ | 96% |
| ⟦BB_INTERVAL⟧ | **delete the phrase** — see §4c |
| ⟦N1N2_RANGE⟧ | **retired** — see §3 |

Two notes on these. **⟦PPC_FRACTION⟧ = 0.46**, so "about a factor of two" and
"by about half" in ¶82 and Discussion ¶3 are both right as written — keep them.
And **⟦PPC_N_TOTAL⟧ / ⟦PPC_N_PASS⟧ are now 34 and 32, not 8 and 8**: the paper
should stop counting the eight targeted checks and count the design grid
instead. See §4a — this is a framing change, not a substitution.

---

## 3. Cuts: the position-indexed and prior-width material

Four passages go. Two are surgical — a neighbouring clause must survive.

**¶71.** Delete only the empirical clause and its figure pointer:
> ~~In this dataset it is not well constrained, and a model free to change both
> leaves neither credible (P = 0.377 … Supplementary Fig. 6).~~

Keep the analytic argument before it (noise and prior width both act through
the shrinkage weight) and the neural justification after it. Adjust the
connective so the paragraph runs into *"We therefore held the prior widths
fixed…"*.

**¶76.** Delete the first half of the last sentence, keep the second:
> ~~Neither conclusion depends on how the noise is indexed, as position-indexed
> models … (Supplementary Fig. 3), nor on~~ **Neither conclusion depends on**
> the memory component's flexibility, since constraining it to Weber's law
> costs 1.6 ELPD (dSE 2.7) and changes no conclusion **(Supplementary Fig. 2c)**.

**Methods, spline paragraph.** Delete the final sentence beginning *"The same
spline form was used for the position-indexed robustness models…"*.

**Methods, identifiability paragraph.** Delete entirely — it exists only to
explain why the position-indexed fits misbehave. Begins *"At the two-parameter
power form, the position-indexed parameterisation…"*.

---

## 4. Substantive text changes

### 4a. ¶78: STOP COUNTING EIGHT CHECKS — the criterion is now the design grid

The paragraph is built around eight targeted posterior predictive checks.
**Eleven of the twenty fitted models pass all eight**, including the model in
which cTBS changes nothing but the magnitude priors, so the count distinguishes
nothing. ¶75 already concedes this (*"which the priors-only model also
passes"*), leaving ¶78 in contradiction with it.

**Do not simply substitute new numbers into the old sentence.** Drop the
eight-check framing and report the design grid, which has 34 cells rather than
8 and does order the models. The eight targeted statistics can stay as a
sentence of reassurance, but they must not carry a claim.

Replacement substance, wording yours:

> We assessed fit against the design's own cells (Supplementary Fig. 3): the 34
> IPS − vertex contrasts defined by five safe payoffs × two presentation orders
> × two stimulation arms, read four ways — by safe payoff, by risky/safe ratio,
> by stake, and as the psychometric slope — each computed per posterior draw.
> The reported model covers 32 of 34. Coverage falls as the mechanism is
> removed: 30 without the prior shift, 29 with perceptual noise alone, 27 with
> no cTBS effect. Coverage of the 68 underlying levels is lower (57) and does
> not order the models at all, so it is the model comparison rather than any
> predictive check that carries the mechanistic claim. The model also
> reproduces every one of eight targeted summary statistics, though so do ten
> of the nineteen alternatives. At the participant level it stays calibrated,
> correlating at r = 0.91 with observed proportions across 420 cells and
> covering 96% of them (Fig. 5h,i).

Anywhere else in the manuscript that says "eight posterior predictive checks"
or similar should be changed the same way.

### 4b. ¶76's ⟦DNU7⟧ is DIRECTIONAL — the wording must match

The value is **+0.029 log units, [−0.003, +0.061], P(Δν > 0) = 0.963**. The
two-sided interval marginally includes zero. Report the posterior probability
as the claim; do not present the interval as excluding zero. Supplementary
Fig. 2c marks credible payoffs by the same directional criterion and says so on
the figure, so the two will agree.

### 4c. ¶81 — keep the probit correlation, delete the interval token

Leave *"r(33) = 0.53 … p = 0.001; Spearman ρ = 0.59"* exactly as written, and
delete the phrase *"95% posterior interval ⟦BB_INTERVAL⟧"*. The probit analysis
gives no such interval. **Do not** substitute the cognitive-model correlation
(−0.195) — that is amplitude against the PMC noise parameter, a different
quantity.

### 4d. NEW paragraph: why the correlation uses probit estimates, not model parameters

Add after ¶81. This is a real limitation and a referee will ask.

> The individual-difference analysis uses the probit estimate of choice
> consistency rather than the corresponding parameter of the cognitive model,
> because the model's per-participant parameters are not estimated precisely
> enough to support a correlation. Assessed by the correlation between two
> independent posterior draws of the whole participant vector, their
> reliabilities are 0.40 for perceptual noise at 7 CHF, 0.53 at 112 CHF, 0.39
> for the risky prior mean and 0.04 for the safe prior mean. The corresponding
> attenuation ceilings — the largest correlation any of them could show with a
> perfectly measured external variable — are 0.61, 0.71, 0.61 and 0.20. The
> model-based versions of the correlation reported above are correspondingly
> weak (r = +0.30 at 7 CHF, −0.27 at 112 CHF), as expected under that
> attenuation rather than as evidence against the link. The model is fitted to
> raw choices with partial pooling and is designed to identify a group-level
> mechanism; individual parameters are a by-product, and we do not interpret
> them.

### 4e. Add the two missing supplementary citations

* **¶75**, after *"an edge in coverage of the design grid (32 of 34 cells …)"*
  → append **"(Supplementary Fig. 3)"**.
* **Figure 4 caption panel d**, and ¶70's *"the spline fits recover the same
  shapes (Fig. 4c,d)"* → append **"; see Supplementary Fig. 4"**.

### 4f. Figure 4 caption: the form count changed

Panel d now shows **eleven** flexible forms, not two. Update the caption to
say so, and note they span 3–7 anchors in both a piecewise-linear and a
natural-cubic basis.

---

## 5. Figure 6 — new, and what still needs checking

`fig6_brain_behavior.pdf`. **a)** Δ nPRF gain × Δ choice consistency across
participants, safe-first trials, r(33) = 0.53. **b)** the same correlation in
five ROIs and both presentation orders, showing it falls off with distance from
the coil and is absent when the risky option came first.

Draft caption:

> **Figure 6. The neural and behavioural cTBS effects are linked across
> participants, specifically at the stimulated site.** **a)** Change in nPRF
> response gain (IPS − vertex) against change in choice consistency for trials
> on which the safe option was presented first; each point is one participant
> (n = 35), the line the posterior regression. **b)** The same correlation
> computed in five numerosity-tuned ROIs, for trials on which the safe option
> came first (dark) and the risky option came first (light). Bars are 95%
> credible intervals.

**ROI provenance — add to Methods.** The five ROIs are fsaverage-space surface
labels (`<bids>/derivatives/surface_masks/desc-*_space-fsaverage_hemi-*.label.gii`),
transformed to each participant's surface with FreeSurfer `SurfaceTransform`
and then to T1w volume (`tms_risk/registration/get_npc_mask.py`). `NPC12r` is
the union of right NPC1 and NPC2; `NPCr2cm-cluster` is the stimulation-site
cluster. Suggested sentence:

> Numerosity-tuned regions of interest were defined as fsaverage-space surface
> labels drawn in Barretto-García et al. (2023), following the nomenclature of
> Harvey et al. (2013), and projected to each participant's cortical surface
> and functional volume. The right parietal region combines NPC1 and NPC2;
> NPC3 was not included, as it lies outside the intraparietal and postcentral
> sulci.

**Panel c is deliberately absent.** The script can also draw a within-subject
panel correlating trial-wise decoding precision with psychometric slope. It is
off by default and should stay off: **no stimulation contrast enters it at
all**, so it is a separate correlational claim rather than part of the cTBS
causal chain this figure makes, and it would invite the reader to ask what it
is doing here. (`--with-trialwise` restores it for a reviewer response, if the
trial-wise decoding table is regenerated.)

## 6. Supplementary captions

### Supplementary Figure 2 — `SUPP_S2_model_comparison.pdf`

> **Where the cTBS effect acts, and how little that depends on the noise
> function.** cTBS cohort, n = 35. **a)** Paired ΔELPD (leave-one-out) against
> the reported model, for models differing only in what cTBS may change; noise
> function held at the power law. Whiskers are ±1 standard error of the paired
> difference. **b)** The estimated change in perceptual noise at the two ends
> of the payoff range under every noise function from Weber (one anchor) to a
> seven-anchor spline, in a piecewise-linear and a natural-cubic basis.
> **c)** The perceptual noise functions themselves after cTBS to IPS (red) and
> vertex (green), one panel per form, with the shared memory component in grey;
> shaded areas are 95% credible intervals. Black bars mark payoffs at which
> P(IPS > vertex) > 0.95. Each panel gives that model's paired ΔELPD (dSE).

### Supplementary Figure 3 — `SUPP_S3_ppc_design_grid.pdf`

> **Posterior predictive checks on the design's own cells.** **a–h)** Observed
> proportion of risky choices (markers) against the reported model's predictive
> median (line) and 95% predictive interval (shaded), after cTBS to IPS (red)
> and vertex (green). Columns are four views of the same choices; rows are
> presentation order. Ringed markers fall outside their interval; the count in
> each panel is that panel's coverage. **i)** Coverage per model on the paired
> IPS − vertex contrast (dark, 34 cells) and on the levels in a–h (pale, 68
> cells). The contrast orders the models monotonically; the levels do not.

### Supplementary Figure 4 — `SUPP_S4_noise_flexibility.pdf`

> **The shape of the baseline noise functions does not depend on the form used
> to estimate it.** Baseline session, n = 73. **a)** Paired ΔELPD against the
> power law, by number of anchors per channel, for a piecewise-linear (blue)
> and natural-cubic (orange) basis. Weber's law (one anchor) is 47.4 ELPD worse
> (dSE 11.2). **b, c)** Perceptual and memory noise functions implied by all
> thirteen fitted forms, coloured by anchor count, with the power law's own 95%
> credible interval shaded behind.

### Supplementary Figure 1 — caption update

The existing caption still describes difference densities. Panel B now shows
the two conditions as levels with 95% credible intervals, joined by a
connector, with Δ and the posterior probability annotated. Add: *"Fitted with
random intercepts only, as in de Hollander et al. (2024a); the connector is
dark where the 95% interval on the difference excludes zero."*
