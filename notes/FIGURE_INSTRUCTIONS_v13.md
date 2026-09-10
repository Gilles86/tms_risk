# Figures for TMS_paper_v13_draft_REDLINE.docx — what to insert where

Audit of `notes/paper/TMS_paper_v13_draft_REDLINE.docx`, 2026-09-10. Read this
alongside `notes/PLACEHOLDERS.md`, which holds the numbers.

**The state of play.** The .docx embeds **six images**: Figures 1–5 and
Supplementary Figure 1. The text also cites Supplementary Figs. 2, 3 and 6,
which do not exist in the file — they are forward references.

**Decided 2026-09-10: the position-indexed (n1/n2) fits and the prior-width
fits are cut from the manuscript entirely.** The position-indexed models do
not sample reliably below four anchors and, whatever the technical
explanation, they cost the reader more than they return. The prior widths are
not fitted for a reason that is analytic rather than empirical — noise and
prior width both act through the same shrinkage weight, so they are not
separately identified — and a failed recovery adds nothing to an argument that
already holds a priori.

That removes what would have been Supplementary Figs. 3 and 6, so the
supplement is **four figures**, renumbered with no gaps. Section 3e lists every
sentence to delete.

---

## 1. The supplementary set

| # | File in `notes/figures/` | Status | Cited at |
|---|---|---|---|
| **S1** | `SUPP_S1_probit_by_stake.pdf` | **already in the docx** as an image; the file is now regenerated under a consistent name — swap it in so the caption and the file agree | caption only |
| **S2** | `SUPP_S2_model_comparison.pdf` | **ready — insert** | ¶75 |
| **S3** | `SUPP_S3_ppc_design_grid.pdf` | **ready — insert, and add citations** (see 3a) | not yet cited |
| **S4** | `SUPP_S4_noise_flexibility.pdf` | **ready — insert, and add a citation** (see 3a) | not yet cited |

Not for the supplement: `SUPP_Sx_noise_flexibility_ctbs.pdf` is S5's panels on
the cTBS cohort rather than the baseline. Keep it for a reviewer response; do
not number it. Having both numbered invites the cohort confusion S5's caption
exists to prevent.

---

## 2. Captions for the new figures

### Supplementary Figure 2 — `SUPP_S2_model_comparison.pdf`

> **Supplementary Figure 2. Where the cTBS effect acts, and how little that
> depends on the noise function.** Fits to the cTBS cohort (n = 35).
> **a)** Paired difference in expected log predictive density (ELPD,
> leave-one-out) against the reported model, for models differing only in what
> cTBS is allowed to change. The noise function is the two-parameter power law
> throughout, so the rungs differ only in the placement of the effect. Bars are
> the paired difference, whiskers ±1 standard error of that difference.
> **b)** The estimated change in perceptual noise at the two ends of the payoff
> range, refitted under every noise function from Weber (one anchor) to a
> seven-anchor spline, in both a piecewise-linear and a natural-cubic
> ("smooth") basis. Markers are posterior medians, lines 95% credible
> intervals. **c)** The perceptual noise functions themselves after cTBS to IPS
> (red) and vertex (green), one panel per noise function, with the shared
> memory component in grey; shaded areas are 95% credible intervals. Black bars
> mark payoffs at which P(IPS > vertex) exceeds 0.95. Each panel gives that
> model's paired ΔELPD against the reported one, with dSE in brackets.

### Supplementary Figure 3 — `SUPP_S3_ppc_design_grid.pdf`

> **Supplementary Figure 3. Posterior predictive checks on the design's own
> cells.** **a–h)** Observed proportion of risky choices (markers) against the
> reported model's posterior predictive median (line) and 95% predictive
> interval (shaded), after cTBS to IPS (red) and vertex (green). Columns are
> four views of the same choices: by safe payoff, by risky/safe ratio, by stake
> tercile, and the psychometric slope by stake. Rows are presentation order.
> Ringed markers fall outside their own interval; the count in each panel is
> how many of that panel's cells are covered. **i)** Coverage per model, on the
> paired IPS − vertex contrast (dark, 34 cells) and on the levels shown in a–h
> (pale, 68 cells). The contrast orders the model set monotonically; the levels
> do not, and the priors-only model leads on them.

### Supplementary Figure 4 — `SUPP_S4_noise_flexibility.pdf`

> **Supplementary Figure 4. The shape of the baseline noise functions does not
> depend on the form used to estimate it.** Pre-stimulation baseline session,
> n = 73. **a)** Paired ΔELPD against the two-parameter power law, against the
> number of anchors per noise channel, for a piecewise-linear (blue) and a
> natural-cubic (orange) basis; shaded areas are ±1 standard error of the
> paired difference. Weber's law (one anchor) is 47.4 ELPD worse (dSE 11.2,
> 4.2 standard errors). **b, c)** The perceptual and memory noise functions
> implied by every fitted form, coloured by number of anchors, with the power
> law's own 95% credible interval shaded behind. The flexible fits lie inside
> that interval almost everywhere; Weber's does not.

---

## 3. Text changes needed

### 3a. Add the two missing citations

Neither S3 nor S4 is cited anywhere. Both need one:

* **¶75**, the sentence ending *"an edge in coverage of the design grid (32 of
  34 cells, and 6 of 6 against the priors-only model's 5 of 6 on the
  psychometric-slope cells, the view in which noise rather than bias should
  show)"* → append **"(Supplementary Fig. 3)"**.
* **¶78**, the posterior-predictive paragraph → add **"(Supplementary Fig. 3)"**
  where the design grid is introduced. See 3b, which rewrites this paragraph
  anyway.
* **Figure 4 caption, panel d**, and the Results sentence at ¶70 ending *"the
  spline fits recover the same shapes (Fig. 4c,d)"* → append **"; see
  Supplementary Fig. 4"**.

### 3b. ¶78 must be rewritten — the current claim is false

The paragraph rests on the eight targeted posterior predictive checks. **Eleven
of the twenty fitted models pass all eight**, including the model in which cTBS
changes nothing but the magnitude priors. A criterion half the model set passes
cannot single anything out, and ¶75 already concedes this in passing
(*"not the targeted posterior predictive checks, which the priors-only model
also passes"*), which leaves ¶78 contradicting it.

Keep the targeted checks as a report of fit; move the discriminating work onto
the design grid and ELPD. Suggested substance, wording yours:

> The reported model reproduces all eight targeted statistics, but so do ten of
> the nineteen alternatives, so we also assessed fit against the design's own
> cells (Supplementary Fig. 3): the 34 IPS − vertex contrasts defined by five
> safe payoffs × two presentation orders × two stimulation arms, read four
> ways, each computed per posterior draw. The reported model covers 32 of 34,
> and coverage falls as the mechanism is removed — 30 without the prior shift,
> 29 with perceptual noise alone, 27 with no cTBS effect. Coverage of the
> levels behind those contrasts is lower (57 of 68) and does not separate the
> models, so the model comparison rather than any predictive check carries the
> mechanistic claim.

### 3c. ¶76's ⟦DNU7⟧ is directional — say so

The draft has *"at 7 CHF the noise SD rose by ⟦DNU7⟧ (95% CrI ⟦DNU7_CRI⟧,
P(Δν > 0) = ⟦DNU7_P⟧)"*. The current fit gives **+0.029 log units,
[−0.003, +0.061], P(Δν > 0) = 0.963**. The two-sided interval marginally
includes zero. Either quote the posterior probability alone and drop the
interval, or quote both and let the interval show what it shows — but do not
present it as excluding zero. S2's significance bars use the same directional
criterion and the figure says so, so the two will agree.

### 3d. ¶81 — keep the probit correlation, drop the interval token

**Decided 2026-09-10: leave ¶81's correlation exactly as it is.** It reports
r(33) = 0.53 between the cTBS-induced change in nPRF amplitude and the change
in probit-estimated choice consistency, which is the quantity the paragraph is
about.

Do NOT substitute the cognitive-model correlation from PLACEHOLDERS Round 4
(−0.195 at the 112 CHF anchor). That is a different quantity — amplitude
against the PMC model's perceptual-noise parameter, not against probit
consistency — and swapping it in would silently change what the paragraph
claims.

⟦BB_INTERVAL⟧ has no value from the probit analysis. Delete the phrase
*"95% posterior interval ⟦BB_INTERVAL⟧"* and report r and p only.

### 3e. Spline-order tokens

The draft correctly uses two tokens. Both resolve once the last fits land:

* ⟦SPLINE_ORDERS_BASELINE⟧ — Figure 4d and ¶70, the n = 73 baseline set.
* ⟦SPLINE_ORDERS_CTBS⟧ — Methods, the n = 35 cTBS set.

They are different sets. Do not collapse them into one token.

---

### 3e. Delete the position-indexed and prior-width material

Four passages go. Each is surgical — in two of them a neighbouring clause must
survive, so do not delete whole sentences without reading them.

**1. ¶71, the prior-width paragraph.** Delete only the empirical clause and its
figure pointer:

> ~~In this dataset it is not well constrained, and a model free to change both
> leaves neither credible (P = 0.377 for the risky prior width and P = 0.187
> for the safe, in a pathfinder-initialised fit; Supplementary Fig. 6).~~

**Keep everything around it.** The paragraph's argument is analytic — noise and
prior width both act through the shrinkage weight σ_p²/(σ_p² + ν²), so a wider
prior and a lower noise level move it the same way — and it is followed by the
neural justification (cTBS lowered amplitude and left preferred numerosity
intact). Both stand on their own. Adjust the connective so the paragraph reads
straight into *"We therefore held the prior widths fixed…"*.

**2. ¶76.** Delete the first half of the final sentence, keep the second:

> ~~Neither conclusion depends on how the noise is indexed, as position-indexed
> models give an increase of roughly ⟦N1N2_RANGE⟧ at low payoffs on both
> options with no credible difference between them (Supplementary Fig. 3),
> nor on~~ **Neither conclusion depends on** the memory component's
> flexibility, since constraining it to Weber's law costs 1.6 ELPD (dSE 2.7)
> and changes no conclusion.

The Weber-memory robustness claim is worth keeping — it is now shown in S2c,
so add **"(Supplementary Fig. 2c)"** to it.

**3. Methods, the spline paragraph.** Delete the final sentence:

> ~~The same spline form was used for the position-indexed robustness models,
> in which the first- and second-presented option carry separate noise
> functions instead of a shared perceptual and an added memory component
> (Supplementary Fig. 3).~~

**4. Methods, the identifiability paragraph.** Delete it entirely — it exists
only to explain why the position-indexed fits misbehave, and with those fits
gone it explains nothing the reader needs. It begins *"At the two-parameter
power form, the position-indexed parameterisation (separate noise functions
for the first- and second-presented option) does not sample reliably…"* and
ends *"…give the same answer (Supplementary Fig. 3)."*

**Tokens retired by these cuts** — remove them from the draft and ignore any
value for them in PLACEHOLDERS: ⟦N1N2_RANGE⟧, ⟦SFIG_N1N2⟧, ⟦PSD_RISKY_P⟧,
⟦PSD_SAFE_P⟧.

---

## 4. Nothing is blocked

Every figure the manuscript now cites exists and is rendered. `SUPP_Sx_noise_
flexibility_ctbs.pdf` remains unnumbered, for a reviewer response only.
