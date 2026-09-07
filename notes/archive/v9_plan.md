# v9 audit: what still needs fixing, and the discussion outline for Christian

Audited 2026-08-19 against `notes/paper/TMS_paper_v9.pdf`, cross-checked against
`notes/PROVENANCE.md`, `notes/v8_stats_check.md`, `notes/checks_20260803.md`, and the
TSVs in `notes/data/`.

**Headline: v9 is in far better shape than v8.** Every heavy analysis it reports is the
current, reproducible one — Fig 3 (probit reanalysis), Fig 4 (Weber vs Flexible + 16-model
ELPD ladder), Fig 5 (mechanism decomposition), Supp Table 1 (`table1_all16.tsv`), and the
brain–behaviour link (r = 0.53) all match the TSVs of record. All four v8-audit errors
(the MAP p = 0.004, the swapped order labels, r(34)→r(33), the inverted mu/sd
descriptives) are fixed. **No refits are needed.** What remains is one figure/analysis
decision, one internal inconsistency, a handful of stale text numbers, and copy edits.

---

## A. Real errors to fix

1. **Risky-first slope CI is a copy-paste error** (Results, psychophysical section):
   "from 2.44 [2.05, 2.85] to 2.41 **[2.19, 2.97]**". The second CI is the risky-second
   *vertex* CI. Correct value from `localnoise_group_posterior.tsv`: **[2.05, 2.80]**.

2. **Two different preferred-numerosity IQRs for the same quantity** on the same Results
   page: "[7, 15]" (first mention) vs "[6, 10]" (second mention + Fig 2B reference).
   [7, 15] is the `encoding_model2.model-1` rebuild; [6, 10] is the published old-tree
   number (cvR² > 0, restricted to plotted range < 30 — see PROVENANCE). Fig 2B is built
   from the old tree, so quote **[6, 10] in both places** (or [6, 13] if the <30
   truncation is deemed too figure-bound — Christian decision, see below).

3. **Stale text numbers in the probit paragraph** — the figure matches the current TSV
   exactly; the text is from an earlier extraction. Against
   `localnoise_group_posterior.tsv`:
   - slope reduction p: text 0.013 → **0.012** (figure already says 0.012)
   - risky-first slope p: text 0.446 → **0.439**
   - risky-first RNP p: text 0.351 → **0.330** (figure: 0.330)
   - consistency interaction: text 0.0338 → **0.035**
   - order-effect replication "p < 0.001": pooled-over-stim draws give **p = 0.003**;
     recompute with the exact pooling the text intends and print that
   - RNP CIs differ in the second decimal throughout (e.g. text [47.2, 58.3] vs TSV
     [46.6, 57.8]) — regenerate the paragraph from the TSV in one pass.

4. **Cross-reference "Fig. 3D"** in the Linking section → **Fig. 3B** (Fig 3 has only
   A/B).

4b. **By-stake paragraph quotes the wrong cell** (found 2026-08-19, details in §B):
   "no such reduction when risky options were presented first (1.73→1.59, p=0.204)" —
   those numbers are the **high-stake safe-first** cell. Reword to "no such reduction
   for high-stake trials (1.74 → 1.59, p = 0.204)" (recommended; matches the
   interaction being within safe-first trials), or quote the actual risky-first
   low-stake cell: 2.83 → 2.60, p = 0.111.

5. **Copy edits** (no numbers): "presented ist" → first; "Moreover,,"; "with the the"
   (Fig 2C caption); "the the risk-neutral" (Methods); "perpetual noise" → perceptual
   (Flexible-PMC Methods); "precepts" → percepts; "Hammiltonian" → Hamiltonian;
   "Magenture" → MagVenture; missing spaces "work(Prat-Carrabin", "sit(Bhatia"; "120
   trials per session/ After" → ". After"; stray apostrophe in the section header
   "…neurocognitive representations'"; "e.g.., didn't"; Methods says "dorsal bank" where
   Results/Fig 1C say "rostral bank" — pick one.

## B. The one open analysis decision: the by-stake block + Supp Fig 1

The by-stake paragraph (slope 3.02→2.32 low-stake safe-first, 1.73→1.59 risky-first,
interaction p = 0.0153) and Supp Fig 1 still come from the **old**
`probit_average_n_full` notebook analysis. Status per `checks_20260803.md`:
- p = 0.0153 reproduces **only** as the named triple-interaction coefficient
  (P = 0.0152); the notebook's own cell-contrast reading of the same trace gives 0.053.
- The new four-cell replacement (`analyze_probit_by_stake.py`) gives, for the low-stake
  risky-second slope: **2.93→2.26, p = 0.0002** with `(1|subject)` (the `ri` tag,
  matching the published RE structure), but only **2.86→2.46, p = 0.059** with full
  random slopes. The two RE structures genuinely disagree here.

**DECIDED 2026-08-19 (Gilles): option (i) — keep the old analysis.** Reproducibility is
now permanent: `behavior/scripts/reproduce_stake_probit.py` rebuilds all eight cell
slopes and the interaction from the stored trace
(`cogmodels/model-probit_average_n_full_trace.netcdf`, on local disk) and writes
`notes/data/probit_stake_cells_published.tsv`. Verified 2026-08-19:

- 3.02 [2.73, 3.31] → 2.33 [2.07, 2.61], p = 0.0002 ✓ (low-stake, safe-first cell)
- 1.74 [1.49, 2.00] → 1.59 [1.33, 1.85], p = 0.204 ✓ — **but this is the HIGH-STAKE
  safe-first cell, not the risky-first cell the text claims.** New erratum: the second
  clause should read "no such reduction for high-stake trials (1.74 → 1.59,
  p = 0.204)", which also makes the paragraph's logic cleaner (stake specificity
  within safe-first trials, matching the interaction coefficient, which is the stake ×
  stimulation term *within* safe-first trials). The actual risky-first low-stake cell
  is 2.83 → 2.60, p = 0.111. Supp Fig 1's slope panel (C) matches the trace; re-check
  its panel-D p-value placement against the TSV when finalizing.
- interaction: named coefficient −0.5425 [−1.04, −0.05], P(>0) = 0.0152 ✓ =
  manuscript's 0.0153. When quoting it, say it is the stake × stimulation interaction
  on safe-first trials (risky_first = 0 is the model's reference, so the 3-way
  coefficient is exactly that). Alternative readings from the same trace: within
  risky-first +4-way term, p = 0.091; averaged over orders, p = 0.009.

Option (ii) (swap to the new `ri` four-cell analysis) stays available as robustness;
note the full-random-slopes variant weakens the low-stake cell to p = 0.059.

## B2. Amplitude test — DECIDED 2026-08-19 (Gilles): keep the preprint version

Panel 2B and its statistic stay as published (old tree, cvR²-either-arm mask,
1.30 → 1.04, t(34) = 1.99, p₁ = .027) — verified reproducible, see §C. The
unthresholded-m1 alternative below (t = 2.76, p₂ = .009) and the selection prototypes
(`notes/figures/prototypes/figure2b_m1_selections.pdf`,
`modeling/scripts/plot_figure2b_prototypes.py`) are exploration on record, available
as a robustness sentence or reviewer response, not for the main text. Original
assessment kept for reference:

The published amplitude test (old tree, cvR²-masked, t(34) = 1.99, p₁ = .027 one-sided)
reproduces and can stand. But `notes/amplitude_effect_voxel_selection.md` (2026-08-03)
documents a stronger, simpler one, re-verified 2026-08-19 from
`notes/data/prf_voxel_table.tsv`: **canonical m1 amplitudes, all 11 022 voxels of the
individualized 2 cm stimulation-site ROI, no functional threshold: Δ = −0.147,
t(34) = −2.76, p = 0.009 two-sided, Wilcoxon p = 0.011, negative in 23/35 subjects** —
with an anatomical gradient (NPCr −2.09, NPC12r −1.87, NF1 −1.63, contralateral NPCl
−0.85, NTO −0.82) and the sign negative in 71/72 ROI × selection × aggregation cells.
Rationale: the cvR² threshold is a near-arbitrary subsample (session-1 vs session-2/3
cvR² correlate at r = 0.011) that costs power without buying independence. Honest
framing: found via a sensitivity sweep, so present the no-threshold whole-ROI test as
the simplest a-priori analysis with the sweep as robustness. Minimal use: one robustness
sentence after the published stat; maximal: make it the headline number + a small
site-specificity panel.

## B3. PLANNED: new composite Figure 2 with the model comparison as its bottom row

Decided 2026-08-19 (Gilles): the encoding-model comparison goes into the **main
Figure 2** as a bottom row, not a supplementary. The full composite is built and
rendered: `notes/figures/figure2_new.{pdf,png,svg}`, produced by
`modeling/scripts/plot_figure2_new.py` — all six panels from local TSVs, no cluster.

| panel | content | data | key stat (verified) |
|---|---|---|---|
| A | nPRF surface map, example subject | `notes/figures/imaging/figure2a_surface.png` | — |
| B | amplitude by preferred numerosity + preferred/presented densities | `prf_voxels_oldtree.tsv` (old tree, cell-7 mask) | 1.30→1.04, t(34)=1.99, p₁=.027 |
| C | decoding accuracy, vertex vs IPS | `bb_decoding.tsv`, NPCr2cm-cluster | 0.142→0.092, p=.032 |
| D | held-out cvR² vs training-mean null, six models | `cvr2_model_grid.tsv` (from `cvr2_vs_null_m0-5.tsv`) | m1 +0.0131, t(34)=2.67, p=.011 |
| E | fraction of voxels beating the null | same | m1 48.9% |
| F | paired Δ cvR² vs canonical m1 | same | all p ≤ .003 |

Design notes: red/green reserved for IPS/vertex throughout; models use canonical =
near-black, tuning (μ+σ) = blue, magnitude (amp+baseline) = orange, others gray.
Panel A is a raster **extracted from the v9 PDF** — no clean local asset exists;
`notes/figures/{paper,imaging}/figure2a.pdf` are an OLDER behavioral figure despite
their names (PROVENANCE's Fig-2 row is mislabeled on this point). For submission,
re-export panel A at full resolution from the original pycortex pipeline if possible.

To do when inserting:
1. Results: one sentence that the canonical amplitude-only model is also the model CV
   selects on the stimulation-site ROI (D–F), and one honesty sentence: CV does **not**
   separate tuning from magnitude models (m5 − m4 = +0.0010, p = 0.27), so the
   "amplitude, not tuning" claim rests on the parameter-level tests.
2. Caption bookkeeping: n = 35; dots subjects, diamonds mean ± SEM; B's band ±1 SEM
   across subjects (not the published bootstrap-over-voxels); B test on per-subject
   mean amplitude, cvR²>0-in-either-arm mask, old tree; C from NPCr2cm-cluster decode;
   D–F: ROI = individualized 2 cm cluster, null = LORO cvR² of predicting the
   training-fold mean (−0.018), paired t-tests df = 34.
3. Voxel-selection robustness (ran 2026-08-19, SLURM 5106922,
   `notes/data/cvr2_vs_null_m0-5_selected.tsv`): restricting to voxels where at least
   one model beats the null (61 % of voxels) changes nothing — m1 still wins all
   pairwise tests (p ≤ 0.001) and m5 − m4 stays null (p = 0.93); m1 is also the
   per-voxel winner in 35 % of those voxels (chance 17 %). Add one caption/text
   sentence to that effect. The all-voxel version stays primary (no selection to
   defend). The voxel-level-winner panel now exists in the standalone
   `plot_encoding_model_comparison.py` figure (4 panels); the composite keeps 3.
Requires no refits.

## C. Verified — no action

- Fig 2 paragraph (amplitude 1.30→1.04 t=1.99 p=.027; mu 14.8/17.8 p=.32; sd 0.77/0.91
  p=.21; R² 6.8→4.9% p=.023; prop cvR²>0 11.1→7.5% p=.027) — all reproduce from the old
  tree (`reproduce_figure2_stats.py`, `figure2_stats.tsv`); condition assignments now
  correct. The old tree `encoding_model.denoise.smoothed` must not be pruned.
- Decoding r=.142 vs .092, F=4.99 p=.032; interaction F=0.86 p=.360 ✓.
- r(33)=0.76 p<.001 and r(33)=−0.51 p<.001 ✓ (v8 fixes landed).
- Every ELPD number in the text (114.1/15.8, 2.0/3.9, 59.9/12.2, 64.3, 69.9, 34.8/17.3)
  matches Supp Table 1 = `table1_all16.tsv` ✓.
- Brain–behaviour block (r=0.53 [0.37, 0.67] p=.001, Spearman .59; risky-first −0.04,
  difference p=.008; pooled 0.28 p=.10) matches the `bb_link_*` results ✓.

## D. Execution plan (all local, ~an afternoon)

1. Decide B with Christian.
2. Regenerate the probit paragraph numbers from `localnoise_group_posterior.tsv`
   (10-line script or by hand from the summary in this note) and fix A.1–A.4.
3. If B(ii): `python -m tms_risk.behavior.scripts.plot_fig3_probit_stake --tag ri` and
   swap Supp Fig 1 + paragraph numbers.
4. Copy edits A.5.
5. Update `notes/PROVENANCE.md` current-draft pointer (done 2026-08-19).
6. Swap in the new composite Figure 2 + the two sentences from B3
   (figure already rendered: `notes/figures/figure2_new.pdf`).

---

# Outline for the conversation with Christian: what changed since v7, and why

## 1. The core issue: the published cognitive-model fits sat on two code bugs

The published (v7) Flexible-PMC fit (`flexible2.6`, 2024-11-05, bauer@`ecc6454`) had:
- **the noise-composition bug**: perceptual noise never actually entered the
  first-presented option (the memory spline basis was used twice), so the fitted "ν₁ =
  perceptual + memory" was not what the Methods described;
- **a choice-rule change afterwards** (`b66c806`) that redefines what ν means, so the old
  trace cannot simply be re-evaluated under fixed code.

Consequence: the published mechanism story — *cTBS raises noise on the second-presented
option, at low payoffs only* — does not survive. The clean refits (current bauer,
provenance-stamped, `flexible2nf` family) attribute the cTBS effect to **shared
perceptual noise on both options: an approximately constant absolute increase (~0.2 CHF),
hence proportionally largest for small payoffs (~12% at 7 CHF)**. Both versions fit the
choices and reproduce the order-specific behavioural effect; the refit is the one that is
internally consistent with released code. v9 reports the refit. This is the main
"results changed because of bugs" item.

## 2. The model comparison was rebuilt and got stronger

v8's ELPD table had three Weber row labels cyclically permuted (values right, names
wrong). Superseded by a 16-model comparison (Supp Table 1): best model = Flexible PMC
with cTBS on shared perceptual noise only; adding a memory-noise effect buys nothing
(2.0 nats); removing the perceptual effect costs 59.9; single-position models cost
64–70; nulls lose by ~114. The order-specificity now *emerges* from the model instead of
being fitted directly — a better story than v7's.

## 3. The brain–behaviour correlation was replaced

The published Δamplitude × Δcognitive-noise r = −0.38 (p = .012) was fragile: it only
reproduces with the ecc6454 spline-basis anchoring (HEAD's `get_sd_curve` gives −0.32 on
identical inputs), and a refactored notebook variant with mis-placed knots had silently
flipped it to +0.11. Since the noise curves themselves were refit (item 1), v9 replaces
it with a model-free link: **Δ nPRF gain × Δ choice consistency on risky-second trials,
r(33) = 0.53, p = .001** (Spearman .59), order-specific (p = .008) and site-specific
(Williams p = .0075 vs occipito-temporal, not currently in the text).
Caveats to put on the table: the permutation FWER depends on the family tested (.012
focused / .25 exhaustive) and the behavioural difference score has poor split-half
reliability, so the point estimate is likely inflated. Decide how much of this goes in
the paper.

## 4. Figure 3 was rebuilt model-free; small numeric shifts

Psychometric curves + paired IPS−vertex difference posteriors replace the mirrored
marginals. Same conclusions (both effects confined to risky-second trials); the exact
p's moved in the third decimal (e.g. slope p .013→.012). Also: v8's swapped order labels
in the split correlations are gone — v9 drops those splits entirely in favour of item 3.

## 5. The by-stake analysis needs his call (see B above)

Old analysis's interaction p = 0.0153 only reproduces under one reading of the trace;
the new four-cell refit gives p = 0.0002 under the published RE structure but p = 0.059
with full random slopes. Which analysis and which caveat do we print?

## 5b. The low-prior caveat (reanalysis_handoff §2.7) — how to frame it

**Units RESOLVED 2026-08-19: the flexible-model prior parameters are NATURAL CHF.**
The flexible PMC is built in natural space (`fit_pmc_noisefix.py`: "B-spline noise
function over magnitude in natural space"; `pin_objective_prior` pins `*_prior_mu`
to the natural-space payoff mean), so the refit priors are genuinely and precisely
low: **safe 3.57 CHF [−3.70, 5.55] against payoffs 7–28; risky 8.75 CHF
[4.23, 10.68] against ~12–112** — both upper CrI bounds below the smallest relevant
payoff, and even lower than the published fit's 10.9 / 18.8 CHF. The percept table
(`pmc_percepts_by_order.flexible2nf.tsv`) confirms the compression in CHF
(objective 7 → perceived 5.8, 14 → 7.6, 20 → 8.5) and the downward cTBS shift.
`reanalysis_handoff.md` §2.7 was right. **Known bug found in passing: the
`chf_mean/chf_lo/chf_hi` columns of `prior_shift.priorshift.tsv` wrongly exponentiate
natural-unit parameters (yielding nonsense like 13,392 CHF) — fix the extraction in
`prior_shift`'s script before those columns are ever quoted; the log_mean columns
are the raw (natural-CHF) parameters despite the name.** Not a fitting-prior
artifact: constrained fits centre the prior on the empirical mean and the posterior
lands 1.1–2.7 SD below it. Read: the model has no utility curvature, so
the low prior IS the compressive value function — it is what produces risk aversion
in this architecture. (An earlier draft of this note called the prior location
"weakly identified"; retracted — the CrIs above are a few CHF wide and the lowness
is decisively established.)

**Why the Weber PMC's priors look fine and the Flexible PMC's don't (2026-08-19,
read from `model-weber2_noisefix.head_trace.netcdf` locally):** the Weber refit's
priors are entirely reasonable — safe exp(2.51) ≈ **12.3 CHF** [8.8, 17.5] vs
presented ~15.8; risky exp(3.02) ≈ **20.5 CHF** [16.8, 25.3] vs ~36. The difference
is architectural, not empirical. The Weber model lives in log space, where shrinkage
X̂ = X^β·e^((1−β)μ) with β < 1 IS a compressive power law — it produces risk
aversion for free, so the prior can sit near the true payoff distribution. In the
Flexible (natural-space) model the percept is X̂ = β(X)·X + (1−β(X))·μ with
β(X) = σ²/(σ² + ν(X)²): the rising noise function does the CURVING (verified against
`pmc_percepts_by_order.flexible2nf.tsv`: implied β falls 0.64 → 0.30 over payoffs
7 → 20, and the implied ν (0.95, 1.27, 1.58, 1.90 CHF) matches the fitted noise
curve — the simple shrinkage arithmetic reproduces the model's own percept table).
What the LOW PRIOR sets is the LEVEL of the percept curve — **and the level is not
behaviorally identified.** The choice rule p·X̂ > Ĉ is invariant to rescaling all
percepts by any k, so choices constrain only the RATIO structure (perceived
risky/safe ratio vs stake → concavity; order asymmetries). The fit still pins a
level because the family is not closed under rescaling — β(X) = σ²/(σ²+ν(X)²) with
one shared σ cannot realize β′ = k·β at every X — so the anchor lands where the
parameterization needs it, not where behavior puts it. Consequences: (1) absolute
percept values and the prior location are family-pinned nuisance coordinates — never
quote them as subjective values (this is WHY reanalysis_handoff §2.7's rule is
right); (2) everything the paper claims lives in the relative structure (noise
curve shape, ratio distortions, IPS−vertex contrasts — Fig 5B–E are all
ratio/difference quantities and thus doubly protected); (3) Discussion framing: the
prior fixes the level of a distortion that choices only constrain up to scale. Do
not compare the two models' prior values as if they were commensurable beliefs. The cTBS
conclusions are IPS − vertex contrasts under shared priors and survive this: cTBS
does not shift the priors themselves (prior-shift fit, p ≈ 0.7; prior-variant model
adds ~nothing in ELPD), and Fig 5 column A deliberately reports percentage change,
not absolute perceived values. The `objprior_perception` fit is the stress test —
with objective priors, ~60% of trials sit below the prior mean and extra noise would
push percepts UP, the wrong direction — i.e. the data genuinely demand the low prior.
Proposed handling: one Discussion passage naming the prior as the regression target
of the compressive distortion (functionally overlapping a concave value function, cf.
Khaw et al. 2020), never quote absolute percept values, and point to
prior-manipulating designs (Renkert et al. 2025) as the way to dissociate.

## 5c. Fig 4B's Weber framing, and Fig 4C's localization (2026-08-19)

**4B:** "log-log slope 0.48 < 1" conflates two departures from Weber's law (whose
defining property is proportionality, i.e. no offset at 0): a compressive power law
vs an additive noise floor on Weber scaling. Against the fitted flexible2nf curve:
power law 0.57·x^0.48 fits the posterior mean almost exactly (rms 0.05 CHF); affine
"Weber + floor" 1.70 + 0.036·x misses the mean's curvature (overshoots at 7,
rms 0.17) but stays largely inside the 95% CrI; pure Weber fails (rms 0.86).
Natural-space prototype with both references:
`notes/figures/prototypes/fig4b_natural.pdf`
(`behavior/scripts/plot_fig4b_natural.py`). Text fix regardless of figure choice:
say "incompatible with pure scalar variability; well described by a compressive
power law" rather than implying slope-1 is the whole content of Weber's law.
Decisive test if wanted: fit an affine-noise variant (bauer has
`AffineNoiseRiskModel`) and add it to the ELPD ladder — one overnight VM fit.

**Spline ladder settled (2026-08-19, `notes/data/ladder_summary.tsv`, run locally on
`cogmodels.ladder`):** df 3/4/6 converged cleanly (r̂ ≤ 1.00, ESS > 2200), df 2 failed
(806 divergences), df 9 failed catastrophically (r̂ 1.59, ESS 7). ELPD: **5 df
(canonical, −4157.7) > 4 df (−4166.1) > 3 df (−4180.3) > 6 df (−4199.4) > 2 df
(−4250.8)**. Predictive fit peaks exactly at the paper's 5 df and falls with more
flexibility (p_loo climbs 220 → 267 at 6 df) — more splines do NOT sharpen the cTBS
effect, they add poorly identified coefficients (the 9-df Δ curve is sampler noise).
This also retro-justifies the Methods' "after some initial model comparisons, we
chose 5 splines". Caveat: ladder fits are the constrained-prior variant
(`tms_risk_constrained = True`), so the df comparison is within that family.

**4C:** the refit posterior does NOT support a localized bump: P(absolute Δν larger
at 7 than 56) = 0.12 (wrong direction), relative-scale localization only p ≈ 0.66–0.73
(`noisecurve_localisation.flexible2nf.tsv`). The 9-spline ladder fit that might have
sharpened it did not converge (r̂ 1.59, ESS 7 — unusable). The sharply localized
effect (+0.44*, 7–28 CHF only) was the pre-bugfix published fit. What IS defensible:
P(perceptual-noise increase over the 7–30 CHF region) ≈ 0.92–0.96
(`noisecurve_regional`), and — the strong, model-free version — ΔP(chose risky)
= +0.160 (p = 0.001) in the 7–17 CHF risky-payoff band (choices move to exactly
chance) vs +0.03–0.05 above, band matching the nPRF preferred-numerosity IQR
(`notes/localized_noise_summary.md`, figure `notes/figures/localized_noise.pdf`
ready with drafted caption). Recommendation: keep 4C as the honest
constant-absolute/relative framing and carry the localization claim with the
model-free figure.

**Figure 4 candidate settled 2026-08-19 (Gilles), after iteration:** `fig4_new` is now
the **v9 layout with only panel B changed to natural space** (A PPCs, B natural-space
noise with pure-Weber dotted + Weber-and-floor dashed references, C relative effect
with no annotation, D ELPD ladder). Set aside along the way: a PPC-of-the-effect
panel by safe payoff (duplicates Fig 5E), the model-free by-risky-payoff panel
(covered by `localized_noise.pdf`), the P(increase, 10–30 CHF) annotation on C, and
B's ν₁ dashed line (confusable with the affine reference; the ~0.1 CHF memory gap
moves to text/caption — the caption's ν₁ sentence must be dropped accordingly, and
the caption's "slope 0.48 vs slope 1" sentence replaced per the text fix below).
Text fix to carry regardless: say "incompatible with pure scalar variability; well
described by a compressive power law" instead of implying slope-1 is the content of
Weber's law. Earlier candidate description (superseded): `notes/figures/fig4_new.{pdf,png,svg}`
via `behavior/scripts/plot_fig4_new.py` (reuses `plot_fig4_model.py`'s PPC panel).
A = PPCs (unchanged); B = noise function in NATURAL space with pure-Weber and
Weber+floor references (replaces the log-log slope framing); C = relative cTBS effect
annotated with P(increase, 10–30 CHF) = 0.96; D = NEW: PPC of the cTBS effect itself —
observed ΔP(chose risky) by SAFE payoff (dots) against the flexible model's 95%
posterior predictive band, one row per order, nPRF-preferred payoffs (≤ 10 CHF)
shaded (reads `ppc_delta_by_safe.flexible2nf.tsv`, i.e. the Fig-5E data — decide with
Christian whether Fig 5E then drops or shrinks to avoid duplication); E = ELPD ladder
(unchanged). The model-free by-RISKY-payoff variant (+0.16 at 7–17 CHF, p = 0.001)
remains available in `notes/figures/localized_noise.pdf` and in the git history of
`plot_fig4_new.py`. Caption needs: D's band is a 95% posterior predictive interval
from choices simulated at the real trials (no s.e.m. on observed points); B's
references are least-squares fits to the vertex posterior-mean curve. Choose vs the
current `fig4_model` with Christian.

**Accentuating the local effect (2026-08-20, Gilles's request):** the honest lever is
promoting the MODEL-FREE localization result, which was absent from v9. Built:
candidate **Figure 3C** (`notes/figures/fig3c_localization.{pdf,png,svg}`,
`behavior/scripts/plot_fig3c_localization.py`, matched to Fig 3's style): observed
ΔP(chose risky) by risky-payoff quintile, risky-second +0.160 [0.067, 0.252]
(p = 0.001) in the 7–17 CHF bin — choices from 35% to exactly chance — vs
+0.03–0.05 above and nothing on risky-first; the probit's own posterior prediction
shown as a band (left-weighted but undershooting the first bin); nPRF-preferred bin
shaded. Caption bookkeeping: bars 95% CIs across subjects (n = 33–35 per cell);
quintiles of the risky payoff; the shaded bin contains the nPRF preferred-numerosity
IQR [6, 10]; band = 95% CrI of the hierarchical probit's predicted difference.
A draft Results paragraph accompanies it (see chat 2026-08-20 / below).
**CHOSEN (2026-08-20, Gilles): the subtle route** — the nPRF-preferred band is now
shaded on Fig 4C's payoff axis in `fig4_new` (light 7–10 CHF span, small vertical
label, no statistic attached; the Δ-noise curve peaks inside it). Caption note: "the
shaded region marks the portion of the nPRF preferred-numerosity IQR ([6, 10];
Fig. 2B) within the presented payoff range." **REVERSED later on 2026-08-20
(Gilles): the band was removed from `fig4_new` — panel C is the bare Δ-noise curve
with its CrI, no nPRF marking and no caption note.** The louder Fig 3C panel and its
paragraph remain built and available (`fig3c_localization.*`) if wanted later or as
a supplementary panel — its annotation now reads "35% to 51% risky, p = 0.001" (the
"to chance" gloss was dropped 2026-08-20: 50% can reflect indifference as much as
random responding, and the localized-noise analysis itself shows a pure flattening
cannot explain the data; avoid the phrase in prose too).

**Order-color convention codified (2026-08-20, now in CLAUDE.md):** presentation
order never gets a hue (red/green = stimulation, blue/orange = model contrasts);
order is encoded by row/panel position, and where both orders share a panel:
risky second = near-black filled, risky first = light gray open. All current paper
figures already comply. Caution for Fig 3B: its dark/light densities encode
SIGNIFICANCE, not order — worth a caption word since the significant row is also
the risky-second row and readers may conflate the two.

**Reconciliation with the small-preferred-numerosity story (2026-08-19):** no
contradiction — the tuning account predicts localized FRACTIONAL information loss,
not an absolute bump. With Δν(x) ≈ ν(x)·f(x)/2 and a rising baseline (ν ~ x^0.5), a
small-number-concentrated fractional hit (f ~ x^−0.5) yields a constant absolute
Δν — exactly the refit result. Fig 4C (Δν/ν in %) already plots the right quantity;
fix the narration: state the prediction in relative terms from the start, and let
behavior (stake split; ΔP band profile) carry the localization evidence. "Raises the
noise floor" is the same story in Weber+floor vocabulary. Weak link to acknowledge
with Christian: the magnitude-resolved NEURAL effect itself (mc_decode expected
uncertainty collapsed; amplitude-by-preferred-numerosity split only p ≈ .07/.23).

## 6. Small reporting corrections already folded into v9

Purely presentational, no conclusion changes, but worth mentioning as part of the audit:
the v7/v8 "r = 0.76, p = 0.004" mixed the Bayesian r with a MAP-fit p (now p < .001);
preferred-numerosity and dispersion descriptives were attached to the wrong conditions
(now fixed); r(34)→r(33); and the preferred-numerosity IQR quoted depends on which
encoding-model tree is used ([6, 10] old tree with cvR² > 0 vs [7, 15] current
canonical model) — v9 currently quotes both, one must go (see A.2).
