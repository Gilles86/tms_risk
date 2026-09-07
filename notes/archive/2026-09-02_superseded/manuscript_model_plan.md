# Plan: adapting the manuscript to the 2026-08 model-comparison work

Status: PLANNING DOCUMENT — the manuscript, PROVENANCE.md, and the paper
figure tree are still untouched. Every section below flags its decision
points for Gilles; nothing here is executed until the primary-model decision
(§1) is made.

Inputs: the 2026-08-20/22 exploratory model set — power-law family
(`cogmodels.power`), log-space flexible family (`cogmodels.logflex`), the
24-cell spline/hyperprior grid (`cogmodels.lfxgrid` on the cluster share),
the flexible2/weber2 head refits (`cogmodels.overnight`), plus
`notes/rdm_magnitude_rt_plan.md` (RT work, incomplete). Ladder TSVs and
figures under `notes/data/` and `notes/figures/` (`power_*`, `logflex_*`,
`lfxgrid_*`, `winner_lfx2.*`).

---

## 0. What changed scientifically this week (one paragraph each)

1. **The prior is load-bearing.** Removing Bayesian shrinkage costs ≈730
   ELPD (21 dSE) in every architecture. The "efficient-coding without
   priors" alternative is dead on these data; adaptive/efficient-coding
   readings survive only as re-implementations of the same
   noise-compression coupling.
2. **The low-prior puzzle is solved, architecturally.** The natural-space
   flexible model's absurd 9-CHF prior is forced by additive-shrinkage
   geometry (the probability weighting keeps the two options' prior pulls
   from cancelling); on a ratio scale (lognormal prior, log-space observer)
   the same data yield priors at the payoff statistics (risky ≈ 17 CHF ×/÷
   1.5). This converts v9_plan §5b's defensive caveat into a positive
   finding.
3. **A leaner, better-behaved primary candidate exists.**
   `lfx2-bs3-sm-dp-b` (log-space, cubic-spline perceptual noise, scalar
   memory noise, TMS on perceptual noise): ELPD −4162.6, statistically tied
   with the natural-space champion (−4157.5, Δ ≈ 5 at cross-set dSE ~10,
   TO BE COMPUTED, §4.1), 10 group parameters vs ~24, r̂ ≤ 1.01, ~0
   divergences, realistic priors, decisive TMS effect (+98 ELPD, z = 7.3).
4. **The cTBS conclusion is modeling-invariant.** TMS-vs-null gains of
   +80…+114 ELPD (6–10 dSE) in every family and every one of the 24 grid
   cells; localization at low payoffs appears in the raw data (ΔP = +0.16,
   p = 0.001, risky-second), in natural-space splines, in the power-law
   tilt, and in log space (region-integrated P(noise increase, 7–30 CHF) =
   0.95–0.97). Perception-only TMS placement wins in both spline families.
5. **Robustness is now demonstrated, not asserted** (the "homework"): a
   24-cell factorial over spline basis × memory structure × hyperpriors
   shows the conclusions are invariant; scalar memory is strictly better
   (fit, diagnostics, power); natural-cubic tail-clamping *loses* fit — the
   high-payoff noise rise is signal.

---

## 1. DECISION A — RESOLVED 2026-08-22: Option 1. The log-space model
## (`lfx2-bs3-sm-dp-b`) is the new primary; Flexible PMC becomes sensitivity.

**Option 1 (recommended): promote the log-space model** (`lfx2-bs3-sm`,
"Log-Flexible PMC") to primary; natural-space Flexible PMC becomes the
lead sensitivity analysis.
  + Kills the low-prior reviewer bait entirely; Discussion §5b becomes a
    result (prior geometry) instead of an apology.
  + Cleanest sampling story (no divergence footnotes), fewest parameters,
    priors quotable as beliefs.
  + The published Weber PMC is already log-space — this is a return to the
    original geometry with a flexible noise function, an easy narrative.
  − Cost: Fig 4/5 pipelines re-run against the new primary (percept tables,
    prior-shift analysis, Fig 5's channel decomposition); Results numbers
    change; more re-writing.
  − The natural-space model fits nominally best; a reviewer could ask why
    we didn't use it (answer in text: statistical tie + identifiability).

**Option 2 (conservative): keep Flexible PMC primary**, add a Model
Comparison section + the log-space model as the interpretability anchor.
  + Minimal disruption to existing Results/Figures.
  − Keeps the low-prior caveat and the divergence-laden fits in the
    spotlight; the strongest new material is demoted to supplement.

Everything below is written for Option 1; Option 2 reuses §3–§5 with
figures 4B/4C left on flexible2nf.

## 2. DECISION B — RT/accumulator work CONTINUES in parallel (agent
## assigned 2026-08-22); inclusion decision deferred until lapse-equipped
## fits land. Manuscript work does not wait for it.

The magnitude→RT effect (p = 0.0008, risky-second p = 4e-5) and the RDM
extension are compelling but incomplete (lapse mixture unimplemented; the
six GPU fits ran lapse-free and are unvalidated). Recommendation: **out of
this manuscript**; one Discussion sentence + the model-free RT effect as a
supplementary observation at most, full treatment reserved for a follow-up.
Alternative: delay submission until the RDM set is done (est. +1–2 weeks).

---

## 3. Manuscript changes (Option 1)

### 3.1 Results — new subsection "Model comparison" (the homework)
Placement: after the current PMC results, before neurobehavioral
correlations. Content:
- **Master ladder table (main text, condensed ~10 rows)**: Weber PMC,
  Flexible PMC (null/TMS), Power-law PMC (incl. no-prior), Log-Flexible PMC
  (null/TMS variants), all on the identical 8,335 trials, ELPD ± paired
  dSE vs best. Full 24-cell grid + power family to a supplementary table.
- Three sentences of inference, in strength order: (i) TMS models beat
  matched nulls by 80–114 ELPD (6–10 dSE) in every family; (ii) removing
  the prior costs ≈730 (21 dSE); (iii) the log-space model matches the
  natural-space champion with realistic priors and 10 parameters.
- Reporting language per the two-tier scheme: ELPD carries existence
  claims; posterior CrIs characterize; the word "significant" reserved for
  the frequentist model-free anchors (localization p = 0.001).

### 3.2 Figure 4 — rebuild around the winner
- **4A** PPC by stake, THREE pairs: Weber | Flexible | Log-Flexible
  (existing `plot_power_fig4a.py` machinery; swap power→logflex, needs the
  winner PPC, §4.2). Miss-arrows convention kept.
- **4B** noise function ν(payoff) by stimulation from `lfx2-bs3-sm-dp-b`
  (winner_lfx2 panel b, log units; natural-space version to supplement).
- **4C** cTBS contrast, absolute Δσ with the region-integrated statistic
  annotated (P = 0.95, mean +0.016 [−0.002, +0.033] over 7–30 CHF).
- **4D** condensed ladder (from §3.1 table).
- New **4E (or Fig 5 panel)**: subjective priors over payoff histograms —
  the architectural point in one panel (winner_lfx2 panel a).

### 3.3 Fig 5 (mechanism/percept figure) — port to log space
Percept tables, prior-shift analysis, and channel decomposition re-derived
from the winner (x̂ = n^w·e^((1−w)μ); report *relative* percept structure
only, consistent with the level-unidentifiability rule which still holds).
The IPS−vertex contrasts are ratio quantities and port cleanly.

### 3.4 Methods additions
- Log-Flexible PMC specification (log-space observer, lognormal priors,
  spline basis anchored on log payoff, scalar memory, choice rule =
  threshold form; ~1 column incl. equations).
- Model set + fitting: 5000+5000, target_accept, LOO/PSIS with paired dSE,
  bauer commit stamps; primary model pre-designated, all others labeled
  sensitivity (multiplicity statement).
- The region-integrated contrast definition (cluster-style posterior test).

### 3.5 Discussion
- Rewrite §5b material as a finding: compression = noise-coupled shrinkage;
  its scale is ratio-metric; Gaussian-in-francs priors cannot be realistic
  in this architecture (the 0.55-weighting argument, one paragraph, cite
  Khaw/Woodford + efficient-coding literature via the power-law analysis).
- One paragraph on efficient coding: the no-prior variant fails (730), the
  power-law reduction captures the code's shape but not the localization —
  positioning vs Frydman & Jin / Heng et al.
- One sentence flagging the RT magnitude effect + accumulator follow-up.

### 3.6 Supplementary material
- S-Table: full ladder (all families incl. 24-cell grid, diagnostics cols).
- S-Fig: grid dashboard (basis/memory/hyperprior invariance).
- S-Fig: prior-geometry comparison (power_vs_flexible + logflex priors).
- S-Fig: natural-space flexible results (continuity with prior drafts).
- Optional S-Note: the simulation walk-through (artifact content) as
  "Why compression, and why ratio-scale priors" — decide after length check.

---

## 4. Work queue BEFORE any manuscript edit (order matters)

4.1 **Cross-set alignment + master ladder** — verify observed_data equality
    between cluster-fit lfxgrid traces and the VM-fit flexible2nf/weber2nf
    traces; compute the single master ladder with paired dSE (extends
    `logflex_full_analysis.py`). Half a day.
4.2 **Winner PPC** — by-stake and ΔP pipelines (`compute_ppc*.py`) pointed
    at `lfx2-bs3-sm-dp-b`; coverage number; Fig-4A row. Half a day.
4.3 **Wandering-μ hygiene** — the winner is clean, but re-fit it once with
    the mild μ-centering (already in tms_risk 9e41a9f) to confirm estimates
    are unchanged; report as sensitivity. One cluster job.
4.4 **Port Fig-5 pipelines** (percepts, prior-shift, channels) to the
    winner; regenerate the tables the Results quote. 1–2 days.
4.5 **Freeze + provenance** — designate trace files as paper-canonical,
    move/copy to `cogmodels/` naming or document paths; add every new
    figure/table/statistic to PROVENANCE.md (first PROVENANCE touch happens
    here, not before). Stamp check: every canonical trace carries
    tms_risk_bauer_commit ∈ {7e2bcea…lineage}; bauer commits pushed to the
    bauer remote (currently local-only: 7e2bcea, 1b73278, 6813cda, 3e5c3ff,
    7f0d1a6).
4.6 Session-1 / no-TMS baseline: fit `lfx2-bs3-sm` variant on session 1 for
    the Fig-2/3-adjacent claims if the text quotes baseline PMC parameters.

## 5. Reviewer-proofing checklist (from the significance discussion)
- Primary inference = pre-named ELPD comparisons; state dSE and z.
- Every posterior probability paired with its CrI; "credible", never
  "significant", for Bayesian quantities.
- Localization: model-free p = 0.001 as the frequentist anchor; model
  contrasts as convergent characterization.
- Multiplicity: one primary model; all other fits named sensitivity
  analyses answering one listed question each (table in supplement).
- Null claims (e.g. no prior shift under cTBS) via ELPD non-gain + ROPE.

## 6. Open questions for Gilles
1. Option 1 vs 2 (§1). Plan assumes 1.
2. RT/RDM scope (§2). Plan assumes out.
3. Does the tutorial simulation note go in supplement or stay a lab note?
4. Push bauer commits to the public bauer repo now or at submission?
5. dp vs tp variant as canonical winner (identical fits; dp = fewer
   arbitrary choices — plan assumes dp).
