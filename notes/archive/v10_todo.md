# v10 todo list

Audited 2026-08-20 against `notes/paper/TMS_paper_v10.docx`, cross-checked against
`notes/v9_plan.md`, the embedded figures (extracted from the docx), and the TSVs of
record. All numbers below recomputed from
`notes/data/localnoise_group_posterior.tsv` (percentile CIs) and
`notes/brain_behavior_link.md` §2.2.

**Already fixed in v10 (verified, no action):** A.1 slope CI, A.2 single IQR [6, 10],
A.3 slope/RNP p-values + interactions, A.4 Fig 3B cross-ref, A.4b by-stake cells,
all A.5 copy edits (incl. dorsal/rostral → "dorsal bank"), new composite Figure 2
with D–F + both Results sentences, Figure 3 = new `fig3_probit`, Figure 5 =
`fig5.flexible2nf`, new Discussion paragraphs.

**Dropped by decision (2026-08-20):** low-prior Discussion passage (§5b);
Fig 2A re-export (embedded raster is ~325 dpi at print width — fine unless the
journal demands vector/600 dpi at final submission).

---

## Results — psychophysical section

### 1. Order-effect replication sentence — Cmd+F `51.0%`

- [ ] `[44.4, 57.0]` → `[44.7, 56.9]`
- [ ] `55.2% [47.5, 63.0]` → `55.2% [49.1, 60.8]` (current CI is from an old
      extraction and matches no current cell)
- [ ] `p < 0.001` → `p = 0.003` (pooled over stimulation, P(RS > RF) = 0.997)

### 2. cTBS-on-RNP sentence — Cmd+F `52.4%`

- [ ] `[95% CI: 47.2%, 58.3%]` → `[46.6%, 57.8%]`
- [ ] `57.9% [51.8%, 64.3%]` → `57.9% [51.4%, 64.1%]`
- [ ] `50.6% [44.3%, 57.45]` → `50.6% [43.8%, 57.0%]` (also fixes the stray "57.45")
- [ ] `51.3% [45.3%, 57.1%]` → `51.3% [45.2%, 57.0%]`

Slope brackets in the preceding sentence are fine — leave them.

## Results — modeling section

### 3. Stale model-count/reference — Cmd+F `among the six models tested`

- [ ] `…lowest among the six models tested (see Table 1)` →
      `…lowest (see Supplementary Table 1)`.
      There is no main Table 1, and the table has 16 models, not six.

### 4. Cross-ref — Cmd+F `captured this gap`

- [ ] `(Fig. 4A-B)` → `(Fig. 4A)` (panel B is the noise function, not the PPC).

### 5. New sentence: encoding-model robustness (plan B3.3)

- [ ] Append to the paragraph ending `…most parsimonious description of the data.`:

> These results did not depend on voxel selection: restricting the comparison to
> voxels in which at least one model outperformed the null predictor (61% of voxels)
> left every pairwise comparison intact (all p ≤ 0.001), and the amplitude-only
> model was the single best-fitting model in 35% of these voxels (chance level: 17%).

## Results — Linking section

### 6. New sentence: site-specificity

- [ ] Insert after `…pooled across both orders the correlation was correspondingly
      weaker (r(33) = 0.28, p = 0.10).`:

> The relationship was also specific to the stimulated site: the same correlation
> computed from amplitude changes in other numerosity-tuned regions fell off with
> distance from the coil (right parietal outside the target mask: r = 0.48; left
> parietal: r = 0.21; frontal: r = 0.17; occipito-temporal: r = 0.02), and was
> significantly weaker than at the stimulation site in the left parietal (Williams
> test, p = 0.020), frontal (p = 0.031), and occipito-temporal (p = 0.0075) regions.

Source: `notes/brain_behavior_link.md` §2.2 (NPCr2cm-cluster r = +0.533 p = .0010;
NPC12r +0.480, vs site p = .33; NPCl +0.211, p = .020; NF1 +0.172, p = .031;
NTO +0.018, p = .0075).

## Figures

### 7. Swap Figure 4

- [ ] Replace the embedded image (still the old `fig4_model`: log-log panel B with
      "Slope 0.48" annotation and ν₁ dashed line) with
      **`notes/figures/fig4_new.pdf`** (`fig4_new.png` for Docs).
- Note: the shaded nPRF-preferred band on panel C was removed 2026-08-20 (Gilles's
  call) — panel C is now just the Δ-noise curve with its CrI.

### 8. Figure 4 caption (two edits, to match the new figure)

- [ ] Panel B: delete `the dashed line shows the first-presented option (ν₁) under
      vertex stimulation` and replace with:
      `The dotted line shows pure Weber scaling (noise proportional to payoff) and
      the dashed line an affine "Weber plus floor" reference; both are least-squares
      fits to the posterior-mean vertex curve.`
      Optionally add: `The first-presented option carries an additional ≈0.1 CHF of
      memory noise (not shown).`
- [ ] Panel B's `Fitted representational noise ν increases sublinearly…` can stay —
      already correct for the new panel.

### 9. Figure 3 caption, panel B — significance ≠ order guard

- [ ] Add: `Dark versus light densities distinguish credible from non-credible
      effects, not presentation order.`
      (The significant row is also the risky-second row; readers may conflate the
      shading with an order palette.)

### 10. Regenerate Supplementary Figure 1 — panel D p-values transposed — **DONE 2026-08-20**

- [x] Figure regenerated from the stored trace:
      `notes/figures/supp_fig1_stake.{pdf,png,svg}`, produced by the new
      `tms_risk/behavior/scripts/plot_supp_fig1_stake.py` (reads
      `cogmodels/model-probit_average_n_full_trace.netcdf`, cross-checks its slope
      cells against `notes/data/probit_stake_cells_published.tsv`, aborts on
      disagreement). Panel-D p-values now land correctly: risky-first/low = 0.111,
      risky-second/high = 0.204.
- [x] Panel B verified from the trace while at it: 0.354 / 0.343 / <0.001 / 0.012 —
      the old figure's B placement was fine (its 0.357/0.349/0.014 were the same
      cells, MC-noise apart).
- [ ] **Insert the new figure into the docx** (replaces the old violin raster). The
      A–D caption still applies (A indifference points with risk-neutral reference,
      B differences, C consistency, D differences); style is now Fig-3-matched
      (posterior mean + 95% CrI, density ridges for the differences).
- [ ] Caption notes when inserting: p-values follow Fig 3's convention
      (min-direction posterior probability), so the risky-first/high-stake slope
      cell reads p = 0.248 where the old figure printed 0.752 — same posterior
      mass, consistent convention. Old caption's "dotted line" → dashed
      risk-neutral reference; RNP panel now shows 95% CrIs, not violins.

## Housekeeping (repo, not the docx)

### 11. `notes/PROVENANCE.md`

- [ ] Line 8: current-draft pointer still says `TMS_paper_v9.pdf` → point to
      `notes/paper/TMS_paper_v10.docx`.
- [ ] Add a provenance row for the site-specificity sentence (item 6) pointing at
      `notes/brain_behavior_link.md` §2.2 / the `bb_link_*` TSVs.
