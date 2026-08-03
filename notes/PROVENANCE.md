# Where every figure and number in the paper comes from

One row per published item. If you want to regenerate something, find it here, run the
command, done. If a row says **stale**, the code has moved on from what the preprint
shows and the number needs recomputing before it is quoted again.

Paper: *Risk Attitudes Causally Rely on Parietal Magnitude Representations*
(de Hollander, Moisa & Ruff). Current draft: `notes/paper/TMS_paper_v8_with_CR_comments.pdf`
(not in git — 260 MB of PDFs and figures are ignored; see `.gitignore`).

---

## The two-stage rule

Almost nothing is computed where it is plotted. Cognitive-model traces are ~1.2 GB each
and live only on the fitting nodes; nPRF derivatives are tens of thousands of small
NIfTIs on the cluster. So every analysis is split:

```
   [ node holding the data ]                    [ laptop ]
   extract_*.py / analyze_*.py    -- TSV -->    plot_*.py  --> PDF/PNG/SVG
        heavy, needs the trace                   light, needs only the TSV
```

* **`notes/data/*.tsv` is the source data of record.** It is tracked in git. Every
  figure can be rebuilt from it with no trace, no bauer, no GPU, no cluster.
* **`notes/figures/` is not tracked.** It is output. Delete it freely.
  It is sorted into one folder per *plot type* (`ppc/`, `noise/`, `decision_space/`,
  `percepts/`, `mechanism/`, `parameters/`, `model_compare/`, `imaging/`) because the
  model label is already in every filename, so grouping this way puts every model's
  version of the same plot side by side. `paper/` holds a copy of whichever variant
  the manuscript currently uses. Re-sort after generating new figures with
  `python -m tms_risk.behavior.scripts.organize_figures --apply`.
* Extraction scripts take `--trace_dir` and `--tag`; plotting scripts take `--label`
  and resolve `notes/data/<something>.<label>.tsv`.

The `<label>` suffix identifies *which fit* a figure came from — this matters more than
usual here, see [Which fit is which](#which-fit-is-which).

---

## Main figures

| Item | Produced by | Reads | Writes |
|---|---|---|---|
| **Fig. 2** — nPRF preferred numerosity, amplitude, stimulation site | `tms_risk/notebooks/figure2.ipynb` | `derivatives/encoding_model2.model-1.smoothed/` | `notes/figures/figure2{a,b,c}.pdf`, `figure2_legend.pdf` |
| **Fig. 3A** — psychometric curves (observed) | `behavior/notebooks/figure4.ipynb` | `get_all_behavior()` | inline |
| **Fig. 3B** — preferred numerosity at the stimulation site | `tms_risk/notebooks/figure2.ipynb` | as Fig. 2 | inline |
| **Fig. 3 (reanalysis)** — psychometric curves + probit parameters, the model-free argument. Supersedes Fig. 3A; block B shows the paired IPS − vertex difference posterior, not the published mirrored marginals (rationale in `reanalysis_handoff.md` §5) | `behavior/scripts/plot_fig3_probit.py` | `localnoise_{group_posterior,delta_by_ratio}.tsv`, `ppc_fig3a.<label>.tsv` | `notes/figures/fig3_probit.*` |
| **Fig. 3 by stake** — the same probit argument refit at low vs high stakes (four independent hierarchical probits, one per order × stake cell). Two random-effects structures: `--random_effects full` (default, per-subject slopes) and `--random_effects intercept --tag ri` (the published `(1\|subject)`). The figure prints which one it is; the two differ substantially — see `checks_20260803.md` §1(a-ter) | `behavior/scripts/analyze_probit_by_stake.py` → `behavior/scripts/plot_fig3_probit_stake.py [--tag ri]` | `/data/ds-tmsrisk` behaviour (fit step); `probit_stake_{group_posterior,by_ratio}[.ri].tsv` (plot step) | `notes/figures/fig3_probit_stake[_ri].*` |
| **Fig. 4A** — posterior predictive check | `behavior/scripts/plot_ppc_fig3a.py` | the **trace** (run on the fitting node) | `notes/data/ppc_{fig3a,by_safe,by_stake,delta_by_safe}.<label>.tsv` → `notes/figures/ppc_fig3a.<label>.*` |
| **Fig. 4B/4C** — noise vs. magnitude, and the cTBS contrast | `behavior/scripts/plot_fig4bc_style.py` | `noisecurve_reparam.<label>.tsv` | `notes/figures/fig4bc_style.<label>.*` |
| **Fig. 4 (reanalysis)** — all of the above on one page: PPC of the Weber fit beside the flexible one, the noise function, the cTBS contrast, the ELPD ladder. Supersedes Figs. 4A–4C | `behavior/scripts/plot_fig4_model.py` | `table1_all16.tsv`, `ppc_by_stake.{weber2nf,flexible2nf}.tsv`, `pmcpars_{curves,relative}.<label>.tsv` | `notes/figures/fig4_model.*` |
| **Table 1** — ELPD model comparison | `behavior/scripts/loo_table.py` | trace `log_likelihood` groups | `<out_stem>.tsv` + `.md` |
| **Fig. 5A/5B** — decision-space heatmaps (total noise, perceived EV ratio) | `behavior/scripts/plot_fig5_style.py` | `decision_space.<label>.tsv`, `pmcpars_curves.<label>.tsv` | `notes/figures/fig5_style.<label>.*` |
| **Fig. 5 (reanalysis)** — five columns, rows = presentation order: (A) perceived value lost to cTBS per option, as % of that option's vertex percept, 95% CrI; (B) perceived risky/safe ratio; (C) leverage; (D) Δ P(chose risky); (E) posterior predictive vs observed Δ P by safe payoff. Supersedes `plot_fig5_style` | `behavior/scripts/plot_fig5.py` | `decision_space.<label>.tsv`, `pmc_percepts_by_order.<label>.tsv`, `ppc_delta_by_safe.<label>.tsv`, `paradigm_payoffs.tsv` | `notes/figures/fig5.<label>.*` |
| **Fig. 5 caption facts** — column A bars are 95% credible intervals, column E's band is a 95% posterior *predictive* interval from simulated choices at the real trials (so the observed points carry no s.e.m.); columns B and D use fixed colour half-ranges of 0.07 and 0.15, the latter matching column E's y axis | — | — | stated here because the figure deliberately does not state it |

### Supplementary

| Item | Produced by | Writes |
|---|---|---|
| **S1.1–S1.4** — perceptual-distortion heatmaps | `behavior/scripts/plot_perceptual_heatmaps.py` | `notes/figures/` |
| **S1.5** — P(risky) heatmaps | `behavior/scripts/plot_decision_space.py` | `notes/figures/decision_space.*` |
| All group-level PMC parameters | `behavior/scripts/plot_pmc_parameters.py` | `notes/figures/pmc_parameters.*` |
| Mechanism walkthrough (cTBS → noise → percept → choice) | `behavior/scripts/plot_pmc_explained.py` | `notes/figures/pmc_explained.*` |
| cvR² encoding-model comparison | `modeling/scripts/compare_cvr2_models.py`, `plot_cvr2_voxels_won.py` | `notes/figures/cvr2_*` |
| Spherical-Ω expected uncertainty | `modeling/scripts/plot_spherical_expected_uncertainty.py` | `notes/figures/spherical_expected_uncertainty.*` |

---

## Reported statistics

| Statistic | Computed in | Status |
|---|---|---|
| nPRF amplitude / `mu` / `sd` / `r2`, IPS vs vertex | `modeling/notebooks/analyze_encoding_model.ipynb`, the `pingouin.pairwise_tests` loop | runs **two-sided**; the paper halves p for directional claims |
| Proportion of voxels with cvR² > 0 | same notebook, the `pg.ttest(..., alternative='less')` cell | the only genuinely one-sided test there |
| Decoding accuracy, decoding × order interaction | `modeling/notebooks/analyze_decoding.ipynb`, the two `rm_anova` cells | ok |
| Indifference point × choice consistency; Δconsistency × Δrisk attitude | `behavior/notebooks/correlation_preference_noise.ipynb`, cell 3 + final cell | ok |
| Δ nPRF amplitude × Δ cognitive noise (the *r* = −0.379) | `behavior/notebooks/neurobehavioral_correlates.ipynb`, cell 4 | **stale** — built its spline basis on `np.arange(7, 50)`, putting the interior knot at 28 rather than the paradigm's 20. Superseded by `behavior/scripts/plot_noise_amplitude_link.py` |
| ELPD table | `behavior/notebooks/comprehensive_model_comparison.ipynb` | **stale** — three Weber row *labels* are a cyclic permutation of the models they name (`11a`=both, `11b`=memory, `11c`=perception). Values fine, names wrong. Superseded by `loo_table.py` |
| Flexible PMC noise curves | `behavior/notebooks/figure4.ipynb` | **stale** — calls bauer's `get_sd_curve`, which anchors knots differently than the fitting commit did |

`notes/v8_stats_check.md` audits the v8 Results line by line against these.

---

## Which fit is which

A stored PMC trace only means something against the bauer commit that produced it, and
a mismatch fails **silently** — see the "Cognitive-model traces are bauer-version-sensitive"
section of `CLAUDE.md`, and `notes/pmc_refit_results.md` for the full audit.

| Label | What it is | Where the trace lives |
|---|---|---|
| `flexible2` | the **published** fit (2024-11-05, bauer@`ecc6454`). Its `ν₁` uses the memory splines twice, so perceptual noise never entered the first option — read it as a relabelled family-1 fit | `derivatives/cogmodels/` |
| `flexible1nf`, `flexible2nf` | refits against current bauer with the intended noise composition. `1` = independent per position (first/second), `2` = shared perceptual + memory. Exact reparameterisations of each other | `derivatives/cogmodels.overnight/` |
| `weber2_noisefix*` | Weber PMC baseline: noise constant in log space (scalar invariance), no spline basis | `derivatives/cogmodels.overnight/` |
| `flexible2.<N>_noisefix` | spline-complexity ladder, `<N>` = number of basis coefficients | `derivatives/cogmodels.ladder/` |
| same labels, `additive` composition | ν₁ = ν₂ + softplus(η_mem), i.e. memory noise constrained ≥ 0. **Rejected: 55.7 nats worse (dSE 10.4).** See `notes/positive_memory_noise.md` | `derivatives/cogmodels.additive/` |

### Where the traces physically are (2026-08-03)

The `Where the trace lives` column names a directory under `derivatives/`, and as of
2026-08-03 that directory exists on **three** tiers, not just the fitting node:

| Tier | Path | Role |
|---|---|---|
| ScienceCloud VMs | `/data/ds-tmsrisk/derivatives/cogmodels.*` | where the fits were produced; still the place to run any extraction that needs a trace |
| Local | `/data/ds-tmsrisk/derivatives/cogmodels.*` | working copy, so a trace can be re-read without a VM |
| Department SMB | `…/dehollander_moisa_ruff_ipsriskydecisionmaking/data/ds-tmsrisk/derivatives/cogmodels.*` | the archive of record; append-only |

The four T4 boxes hold the **numpyro/JAX** fits, and those are the ones every TSV in
`notes/data/` was extracted from — they keep the plain directory names. The CPU VM
independently refitted eight of the same labels with pymc; those are a *different set
of chains*, not copies, and are parked in `cogmodels.ladder.cpu/` and
`cogmodels.overnight.cpu/` so they cannot be mistaken for the canonical fits or mixed
into a LOO comparison with them.

Everything fit from 2026-07 onward stamps its provenance into the trace:
`tms_risk_bauer_commit`, `tms_risk_family`, `tms_risk_spline_order`,
`tms_risk_spline_degree`, `tms_risk_constrained`. Read them with
`behavior/scripts/summarize_traces.py`.

---

## Brain–behaviour link (2026-08-03)

Write-up: `notes/brain_behavior_link.md`. All neural measures come from **m1**; all
model-based behavioural measures from the latest refits (`flexible2nf` / `flexible1nf`).

| Statistic | Script | Reads | Writes |
|---|---|---|---|
| Per-subject Δ nPRF gain (overall, by preferred numerosity, tuning-weighted profile); Δ decoding accuracy / error / posterior width; Δ P(chose risky), psychometric slope, indifference point | `modeling/scripts/extract_brain_behavior_table.py` | `notes/data/prf_voxel_table.tsv`, `derivatives/decoded_pdfs.volume.cv_voxel_selection.denoise.natural_space/`, BIDS events | `notes/data/bb_neural.tsv`, `bb_decoding{,_runs,_trials}.tsv`, `bb_behavior.tsv` |
| Focused-family and exploratory correlation grids with max-\|r\| permutation FWER | `behavior/scripts/analyze_brain_behavior_link.py` | the `bb_*.tsv` above + `subject_noise_shift.<label>.tsv` | `notes/data/bb_link_{master,focused,grid,robustness,composite}.tsv` |
| Group effects, split-half reliability, stability across mask/selection, preferred-numerosity specificity, slope-artifact check | `behavior/scripts/check_brain_behavior_robustness.py` | same | `notes/data/bb_link_stability.tsv` |
| Within-subject trial-by-trial decoding × choice coupling | `behavior/scripts/trialwise_decoding_choice_link.py` | `bb_decoding_trials.tsv` | `notes/data/bb_trialwise*.tsv` |
| Figure (scatter, ROI specificity, trial-level) | `behavior/scripts/plot_brain_behavior_link.py` | `bb_neural.tsv`, `bb_behavior.tsv`, `bb_trialwise_byorder_*.tsv` | `notes/figures/brain_behavior_link.pdf` |

**Headline:** Δ gain at the stimulation site × Δ choice consistency on risky-second
trials, r(33) = +0.53, p = .001 (Spearman .59). Site-specific (Williams p = .0075 vs
occipito-temporal) and order-specific (p = .0082). Caveats — the FWER depends on the
family (.012 focused / .25 exhaustive) and the behavioural difference score has poor
split-half reliability, so the point estimate is very likely inflated. Not a magnitude-
localisation result.

## Encoding-model choice and tuning-width analyses (2026-08-03)

Write-ups: `notes/encoding_model_choice.md`, `notes/tuning_width_by_preference.md`.

| Statistic | Script | Reads | Writes |
|---|---|---|---|
| Per-session held-out cvR² for m0/m1/m2 + reconstruction gate | `modeling/scripts/extract_encoding_model_cv.py` | `derivatives/encoding_model2.model-{0,1,2}.smoothed[.cv]/`, `derivatives/glm_stim1.denoise.smoothed/`, `derivatives/ips_masks/` | `notes/data/encoding_cv_gate.tsv`, `encoding_cvr2_by_session.tsv` |
| Per-voxel cTBS parameter shifts, per model | same | same | `notes/data/encoding_param_shifts.tsv` |
| m2 amplitude/dispersion/baseline trade-off, within subject | same | same | `notes/data/encoding_amp_sd_tradeoff.tsv` |
| Per-voxel mu / sd / amplitude / cvR² table over the numerosity ROIs | `modeling/scripts/extract_prf_voxel_table.py` | `derivatives/encoding_model2.model-{0,1,2}.smoothed[.cv]/`, `derivatives/ips_masks/` | `notes/data/prf_voxel_table.tsv` |
| Tuning width vs preferred numerosity; preferred-numerosity distribution; density-weighted precision | analysis over `prf_voxel_table.tsv` | as above | `notes/tuning_width_by_preference.md` |
| Expected decoding error E(s), ΔE(s), decoding↔behaviour correlations | analysis over `derivatives/monte_carlo_decode.denoise.spherical/` (produced by `modeling/monte_carlo_decode.py`) | m1 parameters, ResidualFitter Ω (spherical), 1000 sims/stimulus | `notes/tuning_width_by_preference.md` §(c) |

**Gate:** `extract_encoding_model_cv.py` rebuilds braincoder's predictions analytically
and asserts they reproduce the stored in-sample `desc-r2` map before writing anything.
All 105 gates (35 subjects × 3 models) passed at r > 0.9999999999, max |diff| 9.5e-07.

**The published preferred-numerosity IQR [6, 10] (Fig. 2B) — RESOLVED.** It comes from
the **old log-space tree** `encoding_model.denoise.smoothed` (still on local disk, 46
subject dirs), ROI `NPCr2cm-cluster`, **cvR² > 0** from `encoding_model.cv.denoise.smoothed`,
exp(mu) pooled over 2089 voxel-sessions: **IQR [6.00, 10.45], median 8.42**. Script:
`scratchpad/old_tree_iqr.py`. Two caveats: unthresholded the same voxels give
[8.00, 30.28], so the claim depends on the cvR² > 0 selection; and under the current
canonical `encoding_model2.model-1` the same quantity is [7.10, 15.22], so regenerating
Fig. 2B with current code changes the number. The presented-numerosity IQR [13, 30]
reproduces exactly from `get_all_behavior` (n1 and n2 identical).

**Figure 2 and its five statistics — reproduced, 2026-08-03.**

| Item | Script | Reads | Writes |
|---|---|---|---|
| The five Figure-2 statistics, each checked against its published value | `modeling/scripts/reproduce_figure2_stats.py` | `encoding_model{,.cv}.denoise.smoothed`, `ips_masks` | `notes/data/figure2_stats.tsv`; `--dump_voxels` → `notes/data/prf_voxels_oldtree.tsv` |
| Figure 2 recreated (amplitude vs preferred numerosity; preferred vs presented distribution) | `modeling/scripts/plot_figure2.py` | `notes/data/prf_voxels_oldtree.tsv` | `notes/figures/figure2_recreated.{pdf,png,svg}` |

Both use the **old tree**, ROI `NPCr2cm-cluster`, notebook-cell-7 mask (cvR² > 0 in
*either* arm). The figure's amplitude annotation is the published statistic itself
(1.3015 → 1.0416, t(34) = 1.99, p₁ = 0.027), so figure and paragraph now come from one
verified computation. One documented deviation: the error band is ±1 SEM across the 35
subjects, not the original's bootstrap over voxels (pseudoreplication);
`--voxel_level_ci` restores the published band.

**Preferred-numerosity IQR under that mask:** [6.13, 12.65] over all masked voxels, and
**[6.00, 10.48]** restricted to the plotted range (preferred numerosity < 30) — the
latter is the published **[6, 10]**. A *model-1* rebuild of the same quantity gives
[7.10, 15.22] (NPC12r) / [6.96, 14.83] (2 cm cluster); that variant is
`--voxel_tsv notes/data/prf_voxel_table.tsv` territory and is **not** what the paper
reports.

**Two problems this surfaced, both open:**

1. ~~The published Fig-2B amplitude effect does not reproduce~~ — **WRONG, retracted.**
   All five Figure-2 statistics reproduce to 3–4 decimals via
   `modeling/scripts/reproduce_figure2_stats.py` → `notes/data/figure2_stats.tsv`.
   The step I had wrong was the mask: notebook cell 7 keeps a voxel where cvR² > 0 in
   **either** arm (`(cvr2.unstack(arm) > 0).any(axis=1)`), preserving the pairing;
   thresholding row-wise drops different voxels from the two arms and none of the
   numbers come out. Source: the OLD tree (`encoding_model.denoise.smoothed` +
   `.cv.`), ROI `NPCr2cm-cluster`, per-subject mean over voxels, median across subjects
   for the descriptives. See `notes/amplitude_effect_voxel_selection.md`.
   **Four of the five are not computable under `encoding_model2.model-1`**, where
   `mu`/`sd`/`r2`/`cvr2` are session-invariant by construction — so the old tree is the
   source of record for that paragraph and must not be pruned.
   The table below is retained only as a record of what other recipes give:

   | source | ROI | vertex → IPS | t | p₁ |
   |---|---|---|---|---|
   | `encoding_model2.model-1` | NPC12r | 0.559 → 0.517 | −1.12 | 0.135 |
   | `encoding_model2.model-1` | NPCr2cm-cluster | 0.610 → 0.532 | −1.19 | 0.120 |
   | old tree NIfTIs, cvR² > 0 | NPCr2cm-cluster | 0.879 → 0.844 | −0.02 | 0.492 |
   | old tree NIfTIs, no threshold | NPCr2cm-cluster | 0.640 → 0.542 | −1.56 | 0.064 |
   | cached `prf_parameters_thr.tsv` | NPCr2cm-cluster | 0.798 → 0.568 | −1.61 | 0.059 |

   None matches the published medians or t. `notes/v8_stats_check.md` §4 documents a
   successful reproduction on 2026-07-30 from `analyze_encoding_model.ipynb`, so the
   recipe exists — it is just not any of the five above. **Resolve against the
   notebook's own `pairwise_tests` cell before the amplitude claim is re-quoted.**
2. **The documented safety copy does not contain what CLAUDE.md says.**
   `prf_parameters_thr.tsv` and `prf_parameters_thr.model-1_20260522.bak.tsv` are
   **byte-identical** (md5 `8434d61b…`). CLAUDE.md trap 3 says the `.bak` preserves the
   pre-2026-05-22 version; it does not. The pre-overwrite table is gone. The old-tree
   NIfTIs (`encoding_model.denoise.smoothed`, 46 subject dirs) survive and are the
   recovery source.

**Does NOT reproduce:** any usable E(s) slope from the monte-carlo decode — the decoder
returns ~59–62 for every stimulus from 7 to 111, so E(s) traces the bounded grid.

## Three checks against existing outputs (2026-08-03)

Write-up: `notes/checks_20260803.md`.

| Statistic | Script | Reads | Writes |
|---|---|---|---|
| Design correlations: stake vs ratio (r = +0.447), stake vs n_risky (+0.986), n_safe vs ratio (+0.022); stake-bin composition | `scratchpad/check1a.py` | `fit_probit.get_data('probit_average_n_full')` | inline |
| Stake × stimulation interaction, read as the named coefficient `x:stimulation_condition:C(average_n_bin)`: **−0.5425 [−1.0382, −0.0470], P(>0) = 0.0152** (risky second) — this is the manuscript's 0.0153. Cross-checked by rebuilding the four cell slopes per draw (agrees to 4.4e-16) | `scratchpad/check1_interaction_triple.py` | `cogmodels/model-probit_average_n_full_trace.netcdf` | inline |
| Same interaction with **random slopes** (`(x*stim*bin\|subject)`), and with ratio carried alongside, and on the n_safe axis | `scratchpad/check1_rs2.py` (new bambi probits) | `fit_probit.get_data` | `scratchpad/probit_{A,B,C}_*_rs.netcdf` |
| Two-stage per-subject ML probit check (order-pooled only; ~60 trials/cell) | `scratchpad/check1_twostage.py` | `fit_probit.get_data` | inline |
| Fig 5 panels a/c per-cell 95% CrIs and sign probabilities | `behavior/scripts/plot_decision_space.py`, extended with `cell_draws()` | `cogmodels.overnight/model-flexible2_noisefix.head_trace.netcdf` (run on `sciencecloud_gpu4`) | `notes/data/decision_space_draws.flexible2nf.tsv` |
| Fig 5 column E: paired Δ P(chose risky) per (order, safe payoff), posterior predictive from 200 draws of simulated choices at the real trials, alongside the observed | `behavior/scripts/plot_ppc_fig3a.py` (`ppc_delta_by_safe` block) | `cogmodels.overnight/model-flexible2_noisefix.head_trace.netcdf` (run on `sciencecloud`) | `notes/data/ppc_delta_by_safe.flexible2nf.tsv` |
| Observed cTBS effect marginals feeding the figures: by safe payoff, and by within-subject ratio sextile with real ratio midpoints (reproduces `localnoise_delta_by_ratio.tsv`'s deltas) | `behavior/scripts/extract_behavior_grid.py` | `/data/ds-tmsrisk` behaviour | `notes/data/behavior_effect_{grid,by_safe,by_ratio}.tsv` |
| Fig 5 sign-crossing analysis: risky-first crosses `cause` = 1 at n_safe 12.58, min 0.9935 at 18.2, P(cause<1) = 0.212 / 0.900 at the ends | `scratchpad/check2.py` | `notes/data/decision_space{,_draws}.flexible2nf.tsv` | inline |
| Split correlations: risky-first r = −0.594447 (p₂ = 0.0001662), risky-second r = −0.328518 (p₂ = 0.0540029), collapsed −0.510759; all n = 35 | `scratchpad/check3.py` | `cogmodels/model-probit_order_trace.netcdf` | inline |

**Does not reproduce:** the manuscript's stake × stimulation `pBayesian = 0.0153`. The
stored `probit_average_n_full` trace gives 0.0532 for the interaction and 0.0556 for the
low-stake cell collapsed over order, under the contrast the source notebook itself uses.

**Now stale:** `notes/reanalysis_handoff.md` §3.2 and `notes/v8_stats_check.md` §3 claim
the manuscript swaps the two split-correlation labels. The current manuscript text
matches the code; the swap was fixed and the audit notes were not updated.

**Caveat:** `plot_decision_space.py --n_draws` defaults to **80**, so the sign
probabilities above are quantised at 1/80.

## Regenerating a figure from scratch

Nothing below needs the cluster or a GPU.

```bash
# every figure for one fit, into notes/figures/<label>/
python -m tms_risk.behavior.scripts.make_model_report --label flexible2nf

# just Fig 4B/4C
python -m tms_risk.behavior.scripts.plot_fig4bc_style --label flexible2nf

# just Fig 5
python -m tms_risk.behavior.scripts.plot_fig5_style --label flexible2nf
```

To regenerate the **TSVs** you need the trace, so run these on the node that holds it
(see `CLAUDE.md` → "Extract remotely, plot locally") and copy the TSV back:

```bash
python -m tms_risk.behavior.scripts.extract_pmc_parameters \
    --trace_dir /data/ds-tmsrisk/derivatives/cogmodels.overnight \
    --label flexible2_noisefix.head --tag flexible2nf
python -m tms_risk.behavior.scripts.noise_curve_inference   --label flexible2nf ...
python -m tms_risk.behavior.scripts.plot_decision_space     --label flexible2nf ...
```

## Deploying code to the fitting nodes

Via GitHub, not `scp`/`tar`:

```bash
git push origin cleanup/ddm-port
ssh sciencecloud 'cd /data/git/tms_risk && git fetch origin && git checkout cleanup/ddm-port && git pull'
```

`libs/bauer` is a submodule and is pinned separately; check `git -C libs/bauer log -1`
on the node matches what the traces are stamped with before trusting a refit.
