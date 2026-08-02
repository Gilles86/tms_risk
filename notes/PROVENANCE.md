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
| **Fig. 4A** — posterior predictive check | `behavior/scripts/plot_ppc_fig3a.py` | the **trace** (run on the fitting node) | `notes/data/ppc_fig3a.<label>.tsv` → `notes/figures/ppc_fig3a.<label>.*` |
| **Fig. 4B/4C** — noise vs. magnitude, and the cTBS contrast | `behavior/scripts/plot_fig4bc_style.py` | `noisecurve_reparam.<label>.tsv` | `notes/figures/fig4bc_style.<label>.*` |
| **Table 1** — ELPD model comparison | `behavior/scripts/loo_table.py` | trace `log_likelihood` groups | `<out_stem>.tsv` + `.md` |
| **Fig. 5A/5B** — decision-space heatmaps (total noise, perceived EV ratio) | `behavior/scripts/plot_fig5_style.py` | `decision_space.<label>.tsv`, `pmcpars_curves.<label>.tsv` | `notes/figures/fig5_style.<label>.*` |

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

Everything fit from 2026-07 onward stamps its provenance into the trace:
`tms_risk_bauer_commit`, `tms_risk_family`, `tms_risk_spline_order`,
`tms_risk_spline_degree`, `tms_risk_constrained`. Read them with
`behavior/scripts/summarize_traces.py`.

---

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
