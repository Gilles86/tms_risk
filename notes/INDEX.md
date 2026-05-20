# Notes index

Manually curated entry point into the `notes/` directory. Treat as
durable text — update when adding a new analysis writeup or figure.

## Paper

- [`paper/TMS paper -v7.pdf`](paper/) — current manuscript draft
  (de Hollander, Moisa & Ruff). Six figures, one table.

## Figure ↔ notebook map

| Figure | What it shows | Notebook (re-derives from saved traces) |
|--------|---------------|------------------------------------------|
| Fig. 1 | Experimental paradigm, nPRF maps, stimulation sites | hand-composed in Illustrator from `tms_risk/visualize/` outputs |
| Fig. 2A | nPRF tuning in a representative subject | `tms_risk/modeling/notebooks/analyze_encoding_model.ipynb` |
| Fig. 2B | Group nPRF amplitude × stimulation condition | `tms_risk/modeling/notebooks/analyze_encoding_model.ipynb` |
| Fig. 2C | Trial-by-trial decoding accuracy | `tms_risk/modeling/notebooks/analyze_decoding.ipynb` |
| Fig. 3  | Psychometric curves + slopes / RNP (paper labels this Fig 3; notebook is misnamed `figure2.ipynb` from an earlier draft) | `tms_risk/notebooks/figure2.ipynb` |
| Fig. 4A | Posterior predictive checks, Weber vs. Flexible PMC | `tms_risk/behavior/notebooks/figure4.ipynb` |
| Fig. 4B | Noise as a function of magnitude (`sd_curves.pdf`) | `tms_risk/behavior/notebooks/figure4.ipynb` |
| Fig. 4C | cTBS effect on noise vs. magnitude (`sd_curves_diff.pdf`) | `tms_risk/behavior/notebooks/figure4.ipynb` |
| Table 1 | ELPD model comparison (Flexible PMC variants × null) | `tms_risk/behavior/notebooks/comprehensive_model_comparison.ipynb` |
| "Linking Neural and Behavioral TMS Effects" (correlations) | brain–behavior bridge: amplitude drop ↔ behavioral noise increase | `tms_risk/behavior/notebooks/analyze_nlc.ipynb` + `tms_risk/modeling/individual_brain_behavior.ipynb` |
| Fig 4 model-figure supporting plots | flexible2 PPCs + per-subject noise curves | `tms_risk/behavior/notebooks/model_figure.ipynb` |
| (new) Phase 5 ELPD: DDM/RDM × Flexible PMC | extends Table 1 with accumulator-model variants | `tms_risk/behavior/notebooks/ddm_rdm_model_comparison.ipynb` |
| (new) Predicted decoding (Fisher + MC) | supplement / addition to Fig 2B–C | `tms_risk/modeling/notebooks/fisher_and_mc_decode.ipynb` |

## Working data

- `data/` (planned) — small TSVs aggregated on the cluster, rsync'd
  back for local plotting.

## See also

- Top-level [`CLAUDE.md`](../CLAUDE.md) — developer-facing recipes,
  module layout, model-label conventions.
- [`STATUS.md`](STATUS.md) — what's done / in progress / blocked.
- [`create_env/README.md`](../create_env/README.md) — conda envs.
