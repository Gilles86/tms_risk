# Status

## Done

- **Cleanup/ddm-port branch (2026-05-18):** house-style reorg of the
  repo. Renamed `cogmodels/` → `behavior/`, `encoding_model/` →
  `modeling/`, `cluster_scripts/` → `slurm_jobs/`, and
  `environments/` → `create_env/`. Pruned ~25 dead model labels in
  `fit_model.py` (archived in `legacy_models.py`). Removed legacy
  Docker stack. Added `tests/test_data.py` (Subject smoke tests).

- **Flexible PMC model (paper v7):** fit complete, posteriors in
  `derivatives/cogmodels/`. 5-spline noise function, both perceptual
  and memory noise modulated by stimulation. ELPD table in Table 1
  of the manuscript.

## In progress

- **DDM × Flexible PMC and RDM × Flexible PMC fits.** Twelve new
  model labels added to `fit_model.py` covering noise-only,
  threshold-only, and combined regressors. Submission script at
  `tms_risk/behavior/slurm_jobs/submit_all_ddm_rdm_models.sh`. Not
  yet submitted; analysis notebook for the extended Table 1 not yet
  written.

- **Fisher-information / predicted-decoding figure.** Computation
  lives in `tms_risk/modeling/fisher_information.py` (mirrors
  neural_priors's `get_fisher_information` pipeline). Plotting
  notebook not yet written; intended as a refinement / addition to
  Figure 2B–C of the paper.

## Blocked / open questions

- Whether the DDM/RDM-extended Table 1 will actually go in the paper
  or stays as a supplement / response-to-reviewers piece.
- Whether the Fisher-information figure replaces or augments the
  current empirical decoding-accuracy panel.

## Recent commits worth knowing about

```
git log --oneline cleanup/ddm-port
```
