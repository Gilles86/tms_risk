# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Analysis code for the combined cTBS-TMS + 7T fMRI study **"Risk Attitudes Causally Rely on Parietal Magnitude Representations"** (de Hollander, Moisa & Ruff). The paper is at `notes/paper/TMS paper -v7.pdf`. The pipeline targets numerosity-tuned right parietal cortex with cTBS (vertex control vs. parietal) and measures effects on (a) nPRF responses, (b) trial-by-trial decoding accuracy, (c) psychophysical choice consistency / risk-neutral probability, and (d) parameters of the Perceptual-and-Memory-based Choice (PMC) model and its **Flexible PMC** extension (B-spline noise function over magnitude).

Three layered analyses sit on top of the same BIDS dataset (`/data/ds-tmsrisk` locally; `/shares/zne.uzh/gdehol/ds-tmsrisk` on the cluster):

1. `tms_risk/encoding_model/` — nPRF fits + Bayesian decoding (uses `braincoder`, TensorFlow). Compute-heavy, runs on cluster GPU/CPU.
2. `tms_risk/glm/` — GLMsingle single-trial betas feeding the encoding model.
3. `tms_risk/cogmodels/` — behavioral PMC / Flexible PMC / probit models (uses `bauer` + PyMC/bambi). Hierarchical Bayesian; runs on CPU.

`tms_risk/tms_targeting/`, `prepare/`, `registration/`, `surface/` support the imaging pipeline; `behavior/` and `visualize/` are analysis/figure notebooks.

## Repo-specific architecture

**Single source of truth for data access**: `tms_risk/utils/data.py`. Every notebook and script should go through `get_all_behavior(...)`, `get_subjects(...)`, or instantiate `Subject(subject_id, bids_folder)` (defined at line 154). Subject methods accept the cross-cutting flags `denoise`, `smoothed`, `natural_space`, `mask`, `n_voxels` that select between GLMsingle variants — touching the same flag in every analysis script is intentional.

**Outliers are hardcoded** to subjects `[22, 49]` in `tms_risk/utils/data.py` (rationale in `tms_risk/behavior/outliers.ipynb`). Pass `exclude_outliers=False` to `get_subjects`/`get_all_behavior` to keep them.

**Model dispatch by string label**: `tms_risk/cogmodels/fit_model.py::build_model(model_label, df)` is a ~300-line `elif` chain mapping short labels (`"1"`, `"5c"`, `"flexible2.6a"`, `"session1_full"`, …) to `bauer.models` constructors with specific regressor formulas and `prior_estimate`/`memory_model` options. The submit scripts (`submit_all_flexible_models.sh`, `submit_all_weber_models.sh`, `fit_model.sh`) iterate over these labels for SLURM arrays. When adding a model variant, add a new label rather than mutating an existing one — downstream notebooks load traces by label from `derivatives/cogmodels/model-<label>_trace.netcdf`.

**`libs/` contains git submodules** (`bauer/`, `braincoder/`) — both are also installed editably elsewhere on disk (`~/git/bauer`, `~/git/braincoder`). The cognitive models in the paper (`FlexibleNoiseRiskRegressionModel`) live in `bauer.models.risky_choice`; DDM and Race-diffusion variants of the same models exist in `bauer.models.{ddm,race}` (`DDMFlexibleNoiseRiskRegressionModel`, `RaceDiffusionFlexibleNoiseRiskRegressionModel`) but are not yet wired into `fit_model.py`.

**Conda environment names** are confusing: both `environment.yml` and `environment_apple_silicon.yml` create an env called `tms_risk`; GPU/CUDA yml files create `tms_risk_gpu` / `tms_risk_cuda`. The CUDA env exists on the cluster and is the one used for `encoding_model` GPU jobs. See `environments/README.md`.

## Common commands

```bash
# Local install (after creating env)
pip install -e .

# Fit one cognitive model locally (label dispatch in fit_model.py)
python -m tms_risk.cogmodels.fit_model flexible2.6 --bids_folder /data/ds-tmsrisk

# Submit full flexible-model sweep on cluster
sbatch tms_risk/cogmodels/submit_all_flexible_models.sh

# Build cluster envs (must run on a GPU node for cuda/gpu)
sbatch environments/build_env.sh cpu
sbatch --gres=gpu:1 environments/build_env.sh cuda
```

There are no automated tests; correctness is checked via posterior predictive notebooks in `cogmodels/notebooks/` (e.g. `analyze_flexible_model_ppcs.ipynb`, `figure4.ipynb`).

## Conventions worth knowing

- Scripts take a positional `subject` arg and `--bids_folder` kwarg, and are submitted as SLURM arrays from `*/cluster_scripts/`. Encoding model scripts use `tms_risk/encoding_model/cluster_scripts/`; GLM scripts use `tms_risk/glm/cluster_scripts/`.
- Cognitive model traces are written to `<bids>/derivatives/cogmodels/model-<label>_trace.netcdf` as ArviZ NetCDF.
- The `tms_keys.yml` and `all_subjects.yml` resource files under `tms_risk/data/` are the authoritative subject lists; `get_tms_subjects()` reads `tms_keys.yml`, `get_all_subject_ids()` reads `all_subjects.yml`.
