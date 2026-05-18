# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Analysis code for the combined cTBS-TMS + 7T fMRI study **"Risk Attitudes Causally Rely on Parietal Magnitude Representations"** (de Hollander, Moisa & Ruff). The paper draft is at `notes/paper/TMS paper -v7.pdf`. The pipeline targets numerosity-tuned right parietal cortex with cTBS (vertex control vs. parietal) and measures effects on (a) nPRF responses, (b) trial-by-trial decoding accuracy, (c) psychophysical choice consistency / risk-neutral probability, and (d) parameters of the Perceptual-and-Memory-based Choice (PMC) model and its **Flexible PMC** extension (B-spline noise function over magnitude).

Three layered analyses sit on top of the same BIDS dataset (`/data/ds-tmsrisk` locally; `/shares/zne.uzh/gdehol/ds-tmsrisk` on the cluster):

1. `tms_risk/modeling/` — nPRF fits + Bayesian decoding (uses `braincoder`, TensorFlow). Compute-heavy, runs on cluster GPU/CPU.
2. `tms_risk/glm/` — GLMsingle single-trial betas feeding the encoding model.
3. `tms_risk/behavior/` — PMC / Flexible PMC / probit models (uses `bauer` + PyMC/bambi). Hierarchical Bayesian; runs on CPU.

`tms_risk/tms_targeting/`, `prepare/`, `registration/`, `surface/` support the imaging pipeline; `visualize/` holds plotting helpers. Exploratory behavioral notebooks live in `tms_risk/behavior/notebooks/archive/`.

## Repo-specific architecture

**Single source of truth for data access**: `tms_risk/utils/data.py`. Every notebook and script should go through `get_all_behavior(...)`, `get_subjects(...)`, or instantiate `Subject(subject_id, bids_folder)` (defined at line 154). Subject methods accept the cross-cutting flags `denoise`, `smoothed`, `natural_space`, `mask`, `n_voxels` that select between GLMsingle variants — touching the same flag in every analysis script is intentional.

**Outliers are hardcoded** to subjects `[22, 49]` in `tms_risk/utils/data.py` (rationale in `tms_risk/behavior/notebooks/archive/outliers.ipynb`). Pass `exclude_outliers=False` to `get_subjects` / `get_all_behavior` to keep them.

**Model dispatch by string label**: `tms_risk/behavior/fit_model.py::build_model(model_label, df)` maps short labels (`"flexible2.6a"`, `"5c"`, `"session1_full"`, …) to `bauer.models` constructors with specific regressor formulas and `prior_estimate` / `memory_model` options. The submit scripts (`submit_all_flexible_models.sh`, `submit_all_weber_models.sh`, `fit_model.sh`) in `tms_risk/behavior/slurm_jobs/` iterate over these labels for SLURM arrays. When adding a model variant, add a new label rather than mutating an existing one — downstream notebooks load traces by label from `<bids>/derivatives/cogmodels/model-<label>_trace.netcdf`.

**`libs/` contains git submodules** (`bauer/` on `refactor`, `braincoder/` on `main`). The cognitive models in the paper (`FlexibleNoiseRiskRegressionModel`) live in `bauer.models.risky_choice`; DDM and Race-diffusion variants of the same model exist in `bauer.models.{ddm,race}` (`DDMFlexibleNoiseRiskRegressionModel`, `RaceDiffusionFlexibleNoiseRiskRegressionModel`).

**Conda environments**: three of them, all installing the same `pymc=5.17` / `bambi=0.13` / `arviz=0.20` stack used to fit the paper's models.

| Env name | YML | Use case |
|----------|-----|----------|
| `tms_risk` | `environment_apple_silicon.yml` (top-level) | Local Mac dev (Metal-accelerated TF) |
| `tms_risk_cpu` | `create_env/environment_cpu.yml` | Cluster CPU jobs (PMC fits, plotting) |
| `tms_risk_cuda` | `create_env/environment_cuda.yml` | Cluster GPU jobs (nPRF fits, decoding) |

## Common commands

```bash
# Local install (after creating env)
pip install -e .

# Fit one cognitive model locally
python -m tms_risk.behavior.fit_model flexible2.6 --bids_folder /data/ds-tmsrisk

# Submit full flexible-model sweep on cluster
cd ~/git/tms_risk/tms_risk/behavior/slurm_jobs && bash submit_all_flexible_models.sh

# Build cluster envs (CUDA env must run on a GPU node)
sbatch create_env/create_cpu_env.sh
sbatch create_env/create_gpu_env.sh
```

There are no automated tests; correctness is checked via posterior predictive notebooks under `tms_risk/behavior/notebooks/` (e.g. `analyze_flexible_model_ppcs.ipynb`, `figure4.ipynb`).

## Conventions worth knowing

- Scripts take a positional `subject` arg and `--bids_folder` kwarg, and are submitted as SLURM arrays from `*/slurm_jobs/`. Each analysis submodule keeps its SLURM wrappers in its own `slurm_jobs/` subfolder.
- Cognitive model traces are written to `<bids>/derivatives/cogmodels/model-<label>_trace.netcdf` as ArviZ NetCDF. The directory name `cogmodels` is historical (kept on disk so existing traces remain loadable).
- The `tms_keys.yml` and `all_subjects.yml` resource files under `tms_risk/data/` are the authoritative subject lists; `get_tms_subjects()` reads `tms_keys.yml`, `get_all_subject_ids()` reads `all_subjects.yml`.
- The `slurm_jobs/fit_model.sh` wrapper activates `tms_risk_cpu`, uses `--account=zne.uzh`, and invokes the script via `python -m tms_risk.behavior.fit_model`.
