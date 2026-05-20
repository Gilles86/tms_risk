# Reproducing the paper

Step-by-step recipe for regenerating every figure and the model-comparison
table in **"Risk Attitudes Causally Rely on Parietal Magnitude
Representations"** (de Hollander, Moisa & Ruff). Manuscript draft at
[`notes/paper/TMS paper -v7.pdf`](notes/paper/).

Everything below assumes the BIDS dataset (`ds-tmsrisk`) is in place
(see [Data](#data)) and that fMRIPrep has already produced
`derivatives/fmriprep/` (out of scope here; see fmriprep docs).

---

## 0. Setup

### Environments

Three conda envs (see [`create_env/README.md`](create_env/README.md) for
details):

| Use | YML | Env name |
|----|-----|----------|
| Local Mac dev (analysis + plotting) | `environment_apple_silicon.yml` | `tms_risk` |
| Cluster CPU (PMC / probit fits) | `create_env/environment_cpu.yml` | `tms_risk_cpu` |
| Cluster GPU (nPRF fits, decoding) | `create_env/environment_cuda.yml` | `tms_risk_cuda` |

```bash
# local (Mac)
conda env create -f environment_apple_silicon.yml
conda activate tms_risk
pip install -e .            # if not already done by the env's pip section

# cluster
sbatch create_env/create_cpu_env.sh
sbatch create_env/create_gpu_env.sh    # GPU node required
```

### Data

| Where | Path |
|---|---|
| Local Mac | `/data/ds-tmsrisk` |
| Cluster | `/shares/zne.uzh/gdehol/ds-tmsrisk` |

Every analysis script accepts `--bids_folder` to point elsewhere.

`tms_risk/data/tms_keys.yml` is the authoritative subject ↔ TMS
condition map for sessions 2/3. `tms_risk/data/all_subjects.yml` lists
all 78 screened subjects (35 went to TMS sessions). Outliers
**22** and **49** are hardcoded in `tms_risk/utils/data.py` and dropped
by default via `exclude_outliers=True`.

---

## 1. Pipeline (cluster)

Each stage produces a `derivatives/<name>/` subtree. Subsequent stages
read from the previous one through the `Subject` class in
`tms_risk/utils/data.py` — never build paths by hand.

### 1a. Single-trial GLM (GLMsingle)

Computes one β per trial, voxel, run. Required input for the nPRF fits.

```bash
# Per subject, sessions 1+2+3 in one go (denoising + RETROICOR variant)
sbatch --array=1-78 tms_risk/glm/cluster_scripts/<glm_submit>.sh
```

(GLM SLURM wrappers live in `tms_risk/glm/cluster_scripts/`. They write
to `derivatives/glmsingle.denoise/`.)

### 1b. Numerical PRF fits (encoding model)

Three model variants under `tms_risk/modeling/fit_regression_nprf.py`,
selected via `--model`:

| `--model` | What varies across stimulation conditions |
|-----------|--------------------------------------------|
| 0 | Nothing — single set of (μ, σ, amp, baseline) per voxel, pooled across sessions |
| **1** | **Amplitude per session** (the paper's main analysis — Fig 2B) |
| 2 | Full session interaction on μ, σ, amplitude, baseline |

```bash
# Paper's main fit (model-1): one job per subject, GPU required
sbatch tms_risk/modeling/slurm_jobs/submit_regression_model.sh
```

Outputs land in `derivatives/encoding_model2.model-{N}.smoothed/`.
Cross-validated version: `submit_regression_model_cv.sh` →
`derivatives/encoding_model2.model-{N}.smoothed.cv/`.

### 1c. Decoding (per-trial payoff posterior)

```bash
# Cross-session decoding for Fig 2C
sbatch tms_risk/modeling/slurm_jobs/submit_cross_session_decode.py
```

This inverts the nPRF model to produce a posterior over the presented
numerosity per trial, then aggregates into decoded mean + accuracy.
Reads from `encoding_model2.model-1.smoothed.cv/`.

### 1d. Behavioral probit (Fig 3)

```bash
# tms_risk_cpu env on cluster (or local)
python -m tms_risk.behavior.fit_probit probit_order --bids_folder /data/ds-tmsrisk
```

`probit_order` is the model in the paper (Bayesian hierarchical probit
with `x * risky_first * stimulation_condition` interaction + random
slopes per subject). Trace goes to
`derivatives/cogmodels/model-probit_order_trace.netcdf`.

### 1e. Behavioral PMC + Flexible PMC (Fig 4, Table 1)

These 8 labels constitute the paper's Table 1:

| Label | Model | TMS regressor |
|-------|-------|--------------|
| `11_null`   | Weber PMC | none (baseline) |
| `11b`       | Weber PMC | memory noise only |
| `11c`       | Weber PMC | perceptual noise only |
| `11a`       | Weber PMC | both noise terms |
| `flexible2_null` | Flexible PMC (5-spline noise) | none |
| `flexible2b` | Flexible PMC | perceptual noise spline only |
| `flexible2a` | Flexible PMC | memory noise spline only |
| `flexible2`  | Flexible PMC | both noise splines (paper's main claim) |

```bash
# Run all 8 — about 6 h each on tms_risk_cpu
cd tms_risk/behavior/slurm_jobs
for label in 11_null 11b 11c 11a flexible2_null flexible2b flexible2a flexible2; do
    sbatch fit_model.sh $label
done
```

Traces go to `derivatives/cogmodels/model-<label>_trace.netcdf`.

---

## 2. Figures (run locally from saved traces)

After 1a–1e have produced the derivatives (and you've `rsync`'d them
to the local Mac), every figure regenerates from notebooks under
`tms_risk/`:

| Paper item | Notebook | Reads |
|------------|----------|-------|
| **Fig 2A** — nPRF maps in a representative subject | [`tms_risk/modeling/notebooks/analyze_encoding_model.ipynb`](tms_risk/modeling/notebooks/analyze_encoding_model.ipynb) | `encoding_model2.model-1.smoothed/` |
| **Fig 2B** — group nPRF amplitude × stim condition | same notebook | same |
| **Fig 2C** — trial-by-trial decoding accuracy | [`tms_risk/modeling/notebooks/analyze_decoding.ipynb`](tms_risk/modeling/notebooks/analyze_decoding.ipynb) | `decoded_pdfs.volume/` |
| **Fig 3** — psychometric curves + slope / RNP | [`tms_risk/notebooks/figure2.ipynb`](tms_risk/notebooks/figure2.ipynb) ⚠ | `cogmodels/model-probit_order_trace.netcdf` |
| **Fig 4A** — PPCs, Weber vs. Flexible PMC | [`tms_risk/behavior/notebooks/figure4.ipynb`](tms_risk/behavior/notebooks/figure4.ipynb) | `cogmodels/model-{11a,flexible2,...}_trace.netcdf` |
| **Fig 4B** — noise as a function of magnitude | same notebook | same |
| **Fig 4C** — cTBS effect on noise vs. magnitude | same notebook | same |
| **Table 1** — ELPD comparison | [`tms_risk/behavior/notebooks/comprehensive_model_comparison.ipynb`](tms_risk/behavior/notebooks/comprehensive_model_comparison.ipynb) | the 8 Table-1 traces |
| Neural ↔ Behavioral link (paper's `r=0.38`, `r=0.76`) | [`tms_risk/behavior/notebooks/analyze_nlc.ipynb`](tms_risk/behavior/notebooks/analyze_nlc.ipynb) + [`tms_risk/modeling/individual_brain_behavior.ipynb`](tms_risk/modeling/individual_brain_behavior.ipynb) | both encoding-model + cogmodel traces |

⚠ `notebooks/figure2.ipynb` is misnamed from an earlier draft — it
actually produces what the v7 manuscript labels Fig 3.

`notes/INDEX.md` keeps this table in one place — when you add a new
figure-producing notebook, update both.

---

## 3. Branches / state

- `main` — paper-pushable state.
- `cleanup/ddm-port` — current working branch with the housekeeping
  reorg (module renames, env cleanup, archive sweep, stripped notebook
  outputs) plus three additive extensions:
  - **DDM × Flexible PMC + RDM × Flexible PMC** comparison (Phase 5)
    — `tms_risk/behavior/fit_model.py::ddm_*`/`rdm_*` labels; submit
    via `tms_risk/behavior/slurm_jobs/submit_all_ddm_rdm_models.sh`;
    analyze via `tms_risk/behavior/notebooks/ddm_rdm_model_comparison.ipynb`.
  - **Fisher information** — `tms_risk/modeling/fisher_information.py`.
  - **Monte Carlo simulate-and-decode** —
    `tms_risk/modeling/monte_carlo_decode.py`. More robust than Fisher
    when the residual has heavy tails. Both feed
    `tms_risk/modeling/notebooks/fisher_and_mc_decode.ipynb`.
- `archive/pre-cleanup` — frozen at the pre-Claude commit
  `5be6eed` for any old exploratory notebooks I removed during the sweep.
  Recover with `git checkout archive/pre-cleanup -- <path>`.

---

## 4. Tests

A live-data smoke test exists under `tests/test_data.py`. Skipped
when `/data/ds-tmsrisk` is absent; otherwise:

```bash
conda activate tms_risk
pip install pytest   # not in the env by default
pytest tests/
```

It exercises the `Subject` class, the outlier list, and the column
contract `behavior.fit_model.get_data` relies on.
