# Conda environments

Four conda environments cover the project's runtime contexts. The PMC
fit envs share the same pymc / bambi / arviz pins used to fit the
paper's models. The DDM env is separate because hssm's deps conflict
with the TF 2.18 / numpy 1.26 stack.

| Use case | YML file | Env name |
|----------|----------|----------|
| Local Mac dev (Apple Silicon, Metal-accelerated TF) | `../environment_apple_silicon.yml` | `tms_risk` |
| Cluster CPU jobs (PMC fits, plotting, aggregation) | `environment_cpu.yml` | `tms_risk_cpu` |
| Cluster GPU jobs (nPRF fits, decoding) | `environment_cuda.yml` | `tms_risk_cuda` |
| Cluster DDM/RDM fits (bauer + hssm) | `environment_ddm.yml` | `tms_risk_ddm` |

## Local (Mac, Apple Silicon)

```bash
conda env create -f environment_apple_silicon.yml
conda activate tms_risk
```

This env uses `tensorflow-metal` for GPU acceleration on M1/M2/M3.

## Cluster (SLURM)

Each env build is its own sbatch wrapper. The CUDA env **must** be
built on a GPU node so the NVIDIA driver is visible at install time.

```bash
sbatch create_env/create_cpu_env.sh   # tms_risk_cpu
sbatch create_env/create_gpu_env.sh   # tms_risk_cuda (GPU node)
sbatch create_env/create_ddm_env.sh   # tms_risk_ddm
```

Logs land in `~/logs/create_{cpu,cuda,ddm}_env_<jobid>.log`. Each
wrapper removes any existing env of the same name before creating.

`tms_risk_ddm` is selected automatically by
`tms_risk/behavior/slurm_jobs/fit_model.sh` for labels starting with
`ddm_` or `rdm_`. The other labels stay on `tms_risk_cpu`.

## Updating after a yml change

Recreate from scratch — `conda env update --prune` doesn't reliably
handle pip-installed editable submodules:

```bash
# local
conda env remove -n tms_risk && conda env create -f ../environment_apple_silicon.yml

# cluster
sbatch create_env/create_cpu_env.sh   # or create_{gpu,ddm}_env.sh
```

## Why four envs

- **`tms_risk` (Mac)** uses `tensorflow-metal`, which only exists on
  Apple Silicon — separate from the cluster builds.
- **`tms_risk_cpu`** uses Intel MKL + TF 2.18 CPU build. Right for
  PMC / Flexible PMC fits (CPU-bound NUTS sampling) and any local
  analysis on a Linux box.
- **`tms_risk_cuda`** pins TF 2.14 because that's what the working
  braincoder pin in `libs/` expects with CUDA 12.6 on the cluster.
  Built on a GPU node; use for nPRF fits and decoding.
- **`tms_risk_ddm`** is for DDM/RDM × Flexible PMC fits. hssm pulls
  numpy 2.x, pymc 5.28, arviz 0.23 and ml-dtypes 0.5 — all
  incompatible with the TF 2.18 stack — so the DDM analyses get
  their own minimal env (bauer + hssm + the Bayesian stack, no
  braincoder/TF).
