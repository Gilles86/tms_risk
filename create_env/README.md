# Conda environments

Three conda environments cover the project's three runtime contexts.
All three install the same pymc/bambi/arviz versions used to fit the
paper's models, plus the in-house `braincoder` and `bauer` libraries
from `libs/` as editable installs.

| Use case | YML file | Env name |
|----------|----------|----------|
| Local Mac dev (Apple Silicon, Metal-accelerated TF) | `../environment_apple_silicon.yml` | `tms_risk` |
| Cluster CPU jobs (PMC fits, plotting, aggregation) | `environment_cpu.yml` | `tms_risk_cpu` |
| Cluster GPU jobs (nPRF fits, decoding) | `environment_cuda.yml` | `tms_risk_cuda` |

## Local (Mac, Apple Silicon)

```bash
conda env create -f environment_apple_silicon.yml
conda activate tms_risk
```

This env uses `tensorflow-metal` for GPU acceleration on M1/M2/M3.

## Cluster (SLURM)

Both env builds are sbatch wrappers. The CUDA env **must** be built
on a GPU node so the NVIDIA driver is visible at install time.

```bash
# CPU env
sbatch create_env/create_cpu_env.sh

# CUDA env (GPU node + cuda module)
sbatch create_env/create_gpu_env.sh
```

Logs land in `~/logs/create_{cpu,cuda}_env_<jobid>.log`. The
wrappers refuse to clobber a healthy existing env unless you
`conda env remove -n tms_risk_{cpu,cuda}` first.

## Updating after a yml change

Recreate from scratch — `conda env update --prune` doesn't reliably
handle pip-installed editable submodules:

```bash
# local
conda env remove -n tms_risk && conda env create -f ../environment_apple_silicon.yml

# cluster
sbatch create_env/create_cpu_env.sh   # or create_gpu_env.sh
```

## Why three envs

- **`tms_risk` (Mac)** uses `tensorflow-metal`, which only exists on
  Apple Silicon — separate from the cluster builds.
- **`tms_risk_cpu`** uses Intel MKL + TF 2.18 CPU build. Right for
  PMC / Flexible PMC fits (CPU-bound NUTS sampling) and any local
  analysis on a Linux box.
- **`tms_risk_cuda`** pins TF 2.14 because that's what the working
  braincoder pin in `libs/` expects with CUDA 12.6 on the cluster.
  Built on a GPU node; use for nPRF fits and decoding.
