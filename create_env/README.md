# Conda environments

These conda environments cover the project's runtime contexts. The PMC
fit envs share the same pymc / bambi / arviz pins used to fit the
paper's models. The DDM env is separate because hssm's deps conflict
with the TF 2.18 / numpy 1.26 stack, and `tms_risk_prf` is separate
because it drops TensorFlow entirely in favour of Keras 3 on JAX.

| Use case | YML file | Env name |
|----------|----------|----------|
| **One env to rule them all** — TF 2.20 + pymc 5.28 + hssm + ssm-simulators + braincoder + bauer (Recommended for new work) | `environment_unified.yml` | `tms_risk_unified` |
| Local Mac dev (Apple Silicon, Metal-accelerated TF) | `../environment_apple_silicon.yml` | `tms_risk` |
| Cluster CPU jobs (PMC fits, plotting, aggregation) — older numpy 1.26 stack | `environment_cpu.yml` | `tms_risk_cpu` |
| Cluster GPU jobs (nPRF fits, decoding) — older TF 2.14 / numpy 1.25 stack | `environment_cuda.yml` | `tms_risk_cuda` |
| Cluster DDM/RDM fits (bauer + hssm) — older split env | `environment_ddm.yml` | `tms_risk_ddm` |
| **ScienceCloud GPU VMs** — nPRF fits + Monte-Carlo decoding on the T4 boxes (Keras 3 / **JAX**, no TensorFlow) | `environment_sciencecloud_prf.yml` | `tms_risk_prf` |

## Unified env (recommended for new work)

`environment_unified.yml` collapses the previous cpu/cuda/ddm split. The
historical conflict (TF≤2.18 needing `numpy<2.1` vs `ssm-simulators`
needing `numpy>=2.0`) dissolves once TF 2.20 drops the cap and
pytensor 2.38 / pymc 5.28 become numpy-2 native. Build on a GPU node so
TF's bundled CUDA wheels see the NVIDIA driver:

```bash
sbatch create_env/create_unified_env.sh   # tms_risk_unified (GPU node)
```

The unified env is the only env that pulls braincoder's `keras-backend`
branch (and therefore `model.get_expected_uncertainty`). For modeling /
decode / mc_decode jobs on the cluster, switch the runner's
`PYTHON_BIN` to `$HOME/data/conda/envs/tms_risk_unified/bin/python`.
Existing fits made with `tms_risk_cuda` remain reproducible.

## ScienceCloud GPU VMs — `tms_risk_prf` (encoding models / decoding)

For the four Tesla-T4 ScienceCloud boxes (SSH aliases `sciencecloud_gpu`,
`sciencecloud_gpu2`, `sciencecloud_gpu3`, `sciencecloud_gpu4`). These have
**no SLURM** — there is no sbatch wrapper, you run the build script on the
box:

```bash
ssh sciencecloud_gpu 'bash /data/git/tms_risk/create_env/create_sciencecloud_prf_env.sh'
```

The script is idempotent (it removes any existing `tms_risk_prf` first),
checks `libs/braincoder` out on the `keras-backend` branch over HTTPS — the
VMs have no GitHub SSH key, and the submodule ships uninitialised — and
finishes with a GPU smoke test that fails loudly. Repeat for each box; they
share nothing. It never touches the behaviour-only `tms_risk_gpu` env that
the bauer/numpyro fits run in.

**The backend is JAX, not TensorFlow.** The braincoder on `keras-backend`
talks to `keras.ops`, so the backend is a free choice, and JAX's CUDA
plugin wheels were already proven on these boxes by the numpyro fits.
Verified stack: python 3.11 · keras 3.15.1 · jax 0.8.3 (`jax[cuda12]`,
CUDA 12.9 wheels on driver 580.173.02) · numpy 2.4 · braincoder 0.5.1
`keras-backend`.

Keras 3 defaults to TensorFlow, which is *not installed here* — so
selecting the backend is not optional. The build script sets it twice:

- `$CONDA_PREFIX/etc/conda/activate.d/keras_backend.sh` for `conda activate`
- `~/.keras/keras.json` for the nohup pattern, which calls the env's python
  binary directly and therefore never runs the activation hooks.

Launch work with `nohup`, one model per box (one GPU each):

```bash
ssh sciencecloud_gpu 'cd /data/git/tms_risk && nohup \
  /data/miniforge3/envs/tms_risk_prf/bin/python -m tms_risk.modeling.monte_carlo_decode \
  01 1 --bids_folder /data/ds-tmsrisk > /data/logs/mc_decode_01_1.log 2>&1 &'
```

Measured on a T4 with a synthetic `ParameterFitter.fit` at realistic scale
(200 000 voxels × 300 iterations): **25 s on the GPU vs 406 s CPU-only**
(`JAX_PLATFORMS=cpu`), a 16× speedup, with ~55 % GPU utilisation and 11 GB
of the 15 GB VRAM in use (JAX preallocates 75 % by default).

Two repo-side snags this env has to work around, both worth fixing at the
source eventually:

- `nilearn` is **capped at `<0.13`**. `modeling/{fit_nprf,fit_regression_nprf,
  fit_regression_nprf_cv}.py` still do `from nilearn.input_data import
  NiftiMasker`; that module was deprecated in 0.9 and deleted in 0.13.
  0.12.x keeps the shim. Lift the cap once those three imports move to
  `from nilearn.maskers import NiftiMasker`.
- `fit_regression_nprf_cv.py` cannot be run with `python -m` — its line 12
  is a bare `from fit_regression_nprf import get_model, get_grid`, which only
  resolves with the script's own directory on `sys.path`. Run it as
  `cd tms_risk/modeling && python fit_regression_nprf_cv.py ...`. Every other
  `tms_risk.modeling.*` entry point works under `python -m`.

### Getting the imaging derivatives onto a box

The VMs ship with only the behavioural `cogmodels.*` traces. Encoding /
decoding work additionally needs `encoding_model2.model-1.smoothed` (91 MB),
`encoding_model2.model-1.smoothed.cv` (602 MB), `ips_masks` (519 MB) and
`glm_stim1.denoise.smoothed` (10 GB) — ~11 GB, against 57 GB free on
`sciencecloud_gpu`/`_gpu2` and ~200 GB on `_gpu3`/`_gpu4`.

**Push from the Mac.** Measured both ways: large files run at ~35 MB/s from
either source, but on the many-small-file trees the Mac does 34 MB/s where
cluster→VM manages only 1.6–2.7 MB/s — the cluster's `/shares` NFS
per-file latency, not the network. Mac→VM RTT is 7.8 ms; the VPN is not the
bottleneck people assume it is.

```bash
rsync -a --info=progress2 \
  /data/ds-tmsrisk/derivatives/{encoding_model2.model-1.smoothed,encoding_model2.model-1.smoothed.cv,ips_masks,glm_stim1.denoise.smoothed} \
  sciencecloud_gpu:/data/ds-tmsrisk/derivatives/
```

Expect ~10 min for the full 11 GB. Nothing under these four trees is a
git-annex symlink, so plain `-a` copies real content.

The cluster copy of `glm_stim1.denoise.smoothed` looks bigger (18 GB vs
10 GB) but is **not** more complete: both hold the same 135
`*_desc-stims1_pe.nii.gz`, and the extra 8 GB is 38 GLMsingle
`TYPED_FITHRF_GLMDENOISE_RR.npy` / `TYPEA_ONOFF.npy` intermediates that the
encoding pipeline never reads. For `ips_masks` the Mac is outright more
complete (3846 files vs 1969).

If you do need the cluster as the source, the VMs have no key for it —
forward your agent rather than installing one:

```bash
ssh -A sciencecloud_gpu 'rsync -a gdehol@cluster.s3it.uzh.ch:/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/<tree>/ /data/ds-tmsrisk/derivatives/<tree>/'
```

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
