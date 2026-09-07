#!/bin/bash
#SBATCH --job-name=create_tms_risk_prf
#SBATCH --account=zne.uzh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gdehol/logs/create_prf_env_%j.log

# Builds `tms_risk_prf` -- Keras 3 on JAX, no TensorFlow -- on a GPU compute node,
# from the same spec verified working on the ScienceCloud T4 boxes.
#
# WHY, rather than rebuilding tms_risk_cuda:
#
#   * `tms_risk_cuda` currently allocates a GPU and then runs on CPU. Its TF 2.14 is
#     built with CUDA support but the env contains ZERO `nvidia-*` CUDA runtime wheels,
#     no CUDA libs of its own, and nothing puts CUDA on LD_LIBRARY_PATH. TF logs
#     "Could not find cuda drivers on your machine, GPU will not be used" and falls
#     back silently. A 74-subject array would burn A100 allocations at CPU speed.
#   * `create_gpu_env.sh` does `module load cuda/12.6.3 || true`, but this cluster now
#     offers only cuda/11.8.0, cuda/13.0.2 and cuda/13.1.1. The `|| true` swallows the
#     failure, which is a good candidate for how the env drifted into this state.
#   * braincoder is now backend-agnostic (Keras 3, `keras.ops`), so TensorFlow is not
#     required at all. `jax[cuda12]` ships its own CUDA via pip wheels and needs only
#     the driver, so it does not depend on which CUDA module happens to exist.
#
# NON-DESTRUCTIVE: creates a NEW env. `tms_risk_cuda` is left untouched, so nothing
# that currently runs (however slowly) is broken by this.
#
#   sbatch create_env/create_cluster_prf_env.sh

set -e
mkdir -p "$HOME/logs"

echo "=== Building tms_risk_prf (Keras 3 / JAX) ==="
echo "Started $(date) on $(hostname), job $SLURM_JOB_ID"
nvidia-smi || { echo "NO GPU VISIBLE -- aborting, this must run on a GPU node"; exit 1; }

source "$(conda info --base)/etc/profile.d/conda.sh"
cd "$HOME/git/tms_risk"

# The submodule must carry the ParameterFitter tuple-label fix (2a548e7), without which
# every RegressionGaussianPRF fit dies with
# "TypeError: sequence item 0: expected str instance, tuple found".
git -C libs/braincoder fetch -q origin
git -C libs/braincoder checkout -q origin/keras-backend
echo "braincoder at: $(git -C libs/braincoder log --oneline -1)"

ENV_NAME=tms_risk_prf
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Env ${ENV_NAME} exists; updating in place."
    conda env update -n "${ENV_NAME}" -f create_env/environment_sciencecloud_prf.yml --prune
else
    conda env create -f create_env/environment_sciencecloud_prf.yml
fi

conda activate "${ENV_NAME}"

# Keras 3 defaults to TensorFlow, which is deliberately absent here, so the backend has
# to be pinned for bare `python -m ...` launches that never run activation hooks.
mkdir -p "$HOME/.keras"
cat > "$HOME/.keras/keras.json" <<'JSON'
{"floatx": "float32", "epsilon": 1e-07, "backend": "jax", "image_data_format": "channels_last"}
JSON
ACT_DIR="$CONDA_PREFIX/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
echo 'export KERAS_BACKEND=jax' > "$ACT_DIR/keras_backend.sh"

echo ""
echo "=== Sanity checks ==="
KERAS_BACKEND=jax python - <<'PY'
import jax, keras, numpy as np
print("jax", jax.__version__, "devices:", jax.devices())
assert any(d.platform == "gpu" for d in jax.devices()), "NO GPU DEVICE -- build failed"
print("keras", keras.__version__, "backend:", keras.backend.backend())
import braincoder
from braincoder.models import GaussianPRF, RegressionGaussianPRF, LogGaussianPRF
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.utils import get_rsq
print("braincoder", braincoder.__version__)

# numerical check against the analytic Gaussian, the formula tms_risk relies on
import pandas as pd
x = np.linspace(1, 5, 40, dtype="float32")
pars = pd.DataFrame({"mu": [2.5], "sd": [0.8], "amplitude": [1.7], "baseline": [0.3]})
pred = GaussianPRF(paradigm=x, parameters=pars).predict().values.ravel()
ref = 1.7 * np.exp(-0.5 * ((x - 2.5) / 0.8) ** 2) + 0.3
err = float(np.max(np.abs(pred - ref)))
print("GaussianPRF vs analytic, max abs err:", err)
assert err < 1e-5, "prediction mismatch"
print("ALL CHECKS PASSED")
PY

echo ""
echo "=== Done $(date) ==="
echo "Run with: conda activate ${ENV_NAME}   (or KERAS_BACKEND=jax <env>/bin/python)"
