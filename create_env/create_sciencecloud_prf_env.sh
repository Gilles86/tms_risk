#!/usr/bin/env bash
# Build the `tms_risk_prf` env (nPRF fits + Monte-Carlo decoding) on a
# ScienceCloud GPU VM. NOT a SLURM script — these boxes have no scheduler.
#
# Run it ON the box (the JAX CUDA plugin wheels want the driver present):
#
#   ssh sciencecloud_gpu 'bash /data/git/tms_risk/create_env/create_sciencecloud_prf_env.sh'
#
# Idempotent: removes any existing tms_risk_prf first. Leaves the
# behaviour-only `tms_risk_gpu` env completely untouched.
set -euo pipefail

REPO=${REPO:-/data/git/tms_risk}
CONDA_ROOT=${CONDA_ROOT:-/data/miniforge3}
ENV_NAME=tms_risk_prf
BRAINCODER_BRANCH=keras-backend

echo "== 1. braincoder submodule =="
# The submodule is often uninitialised on these VMs. Clone over HTTPS (the
# boxes have no GitHub SSH key) and pin the Keras-3 branch.
if [ ! -f "$REPO/libs/braincoder/pyproject.toml" ]; then
    rmdir "$REPO/libs/braincoder" 2>/dev/null || true
    git clone -q https://github.com/Gilles86/braincoder.git "$REPO/libs/braincoder"
fi
git -C "$REPO/libs/braincoder" fetch -q origin
git -C "$REPO/libs/braincoder" checkout -q "$BRAINCODER_BRANCH"
git -C "$REPO/libs/braincoder" pull -q --ff-only
echo "   braincoder @ $(git -C "$REPO/libs/braincoder" log --oneline -1)"

echo "== 2. conda env =="
"$CONDA_ROOT/bin/conda" env remove -n "$ENV_NAME" -y 2>/dev/null || true
cd "$REPO/create_env"
"$CONDA_ROOT/bin/conda" env create -f environment_sciencecloud_prf.yml -y

echo "== 3. keras backend = jax =="
# Two belts. `activate.d` covers `conda activate tms_risk_prf`; keras.json
# covers the nohup pattern, which calls the env's python binary directly and
# so never runs the activation hooks. Without either, Keras 3 defaults to
# TensorFlow — which is not installed — and every import dies.
mkdir -p "$CONDA_ROOT/envs/$ENV_NAME/etc/conda/activate.d"
echo "export KERAS_BACKEND=jax" > "$CONDA_ROOT/envs/$ENV_NAME/etc/conda/activate.d/keras_backend.sh"
mkdir -p "$HOME/.keras"
cat > "$HOME/.keras/keras.json" <<'JSON'
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "jax",
    "image_data_format": "channels_last"
}
JSON

echo "== 4. smoke test =="
"$CONDA_ROOT/envs/$ENV_NAME/bin/python" - <<'PY'
import keras, jax, numpy as np, pandas as pd
from braincoder.models import GaussianPRF, LogGaussianPRF
from braincoder.optimize import ParameterFitter, ResidualFitter
from braincoder.utils import get_rsq

assert keras.backend.backend() == 'jax', keras.backend.backend()
devices = jax.devices()
assert any(d.platform == 'gpu' for d in devices), f'no GPU: {devices}'

par = pd.DataFrame({'mu': [12., 25.], 'sd': [4., 8.],
                    'amplitude': [1.5, .7], 'baseline': [.2, -.3]})
x = pd.Series(np.linspace(5., 35., 40), name='x')
pred = np.asarray(GaussianPRF(parameters=par).predict(paradigm=x))
ref = (par['amplitude'].values * np.exp(-.5 * (x.values[:, None] - par['mu'].values) ** 2
                                        / par['sd'].values ** 2) + par['baseline'].values)
err = np.abs(pred - ref).max()
assert err < 1e-5, err
print(f"OK  keras {keras.__version__} / jax {jax.__version__} / {devices} "
      f"| GaussianPRF max err {err:.2e}")
PY

echo "== done: $ENV_NAME =="
