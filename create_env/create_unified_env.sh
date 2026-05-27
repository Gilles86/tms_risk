#!/bin/bash
#SBATCH --job-name=create_tms_risk_unified_env
#SBATCH --account=zne.uzh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gdehol/logs/create_unified_env_%j.log

# Build the unified env on a GPU compute node so TF 2.20's CUDA wheel
# variant is selected by pip (it inspects the visible NVIDIA driver at
# install time, same constraint as create_gpu_env.sh).

set -e

mkdir -p "$HOME/logs"

module load cuda/12.6.3 || true

echo "=== Creating tms_risk_unified environment ==="
echo "Starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU info:"
nvidia-smi

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME=tms_risk_unified
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
fi

cd "$HOME/git/tms_risk"

# Make sure the braincoder submodule is up-to-date — the unified env
# expects the keras-backend branch (where get_expected_uncertainty lives).
git submodule update --init --recursive

conda env create -f create_env/environment_unified.yml

conda activate "${ENV_NAME}"

echo ""
echo "=== Sanity check: full stack ==="
python -c "
import tensorflow as tf, pymc, hssm, ssms, bauer, braincoder, tms_risk
print('python   ', __import__('sys').version.split()[0])
print('tf       ', tf.__version__)
print('pymc     ', pymc.__version__)
print('hssm     ', hssm.__version__)
print('ssms     ', ssms.__file__)
print('bauer    ', bauer.__file__)
print('braincoder', braincoder.__file__)
print('GPUs     ', tf.config.list_physical_devices('GPU'))
from braincoder.models import LogGaussianPRF
assert hasattr(LogGaussianPRF, 'get_expected_uncertainty'), \
    'braincoder is missing get_expected_uncertainty — submodule on wrong branch?'
print('braincoder.get_expected_uncertainty: present ✓')
"

echo ""
echo "=== Environment created successfully ==="
echo "Finished at $(date)"
echo "Activate with: conda activate ${ENV_NAME}"
