#!/bin/bash
#SBATCH --job-name=create_tms_risk_cuda_env
#SBATCH --account=zne.uzh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/gdehol/logs/create_cuda_env_%j.log

# Builds the GPU/CUDA env on a GPU compute node so the NVIDIA driver is
# visible to pip when it picks tensorflow's CUDA wheel variant.

set -e

mkdir -p "$HOME/logs"

module load cuda/12.6.3 || true

echo "=== Creating tms_risk_cuda environment ==="
echo "Starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU info:"
nvidia-smi

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME=tms_risk_cuda
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
fi

cd "$HOME/git/tms_risk"
conda env create -f create_env/environment_cuda.yml

conda activate "${ENV_NAME}"
echo ""
echo "Sanity check:"
python -c "import tensorflow as tf; print('TF', tf.__version__); print('GPUs:', tf.config.list_physical_devices('GPU'))"

echo ""
echo "=== Environment created successfully ==="
echo "Finished at $(date)"
echo "Activate with: conda activate ${ENV_NAME}"
