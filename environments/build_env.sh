#!/bin/bash

# Script to build conda environments on the cluster
# Usage: 
#   For CPU: sbatch build_env.sh cpu
#   For GPU: sbatch --gres=gpu:1 build_env.sh gpu
#   For CUDA: sbatch --gres=gpu:1 build_env.sh cuda
#
# Note: GPU and CUDA environments MUST be built on GPU nodes with --gres=gpu:1

set -e

ENV_NAME=${1:-cpu}

# Check if GPU is required and available
case $ENV_NAME in
    gpu|cuda)
        if ! nvidia-smi &> /dev/null; then
            echo "ERROR: Building ${ENV_NAME} environment requires a GPU node!"
            echo "Please resubmit with: sbatch --gres=gpu:1 build_env.sh ${ENV_NAME}"
            exit 1
        fi
        echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1)"
        ;;
    cpu)
        echo "Building CPU-only environment"
        ;;
esac

echo "Building environment: ${ENV_NAME}"
echo "Started at: $(date)"
echo "Node: $(hostname)"

# Initialize conda
source "$HOME/init_conda.sh"

# Map environment name to yml file
case $ENV_NAME in
    cpu)
        ENV_FILE="environment.yml"
        CONDA_ENV_NAME="tms_risk"
        ;;
    gpu)
        ENV_FILE="environment_gpu.yml"
        CONDA_ENV_NAME="tms_risk_gpu"
        ;;
    cuda)
        ENV_FILE="environment_cuda.yml"
        CONDA_ENV_NAME="tms_risk_cuda"
        ;;
    *)
        echo "Unknown environment: ${ENV_NAME}"
        echo "Valid options: cpu, gpu, cuda"
        exit 1
        ;;
esac

cd "$HOME/git/tms_risk/environments"

# Remove existing environment if it exists
if conda env list | grep -q "^${CONDA_ENV_NAME} "; then
    echo "Removing existing environment: ${CONDA_ENV_NAME}"
    conda env remove -n "${CONDA_ENV_NAME}" -y
fi

# Create new environment
echo "Creating environment from ${ENV_FILE}"
conda env create -f "${ENV_FILE}"

echo "Environment ${CONDA_ENV_NAME} built successfully"
echo "Finished at: $(date)"

# Show environment info
conda activate "${CONDA_ENV_NAME}"
echo ""
echo "Python version:"
python --version
echo ""
echo "TensorFlow version:"
python -c "import tensorflow as tf; print(tf.__version__)"
echo ""
echo "Available devices:"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
