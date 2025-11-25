#!/bin/bash
#SBATCH --job-name=install_tms_risk_cuda
#SBATCH --output=/home/gdehol/logs/install_tms_risk_cuda_%j.txt
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00

# Load the GPU module first (if required by your cluster)
module load gpu

# Load the correct CUDA module
module load cuda/12.2.1

# Load conda
. "$HOME/init_conda.sh"

# Create the environment
echo "Creating tms_risk_cuda environment..."
conda env create -n tms_risk_cuda -f "$HOME/git/tms_risk/environment_cuda.yml"

# Verify installation
echo "Verifying TensorFlow and GPU..."
conda activate tms_risk_cuda
python -c "
import tensorflow as tf
print('TensorFlow version:', tf.__version__)
print('GPU Available:', tf.config.list_physical_devices('GPU'))
"