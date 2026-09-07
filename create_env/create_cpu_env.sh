#!/bin/bash
#SBATCH --job-name=create_tms_risk_cpu_env
#SBATCH --account=zne.uzh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/gdehol/logs/create_cpu_env_%j.log

set -e

mkdir -p "$HOME/logs"

echo "=== Creating tms_risk_cpu environment ==="
echo "Starting at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"

source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_NAME=tms_risk_cpu
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Removing existing environment: ${ENV_NAME}"
    conda env remove -n "${ENV_NAME}" -y
fi

cd "$HOME/git/tms_risk"
conda env create -f create_env/environment_cpu.yml

echo ""
echo "=== Environment created successfully ==="
echo "Finished at $(date)"
echo "Activate with: conda activate ${ENV_NAME}"
