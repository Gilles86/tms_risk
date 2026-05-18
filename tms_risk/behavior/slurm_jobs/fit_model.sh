#!/bin/bash
#SBATCH --job-name=fit_model
#SBATCH --account=zne.uzh
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/gdehol/logs/fit_model_%j.out

mkdir -p "$HOME/logs"

. "$HOME/init_conda.sh"
conda activate tms_risk_cpu

MODEL_LABEL="$1"
BIDS_FOLDER="/shares/zne.uzh/gdehol/ds-tmsrisk"

python -m tms_risk.behavior.fit_model "$MODEL_LABEL" --bids_folder "$BIDS_FOLDER" \
    > "$HOME/logs/fit_model_${MODEL_LABEL}.out" 2>&1
