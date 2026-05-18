#!/bin/bash
#SBATCH --job-name=regression_prf_ses1
#SBATCH --output=/home/gdehol/logs/regression_prf_ses1_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00-00:45:00

# Initialize conda
. "$HOME/init_conda.sh"

# Activate environment
conda activate tf2-gpu

# Format subject ID as two digits
PARTICIPANT_LABEL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")

# Set model_label (passed as the first argument to this script)
MODEL_LABEL="$1"

# Run the script
python "$HOME/git/tms_risk/tms_risk/encoding_model/fit_regression_nprf.py" \
    "$PARTICIPANT_LABEL" \
    "$MODEL_LABEL" \
    --smoothed \
    --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk
