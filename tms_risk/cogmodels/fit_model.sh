#!/bin/bash
#SBATCH --job-name=fit_model        # Job name
#SBATCH --time=06:00:00             # Maximum runtime set to 6 hours
#SBATCH --mem=32G                   # Memory allocation (adjust as needed)
#SBATCH --cpus-per-task=4           # Number of CPUs
#SBATCH --output=/home/gdehol/logs/fit_model_%j.out  # Output log file with job ID

# Ensure logs directory exists
mkdir -p $HOME/logs

# Load Conda environment
. $HOME/init_conda.sh               # Load Conda (adjust to your init script)
conda activate tms_risk             # Activate the CPU-only environment

# Define variables
MODEL_LABEL="$1"                     # Model label argument (pass as first argument when running this script)
BIDS_FOLDER="/shares/zne.uzh/gdehol/ds-tmsrisk"  # BIDS folder path

# Run the Python script and redirect output to a log file with MODEL_LABEL in the name
python /home/gdehol/git/tms_risk/tms_risk/cogmodels/fit_model.py "$MODEL_LABEL" --bids_folder "$BIDS_FOLDER" > "$HOME/logs/fit_model_${MODEL_LABEL}.out" 2>&1