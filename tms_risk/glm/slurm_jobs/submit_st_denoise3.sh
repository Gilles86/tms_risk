#!/bin/bash
#SBATCH --job-name=fit_st_denoise3
#SBATCH --account=zne.uzh
#SBATCH --output=/home/gdehol/logs/fit_st_denoise_ses3_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00

PARTICIPANT_LABEL=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
PY="$HOME/data/conda/envs/tms_risk_cpu/bin/python"
SCRIPT="$HOME/git/tms_risk/tms_risk/glm/fit_single_trials_denoise.py"
BIDS=/shares/zne.uzh/gdehol/ds-tmsrisk

"$PY" "$SCRIPT" "$PARTICIPANT_LABEL" 3 --bids_folder "$BIDS" --smoothed
"$PY" "$SCRIPT" "$PARTICIPANT_LABEL" 3 --bids_folder "$BIDS"
