#!/bin/bash

# Runtime script for a single expected uncertainty job.
# Expects environment variables:
#   SUBJECT_ID   (numeric subject id, provided by SLURM_ARRAY_TASK_ID)
#   SPHERICAL    (1 for spherical model, 0 otherwise)
#   USE_PRIOR    (1 to use empirical prior, 0 otherwise)
#   BIDS_FOLDER  (path to BIDS dataset)
# Resources (GPU, memory, time) are specified in the submitting sbatch command.

set -e

SUBJECT_ID=$(printf "%02d" "${SUBJECT_ID:-${SLURM_ARRAY_TASK_ID}}")
BIDS_FOLDER=${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-tmsrisk}
SPHERICAL=${SPHERICAL:-0}
USE_PRIOR=${USE_PRIOR:-0}

echo "Running expected uncertainty for subject ${SUBJECT_ID} (spherical=${SPHERICAL}, prior=${USE_PRIOR})"

# Initialize conda and activate environment
source "$HOME/init_conda.sh"
conda activate tf2-gpu

FLAGS=""
if [ "$SPHERICAL" = "1" ]; then
  FLAGS+=" --spherical"
fi
if [ "$USE_PRIOR" = "1" ]; then
  FLAGS+=" --use_prior"
fi

python "$HOME/git/tms_risk/tms_risk/encoding_model/calculate_expected_uncertainty.py" \
  "$SUBJECT_ID" \
  --bids_folder "$BIDS_FOLDER" \
  $FLAGS
