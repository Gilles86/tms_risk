#!/bin/bash
#
# Runtime script for a single Fisher-information job.
# Submitted as an array by submit_fisher_information.sh.
#
# Expects env vars:
#   SUBJECT_ID   subject id (from SLURM_ARRAY_TASK_ID)
#   SESSION      1, 2, or 3
#   BIDS_FOLDER  path to BIDS dataset
#   N_VOXELS     top-k voxels by R² (default 100)
#   ROI          mask name (default NPC12r — matches the paper)

set -e

SUBJECT_ID=$(printf "%02d" "${SUBJECT_ID:-${SLURM_ARRAY_TASK_ID}}")
SESSION=${SESSION:-2}
BIDS_FOLDER=${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-tmsrisk}
N_VOXELS=${N_VOXELS:-100}
ROI=${ROI:-NPC12r}

echo "Fisher information: sub-${SUBJECT_ID} ses-${SESSION} roi=${ROI} n_voxels=${N_VOXELS}"

source "$HOME/init_conda.sh"
conda activate tms_risk_cuda

python "$HOME/git/tms_risk/tms_risk/modeling/fisher_information.py" \
  "$SUBJECT_ID" "$SESSION" \
  --bids_folder "$BIDS_FOLDER" \
  --mask "$ROI" \
  --n_voxels "$N_VOXELS" \
  --denoise --natural_space
