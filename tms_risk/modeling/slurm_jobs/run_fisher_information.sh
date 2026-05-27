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
PYTHON_BIN=${PYTHON_BIN:-$HOME/data/conda/envs/tms_risk_cuda/bin/python}
# Set SPHERICAL=1 to use a diagonal noise covariance (per-voxel τ, no
# cross-voxel ρ). The full Ω tends to over-estimate covariance and
# collapse the decoder toward the stimulus-range mean. Diagonal Ω
# restores individual-RF tuning.
SPHERICAL_FLAG=""
if [ -n "${SPHERICAL:-}" ]; then
    SPHERICAL_FLAG="--spherical"
fi

echo "Fisher information: sub-${SUBJECT_ID} ses-${SESSION} roi=${ROI} n_voxels=${N_VOXELS} spherical=${SPHERICAL:-0}"
echo "Python: ${PYTHON_BIN}"

# Direct env binary — avoids the `conda activate` path, which interacts
# badly with `set -e` on this cluster (init_conda.sh uses a zsh hook
# that bails under bash strict mode).
"${PYTHON_BIN}" "$HOME/git/tms_risk/tms_risk/modeling/fisher_information.py" \
  "$SUBJECT_ID" "$SESSION" \
  --bids_folder "$BIDS_FOLDER" \
  --mask "$ROI" \
  --n_voxels "$N_VOXELS" \
  --denoise --natural_space ${SPHERICAL_FLAG}
