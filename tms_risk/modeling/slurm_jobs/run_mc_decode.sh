#!/bin/bash
#
# Runtime script for a single Monte Carlo simulate-and-decode job.
# Submitted as an array — one task per subject — by submit_mc_decode.sh.
#
# Expects environment variables:
#   SUBJECT_ID   numeric subject id (from SLURM_ARRAY_TASK_ID)
#   SESSION      1, 2, or 3
#   BIDS_FOLDER  path to BIDS dataset
#   N_VOXELS     top-k voxels by R² (default 100)
#   N_REPEATS    Monte Carlo sample count per true stimulus (default 1000)
#   ROI          mask name passed to Subject (default wang15_ips)
#
# Resources are set by the submitting sbatch command (typically 1 GPU,
# ~24G RAM, ~30 min — model.simulate + get_stimulus_pdf are batched).

set -e

SUBJECT_ID=$(printf "%02d" "${SUBJECT_ID:-${SLURM_ARRAY_TASK_ID}}")
SESSION=${SESSION:-2}
BIDS_FOLDER=${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-tmsrisk}
N_VOXELS=${N_VOXELS:-100}
N_REPEATS=${N_REPEATS:-1000}
ROI=${ROI:-NPC12r}

PYTHON_BIN=${PYTHON_BIN:-$HOME/data/conda/envs/tms_risk_cuda/bin/python}
# Set SPHERICAL=1 to use a diagonal noise covariance (per-voxel τ, no ρ).
SPHERICAL_FLAG=""
if [ -n "${SPHERICAL:-}" ]; then
    SPHERICAL_FLAG="--spherical"
fi

echo "Monte-Carlo decode: sub-${SUBJECT_ID} ses-${SESSION} roi=${ROI} n_voxels=${N_VOXELS} n_repeats=${N_REPEATS} spherical=${SPHERICAL:-0}"
echo "Python: ${PYTHON_BIN}"

# Direct env binary (see run_fisher_information.sh for rationale).
"${PYTHON_BIN}" "$HOME/git/tms_risk/tms_risk/modeling/monte_carlo_decode.py" \
  "$SUBJECT_ID" "$SESSION" \
  --bids_folder "$BIDS_FOLDER" \
  --mask "$ROI" \
  --n_voxels "$N_VOXELS" \
  --n_repeats "$N_REPEATS" \
  --denoise --natural_space ${SPHERICAL_FLAG}
