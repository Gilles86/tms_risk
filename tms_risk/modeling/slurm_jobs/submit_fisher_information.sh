#!/bin/bash
#
# Submit Fisher-information jobs for every TMS subject × session 2/3.
# Companion to submit_mc_decode.sh — same subject list, same ROI default.

set -e

## TMS subjects with PRF fits available — intersection of tms_keys.yml
## (the actual cTBS cohort) with encoding_model2.model-1.smoothed/ on disk.
## Excludes sub-22 and sub-49 (TMS subjects without PRF fits) and the
## non-TMS PRF subjects 35/36/37 etc.
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,9,10,11,18,19,21,25,26,29,30,31,34,35,36,37,45,46,47,50,53,56,59,62,63,67,69,72,74}"
BIDS_FOLDER=${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-tmsrisk}
# Pass SPHERICAL=1 to use a diagonal noise covariance (per-voxel τ, no ρ).
# Outputs land under derivatives/fisher_information.…spherical/ so the
# two flavors don't collide.
SPHERICAL=${SPHERICAL:-}
MODEL_LABEL=${MODEL_LABEL:-}

for SESSION in 2 3; do
    JOB_NAME="fisher_ses${SESSION}"
    [ -n "$SPHERICAL" ] && JOB_NAME="${JOB_NAME}_sph"
    [ -n "$MODEL_LABEL" ] && JOB_NAME="${JOB_NAME}_m${MODEL_LABEL}"
    sbatch --array="${SUBJECTS}" \
      --job-name="${JOB_NAME}" \
      --account=zne.uzh \
      --output="/home/gdehol/logs/${JOB_NAME}_%A-%a.txt" \
      --ntasks=1 \
      --cpus-per-task=4 \
      --gpus=1 \
      --mem=24G \
      --time=00:20:00 \
      --export=ALL,SESSION=${SESSION},BIDS_FOLDER=${BIDS_FOLDER},SPHERICAL=${SPHERICAL},MODEL_LABEL=${MODEL_LABEL} \
      "$HOME/git/tms_risk/tms_risk/modeling/slurm_jobs/run_fisher_information.sh"
    echo "Submitted: ${JOB_NAME}, subjects=[${SUBJECTS}], spherical=${SPHERICAL:-0}"
done
