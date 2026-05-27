#!/bin/bash
#
# Submit Monte Carlo simulate-and-decode for every TMS subject × session 2/3.
# Logs land in ~/logs/mc_decode_<jobid>-<arrayidx>.txt.
#
# Subject list mirrors submit_expected_uncertainty.sh — keep them in sync.

set -e

## TMS subjects with PRF fits available — see comment in
## submit_fisher_information.sh for derivation.
SUBJECTS="${SUBJECTS:-1,2,3,4,5,6,7,9,10,11,18,19,21,25,26,29,30,31,34,35,36,37,45,46,47,50,53,56,59,62,63,67,69,72,74}"
BIDS_FOLDER=${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-tmsrisk}
# Pass SPHERICAL=1 to use a diagonal noise covariance.
SPHERICAL=${SPHERICAL:-}

for SESSION in 2 3; do
    JOB_NAME="mc_decode_ses${SESSION}"
    [ -n "$SPHERICAL" ] && JOB_NAME="${JOB_NAME}_sph"
    sbatch --array="${SUBJECTS}" \
      --job-name="${JOB_NAME}" \
      --account=zne.uzh \
      --output="/home/gdehol/logs/${JOB_NAME}_%A-%a.txt" \
      --ntasks=1 \
      --cpus-per-task=4 \
      --gpus=1 \
      --mem=24G \
      --time=00:30:00 \
      --export=ALL,SESSION=${SESSION},BIDS_FOLDER=${BIDS_FOLDER},SPHERICAL=${SPHERICAL} \
      "$HOME/git/tms_risk/tms_risk/modeling/slurm_jobs/run_mc_decode.sh"
    echo "Submitted: ${JOB_NAME}, subjects=[${SUBJECTS}], spherical=${SPHERICAL:-0}"
done
