#!/bin/bash
#
# Submit Fisher-information jobs for every TMS subject × session 2/3.
# Companion to submit_mc_decode.sh — same subject list, same ROI default.

set -e

SUBJECTS="${SUBJECTS:-2,5,7,8,9,10,11,12,13,14,15,16,18,19,20,21,24,25,26,27,30,31,32,33,34,38,42,44,45,46,47,48,50,51,52}"
BIDS_FOLDER=${BIDS_FOLDER:-/shares/zne.uzh/gdehol/ds-tmsrisk}

for SESSION in 2 3; do
    sbatch --array="${SUBJECTS}" \
      --job-name="fisher_ses${SESSION}" \
      --account=zne.uzh \
      --output="/home/gdehol/logs/fisher_ses${SESSION}_%A-%a.txt" \
      --ntasks=1 \
      --cpus-per-task=4 \
      --gpus=1 \
      --mem=24G \
      --time=00:20:00 \
      --export=ALL,SESSION=${SESSION},BIDS_FOLDER=${BIDS_FOLDER} \
      "$HOME/git/tms_risk/tms_risk/modeling/slurm_jobs/run_fisher_information.sh"
    echo "Submitted: fisher_ses${SESSION}, subjects=[${SUBJECTS}]"
done
