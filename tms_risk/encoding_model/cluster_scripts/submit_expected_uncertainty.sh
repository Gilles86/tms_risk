#!/bin/bash

# Submit calculate_expected_uncertainty.py jobs to SLURM cluster
#
# Usage:
#   ./submit_expected_uncertainty.sh                  # Default: all subjects, non-spherical, no prior
#   ./submit_expected_uncertainty.sh --spherical      # Spherical likelihood
#   ./submit_expected_uncertainty.sh --use_prior      # With empirical prior
#   ./submit_expected_uncertainty.sh --spherical --use_prior  # Both options
#   ./submit_expected_uncertainty.sh --all            # All 4 combinations
#   ./submit_expected_uncertainty.sh --subjects 4,5,6 # Specific subjects

# Default subject set (comma-separated)
SUBJECTS="1,2,3,4,5,6,7,9,10,11,18,19,21,25,26,29,30,31,34,35,36,37,45,46,47,50,53,56,59,62,63,67,69,72,74"
SPHERICAL=0
USE_PRIOR=0
ALL_COMBINATIONS=false
BIDS_FOLDER="/shares/zne.uzh/gdehol/ds-tmsrisk"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spherical)
            SPHERICAL=1
            shift
            ;;
        --use_prior)
            USE_PRIOR=1
            shift
            ;;
        --all)
            ALL_COMBINATIONS=true
            shift
            ;;
        --subjects)
            SUBJECTS="$2"
            shift 2
            ;;
        --bids_folder)
            BIDS_FOLDER="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--spherical] [--use_prior] [--all] [--subjects 4,5,6]"
            exit 1
            ;;
    esac
done

# Function to submit a single job configuration
submit_job() {
    local spherical=$1   # 0/1
    local prior=$2       # 0/1
    local suffix=$3

    sbatch --array="${SUBJECTS}" \
        --job-name="exp_unc${suffix}" \
        --output="/home/gdehol/logs/expected_uncertainty${suffix}_%A-%a.txt" \
        --ntasks=1 \
        --mem=64G \
        --gres=gpu:1 \
        --time=00-02:00:00 \
        --export=ALL,SPHERICAL=${spherical},USE_PRIOR=${prior},BIDS_FOLDER=${BIDS_FOLDER} \
        "$HOME/git/tms_risk/tms_risk/encoding_model/cluster_scripts/run_expected_uncertainty.sh"

    echo "Submitted: exp_unc${suffix} (SPHERICAL=${spherical}, USE_PRIOR=${prior}) subjects=[${SUBJECTS}]"
}

# Submit jobs
if [ "$ALL_COMBINATIONS" = true ]; then
    echo "Submitting all 4 combinations..."
    submit_job 0 0 "_default"
    submit_job 1 0 "_sph"
    submit_job 0 1 "_prior"
    submit_job 1 1 "_sph_prior"
else
    JOB_SUFFIX=""
    [ "$SPHERICAL" = "1" ] && JOB_SUFFIX="${JOB_SUFFIX}_sph"
    [ "$USE_PRIOR" = "1" ] && JOB_SUFFIX="${JOB_SUFFIX}_prior"
    [ -z "$JOB_SUFFIX" ] && JOB_SUFFIX="_default"
    submit_job "$SPHERICAL" "$USE_PRIOR" "$JOB_SUFFIX"
fi

echo "Done submission. Use squeue -u $USER to monitor."
