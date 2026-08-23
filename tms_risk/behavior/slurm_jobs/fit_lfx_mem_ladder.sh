#!/bin/bash
#SBATCH --job-name=lfxmem
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-6
#SBATCH --output=/home/gdehol/logs/lfxmem_%A_%a.txt

# Memory-spline df ladder for the log-space flexible PMC: between the scalar
# memory (sm, FAILS the order-asymmetry delta-PPC at 0.67) and the 5-df
# memory (logflex2, PASSES at 1.0 but heavily parameterized), sample df=2
# (linear in log n) and df=3 (quadratic) with the three TMS levers.
# bauer from ~/git/bauer-powerlaw via PYTHONPATH.
set -eo pipefail

LABELS=(lfx2-bs3-m2-dp-null lfx2-bs3-m2-dp-b lfx2-bs3-m2-dp-bm \
        lfx2-bs3-m3-dp-null lfx2-bs3-m3-dp-b lfx2-bs3-m3-dp-bm)
LABEL=${LABELS[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "task ${SLURM_ARRAY_TASK_ID}: ${LABEL}"

cd "$HOME/git/tms_risk"
export PYTHONPATH="$HOME/git/bauer-powerlaw"
export PYTHONUNBUFFERED=1
# pytensor C-cache + all temp files on /scratch (home-quota / node-local /tmp
# post-mortems: jobs 5224953_7, 5224974_20; general rule in the
# sciencecluster skill)
export PYTENSOR_FLAGS="base_compiledir=/scratch/gdehol/pytensor/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
export TMPDIR="/scratch/gdehol/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

exec "$HOME/data/conda/envs/tms_risk_cpu/bin/python" -u -m tms_risk.behavior.fit_model \
    "$LABEL" \
    --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk \
    --out_folder cogmodels.lfxgrid
