#!/bin/bash
#SBATCH --job-name=lfxmemb
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-12
#SBATCH --output=/home/gdehol/logs/lfxmemb_%A_%a.txt

# Basis robustness for the memory-df ladder: the bs3 m2/m3 cells (job
# 5233031) rerun with quadratic B-splines (bs2, less wiggly everywhere)
# and natural cubic splines (cr3, linear beyond the boundary knots -- tames
# the data-poor high-payoff tail). bauer from ~/git/bauer-powerlaw.
set -eo pipefail

LABELS=()
for basis in bs2 cr3; do
  for mem in m2 m3; do
    for tms in null b bm; do
      LABELS+=("lfx2-${basis}-${mem}-dp-${tms}")
    done
  done
done
LABEL=${LABELS[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "task ${SLURM_ARRAY_TASK_ID}: ${LABEL}"

cd "$HOME/git/tms_risk"
export PYTHONPATH="$HOME/git/bauer-powerlaw"
export PYTHONUNBUFFERED=1
export PYTENSOR_FLAGS="base_compiledir=/scratch/gdehol/pytensor/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
export TMPDIR="/scratch/gdehol/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

exec "$HOME/data/conda/envs/tms_risk_cpu/bin/python" -u -m tms_risk.behavior.fit_model \
    "$LABEL" \
    --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk \
    --out_folder cogmodels.lfxgrid
