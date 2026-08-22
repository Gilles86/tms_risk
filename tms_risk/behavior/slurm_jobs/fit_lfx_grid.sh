#!/bin/bash
#SBATCH --job-name=lfxgrid
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --qos=normal
#SBATCH --time=07:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --array=1-24%12
#SBATCH --output=/home/gdehol/logs/lfxgrid_%A_%a.txt

# Systematic spline/hyperprior grid for the log-space flexible PMC:
# lfx2-{bs3|bs2|cr3}-{fm|sm}-{dp|tp}-{null|b}, 24 cells (see
# fit_model._build_lfx_grid). bauer comes from ~/git/bauer-powerlaw via
# PYTHONPATH (pinned checkout, commit stamped into each trace).
set -eo pipefail

LABELS=()
for basis in bs3 bs2 cr3; do
  for mem in fm sm; do
    for hp in dp tp; do
      for tms in null b; do
        LABELS+=("lfx2-${basis}-${mem}-${hp}-${tms}")
      done
    done
  done
done
LABEL=${LABELS[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "task ${SLURM_ARRAY_TASK_ID}: ${LABEL}"

cd "$HOME/git/tms_risk"
export PYTHONPATH="$HOME/git/bauer-powerlaw"
export PYTHONUNBUFFERED=1
# pytensor C-compilation cache: per-task dir on /scratch. The default
# ($HOME/.pytensor) tripped the home quota when 12 tasks compiled at once
# (observed 2026-08-22, job 5224953_7) and shares one lock across tasks.
export PYTENSOR_FLAGS="base_compiledir=/scratch/gdehol/pytensor/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"

exec "$HOME/data/conda/envs/tms_risk_cpu/bin/python" -u -m tms_risk.behavior.fit_model \
    "$LABEL" \
    --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk \
    --out_folder cogmodels.lfxgrid
