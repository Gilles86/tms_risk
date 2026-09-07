#!/bin/bash
#SBATCH --job-name=anchrep
#SBATCH --account=zne.uzh
#SBATCH --partition=standard
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --array=1-72
#SBATCH --output=/home/gdehol/logs/anchrep_%A_%a.txt
set -eo pipefail
# One task per model label: the posterior-predictive summary and the ELPD that
# the results PDF is built from. No throttle -- the partition has the cores and
# each task is a few minutes.
LABEL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" /scratch/gdehol/anchor_labels.txt)
TRACE=/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/cogmodels.anchor
if [ ! -f "${TRACE}/model-${LABEL}_trace.netcdf" ]; then
    echo "no trace for ${LABEL}, skipping"; exit 0
fi
cd "$HOME/git/tms_risk"
export PYTHONPATH=/scratch/gdehol/bauer_anchor
export PYTHONUNBUFFERED=1
export PYTENSOR_FLAGS="base_compiledir=/scratch/gdehol/pytensor/rep_${SLURM_ARRAY_TASK_ID}"
PY="$HOME/data/conda/envs/tms_risk_cpu/bin/python"
$PY -u -m tms_risk.behavior.scripts.extract_anchor_loo "$LABEL" \
    --trace_dir "$TRACE" --out_dir /home/gdehol/loo_anchor
$PY -u -m tms_risk.behavior.scripts.extract_anchor_ppc "$LABEL" \
    --trace_dir "$TRACE" --out_dir /home/gdehol/ppc_anchor
