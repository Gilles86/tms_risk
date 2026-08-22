#!/bin/bash
#SBATCH --job-name=rdmlogflex
#SBATCH --account=zne.uzh
#SBATCH --partition=lowprio
#SBATCH --gres=gpu:L4:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --array=1-6
#SBATCH --output=/home/gdehol/logs/rdmlogflex_%A_%a.txt

# Accumulator (DDM/RDM) x log-space flexible PMC — the magnitude->RT program
# (notes/rdm_magnitude_rt_plan.md). numpyro on one L4, vectorized chains,
# tune=2000/draws=1000/target_accept=0.99 + mapjitter via fit_model's
# accumulator branch. bauer from ~/git/bauer-powerlaw via PYTHONPATH.
set -eo pipefail

LABELS=(rdm_logflex2_null rdm_logflex2b rdm_logflex2b_ws0 \
        ddm_logflex2_null ddm_logflex2b ddm_logflex2_threshold)
LABEL=${LABELS[$((SLURM_ARRAY_TASK_ID - 1))]}
echo "task ${SLURM_ARRAY_TASK_ID}: ${LABEL}"

cd "$HOME/git/tms_risk"
export PYTHONPATH="$HOME/git/bauer-powerlaw"
export PYTHONUNBUFFERED=1
# pytensor C-cache on /scratch, per-task (home-quota + lock contention fix,
# see fit_lfx_grid.sh / job 5224953_7 post-mortem)
export PYTENSOR_FLAGS="base_compiledir=/scratch/gdehol/pytensor/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
# ... and ALL temp files off node-local /tmp (python tempfile inside
# pytensor's jax linker killed the first wave: OSError 122 in
# NamedTemporaryFile, jobs 5225709_1-6). General golden rule now in the
# sciencecluster skill.
export TMPDIR="/scratch/gdehol/tmp/${SLURM_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$TMPDIR"

# CUDA env: activate (activate.d hooks), then exec so SIGTERM reaches python
source "$HOME/data/miniforge3/etc/profile.d/conda.sh"
conda activate bauer_cuda
# NB: never `nvidia-smi | head` under `set -o pipefail` — head's early
# close SIGPIPEs nvidia-smi (exit 141) and -e kills the job pre-exec.
nvidia-smi -L || true

exec python -u -m tms_risk.behavior.fit_model \
    "$LABEL" \
    --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk \
    --out_folder cogmodels.rdmlogflex
