#!/bin/bash
#SBATCH --job-name=fit_model
#SBATCH --account=zne.uzh
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/gdehol/logs/fit_model_%j.out

# Resource layout per bauer/notes/tms_risk_ddm_fitting_brief.md:
#   DDM/RDM → GPU (L4), 4 h walltime, bauer_cuda env (has jax[cuda12] + hssm).
#   PMC      → CPU, 12 h walltime, tms_risk_cpu env (pymc/bambi/braincoder).
# `recommended_init='mapjitter'` in bauer makes DDM/RDM converge reliably,
# so 4 h is plenty under numpyro vectorized on a GPU.

mkdir -p "$HOME/logs"
. "$HOME/init_conda.sh"

MODEL_LABEL="$1"
BIDS_FOLDER="/shares/zne.uzh/gdehol/ds-tmsrisk"

case "$MODEL_LABEL" in
    ddm_*|rdm_*)
        conda activate bauer_cuda ;;
    *)
        conda activate tms_risk_cpu ;;
esac

python -m tms_risk.behavior.fit_model "$MODEL_LABEL" --bids_folder "$BIDS_FOLDER" \
    > "$HOME/logs/fit_model_${MODEL_LABEL}.out" 2>&1
