#!/bin/bash
#SBATCH --job-name=fit_model
#SBATCH --account=zne.uzh
#SBATCH --time=24:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=/home/gdehol/logs/fit_model_%j.out

mkdir -p "$HOME/logs"

. "$HOME/init_conda.sh"

MODEL_LABEL="$1"
BIDS_FOLDER="/shares/zne.uzh/gdehol/ds-tmsrisk"

# DDM / RDM fits need hssm, which conflicts with the TF 2.18 / numpy 1.26
# stack in tms_risk_cpu. Route those labels to a dedicated tms_risk_ddm env.
case "$MODEL_LABEL" in
    ddm_*|rdm_*)
        conda activate tms_risk_ddm ;;
    *)
        conda activate tms_risk_cpu ;;
esac

python -m tms_risk.behavior.fit_model "$MODEL_LABEL" --bids_folder "$BIDS_FOLDER" \
    > "$HOME/logs/fit_model_${MODEL_LABEL}.out" 2>&1
