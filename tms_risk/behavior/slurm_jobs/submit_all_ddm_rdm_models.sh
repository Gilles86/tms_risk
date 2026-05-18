#!/bin/bash
#
# Submit the DDM × Flexible-PMC and RDM × Flexible-PMC model sweep for the
# Table-1-style ELPD comparison adding accumulator-model variants.
#
# Six variants per accumulator family (DDM and RDM):
#   _null            no TMS regressor (baseline)
#   _perception      TMS on perceptual noise only
#   _memory          TMS on memory noise only
#   (bare)           TMS on both noise terms (matches Flexible PMC paper claim)
#   _threshold       TMS on accumulator threshold only (alternative hypothesis)
#   _noise_threshold TMS on both noise terms + threshold
#
# Run from this directory: `bash submit_all_ddm_rdm_models.sh`
# Each job is a fit_model.sh sbatch.

set -e

model_labels=(
    # DDM × Flexible PMC family
    "ddm_flexible"
    "ddm_flexible_null"
    "ddm_flexible_perception"
    "ddm_flexible_memory"
    "ddm_flexible_threshold"
    "ddm_flexible_noise_threshold"

    # Race-diffusion × Flexible PMC family
    "rdm_flexible"
    "rdm_flexible_null"
    "rdm_flexible_perception"
    "rdm_flexible_memory"
    "rdm_flexible_threshold"
    "rdm_flexible_noise_threshold"
)

for MODEL_LABEL in "${model_labels[@]}"; do
    sbatch fit_model.sh "$MODEL_LABEL"
    echo "Submitted: $MODEL_LABEL"
done
