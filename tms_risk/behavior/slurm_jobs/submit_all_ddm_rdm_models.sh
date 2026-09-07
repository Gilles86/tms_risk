#!/bin/bash
#
# Submit the full SSM × PMC analogue sweep for the Table-1-style ELPD
# comparison adding accumulator-model variants.
#
# Two families × two noise structures × 4-6 regressor variants:
#
#   Weber-noise:    analogues of the paper's 11_* PMC family
#                   (no splines, single noise param per term)
#   Flexible-noise: analogues of the paper's flexible2_* family
#                   (5-spline noise function over magnitude)
#
# Regressor suffixes (same in both noise structures):
#   _null            no TMS regressor (baseline)
#   _perception      TMS on perceptual noise only
#   _memory          TMS on memory noise only
#   (bare)           TMS on both noise terms (paper claim analogue)
#   _threshold       Flexible only — TMS on accumulator threshold
#   _noise_threshold Flexible only — TMS on both noise + threshold
#
# Run from this directory: `bash submit_all_ddm_rdm_models.sh`

set -e

model_labels=(
    # ── Weber-noise, shared_perceptual_noise (paper analogue of 11_*) ──
    "ddm_weber_null"
    "ddm_weber_perception"
    "ddm_weber_memory"
    "ddm_weber"

    "rdm_weber_null"
    "rdm_weber_perception"
    "rdm_weber_memory"
    "rdm_weber"

    # ── Weber-noise, independent memory model (n1/n2_evidence_sd) ──
    # Same evidence-noise structure, different decomposition. Bauer's
    # default; used in notes/tms_risk_ddm_fitting_brief.md's recipe.
    "ddm_indep_null"
    "ddm_indep_n1"
    "ddm_indep_n2"
    "ddm_indep"

    "rdm_indep_null"
    "rdm_indep_n1"
    "rdm_indep_n2"
    "rdm_indep"

    # ── Flexible-noise SSM family (paper Flexible PMC analogue: flexible2_*) ──
    "ddm_flexible_null"
    "ddm_flexible_perception"
    "ddm_flexible_memory"
    "ddm_flexible"
    "ddm_flexible_threshold"
    "ddm_flexible_noise_threshold"

    "rdm_flexible_null"
    "rdm_flexible_perception"
    "rdm_flexible_memory"
    "rdm_flexible"
    "rdm_flexible_threshold"
    "rdm_flexible_noise_threshold"
)

for MODEL_LABEL in "${model_labels[@]}"; do
    # GPU L4, lowprio (more L4 nodes there) — see
    # bauer/notes/tms_risk_ddm_fitting_brief.md.
    sbatch --gres=gpu:L4:1 --partition=lowprio fit_model.sh "$MODEL_LABEL"
    echo "Submitted: $MODEL_LABEL (GPU L4)"
done
