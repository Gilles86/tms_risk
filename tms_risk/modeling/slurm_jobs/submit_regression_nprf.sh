#!/bin/bash
# Regression-nPRF fit (encoding_model2) for one subject, one model label.
#
#   sbatch --array=1-74 submit_regression_nprf.sh 3        # main fit, model 3
#   sbatch --array=1-74 submit_regression_nprf.sh 3 --cv   # cross-validated
#
# Model labels (fit_regression_nprf.py::get_model):
#   0  pooled across sessions
#   1  amplitude per session                     <- the paper's canonical model
#   2  mu, sd, amplitude, baseline per session
#   3  amplitude AND sd per session, mu pooled   <- gain + dispersion, tuning fixed
#
# NOTE the CV script now defaults to max_n_iterations=10000, matching the main fit.
# It previously shipped with 10, so every stored cvR2 came from a 1000x
# under-converged optimisation. Expect CV jobs to take much longer than before.
#SBATCH --job-name=regression_nprf
#SBATCH --output=/home/gdehol/logs/regression_nprf_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=4:00:00
#SBATCH --account=zne.uzh

. $HOME/init_conda.sh
# tms_risk_prf (Keras 3 on JAX), NOT tms_risk_cuda: the latter allocates a GPU and then
# runs on CPU -- its TF 2.14 has no CUDA runtime wheels. Verified 2026-08-05: JAX sees
# the GPU on the same nodes where TF could not, so this was never a node problem.
conda activate tms_risk_prf
export KERAS_BACKEND=jax

MODEL_LABEL=${1:-1}
MODE=${2:-}
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
BIDS=/shares/zne.uzh/gdehol/ds-tmsrisk

cd $HOME/git/tms_risk

if [ "$MODE" == "--cv" ]; then
    python -m tms_risk.modeling.fit_regression_nprf_cv \
        $PARTICIPANT_LABEL $MODEL_LABEL --bids_folder $BIDS
else
    python -m tms_risk.modeling.fit_regression_nprf \
        $PARTICIPANT_LABEL $MODEL_LABEL --bids_folder $BIDS
fi
