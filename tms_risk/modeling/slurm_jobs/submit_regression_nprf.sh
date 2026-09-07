#!/bin/bash
# Regression-nPRF fit (encoding_model2) for one subject, one model label.
#
#   sbatch --array=<subs> submit_regression_nprf.sh 3                    # main fit
#   sbatch --array=<subs> submit_regression_nprf.sh 3 --cv                # cross-validated
#   sbatch --array=<subs> submit_regression_nprf.sh 3 '' .refit2026       # separate tree
#
# Model labels (fit_regression_nprf.py::get_model):
#   0  pooled across sessions
#   1  amplitude per session                     <- the paper's canonical model
#   2  mu, sd, amplitude, baseline per session
#   3  amplitude AND sd per session              <- mixes gain and shape; superseded
#   4  mu AND sd per session                     <- TUNING: what it is tuned to
#   5  amplitude AND baseline per session        <- RESPONSE MAGNITUDE: how hard it responds
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
# Third arg: output-tree suffix. Use '.refit2026' to keep a run OUT of the legacy tree
# encoding_model2.model-{0,1,2}.smoothed, which every published analysis reads.
OUT_SUFFIX=${3:-}
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
BIDS=/shares/zne.uzh/gdehol/ds-tmsrisk

cd $HOME/git/tms_risk

if [ "$MODE" == "--cv" ]; then
    python -m tms_risk.modeling.fit_regression_nprf_cv \
        $PARTICIPANT_LABEL $MODEL_LABEL --bids_folder $BIDS
else
    python -m tms_risk.modeling.fit_regression_nprf \
        $PARTICIPANT_LABEL $MODEL_LABEL --bids_folder $BIDS --out_suffix "$OUT_SUFFIX"
fi
