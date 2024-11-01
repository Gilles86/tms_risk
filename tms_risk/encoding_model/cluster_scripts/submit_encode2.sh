#!/bin/bash
#SBATCH --job-name=fit_nprf_ses2
#SBATCH --output=/home/gdehol/logs/fit_nprf_ses2_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=10:00

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

conda activate tf2-gpu
python $HOME/git/tms_risk/tms_risk/encoding_model/fit_nprf.py $PARTICIPANT_LABEL 2 --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk --denoise
python $HOME/git/tms_risk/tms_risk/encoding_model/fit_nprf.py $PARTICIPANT_LABEL 2 --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk --denoise --smoothed
