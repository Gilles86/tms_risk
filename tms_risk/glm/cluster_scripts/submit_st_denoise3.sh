#!/bin/bash
#SBATCH --job-name=fit_st_denoise3
#SBATCH --output=/home/gdehol/logs/fit_st_denoise3_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH -c 16
#SBATCH --time=45:00

. $HOME/init_conda.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

python $HOME/git/tms_risk/tms_risk/glm/fit_single_trials_denoise.py $PARTICIPANT_LABEL 3 --bids_folder /scratch/gdehol/ds-tmsrisk --smoothed
python $HOME/git/tms_risk/tms_risk/glm/fit_single_trials_denoise.py $PARTICIPANT_LABEL 3 --bids_folder /scratch/gdehol/ds-tmsrisk
