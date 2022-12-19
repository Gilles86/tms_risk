#!/bin/bash
#SBATCH --job-name=task_fit_cv
#SBATCH --output=/home/cluster/gdehol/logs/fit_nprf_smoothed_surf_%A-%a.txt
#SBATCH --partition=volta
#SBATCH --ntasks=1
#SBATCH --mem=96G
#SBATCH --gres gpu:1
#SBATCH --time=1:00:00
module load volta
module load nvidia/cuda11.2-cudnn8.1.0

# . $HOME/init_conda.sh
# . $HOME/init_freesurfer.sh
. $HOME/.bashrc.sh

export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)

source activate tf2-gpu
python $HOME/git/tms_risk/tms_risk/encoding_model/fit_task_cv.py $PARTICIPANT_LABEL 3 --bids_folder /scratch/gdehol/ds-tmsrisk --denoise 
python $HOME/git/tms_risk/tms_risk/encoding_model/fit_task_cv.py $PARTICIPANT_LABEL 3 --bids_folder /scratch/gdehol/ds-tmsrisk --denoise --smoothed
