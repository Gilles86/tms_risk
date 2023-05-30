#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/gdehol/logs/res_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=08:00:00
module load singularityce
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
singularity run -B /scratch/gdehol/templateflow:/opt/templateflow,/scratch/gdehol/ds-tmsrisk --cleanenv /scratch/gdehol/containers/fmriprep-20.2.3.simg /scratch/gdehol/ds-tmsrisk /scratch/gdehol/ds-tmsrisk/derivatives participant --participant-label $PARTICIPANT_LABEL  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative  --dummy-scans 3 --skip_bids_validation -w /scratch/gdehol/workflow_folders --no-submm-recon
