#!/bin/bash
#SBATCH --job-name=fmriprep
#SBATCH --output=/home/gdehol/logs/res_fmriprep_%A-%a.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
source /etc/profile.d/lmod.sh
module load singularityce/3.10.2
export SINGULARITYENV_FS_LICENSE=$HOME/freesurfer/license.txt
export PARTICIPANT_LABEL=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
export SINGULARITYENV_TEMPLATEFLOW_HOME=/opt/templateflow
singularity run -B /shares/zne.uzh/containers/templateflow:/opt/templateflow -B /shares/zne.uzh/gdehol/ds-tmsrisk:/data -B /scratch/gdehol:/workflow --cleanenv /shares/zne.uzh/containers/fmriprep-20.2.3 /data /data/derivatives/fmriprep participant --participant-label $PARTICIPANT_LABEL  --output-spaces MNI152NLin2009cAsym T1w fsaverage fsnative  --dummy-scans 3 --skip_bids_validation -w /workflow --no-submm-recon