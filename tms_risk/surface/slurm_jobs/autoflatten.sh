#!/bin/bash
#SBATCH --job-name=autoflatten
#SBATCH --account=zne.uzh
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/home/gdehol/logs/autoflatten_%A_%a.txt
#
# Cut and flatten one participant's cortex with autoflatten, so the pycortex
# viewers (tms_risk/visualize/make_static_viewers.py) can show flatmaps.
#
# The flat patch must come from the SAME reconstruction the pycortex subject
# tms.sub-XX was imported from: the local <bids>/derivatives/freesurfer. The
# cluster's fmriprep/sourcedata/freesurfer is not that recon (sub-45 differs by
# ~370 vertices) and has no surf/ files, so the needed surf files are uploaded
# into $SUBJECTS_DIR first:
#
#   cd /data/ds-tmsrisk/derivatives/freesurfer && tar czf - sub-*/surf/{lh,rh}.{white,pial,fiducial,smoothwm,inflated,sphere.reg,curv,sulc} \
#     | ssh sciencecluster 'tar xzf - -C /shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/autoflatten/subjects'
#   ln -s $FREESURFER_HOME/subjects/fsaverage $SUBJECTS_DIR/fsaverage   # template cuts
#
#   sbatch --array=1-35 autoflatten.sh
#
# Then pull back sub-XX/surf/{lh,rh}.autoflatten.flat.patch.3d and import with
# cortex.freesurfer.import_flat(..., patch='autoflatten') (pycortex appends
# '.flat' itself).

SUBJECTS=(01 02 03 04 05 06 07 09 10 11 18 19 21 25 26 29 30 31 34 35 36 37 45 46 47 50 53 56 59 62 63 67 69 72 74)
SUB=sub-${SUBJECTS[$((SLURM_ARRAY_TASK_ID - 1))]}

export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer
export PATH=$FREESURFER_HOME/bin:$PATH
export FS_LICENSE=$HOME/freesurfer/license.txt      # mri_label2label silently maps 0 vertices without it
export SUBJECTS_DIR=/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/autoflatten/subjects

source ~/data/miniforge3/etc/profile.d/conda.sh
conda activate autoflatten
# jaxlib 0.11 aborts on an XLA flag autoflatten sets; see the sitecustomize.py there
export PYTHONPATH=/shares/zne.uzh/gdehol/ds-tmsrisk/derivatives/autoflatten/xla_fix${PYTHONPATH:+:$PYTHONPATH}

echo "$SUB on $(hostname)"
time autoflatten run "$SUBJECTS_DIR/$SUB" --parallel --overwrite \
    --output-dir "$SUBJECTS_DIR/$SUB/surf" --n-cores "$SLURM_CPUS_PER_TASK"
