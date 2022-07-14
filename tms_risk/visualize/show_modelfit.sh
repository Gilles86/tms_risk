export SUBJECT=$1
BIDS_DIR=/data/ds-tmsrisk
DERIVATIVES_DIR=$BIDS_DIR/derivatives

fsleyes $DERIVATIVES_DIR/fmriprep/sub-${SUBJECT}/ses-1/anat/sub-${SUBJECT}_ses-1_desc-preproc_T1w.nii.gz $DERIVATIVES_DIR/encoding_model.smoothed/sub-${SUBJECT}/ses-1/func/sub-${SUBJECT}_ses-1_desc-r2.optim_space-T1w_pars.nii.gz -cm hot -dr 0.125 0.45 
