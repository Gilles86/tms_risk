export SUBJECT=$1
BIDS_DIR=/data/ds-tmsrisk
DERIVATIVES_DIR=$BIDS_DIR/derivatives

#fsleyes $DERIVATIVES_DIR/fmriprep/sub-${SUBJECT}/ses-1/anat/sub-${SUBJECT}_ses-1_desc-preproc_T1w.nii.gz $DERIVATIVES_DIR/ips_masks/sub-${SUBJECT}/anat/sub-${SUBJECT}_space-T1w_desc-NPC12r_mask.nii.gz $DERIVATIVES_DIR/encoding_model.smoothed.pca_confounds/sub-${SUBJECT}/ses-1/func/sub-${SUBJECT}_ses-1_desc-r2.optim_space-T1w_pars.nii.gz -cm hot -dr 0.15 0.45 
fsleyes $DERIVATIVES_DIR/fmriprep/sub-${SUBJECT}/ses-1/anat/sub-${SUBJECT}_ses-1_desc-preproc_T1w.nii.gz $DERIVATIVES_DIR/encoding_model.smoothed.pca_confounds/sub-${SUBJECT}/ses-1/func/sub-${SUBJECT}_ses-1_desc-r2.optim_space-T1w_pars.nii.gz -cm hot -dr 0.15 0.45 
