export SUBJECT=$1
BISD_DIR=/data/ds-tmsrisk
TMS_DIR=/data/ds-tmsrisk/derivatives/tms_navigation/sub-$SUBJECT/

fsleyes $TMS_DIR/sub-${SUBJECT}_T1w.nii.gz $TMS_DIR/sub-${SUBJECT}_NPC12.nii.gz -cm blue-lightblue $TMS_DIR/sub-${SUBJECT}_com.nii.gz -cm green $TMS_DIR/sub-${SUBJECT}_r2.thr.nii.gz -cm hot -dr 0.125 0.45 
