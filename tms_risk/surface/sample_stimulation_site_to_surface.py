import argparse
import os.path as op
from tms_risk.utils.data import Subject
from nilearn import surface
import nibabel as nb
from tms_risk.encoding_model.fit_nprf import get_key_target_dir
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform
import numpy as np
from tms_risk.utils.data import get_tms_subjects


def transform_fsaverage(in_file, fs_hemi, source_subject, bids_folder):

        subjects_dir = op.join(bids_folder, 'derivatives', 'freesurfer')

        sxfm = SurfaceTransform(subjects_dir=subjects_dir)
        sxfm.inputs.source_file = in_file
        sxfm.inputs.out_file = in_file.replace('fsnative', 'fsaverage')
        sxfm.inputs.source_subject = source_subject
        sxfm.inputs.target_subject = 'fsaverage'
        sxfm.inputs.hemi = fs_hemi

        r = sxfm.run()
        return r

def main(subject, bids_folder):
    
    sub = Subject(subject, bids_folder=bids_folder)
    surfinfo = sub.get_surf_info()

    target_dir = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat')

    map_5mm = op.join(target_dir, f'sub-{subject}_space-T1w_desc-NPCr1cm-surface_mask.nii.gz')

    for hemi in ['R']:
        samples = surface.vol_to_surf(map_5mm, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples.astype(np.float32))])
        target_fn =  op.join(target_dir, f'sub-{subject}_space-fsnative_desc-NPCr1cm_hemi-{hemi}.anat.gii')
        nb.save(im, target_fn)
        transform_fsaverage(target_fn, fs_hemi, f'sub-{subject}', bids_folder)

def transform_group(bids_folder):
    subjects = [f'{subject:02d}' for subject in get_tms_subjects(bids_folder)]

    for subject in tqdm(subjects):
        try:
            main(subject, bids_folder)
        except Exception as e:
            print(e)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--transform_group', action='store_true')
    args = parser.parse_args()

    if args.transform_group:
        transform_group(args.bids_folder)
        # pass
    else:
        main(args.subject, bids_folder=args.bids_folder)
