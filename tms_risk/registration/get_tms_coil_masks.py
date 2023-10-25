import os
import os.path as op
import argparse
import pandas as pd
import nipype.pipeline.engine as pe
from nipype.interfaces import utility as niu
from nipype.interfaces import io as nio
from nipype.interfaces import fsl
import nibabel as nb
import numpy as np
from nilearn import image


def main(subject, bids_folder):


    target_dir = op.join(bids_folder, 'derivatives', 'stim_coordinates', f'sub-{subject}')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    coords = np.loadtxt(op.join(bids_folder, 'derivatives', 'stim_coordinates', f'sub-{subject}', f'sub-{subject}_coords_warped.txt'))
    T1w = image.load_img(op.join(bids_folder, f'derivatives/fmriprep/sub-{subject}/ses-1/anat/sub-{subject}_ses-1_desc-preproc_T1w.nii.gz'))
    npcr = image.load_img(op.join(bids_folder, f'derivatives/ips_masks/sub-{subject}/anat/sub-{subject}_space-T1w_desc-NPCr_mask.nii.gz'))

    if len(npcr.shape) > 3:
        npcr = image.index_img(npcr, 0)

    x, y, z = np.indices(image.load_img(T1w).shape)
    xyz1 = np.vstack((x.ravel(), y.ravel(), z.ravel(), np.ones_like(x).ravel()))
    t1w_coords = image.new_img_like(T1w, np.dot(T1w.affine, xyz1).T.reshape(T1w.shape + (4,)))

    t1w_coords.to_filename(op.join(target_dir, f'sub-{subject}_coords_T1w.nii.gz'))

    for label, c in zip(['surface', 'cluster'], coords):
        c = np.append(c, 1)
        distance = np.sqrt(((t1w_coords.get_fdata() - c[None, None, None, :])**2).sum(axis=-1))
        distance_img = image.new_img_like(T1w, distance)
        distance_img.to_filename(op.join(target_dir, f'sub-{subject}_coords_{label}_distance.nii.gz'))
        mask_1cm = image.math_img('im < 10', im=distance_img)
        mask_2cm = image.math_img('im < 20', im=distance_img)

        mask_1cm.to_filename(op.join(target_dir, f'sub-{subject}_coords_{label}_mask_1cm.nii.gz'))
        mask_2cm.to_filename(op.join(target_dir, f'sub-{subject}_coords_{label}_mask_2cm.nii.gz'))

        npcr1cm = image.math_img('npcr * mask', mask=mask_1cm, npcr=npcr)
        npcr2cm = image.math_img('npcr * mask', mask=mask_2cm, npcr=npcr)

        npcr1cm.to_filename(op.join(bids_folder, f'derivatives/ips_masks/sub-{subject}/anat/sub-{subject}_space-T1w_desc-NPCr1cm-{label}_mask.nii.gz'))
        npcr2cm.to_filename(op.join(bids_folder, f'derivatives/ips_masks/sub-{subject}/anat/sub-{subject}_space-T1w_desc-NPCr2cm-{label}_mask.nii.gz'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.subject, args.bids_folder)
