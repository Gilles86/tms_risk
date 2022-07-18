import numpy as np
import os
import os.path as op
import argparse
from tkinter import W
from get_npc_mask import main as get_npc
from nilearn import image
from tms_risk.utils import Subject
from nilearn.input_data.nifti_spheres_masker import _apply_mask_and_get_affinity

def main(subject, bids_folder):

    s = Subject(subject, bids_folder)
    t1w = s.t1w

    npc_mask = s.get_volume_mask('NPC12r')
    npc_mask = image.resample_to_img(npc_mask, t1w)

    if npc_mask.ndim > 3:
        npc_mask = image.index_img(npc_mask, 0)

    # r2_unsmoothed = s.get_nprf_pars(model='encoding_model')
    # r2_unsmoothed = image.resample_to_img(r2_unsmoothed, t1w)
    
    r2 = s.get_nprf_pars(model='encoding_model.smoothed.pca_confounds')
    r2 = image.resample_to_img(r2, t1w)
    thr_r2_90 = image.math_img('np.where(r2 > .15, r2, 0.0)', r2=r2)

    thr_r2 = image.math_img('npc_mask*r2', r2=r2, npc_mask=npc_mask)



    coords, ball = get_coords_ball(thr_r2)

    target_dir = op.join(bids_folder, 'derivatives',
    'tms_navigation', f'sub-{subject}')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for im, label, interpolation in zip([t1w, npc_mask, r2, ball, thr_r2_90],
    ['T1w', 'NPC12', 'r2', 'com', 'r2.thr'], ['linear', 'nearest', 'linear', 'nearest', 'linear']):

        im = go_to_marius_space(im, interpolation=interpolation)
        im.to_filename(op.join(target_dir, f'sub-{subject}_{label}.nii.gz'))



def go_to_marius_space(im, interpolation='continuous'):
    im = image.load_img(im)
    new_affine = np.identity(4)
    new_affine[:3, -1] = np.linalg.inv(im.affine[:3, :3]).dot(im.affine[:3, -1])
    
    im2 = image.resample_img(im, new_affine, interpolation=interpolation)
    
    im2.affine[:3, -1] = 0.0
    
    return im2

def get_coords_ball(r2):
    vox_coords = np.unravel_index(r2.get_fdata().argmax(), r2.shape)
    coords = image.coord_transform(*vox_coords, r2.affine)
    _, ball =_apply_mask_and_get_affinity([list(coords)], image.concat_imgs([r2]), 5., True)
    ball = ball.toarray().reshape(r2.shape)* 100.
    ball = image.new_img_like(r2, ball.astype(int))
    
    return coords, ball


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument(
        '--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.subject, args.bids_folder)
