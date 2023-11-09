import argparse
import cortex
import os.path as op
from cortex import webgl 
from nilearn import image
from nilearn import surface
import numpy as np


def main(subject, bids_folder, thr=.1):

    subject = int(subject)

    r2 = op.join(bids_folder, 'derivatives', 'encoding_model.denoise.smoothed.natural_space',
            f'sub-{subject:02d}', f'ses-1', 'func', f'sub-{subject:02d}_ses-1_desc-r2.optim_space-T1w_pars.nii.gz')

    r2 = image.load_img(r2)

    mu = op.join(bids_folder, 'derivatives', 'encoding_model.denoise.smoothed.natural_space',
            f'sub-{subject:02d}', f'ses-1', 'func', f'sub-{subject:02d}_ses-1_desc-mu.optim_space-T1w_pars.nii.gz')

    mu = image.load_img(mu).get_data()
    mu[r2.get_data() < thr] = np.nan
    mu = cortex.Volume(mu.T, f'tms.sub-{subject:02d}', 'epi.identity',
            vmin=1, vmax=3.)

#     npc1_r = op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer', f'sub-{subject:02d}', 'surf', 'rh.npc1.mgz')
#     npc1_r = image.load_img(npc1_r).get_data().ravel()
#     npc1_r = np.where(npc1_r != 0.0, npc1_r, np.nan)
#     npc1_r = cortex.Vertex(npc1_r, f'tms.sub-{subject:02d}', vmin=0, vmax=1)

#     npc2_r = op.join(bids_folder, 'derivatives', 'fmriprep', 'sourcedata', 'freesurfer', f'sub-{subject:02d}', 'surf', 'rh.npc2.mgz')
#     npc2_r = image.load_img(npc2_r).get_data().ravel()
#     npc2_r = np.where(npc2_r != 0.0, npc2_r, np.nan)
#     npc2_r = cortex.Vertex(npc2_r, f'tms.sub-{subject:02d}', vmin=0, vmax=1)

    r2  = image.math_img(f'np.where(r2>{thr}, r2, np.nan)', r2=r2)
    r2 = cortex.Volume(r2.get_data().T, f'tms.sub-{subject:02d}', 'epi.identity',
            vmin=0.1, vmax=.25, cmap='hot')

    ds = cortex.Dataset(r2=r2, mu=mu)

    webgl.show(ds)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject')
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--fsnative', action='store_true')
    args = parser.parse_args()
    main(args.subject, args.bids_folder)
