import argparse
import os.path as op
from tms_risk.utils.data import Subject
from nilearn import surface
import nibabel as nb
from tms_risk.encoding_model.fit_nprf import get_key_target_dir
from tqdm import tqdm
from nipype.interfaces.freesurfer import SurfaceTransform


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

def main(subject, session, bids_folder, smoothed):
    
    sub = Subject(subject, bids_folder=bids_folder)
    surfinfo = sub.get_surf_info()

    par_keys = ['mu', 'sd', 'amplitude', 'baseline', 'cvr2', 'r2']

    prf_pars_volume = sub.get_prf_parameters_volume(session, smoothed=smoothed, denoise=True, keys=par_keys, natural_space=True, retroicor=False, pca_confounds=False,
                                                    return_image=True, cross_validated=False)

    _, target_dir = get_key_target_dir(f'{int(subject):02d}', session, bids_folder, smoothed, denoise=True, pca_confounds=False, retroicor=False, natural_space=True)    

    print(f'Writing to {target_dir}')

    for hemi in ['L', 'R']:
        samples = surface.vol_to_surf(prf_pars_volume, surfinfo[hemi]['outer'], inner_mesh=surfinfo[hemi]['inner'])
        fs_hemi = 'lh' if hemi == 'L' else 'rh'

        for ix, par in enumerate(par_keys):
            im = nb.gifti.GiftiImage(darrays=[nb.gifti.GiftiDataArray(samples[:, ix])])
            target_fn =  op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim.nilearn_space-fsnative_hemi-{hemi}.func.gii')
            nb.save(im, target_fn)

            transform_fsaverage(target_fn, fs_hemi, f'sub-{subject}', bids_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, smoothed=args.smoothed)
