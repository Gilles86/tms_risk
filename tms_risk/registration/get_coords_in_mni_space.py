import argparse
from nipype.interfaces import ants
import os
import os.path as op
from nilearn import image
from nilearn import plotting
from tms_risk.utils.data import get_tms_subjects
from tqdm import tqdm


def main(subject, bids_folder):

    if type(subject) == int:
        subject = f'{subject:02d}'

    distance_map = op.join(bids_folder, 'derivatives', 'stim_coordinates', f'sub-{subject}', f'sub-{subject}_coords_surface_distance.nii.gz')  

    map_5mm = image.load_img(distance_map)
    map_5mm = image.math_img('im<5', im=map_5mm)
    map_5mm.to_filename(op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-NPCr5mm-surface_mask.nii.gz'))

    transformer = ants.ApplyTransforms()
    transformer.inputs.dimension = 3
    transformer.inputs.input_image = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-T1w_desc-NPCr5mm-surface_mask.nii.gz')
    fsl_dir = '/usr/local/fsl'
    transformer.inputs.reference_image = op.join(fsl_dir, 'data', 'standard', 'MNI152_T1_0.5mm.nii.gz')
    transformer.inputs.transforms = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}', 'ses-1', 'anat', f'sub-{subject}_ses-1_from-T1w_to-MNI152NLin2009cAsym_mode-image_xfm.h5')
    transformer.inputs.environ = {'ANTSPATH': '/Users/gdehol/ants/bin', 'PATH': '/Users/gdehol/ants/bin:/usr/local/fsl/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:/opt/X11/bin:/Library/TeX/texbin:/usr/local/go/bin:/Users/gdehol/go/bin:/Users/gdehol/ants/bin'}
    transformer.inputs.output_image = op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject}', 'anat', f'sub-{subject}_space-MNI152NLin2009cAsym_desc-NPCr5mm-surface_mask.nii.gz')
    transformer.run() 

def fit_group(bids_folder):
    subjects = get_tms_subjects(bids_folder)

    for subject in tqdm(subjects):
        main(subject, bids_folder)

    maps = [op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{subject:02d}', 'anat', f'sub-{subject:02d}_space-MNI152NLin2009cAsym_desc-NPCr5mm-surface_mask.nii.gz') for subject in subjects]

    maps = image.concat_imgs(maps)
    mean_map = image.mean_img(maps)

    if not op.exists(op.join(bids_folder, 'derivatives', 'ips_masks', 'group', 'anat')):
        os.makedirs(op.join(bids_folder, 'derivatives', 'ips_masks', 'group', 'anat'))

    mean_map.to_filename(op.join(bids_folder, 'derivatives', 'ips_masks', 'group', 'anat', 'group_space-MNI152NLin2009cAsym_desc-NPCr5mm-surface_mean_mask.nii.gz'))

    thr_maps = [image.math_img('im>0.5', im=map) for map in image.iter_img(maps)]
    thr_maps = image.concat_imgs(thr_maps)
    mean_thr_map = image.mean_img(thr_maps)
    mean_thr_map = image.math_img('im*35', im=mean_thr_map)
    mean_thr_map.to_filename(op.join(bids_folder, 'derivatives', 'ips_masks', 'group', 'anat', 'group_space-MNI152NLin2009cAsym_desc-NPCr5mm-surface_mean_thr_mask.nii.gz'))

if __name__ == '__main__':
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument('subject', default=None)
    argument_parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    argument_parser.add_argument('--fit_group', action='store_true')
    args = argument_parser.parse_args()

    if args.fit_group:
        fit_group(args.bids_folder)
    else:
        main(args.subject, args.bids_folder)