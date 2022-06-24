import re
import os
import os.path as op
import shutil
import argparse
import pandas as pd
import glob
from nilearn import image
import json

def main(subject, session, bids_folder='/data/ds-tms'):
    sourcedata_root = op.join(bids_folder, 'sourcedata', 'mri',
    f'SNS_MRI_RTMS_S{subject:05d}_{session:02d}')

    # *** ANATOMICAL DATA ***
    t1w = glob.glob(op.join(sourcedata_root, '*t1w*.nii'))
    assert(len(t1w) == 1), "More than 1/no T1w {t1w}"

    flair = glob.glob(op.join(sourcedata_root, '*flair*.nii'))
    assert(len(flair) == 1), f"More than 1/no FLAIR {flair}"

    target_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'anat')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    shutil.copy(t1w[0], op.join(target_dir, f'sub-{subject:02d}_ses-{session}_T1w.nii'))
    shutil.copy(flair[0], op.join(target_dir, f'sub-{subject:02d}_ses-{session}_FLAIR.nii'))

    # *** FUNCTIONAL DATA ***
    with open(op.abspath('./bold_template.json'), 'r') as f:
        json_template = json.load(f)
        print(json_template)

    reg = re.compile('.*run(?P<run>[0-9]+).*')
    funcs = glob.glob(op.join(sourcedata_root, '*run*.nii'))

    runs = [int(reg.match(fn).group(1)) for fn in funcs]

    target_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'func')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    for run, fn in zip(runs, funcs):
        shutil.copy(fn, op.join(target_dir, f'sub-{subject:02d}_ses-{session}_task-task_run-{run}_bold.nii'))

        json_sidecar = json_template
        json_sidecar['PhaseEncodingDirection'] = 'i' if (run % 2 == 1) else 'i-'

        with open(op.join(target_dir, f'sub-{subject:02d}_ses-{session}_task-task_run-{run}_bold.json'), 'w') as f:
            json.dump(json_sidecar, f)

    # *** Fieldmaps ***
    func_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'func')
    target_dir = op.join(bids_folder, f'sub-{subject:02d}', f'ses-{session}', 'fmap')
    if not op.exists(target_dir):
        os.makedirs(target_dir)

    with open(op.abspath('./fmap_template.json'), 'r') as f:
        json_template = json.load(f)
        print(json_template)
  
    for target_run in range(1, 7):
        source_run = target_run + 1
        index_slice = slice(5, 10)
        if source_run == 7:
            source_run = 2
            index_slice = slice(-10, -5)
        
        direction = 'left' if (source_run % 2 == 0) else 'right'

        epi = op.join(func_dir, f'sub-{subject:02d}_ses-{session}_task-task_run-{source_run}_bold.nii')
        epi = image.index_img(epi, index_slice)

        target_fn = op.join(target_dir, f'sub-{subject:02d}_ses-{session}_dir-{direction}_run-{target_run}_epi.nii')
        epi.to_filename(target_fn)

        json_sidecar = json_template
        json_sidecar['PhaseEncodingDirection'] = 'i' if (source_run % 2 == 1) else 'i-'
        json_sidecar['IntendedFor'] = f'ses-{session}/func/sub-{subject:02d}_ses-{session}_task-task_run-{target_run}_bold.nii'

        with open(op.join(target_dir, f'sub-{subject:02d}_ses-{session}_dir-{direction}_run-{target_run}_epi.json'), 'w') as f:
            json.dump(json_sidecar, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', type=int)
    parser.add_argument('session', type=int)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk/')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder)