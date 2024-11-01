#!/usr/bin/env python

import os
import os.path as op
from datetime import datetime
from itertools import product
from tms_risk.utils.data import get_subjects

def mkdir_p(dir):
    '''make a directory (dir) if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)
    
job_directory = "%s/.job" %os.getcwd()

if not op.exists(job_directory):
    os.makedirs(job_directory)


bids_folder = '/scratch/gdehol/ds-tmsrisk'

subjects = [subject.subject for subject in get_subjects(bids_folder=bids_folder, all_tms_conditions=True)]

subjects = [3, 4, 21, 31, 35]

sessions = ['1', '2', '3']
masks = ['NPC1l', 'NPC1r', 'NPC2l', 'NPC2r', 'NPC3l', 'NPC3r', 'NTOl', 'NTOr', 'NF1l', 'NF1r', 'NF2l', 'NF2r']

smoothed = [False]
pca_confounds = [False]
retroicors = [False]
denoises = [True]
natural_space = [True]

for ix, (subject, session, mask, smooth, pcc, denoise, retroicor, ns) in enumerate(product(subjects, sessions, masks, smoothed, pca_confounds, denoises, retroicors, natural_space)):
# for ix, (nv, subject, session, mask) in enumerate(missing):
    print(f'*** RUNNING {subject}, {mask}')

    job_file = os.path.join(job_directory, f"{ix}.job")
    
    now = datetime.now()
    time_str = now.strftime("%Y.%m.%d_%H.%M.%S")

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        id = f'{subject}.{session}.{mask}.{smooth}.{pcc}.{retroicor}.{denoise}{time_str}'

        if denoise:
            id += '.denoise'

        if smooth:
            id += '.smoothed'

        if pcc:
            id += '.pca_confounds'

        if retroicor:
            id += '.retroicor'

        if natural_space:
            id += '.natural_space'

        fh.writelines(f"#SBATCH --job-name=decode_cv_voxels_volume.{id}.job\n")
        fh.writelines(f"#SBATCH --output=/data/{os.environ['USER']}/.out/decode_cv_voxels_volume.{id}.txt\n")
        fh.writelines("#SBATCH --time=45:00\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --mem=96G\n")
        # fh.writelines("#SBATCH --gres gpu:1\n")
        fh.writelines(". $HOME/init_conda.sh\n")
        fh.writelines("conda activate tf2-gpu\n")
        cmd = f"python $HOME/git/tms_risk/tms_risk/encoding_model/decode_select_voxels_cv.py {subject} {session} --bids_folder /shares/hare.econ.uzh/ds-tmsrisk --mask {mask} --keep_cached"

        if denoise:
            cmd += ' --denoise'

        if smooth:
            cmd += ' --smoothed'

        if pcc:
            cmd += ' --pca_confounds'

        if retroicor:
            cmd += ' --retroicor'

        if ns:
            cmd += ' --natural_space'

        fh.writelines(cmd)
        print(cmd)

    os.system("sbatch %s" %job_file)
