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
sessions = ['1', '2', '3']#[1:]
masks = ['NPCr']

n_voxels = [0, 100]

smoothed = [False]
pca_confounds = [False]
retroicors = [False]

denoises = [True]

subjects = ['04', '05', '06', '07', '09']

sessions = ['1', '2', '3']

# subjects = ['45', '72']
# sessions = ['2']
subjects = ['10']
sessions = ['1']

subjects = ['45']
sessions = ['2']

for ix, (subject, session, mask, nv, smooth, pcc, denoise, retroicor) in enumerate(product(subjects, sessions, masks, n_voxels, smoothed, pca_confounds, denoises, retroicors)):
# for ix, (nv, subject, session, mask) in enumerate(missing):
    print(f'*** RUNNING {subject}, {mask}, {nv}')

    job_file = os.path.join(job_directory, f"{ix}.job")
    
    now = datetime.now()
    time_str = now.strftime("%Y.%m.%d_%H.%M.%S")

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        id = f'{subject}.{session}.{mask}.{nv}.{smooth}.{pcc}.{retroicor}.{denoise}{time_str}'

        if denoise:
            id += '.denoise'

        if smooth:
            id += '.smoothed'

        if pcc:
            id += '.pca_confounds'

        if retroicor:
            id += '.retroicor'

        fh.writelines(f"#SBATCH --job-name=decode_volume.{id}.job\n")
        fh.writelines(f"#SBATCH --output={os.environ['HOME']}/.out/decode_volume.{id}.txt\n")
        fh.writelines("#SBATCH --time=30:00\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --mem=96G\n")
        fh.writelines("#SBATCH --gres gpu:1\n")
        fh.writelines(". $HOME/init_conda.sh\n")
        fh.writelines("conda activate tf2-gpu\n")
        cmd = f"python $HOME/git/tms_risk/tms_risk/encoding_model/decode.py {subject} {session} --bids_folder /home/gdehol/share/ds-tmsrisk --n_voxels {nv} --mask {mask}"

        if denoise:
            cmd += ' --denoise'

        if smooth:
            cmd += ' --smoothed'

        if pcc:
            cmd += ' --pca_confounds'

        if retroicor:
            cmd += ' --retroicor'

        fh.writelines(cmd)
        print(cmd)

    os.system("sbatch %s" %job_file)
