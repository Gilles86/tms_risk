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
# masks = ['NPCr1cm-cluster', 'NPCr1cm-surface', 'NPCr2cm-cluster', 'NPCr2cm-surface']
masks = ['NPCr', 'NPC12r']

n_voxels = [0, 100]



smoothed = [False]
pca_confounds = [False]
retroicors = [False]

denoises = [True]
natural_space = [True]

sessions = [2]

sessions = [3]
masks = ['NPC12r']
subjects = ['31', '35'][-2:]

# sessions = [3]
# masks = ['NPC12r']
# subjects = ['50', '53', '56', '59', '62', '63', '67', '69', '72', '74'][-2:]


# subjects = ['06', '10', '11', '25', '29', '45', '53', '56', '59', '62', '63', '67', '69', '72', '74']
# sessions = ['2']
# n_voxels = [100]

# subjects = ['05', '10', '25', '53', '56', '59', '62', '63', '67', '69', '74']
# sessions = ['3']
# n_voxels = [100]

# subjects = ['11', '72']
# sessions = [2]
# masks = ['NPCr']

# subjects = ['11', '18', '37', '47']
# sessions = [2]
# masks = ['NPC12r']

# subjects = ['37', '47', '50', '63']
# sessions = [2]
# masks = ['NPCr']
# n_voxels = [100]

# subjects = ['10', '11', '18', '26', '35', '37', '50', '63', '67']
# sessions = [2]
# masks = ['NPC12r']
# n_voxels = [100]

# subjects = ['21', '26', '35']
# sessions = [3]
# masks = ['NPCr']
# n_voxels = [0]

for ix, (subject, session, mask, nv, smooth, pcc, denoise, retroicor, ns) in enumerate(product(subjects, sessions, masks, n_voxels, smoothed, pca_confounds, denoises, retroicors, natural_space)):
# for ix, (nv, subject, session, mask) in enumerate(missing):
    print(f'*** RUNNING {subject}, {mask}, {nv}')

    job_file = os.path.join(job_directory, f"{ix}.job")
    
    now = datetime.now()
    time_str = now.strftime("%Y.%m.%d_%H.%M.%S")

    with open(job_file, 'w') as fh:
        fh.writelines("#!/bin/bash\n")
        id = f'cross_session.{subject}.{session}.{mask}.{nv}.{smooth}.{pcc}.{retroicor}.{denoise}{time_str}'

        if denoise:
            id += '.denoise'

        if smooth:
            id += '.smoothed'

        if pcc:
            id += '.pca_confounds'

        if retroicor:
            id += '.retroicor'

        if ns:
            id += '.natural_space'

        fh.writelines(f"#SBATCH --job-name=decode_volume.crosssession.{id}.job\n")
        fh.writelines(f"#SBATCH --output={os.environ['HOME']}/.out/decode_volume.crosssession.{id}.txt\n")
        fh.writelines("#SBATCH --time=45:00\n")
        fh.writelines("#SBATCH --ntasks=1\n")
        fh.writelines("#SBATCH --mem=96G\n")
        fh.writelines("#SBATCH --gres gpu:1\n")
        fh.writelines(". $HOME/init_conda.sh\n")
        fh.writelines("conda activate tf2-gpu\n")
        cmd = f"python $HOME/git/tms_risk/tms_risk/encoding_model/cross_session_decode.py {subject} {session} --bids_folder /home/gdehol/share/ds-tmsrisk --n_voxels {nv} --mask {mask}"

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
