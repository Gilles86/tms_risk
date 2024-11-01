import argparse
import cortex
import os.path as op
from cortex import webgl 
from nilearn import image
from nilearn import surface
import numpy as np
from utils import get_alpha_vertex

from tms_risk.utils.data import Subject, get_subjects
from tqdm.contrib.itertools import product


def main(bids_folder, thr=.1):


    subjects = get_subjects(bids_folder, all_tms_conditions=True)


    vertices = {}
    
    fs_subject = space = 'fsaverage'

    for sub, session in product(subjects, [1,2,3][:1]):
        prf_pars = sub.get_prf_parameters_surf(session, ['r2'],  smoothed=True, nilearn=True, space=space)
        mask = (prf_pars['r2'] > 0.1).values
        # mu_vertex = get_alpha_vertex(prf_pars['mu'].values, mask, vmin=5, vmax=28, subject=fs_subject) 
        r2_vertex = get_alpha_vertex(prf_pars['r2'].values, mask, cmap='hot', vmin=0.0, vmax=0.25, subject=fs_subject)
        # cvr2_vertex = get_alpha_vertex(prf_pars['cvr2'].values, mask, cmap='hot', vmin=0.0, vmax=0.25, subject=fs_subject)

        if sub.subject != '45':
            target = surface.load_surf_data(op.join(bids_folder, 'derivatives', 'ips_masks', f'sub-{sub.subject}', 'anat', f'sub-{sub.subject}_space-fsaverage_desc-NPCr5mm_geodesic_hemi-R.anat.gii'))
            target = np.concatenate((np.zeros_like(target), target))
            vertices[f'{sub.subject}_target'] = cortex.Vertex(target, 'fsaverage', vmin=0.0, vmax=1.0, cmap='viridis')

        # vertices[f"{sub.subject}_mu_{session}"] = mu_vertex
        vertices[f"{sub.subject}_r2_{session}"] = r2_vertex
        # vertices[f"{sub.subject_id}_cvr2_{session}"] = cvr2_vertex

    vertices = {k: v for k, v in sorted(vertices.items(), key=lambda item: item[0])}
    webgl.show(vertices)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()
    main(args.bids_folder)
