import argparse
import pandas as pd
from tqdm.contrib.itertools import product
import matplotlib.pyplot as plt
import numpy as np
import cortex
from tms_risk.utils import get_subjects
from utils import get_alpha_vertex
from scipy import stats as ss
from tms_risk.utils.data import get_tms_conditions
from tqdm import tqdm

vranges = {'mu':(5, 28), 'cvr2':(0.0, 0.05), 'r2':(0.0, 0.10), 'mu_natural':(5, 28), 'sd':(2, 12), 'amplitude':(0, 2)}
cmaps = {'mu':'nipy_spectral', 'cvr2':'hot', 'r2':'plasma', 'mu_natural':'nipy_spectral', 
'sd':'hot', 'amplitude':'viridis'}

def main(bids_folder, thr, smoothed, threshold_on='r2'):


    subjects = get_subjects(bids_folder, all_tms_conditions=True)
    

    pars = []

    for sub in tqdm(subjects):
        pars.append(sub.get_prf_parameters_surf(1, space='fsaverage', smoothed=smoothed, parameters=['cvr2', 'r2'], nilearn=True))
    
    pars = pd.concat(pars, keys=[sub.subject for sub in subjects], names=['subject'])


    mean_pars = pars.groupby(['hemi', 'vertex']).mean()

    alpha = np.round(ss.norm(thr, 0.005).cdf(mean_pars[threshold_on].values), 2)
    alpha_cvr2 = np.round(ss.norm(0.0, 0.005).cdf(mean_pars['cvr2'].values), 2)

    ds = {}

    ds['r2'] = get_alpha_vertex(mean_pars['r2'].values, alpha, standard_space=True, subject='fsaverage',
                                vmin=vranges['r2'][0], vmax=vranges['r2'][1], cmap=cmaps['r2'])

    ds['cvr2'] = get_alpha_vertex(mean_pars['cvr2'].values, alpha, standard_space=True, subject='fsaverage',
                                vmin=vranges['cvr2'][0], vmax=vranges['cvr2'][1], cmap=cmaps['cvr2'])

    ds['r2_unthr'] = cortex.Vertex(mean_pars['r2'].values, 'fsaverage', vmin=vranges['r2'][0], vmax=vranges['r2'][1], cmap='afmhot')
    

    cortex.webshow(ds)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()
    main(args.bids_folder, thr=args.threshold, smoothed=args.smoothed,)
