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

vranges = {'mu':(5, 80), 'cvr2':(0.0, 0.05), 'r2':(0.0, 0.09), 'mu_natural':(5, 80), 'sd':(2, 12), 'amplitude':(0, 2)}
cmaps = {'mu':'nipy_spectral', 'cvr2':'afmhot', 'r2':'afmhot', 'mu_natural':'nipy_spectral', 
'sd':'hot', 'amplitude':'viridis'}

def get_average_vertex(session, bids_folder, thr, smoothed, show_unthresholded_map=True, show_pars=None,
threshold_on='r2'):


    subjects = get_subjects(bids_folder, all_tms_conditions=True)

    pars = []

    for sub in subjects:
        try:
            print(sub.subject, session)
            if str(session) not in ['1', '2', '3']:
                tms_mapping = {v: k for k, v in sub.tms_conditions.items()}
                session_ = tms_mapping[session]
            else:
                session_ = session

            pars.append(sub.get_prf_parameters_surf(session_, space='fsaverage', smoothed=smoothed, parameters=['mu', 'sd', 'cvr2', 'r2', 'amplitude']))
        except Exception as e:
            print(f'Problem with subject {sub.subject}: {e} ')
    
    pars = pd.concat(pars, keys=[sub.subject for sub in subjects], names=['subject'])
    # pars['mu_natural'] = np.exp(pars['mu'])

    if 'cvr2' in pars.columns:
        pars['cvr2'] = np.clip(pars['cvr2'], -0.01, 1)
        pars.loc[pars['r2'] < 0.01, 'r2'] = 0.0
        print(pars['cvr2'])
        print(pars['r2'])

    mean_pars = pars.groupby(['hemi', 'vertex']).mean()
    print(mean_pars)

    alpha = np.round(ss.norm(thr, 0.005).cdf(mean_pars[threshold_on].values), 2)
    alpha_cvr2 = np.round(ss.norm(0.0, 0.005).cdf(mean_pars['cvr2'].values), 2)


    ds = {}

    if show_pars is None:
        keys = pars.columns
    else:
        keys = show_pars

    print(keys)
    for key in keys:

        if show_unthresholded_map:
            ds[f'{session}.{key}'] = cortex.Vertex(mean_pars[key].values, 'fsaverage')

        if key in vranges.keys():
            vmin, vmax = vranges[key]
            cmap = cmaps[key]
            if key == 'cvr2':
                ds[f'{session}.{key}_thr'] = get_alpha_vertex(mean_pars['cvr2'].values, alpha_cvr2, standard_space=True, subject='fsaverage',
                vmin=vmin, vmax=vmax, cmap=cmap)
            else:
                ds[f'{session}.{key}_thr'] = get_alpha_vertex(mean_pars[key].values, alpha, standard_space=True, subject='fsaverage',
                vmin=vmin, vmax=vmax, cmap=cmap)
    
    return ds

def main(sessions, bids_folder, thr=-np.inf, smoothed=False, show_pars=None, threshold=0.05):

    if sessions is None:
        # sessions = ['3t2', '7t2']
        # sessions = ['1', '2', '3']
        sessions = ['baseline', 'ips', 'vertex']

    ds = {}
    for session in sessions:
        ds.update(get_average_vertex(session, bids_folder, threshold, smoothed, show_pars=show_pars))

    ds = cortex.Dataset(**ds)

    cortex.webshow(ds)

    vmin, vmax = vranges['mu']
    x = np.linspace(0, 1, 101, True)
    
    # Width is 80 x 1, so 
    im = plt.imshow(plt.cm.nipy_spectral(x)[np.newaxis, ...],
            extent=[vmin, vmax, 0, 1], aspect=1.*(vmax-vmin) / 20.,
            origin='lower')
    print(im.get_extent())
    plt.yticks([])
    plt.tight_layout()

    ns = np.array([5, 7, 10, 14, 20, 28, 40, 56, 80])
    ns = ns[ns <= vmax]
    plt.xticks(ns)
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--threshold', default=0.05, type=float)
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pars', nargs='+', default=None)
    parser.add_argument('--sessions', nargs='+', default=None)
    args = parser.parse_args()
    main(args.sessions, args.bids_folder, thr=args.threshold,
            smoothed=args.smoothed,)
