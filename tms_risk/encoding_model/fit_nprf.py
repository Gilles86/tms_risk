import argparse
import pandas as pd
from braincoder.models import GaussianPRF, LogGaussianPRF
from braincoder.optimize import ParameterFitter
from nilearn.input_data import NiftiMasker
from tms_risk.utils import get_target_dir, Subject

import os
import os.path as op
import numpy as np

def get_key_target_dir(subject, session, bids_folder, smoothed, denoise, pca_confounds, retroicor, natural_space, only_target_key=False,
                       new_parameterisation=False):

    key = 'glm_stim1'
    target_dir = 'encoding_model'

    if denoise:
        key += '.denoise'
        target_dir += '.denoise'

    if (retroicor) and (not denoise):
        raise Exception("When not using GLMSingle RETROICOR is *always* used!")

    if retroicor:
        key += '.retroicor'
        target_dir += '.retroicor'

    if smoothed:
        key += '.smoothed'
        target_dir += '.smoothed'

    if pca_confounds:
        target_dir += '.pca_confounds'
        key += '.pca_confounds'

    if natural_space:
        target_dir += '.natural_space'

    if new_parameterisation:
        target_dir += '.new_parameterisation'

    if only_target_key:
        return target_dir

    target_dir = op.join(bids_folder, 'derivatives', target_dir, f'sub-{subject}', f'ses-{session}', 'func' )

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    return key, target_dir

def main(subject, session, bids_folder='/data/ds-tmsrisk', smoothed=False,
        denoise=False,
        pca_confounds=False, retroicor=False, natural_space=False,
        new_parameterisation=False):

    if new_parameterisation and (not natural_space):
        raise ValueError("New parameterisation only makes sense in natural space")

    sub = Subject(subject, bids_folder=bids_folder)

    key, target_dir = get_key_target_dir(subject, session, bids_folder, smoothed, denoise, pca_confounds, retroicor, natural_space)

    print("TARGET DIR", target_dir)

    runs = range(1, 7)
    if (str(subject) == '10') & (str(session) == '1'):
        runs = range(1, 6)


    paradigm = [pd.read_csv(op.join(bids_folder, f'sub-{subject}', f'ses-{session}',
                               'func', f'sub-{subject}_ses-{session}_task-task_run-{run}_events.tsv'), sep='\t')
                for run in runs]
    paradigm = pd.concat(paradigm, keys=runs, names=['run'])
    paradigm = paradigm[paradigm.trial_type == 'stimulus 1'].set_index('trial_nr')
    paradigm['log(n1)'] = np.log(paradigm['n1'])


    if natural_space:
        paradigm = paradigm['n1']
        
        if new_parameterisation:
            model = LogGaussianPRF(parameterisation='mode_fwhm_natural')
            # SET UP GRID
            modes = np.linspace(5, 80, 60, dtype=np.float32)
            fwhms = np.linspace(5, 40, 60, dtype=np.float32)
            amplitudes = np.array([1.], dtype=np.float32)
            baselines = np.array([0], dtype=np.float32)
        else:
            model = LogGaussianPRF()
            # SET UP GRID
            mus = np.linspace(5, 80, 60, dtype=np.float32)
            sds = np.linspace(5, 40, 60, dtype=np.float32)
            amplitudes = np.array([1.], dtype=np.float32)
            baselines = np.array([0], dtype=np.float32)
    else:
        paradigm = paradigm['log(n1)']
        model = GaussianPRF()
        # SET UP GRID
        mus = np.log(np.linspace(5, 80, 60, dtype=np.float32))
        sds = np.log(np.linspace(2, 30, 60, dtype=np.float32))
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)

    # mask = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')
    mask = sub.get_volume_mask(session=session, roi=None, epi_space=True)
    masker = NiftiMasker(mask_img=mask)

    data = op.join(bids_folder, 'derivatives', key,
                                          f'sub-{subject}', f'ses-{session}', 'func', f'sub-{subject}_ses-{session}_task-task_space-T1w_desc-stims1_pe.nii.gz')

    data = pd.DataFrame(masker.fit_transform(data), index=paradigm.index)
    print(data)

    data = pd.DataFrame(data, index=paradigm.index)

    optimizer = ParameterFitter(model, data, paradigm)

    if new_parameterisation:
        grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)
        grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=2)
        optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                      fixed_parameters=['mode', 'fwhm'],
                        r2_atol=0.00001)
    else:
        grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, baselines, use_correlation_cost=True)
        grid_parameters = optimizer.refine_baseline_and_amplitude(grid_parameters, n_iterations=2)
        optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                      fixed_parameters=['mu', 'sd'],
                        r2_atol=0.00001)


    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
            r2_atol=0.00001)

    target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-r2.optim_space-T1w_pars.nii.gz')

    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    for par, values in optimizer.estimated_parameters.T.iterrows():
        print(values)
        target_fn = op.join(target_dir, f'sub-{subject}_ses-{session}_desc-{par}.optim_space-T1w_pars.nii.gz')
        masker.inverse_transform(values).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--retroicor', action='store_true')
    parser.add_argument('--natural_space', action='store_true')
    parser.add_argument('--new_parameterisation', action='store_true')
    args = parser.parse_args()

    main(args.subject, args.session, bids_folder=args.bids_folder, smoothed=args.smoothed,
            pca_confounds=args.pca_confounds, denoise=args.denoise, retroicor=args.retroicor,
            natural_space=args.natural_space, new_parameterisation=args.new_parameterisation)
