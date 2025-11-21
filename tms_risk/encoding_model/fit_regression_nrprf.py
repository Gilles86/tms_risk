import argparse
import pandas as pd
from braincoder.models import RegressionGaussianPRF
from braincoder.optimize import ParameterFitter
from nilearn.input_data import NiftiMasker
from experiment import session
from tms_risk.utils import get_target_dir, Subject
from pathlib import Path
import os
import os.path as op
import numpy as np

def main(subject, bids_folder='/data/ds-tmsrisk', natural_space=False):

    target_dir = Path(bids_folder) / 'derivatives' / 'encoding_model2.denoise.smoothed'

    sub = Subject(subject, bids_folder=bids_folder)

    paradigm = sub.get_paradigm()

    if natural_space:
        raise NotImplementedError("Natural space not implemented yet for regression nPRF")
    else:
        paradigm = paradigm.reset_index('session')[['log(n1)', 'session']].rename(columns={'log(n1)': 'x'}).astype(np.float32)
        model = RegressionGaussianPRF(paradigm=paradigm, regressors={'amplitude': '0 + C(session)'},)
        # SET UP GRID
        mus = np.log(np.linspace(5, 80, 20, dtype=np.float32))
        sds = np.log(np.linspace(2, 30, 20, dtype=np.float32))
        amplitudes = np.array([1.], dtype=np.float32)
        baselines = np.array([0], dtype=np.float32)

    # mask = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')
    mask = bids_folder / 'derivatives' / 'fmriprep' / f'sub-{subject}' / f'ses-1' / 'func' / f'sub-{subject}_ses-1_task-task_run-1_space-T1w_desc-brain_mask.nii.gz'
    mask = sub.get_volume_mask(session=1, roi=None, epi_space=True)
    masker = NiftiMasker(mask_img=mask)

    data1 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject}' / f'ses-1' / 'func' / f'sub-{subject}_ses-1_task-task_space-T1w_desc-stims1_pe.nii.gz'
    data2 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject}' / f'ses-2' / 'func' / f'sub-{subject}_ses-2_task-task_space-T1w_desc-stims1_pe.nii.gz'

    data = pd.DataFrame(np.vstack([masker.fit_transform(data1), masker.fit_transform(data2)]), index=paradigm.index)

    data = pd.DataFrame(data, index=paradigm.index).astype(np.float32)

    optimizer = ParameterFitter(model, data, paradigm)

    grid_parameters = optimizer.fit_grid(mus, sds, amplitudes, amplitudes, baselines, use_correlation_cost=True)
    grid_parameters = optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                    fixed_pars=[('mu_unbounded', 'Intercept'), ('sd_unbounded', 'Intercept')],
                    r2_atol=0.00001)


    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
            r2_atol=0.00001)


    target_fn = target_dir / f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz'
    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    for par, values in optimizer.estimated_parameters.T.iterrows():
        print(values)
        par_label =f'{par[0]}_{par[1]}' if isinstance(par, tuple) else par
        target_fn = op.join(target_dir, f'sub-{subject}_desc-{par_label}.optim_space-T1w_pars.nii.gz')
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
