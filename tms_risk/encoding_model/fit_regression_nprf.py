import argparse
import pandas as pd
from braincoder.models import RegressionGaussianPRF
from braincoder.optimize import ParameterFitter
from nilearn.input_data import NiftiMasker
from tms_risk.utils import get_target_dir, Subject
from pathlib import Path
import os
import os.path as op
import numpy as np
import re

def get_model(model_label, paradigm):
    if model_label == 1:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors={'amplitude': '0 + C(session)'},)
    elif model_label == 0:
        model = RegressionGaussianPRF(paradigm=paradigm)
    elif model_label == 2:
        model = RegressionGaussianPRF(paradigm=paradigm, regressors={'amplitude': '0 + C(session)',
                                                                     'mu':'0 + C(session)',
                                                                     'sd':'0 + C(session)',
                                                                     'baseline': '0 + C(session)'},)
    else:
        raise NotImplementedError(f'Model label {model_label} has not been implemented')

    return model

def get_grid(model_label):

    mus = np.log(np.linspace(5, 80, 50, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 50, dtype=np.float32))
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)    

    if model_label == 1:
        return mus, sds, amplitudes, amplitudes, baselines
    elif model_label == 0:
        return mus, sds, amplitudes, baselines
    elif model_label == 2:
        return mus[::5], mus[::5], sds[::5], sds[::5], amplitudes, amplitudes, baselines, baselines


def main(subject, model_label=1, bids_folder='/data/ds-tmsrisk', natural_space=False):

    bids_folder = Path(bids_folder)

    target_dir = bids_folder / 'derivatives' / f'encoding_model2.model-{model_label}.smoothed' / f'sub-{subject}'

    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder=bids_folder)

    paradigm = sub.get_paradigm()

    if natural_space:
        raise NotImplementedError("Natural space not implemented yet for regression nPRF")

    paradigm = paradigm.reset_index('session')[['log(n1)', 'session']].rename(columns={'log(n1)': 'x'}).astype(np.float32)
    model = get_model(model_label, paradigm)


    # mask = op.join(bids_folder, 'derivatives', 'fmriprep', f'sub-{subject}/ses-{session}/func/sub-{subject}_ses-{session}_task-task_run-1_space-T1w_desc-brain_mask.nii.gz')
    mask = bids_folder / 'derivatives' / 'fmriprep' / f'sub-{subject}' / f'ses-1' / 'func' / f'sub-{subject}_ses-1_task-task_run-1_space-T1w_desc-brain_mask.nii.gz'
    mask = sub.get_volume_mask(session=1, roi=None, epi_space=True)
    masker = NiftiMasker(mask_img=mask)

    data2 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject}' / f'ses-2' / 'func' / f'sub-{subject}_ses-2_task-task_space-T1w_desc-stims1_pe.nii.gz'
    data3 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject}' / f'ses-3' / 'func' / f'sub-{subject}_ses-3_task-task_space-T1w_desc-stims1_pe.nii.gz'

    data = pd.DataFrame(np.vstack([masker.fit_transform(data2), masker.fit_transform(data3)]), index=paradigm.index)

    data = pd.DataFrame(data, index=paradigm.index).astype(np.float32)

    optimizer = ParameterFitter(model, data, paradigm)

    grid = get_grid(model_label)

    grid_parameters = optimizer.fit_grid(*grid, use_correlation_cost=True)
    

    if model_label in [0, 1]:
        fixed_pars = [('mu_unbounded', 'Intercept'), ('sd_unbounded', 'Intercept')]
    elif model_label in [2]:
        fixed_pars = [('mu_unbounded', 'C(session)[2.0]'),('mu_unbounded', 'C(session)[3.0]'),
                      ('sd_unbounded', 'C(session)[2.0]'), ('sd_unbounded', 'C(session)[3.0]')]
        
    grid_parameters = optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
                    fixed_pars=fixed_pars,
                    r2_atol=0.00001)


    optimizer.fit(init_pars=grid_parameters, learning_rate=.05, store_intermediate_parameters=False, max_n_iterations=10000,
            r2_atol=0.00001)


    target_fn = target_dir / f'sub-{subject}_desc-r2.optim_space-T1w_pars.nii.gz'
    masker.inverse_transform(optimizer.r2).to_filename(target_fn)

    def sanitize_filename(label):
        # Replace problematic characters with underscores
        label = re.sub(r'[(),\s]', '_', label)
        # Remove or replace other special characters if needed
        label = re.sub(r'[^\w\-_]', '', label)
        return label    

    conditions = pd.DataFrame({'session': [2, 3]}).set_index('session', drop=False)

    pars = model.get_conditionspecific_parameters(conditions, optimizer.estimated_parameters)


    for session, pars_ in pars.groupby('session'):
        session_dir = target_dir / f'ses-{session}'
        session_dir.mkdir(parents=True, exist_ok=True)

        for par, values in pars_.T.iterrows():
            print(values)
            par_label = sanitize_filename(f'{par[0]}_{par[1]}') if isinstance(par, tuple) else par
            target_fn = session_dir / f'sub-{subject}_ses-{session}_desc-{par_label}.optim_space-T1w_pars.nii.gz'
            print(f'Writing {par_label} for session {session} to {target_fn}.')
            masker.inverse_transform(values).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('model_label', default=1, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, bids_folder=args.bids_folder)
