import argparse
import pandas as pd
from braincoder.models import RegressionGaussianPRF
from braincoder.optimize import ParameterFitter
from braincoder.utils import get_rsq
from tms_risk.utils import get_target_dir, Subject
from nilearn.input_data import NiftiMasker
from pathlib import Path
import os.path as op
import numpy as np
import re
from fit_regression_nprf import get_model, get_grid

def main(subject, model_label=1, bids_folder='/data/ds-tmsrisk', natural_space=False):
    bids_folder = Path(bids_folder)
    target_dir = bids_folder / 'derivatives' / f'encoding_model2.model-{model_label}.smoothed.cv' / f'sub-{subject}'
    target_dir.mkdir(parents=True, exist_ok=True)

    sub = Subject(subject, bids_folder=bids_folder)
    paradigm = sub.get_paradigm()
    if natural_space:
        raise NotImplementedError("Natural space not implemented yet for regression nPRF")
    print(paradigm)

    paradigm = paradigm.reset_index(['session'])[['log(n1)', 'session']].rename(columns={'log(n1)': 'x'}).astype(np.float32)

    mask = sub.get_volume_mask(session=1, roi=None, epi_space=True)
    masker = NiftiMasker(mask_img=mask)

    # Load data for both sessions
    data2 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject}' / f'ses-2' / 'func' / f'sub-{subject}_ses-2_task-task_space-T1w_desc-stims1_pe.nii.gz'
    data3 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject}' / f'ses-3' / 'func' / f'sub-{subject}_ses-3_task-task_space-T1w_desc-stims1_pe.nii.gz'

    # Stack data and paradigm
    data = pd.DataFrame(
        np.vstack([masker.fit_transform(data2), masker.fit_transform(data3)]),
        index=paradigm.index
    ).astype(np.float32)

    cv_r2s = []
    # paradigm = paradigm.set_index('run', append=True)
    runs = paradigm.index.unique(level='run')

    for test_run in runs:
        # Exclude the same run from both sessions
        test_data = data.loc[paradigm.index.get_level_values('run') == test_run].copy()
        test_paradigm = paradigm.loc[paradigm.index.get_level_values('run') == test_run].copy()

        train_data = data.loc[paradigm.index.get_level_values('run') != test_run].copy()
        train_paradigm = paradigm.loc[paradigm.index.get_level_values('run') != test_run].copy()

        train_model = get_model(model_label, train_paradigm)
        optimizer = ParameterFitter(train_model, train_data, train_paradigm)
        grid = get_grid(model_label)
        grid_parameters = optimizer.fit_grid(*grid, use_correlation_cost=True)

        if model_label in [0, 1]:
            fixed_pars = [('mu_unbounded', 'Intercept'), ('sd_unbounded', 'Intercept')]
        elif model_label in [2]:
            fixed_pars = [('mu_unbounded', 'C(session)[2.0]'),('mu_unbounded', 'C(session)[3.0]'),
                        ('sd_unbounded', 'C(session)[2.0]'), ('sd_unbounded', 'C(session)[3.0]')]

        # Refine and fit
        grid_parameters = optimizer.fit(
            init_pars=grid_parameters,
            learning_rate=.05,
            store_intermediate_parameters=False,
            max_n_iterations=10,
            fixed_pars=fixed_pars,
            r2_atol=0.00001
        )
        optimizer.fit(
            init_pars=grid_parameters,
            learning_rate=.05,
            store_intermediate_parameters=False,
            max_n_iterations=10,
            r2_atol=0.00001
        )

        # Save R² for this fold
        target_fn = target_dir / f'sub-{subject}_run-{test_run}_desc-r2.optim_space-T1w_pars.nii.gz'
        masker.inverse_transform(optimizer.r2).to_filename(target_fn)

        test_model = get_model(model_label, test_paradigm)
        # Calculate and save cross-validated R²
        cv_r2 = get_rsq(
            test_data,
            test_model.predict(parameters=optimizer.estimated_parameters, paradigm=test_paradigm.astype(np.float32))
        )
        target_fn = target_dir / f'sub-{subject}_run-{test_run}_desc-cvr2.optim_space-T1w_pars.nii.gz'
        masker.inverse_transform(cv_r2).to_filename(target_fn)

        # Save session-specific parameters for this fold
        conditions = pd.DataFrame({'session': [2, 3]}).set_index('session', drop=False)
        pars = train_model.get_conditionspecific_parameters(conditions, optimizer.estimated_parameters)
        for session, pars_ in pars.groupby('session'):
            session_dir = target_dir / f'ses-{session}'
            session_dir.mkdir(parents=True, exist_ok=True)
            for par, values in pars_.T.iterrows():
                par_label = sanitize_filename(f'{par[0]}_{par[1]}') if isinstance(par, tuple) else par
                target_fn = session_dir / f'sub-{subject}_ses-{session}_run-{test_run}_desc-{par_label}.optim_space-T1w_pars.nii.gz'
                print(f'Writing {par_label} for session {session} (run {test_run}) to {target_fn}.')
                masker.inverse_transform(values).to_filename(target_fn)

        cv_r2s.append(cv_r2)

    # Average cross-validated R² across runs
    cv_r2 = pd.concat(cv_r2s, keys=runs, names=['run']).groupby(level=1, axis=0).mean()
    target_fn = target_dir / f'sub-{subject}_desc-cvr2.optim_space-T1w_pars.nii.gz'
    masker.inverse_transform(cv_r2).to_filename(target_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('model_label', default=1, type=int)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()
    main(args.subject, model_label=args.model_label, bids_folder=args.bids_folder)
