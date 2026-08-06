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

# Which parameters are allowed to differ BETWEEN SESSIONS, per model label. Everything
# else is pooled. Sessions 2 and 3 are the two TMS arms; session 1 is the pre-TMS
# baseline and is only included when --sessions asks for it.
#
#   0  nothing                    5  amplitude + baseline  (RESPONSE MAGNITUDE)
#   1  amplitude                  4  mu + sd               (TUNING)
#   3  amplitude + sd             2  all four
#
# m4 vs m5 is the contrast that separates "what the population is tuned to" from "how
# strongly it responds" -- the paper's specificity claim. m3 mixes one of each and is
# kept only because it was fitted before that distinction was drawn.
SESSION_VARYING = {
    0: [],
    1: ['amplitude'],
    2: ['amplitude', 'mu', 'sd', 'baseline'],
    3: ['amplitude', 'sd'],
    4: ['mu', 'sd'],
    5: ['amplitude', 'baseline'],
}


# The two TMS sessions. Session 1 is the pre-TMS baseline and is deliberately NOT part
# of these fits -- the models describe the IPS-vs-vertex contrast.
SESSIONS = (2, 3)


def get_model(model_label, paradigm):
    if model_label not in SESSION_VARYING:
        raise NotImplementedError(f'Model label {model_label} has not been implemented')
    regressors = {p: '0 + C(session)' for p in SESSION_VARYING[model_label]}
    if not regressors:
        return RegressionGaussianPRF(paradigm=paradigm)
    return RegressionGaussianPRF(paradigm=paradigm, regressors=regressors)


def get_grid(model_label, n_sessions=2):
    """Grid in braincoder's parameter order: mu, sd, amplitude, baseline, with one entry
    per session for whichever of them is session-varying.

    mu and sd are coarsened 5x when they are session-varying, exactly as the original
    hard-coded m2 grid did -- otherwise the grid is 50^(2*n_sessions) and unusable.
    Verified to reproduce the previous hard-coded tuples for every label at n_sessions=2.
    """
    mus = np.log(np.linspace(5, 80, 50, dtype=np.float32))
    sds = np.log(np.linspace(2, 30, 50, dtype=np.float32))
    amplitudes = np.array([1.], dtype=np.float32)
    baselines = np.array([0], dtype=np.float32)

    varying = SESSION_VARYING[model_label]
    grid = []
    for name, values in [('mu', mus), ('sd', sds), ('amplitude', amplitudes),
                         ('baseline', baselines)]:
        if name in varying:
            v = values[::5] if name in ('mu', 'sd') else values
            grid.extend([v] * n_sessions)
        else:
            grid.append(values)
    return tuple(grid)


def get_fixed_pars(model_label, sessions):
    """Stage 1 pins mu and sd, under whichever names they have in this model."""
    varying = SESSION_VARYING[model_label]
    fixed = []
    for name in ('mu', 'sd'):
        if name in varying:
            fixed += [(f'{name}_unbounded', f'C(session)[{float(s)}]') for s in sessions]
        else:
            fixed.append((f'{name}_unbounded', 'Intercept'))
    return fixed


def main(subject, model_label=1, bids_folder='/data/ds-tmsrisk', natural_space=False,
         out_suffix=''):
    """Fit on ALL of sessions 2+3 -- no folds held out -- to get the best-fitting
    parameters for downstream use. `out_suffix` keeps a run out of the legacy tree:
    with `.refit2026` the output lands in
    `encoding_model2.refit2026.model-N.smoothed/`, leaving
    `encoding_model2.model-{0,1,2}.smoothed/` (the fits every published analysis reads)
    untouched."""

    bids_folder = Path(bids_folder)

    target_dir = (bids_folder / 'derivatives'
                  / f'encoding_model2{out_suffix}.model-{model_label}.smoothed'
                  / f'sub-{subject}')

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
    

    fixed_pars = get_fixed_pars(model_label, SESSIONS)

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
    parser.add_argument('--out_suffix', default='',
                        help="e.g. '.refit2026' to write outside the legacy tree")
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--smoothed', action='store_true')
    args = parser.parse_args()

    main(args.subject, model_label=args.model_label, bids_folder=args.bids_folder,
         out_suffix=args.out_suffix)
