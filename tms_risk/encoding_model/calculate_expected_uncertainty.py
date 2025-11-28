from argparse import ArgumentParser
from tms_risk.utils.data import Subject
from pathlib import Path
import pandas as pd
import numpy as np
from nilearn.input_data import NiftiMasker
from tms_risk.encoding_model.fit_regression_nprf import get_model
from braincoder.optimize import ResidualFitter
from braincoder.utils import get_rsq
from braincoder.utils.math import get_expected_value


def main(subject, roi='NPCr2cm-cluster', bids_folder='/data/ds-tmsrisk'):

    sub = Subject(subject)

    bids_folder = Path(bids_folder)

    target_dir = bids_folder / 'derivatives' / 'expected_uncertainty' / f'sub-{subject:02d}' / 'func'
    target_dir.mkdir(parents=True, exist_ok=True)

    pars = sub.get_prf_parameters2(model_label=1, roi=roi)

    paradigm = sub.get_paradigm()
    paradigm = paradigm.reset_index('session')[['log(n1)', 'session']].rename(columns={'log(n1)': 'x'}).astype(np.float32).droplevel(1)

    # Get prf parameters
    print('Getting prf parameters...')
    model = get_model(1, paradigm)

    raw_pars2 = model._base_transform_parameters_backward(pars[['mu', 'sd', 'amplitude', 'baseline']].xs(2, level=1, axis=1))
    raw_pars3 = model._base_transform_parameters_backward(pars[['mu', 'sd', 'amplitude', 'baseline']].xs(3, level=1, axis=1))

    raw_pars = np.stack([raw_pars2[:, 0], raw_pars2[:,1], raw_pars2[:, 2], raw_pars3[:, 2], raw_pars2[:, 3]], axis=1)
    raw_pars = pd.DataFrame(raw_pars, columns=model.parameter_labels)

    cvr2 = pars[('cvr2', None)]
    cvr2_mask = (cvr2 > 0.0) & (np.isfinite(raw_pars).all(axis=1))
    
    # Get mask
    mask = bids_folder / 'derivatives' / 'fmriprep' / f'sub-{subject}' / f'ses-1' / 'func' / f'sub-{subject}_ses-1_task-task_run-1_space-T1w_desc-brain_mask.nii.gz'
    mask = sub.get_volume_mask(session=1, roi=roi, epi_space=True)
    masker = NiftiMasker(mask_img=mask)

    # Get single-trial data
    data2 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject:02d}' / f'ses-2' / 'func' / f'sub-{subject:02d}_ses-2_task-task_space-T1w_desc-stims1_pe.nii.gz'
    data3 = bids_folder / 'derivatives' / 'glm_stim1.denoise.smoothed' / f'sub-{subject:02d}' / f'ses-3' / 'func' / f'sub-{subject:02d}_ses-3_task-task_space-T1w_desc-stims1_pe.nii.gz'

    data = pd.DataFrame(np.vstack([masker.fit_transform(data2), masker.fit_transform(data3)]), index=paradigm.index)
    data = pd.DataFrame(data, index=paradigm.index).astype(np.float32)

    stimulus_range = np.linspace(paradigm['x'].min(), paradigm['x'].max(), 100).astype(np.float32)
    model.init_pseudoWWT(stimulus_range, raw_pars)

    pred = model.predict(paradigm, raw_pars)
    r2 = get_rsq(data, pred)
    print(r2)
    assert(r2[cvr2_mask].min() > 0.0), "Some voxels have negative R², cannot proceed"

    print('Masking data.')
    data = data.loc[:, cvr2_mask]
    raw_pars = raw_pars.loc[cvr2_mask, :]

    resid_fitter = ResidualFitter(model, data, paradigm, raw_pars)
    omega, dof = resid_fitter.fit(method='gauss', spherical=True)

    print('Simulating data...')
    x = np.log(np.arange(5, 28*4))
    sessions = [2.0, 3.0]
    fake_paradigm = pd.DataFrame({
        'x': np.tile(x, len(sessions)),
        'session': np.repeat(sessions, len(x))
    }).astype(np.float32)

    model = get_model(1, fake_paradigm)
    simulated_data = model.simulate(paradigm=fake_paradigm, parameters=raw_pars, noise=omega, dof=dof, n_repeats=1000)

    # Calculating pdf
    print('Calculating pdf...')
    pdf = model.get_stimulus_pdf(simulated_data, stimulus_range=fake_paradigm, parameters=raw_pars, omega=omega, dof=dof, normalize=True)
    pdf = pdf.unstack('repeat')
    pdf.index = pd.MultiIndex.from_frame(fake_paradigm)
    print(pdf)

    pdf = pdf.apply(lambda d: d.xs(d.name[1], level='session'), axis=1).stack('repeat')
    pdf.columns = np.exp(pdf.columns)

    # x-level of index should take the exponential
    pdf = pdf.reset_index()
    print(pdf)
    pdf['x'] = np.exp(pdf['x'])
    pdf = pdf.set_index(['x', 'session', 'repeat'])
    print(pdf)

    pars = pd.DataFrame(index=pdf.index)
    pars['E'] = get_expected_value(pdf, normalize=True)
    pars['error'] = pars['E'] - pars.index.get_level_values('x')
    pars['abs(error)'] = pars['error'].abs()

    mean_E = pars.groupby(['session', 'x'])['E'].mean().to_frame("mean_E")
    mean_error = pars.groupby(['session', 'x'])['error'].mean().to_frame("mean_error")
    var_E = pars.groupby(['session', 'x'])['E'].var().to_frame("var_E")
    abs_error = pars.groupby(['session', 'x'])['abs(error)'].mean().to_frame("mean_abs_error")

    decode_pars = mean_E.join(mean_error).join(var_E).join(abs_error)

    decode_pars.to_csv(target_dir / f'sub-{subject:02d}_roi-{roi}_desc-expected_error.tsv', sep='\t')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('subject', type=int, help='Subject identifier')
    parser.add_argument('--bids_folder', type=str, default='/data/ds-tmsrisk', help='Path to BIDS folder')
    args = parser.parse_args()

    main(args.subject, bids_folder=args.bids_folder)