from tms_risk.utils import Subject
import argparse
import os.path as op
import os

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter
import numpy as np

from braincoder.utils import get_rsq

stimulus_range = np.arange(7, 28*4)


def main(subject, session, smoothed, pca_confounds, denoise, n_voxels=1000, bids_folder='/data',
        retroicor=False,
        natural_space=False,
        spherical=False,
        roi='wang15_ips'):

    target_dir = op.join(bids_folder, 'derivatives', 'fisher_information')

    if denoise:
        target_dir += '.denoise'

    if (retroicor) and (not denoise):
        raise Exception("When not using GLMSingle RETROICOR is *always* used!")

    if retroicor:
        target_dir += '.retroicor'

    if smoothed:
        target_dir += '.smoothed'

    if pca_confounds:
        target_dir += '.pca_confounds'

    if spherical:
        # Diagonal noise: skip the cross-voxel correlation term in
        # ResidualFitter so omega = diag(τ²). Empirically the full Ω
        # over-estimates covariance and the decoder collapses toward the
        # stimulus-range mean, washing out individual RF tuning.
        target_dir += '.spherical'

    target_dir = op.join(target_dir, f'sub-{subject}', f'ses-{session}', 'func')
    print(denoise, target_dir)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    sub = Subject(subject, bids_folder)

    pars = sub.get_prf_parameters(model_label=1, session=session, roi=roi)
    data = sub.get_single_trial_volume(session, roi, smoothed=smoothed, retroicor=False, denoise=True)
    paradigm = sub.get_behavior(sessions=session, drop_no_responses=False)
    paradigm = paradigm.droplevel(['subject', 'session'])

    if n_voxels == 0:
        if session == 1:
            raise Exception("Session 1 is used for voxel selection!")

        session1_pars = sub.get_prf_parameters(model_label=1, session=1, roi=roi)
        r2_mask = session1_pars['cvr2'] > 0.0
        print(f"Using session 1 to select voxels. Mask {r2_mask.sum()} voxels big")
        r2_mask = r2_mask[r2_mask].index

    elif n_voxels == 1:
        r2_mask = pars['cvr2'] > 0.0
        print(f"Using current sessions to select voxels. Mask {r2_mask.sum()} voxels big")
        r2_mask = r2_mask[r2_mask].index

    else:
        r2_mask = pars['r2'].sort_values(ascending=False).index[:n_voxels]
        print(f"Using {len(r2_mask)} best voxels")

    data = data.loc[:, r2_mask]
    pars = pars.loc[r2_mask]

    if not natural_space:
        raise NotImplementedError("Only natural space is implemented")

    # Same index + NaN-voxel fixes applied to monte_carlo_decode.py — the
    # keras-backend braincoder's ResidualFitter re-runs predict() internally
    # and aligns on pandas index, so paradigm + data must share an index.
    # And LogGaussianPRF.predict returns NaN for voxels with strongly
    # negative `mu` (preferred numerosity < 1) which were silently producing
    # garbage in the old branch — drop them so the residual fit isn't
    # poisoned.
    import pandas as pd
    n1_paradigm = pd.Series(
        paradigm['n1'].values.astype(np.float32),
        index=pd.RangeIndex(len(paradigm), name='trial'),
        name='n1',
    )
    data = data.copy()
    data.index = n1_paradigm.index

    model = LogGaussianPRF(parameters=pars, paradigm=n1_paradigm)
    predictions = model.predict()
    finite_voxels = predictions.columns[~predictions.isna().any(axis=0)]
    n_dropped = predictions.shape[1] - len(finite_voxels)
    if n_dropped > 0:
        print(f'[fisher] dropping {n_dropped}/{predictions.shape[1]} voxels with NaN predictions '
              f'(likely negative-mu PRFs)')
        pars = pars.loc[finite_voxels]
        data = data.loc[:, finite_voxels]
        model = LogGaussianPRF(parameters=pars, paradigm=n1_paradigm)

    model.init_pseudoWWT(stimulus_range=stimulus_range, parameters=pars)

    residfit = ResidualFitter(model, data, n1_paradigm)

    omega, dof = residfit.fit(init_sigma2=1.0,
            init_dof=10.0,
            method='t',
            learning_rate=0.005,
            max_n_iterations=20000,
            spherical=spherical)
    # ResidualFitter returns a TF EagerTensor — convert to numpy.
    omega = np.asarray(omega, dtype=np.float32)
    dof = float(dof)


    fi = model.get_fisher_information(stimulus_range.astype(np.float32), omega, dof)
    fi.to_csv(op.join(target_dir, f'sub-{subject}_ses-{session}_roi-{roi}_nvoxels-{n_voxels}_fisher_information.tsv'), sep='\t')


if __name__ == '__main__':
    print('ues')
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None, type=int)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--pca_confounds', action='store_true')
    parser.add_argument('--retroicor', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--mask', default='wang15_ips')
    parser.add_argument('--natural_space', action='store_true')
    parser.add_argument('--spherical', action='store_true',
                        help='Use diagonal noise covariance (per-voxel τ, no ρ).')
    parser.add_argument('--n_voxels', default=100, type=int)
    args = parser.parse_args()

    main(subject=args.subject, session=args.session, smoothed=args.smoothed, pca_confounds=args.pca_confounds, denoise=args.denoise,
            n_voxels=args.n_voxels,
            natural_space=args.natural_space,
            spherical=args.spherical,
            bids_folder=args.bids_folder, roi=args.mask)
