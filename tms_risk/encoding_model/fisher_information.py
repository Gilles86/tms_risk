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

    target_dir = op.join(target_dir, f'sub-{subject}', f'ses-{session}', 'func')
    print(denoise, target_dir)

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    sub = Subject(subject, bids_folder)

    pars = sub.get_prf_parameters_volume(session, smoothed=smoothed, retroicor=False, denoise=True, cross_validated=False, natural_space=True, roi=roi)
    data = sub.get_single_trial_volume(session, roi, smoothed=smoothed, retroicor=False, denoise=True)
    paradigm = sub.get_behavior(sessions=session, drop_no_responses=False)
    paradigm = paradigm.droplevel(['subject', 'session'])

    if n_voxels == 0:
        if session == 1:
            raise Exception("Session 1 is used for voxel selection!")

        session1_pars = sub.get_prf_parameters_volume(1, run=None, smoothed=smoothed, pca_confounds=pca_confounds, denoise=denoise, retroicor=retroicor, cross_validated=False, natural_space=natural_space, roi=roi)
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

    model = LogGaussianPRF(parameters=pars, paradigm=paradigm['n1'].astype(np.float32))
    predictions = model.predict()

    data.index = predictions.index

    model.init_pseudoWWT(stimulus_range=stimulus_range, parameters=pars)

    residfit = ResidualFitter(model, data,
                                paradigm['n1'].astype(np.float32))

    omega, dof = residfit.fit(init_sigma2=1.0,
            init_dof=10.0,
            method='t',
            learning_rate=0.005,
            max_n_iterations=20000)


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
    parser.add_argument('--n_voxels', default=100, type=int)
    args = parser.parse_args()

    main(subject=args.subject, session=args.session, smoothed=args.smoothed, pca_confounds=args.pca_confounds, denoise=args.denoise,
            n_voxels=args.n_voxels,
            natural_space=args.natural_space,
            bids_folder=args.bids_folder, roi=args.mask)
