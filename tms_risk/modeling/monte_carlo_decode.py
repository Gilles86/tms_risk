"""Monte Carlo simulate-and-decode for predicted decoding accuracy.

Companion to ``fisher_information.py``. Where Fisher info is the local
curvature of the log-likelihood at the true stimulus (analytical, fast,
but assumes Gaussian-like noise), this script samples from the fitted
nPRF + residual covariance, inverts each simulated trial with
``model.get_stimulus_pdf``, and aggregates decoded-vs-true error across
repeats. More robust than Fisher when PRFs are wide / overlapping or
when the residual is heavy-tailed (Student-t with finite dof).

Output: a TSV per (subject, session, ROI, n_voxels) with one row per
``(true_stim, repeat)`` carrying decoded mean and posterior SD. Group
plots downstream aggregate this into the standard "predicted decoding
accuracy" curve.

Mirrors the CLI of ``fisher_information.py`` so the SLURM wrapper can
be a near-copy.
"""
import argparse
import os
import os.path as op

import numpy as np
import pandas as pd

from braincoder.models import LogGaussianPRF
from braincoder.optimize import ResidualFitter
from braincoder.utils.math import get_expected_value, get_sd_posterior

from tms_risk.utils import Subject


# Grid over which the posterior is evaluated. Same as fisher_information.py
# (7..112 covers the experimental range with a margin).
STIMULUS_RANGE = np.arange(7, 28 * 4)


def _sample_paradigm(n_trials, rng, stimulus_pool=None):
    """Draw `n_trials` stimuli to forward-simulate.

    ``stimulus_pool`` is the discrete set of experimentally-shown
    numerosities. If None, uniformly sample integers across STIMULUS_RANGE.
    """
    if stimulus_pool is None:
        stimulus_pool = STIMULUS_RANGE
    return pd.DataFrame(
        rng.choice(stimulus_pool, size=n_trials),
        columns=['n1'],
    )


def main(subject, session, smoothed=False, denoise=True, n_voxels=100,
         bids_folder='/data', roi='wang15_ips',
         n_repeats=1000, seed=0, natural_space=True, spherical=False):

    target_dir = op.join(bids_folder, 'derivatives', 'monte_carlo_decode')
    if denoise:
        target_dir += '.denoise'
    if smoothed:
        target_dir += '.smoothed'
    if spherical:
        target_dir += '.spherical'
    target_dir = op.join(target_dir, f'sub-{subject}', f'ses-{session}', 'func')
    os.makedirs(target_dir, exist_ok=True)

    sub = Subject(subject, bids_folder)

    # Same parameter / data loading as fisher_information.py — keep the two
    # in lockstep so the predicted-decoding figure compares apples to apples.
    pars = sub.get_prf_parameters(model_label=1, session=session, roi=roi)
    data = sub.get_single_trial_volume(
        session, roi, smoothed=smoothed, retroicor=False, denoise=denoise,
    )
    paradigm_obs = sub.get_behavior(sessions=session, drop_no_responses=False)
    paradigm_obs = paradigm_obs.droplevel(['subject', 'session'])

    # Voxel selection: top n_voxels by in-session R² (matches fisher_info).
    if n_voxels == 0:
        if session == 1:
            raise Exception("Session 1 is used for voxel selection!")
        session1_pars = sub.get_prf_parameters(model_label=1, session=1, roi=roi)
        mask_idx = session1_pars.index[session1_pars['cvr2'] > 0.0]
    elif n_voxels == 1:
        mask_idx = pars.index[pars['cvr2'] > 0.0]
    else:
        mask_idx = pars['r2'].sort_values(ascending=False).index[:n_voxels]

    data = data.loc[:, mask_idx]
    pars = pars.loc[mask_idx]

    if not natural_space:
        raise NotImplementedError("Only natural space is implemented")

    # Build the nPRF + fit the residual covariance.
    model = LogGaussianPRF(parameters=pars,
                           paradigm=paradigm_obs['n1'].astype(np.float32))
    predictions = model.predict()
    data.index = predictions.index
    model.init_pseudoWWT(stimulus_range=STIMULUS_RANGE, parameters=pars)

    omega, dof = ResidualFitter(
        model, data, paradigm_obs['n1'].astype(np.float32),
    ).fit(init_sigma2=1.0, init_dof=10.0, method='t',
          learning_rate=0.005, max_n_iterations=20000,
          spherical=spherical)

    # Forward-simulate `n_repeats` × len(STIMULUS_RANGE) trials. Stratified:
    # one repeat = full coverage of the stimulus grid → predicted decoding
    # bias / SD as functions of true magnitude.
    rng = np.random.default_rng(seed)
    paradigm_sim = pd.DataFrame(
        np.tile(STIMULUS_RANGE, n_repeats).astype(np.float32),
        columns=['n1'],
    )
    paradigm_sim.index = pd.MultiIndex.from_product(
        [np.arange(n_repeats), STIMULUS_RANGE],
        names=['repeat', 'true_stim'],
    )

    # `model.simulate` adds noise drawn from omega (with `dof` → MVT,
    # else MVN). One shot does all reps × all stimuli.
    sim_data = model.simulate(
        paradigm=paradigm_sim['n1'], noise=omega, dof=dof, n_repeats=1,
    )
    sim_data.index = paradigm_sim.index

    # Invert: per simulated trial, evaluate the posterior over the
    # stimulus grid in one vectorized call.
    pdf = model.get_stimulus_pdf(
        sim_data, stimulus_range=STIMULUS_RANGE.astype(np.float32),
        omega=omega, dof=dof,
    )

    decoded_mean = get_expected_value(pdf)
    decoded_sd = get_sd_posterior(pdf)

    out = pd.DataFrame({
        'decoded_mean': decoded_mean.values,
        'decoded_sd': decoded_sd.values,
    }, index=paradigm_sim.index)
    out['true_stim'] = out.index.get_level_values('true_stim')
    out['bias'] = out['decoded_mean'] - out['true_stim']

    out_path = op.join(
        target_dir,
        f'sub-{subject}_ses-{session}_roi-{roi}_nvoxels-{n_voxels}_mc_decode.tsv',
    )
    out.to_csv(out_path, sep='\t')
    print(f'Wrote {out_path}  ({len(out):,} rows; n_repeats={n_repeats})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('subject', default=None)
    parser.add_argument('session', default=None, type=int)
    parser.add_argument('--bids_folder', default='/data')
    parser.add_argument('--smoothed', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--mask', default='wang15_ips')
    parser.add_argument('--natural_space', action='store_true')
    parser.add_argument('--spherical', action='store_true',
                        help='Use diagonal noise covariance (per-voxel τ, no ρ).')
    parser.add_argument('--n_voxels', default=100, type=int)
    parser.add_argument('--n_repeats', default=1000, type=int)
    parser.add_argument('--seed', default=0, type=int)
    args = parser.parse_args()

    main(
        subject=args.subject, session=args.session,
        smoothed=args.smoothed, denoise=args.denoise,
        n_voxels=args.n_voxels, bids_folder=args.bids_folder,
        roi=args.mask, n_repeats=args.n_repeats, seed=args.seed,
        natural_space=args.natural_space,
        spherical=args.spherical,
    )
