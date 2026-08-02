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

from tms_risk.utils import Subject


# Grid over which the posterior is evaluated. Same as fisher_information.py
# (7..112 covers the experimental range with a margin).
STIMULUS_RANGE = np.arange(7, 28 * 4)
# Log-spaced (geometric) alternative. braincoder treats the stimulus grid as a
# discrete support with uniform weight per grid point, so a geometric grid =
# a *flat prior in log space* (the objective/Weber prior for numerosity). This
# (a) moves the central-tendency collapse target from the arithmetic mean (~59)
# to the geometric mean (~28), and (b) concentrates grid resolution at small
# numerosities where the cTBS effect lives.
STIMULUS_RANGE_LOG = np.geomspace(7, 28 * 4 - 1, num=len(STIMULUS_RANGE))


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
         n_repeats=1000, seed=0, natural_space=True, spherical=False,
         model_label=1, selection=None, log_prior=False):

    stim_grid = STIMULUS_RANGE_LOG if log_prior else STIMULUS_RANGE

    target_dir = op.join(bids_folder, 'derivatives', 'monte_carlo_decode')
    if denoise:
        target_dir += '.denoise'
    if smoothed:
        target_dir += '.smoothed'
    if spherical:
        target_dir += '.spherical'
    if log_prior:
        target_dir += '.logprior'
    if model_label != 1:
        target_dir += f'.model{model_label}'
    target_dir = op.join(target_dir, f'sub-{subject}', f'ses-{session}', 'func')
    os.makedirs(target_dir, exist_ok=True)

    sub = Subject(subject, bids_folder)

    # Same parameter / data loading as fisher_information.py — keep the two
    # in lockstep so the predicted-decoding figure compares apples to apples.
    pars = sub.get_prf_parameters(model_label=model_label, session=session, roi=roi)
    data = sub.get_single_trial_volume(
        session, roi, smoothed=smoothed, retroicor=False, denoise=denoise,
    )
    paradigm_obs = sub.get_behavior(sessions=session, drop_no_responses=False)
    paradigm_obs = paradigm_obs.droplevel(['subject', 'session'])

    # Voxel selection. `selection` (if given) overrides the n_voxels logic;
    # otherwise n_voxels selects: 0 -> session-1 cvR²>0, 1 -> in-session
    # cvR²>0, >=2 -> top-n by in-session R². The filename token `sel_label`
    # records which rule was used so downstream plots can distinguish runs.
    sel_label = str(n_voxels)
    if selection == 'mixture':
        # Paper used a hard cvR²>0 cut within the targeted ROI; this is the
        # principled per-subject version: fit a 2-component signal/noise
        # mixture to session-1 cvR² and keep voxels with P(signal) >= 0.5.
        if session == 1:
            raise Exception("Session 1 is used for voxel selection!")
        from braincoder.utils.stats import fit_r2_mixture, r2_p_signal_threshold
        session1_pars = sub.get_prf_parameters(model_label=model_label, session=1, roi=roi)
        cvr2_s1 = session1_pars['cvr2']
        fit = fit_r2_mixture(cvr2_s1.values)
        thr = r2_p_signal_threshold(fit, p=0.5)
        mask_idx = session1_pars.index[cvr2_s1 >= thr]
        sel_label = 'mixture'
        print(f'[mc_decode] R² mixture (ses-1): signal_mean_r2={fit["signal_mean_r2"]:.3f} '
              f'noise_mean_r2={fit["noise_mean_r2"]:.3f} signal_weight={fit["signal_weight"]:.2f} '
              f'p>=0.5 thr={thr:.4f} -> {len(mask_idx)} voxels')
    elif n_voxels == 0:
        if session == 1:
            raise Exception("Session 1 is used for voxel selection!")
        session1_pars = sub.get_prf_parameters(model_label=model_label, session=1, roi=roi)
        mask_idx = session1_pars.index[session1_pars['cvr2'] > 0.0]
        sel_label = 'ses1cvr2'
    elif n_voxels == 1:
        mask_idx = pars.index[pars['cvr2'] > 0.0]
    else:
        mask_idx = pars['r2'].sort_values(ascending=False).index[:n_voxels]

    data = data.loc[:, mask_idx]
    pars = pars.loc[mask_idx]

    if not natural_space:
        raise NotImplementedError("Only natural space is implemented")

    # Build the nPRF + fit the residual covariance.
    # The new braincoder's ResidualFitter re-runs predict() internally to get
    # residuals and then aligns on pandas index. Force both `data` and
    # `paradigm` to a simple RangeIndex so the predict-output index always
    # matches and the alignment can't drop rows.
    n1_paradigm = pd.Series(
        paradigm_obs['n1'].values.astype(np.float32),
        index=pd.RangeIndex(len(paradigm_obs), name='trial'),
        name='n1',
    )
    data = data.copy()
    data.index = n1_paradigm.index
    model = LogGaussianPRF(parameters=pars, paradigm=n1_paradigm)

    # Drop voxels whose PRF predictions blow up. In keras-backend braincoder,
    # LogGaussianPRF returns NaN when `mu` is very negative (preferred
    # numerosity below 1) — voxels that were never going to contribute anyway
    # (cvr2 near zero). The older braincoder silently produced garbage; the
    # new one propagates NaN, which then poisons the residual covariance fit.
    preds = model.predict()
    finite_voxels = preds.columns[~preds.isna().any(axis=0)]
    n_dropped = preds.shape[1] - len(finite_voxels)
    if n_dropped > 0:
        print(f'[mc_decode] dropping {n_dropped}/{preds.shape[1]} voxels with NaN predictions '
              f'(likely negative-mu PRFs)')
        pars = pars.loc[finite_voxels]
        data = data.loc[:, finite_voxels]
        model = LogGaussianPRF(parameters=pars, paradigm=n1_paradigm)

    model.init_pseudoWWT(stimulus_range=stim_grid, parameters=pars)

    omega, dof = ResidualFitter(
        model, data, n1_paradigm,
    ).fit(init_sigma2=1.0, init_dof=10.0, method='t',
          learning_rate=0.005, max_n_iterations=20000,
          spherical=spherical)
    # ResidualFitter returns a TF EagerTensor in the new branch — convert to
    # numpy so `model.simulate` inside get_expected_uncertainty can call
    # `.astype(np.float32)` on it without a TypeError.
    omega = np.asarray(omega, dtype=np.float32)
    dof = float(dof)

    # Simulate `n_repeats` noisy responses per stimulus and decode each
    # via `model.get_expected_uncertainty` (canonical braincoder API in the
    # keras-backend branch). Returns one row per true stimulus with
    # mean_E (posterior mean averaged across reps), var_E (empirical
    # variance of the posterior mean across reps), mean_error, etc.
    out = model.get_expected_uncertainty(
        stimuli=stim_grid.astype(np.float32),
        omega=omega, dof=dof,
        n_simulations=n_repeats,
        progress=True,
    )

    out_path = op.join(
        target_dir,
        f'sub-{subject}_ses-{session}_roi-{roi}_nvoxels-{sel_label}_mc_decode.tsv',
    )
    out.to_csv(out_path, sep='\t')
    print(f'Wrote {out_path}  ({len(out):,} rows; n_simulations={n_repeats})')


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
    parser.add_argument('--model_label', type=int, default=1,
                        help='PRF model variant 0/1/2 (default 1, paper Fig 2).')
    parser.add_argument('--n_voxels', default=100, type=int)
    parser.add_argument('--selection', default=None, choices=[None, 'mixture'],
                        help='Override n_voxels: "mixture" = R²-mixture P(signal)>=0.5 '
                             'on session-1 cvR².')
    parser.add_argument('--log_prior', action='store_true',
                        help='Use a geometric (log-spaced) stimulus grid = flat '
                             'prior in log space (objective/Weber prior).')
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
        model_label=args.model_label,
        selection=args.selection,
        log_prior=args.log_prior,
    )
