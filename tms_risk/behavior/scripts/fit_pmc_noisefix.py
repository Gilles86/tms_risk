"""Refit the Flexible PMC with the *intended* noise composition.

Fully separate from the published fits: separate bauer checkout, separate model
labels, separate derivatives directory. Nothing here reads or writes anything
under `derivatives/cogmodels/`, and `fit_model.py` is untouched.

Background
----------
The published `flexible2.6` fits were produced with bauer@ecc6454, whose
`_get_trialwise_evidence_sd` built the first option's noise as

    n1_evidence_sd = softplus( sum(memory_coefs * basis_memory)
                             + sum(memory_coefs * basis_perceptual) )   # <- bug

i.e. the memory spline coefficients were used for both terms, so the perceptual
noise function never entered the first-presented option. The Methods describe
nu_1 = nu_perceptual + nu_memory, so the intended line is

    n1_evidence_sd = softplus( sum(memory_coefs * basis_memory)
                             + sum(perceptual_coefs * basis_perceptual) )

Upstream fixed this in b66c806 (2026-04-03) -- but that same commit also changed
the 'payoff' branch of `_get_choice_predictions` (the diff_sd formula), so simply
checking out current bauer changes two things at once. To isolate the noise
composition we apply a one-line patch to ecc6454 instead.

Variants
--------
`--variant noisefix`  ecc6454 + notes/patches/bauer-ecc6454-noisefix.patch
                      Exactly one difference from the published model.
`--variant head`      current libs/bauer HEAD (noise fix *and* new diff_sd).
                      What future work will use; two differences.

Usage
-----
    python -m tms_risk.behavior.scripts.fit_pmc_noisefix flexible2.6_noisefix
    python -m tms_risk.behavior.scripts.fit_pmc_noisefix flexible2.6_noisefix_null

Labels: `flexible2.6_noisefix[_null|_memory|_perception]` -- TMS on both noise
terms, neither, memory only, or perceptual only. Traces land in
`<bids>/derivatives/cogmodels.noisefix/model-<label>_trace.netcdf`.

`--memory_composition additive` (needs `--variant head`) additionally constrains
the memory contribution to be non-negative; the composition used is stamped into
`trace.posterior.attrs['tms_risk_memory_composition']`.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / 'notes' / 'patches' / 'bauer-ecc6454-noisefix.patch'
BASE_COMMIT = 'ecc6454'

# family 2 = shared_perceptual_noise (memory/perceptual); family 1 = independent
# (first/second option). They are exact reparameterisations of one another
# (c1 = memory + perceptual, c2 = perceptual), so the likelihoods are identical;
# only the prior coordinates and the sampling geometry differ. Family 1 is the
# better-conditioned one -- family 2's coordinates make the posterior bimodal.
# The `_prior*` suffixes put the cTBS regressor on the magnitude PRIOR instead of (or
# as well as) the noise. They are the model-level version of the paper's central
# dichotomy: did parietal cTBS degrade how precisely payoffs are represented, or did it
# shift the observer's prior -- i.e. their preferences? Every other row of Table 1 only
# asks *where in the noise* the effect sits, so without these the "noise, not
# preference" claim rests on the model-free analyses alone.
SUFFIX_REGRESSORS = {
    2: {'': ['memory_noise_sd', 'perceptual_noise_sd'], '_null': [],
        '_memory': ['memory_noise_sd'], '_perception': ['perceptual_noise_sd'],
        '_prior': ['risky_prior_mu', 'safe_prior_mu'],
        '_priorsd': ['risky_prior_sd', 'safe_prior_sd'],
        '_perception_prior': ['perceptual_noise_sd',
                              'risky_prior_mu', 'safe_prior_mu']},
    1: {'': ['n1_evidence_sd', 'n2_evidence_sd'], '_null': [],
        '_first': ['n1_evidence_sd'], '_second': ['n2_evidence_sd'],
        '_prior': ['risky_prior_mu', 'safe_prior_mu']},
}


def prepare_bauer(variant, worktree):
    """Return a path to the bauer checkout to use, creating it if needed."""
    if variant == 'head':
        return REPO / 'libs' / 'bauer'

    worktree = Path(worktree)
    if not (worktree / 'bauer' / 'models.py').exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        bauer_repo = REPO / 'libs' / 'bauer'
        subprocess.run(['git', '-C', str(bauer_repo), 'worktree', 'prune'], check=True)
        subprocess.run(['git', '-C', str(bauer_repo), 'worktree', 'add', '--detach',
                        str(worktree), BASE_COMMIT], check=True)
        subprocess.run(['git', '-C', str(worktree), 'apply', str(PATCH)], check=True)
        print(f'created patched bauer worktree at {worktree}')
    # verify the patch is in place -- a silent revert here would refit the bug
    src = (worktree / 'bauer' / 'models.py').read_text()
    assert 'spline_pars2 = pt.stack([parameters[l2] for l2 in labels2], axis=1)' in src, \
        f'{worktree} does not carry the noise-composition fix'
    return worktree


def bauer_commit(path):
    """Record exactly which bauer produced this fit -- these models are
    version-sensitive and the label alone does not pin the code."""
    try:
        out = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(path), 'status', '--porcelain'],
                               capture_output=True, text=True).stdout.strip()
        return out + ('+patched' if dirty else '')
    except Exception:
        return 'unknown'


def build(df, regressor_names, spline_order=6, family=2, spline_degree=3,
          noise='flexible', prior_estimate='full',
          memory_composition='sum_then_softplus'):
    """Build the noise model.

    `noise='flexible'`  B-spline noise function over magnitude in natural space.
    `noise='weber'`     the original PMC: a single noise sd per term, applied in
                        log space, i.e. scalar invariance / Weber's law. This is
                        the paper's Table-1 baseline family (`11a`-`11c`,
                        `11_null` in fit_model.py) and takes no spline arguments.

    `memory_composition='additive'` constrains the memory contribution to the
    first-presented option's noise to be non-negative (nu_1 = nu_2 +
    softplus(eta_memory)). The default keeps bauer's historical composition,
    softplus(eta_memory + eta_perceptual), under which nu_1 < nu_2 wherever
    eta_memory < 0 -- which it is below ~12 CHF in these fits.
    """
    import inspect
    import bauer.models as bm
    kw = dict(regressors={n: 'stimulation_condition' for n in regressor_names},
              memory_model='shared_perceptual_noise' if family == 2 else 'independent',
              prior_estimate=prior_estimate)
    if memory_composition != 'sum_then_softplus':
        # Only family 2 decomposes noise into memory + perceptual terms.
        if family != 2:
            raise SystemExit('--memory_composition only applies to family 2 '
                             '(shared_perceptual_noise); family 1 has no memory term')
        if noise == 'weber':
            raise SystemExit('--memory_composition is flexible-noise only; '
                             'the weber front-end has no spline noise functions')
        kw['memory_composition'] = memory_composition
    if prior_estimate == 'objective':
        # bauer's FlexibleNoiseRiskModel raises NotImplementedError for 'objective',
        # so the prior is instead PINNED numerically in `pin_objective_prior` below:
        # built as 'full', then given a near-degenerate hyperprior at the empirical
        # payoff mean/SD. Same effect, no bauer change.
        kw['prior_estimate'] = 'full'
    if noise == 'weber':
        return bm.RiskRegressionModel(df, **kw)
    cls = bm.FlexibleNoiseRiskRegressionModel
    signature = inspect.signature(cls).parameters
    key = 'spline_order' if 'spline_order' in signature else 'polynomial_order'
    kw[key] = spline_order
    if 'spline_degree' in signature:
        kw['spline_degree'] = spline_degree
    elif spline_degree != 3:
        raise SystemExit('this bauer has no spline_degree; use the patched checkout')
    if 'memory_composition' in kw and 'memory_composition' not in signature:
        # ecc6454 (`--variant noisefix`) predates the option entirely.
        raise SystemExit('this bauer has no memory_composition; use --variant head')
    return cls(df, **kw)


def pin_objective_prior(model, df, verbose=True, eps=0.01):
    """Fix the magnitude prior to the OBJECTIVE payoff distribution.

    The fitted priors sit far below the stimulus range (safe_prior_mu ~ 4 CHF against
    a 7-112 CHF range), compressing an objective 28 CHF into a perceived 9 CHF. That
    is hard to read as a belief about payoffs, and suggests the prior is standing in
    for a compressive value function -- which is how this architecture produces risk
    aversion at all.

    This variant removes that freedom: each prior parameter gets a near-degenerate
    hyperprior (sigma = eps) at the empirical mean/SD of the payoffs the participant
    actually saw, and the between-subject dispersion is squeezed to match, so the
    prior is fixed rather than merely regularised. The ELPD cost of the pin measures
    how much predictive work the compression was doing.
    """
    stats = {'safe': (float(df['n_safe'].mean()), float(df['n_safe'].std())),
             'risky': (float(df['n_risky'].mean()), float(df['n_risky'].std()))}
    pinned = []
    for key, info in model.free_parameters.items():
        for opt, (m, sd) in stats.items():
            if key == f'{opt}_prior_mu':
                target = m
            elif key == f'{opt}_prior_sd':
                # transform is softplus; invert it so the transformed value equals sd
                target = float(np.log(np.expm1(sd))) if sd < 30 else sd
            else:
                continue
            info['mu_intercept'], info['sigma_intercept'] = target, eps
            info['cauchy_sigma_intercept'] = eps      # squeeze between-subject spread
            pinned.append(f'{key}={target:.2f}')
    if verbose:
        print('pinned prior to the objective payoff distribution: ' + '; '.join(pinned))
    return model


def constrain_priors(model, df, verbose=True, prior_mu_sigma=10., prior_sd_mu=3.):
    """Replace bauer's very wide default priors with ones that respect the payoff scale.

    The defaults let the sampler reach degenerate regions: noise spline intercepts are
    Normal(5, 5) on the pre-softplus scale, i.e. nu is centred at ~5 CHF and can reach
    tens; and prior_sd is centred near the mean payoff, so sigma ~ 20 is a priori
    ordinary. At sigma = 20 the shrinkage weight sigma^2/(sigma^2 + nu^2) is ~0.99, the
    prior stops doing any work, and prior_mu becomes a flat direction that wanders to
    +/-190 CHF. Both observed failure modes live there.

    Constraints, all on the pre-transform (unconstrained Normal) scale:
      noise splines   Normal(0.5, 1.5)  -> nu ~ softplus(.) centred near 1 CHF,
                                           comfortably reaching ~4 CHF
      *_prior_mu      Normal(empirical mean payoff for that option type, 10)
      *_prior_sd      Normal(3, 1)      -> sigma ~ 3 CHF, matching the published fits
    """
    safe_mu = float(df['n_safe'].mean()) if 'n_safe' in df else 16.
    risky_mu = float(df['n_risky'].mean()) if 'n_risky' in df else 36.
    # With prior_estimate='objective' the prior is fixed to the empirical payoff
    # distribution and carries no free parameters, so only the noise splines remain
    # to constrain. The loop below is keyed on parameter names, so this is automatic.
    changed = []
    for key, info in model.free_parameters.items():
        if 'spline' in key and ('noise_sd' in key or 'evidence_sd' in key):
            info['mu_intercept'], info['sigma_intercept'] = 0.5, 1.5
        elif key.endswith('_prior_mu'):
            info['mu_intercept'] = risky_mu if key.startswith('risky') else safe_mu
            info['sigma_intercept'] = prior_mu_sigma
        elif key.endswith('_prior_sd'):
            info['mu_intercept'], info['sigma_intercept'] = prior_sd_mu, 1.
        else:
            continue
        changed.append(f"{key}~N({info['mu_intercept']:.1f},{info['sigma_intercept']:.1f})")
    if verbose:
        print(f'constrained {len(changed)} priors, e.g. ' + '; '.join(changed[:3]))
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label')
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--variant', default='noisefix', choices=['noisefix', 'head'])
    parser.add_argument('--spline_degree', default=3, type=int,
                        help='B-spline polynomial degree; 2 doubles the interior '
                             'knots at the same parameter count')
    parser.add_argument('--worktree',
                        default='/private/tmp/claude-1763273667/-Users-gdehol-git-tms-risk/'
                                '05755856-de7c-4368-9eca-1e14a3aa0161/scratchpad/bauer_noisefix')
    parser.add_argument('--tune', default=5000, type=int)
    parser.add_argument('--draws', default=5000, type=int)
    parser.add_argument('--chains', default=4, type=int)
    parser.add_argument('--cores', default=4, type=int)
    parser.add_argument('--target_accept', default=0.9, type=float)
    parser.add_argument('--prior_estimate', default='full',
                        choices=['full', 'shared', 'objective'],
                        help="'full' fits a (mu, sd) prior per option type; 'objective' "
                             'FIXES the prior to the empirical payoff distribution, '
                             'removing it as a free parameter. The latter is the '
                             'robustness check for whether the fitted prior is doing '
                             'the work of a compressive value function.')
    parser.add_argument('--memory_composition', default='sum_then_softplus',
                        choices=['sum_then_softplus', 'additive'],
                        help="how perceptual and memory noise combine into the "
                             "first-presented option's noise. 'sum_then_softplus' "
                             '(default, and what every published trace was fit '
                             'under) is softplus(eta_memory + eta_perceptual), '
                             'which lets the memory contribution go NEGATIVE '
                             'wherever eta_memory < 0. "additive" is nu_2 + '
                             'softplus(eta_memory), so holding an option in '
                             'memory can only add noise. Family 2 + flexible only.')
    parser.add_argument('--group_sd', default=None, choices=['hp'],
                        help="'hp' = HalfNormal group SDs with per-parameter "
                             "scales, identical to the lfx2 '-hp' token. Use it "
                             "to put a natural-space fit on the SAME prior as a "
                             "log-space one; without it the two families differ "
                             "in prior as well as in scale, and their credible "
                             "intervals are not comparable.")
    parser.add_argument('--constrain', action='store_true',
                        help='use payoff-scale priors instead of bauer defaults')
    parser.add_argument('--backend', default='pymc',
                        choices=['pymc', 'numpyro', 'blackjax'],
                        help='numpyro is the JAX/GPU path; init= is pymc-only so it is '
                             'dropped automatically when backend != pymc')
    parser.add_argument('--find_init', default=None,
                        help="bauer's starting-point finder, e.g. 'mapjitter' (MAP centre "
                             "+ prior-scaled jitter). Only ddm/race set it by default, so "
                             "the risky-choice family otherwise gets pymc's generic jitter "
                             "init -- which bauer/notes/ddm_convergence_lessons.md reports "
                             "converging in only 2/16 seeds.")
    parser.add_argument('--init', default='adapt_diag',
                        help="pymc init. 'adapt_diag' starts every chain from the "
                             "same point; pymc's default 'jitter+adapt_diag' "
                             "randomises starts and can scatter chains into "
                             "different basins on this model.")
    parser.add_argument('--out_dir', default=None,
                        help='override the output directory (for smoke tests)')
    parser.add_argument('--no_log_likelihood', action='store_true',
                        help='skip pm.compute_log_likelihood (much smaller file, no LOO)')
    args = parser.parse_args()

    label = args.model_label
    m = re.fullmatch(r'(flexible|weber)([12])(\.\d)?_noisefix(.*)', label)
    if not m:
        raise SystemExit('label must look like {flexible|weber}[1|2][.4|.6]_noisefix[suffix] '
                         'so these fits can never be confused with the published ones')
    noise = 'weber' if m.group(1) == 'weber' else 'flexible'
    family = int(m.group(2))
    # bare `flexible2` is 5 splines -- that is the paper's model (Methods: "we chose
    # to use 5 splines"), confirmed by comprehensive_model_comparison.ipynb.
    spline_order = 5 if m.group(3) is None else int(m.group(3)[1:])
    suffix = m.group(4)
    if noise == 'weber' and (m.group(3) or args.spline_degree != 3):
        raise SystemExit('weber models have no spline basis; drop .N / --spline_degree')
    if suffix not in SUFFIX_REGRESSORS[family]:
        raise SystemExit(f'unknown suffix {suffix!r} for family {family}; '
                         f'expected one of {sorted(SUFFIX_REGRESSORS[family])}')

    bauer_path = prepare_bauer(args.variant, args.worktree)
    sys.path.insert(0, str(bauer_path))
    sys.path.insert(0, str(REPO / 'tms_risk' / 'behavior'))

    import arviz as az
    import pymc as pm
    import bauer
    from fit_model import get_data

    print(f'bauer     {Path(bauer.__file__).parent}  (variant: {args.variant})')
    print(f'label     {label}  noise: {noise}  prior: {args.prior_estimate}  '
          + (f'splines: {spline_order} (degree {args.spline_degree})  ' if noise == 'flexible' else '')
          + f'family {family}  regressors: {SUFFIX_REGRESSORS[family][suffix] or "none"}'
          + (f'  memory: {args.memory_composition}' if family == 2 else ''))

    target = (Path(args.out_dir) if args.out_dir else
              Path(args.bids_folder) / 'derivatives' / 'cogmodels.noisefix')
    target.mkdir(parents=True, exist_ok=True)
    out = target / f'model-{label}.{args.variant}_trace.netcdf'
    if out.exists():
        raise SystemExit(f'{out} already exists; refusing to overwrite')

    df = get_data(args.bids_folder)
    print(f'data      {df.index.get_level_values("subject").nunique()} subjects, '
          f'{len(df)} trials')

    model = build(df, SUFFIX_REGRESSORS[family][suffix],
                  spline_order=spline_order, family=family,
                  spline_degree=args.spline_degree, noise=noise,
                  prior_estimate=args.prior_estimate,
                  memory_composition=args.memory_composition)
    if args.constrain:
        if noise == 'weber':
            # Weber's prior mu/sd live in LOG space, so the payoff-scale numbers
            # constrain_priors installs (N(16, 10) etc.) are meaningless there.
            # bauer's own defaults are already anchored to log(n) and the
            # published Weber fits converged under them.
            raise SystemExit('--constrain is natural-space only; weber uses bauer defaults')
        constrain_priors(model, df)
    if args.prior_estimate == 'objective':
        pin_objective_prior(model, df)
    if args.group_sd == 'hp':
        # Two different mechanisms exist in the wild and neither raises if you
        # use the wrong one. bauer >= 0.3.0 (the GPU boxes' HEAD) reads a
        # PER-INSTANCE `group_sd_dist`, and already DEFAULTS it to 'halfnormal';
        # the patched older clones read module-level GROUP_SD_DIST /
        # GROUP_SD_SCALE. Set whichever is present, and refuse if neither is.
        from bauer import core as _bauer_core
        if hasattr(model, 'group_sd_dist'):
            model.group_sd_dist = 'halfnormal'
            how = 'per-instance group_sd_dist'
        elif hasattr(_bauer_core, '_group_sd'):
            _bauer_core.GROUP_SD_DIST = 'halfnormal'
            _bauer_core.GROUP_SD_SCALE = 1.0
            how = 'module-level GROUP_SD_DIST'
        else:
            raise SystemExit(
                f'--group_sd hp: {_bauer_core.__file__} has neither a '
                f'per-instance group_sd_dist nor a _group_sd helper, so the '
                f'request would be silently ignored.')
        from tms_risk.behavior.fit_model import _scale_group_sds
        _scale_group_sds(model)
        print(f'group SDs: HalfNormal via {how}, per-parameter scales '
              f'(matching lfx2 -hp)')
    model.build_estimation_model()
    print(f'sampling  {args.chains} chains, {args.tune} tune + {args.draws} draws, '
          f'target_accept={args.target_accept}, backend={args.backend}, '
          f'init={args.init if args.backend == "pymc" else "n/a"}, '
          f'find_init={args.find_init}', flush=True)
    kw = {}
    if args.find_init:
        kw['find_init'] = args.find_init
    if args.backend == 'pymc':
        kw['init'] = args.init          # pymc-only kwarg
        kw['cores'] = args.cores
    trace = model.sample(draws=args.draws, tune=args.tune,
                         target_accept=args.target_accept, chains=args.chains,
                         backend=args.backend, **kw)

    n_div = int(trace.sample_stats['diverging'].sum())
    print(f'divergences {n_div} / {args.chains * args.draws}')

    # Convergence gate from bauer/notes/fitting_ddm_models.md: r_hat <= 1.01, ESS >= 400.
    group = [v for v in trace.posterior.data_vars if v.endswith('_mu')]
    summ = az.summary(trace, var_names=group, hdi_prob=.95)
    print(f'CONVERGENCE  max r_hat {summ.r_hat.max():.3f}  min ess_bulk {summ.ess_bulk.min():.0f}'
          f'  -> {"OK" if summ.r_hat.max() <= 1.01 and summ.ess_bulk.min() >= 400 else "FAILED"}')
    for v in ['risky_prior_mu_mu', 'safe_prior_mu_mu']:
        if v in trace.posterior:
            x = trace.posterior[v].values.reshape(args.chains, -1)
            print(f'  per-chain mean {v:20s} ' + '  '.join(f'{c:8.2f}' for c in x.mean(1)))

    if not args.no_log_likelihood:
        try:
            with model.estimation_model:
                pm.compute_log_likelihood(trace)
        except Exception as e:
            print(f'WARNING: compute_log_likelihood failed ({type(e).__name__}: {e})')

    trace.posterior.attrs['tms_risk_variant'] = args.variant
    trace.posterior.attrs['tms_risk_bauer'] = str(bauer_path)
    trace.posterior.attrs['tms_risk_bauer_commit'] = bauer_commit(bauer_path)
    trace.posterior.attrs['tms_risk_regressors'] = ','.join(SUFFIX_REGRESSORS[family][suffix])
    trace.posterior.attrs['tms_risk_family'] = family
    trace.posterior.attrs['tms_risk_noise'] = noise
    trace.posterior.attrs['tms_risk_prior_estimate'] = args.prior_estimate
    # Read back off the model, not off argv: this is the composition the
    # likelihood actually used, and get_sd_curve must be told the same thing.
    trace.posterior.attrs['tms_risk_memory_composition'] = getattr(
        model, 'memory_composition', 'sum_then_softplus')
    trace.posterior.attrs['tms_risk_spline_order'] = spline_order if noise == 'flexible' else 0
    trace.posterior.attrs['tms_risk_spline_degree'] = args.spline_degree
    trace.posterior.attrs['tms_risk_init'] = args.init
    trace.posterior.attrs['tms_risk_find_init'] = str(args.find_init)
    trace.posterior.attrs['tms_risk_backend'] = args.backend
    trace.posterior.attrs['tms_risk_constrained'] = str(args.constrain)
    trace.posterior.attrs['tms_risk_group_sd'] = str(args.group_sd)
    az.to_netcdf(trace, str(out))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
