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
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PATCH = REPO / 'notes' / 'patches' / 'bauer-ecc6454-noisefix.patch'
BASE_COMMIT = 'ecc6454'

# family 2 = shared_perceptual_noise (memory/perceptual); family 1 = independent
# (first/second option). They are exact reparameterisations of one another
# (c1 = memory + perceptual, c2 = perceptual), so the likelihoods are identical;
# only the prior coordinates and the sampling geometry differ. Family 1 is the
# better-conditioned one -- family 2's coordinates make the posterior bimodal.
SUFFIX_REGRESSORS = {
    2: {'': ['memory_noise_sd', 'perceptual_noise_sd'], '_null': [],
        '_memory': ['memory_noise_sd'], '_perception': ['perceptual_noise_sd']},
    1: {'': ['n1_evidence_sd', 'n2_evidence_sd'], '_null': [],
        '_first': ['n1_evidence_sd'], '_second': ['n2_evidence_sd']},
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
          noise='flexible', prior_estimate='full'):
    """Build the noise model.

    `noise='flexible'`  B-spline noise function over magnitude in natural space.
    `noise='weber'`     the original PMC: a single noise sd per term, applied in
                        log space, i.e. scalar invariance / Weber's law. This is
                        the paper's Table-1 baseline family (`11a`-`11c`,
                        `11_null` in fit_model.py) and takes no spline arguments.
    """
    import inspect
    import bauer.models as bm
    kw = dict(regressors={n: 'stimulation_condition' for n in regressor_names},
              memory_model='shared_perceptual_noise' if family == 2 else 'independent',
              prior_estimate=prior_estimate)
    if noise == 'weber':
        return bm.RiskRegressionModel(df, **kw)
    cls = bm.FlexibleNoiseRiskRegressionModel
    key = ('spline_order' if 'spline_order' in inspect.signature(cls).parameters
           else 'polynomial_order')
    kw[key] = spline_order
    if 'spline_degree' in inspect.signature(cls).parameters:
        kw['spline_degree'] = spline_degree
    elif spline_degree != 3:
        raise SystemExit('this bauer has no spline_degree; use the patched checkout')
    return cls(df, **kw)


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
          + f'family {family}  regressors: {SUFFIX_REGRESSORS[family][suffix] or "none"}')

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
                  prior_estimate=args.prior_estimate)
    if args.constrain:
        if noise == 'weber':
            # Weber's prior mu/sd live in LOG space, so the payoff-scale numbers
            # constrain_priors installs (N(16, 10) etc.) are meaningless there.
            # bauer's own defaults are already anchored to log(n) and the
            # published Weber fits converged under them.
            raise SystemExit('--constrain is natural-space only; weber uses bauer defaults')
        constrain_priors(model, df)
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
    trace.posterior.attrs['tms_risk_spline_order'] = spline_order if noise == 'flexible' else 0
    trace.posterior.attrs['tms_risk_spline_degree'] = args.spline_degree
    trace.posterior.attrs['tms_risk_init'] = args.init
    trace.posterior.attrs['tms_risk_find_init'] = str(args.find_init)
    trace.posterior.attrs['tms_risk_backend'] = args.backend
    trace.posterior.attrs['tms_risk_constrained'] = str(args.constrain)
    az.to_netcdf(trace, str(out))
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
