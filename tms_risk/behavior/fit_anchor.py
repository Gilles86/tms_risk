"""Fit the anchor-parameterised PMC grid.

Label grammar -- three fields, every one of them structure, no knobs:

    <space>-<form>-<placement>

    space      log | chf                 inference scale of the observer
    form       weber | affine | power | genweber | spl3 | spl5

NOTE on `power` with a memory/perceptual placement (perc / mem / percmem /
null): the two CHANNELS are power laws, but sigma_n1 = sigma_perc + sigma_mem
and a sum of two power laws is not one (it departs from the closest single
power law by ~106%). The model is coherent; the name refers to the channels.
Anything reporting it should say "power-law channels", not "power-law noise".
Affine, generalized Weber and Weber ARE closed under addition, so for those the
composed n1 stays in the family.
    placement  null | n1 | n2 | n1n2     independent parameterisation
               perc | mem | percmem      memory/perceptual parameterisation

e.g. ``log-affine-perc``, ``chf-spl5-n1n2``, ``log-genweber-null``.

Everything the old `lfx2-...` grammar expressed with suffixes -- `dp/tp`, `-pN`,
`-fx`, `-op/-sp/-fs/-f1/-fp`, `-hn/-ts`, mapjitter -- is gone. Each was a
convergence workaround for priors that PRIOR_SPEC now sets once, and each was a
way for two fits to differ without the label saying so.

    python -m tms_risk.behavior.fit_anchor log-affine-perc \\
        --bids_folder /shares/zne.uzh/gdehol/ds-tmsrisk
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

#: Bumped whenever any number below changes. Stamped into every trace, so a
#: fit can always be traced back to the exact prior that produced it.
PRIOR_SPEC = 'v1-2026-08-28'

#: `spl5+affine` gives the two channels different forms: five anchors on the
#: perceptual (or n1) channel, two on the memory (or n2) one. The memory channel
#: is only ever seen through the first-presented option, so it is identified by
#: far less data and does not warrant the same flexibility.
LABEL_RE = re.compile(
    r'^(log|chf)-((?:weber|affine|power|genweber|spl3|spl5|spl7|spl9|cspl3|cspl5|cspl7)'
    r'(?:\+(?:weber|affine|power|genweber|spl3|spl5|spl7|spl9|cspl3|cspl5|cspl7))?)-'
    r'(null|nullind|n1|n2|n1n2|perc|mem|percmem'
    r'|pmu|psd|pmusd|n1n2pmu|n1n2psd|n1psd|n2psd|n2pmusd|n1n2pmusd'
    r'|spsd|spmusd|percpsd|percmempsd'
    r'|n2x|n1n2x|percx|percmemx)$')

#: placement -> (memory_model, noise channels carrying the cTBS regressor,
#: prior parameters carrying it). Prior names are given bare here and get the
#: space prefix in `build_model`, because `log_risky_prior_mu` and
#: `chf_risky_prior_mu` are different quantities and the model names them so.
#:
#: Why prior placements exist at all: in the original grid cTBS could only ever
#: move the noise function, and every one of those 72 fits underpredicts the
#: observed order contrast by about half. A prior that shifts or narrows under
#: stimulation is the obvious thing the model could not say. Note `psd` is
#: partly degenerate with noise -- the shrinkage weight is
#: sd_prior^2 / (sd_prior^2 + sd_evidence^2), so widening the prior and lowering
#: the noise move the same quantity -- while `pmu` is not: it moves WHERE the
#: percept is pulled to, and can do so asymmetrically for the risky and safe
#: roles.
# Two nulls, not one. In the shared parameterisation sigma_n1 = exp(theta_perc)
# + exp(theta_mem), a sum of two lognormals, whose prior median sits 2.25x above
# the independent family's exp(theta_n1) -- in every form and both spaces. One
# shared null would score the two families against baselines on different
# priors, so each family gets its own.
#: The default cTBS regressor. `X` adds the interaction with presentation
#: order, which is the one thing no placement in the original grid could say.
#: Every noise channel is indexed by POSITION (n1/n2) or by STAGE (perc/mem),
#: while the observed effect is confined to risky-SECOND trials. Raising
#: sigma_n2 flattens the psychometric curve in both orders -- it is the noise on
#: whichever option came second, risky or safe -- so the fit compromises to a
#: small effect. The interaction lets the cTBS effect on a channel differ by
#: order, which is what the data actually show.
REG = 'stimulation_condition'
REGX = 'stimulation_condition*risky_first'

_N12 = ['n1_evidence_sd', 'n2_evidence_sd']
_PMU = ['risky_prior_mu', 'safe_prior_mu']
_PSD = ['risky_prior_sd', 'safe_prior_sd']

PLACEMENT = {
    'null':     ('shared_perceptual_noise', [], []),
    'nullind':  ('independent', [], []),
    'n1':       ('independent', ['n1_evidence_sd'], []),
    'n2':       ('independent', ['n2_evidence_sd'], []),
    'n1n2':     ('independent', _N12, []),
    'perc':     ('shared_perceptual_noise', ['perceptual_noise_sd'], []),
    'mem':      ('shared_perceptual_noise', ['memory_noise_sd'], []),
    'percmem':  ('shared_perceptual_noise',
                 ['perceptual_noise_sd', 'memory_noise_sd'], []),
    # cTBS on the magnitude prior instead of, or as well as, the noise.
    # Baseline for all of these is `nullind`.
    'pmu':      ('independent', [], _PMU),
    'psd':      ('independent', [], _PSD),
    'pmusd':    ('independent', [], _PMU + _PSD),
    'n1n2pmu':  ('independent', _N12, _PMU),
    'n1n2psd':  ('independent', _N12, _PSD),
    # cTBS on ONE presentation position plus the prior width. Adding the prior
    # parameters to `n1n2` costs the n2 noise effect its credibility (p .980 ->
    # .936), so these ask whether the noise effect survives once the
    # unaffected position is not also free to move.
    'n1psd':    ('independent', ['n1_evidence_sd'], _PSD),
    'n2psd':    ('independent', ['n2_evidence_sd'], _PSD),
    # noise on the second-presented option PLUS both prior parameters. Tests
    # whether the option-type effect the data want is better carried by the
    # prior MEAN than by its width -- a narrower prior and a lower prior mean
    # both pull the safe option down, but they are different claims.
    'n2pmusd':  ('independent', ['n2_evidence_sd'], _PMU + _PSD),
    'n1n2pmusd': ('independent', _N12, _PMU + _PSD),
    # the same prior placements inside the SHARED perc/mem family, which is
    # markedly better conditioned (100% of Rule-A fits pass the gate against
    # 50% for the independent family) because sigma_n1 = sigma_perc +
    # sigma_mem makes sigma_n1 > sigma_n2 true by construction
    'spsd':       ('shared_perceptual_noise', [], _PSD),
    'spmusd':     ('shared_perceptual_noise', [], _PMU + _PSD),
    'percpsd':    ('shared_perceptual_noise', ['perceptual_noise_sd'], _PSD),
    'percmempsd': ('shared_perceptual_noise',
                   ['perceptual_noise_sd', 'memory_noise_sd'], _PSD),
    # cTBS x presentation-order interaction on the noise channels
    'n2x':      ('independent', ['n2_evidence_sd'], [], REGX),
    'n1n2x':    ('independent', _N12, [], REGX),
    'percx':    ('shared_perceptual_noise', ['perceptual_noise_sd'], [], REGX),
    'percmemx': ('shared_perceptual_noise',
                 ['perceptual_noise_sd', 'memory_noise_sd'], [], REGX),
}

# ---------------------------------------------------------------------------
# The prior specification. Every number here is set from the units of the
# quantity it governs, and from what the data can actually resolve -- measured,
# not guessed. See notes/refit_plan_2026-08-28.md section 3.
#
#   noise anchors   theta = log sigma. Centre at sigma = 0.25 (the middle of
#                   every fitted value we have; for the natural-space observer
#                   the same RELATIVE noise at the mean safe payoff). tau on the
#                   intercept is tight because this is the weakly-identified
#                   parameter: subject estimates use only 34-59% of the
#                   allowance the group SD gives them.
#   prior_mu        log CHF (log space) or CHF (natural space). tau is LOOSE:
#                   measured tau is 0.60-0.72 and subject estimates use 85-87%
#                   of it, so this is the best-identified parameter in the
#                   model and must not be shrunk.
#   prior_sd        log of the spread. Centred on the empirical value -- through
#                   the log, not through a softplus applied afterwards.
#   cTBS slopes     every regression slope in these models is the cTBS contrast.
#                   `sigma_slope` is the GROUP-MEAN prior on that slope; bauer
#                   defaults it to 1.0, which on a log-noise scale means a cTBS
#                   effect of e^1 = 2.7x per SD. The prior predictive caught
#                   that: it put the perceptual noise at a median of 0.46 and a
#                   97.5th percentile of 5.2 instead of the intended 0.25.
#                   0.25 gives +/-65% at 2 SD, generous against the ~10%
#                   effects actually observed.
# ---------------------------------------------------------------------------
PRIORS = {
    'noise':    dict(sigma_intercept=0.75, sigma_slope=0.25,
                     tau_intercept=0.30, tau_slope=0.30),
    'prior_mu': dict(sigma_intercept=1.00, sigma_slope=0.25,
                     tau_intercept=0.75, tau_slope=0.15),
    'prior_sd': dict(sigma_intercept=0.50, sigma_slope=0.25,
                     tau_intercept=0.40, tau_slope=0.15),
}
NOISE_CENTRE_REL = 0.25          # relative noise SD the prior is centred on


def parse_label(label):
    m = LABEL_RE.match(label)
    if not m:
        raise SystemExit(
            f'{label!r} is not a valid label. Grammar: <space>-<form>-'
            f'<placement>, e.g. log-affine-perc. '
            f'space: log|chf; form: weber|affine|power|genweber|spl3|spl5; '
            f'placement: {"|".join(PLACEMENT)}')
    return m.groups()


def build_model(label, df, role_scale=None):
    """Construct the model and install PRIOR_SPEC on it."""
    import bauer.models as bm
    space, form, placement = parse_label(label)
    spec = PLACEMENT[placement]
    memory_model, noise_targets, prior_targets = spec[:3]
    formula = spec[3] if len(spec) > 3 else REG
    targets = list(noise_targets) + [f'{space}_{p}' for p in prior_targets]
    cls = (bm.LogAnchorNoiseRiskRegressionModel if space == 'log'
           else bm.AnchorNoiseRiskRegressionModel)
    # `role_scale` only exists in the bauer clone that has the role-indexed
    # noise; pass it only when asked for, so the same fit_anchor works against
    # both checkouts instead of failing on the default value.
    kw = {} if role_scale is None else dict(role_scale=role_scale)
    model = cls(df, noise_form=form, memory_model=memory_model,
                regressors={t: formula for t in targets},
                prior_estimate='full', **kw)
    apply_priors(model, df, space)
    return model


def apply_priors(model, df, space):
    """Write PRIOR_SPEC onto the model's free parameters, by name.

    Raises if a parameter is not recognised, rather than leaving it on bauer's
    defaults: a prior that silently fails to apply is the exact failure mode
    this whole refit exists to remove.
    """
    pay = np.concatenate([np.asarray(df['n1'], float),
                          np.asarray(df['n2'], float)])
    safe = np.asarray(df['n_safe'], float) if 'n_safe' in df else pay
    # the natural-space observer's noise is an SD in CHF, so the same relative
    # noise corresponds to a different absolute number
    scale = 1.0 if space == 'log' else float(np.mean(safe))
    centre = np.log(NOISE_CENTRE_REL * scale)
    # prior_mu lives in log CHF for the log observer and in CHF for the natural
    # one, so a width of 1.0 means a factor of e in one and 1.03x in the other.
    # Multiply through by the payoff scale so both observers get the SAME
    # relative freedom -- otherwise the natural arm is pinned at its empirical
    # values and the linear-vs-log comparison is confounded by the prior.
    mu_scale = 1.0 if space == 'log' else float(np.mean(safe))

    risky = np.asarray(df['n_risky'], float) if 'n_risky' in df else pay
    unknown = []
    for key, info in model.free_parameters.items():
        if '_prior_mu' in key:
            p = dict(PRIORS['prior_mu'])
            emp = risky if 'risky' in key else safe
            # scale by THIS parameter's own payoff mean, so the risky and safe
            # priors get the same relative freedom as each other and as the
            # log-space observer's
            k_scale = 1.0 if space == 'log' else float(np.mean(emp))
            for k2 in ('sigma_intercept', 'sigma_slope',
                       'tau_intercept', 'tau_slope'):
                p[k2] = p[k2] * k_scale
            info['mu_intercept'] = float(np.mean(np.log(emp)) if space == 'log'
                                         else np.mean(emp))
        elif '_prior_sd' in key:
            p = PRIORS['prior_sd']
            emp = (np.std(np.log(risky if 'risky' in key else safe))
                   if space == 'log'
                   else np.std(risky if 'risky' in key else safe))
            info['mu_intercept'] = float(np.log(emp))
        elif False:
            pass
        elif key.endswith('_risky_noise_scale'):
            # log multiplier on the risky option's noise. Centred at 0
            # (payoff-indexing) with sigma 0.5, which puts EV-indexing
            # (b*log 0.55 = -0.20 for the fitted exponent) well inside one SD.
            p = dict(PRIORS['noise'], sigma_intercept=0.5)
            info['mu_intercept'] = 0.0
        elif any(t in key for t in ('_weber_', '_affine_', '_power_',
                                    '_genweber_', '_spl3_', '_spl5_',
                                    '_spl7_', '_spl9_',
                                    '_cspl3_', '_cspl5_', '_cspl7_')):
            p = PRIORS['noise']
            info['mu_intercept'] = float(centre)
        else:
            unknown.append(key)
            continue
        info['sigma_intercept'] = p['sigma_intercept']
        info['sigma_regressors'] = p['sigma_slope']
        info['cauchy_sigma_intercept'] = p['tau_intercept']
        info['cauchy_sigma_regressors'] = p['tau_slope']
    if unknown:
        raise SystemExit(f'no prior rule for {unknown}; refusing to fit with '
                         f'bauer defaults on them')
    model.group_sd_dist = 'halfnormal'
    return model


def bauer_commit():
    import bauer
    root = Path(bauer.__file__).resolve().parents[1]
    # a shipped clone has no .git; the deploy writes the commit to COMMIT
    stamped = root / 'COMMIT'
    if stamped.exists():
        return stamped.read_text().strip()
    try:
        out = subprocess.run(['git', '-C', str(root), 'rev-parse', 'HEAD'],
                             capture_output=True, text=True,
                             check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(root), 'status', '--porcelain'],
                               capture_output=True, text=True).stdout.strip()
        return out + ('+dirty' if dirty else '')
    except Exception:
        return 'unknown'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out_folder', default='cogmodels.anchor')
    ap.add_argument('--tune', default=3000, type=int)
    ap.add_argument('--draws', default=3000, type=int)
    ap.add_argument('--chains', default=4, type=int)
    ap.add_argument('--target_accept', default=0.9, type=float)
    ap.add_argument('--sigma_slope', default=None, type=float,
                    help='override PRIORS[*][sigma_slope], the group-mean prior '
                         'on every cTBS contrast. Default 0.25 on a log scale '
                         '(+/-65%% at 2 SD).')
    ap.add_argument('--tau_slope', default=None, type=float,
                    help='override PRIORS[*][tau_slope], the half-normal scale '
                         'on the BETWEEN-SUBJECT SD of every cTBS contrast. '
                         'Default 0.30 -- and the posteriors come back at '
                         '0.32-0.44, above the prior mode, which is what a '
                         'restraining prior looks like.')
    ap.add_argument('--role_scale', default=None, choices=['ev', 'free'],
                    help="index the noise function by expected value (p*x) "
                         "instead of payoff ('ev'), or give the risky option a "
                         "free log noise multiplier ('free', which nests both). "
                         "Payoff-indexing silently makes risky options ~29%% "
                         "noisier than safe ones because their payoffs are "
                         "larger; nothing in the grid was indexed by ROLE.")
    ap.add_argument('--consistent_choice_noise', action='store_true',
                    help='use the KLW-consistent decision noise, '
                         'sqrt((p1 w1 nu1)^2 + (p2 w2 nu2)^2), instead of '
                         'bauer\'s back-compat sqrt(nu1^2 + nu2^2). See '
                         'notes/klw_variance_analysis.md: the default shrinks '
                         'the numerator by w and leaves the denominator raw, so '
                         'the prior width changes the psychometric SLOPE as an '
                         'artefact of the normalisation. Under the consistent '
                         'rule w scales both and largely cancels.')
    ap.add_argument('--find_init', default=None,
                    choices=['mapjitter', 'priorjitter', 'pathfinder'],
                    help='starting-point strategy; pathfinder seeds each chain '
                         'from a variational draw in the typical set, which is '
                         'what a chain trapped in a secondary mode needs')
    ap.add_argument('--tau_intercept', default=None, type=float,
                    help='half-Cauchy scale on the GROUP SD of every intercept '
                         '(PRIOR_SPEC: 0.30 noise / 0.75 prior_mu / 0.40 '
                         'prior_sd). These are the parameters that fail to mix '
                         'in log-power-n2 and n2psd -- r_hat 1.09-1.13 with no '
                         'rogue chain, i.e. funnel geometry, not a stuck chain. '
                         'A heavy-tailed prior on a group SD is what opens the '
                         'funnel; tightening it regularises the neck.')
    ap.add_argument('--sigma_prior_mu', default=None, type=float,
                    help='group-mean prior SD on *_prior_mu. PRIOR_SPEC sets '
                         '1.0, which in log space lets the group prior mean sit '
                         'a factor of e from the payoff geometric mean -- wide '
                         'enough to open a ridge trading prior location and '
                         'width against second-option noise, which strands '
                         'roughly one chain in eight (measured over 32 chains x '
                         '4 inits on log-weber+affine-n1n2). 0.4 puts a 2.2x '
                         'displacement at 2 SD and closes it.')
    ap.add_argument('--data_label', default=None,
                    help="'baseline_all' = session 1, all 73 participants")
    ap.add_argument('--dry_run', action='store_true',
                    help='build and report the priors, sample nothing')
    args = ap.parse_args()

    # prior-sensitivity overrides, applied before any model is built
    for key in ('noise', 'prior_mu', 'prior_sd'):
        if args.sigma_slope is not None:
            PRIORS[key]['sigma_slope'] = args.sigma_slope
        if args.tau_slope is not None:
            PRIORS[key]['tau_slope'] = args.tau_slope
    if args.sigma_prior_mu is not None:
        PRIORS['prior_mu']['sigma_intercept'] = args.sigma_prior_mu
    if args.tau_intercept is not None:
        for key in ('noise', 'prior_mu', 'prior_sd'):
            PRIORS[key]['tau_intercept'] = args.tau_intercept

    from tms_risk.behavior.fit_model import get_data
    space, form, placement = parse_label(args.label)
    # 'baseline_all' fits the pre-stimulation session for the FULL cohort,
    # including the 38 participants who were never stimulated. `get_data`'s own
    # session-1 path keeps only the TMS cohort (35), which throws away half the
    # sample for a question that has nothing to do with stimulation.
    if getattr(args, 'data_label', None) == 'baseline_all':
        from tms_risk.utils.data import get_all_behavior
        df = get_all_behavior(bids_folder=args.bids_folder,
                              all_tms_conditions=False, exclude_outliers=True)
        df = df.xs(1, 0, 'session', drop_level=False)
        df = df.reset_index('stimulation_condition').reset_index('session')
        df['choice'] = df['choice'] == 2.0
        print(f'baseline_all: {len(df)} trials, '
              f'{df.index.get_level_values("subject").nunique()} subjects')
    else:
        df = get_data(args.bids_folder,
                      model_label=getattr(args, 'data_label', None)
                      or 'lfx2-bs3-m2-dp-bm')
    model = build_model(args.label, df, role_scale=args.role_scale)
    if args.consistent_choice_noise:
        model.consistent_choice_noise = True

    print(f'{args.label}: {space} space, {form} noise, cTBS on {placement}, '
          f'choice noise '
          f'{"consistent (KLW)" if args.consistent_choice_noise else "raw"}')
    print(f'  bauer {bauer_commit()[:9]} | prior spec {PRIOR_SPEC}')
    print(f'  anchors {np.round(model.anchors, 1)}')
    print(f'  {len(model.free_parameters)} free parameters:')
    for k, v in model.free_parameters.items():
        print(f'    {k:32s} mu={v.get("mu_intercept")!s:>8.8s} '
              f'sigma={v.get("sigma_intercept")} '
              f'tau_icpt={v.get("cauchy_sigma_intercept")} '
              f'tau_slope={v.get("cauchy_sigma_regressors")}')
    if args.dry_run:
        return

    ap_suffix = '' if args.find_init is None else f'.{args.find_init}'
    if args.consistent_choice_noise:
        ap_suffix += '.klw'
    if args.role_scale:
        ap_suffix += f'.{args.role_scale}'
    if args.sigma_slope is not None or args.tau_slope is not None:
        ap_suffix += (f'.ss{args.sigma_slope or 0.25:g}'
                      f'-ts{args.tau_slope or 0.30:g}')
    if args.sigma_prior_mu is not None:
        ap_suffix += f'.spm{args.sigma_prior_mu:g}'
    if args.tau_intercept is not None:
        ap_suffix += f'.ti{args.tau_intercept:g}'
    out = Path(args.bids_folder) / 'derivatives' / args.out_folder
    out.mkdir(parents=True, exist_ok=True)
    model.build_estimation_model()
    kw = {} if args.find_init is None else dict(find_init=args.find_init)
    trace = model.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                         target_accept=args.target_accept, **kw)
    # pymc 5.17 does not compute this by default and BaseModel.sample never
    # passes idata_kwargs, so without it every trace lands with no
    # log_likelihood group and the whole ELPD ladder is unobtainable.
    import pymc as pm
    with model.estimation_model:
        pm.compute_log_likelihood(trace)
    if 'log_likelihood' not in trace.groups():
        raise SystemExit('log_likelihood missing after compute; refusing to '
                         'write a trace that cannot be compared')
    a = trace.posterior.attrs
    a['tms_risk_label'] = args.label
    a['tms_risk_space'] = space
    a['tms_risk_noise_form'] = form
    a['tms_risk_placement'] = placement
    a['tms_risk_prior_spec'] = PRIOR_SPEC + (
        '' if args.sigma_prior_mu is None else f'+spm{args.sigma_prior_mu:g}') + (
        '' if args.tau_intercept is None else f'+ti{args.tau_intercept:g}')
    a['tms_risk_bauer_commit'] = bauer_commit()
    a['tms_risk_anchors'] = ','.join(f'{v:.0f}' for v in model.anchors)
    a['tms_risk_choice_noise'] = ('consistent' if args.consistent_choice_noise
                                  else 'raw_evidence_sd')
    a['tms_risk_role_scale'] = str(args.role_scale)
    a['tms_risk_slope_priors'] = (f'sigma_slope={args.sigma_slope} '
                                  f'tau_slope={args.tau_slope}')
    a['tms_risk_sampler'] = (f'chains={args.chains} tune={args.tune} '
                             f'draws={args.draws} ta={args.target_accept} '
                             f'find_init={args.find_init}')
    a['tms_risk_parameters'] = ','.join(sorted(model.free_parameters))
    fn = out / f'model-{args.label}{ap_suffix}_trace.netcdf'
    trace.to_netcdf(str(fn))
    print(f'wrote {fn}')


if __name__ == '__main__':
    main()
