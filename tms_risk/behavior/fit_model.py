"""Fit bauer cognitive models by short string label.

Live model labels (referenced by analysis notebooks):

    Weber PMC family (RiskRegressionModel):
        1c                      n2_evidence_sd ~ stim (single noise regressor)
        10_null, 10a, 10b, 10c  TMS on n1/n2 evidence noise (independent memory)
        11_null, 11a, 11b, 11c  TMS on perceptual / memory noise (shared mem)
        12a, 12b, 12c, 12d, 12e TMS on prior μ / σ for risky / safe options

    Flexible PMC family (FlexibleNoiseRiskRegressionModel) — paper's main model:
        flexible1[.4|.6][_null|a|b]  TMS on n1/n2 evidence noise, polynomial orders 3/4/6
        flexible2[.4|.6][_null|a|b]  TMS on perceptual / memory noise, polynomial orders 3/4/6

        Suffix legend: '_null' = no TMS regressor; 'a' = only n1/memory;
        'b' = only n2/perceptual; bare label = both.

    Power-law-noise PMC family (PowerLawNoiseRiskRegressionModel) — the
    efficient-coding comparison (SD_k(n) = exp(intercept_k) · n^exponent):
        power{1|2}[_flat][_null|_exp|_full]
        1 = independent noise, 2 = shared perceptual/memory noise;
        '_flat' = no prior / no shrinkage (Thurstonian observer);
        '_null' = no TMS regressor, bare = TMS on noise intercepts,
        '_exp' = TMS on the exponent, '_full' = TMS on both.

    DDM × Flexible PMC family (DDMFlexibleNoiseRiskRegressionModel) — Phase 5:
        ddm_flexible[_null|_perception|_memory|_threshold|_noise_threshold]

    Race-diffusion × Flexible PMC family (RaceDiffusionFlexibleNoiseRiskRegressionModel):
        rdm_flexible[_null|_perception|_memory|_threshold|_noise_threshold]

    Session-1 baselines (RiskModel, no TMS regressor):
        everyone                fit pooled across all sessions/subjects
        session1_full           prior_estimate='full', separate n1/n2 noise

Legacy labels (kept for reference in legacy_models.py): 1, 1_null, 1a, 1b,
1_session, 2*, 3*, 5*, 6*, 7, session1_*, 20, 21. They were exploratory
variants that no live notebook references — pruning them keeps this
dispatch readable. See legacy_models.py to resurrect one.
"""
import argparse
from pathlib import Path

import arviz as az
import numpy as np

from bauer.models import (
    RiskModel,
    RiskRegressionModel,
    FlexibleNoiseRiskRegressionModel,
    PowerLawNoiseRiskRegressionModel,
    LogFlexibleNoiseRiskRegressionModel,
)
try:
    from bauer.models import (
        DDMFlexibleNoiseRiskRegressionModel,
        DDMRiskRegressionModel,
    )
except ImportError:
    DDMFlexibleNoiseRiskRegressionModel = None
    DDMRiskRegressionModel = None
try:
    from bauer.models import (
        RaceDiffusionFlexibleNoiseRiskRegressionModel,
        RaceDiffusionRiskRegressionModel,
    )
except ImportError:
    RaceDiffusionFlexibleNoiseRiskRegressionModel = None
    RaceDiffusionRiskRegressionModel = None

from tms_risk.utils.data import get_all_behavior


def main(model_label, burnin=None, samples=None, bids_folder='/data/ds-tmsrisk',
         backend=None, out_folder=None, group_sd=None, target_accept=None):

    # Group-SD prior family, applied to every hierarchical node of whatever
    # model follows. HalfNormal clips the fat tail that creates a
    # group-SD/subject-offset funnel (dyscalculic_ddm lesson 2, after Gelman).
    if group_sd is not None:
        from bauer import core as _bauer_core
        _bauer_core.GROUP_SD_DIST = group_sd

    df = get_data(bids_folder, model_label=model_label)

    target_folder = Path(bids_folder) / 'derivatives' / (out_folder or 'cogmodels')
    target_folder.mkdir(parents=True, exist_ok=True)

    is_accumulator = model_label.startswith('ddm_') or model_label.startswith('rdm_')
    target_accept_override = target_accept

    # DDM/RDM fits use the recipe from bauer's
    # notes/tms_risk_ddm_fitting_brief.md: numpyro backend, tune=2000,
    # target_accept=0.99, and bauer's `mapjitter` starting-point finder
    # (on by default for DDM/Race — MAP centre + prior-scaled jitter).
    # That recipe took the failing config from ~12% to 100% convergence.
    if is_accumulator:
        burnin = burnin or 2000
        samples = samples or 1000
        backend = backend or 'numpyro'
        target_accept = 0.99
    elif model_label.startswith('flexible') or model_label.startswith('session1'):
        burnin = burnin or 5000
        samples = samples or 5000
        backend = backend or 'pymc'
        target_accept = 0.9
    elif (model_label.startswith('power') or model_label.startswith('logflex')
          or model_label.startswith('lfx2-')):
        # The hierarchical prior-SD funnel gives ~4-8% divergences at 0.8
        # (observed on power1_null / power1, 2026-08-20); 0.95 is needed for
        # trustworthy stimulation-coefficient posteriors.
        burnin = burnin or 5000
        samples = samples or 5000
        backend = backend or 'pymc'
        target_accept = 0.95
    else:
        burnin = burnin or 5000
        samples = samples or 5000
        backend = backend or 'pymc'
        target_accept = 0.8

    if target_accept_override is not None:
        target_accept = target_accept_override

    model = build_model(model_label, df)
    model.build_estimation_model()
    sample_kwargs = {}
    if is_accumulator and backend == 'numpyro':
        # one GPU: vectorized chains run in parallel (bauer fitting brief)
        sample_kwargs['chain_method'] = 'vectorized'
    trace = model.sample(burnin, samples, target_accept=target_accept,
                         backend=backend, **sample_kwargs)

    # Compute per-observation log-likelihood in-place so downstream LOO /
    # WAIC works without rebuilding the model. Without this the comparison
    # notebooks have to rebuild the model in an env that knows the model
    # class — for DDM/RDM that means an hssm-enabled env, which we don't
    # always have locally for plotting.
    try:
        import pymc as pm
        with model.estimation_model:
            pm.compute_log_likelihood(trace)
    except Exception as e:
        print(f'WARNING: pm.compute_log_likelihood failed ({type(e).__name__}: {e}); '
              f'LOO / WAIC will need a manual rebuild step.')

    # Stamp the bauer commit — a stored trace only means something against the
    # bauer code that produced it (see CLAUDE.md, "Cognitive-model traces are
    # bauer-version-sensitive").
    try:
        import subprocess
        import bauer as _bauer
        bauer_repo = Path(_bauer.__file__).resolve().parent.parent
        commit = subprocess.run(['git', '-C', str(bauer_repo), 'rev-parse', 'HEAD'],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(bauer_repo), 'status', '--porcelain'],
                               capture_output=True, text=True, check=True).stdout.strip()
        trace.posterior.attrs['tms_risk_bauer_commit'] = commit + ('+dirty' if dirty else '')
    except Exception as e:
        print(f'WARNING: could not stamp bauer commit ({type(e).__name__}: {e})')
    if hasattr(model, 'p_lapse'):
        trace.posterior.attrs['tms_risk_p_lapse'] = (
            'fitted' if getattr(model, 'fit_p_lapse', False)
            else float(model.p_lapse))

    az.to_netcdf(trace, str(target_folder / f'model-{model_label}_trace.netcdf'))


# ---------------------------------------------------------------------------
# Helpers for repeated regressor recipes
# ---------------------------------------------------------------------------

def _stim(*names):
    """Build {name: 'stimulation_condition', ...} regressor dict."""
    return {n: 'stimulation_condition' for n in names}


def _flexible_noise_regressors(suffix, memory_model):
    """Regressor dict for the flexible PMC family.

    suffix: '' (both noise terms), 'a' (n1/memory only), 'b' (n2/perceptual only),
            '_null' (no TMS effect on noise).
    memory_model: 'independent' (flexible1 family) → uses n1/n2_evidence_sd
                  'shared_perceptual_noise' (flexible2 family) → uses
                  memory_noise_sd / perceptual_noise_sd
    """
    if suffix == '_null':
        return {}
    if memory_model == 'independent':
        first, second = 'n1_evidence_sd', 'n2_evidence_sd'
    else:
        first, second = 'memory_noise_sd', 'perceptual_noise_sd'
    if suffix == 'a':
        return _stim(first)
    if suffix == 'b':
        return _stim(second)
    return _stim(first, second)


def _build_flexible(model_label, df):
    """Dispatch flexible1[.4|.6][_null|a|b] / flexible2[.4|.6][_null|a|b]."""
    head, _, suffix = model_label.partition('_')
    if suffix not in ('', 'null'):
        # 'flexible1a' etc — suffix lives in the tail of `head`, not after `_`
        suffix = ''
    # Re-parse: family ∈ {flexible1, flexible2}; optional .4/.6; optional a/b
    rest = model_label[len('flexible'):]
    family_digit = rest[0]            # '1' or '2'
    rest = rest[1:]
    polynomial_order = 5
    if rest.startswith('.4'):
        polynomial_order = 4
        rest = rest[2:]
    elif rest.startswith('.6'):
        polynomial_order = 6
        rest = rest[2:]
    if rest == '_null':
        suffix = '_null'
    elif rest in ('a', 'b'):
        suffix = rest
    elif rest == '':
        suffix = ''
    else:
        raise Exception(f'Unrecognised flexible suffix: {rest!r}')

    memory_model = 'independent' if family_digit == '1' else 'shared_perceptual_noise'
    regressors = _flexible_noise_regressors(suffix, memory_model)
    return FlexibleNoiseRiskRegressionModel(
        df, regressors=regressors,
        spline_order=polynomial_order,   # bauer renamed polynomial_order → spline_order
        memory_model=memory_model,
        prior_estimate='full',
    )


def _build_logflex(model_label, df):
    """Dispatch logflex{1,2}[_null|a|b] — log-space flexible PMC.

    Weber RiskModel front-end (log-payoff evidence, log(p2/p1) threshold,
    lognormal priors) + spline noise over log payoff. Family and suffix
    semantics identical to the flexible family: 1 = independent (n1/n2),
    2 = shared perceptual/memory; '_null' no TMS, 'a' first/memory only,
    'b' second/perceptual only, bare = both noise terms.
    """
    rest = model_label[len('logflex'):]
    scalar_mem = rest.startswith('m')     # scalar memory + tightened hyperpriors
    tight_only = rest.startswith('t')     # tightened hyperpriors, 5-df memory
    natural = rest.startswith('n')        # natural cubic basis (cr), default priors
    quadratic = rest.startswith('q')      # degree-2 B-splines, default priors
    if scalar_mem or tight_only or natural or quadratic:
        rest = rest[1:]
    family_digit, rest = rest[0], rest[1:]
    if family_digit not in ('1', '2') or rest not in ('', '_null', 'a', 'b'):
        raise Exception(f'Unrecognised logflex label: {model_label!r}')
    memory_model = 'independent' if family_digit == '1' else 'shared_perceptual_noise'
    regressors = _flexible_noise_regressors(rest, memory_model)
    # logflexm*: scalar (Weber-style) memory noise + 5-df perceptual spline,
    # with tightened hyperpriors on the noise machinery: TMS coefficients get
    # prior scale 0.4 (pre-softplus, ~"changes beyond +-50% implausible")
    # instead of the default 1.0, and spline intercepts 1.2 instead of 1.5 --
    # the constrain_priors philosophy, aimed at the high-payoff variance.
    model = LogFlexibleNoiseRiskRegressionModel(
        df, regressors=regressors,
        spline_order=(1, 5) if scalar_mem else 5,
        memory_model=memory_model, prior_estimate='full',
        spline_basis='cr' if natural else 'bs',
        spline_degree=2 if quadratic else 3,
    )
    if scalar_mem or tight_only:
        _tighten_noise_hyperpriors(model)
    return model


def _tighten_noise_hyperpriors(model):
    """Shrink the noise-spline hyperpriors: coefficient scale 1.5 -> 1.2 and
    TMS regression-coefficient scale 1.0 -> 0.4 (pre-softplus; ~'noise changes
    beyond +-50% implausible'). Applied by wrapping get_free_parameters, which
    is what feeds build_hierarchical_nodes at graph-build time."""
    orig_gfp = model.get_free_parameters

    def tightened():
        fp = orig_gfp()
        for k, info in fp.items():
            if 'spline' in k:
                info['sigma_intercept'] = 1.2
                info['sigma_regressors'] = 0.4
        return fp

    model.get_free_parameters = tightened
    return model


def _build_accumulator_logflex(model_label, df):
    """Dispatch {ddm|rdm}_logflex2[_null|b|_threshold][_ws0].

    Accumulator versions of the log-space flexible PMC (see
    notes/rdm_magnitude_rt_plan.md). Family 2 = shared perceptual/memory
    noise, prior_estimate='full'. Suffixes: '_null' no TMS regressor;
    'b' TMS on perceptual splines; '_threshold' TMS on the decision bound
    `a` (caution confound control). '_ws0' (race only) ablates the w_s
    magnitude→RT sum channel. Ships the prior-mean wandering mitigation:
    prior-μ hyperprior centering tightened to σ = 0.5 log-units.
    """
    kind = 'ddm' if model_label.startswith('ddm_') else 'rdm'
    rest = model_label[len(f'{kind}_logflex'):]
    if not rest.startswith('2'):
        raise Exception(f'Only family 2 supported: {model_label!r}')
    rest = rest[1:]
    # optional memory-df marker mirroring the static grid's m2/m3 cells:
    # 'm2' = linear memory noise in log payoff, 'm3' = quadratic. Kills the
    # 5-df memory-spline hyperprior funnel that broke the first RT wave
    # (0/6 converged, worst r-hats on memory_noise_sd_spline*_sd).
    mem_df, perc_df = 5, 5
    if rest[:2] in ('m2', 'm3'):
        mem_df = int(rest[1])
        rest = rest[2:]
    elif rest[:1] == 'w':
        # 'w' = log-space Weber: BOTH noise channels scalar. The chain
        # diagnostic on the m2 op fits put the residual pathology in the
        # 5-df perceptual spline's adjacent-coefficient collinearity
        # (r = -0.60 to -0.70), not in the priors/w_d/SD tails that the
        # earlier arms fixed. Scalar noise removes it, and matches the
        # dyscalculic_ddm spec that converged.
        mem_df, perc_df = 1, 1
        rest = rest[1:]
    # '_dm0': diagonal mass matrix. RaceMixin/DDMMixin recommend
    # dense_mass=True, but these models carry ~666 latent dims (35 subjects
    # x per-parameter offsets), so a full mass matrix means estimating
    # ~222k covariance entries from 2000 tuning draws — under-determined,
    # and a noisy near-singular metric wrecks mixing. The *choice* models,
    # which converge fine, use the default diagonal metric.
    diag_mass = rest.endswith('_dm0')
    if diag_mass:
        rest = rest[:-len('_dm0')]
    # '_op': objective priors — pin the observer's prior at the log-payoff
    # statistics, removing the prior block (and its hyperprior funnel, the
    # worst-mixing parameters of the m2 RT wave) from the fit entirely.
    # Mirrors the dyscalculic_ddm winning spec, which had no free priors.
    obj_prior = rest.endswith('_op')
    if obj_prior:
        rest = rest[:-len('_op')]
    # '_hn': HalfNormal instead of HalfCauchy on ALL group SDs (kills the
    # fat-tail funnel; dyscalculic_ddm lesson 2). Global switch in
    # bauer.core, so it applies to every hierarchical node of this model.
    from bauer import core as _bauer_core
    _bauer_core.GROUP_SD_DIST = 'halfcauchy'
    if rest.endswith('_hn'):
        _bauer_core.GROUP_SD_DIST = 'halfnormal'
        rest = rest[:-len('_hn')]
    # '_wd1' (race only): pin the evidence-to-drift gain w_d = 1, breaking
    # the exact (σ, w's, a) scale ridge so RTs identify the noise LEVEL.
    wd1 = rest.endswith('_wd1')
    if wd1:
        if kind == 'ddm':
            raise Exception('_wd1 is race-only (the DDM pins v_scale=1 already)')
        rest = rest[:-len('_wd1')]
    ws0 = rest.endswith('_ws0')
    if ws0:
        if kind == 'ddm':
            raise Exception('_ws0 is race-only (the DDM has no sum channel)')
        rest = rest[:-len('_ws0')]
    if rest == '_null':
        regressors = {}
    elif rest == 'b':
        regressors = _stim('perceptual_noise_sd')
    elif rest == 'bm':
        regressors = _stim('perceptual_noise_sd', 'memory_noise_sd')
    elif rest == '_threshold':
        regressors = _stim('a')
    else:
        raise Exception(f'Unrecognised accumulator-logflex suffix: {rest!r}')

    from bauer.models import (DDMLogFlexibleNoiseRiskRegressionModel,
                              RaceDiffusionLogFlexibleNoiseRiskRegressionModel)
    spline_order = (mem_df, perc_df)
    prior_estimate = 'objective' if obj_prior else 'full'
    if kind == 'ddm':
        model = DDMLogFlexibleNoiseRiskRegressionModel(
            df, regressors=regressors, prior_estimate=prior_estimate,
            memory_model='shared_perceptual_noise', spline_order=spline_order)
    else:
        model = RaceDiffusionLogFlexibleNoiseRiskRegressionModel(
            df, regressors=regressors, prior_estimate=prior_estimate,
            memory_model='shared_perceptual_noise', spline_order=spline_order,
            fit_w_s=not ws0, fit_w_d=not wd1)

    if diag_mass:
        # instance attribute shadows the mixin's class-level recommendation
        model.recommended_nuts_kwargs = {}

    # prior-mean wandering mitigation (memo §6): tighter centering on μ.
    orig_gfp = model.get_free_parameters

    def centered():
        fp = orig_gfp()
        for k, info in fp.items():
            if k.endswith('_prior_mu'):
                info['sigma_intercept'] = 0.5
        return fp

    model.get_free_parameters = centered
    return model


def _build_lfx_grid(model_label, df):
    """Systematic spline/hyperprior grid for the log-space flexible PMC.

    Label grammar:  lfx2-{bs3|bs2|cr3}-{fm|sm}-{dp|tp}-{null|b}

        bs3 / bs2 / cr3   cubic B-spline / quadratic B-spline / natural cubic
        fm / sm           flexible (5-df spline) vs scalar memory noise
        dp / tp           default vs tightened noise hyperpriors
        null / b          no TMS regressor vs TMS on perceptual noise

    3 x 2 x 2 x 2 = 24 cells; shared_perceptual_noise, prior_estimate='full'
    throughout. Traces land wherever --out_folder points (cogmodels.lfxgrid
    for the 2026-08 cluster sweep).
    """
    import re
    m = re.fullmatch(r'lfx2-(bs3|bs2|cr3)-(fm|sm|m2|m3|w|sd2|sd3|sd5)-(dp|tp)-(null|b|bm|t)'
                     r'(-op|-sp|-fs|-f1)?(-hn)?', model_label)
    if not m:
        raise Exception(f'Bad lfx2 grid label: {model_label!r}')
    basis, mem, hp, tms, pri, hn = m.groups()
    # '-hn': HalfNormal rather than HalfCauchy on every group SD. Set
    # explicitly either way — GROUP_SD_DIST is module-level state, so a
    # script that builds several models in one process must not inherit it.
    from bauer import core as _bauer_core
    _bauer_core.GROUP_SD_DIST = 'halfnormal' if hn else 'halfcauchy'
    # memory-spline df ladder: sm=1 (scalar), m2=2 (linear in log n),
    # m3=3 (quadratic), fm=5. spline_order = (memory, perceptual).
    # 'w' = log-space Weber: BOTH noises scalar, so each TMS lever is a
    # single coefficient — the maximally-powered test of "cTBS raises
    # noise", at the cost of the magnitude dependence the flexible cells
    # need to reproduce the order asymmetry.
    # sd2/sd3/sd5: sum/difference rotation of the shared-noise model -- fit
    # total and split noise instead of perceptual and memory. The digit is
    # the split-function df; total always gets 5.
    sumdiff = mem.startswith('sd')
    mem_df = ({'sm': 1, 'm2': 2, 'm3': 3, 'fm': 5, 'w': 1}[mem] if not sumdiff
              else int(mem[2]))
    perc_df = 1 if mem == 'w' else 5
    model = LogFlexibleNoiseRiskRegressionModel(
        df,
        regressors=({} if tms == 'null' else
                    # 't': cTBS on TOTAL noise only -- a restriction of 'bm',
                    # one effect function, no channel ambiguity.
                    _stim('total_noise_sd') if tms == 't' else
                    _stim('total_noise_sd', 'split_noise_sd') if sumdiff else
                    _stim('perceptual_noise_sd') if tms == 'b' else
                    _stim('perceptual_noise_sd', 'memory_noise_sd')),
        spline_order=(mem_df, perc_df),
        memory_model='sum_difference' if sumdiff else 'shared_perceptual_noise',
        # Prior block: 'full' = four free (risky/safe) x (mu/sd) params,
        # each hierarchical. '-sp' = one prior shared across the two roles
        # (2 params, subject variation kept). '-op' = pinned at the
        # log-payoff statistics, no free params and no subject variation --
        # which converges instantly but costs 580 +- 31 ELPD, so it is a
        # diagnostic, not a candidate.
        prior_estimate=({'-op': 'objective', '-sp': 'shared',
                         '-fs': 'fix_prior_sd',
                         '-f1': 'fix_safe_prior_sd'}[pri]
                        if pri else 'full'),
        spline_basis='cr' if basis == 'cr3' else 'bs',
        spline_degree=2 if basis == 'bs2' else 3,
    )
    if hp == 'tp':
        _tighten_noise_hyperpriors(model)
    return model


def _build_power(model_label, df):
    """Dispatch power{1,2}[_flat][|_null|_exp|_full] — power-law-noise PMC.

    Noise: SD_k(n) = exp(log_sd_intercept_k) · n^noise_exponent, exponent
    shared across options (β = 1 − α indexes Stevens compression).

    Family: power1 = memory_model 'independent' (n1/n2 intercepts, flexible1
    analogue); power2 = 'shared_perceptual_noise' (perceptual/memory
    intercepts, flexible2 analogue).

    '_flat' marker = prior_estimate 'none' — the Thurstonian /
    efficient-coding observer with no shrinkage (exact prior_sd → ∞ limit of
    the Bayesian model, so the pairs are nested). Without it,
    prior_estimate='full' as in the flexible family.

    Suffix: '_null' no TMS regressors; '' (bare) TMS on the two noise
    intercepts (noise-scale change only); '_exp' TMS on noise_exponent only
    (compression change only); '_full' TMS on intercepts + exponent.
    """
    rest = model_label[len('power'):]
    family_digit, rest = rest[0], rest[1:]
    if family_digit not in ('1', '2'):
        raise Exception(f'Unrecognised power family: {model_label!r}')
    memory_model = 'independent' if family_digit == '1' else 'shared_perceptual_noise'

    prior_estimate = 'full'
    if rest.startswith('_flat'):
        prior_estimate = 'none'
        rest = rest[len('_flat'):]

    if memory_model == 'independent':
        intercepts = ('n1_log_sd_intercept', 'n2_log_sd_intercept')
    else:
        intercepts = ('perceptual_log_sd_intercept', 'memory_log_sd_intercept')

    if rest == '_null':
        regressors = {}
    elif rest == '':
        regressors = _stim(*intercepts)
    elif rest == '_exp':
        regressors = _stim('noise_exponent')
    elif rest == '_full':
        regressors = _stim('noise_exponent', *intercepts)
    else:
        raise Exception(f'Unrecognised power suffix: {rest!r}')

    return PowerLawNoiseRiskRegressionModel(
        df, regressors=regressors,
        prior_estimate=prior_estimate,
        memory_model=memory_model,
    )


def _build_ddm_or_rdm(model_label, df):
    """Dispatch ddm_*/rdm_* labels (Weber- and Flexible-noise; two memory_models).

    Three noise structures, two SSM kinds (DDM, RDM):

    - ``*_weber_*`` — Weber (scalar) noise, ``memory_model='shared_perceptual_noise'``
      (paper-analogue of `11_*`). Regressors are on `perceptual_noise_sd` /
      `memory_noise_sd`.
    - ``*_indep_*`` — Weber (scalar) noise, ``memory_model='independent'``
      (bauer's default). Regressors are on `n1_evidence_sd` / `n2_evidence_sd`.
      Different decomposition of the same evidence-noise structure.
    - ``*_flexible_*`` — 5-spline noise per term, ``memory_model='shared_perceptual_noise'``
      (paper-analogue of `flexible2_*`).

    Suffix legend (slightly different per noise structure):
        _null            no TMS regressor (baseline)
        ── weber / flexible (shared_perceptual_noise) ──
        _perception      TMS on perceptual_noise_sd only
        _memory          TMS on memory_noise_sd only
        ''               TMS on both noise terms (paper's main claim analogue)
        ── indep (independent memory model) ──
        _n1              TMS on n1_evidence_sd only
        _n2              TMS on n2_evidence_sd only
        ''               TMS on both
        ── flexible only ──
        _threshold       TMS on accumulator threshold ``a`` only
        _noise_threshold TMS on both noise terms + threshold
    """
    if model_label.startswith('ddm_'):
        kind = 'ddm'
        weber_cls = DDMRiskRegressionModel
        flex_cls = DDMFlexibleNoiseRiskRegressionModel
    elif model_label.startswith('rdm_'):
        kind = 'rdm'
        weber_cls = RaceDiffusionRiskRegressionModel
        flex_cls = RaceDiffusionFlexibleNoiseRiskRegressionModel
    else:
        raise Exception(f'Not a DDM/RDM label: {model_label!r}')

    rest = model_label[len(kind) + 1:]   # strip "ddm_" or "rdm_"
    if rest.startswith('flexible'):
        cls = flex_cls
        suffix = rest[len('flexible'):]
        memory_model = 'shared_perceptual_noise'
        is_flex = True
    elif rest.startswith('weber'):
        cls = weber_cls
        suffix = rest[len('weber'):]
        memory_model = 'shared_perceptual_noise'
        is_flex = False
    elif rest.startswith('indep'):
        cls = weber_cls
        suffix = rest[len('indep'):]
        memory_model = 'independent'
        is_flex = False
    else:
        raise Exception(
            f'{kind} label must start with {kind}_weber_* / {kind}_indep_* '
            f'/ {kind}_flexible_*, got {model_label!r}')

    if cls is None:
        raise Exception(
            f'{kind} class for label {model_label!r} is not available — '
            f'check libs/bauer is at a recent commit + the DDM env has hssm.')

    if suffix.startswith('_'):
        suffix = suffix[1:]

    regressors = {}
    if memory_model == 'shared_perceptual_noise':
        n1_term, n2_term = 'memory_noise_sd', 'perceptual_noise_sd'
        n1_label, n2_label = 'memory', 'perception'
    else:
        n1_term, n2_term = 'n1_evidence_sd', 'n2_evidence_sd'
        n1_label, n2_label = 'n1', 'n2'

    if suffix == 'null':
        pass
    elif suffix == n1_label:
        regressors = _stim(n1_term)
    elif suffix == n2_label:
        regressors = _stim(n2_term)
    elif suffix == '':
        regressors = _stim(n1_term, n2_term)
    elif suffix == 'threshold' and is_flex:
        regressors = _stim('a')
    elif suffix == 'noise_threshold' and is_flex:
        regressors = _stim(n1_term, n2_term, 'a')
    else:
        raise Exception(
            f'Unrecognised {kind} suffix {suffix!r} for {memory_model} '
            f'(valid: null, {n1_label}, {n2_label}, '''
            + (', threshold, noise_threshold' if is_flex else '') + ')')

    kwargs = dict(prior_estimate='full', memory_model=memory_model)
    if is_flex:
        kwargs['spline_order'] = 5
    return cls(df, regressors=regressors, **kwargs)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def build_model(model_label, df):

    # Flexible PMC family — paper's main model
    if model_label.startswith('flexible'):
        return _build_flexible(model_label, df)

    # Log-space flexible PMC (Weber front-end + spline noise over log payoff)
    if model_label.startswith('lfx2-'):
        return _build_lfx_grid(model_label, df)
    if model_label.startswith('logflex'):
        return _build_logflex(model_label, df)

    # Power-law-noise PMC family (efficient-coding comparison)
    if model_label.startswith('power'):
        return _build_power(model_label, df)

    # DDM / RDM × log-space flexible PMC (magnitude→RT program, 2026-08-22)
    if model_label.startswith(('ddm_logflex', 'rdm_logflex')):
        return _build_accumulator_logflex(model_label, df)

    # DDM / RDM × Flexible PMC family (Phase 5)
    if model_label.startswith('ddm_') or model_label.startswith('rdm_'):
        return _build_ddm_or_rdm(model_label, df)

    # Weber PMC family (RiskRegressionModel)
    if model_label == '1c':
        return RiskRegressionModel(
            df, regressors={'n2_evidence_sd': 'stimulation_condition'},
            prior_estimate='full',
        )

    if model_label == '10_null':
        return RiskRegressionModel(df, regressors={}, prior_estimate='full')
    if model_label == '10a':
        return RiskRegressionModel(
            df, regressors=_stim('n1_evidence_sd', 'n2_evidence_sd'),
            prior_estimate='full',
        )
    if model_label == '10b':
        return RiskRegressionModel(
            df, regressors=_stim('n1_evidence_sd'), prior_estimate='full',
        )
    if model_label == '10c':
        return RiskRegressionModel(
            df, regressors=_stim('n2_evidence_sd'), prior_estimate='full',
        )

    if model_label == '11_null':
        return RiskRegressionModel(
            df, regressors={},
            memory_model='shared_perceptual_noise', prior_estimate='full',
        )
    if model_label == '11a':
        return RiskRegressionModel(
            df, regressors=_stim('memory_noise_sd', 'perceptual_noise_sd'),
            memory_model='shared_perceptual_noise', prior_estimate='full',
        )
    if model_label == '11b':
        return RiskRegressionModel(
            df, regressors=_stim('memory_noise_sd'),
            memory_model='shared_perceptual_noise', prior_estimate='full',
        )
    if model_label == '11c':
        return RiskRegressionModel(
            df, regressors=_stim('perceptual_noise_sd'),
            memory_model='shared_perceptual_noise', prior_estimate='full',
        )

    if model_label == '12a':
        return RiskRegressionModel(
            df, regressors=_stim('risky_prior_mu', 'safe_prior_mu'),
            prior_estimate='full',
        )
    if model_label == '12b':
        return RiskRegressionModel(
            df, regressors=_stim('risky_prior_mu'), prior_estimate='full',
        )
    if model_label == '12c':
        return RiskRegressionModel(
            df, regressors=_stim('safe_prior_mu'), prior_estimate='full',
        )
    if model_label == '12d':
        return RiskRegressionModel(
            df, regressors=_stim('risky_prior_sd'), prior_estimate='full',
        )
    if model_label == '12e':
        return RiskRegressionModel(
            df, regressors=_stim('risky_prior_sd', 'safe_prior_sd'),
            prior_estimate='full',
        )

    # Session-1 baselines (no TMS regressor)
    if model_label == 'everyone':
        return RiskModel(df)
    if model_label == 'session1_full':
        return RiskModel(df, prior_estimate='full', fit_seperate_evidence_sd=True)

    # Legacy / dead labels — see legacy_models.py
    try:
        from tms_risk.behavior.legacy_models import build_legacy_model
    except ImportError:
        build_legacy_model = None
    if build_legacy_model is not None:
        legacy = build_legacy_model(model_label, df)
        if legacy is not None:
            return legacy

    raise Exception(f'Do not know model label {model_label}')


def get_data(bids_folder='/data/ds-tmsrisk', model_label=None):

    if model_label is not None and model_label.endswith('everyone'):
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=False, exclude_outliers=True)
        df = df.xs(1, 0, 'session', drop_level=False)
    elif model_label is not None and model_label.startswith('session1'):
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
        df = df.xs(1, 0, 'session', drop_level=False)
    else:
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
        df = df.drop('baseline', level='stimulation_condition')

    df = df.reset_index('stimulation_condition')
    df = df.reset_index('session')
    df['choice'] = df['choice'] == 2.0

    # DDM/RDM likelihoods (WFPT) require t0 < min(rt) per subject. Trials with
    # implausibly short RTs let the sampler wander into the t0 > rt region where
    # the WFPT log-likelihood floors at -66.1 and the gradient w.r.t. t0 is
    # exactly zero, which can trap NUTS at a wrong posterior. Mirror the
    # 0.20 s cutoff used in the bauer lesson 8 tutorial.
    if model_label is not None and (model_label.startswith('ddm_')
                                     or model_label.startswith('rdm_')):
        before = len(df)
        df = df[df['rt'] >= 0.20].copy()
        dropped = before - len(df)
        if dropped:
            print(f'Dropped {dropped} / {before} trials with rt < 0.20s '
                  f'({100 * dropped / before:.1f}%) for DDM/RDM fit.')

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--burnin', type=int, default=None,
                        help='tuning draws (default: per-family recipe)')
    parser.add_argument('--samples', type=int, default=None,
                        help='post-warmup draws (default: per-family recipe)')
    parser.add_argument('--group_sd', default=None,
                        choices=['halfcauchy', 'halfnormal'],
                        help='prior family for every group SD')
    parser.add_argument('--target_accept', type=float, default=None)
    parser.add_argument('--out_folder', default=None,
                        help='derivatives subfolder for the trace '
                             '(default: cogmodels)')
    args = parser.parse_args()
    main(args.model_label, bids_folder=args.bids_folder,
         burnin=args.burnin, samples=args.samples,
         group_sd=args.group_sd, target_accept=args.target_accept,
         out_folder=args.out_folder)
