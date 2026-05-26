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
         backend=None):

    df = get_data(bids_folder, model_label=model_label)

    target_folder = Path(bids_folder) / 'derivatives' / 'cogmodels'
    target_folder.mkdir(parents=True, exist_ok=True)

    is_accumulator = model_label.startswith('ddm_') or model_label.startswith('rdm_')

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
    else:
        burnin = burnin or 5000
        samples = samples or 5000
        backend = backend or 'pymc'
        target_accept = 0.8

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept,
                         backend=backend)

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
        polynomial_order=polynomial_order,
        memory_model=memory_model,
        prior_estimate='full',
    )


def _build_ddm_or_rdm(model_label, df):
    """Dispatch ddm_*/rdm_* labels (both Weber-noise and Flexible-noise variants).

    Two families, mirroring the paper's two PMC families:

    - **Weber-noise** (analogue of the paper's `11_*` family):
      ``ddm_weber_*`` / ``rdm_weber_*``. Uses
      ``{DDM,RaceDiffusion}RiskRegressionModel`` from bauer.

    - **Flexible-noise** (analogue of the paper's `flexible2_*` family):
      ``ddm_flexible_*`` / ``rdm_flexible_*``. Uses
      ``{DDM,RaceDiffusion}FlexibleNoiseRiskRegressionModel`` with 5
      B-splines on each noise term.

    Suffix legend (same in both families):
        _null              no TMS regressor (baseline)
        _perception        TMS on perceptual_noise_sd only
        _memory            TMS on memory_noise_sd only
        ''                 TMS on both noise terms (paper's main claim analogue)
        _threshold         TMS on accumulator threshold ``a`` only
        _noise_threshold   TMS on both noise terms + threshold
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
    elif rest.startswith('weber'):
        cls = weber_cls
        suffix = rest[len('weber'):]
    else:
        raise Exception(
            f'{kind} label must be {kind}_weber_* or {kind}_flexible_*, '
            f'got {model_label!r}')

    if cls is None:
        raise Exception(
            f'{kind} class for label {model_label!r} is not available — '
            f'check libs/bauer is at a recent commit + the DDM env has hssm.')

    if suffix.startswith('_'):
        suffix = suffix[1:]

    regressors = {}
    if suffix == 'null':
        pass
    elif suffix == 'perception':
        regressors = _stim('perceptual_noise_sd')
    elif suffix == 'memory':
        regressors = _stim('memory_noise_sd')
    elif suffix == '':
        regressors = _stim('perceptual_noise_sd', 'memory_noise_sd')
    elif suffix == 'threshold':
        regressors = _stim('a')
    elif suffix == 'noise_threshold':
        regressors = _stim('perceptual_noise_sd', 'memory_noise_sd', 'a')
    else:
        raise Exception(f'Unrecognised {kind} suffix: {suffix!r}')

    kwargs = dict(prior_estimate='full', memory_model='shared_perceptual_noise')
    if cls is flex_cls:
        kwargs['spline_order'] = 5
    return cls(df, regressors=regressors, **kwargs)


# ---------------------------------------------------------------------------
# Main dispatch
# ---------------------------------------------------------------------------

def build_model(model_label, df):

    # Flexible PMC family — paper's main model
    if model_label.startswith('flexible'):
        return _build_flexible(model_label, df)

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
    args = parser.parse_args()
    main(args.model_label, bids_folder=args.bids_folder)
