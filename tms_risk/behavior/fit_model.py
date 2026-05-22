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
    from bauer.models import DDMFlexibleNoiseRiskRegressionModel
except ImportError:
    DDMFlexibleNoiseRiskRegressionModel = None
try:
    from bauer.models import RaceDiffusionFlexibleNoiseRiskRegressionModel
except ImportError:
    RaceDiffusionFlexibleNoiseRiskRegressionModel = None

from tms_risk.utils.data import get_all_behavior


def main(model_label, burnin=None, samples=None, bids_folder='/data/ds-tmsrisk',
         backend=None):

    df = get_data(bids_folder, model_label=model_label)

    target_folder = Path(bids_folder) / 'derivatives' / 'cogmodels'
    target_folder.mkdir(parents=True, exist_ok=True)

    is_accumulator = model_label.startswith('ddm_') or model_label.startswith('rdm_')

    if (model_label.startswith('flexible')
            or is_accumulator
            or model_label.startswith('session1')):
        target_accept = 0.9
    else:
        target_accept = 0.8

    # DDM/RDM fits are slow under pymc's NUTS — bauer's lesson 8 puts them
    # on the numpyro backend, which is 3–10× faster on CPU and parallelises
    # cleanly. Use shorter chains there (1000+1000 is what lesson 8 uses
    # and what passes diagnostics on the Garcia 2022 dataset).
    if is_accumulator:
        burnin = burnin or 1000
        samples = samples or 1000
        backend = backend or 'numpyro'
    else:
        burnin = burnin or 5000
        samples = samples or 5000
        backend = backend or 'pymc'

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
    """Dispatch ddm_flexible_* and rdm_flexible_* labels.

    Suffix legend:
        _null              no TMS regressor (baseline)
        _perception        TMS on perceptual_noise_sd only
        _memory            TMS on memory_noise_sd only
        ''                 TMS on both noise terms (matches Flexible PMC main)
        _threshold         TMS on accumulator threshold (a) only
        _noise_threshold   TMS on both noise terms + threshold

    All use polynomial_order=5 and memory_model='shared_perceptual_noise',
    matching the paper's Flexible PMC model.
    """
    if model_label.startswith('ddm_'):
        if DDMFlexibleNoiseRiskRegressionModel is None:
            raise Exception(
                'DDMFlexibleNoiseRiskRegressionModel is not available — '
                'check libs/bauer is installed with DDM extras (hssm/pymc-ddm).'
            )
        cls = DDMFlexibleNoiseRiskRegressionModel
        kind = 'ddm'
    elif model_label.startswith('rdm_'):
        if RaceDiffusionFlexibleNoiseRiskRegressionModel is None:
            raise Exception('RaceDiffusionFlexibleNoiseRiskRegressionModel unavailable.')
        cls = RaceDiffusionFlexibleNoiseRiskRegressionModel
        kind = 'rdm'
    else:
        raise Exception(f'Not a DDM/RDM label: {model_label!r}')

    suffix = model_label[len(kind) + 1 + len('flexible'):]
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

    return cls(
        df, regressors=regressors,
        prior_estimate='full',
        memory_model='shared_perceptual_noise',
        spline_order=5,
    )


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
