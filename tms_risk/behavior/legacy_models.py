"""Legacy model-label dispatch.

Kept as a graveyard for the long elif chain that used to live in
``fit_model.py``. No live notebook (anything outside
``behavior/notebooks/archive/``) references these labels — they are
exploratory variants that never made it into the paper. Removed from
``fit_model.py``'s main dispatch on the cleanup/ddm-port branch to make
the live model surface readable.

``build_legacy_model(label, df)`` is called as a last-resort fallback
from ``fit_model.build_model`` before it raises. If you need to resurrect
one of these, prefer copying the constructor back into ``fit_model.py``
with a fresh label rather than relying on this fallback.

One-line summary of each retired label:

- ``1``         RiskRegressionModel, TMS on every noise + prior parameter
- ``1_null``    same family, no TMS regressor
- ``1a``        TMS on n1/n2 noise + risky_prior_mu only
- ``1b``        TMS on n1/n2 noise only
- ``1_session`` 1 with extra session regressor
- ``2``         RiskRegressionModel, prior_estimate='different'
- ``2a``        2 with reduced regressor set
- ``2_null``    2 with no regressor
- ``3``         RiskRegressionModel, prior_estimate='shared'
- ``3_null``    3 with no regressor
- ``5``         Earlier Weber PMC variant (superseded by 11a)
- ``5a/b/c``    Ablations of 5
- ``5_everyone`` 5 fit pooled
- ``6/6a/6b``   stimulation_condition01 0/1-coded variant
- ``7``         interaction with risky_first
- ``20``        PsychometricRegressionModel — see fit_probit.py for the
                ``probit_*`` labels that are actually used
- ``21``        Psychometric with risky_first*stim interaction
- ``session1_*`` various session-1-only fits (full kept in fit_model.py)
"""
from bauer.models import (
    RiskModel,
    RiskRegressionModel,
    PsychometricRegressionModel,
)


def build_legacy_model(model_label, df):
    """Return a model for retired labels, or None if the label is unknown."""

    if model_label == '1':
        return RiskRegressionModel(
            df,
            regressors={
                'n1_evidence_sd': 'stimulation_condition',
                'n2_evidence_sd': 'stimulation_condition',
                'risky_prior_mu': 'stimulation_condition',
                'risky_prior_sd': 'stimulation_condition',
                'safe_prior_mu': 'stimulation_condition',
                'safe_prior_sd': 'stimulation_condition',
            },
            prior_estimate='full',
        )
    if model_label == '1_null':
        return RiskRegressionModel(df, regressors={}, prior_estimate='full')
    if model_label == '1a':
        return RiskRegressionModel(
            df,
            regressors={
                'n1_evidence_sd': 'stimulation_condition',
                'n2_evidence_sd': 'stimulation_condition',
                'risky_prior_mu': 'stimulation_condition',
            },
            prior_estimate='full',
        )
    if model_label == '1b':
        return RiskRegressionModel(
            df,
            regressors={
                'n1_evidence_sd': 'stimulation_condition',
                'n2_evidence_sd': 'stimulation_condition',
            },
            prior_estimate='full',
        )
    if model_label == '1_session':
        return RiskRegressionModel(
            df,
            regressors={
                'n1_evidence_sd': 'stimulation_condition+session',
                'n2_evidence_sd': 'stimulation_condition+session',
                'risky_prior_mu': 'stimulation_condition+session',
                'risky_prior_std': 'stimulation_condition+session',
                'safe_prior_mu': 'stimulation_condition+session',
                'safe_prior_std': 'stimulation_condition+session',
            },
            prior_estimate='full',
        )

    if model_label == '2':
        return RiskRegressionModel(
            df,
            regressors={
                'n1_evidence_sd': 'stimulation_condition',
                'n2_evidence_sd': 'stimulation_condition',
                'risky_prior_mu': 'stimulation_condition',
                'risky_prior_std': 'stimulation_condition',
            },
            prior_estimate='different',
        )
    if model_label == '2a':
        return RiskRegressionModel(
            df,
            regressors={
                'n2_evidence_sd': 'stimulation_condition',
                'risky_prior_mu': 'stimulation_condition',
            },
            prior_estimate='different',
        )
    if model_label == '2_null':
        return RiskRegressionModel(df, regressors={}, prior_estimate='different')

    if model_label == '3':
        return RiskRegressionModel(
            df,
            regressors={
                'n1_evidence_sd': 'stimulation_condition',
                'n2_evidence_sd': 'stimulation_condition',
                'prior_mu': 'stimulation_condition',
                'prior_std': 'stimulation_condition',
            },
            prior_estimate='shared',
        )
    if model_label == '3_null':
        return RiskRegressionModel(df, regressors={}, prior_estimate='different')

    if model_label == '5':
        return RiskRegressionModel(
            df,
            regressors={
                'perceptual_noise_sd': 'stimulation_condition',
                'memory_noise_sd': 'stimulation_condition',
                'risky_prior_mu': 'stimulation_condition',
                'risky_prior_std': 'stimulation_condition',
                'safe_prior_mu': 'stimulation_condition',
                'safe_prior_std': 'stimulation_condition',
            },
            prior_estimate='full', memory_model='shared_perceptual_noise',
        )
    if model_label == '5a':
        return RiskRegressionModel(
            df,
            regressors={'perceptual_noise_sd': 'stimulation_condition'},
            prior_estimate='full', memory_model='shared_perceptual_noise',
        )
    if model_label == '5b':
        return RiskRegressionModel(
            df,
            regressors={
                'perceptual_noise_sd': 'stimulation_condition',
                'memory_noise_sd': 'stimulation_condition',
            },
            prior_estimate='full', memory_model='shared_perceptual_noise',
        )
    if model_label == '5c':
        return RiskRegressionModel(
            df,
            regressors={
                'perceptual_noise_sd': 'stimulation_condition*risky_first',
                'memory_noise_sd': 'stimulation_condition*risky_first',
            },
            prior_estimate='shared', memory_model='shared_perceptual_noise',
        )
    if model_label == '5_everyone':
        return RiskModel(df, memory_model='shared_perceptual_noise')

    if model_label == '6':
        return RiskRegressionModel(
            df,
            regressors={
                'evidence_sd': '0 + stimulation_condition01',
                'risky_prior_mu': 'stimulation_condition',
                'risky_prior_std': 'stimulation_condition',
                'safe_prior_mu': 'stimulation_condition',
                'safe_prior_std': 'stimulation_condition',
            },
            prior_estimate='full',
        )
    if model_label == '6a':
        return RiskRegressionModel(
            df, regressors={'evidence_sd': '0 + stimulation_condition01'},
            prior_estimate='full',
        )
    if model_label == '6b':
        return RiskRegressionModel(
            df, regressors={'evidence_sd': '0 + stimulation_condition'},
            prior_estimate='full',
        )

    if model_label == '7':
        return RiskRegressionModel(
            df,
            regressors={
                'n1_evidence_sd': 'risky_first*stimulation_condition',
                'n2_evidence_sd': 'risky_first*stimulation_condition',
            },
            prior_estimate='shared',
        )

    if model_label == 'session1_tms':
        return RiskModel(df, prior_estimate='full')
    if model_label == 'session1_different_evidence':
        return RiskModel(df, prior_estimate='shared', fit_seperate_evidence_sd=True)
    if model_label == 'session1_different_priors':
        return RiskModel(df, prior_estimate='full', fit_seperate_evidence_sd=False)
    if model_label == 'session1_simple':
        return RiskModel(df, prior_estimate='shared', fit_seperate_evidence_sd=False)

    if model_label == '20':
        return PsychometricRegressionModel(
            df,
            regressors={
                'bias': 'stimulation_condition',
                'nu': 'stimulation_condition',
            },
        )
    if model_label == '21':
        return PsychometricRegressionModel(
            df,
            regressors={
                'bias': 'stimulation_condition*risky_first',
                'nu': 'stimulation_condition*risky_first',
            },
        )

    return None
