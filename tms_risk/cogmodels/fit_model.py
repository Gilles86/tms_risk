import argparse
from bauer.models import RiskRegressionModel, RiskModel, FlexibleSDRiskRegressionModel
from tms_risk.utils.data import get_all_behavior
import os.path as op
import os
import arviz as az
import numpy as np

def main(model_label, burnin=1000, samples=1000, bids_folder='/data/ds-tmsrisk'):

    df = get_data(bids_folder, model_label=model_label)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    if (model_label in ['0', '5', '1_session']) or model_label.startswith("flexible"):
        target_accept = 0.9
    else:
        target_accept = 0.8

    model = build_model(model_label, df)
    model.build_estimation_model()
    trace = model.sample(burnin, samples, target_accept=target_accept)
    az.to_netcdf(trace,
                 op.join(target_folder, f'model-{model_label}_trace.netcdf'))

def build_model(model_label, df):
    if model_label == '1':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition',
         'n2_evidence_sd':'stimulation_condition', 'risky_prior_mu':'stimulation_condition', 'risky_prior_std':'stimulation_condition',
          'safe_prior_mu':'stimulation_condition', 'safe_prior_std':'stimulation_condition'},
         prior_estimate='full')
    elif model_label == '1_null':
        model = RiskRegressionModel(df, regressors={},
         prior_estimate='full')
    elif model_label == '1a':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition', 'n2_evidence_sd':'stimulation_condition',
        'risky_prior_mu':'stimulation_condition'},
         prior_estimate='full')
    elif model_label == '1b':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition', 'n2_evidence_sd':'stimulation_condition'},
         prior_estimate='full')
    elif model_label == '1c':
        model = RiskRegressionModel(df, regressors={'n2_evidence_sd':'stimulation_condition'},
         prior_estimate='full')
    elif model_label == '1_session':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition+session',
         'n2_evidence_sd':'stimulation_condition+session', 'risky_prior_mu':'stimulation_condition+session', 'risky_prior_std':'stimulation_condition+session',
          'safe_prior_mu':'stimulation_condition+session', 'safe_prior_std':'stimulation_condition+session'},
         prior_estimate='full')
    elif model_label == '2':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition',
         'n2_evidence_sd':'stimulation_condition', 'risky_prior_mu':'stimulation_condition', 'risky_prior_std':'stimulation_condition'},
         prior_estimate='different')
    elif model_label == '2a':
        model = RiskRegressionModel(df, regressors={'n2_evidence_sd':'stimulation_condition', 'risky_prior_mu':'stimulation_condition'},
         prior_estimate='different')
    elif model_label == '2_null':
        model = RiskRegressionModel(df, regressors={},
         prior_estimate='different')
    elif model_label == '3':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition',
         'n2_evidence_sd':'stimulation_condition', 'prior_mu':'stimulation_condition', 'prior_std':'stimulation_condition'},
         prior_estimate='shared')
    elif model_label == '3_null':
        model = RiskRegressionModel(df, regressors={},
         prior_estimate='different')    
    elif model_label == 'everyone':
        model = RiskModel(df)
    elif model_label == 'session1_tms':
        model = RiskModel(df, prior_estimate='full')
    elif model_label == '5':
        model = RiskRegressionModel(df, regressors={'perceptual_noise_sd':'stimulation_condition',
         'memory_noise_sd':'stimulation_condition', 'risky_prior_mu':'stimulation_condition', 'risky_prior_std':'stimulation_condition',
          'safe_prior_mu':'stimulation_condition', 'safe_prior_std':'stimulation_condition'},
         prior_estimate='full', memory_model='shared_perceptual_noise')
    elif model_label == '5a':
        model = RiskRegressionModel(df, regressors={'perceptual_noise_sd':'stimulation_condition',},
         prior_estimate='full', memory_model='shared_perceptual_noise')
    elif model_label == '5b':
        model = RiskRegressionModel(df, regressors={'perceptual_noise_sd':'stimulation_condition', 'memory_noise_sd':'stimulation_condition'},
         prior_estimate='full', memory_model='shared_perceptual_noise')
    elif model_label == '5_everyone':
        model = RiskModel(df, memory_model='shared_perceptual_noise')
    elif model_label == '6':
        model = RiskRegressionModel(df, regressors={'evidence_sd':'0 + stimulation_condition01', 
                                                    'risky_prior_mu':'stimulation_condition', 'risky_prior_std':'stimulation_condition',
                                                    'safe_prior_mu':'stimulation_condition', 'safe_prior_std':'stimulation_condition'},
         prior_estimate='full')
    elif model_label == '6a':
        model = RiskRegressionModel(df, regressors={'evidence_sd':'0 + stimulation_condition01'},
         prior_estimate='full')
    elif model_label == 'flexible1':
        model = FlexibleSDRiskRegressionModel(df, regressors={'n1_evidence_sd_poly0':'stimulation_condition',
                                        'n1_evidence_sd_poly1':'stimulation_condition',
                                        'n1_evidence_sd_poly2':'stimulation_condition',
                                        'n1_evidence_sd_poly3':'stimulation_condition',
                                        'n1_evidence_sd_poly4':'stimulation_condition',
                                        'n2_evidence_sd_poly0':'stimulation_condition',
                                        'n2_evidence_sd_poly1':'stimulation_condition',
                                        'n2_evidence_sd_poly2':'stimulation_condition',
                                        'n2_evidence_sd_poly3':'stimulation_condition',
                                        'n2_evidence_sd_poly4':'stimulation_condition',}, bspline=True,
                                        prior_estimate='full')
    elif model_label == 'flexible2':
        model = FlexibleSDRiskRegressionModel(df, regressors={'memory_noise_sd_poly0': 'stimulation_condition',
                                                            'memory_noise_sd_poly1': 'stimulation_condition',
                                                            'memory_noise_sd_poly2': 'stimulation_condition',
                                                            'memory_noise_sd_poly3': 'stimulation_condition',
                                                            'memory_noise_sd_poly4': 'stimulation_condition',
                                                            'perceptual_noise_sd_poly0': 'stimulation_condition',
                                                            'perceptual_noise_sd_poly1': 'stimulation_condition',
                                                            'perceptual_noise_sd_poly2': 'stimulation_condition',
                                                            'perceptual_noise_sd_poly3': 'stimulation_condition',
                                                            'perceptual_noise_sd_poly4': 'stimulation_condition', }, bspline=True,
                                            memory_model='shared_perceptual_noise',
                                            prior_estimate='full')
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(bids_folder='/data/ds-tmsrisk', model_label=None):

    if (model_label is not None) and model_label.endswith('everyone'):
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=False, exclude_outliers=True)
        df = df.xs(1, 0, 'session', drop_level=False)
    elif (model_label is not None) and (model_label == 'session1_tms'):
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
        df = df.xs(1, 0, 'session', drop_level=False)
    else:
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
        df = df.drop('baseline', level='stimulation_condition')
        print('Dropping the baseline condition')

    df = df.reset_index('stimulation_condition')

    df = df.reset_index('session')

    if (model_label is not None) & model_label.startswith('6'):
        df['stimulation_condition01'] = df['stimulation_condition'].map({'vertex':0, 'ips':1}).astype(float)
        print('yo')

    df['choice'] = df['choice'] == 2.0
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)