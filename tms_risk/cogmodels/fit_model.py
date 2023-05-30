import argparse
from bauer.models import RiskRegressionModel
from tms_risk.utils.data import get_all_behavior
import os.path as op
import os
import arviz as az
import numpy as np

def main(model_label, burnin=1000, samples=1000, bids_folder='/data/ds-tmsrisk'):

    df = get_data(bids_folder)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    if model_label in ['0', '5']:
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
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition'},
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
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(bids_folder='/data/ds-tmsrisk'):

    df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
    df = df.drop('baseline', level='stimulation_condition')
    print('Dropping the baseline condition')
    df = df.reset_index('stimulation_condition')
    df['choice'] = df['choice'] == 2.0
    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)



