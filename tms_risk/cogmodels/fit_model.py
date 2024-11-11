import argparse
from bauer.models import RiskRegressionModel, RiskModel, FlexibleNoiseRiskRegressionModel, PsychometricRegressionModel #FlexibleSDRiskRegressionModel, 
from tms_risk.utils.data import get_all_behavior
import os.path as op
import os
import arviz as az
import numpy as np

def main(model_label, burnin=5000, samples=5000, bids_folder='/data/ds-tmsrisk'):

    df = get_data(bids_folder, model_label=model_label)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    if (model_label in ['0', '5', '1_session']) or model_label.startswith("flexible") or model_label.startswith('session1'):
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
    elif model_label == '5c':
        model = RiskRegressionModel(df, regressors={'perceptual_noise_sd':'stimulation_condition*risky_first', 'memory_noise_sd':'stimulation_condition*risky_first'},
         prior_estimate='shared', memory_model='shared_perceptual_noise')
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
    elif model_label == '6b':
        model = RiskRegressionModel(df, regressors={'evidence_sd':'0 + stimulation_condition'},
         prior_estimate='full')
    elif model_label == '7':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'risky_first*stimulation_condition', 
                                                    'n2_evidence_sd':'risky_first*stimulation_condition',},
         prior_estimate='shared')
    elif model_label == 'flexible1':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n1_evidence_sd': 'stimulation_condition',
                                        'n2_evidence_sd': 'stimulation_condition'}, 
                                        prior_estimate='full')

    elif model_label == 'flexible1_null':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={}, prior_estimate='full')

    elif model_label == 'flexible1a':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n1_evidence_sd': 'stimulation_condition'}, 
                                        prior_estimate='full')

    elif model_label == 'flexible1b':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n2_evidence_sd': 'stimulation_condition'}, 
                                        prior_estimate='full')

    elif model_label == 'flexible1.4':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n1_evidence_sd': 'stimulation_condition',
                                        'n2_evidence_sd': 'stimulation_condition'}, 
                                        polynomial_order=4,
                                        prior_estimate='full')

    elif model_label == 'flexible1.4_null':
        model = FlexibleNoiseRiskRegressionModel(df, polynomial_order=4, regressors={}, prior_estimate='full')

    elif model_label == 'flexible1.4a':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n1_evidence_sd': 'stimulation_condition'}, 
                                        polynomial_order=4,
                                        prior_estimate='full')

    elif model_label == 'flexible1.4b':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n2_evidence_sd': 'stimulation_condition'}, 
                                        polynomial_order=4,
                                        prior_estimate='full')

    elif model_label == 'flexible2':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'memory_noise_sd': 'stimulation_condition',
                                        'perceptual_noise_sd': 'stimulation_condition'}, 
                                        memory_model='shared_perceptual_noise',
                                        prior_estimate='full')

    elif model_label == 'flexible2a':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'memory_noise_sd': 'stimulation_condition'}, 
                                        memory_model='shared_perceptual_noise',
                                        prior_estimate='full')

    elif model_label == 'flexible2b':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'perceptual_noise_sd': 'stimulation_condition'}, 
                                        memory_model='shared_perceptual_noise',
                                        prior_estimate='full')

    elif model_label == 'flexible2_null':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={}, 
                                        memory_model='shared_perceptual_noise', 
                                        prior_estimate='full')

    elif model_label == 'flexible2.4':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'memory_noise_sd': 'stimulation_condition',
                                        'perceptual_noise_sd': 'stimulation_condition'},  
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=4,
                                        prior_estimate='full')

    elif model_label == 'flexible2.4_null':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={},  
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=4,
                                        prior_estimate='full')

    elif model_label == 'flexible2.4a':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'memory_noise_sd': 'stimulation_condition'}, 
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=4,
                                        prior_estimate='full')

    elif model_label == 'flexible2.4b':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'perceptual_noise_sd': 'stimulation_condition'}, 
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=4,
                                        prior_estimate='full')

    elif model_label == 'flexible1.6':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n1_evidence_sd': 'stimulation_condition',
                                        'n2_evidence_sd': 'stimulation_condition'}, 
                                        polynomial_order=6,
                                        prior_estimate='full')

    elif model_label == 'flexible1.6_null':
        model = FlexibleNoiseRiskRegressionModel(df, polynomial_order=6, regressors={}, prior_estimate='full')

    elif model_label == 'flexible1.6a':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n1_evidence_sd': 'stimulation_condition'}, 
                                        polynomial_order=6,
                                        prior_estimate='full')

    elif model_label == 'flexible1.6b':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'n2_evidence_sd': 'stimulation_condition'}, 
                                        polynomial_order=6,
                                        prior_estimate='full')

    elif model_label == 'flexible2.6':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'memory_noise_sd': 'stimulation_condition',
                                        'perceptual_noise_sd': 'stimulation_condition'},  
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=6,
                                        prior_estimate='full')

    elif model_label == 'flexible2.6_null':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={},  
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=6,
                                        prior_estimate='full')

    elif model_label == 'flexible2.6a':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'memory_noise_sd': 'stimulation_condition'}, 
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=6,
                                        prior_estimate='full')

    elif model_label == 'flexible2.6b':
        model = FlexibleNoiseRiskRegressionModel(df, regressors={
                                        'perceptual_noise_sd': 'stimulation_condition'}, 
                                        memory_model='shared_perceptual_noise',
                                        polynomial_order=6,
                                        prior_estimate='full')
    elif model_label == 'session1_full':
        model = RiskModel(df, prior_estimate='full', fit_seperate_evidence_sd=True)
    elif model_label == 'session1_different_evidence':
        model = RiskModel(df, prior_estimate='shared', fit_seperate_evidence_sd=True)
    elif model_label == 'session1_different_priors':
        model = RiskModel(df, prior_estimate='full', fit_seperate_evidence_sd=False)
    elif model_label == 'session1_simple':
        model = RiskModel(df, prior_estimate='shared', fit_seperate_evidence_sd=False)
    elif model_label == '10_null':
        model = RiskRegressionModel(df, regressors={},
                                    prior_estimate='full')
    elif model_label == '10a':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition', 'n2_evidence_sd':'stimulation_condition', },
                                    prior_estimate='full')
    elif model_label == '10b':
        model = RiskRegressionModel(df, regressors={'n1_evidence_sd':'stimulation_condition'},
                                    prior_estimate='full')
    elif model_label == '10c':
        model = RiskRegressionModel(df, regressors={'n2_evidence_sd':'stimulation_condition'},
                                    prior_estimate='full')
    elif model_label == '11_null':
        model = RiskRegressionModel(df, regressors={},
                                    memory_model='shared_perceptual_noise',
                                    prior_estimate='full')
    elif model_label == '11a':
        model = RiskRegressionModel(df, regressors={'memory_noise_sd':'stimulation_condition',
                                                    'perceptual_noise_sd':'stimulation_condition'},
                                    memory_model='shared_perceptual_noise',
                                    prior_estimate='full')
    elif model_label == '11b':
        model = RiskRegressionModel(df, regressors={'memory_noise_sd':'stimulation_condition'},
                                    memory_model='shared_perceptual_noise',
                                    prior_estimate='full')
    elif model_label == '11c':
        model = RiskRegressionModel(df, regressors={'perceptual_noise_sd':'stimulation_condition'},
                                    memory_model='shared_perceptual_noise',
                                    prior_estimate='full')

    elif model_label == '12a':
        model = RiskRegressionModel(df, regressors={'risky_prior_mu':'stimulation_condition', 'safe_prior_mu':'stimulation_condition', },
                                    prior_estimate='full')
    elif model_label == '12b':
        model = RiskRegressionModel(df, regressors={'risky_prior_mu':'stimulation_condition'},
                                    prior_estimate='full')
    elif model_label == '12c':
        model = RiskRegressionModel(df, regressors={'safe_prior_mu':'stimulation_condition'},
                                    prior_estimate='full')
    elif model_label == '12d':
        model = RiskRegressionModel(df, regressors={'risky_prior_std':'stimulation_condition', },
                                    prior_estimate='full')
    elif model_label == '12e':
        model = RiskRegressionModel(df, regressors={'safe_prior_std':'stimulation_condition', },
                                    prior_estimate='full')
    elif model_label == '12e':
        model = RiskRegressionModel(df, regressors={'risky_prior_std':'stimulation_condition', 'safe_prior_std':'stimulation_condition', },
                                    prior_estimate='full')
    elif model_label == '20':
        model = PsychometricRegressionModel(df, regressors={'bias':'stimulation_condition', 'nu':'stimulation_condition'},)
    elif model_label == '21':
        model = PsychometricRegressionModel(df, regressors={'bias':'stimulation_condition*risky_first', 'nu':'stimulation_condition*risky_first'},)
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(bids_folder='/data/ds-tmsrisk', model_label=None):

    if (model_label is not None) and model_label.endswith('everyone'):
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=False, exclude_outliers=True)
        df = df.xs(1, 0, 'session', drop_level=False)
    elif (model_label is not None) and model_label.startswith('session1'):
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
        df = df.xs(1, 0, 'session', drop_level=False)
    else:
        df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True, exclude_outliers=True)
        df = df.drop('baseline', level='stimulation_condition')

    df = df.reset_index('stimulation_condition')

    df = df.reset_index('session')

    if (model_label is not None) & model_label.startswith('6'):
        df['stimulation_condition01'] = df['stimulation_condition'].map({'vertex':0, 'ips':1}).astype(float)
        print('yo')

    if model_label.startswith('2'):
        df['x1'] = np.log(df['n_safe'])
        df['x2'] = np.log(df['n_risky'])
        df['choice'] = df['chose_risky']
    else:
        df['choice'] = df['choice'] == 2.0

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)