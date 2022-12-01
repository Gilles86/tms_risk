import argparse
from tms_risk.utils.data import get_all_behavior
import os.path as op
import os
import arviz as az
import bambi

def main(model_label, burnin=1000, samples=1000, bids_folder='/data/ds-tmsrisk'):

    df = get_data(model_label, bids_folder)
    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    target_accept = 0.9

    model = build_model(model_label, df)
    trace = model.fit(burnin, samples, init='adapt_diag', target_accept=target_accept)
    az.to_netcdf(trace,
                 op.join(target_folder, f'model-{model_label}_trace.netcdf'))

def build_model(model_label, df):
    if model_label == 'probit_simple':
        model = bambi.Model('chose_risky ~ x*stimulation_condition + (x*stimulation_condition|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_simple_fixed':
        model = bambi.Model('chose_risky ~ x*stimulation_condition + (x|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_simple_fixed0':
        model = bambi.Model('chose_risky ~ 0+x*stimulation_condition + (x|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_simple_session':
        model = bambi.Model('chose_risky ~ x*stimulation_condition*session3 + (x*stimulation_condition|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_order':
        model = bambi.Model('chose_risky ~ x*risky_first*stimulation_condition + (x*risky_first*stimulation_condition|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_order_fixed0':
        model = bambi.Model('chose_risky ~ 0+x*risky_first*stimulation_condition + (x*risky_first|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_order_session':
        model = bambi.Model('chose_risky ~ x*risky_first*stimulation_condition*C(session) + (x*risky_first*stimulation_condition|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_full':
        model = bambi.Model('chose_risky ~ x*risky_first*stimulation_condition*C(n_safe) + (x*risky_first*stimulation_condition+C(n_safe)|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_full_fixed':
        model = bambi.Model('chose_risky ~ x*risky_first*stimulation_condition*C(n_safe) + (x*risky_first*C(n_safe)|subject)', df.reset_index(), link='probit', family='bernoulli')
    elif model_label == 'probit_full_fixed2':
        model = bambi.Model('chose_risky ~ x*risky_first*stimulation_condition*C(n_safe) + (x|subject)', df.reset_index(), link='probit', family='bernoulli')
    else:
        raise Exception(f'Do not know model label {model_label}')

    return model

def get_data(model_label=None, bids_folder='/data/ds-tmsrisk'):
    df = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True)
    df['x'] = df['log(risky/safe)']
    df['session3'] = (df.index.get_level_values('session') == 3).astype(int)

    df = df.drop('baseline', level='stimulation_condition')
    print('Dropping the baseline condition')

    df = df.reset_index('stimulation_condition')

    return df

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)



