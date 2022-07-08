import argparse
import arviz as az
import os
import os.path as op
from tms_risk.utils import get_all_behavior
from tqdm.contrib.itertools import product
import pandas as pd
import numpy as np
import bambi as bmb
from scipy.stats import zscore

bids_folder = '/data'


models = { 'model1': 'chose_risky ~ 1 + C(n_safe)*x + x*risky_first + (C(n_safe)*x|subject) + (x*risky_first|subject)',
           'model2':'chose_risky ~ 1 + C(n_safe)*x*risky_first + (C(n_safe)*x*risky_first|subject)'}

def build_model(model, bids_folder='/data/ds-tmsrisk'):
    df = get_all_behavior(bids_folder=bids_folder)
    df = df[~df.choice.isnull()]

    df['x'] = df['log(risky/safe)']
    formula = models[model]

    model_probit = bmb.Model(formula, df[['chose_risky', 'x', 'risky_first', 'n_safe']].reset_index(
    ), family="bernoulli", link='probit')

    return model_probit
def create_test_matrix(df, empirical=True):
    df = df.copy()

    if empirical: 
        df['x'] = df.groupby(['subject'])['x'].apply(lambda d: pd.qcut(d, 6)).apply(lambda x: x.mid, 1)
        
        return df.groupby(['subject', 'risky_first', 'session', 'x', 'n_safe']).size().to_frame('size').reset_index().drop('size', 1)

    else:
        unique_subjects = df.index.unique(level='subject')
        
        risky_safe = np.linspace(np.log(1), np.log(4), 20)
        risky_first = [True, False]
        n_safe = [7, 10, 14, 20, 28]
        session = [1]
        
        d = pd.DataFrame(np.array([e.ravel() for e in np.meshgrid(unique_subjects, risky_safe, risky_first, n_safe, session)]).T,
                        columns=['subject', 'x', 'risky_first', 'n_safe', 'session'])
        
        d['risky_first'] = d['risky_first'].astype(bool)
        d['session'] = d['session'].astype(int)

        return d


def get_predictions(model, trace, bids_folder='/data/ds-tmsrisk',
return_summary_stats=True, thin=5):

    model = build_model(model, bids_folder)
    df = model.data.set_index(['subject', 'session', 'run', 'trial_nr'])

    test_data = create_test_matrix(df)
    test_data.index.name = 'test_values'
    pred = model.predict(trace, 'mean', test_data, inplace=False,)['posterior']['chose_risky_mean'].to_dataframe()
    pred.index = pred.index.set_names('test_values', -1)
    pred = pred.join(test_data).loc[(slice(None), slice(None, None, thin)), :]

    if return_summary_stats:
        m = pred.groupby(['subject', 'x', 'risky_first', 'n_safe'])[['chose_risky_mean']].mean()
        ci = pred.groupby(['subject', 'x', 'risky_first', 'n_safe'])['chose_risky_mean'].apply(lambda x: pd.Series(az.hdi(x.values),
                                                                                                         index=['lower', 'higher'])).unstack()

        m = m.join(ci)
        return m

    else:
        return pred



    
def main(model, bids_folder='/data'):

    model_probit = build_model(model, bids_folder)

    results = model_probit.fit(2000, 2000, target_accept=.85, init='adapt_diag')

    target_dir = op.join(bids_folder, 'derivatives', 'probit_models')

    if not op.exists(target_dir):
        os.makedirs(target_dir)

    az.to_netcdf(results, op.join(target_dir, f'group_model-{model}_behavioralmodel.nc'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', default=None)
    parser.add_argument(
        '--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model, bids_folder=args.bids_folder)
5


