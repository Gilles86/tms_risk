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


models = { 'model1': 'chose_risky ~ 1 + C(n_safe)*x + x*session + (C(n_safe)*x|subject) + (x*session|subject)'}

def build_model(model, bids_folder='/data/ds-tmsrisk'):
    df = get_all_behavior(bids_folder=bids_folder)
    df = df[~df.choice.isnull()]

    df['x'] = df['log(risky/safe)']
    formula = models[model]

    df = df.loc[:5]


    model_probit = bmb.Model(formula, df[['chose_risky', 'x', 'risky_first', 'n_safe']].reset_index(
    ), family="bernoulli", link='probit')

    return model_probit
    
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


