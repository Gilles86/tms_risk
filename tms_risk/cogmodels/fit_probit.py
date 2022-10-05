import os
import os.path as op
import arviz as az
from tms_risk.utils.data import get_all_behavior
import argparse
import bambi


def main(model_label=1, bids_folder='/data/ds-tmsrisk'):

    model = build_model(model_label, bids_folder)

    idata = model.fit(init='adapt_diag',
    target_accept=0.9, draws=500, tune=500)

    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels')

    if not op.exists(target_folder):
        os.makedirs(target_folder)

    az.to_netcdf(idata,
                 op.join(target_folder, f'model-probit{model_label}_trace.netcdf'))

def build_model(model_label, df, bids_folder='/data/ds-tmsrisk'):

    model_label = int(model_label)

    df = get_all_behavior(bids_folder=bids_folder)
    df = df.xs(1, 0, 'session')

    df['x'] = df['log(risky/safe)']
    df['chose_risky'] = df['chose_risky'].astype(bool)

    df = df.reset_index()


    if model_label == 1:
        formula = 'chose_risky ~ x*C(risky_first)*C(n_safe) + (x*C(risky_first)*C(n_safe)|subject)'
    elif model_label == 2:
        formula = 'chose_risky ~ x + (x|subject)'
    elif model_label == 3:
        formula = 'chose_risky ~ x*C(risky_first) + (x*C(risky_first)|subject)'

    return bambi.Model(formula, data=df, link='probit', family='bernoulli')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder)
