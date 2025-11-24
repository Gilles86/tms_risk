from itertools import combinations
import pingouin
import matplotlib.pyplot as plt
import seaborn
import os
from pathlib import Path
import argparse
import arviz as az
from fit_model import get_data, build_model
import numpy as np
import seaborn as sns
from bauer.utils.bayes import softplus
from utils import plot_ppc

def main(model_label, bids_folder='/data/ds-tmsrisk', col_wrap=5, only_ppc=False,
plot_traces=False):


    df = get_data(model_label=model_label, bids_folder=bids_folder)
    model = build_model(model_label, df)
    model.build_estimation_model()

    idata = az.from_netcdf(Path(bids_folder) / 'derivatives' / 'cogmodels' / f'model-{model_label}_trace.netcdf')

    target_folder = Path(bids_folder) / 'derivatives' / 'cogmodels' / 'figures' / model_label
    target_folder.mkdir(parents=True, exist_ok=True)

    if plot_traces:
        az.plot_trace(idata, var_names=['~p'])
        plt.savefig(str(target_folder / 'traces.pdf'))


    for par in model.free_parameters:
        traces = idata.posterior[par+'_mu'].to_dataframe()


        for regressor, t in traces.groupby(par+'_regressors'):
            t = t.copy()
            print(regressor, t)
            if (par in ['prior_std', 'risky_prior_std', 'safe_prior_std', 'n1_evidence_sd', 'n2_evidence_sd', 'evidence_sd']) & (regressor == 'Intercept'):
                t = softplus(t)

            plt.figure()
            sns.kdeplot(t, fill=True)
            if regressor != 'Intercept':
                plt.axvline(0.0, c='k', ls='--')
                txt = f'p({par} < 0.0) = {np.round((t.values < 0.0).mean(), 3)}'
                plt.xlabel(txt)

            else:
                if par == 'risky_prior_mu':
                    plt.axvline(np.log(df['n_risky']).mean(), c='k', ls='--')
                elif par == 'risky_prior_std':
                    plt.axvline(np.log(df['n_risky']).std(), c='k', ls='--')
                elif par == 'safe_prior_mu':
                    for n_safe in np.log([7., 10., 14., 20., 28.]):
                        plt.axvline(n_safe, c='k', ls='--')

                    plt.axvline(np.log(df['n_safe']).mean(), c='k', ls='--', lw=2)
                elif par == 'safe_prior_std':
                    plt.axvline(np.log(df['n_safe']).std(), c='k', ls='--')

            plt.savefig(str(target_folder / f'group_par-{par}.{regressor}.pdf'))
            plt.close()

        


    free_tms_parameters = ['n1_evidence_sd', 'n2_evidence_sd', 'risky_prior_mu']
    for par1, par2 in combinations(free_tms_parameters, 2):
        if ('stimulation_condition' in model.regressors[par1]) & ('stimulation_condition' in model.regressors[par2]):
            trace1 = idata.posterior[par1].to_dataframe().xs('stimulation_condition[T.vertex]', 0, f'{par1}_regressors')
            trace2 = idata.posterior[par2].to_dataframe().xs('stimulation_condition[T.vertex]', 0, f'{par2}_regressors')

            par1_subjectwise = trace1.groupby('subject').mean()
            par2_subjectwise = trace2.groupby('subject').mean()
            pars_subjectwise = par1_subjectwise.join(par2_subjectwise)
            g = sns.lmplot(data=pars_subjectwise, x=par1,  y=par2)
            r = pingouin.corr(pars_subjectwise[par1], pars_subjectwise[par2])
            print(r)
        # g.set_titles(f'r={r["r"].iloc[0]:0.2f}, p={r["p"].iloc[0]:0.4f}')
            g.savefig(op.join(target_folder, f'group-corrtmseffect_{par1}_{par2}.pdf'))


    ppc = model.ppc(trace=idata.sel(draw=slice(None, None, 10)), data=df)

    # "Chose risky" vs "chose 2nd option coding"
    ppc.loc[ppc.index.get_level_values('risky_first')] = 1 - ppc.loc[ppc.index.get_level_values('risky_first')]

    for plot_type in [1,2,3, 5, 6, 7, 8, 9]:
        for var_name in ['p', 'll_bernoulli']:
            for level in ['group']:
                try:
                    target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'figures', model_label, var_name)

                    if not op.exists(target_folder):
                        os.makedirs(target_folder)

                    fn = f'{level}_plot-{plot_type}_model-{model_label}_pred.pdf'
                    plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap).savefig(
                        op.join(target_folder, fn))
                except Exception as e:
                    print(f'Could not plot {level} {plot_type} {var_name}: {e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--only_ppc', action='store_true')
    parser.add_argument('--no_trace', dest='plot_traces', action='store_false')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, only_ppc=args.only_ppc, plot_traces=args.plot_traces)