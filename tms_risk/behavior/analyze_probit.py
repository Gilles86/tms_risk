import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from fit_probit import build_model, get_data
import arviz as az
from utils import extract_intercept_gamma, get_rnp, plot_ppc, format_bambi_ppc


def main(model_label, bids_folder='/data/ds-tmsrisk', col_wrap=5, only_ppc=False):


    df = get_data()
    model = build_model(model_label, df)

    idata = az.from_netcdf(Path(bids_folder) / 'derivatives' / 'cogmodels' / f'model-{model_label}_trace.netcdf')

    target_folder = Path(bids_folder) / 'derivatives' / 'cogmodels' / 'figures' / model_label
    target_folder.mkdir(parents=True, exist_ok=True)

    intercept_group, gamma_group = extract_intercept_gamma(idata, model, df, True)
    rnp_group = get_rnp(intercept_group, gamma_group)

    gamma_group['Order'] = gamma_group.index.get_level_values('risky_first').map({True:'Risky first', False:'Safe first'})
    rnp_group['Order'] = gamma_group['Order']
    rnp_group.set_index('Order', inplace=True, append=True)
    gamma_group.set_index('Order', inplace=True, append=True)

    intercept, gamma = extract_intercept_gamma(idata, model, df, False)
    rnp = np.clip(get_rnp(intercept, gamma), 0, 1)

    gamma['Order'] = gamma.index.get_level_values('risky_first').map({True:'Risky first', False:'Safe first'})
    rnp['Order'] = gamma['Order']
    rnp.set_index('Order', inplace=True, append=True)
    gamma.set_index('Order', inplace=True, append=True)

    print(f'*** only_ppc: {only_ppc}')
    if not only_ppc:
        if model_label.startswith('probit_simple'):
            # Violin plot of conditions
            gamma_groupfig = sns.catplot(gamma_group.stack().stack().reset_index(), y='gamma', x='stimulation_condition', kind='violin')

            # Distribution of difference
            plt.figure()
            gamma_diff = gamma_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            sns.distplot(gamma_diff[('ips')] - gamma_diff[('vertex')])
            plt.axvline(0.0, c='k', ls='--')
            plt.title((gamma_diff['ips'] - gamma_diff['vertex'] > 0.0).mean())
            plt.title(np.round((gamma_diff['ips'] - gamma_diff['vertex'] > 0.0).mean(), 3))
            plt.savefig(str(target_folder / 'group_gamma_diff.pdf'))
            plt.close()

            rnp_groupfig = sns.catplot(rnp_group.stack().stack().reset_index(), y='rnp', x='stimulation_condition', kind='violin')
            plt.axhline(0.55, c='k', ls='--')

            plt.figure()
            rnp_diff = rnp_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            sns.distplot(rnp_diff[('ips')] - rnp_diff[('vertex')])
            plt.title(np.round((rnp_diff['ips'] - rnp_diff['vertex'] > 0.0).mean(), 3))
            plt.axvline(0.0)
            plt.savefig(str(target_folder / 'group_rnp_diff.pdf'))
            plt.close()

            # Subjectwise parameter plots
            gamma_fig = sns.catplot(gamma.stack([1, 2]).reset_index(), x='subject', y='gamma', hue='stimulation_condition', kind='violin', aspect=3.)
            rnp_fig = sns.catplot(rnp.stack([1, 2]).reset_index(), x='subject', y='rnp', hue='stimulation_condition', kind='violin', aspect=3.)
            plt.axhline(0.55)

        elif model_label.startswith('probit_order'):
            # Violin plot of conditions
            gamma_groupfig = sns.catplot(gamma_group.stack().stack().reset_index(), y='gamma', x='Order', hue='stimulation_condition', kind='violin')

            # Distribution of difference
            plt.figure()
            gamma_diff = gamma_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            gamma_diff = (gamma_diff['ips'] - gamma_diff['vertex']).to_frame('gamma_diff')
            print(gamma_diff)

            fac = sns.FacetGrid(gamma_diff.reset_index(), hue='Order', palette=sns.color_palette()[2:])
            fac.map(sns.distplot, 'gamma_diff')
            fac.map(lambda *args, **kwargs: plt.axvline(0.0, c='k', ls='--'))
            fac.add_legend()
            fac.savefig(op.join(target_folder, 'group_gamma_diff.pdf'))
            plt.close()

            plt.figure()
            rnp_diff = rnp_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            rnp_diff = (rnp_diff['ips'] - rnp_diff['vertex']).to_frame('rnp_diff')

            fac = sns.FacetGrid(rnp_diff.reset_index(), hue='Order', palette=sns.color_palette()[2:])
            fac.map(sns.distplot, 'rnp_diff')
            fac.map(lambda *args, **kwargs: plt.axvline(0.0, c='k', ls='--'))
            fac.add_legend()
            plt.savefig(op.join(target_folder, 'group_rnp_diff.pdf'))
            plt.close()

            rnp_groupfig = sns.catplot(rnp_group.stack().stack().reset_index(), y='rnp',
            x='Order', hue='stimulation_condition', kind='violin')
            plt.axhline(0.55, c='k', ls='--')


            # Subjectwise parameter plots
            gamma_fig = sns.catplot(gamma.stack([1, 2]).reset_index(), x='subject', y='gamma', row='Order', hue='stimulation_condition', kind='violin', aspect=3.)
            rnp_fig = sns.catplot(rnp.stack([1, 2]).reset_index(), x='subject', y='rnp', row='Order', hue='stimulation_condition', kind='violin', aspect=3.)
            rnp_fig.map(lambda *arsg, **kwargs: plt.axhline(0.55, c='k', ls='--'))

            plt.axhline(0.55, c='k', ls='--')

        elif model_label.startswith('probit_full'):
            # Violin plot of conditions
            gamma_groupfig = sns.catplot(gamma_group.stack().stack().reset_index(), y='gamma', x='n_safe',
            col='Order',
            hue='stimulation_condition', kind='violin')

            # Distribution of difference
            plt.figure()
            gamma_diff = gamma_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            gamma_diff = (gamma_diff['ips'] - gamma_diff['vertex']).to_frame('gamma_diff')
            print(gamma_diff)

            fac = sns.FacetGrid(gamma_diff.reset_index(), col='n_safe', hue='Order', palette=sns.color_palette()[2:])
            fac.map(sns.distplot, 'gamma_diff')
            fac.map(lambda *args, **kwargs: plt.axvline(0.0, c='k', ls='--'))
            fac.add_legend()
            fac.savefig(op.join(target_folder, 'group_gamma_diff.pdf'))
            plt.close()

            plt.figure()
            rnp_diff = rnp_group.stack([0, 1, 2]).unstack(['stimulation_condition'])
            rnp_diff = (rnp_diff['ips'] - rnp_diff['vertex']).to_frame('rnp_diff')

            fac = sns.FacetGrid(rnp_diff.reset_index(), hue='Order', col='n_safe', palette=sns.color_palette()[2:])
            fac.map(sns.distplot, 'rnp_diff')
            fac.map(lambda *args, **kwargs: plt.axvline(0.0, c='k', ls='--'))
            fac.add_legend()
            plt.savefig(op.join(target_folder, 'group_rnp_diff.pdf'))
            plt.close()

            rnp_groupfig = sns.catplot(rnp_group.stack().stack().reset_index(), y='rnp',
            x='n_safe', col='Order', hue='stimulation_condition', kind='violin')
            rnp_groupfig.map(lambda *args, **kwargs: plt.axhline(0.55, c='k', ls='--'))


            # Subjectwise parameter plots
            gamma_fig = sns.catplot(gamma.stack([1, 2]).reset_index(), x='subject', y='gamma', row='Order', hue='n_safe', col='stimulation_condition', kind='violin', aspect=3.)
            rnp_fig = sns.catplot(rnp.stack([1, 2]).reset_index(), x='subject', y='rnp', row='Order', hue='n_safe', col='stimulation_condition', kind='violin', aspect=3.)
            rnp_fig.map(lambda *arsg, **kwargs: plt.axhline(0.55, c='k', ls='--'))

            plt.axhline(0.55, c='k', ls='--')
        else:
            raise NotImplementedError


        gamma_groupfig.fig.savefig(op.join(target_folder, 'group_pars_gamma.pdf'))
        rnp_groupfig.fig.savefig(op.join(target_folder, 'group_pars_rnp.pdf'))
        gamma_fig.fig.savefig(op.join(target_folder, 'subject_pars_gamma.pdf'))
        rnp_fig.fig.savefig(op.join(target_folder, 'subject_pars_rnp.pdf'))
        
    # PPC
    for plot_type in [1,2,3, 5, 6, 7, 8, 9]:
        for var_name in ['p', 'll_bernoulli']:
            for level in ['group']:
                target_folder = op.join(bids_folder, 'derivatives', 'cogmodels', 'figures', model_label, var_name)

                if not op.exists(target_folder):
                    os.makedirs(target_folder)

                fn = f'{level}_plot-{plot_type}_model-{model_label}_pred.pdf'
                ppc = format_bambi_ppc(idata, model, df)
                plot_ppc(df, ppc, level=level, plot_type=plot_type, var_name=var_name, col_wrap=col_wrap).savefig(
                    op.join(target_folder, fn))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_label', default=None)
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--only_ppc', action='store_true')
    args = parser.parse_args()

    main(args.model_label, bids_folder=args.bids_folder, only_ppc=args.only_ppc)