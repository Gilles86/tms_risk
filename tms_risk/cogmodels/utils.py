import pandas as pd
import arviz as az
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def cluster_offers(d, n=6, key='log(risky/safe)'):
    return pd.qcut(d[key], n, duplicates='drop').apply(lambda x: x.mid)


def plot_ppc(df, ppc, plot_type=1, var_name='ll_bernoulli', level='subject', col_wrap=5):

    assert (var_name in ['p', 'll_bernoulli'])

    ppc = ppc.xs(var_name, 0, 'variable').copy()

    df = df.copy()

    # Make sure that we group data from (Roughly) same fractions
    if not (df.groupby(['subject', 'log(risky/safe)']).size().groupby('subject').size() < 7).all():
        df['log(risky/safe)'] = df.groupby(['subject'],
                                        group_keys=False).apply(cluster_offers)

    if level == 'group':
        df['log(risky/safe)'] = df['bin(risky/safe)']
        ppc = ppc.reset_index('log(risky/safe)')
        ppc['log(risky/safe)'] = ppc.index.get_level_values('bin(risky/safe)')

    if plot_type == 1:
        groupby = ['risky_first', 'log(risky/safe)']
    elif plot_type in [2, 4]:
        groupby = ['risky_first', 'n_safe']
    elif plot_type == 3:
        groupby = ['risky_first', 'n_safe', 'log(risky/safe)']
    else:
        raise NotImplementedError

    if level == 'group':
        ppc = ppc.groupby(['subject']+groupby).mean()

    if level == 'subject':
        groupby = ['subject'] + groupby

    ppc_summary = summarize_ppc(ppc, groupby=groupby)
    p = df.groupby(groupby).mean()[['chose_risky']]
    # ppc_summary = pd.concat((p, ppc_summary)).sort_index()
    ppc_summary = ppc_summary.join(p)


    if plot_type in [1, 2]:
        if plot_type == 1:
            x = 'log(risky/safe)'

        if plot_type == 2:
            x = 'n_safe'

        fac = sns.FacetGrid(ppc_summary.reset_index(),
                            col='subject' if level == 'subject' else None,
                            hue='risky_first',
                            col_wrap=col_wrap if level == 'subject' else None)

    elif plot_type == 3:
        x = 'log(risky/safe)'

        fac = sns.FacetGrid(ppc_summary.reset_index(),
                            col='n_safe',
                            hue='risky_first',
                            row='subject' if level == 'subject' else None)
    elif plot_type == 4:


        if level == 'group':
            rnp = df.groupby(['subject'] + groupby, group_keys=False).apply(get_rnp).to_frame('rnp')
            rnp = rnp.groupby(groupby).mean()
        else:
            rnp = df.groupby(groupby, group_keys=False).apply(get_rnp).to_frame('rnp')

        print(ppc_summary)
        print(rnp)
        ppc_summary = ppc_summary.join(rnp)
        print(ppc_summary)
        fac = sns.FacetGrid(ppc_summary.reset_index(),
                            hue='risky_first',
                            col='subject' if level == 'subject' else None,
                            col_wrap=col_wrap if level == 'subject' else None)

        print(ppc_summary)
        fac.map_dataframe(plot_prediction, x='n_safe', y='p_predicted')
        fac.map(plt.scatter, 'n_safe', 'rnp')
        fac.map(lambda *args, **kwargs: plt.axhline(.55, c='k', ls='--'))

    if plot_type in [1,2,3]:
        fac.map_dataframe(plot_prediction, x=x)
        fac.map(plt.scatter, x, 'chose_risky')
        fac.map(lambda *args, **kwargs: plt.axhline(.5, c='k', ls='--'))

    if plot_type in [1, 3]:
        if level == 'subject':
            fac.map(lambda *args, **kwargs: plt.axvline(np.log(1./.55), c='k', ls='--'))
        else:
            fac.map(lambda *args, **kwargs: plt.axvline(3.5, c='k', ls='--'))

    return fac


def plot_prediction(data, x, color, y='p_predicted', alpha=.25, **kwargs):
    data = data[~data['hdi025'].isnull()]

    plt.fill_between(data[x], data['hdi025'],
                     data['hdi975'], color=color, alpha=alpha)
    plt.plot(data[x], data[y], color=color)


def summarize_ppc(ppc, groupby=None):

    if groupby is not None:
        ppc = ppc.groupby(groupby).mean()

    e = ppc.mean(1).to_frame('p_predicted')
    hdi = pd.DataFrame(az.hdi(ppc.T.values), index=ppc.index,
                       columns=['hdi025', 'hdi975'])

    return pd.concat((e, hdi), axis=1)

def get_rnp(d):
    return (1./d['frac']).quantile(d.chose_risky.mean())