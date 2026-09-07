"""The cognitive model against the paper's own psychophysical analysis.

a  Psychometric functions in the four stake x order cells: observed choice
   proportions with the model's 95% posterior predictive band.
b  The cTBS effect on the two probit parameters, model beside data, per cell.
   Model values are DERIVED from the PMC parameters in closed form
   (`extract_anchor_probit`), not fitted to simulated choices -- so the
   comparison carries no simulation noise and uses the same quantities the
   published Figure 3 reports.

The claim to check is in the bottom-left cell of a and the highlighted row of b:
cTBS flattens the psychometric function and raises the risk-neutral probability,
when the risky option came second and the stakes were low, and not elsewhere.

    python -m tms_risk.behavior.scripts.plot_fig3_ppc_anchor --model_label log-power-n1n2
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

IPS, VERTEX = '#d62728', '#2ca02c'
MODEL, DATA = '#3B5BA5', '0.15'
READ = dict(sep='\t', keep_default_na=False, na_values=[''])
CELLS = [('Risky first', 0), ('Risky second', 0),
         ('Risky first', 1), ('Risky second', 1)]
NAMES = {0: 'Low stake', 1: 'High stake'}

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def logx(ax):
    """The probit is fitted on log(risky/safe), so the published Figure 3A puts
    the ratio on a log axis -- and a constant ABSOLUTE noise change only reads as
    magnitude-specific on that scale. Match it."""
    ax.set_xscale('log')
    ax.set_xticks([1.5, 2, 2.5, 3])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(1.5, 3.25)


def within_subject_points(bids_folder):
    """Observed proportions with CousineauMorey within-subject error bars.

    The band in panel a is a comparison BETWEEN two conditions measured in the
    same people, but a plain between-subject SEM carries the between-subject
    variance that cancels in that comparison -- so it overstates the uncertainty
    on the very thing the panel is about, and the effect looks invisible when it
    is not. Normalising each participant to their own two-condition mean and
    applying the Morey factor sqrt(k/(k-1)) gives the error bar that matches the
    paired test.
    """
    from tms_risk.behavior.fit_model import get_data
    d = get_data(bids_folder, model_label='lfx2-bs3-m2-dp-bm').reset_index()
    d['order'] = d['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    d['y'] = d['chose_risky'].astype(float)
    d['stake'] = (d['n_safe'] + d['n_risky']) / 2
    d['stake2'] = (d.groupby('subject', group_keys=False)['stake']
                   .apply(lambda v: (v > v.median()).astype(int)))
    d['rung'] = (d.groupby(['subject', 'n_safe'], group_keys=False)['frac']
                 .rank(method='dense').astype(int))
    keys = ['subject', 'order', 'stake2', 'rung', 'stimulation_condition']
    cell = d.groupby(keys)['y'].mean().rename('p').reset_index()
    grp = ['subject', 'order', 'stake2', 'rung']
    cell['norm'] = (cell['p'] - cell.groupby(grp)['p'].transform('mean')
                    + cell['p'].mean())
    k = 2
    g = ['order', 'stake2', 'rung', 'stimulation_condition']
    out = cell.groupby(g).agg(observed=('p', 'mean'),
                              wsem=('norm', 'sem'), n=('p', 'size')).reset_index()
    out['wsem'] = out['wsem'] * np.sqrt(k / (k - 1))
    out = out.join(d.groupby(g)['frac'].mean().rename('frac'), on=g)
    return out.rename(columns={'stimulation_condition': 'stim'})


def z(p, eps=.012):
    """Probit transform, clipped. A slope change is a fan of straight lines
    here; on the sigmoid it is a curvature change nobody can see."""
    return norm.ppf(np.clip(p, eps, 1 - eps))


def draws(s):
    return np.array([float(v) for v in s.split(',')])


def derived_effects(f):
    d = pd.read_csv(f, **READ)
    out = {}
    for (o, sb), g in d[d.parameter.isin(['slope', 'logfrac_star'])
                        ].groupby(['order', 'stake2']):
        cur = {}
        for par in ('slope', 'logfrac_star'):
            gg = g[g.parameter == par]
            i = draws(gg[gg.stimulation_condition == 'ips'].draws.iloc[0])
            v = draws(gg[gg.stimulation_condition == 'vertex'].draws.iloc[0])
            n = min(len(i), len(v))
            cur[par] = (i[:n], v[:n])
        # published rnp = exp(b0/b1) = exp(-log frac*), so it RISES with
        # risk seeking -- same convention as analyze_probit_by_stake
        out[(o, int(sb))] = {
            'slope': cur['slope'][0] - cur['slope'][1],
            'rnp': np.exp(-cur['logfrac_star'][0]) - np.exp(-cur['logfrac_star'][1])}
    return out


def observed_effects(f):
    g = pd.read_csv(f, **READ)
    p = g.pivot_table(index=['parameter', 'order', 'stake', 'draw'],
                      columns='stimulation_condition', values='value')
    p['d'] = p['ips'] - p['vertex']
    out = {}
    for (par, o, st), gg in p.groupby(level=['parameter', 'order', 'stake']):
        out[(par, o, 0 if st.startswith('Low') else 1)] = gg['d'].values
    return out


def main(data_dir, out_stem, label, bids_folder, diff=False):
    dd = Path(data_dir)
    ppc = pd.read_csv(dd / f'ppc_anchor/ppc_anchor.stakerung.{label}.tsv', **READ)
    pts = within_subject_points(bids_folder)
    mod = derived_effects(dd / f'probit_derived/probit_derived.{label}.tsv')
    obs = observed_effects(dd / 'probit_stake_group_posterior.tsv')

    fig = plt.figure(figsize=(7.25, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, .9, .9])

    for k, (order, sb) in enumerate(CELLS):
        ax = fig.add_subplot(gs[k % 2, k // 2])
        s = ppc[(ppc.order == order) & (ppc.stake2 == sb)]
        q = pts[(pts.order == order) & (pts.stake2 == sb)]
        if diff:
            # The claim is a DIFFERENCE between two conditions measured in the
            # same people. Plotting levels spends the whole axis on
            # between-subject spread that cancels; the effect is 5 percentage
            # points and will never look large there. Plot the thing itself.
            mi = s[s.stim == 'ips'].sort_values('frac')
            mv = s[s.stim == 'vertex'].sort_values('frac')
            oi = q[q.stim == 'ips'].sort_values('frac')
            ov = q[q.stim == 'vertex'].sort_values('frac')
            dobs = oi.observed.values - ov.observed.values
            # paired SEM: the two normalised SEMs are of the same paired
            # quantity, so combine them rather than treating them as independent
            dsem = np.sqrt(oi['wsem'].values ** 2 + ov['wsem'].values ** 2)
            ax.axhline(0, color='0.8', lw=.7, ls='--', zorder=0)
            ax.plot(mi.frac.values, mi.model.values - mv.model.values,
                    color=MODEL, lw=1.4, zorder=3)
            ax.errorbar(oi.frac.values, dobs, yerr=dsem, fmt='o', ms=3.4,
                        color=DATA, lw=0, elinewidth=.9, capsize=0, zorder=4)
            ax.set_ylim(-.13, .21)
            ax.set_yticks([-.1, 0, .1, .2])
            key = order == 'Risky second' and sb == 0
            ax.set_title(f'{NAMES[sb]} · {order.lower()}', fontsize=7.5, pad=3,
                         color='#8c2d04' if key else '0.15')
            logx(ax)
            if k // 2 == 0:
                ax.set_ylabel('Δ P(chose risky)\nIPS − vertex')
            else:
                ax.set_yticklabels([])
            if k % 2 == 1:
                ax.set_xlabel('Risky/safe payoff ratio')
            else:
                ax.set_xticklabels([])
            continue
        for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
            o = s[s.stim == stim].sort_values('frac')
            ax.fill_between(o.frac, z(o.lo), z(o.hi), color=col, alpha=.18, lw=0)
            ax.plot(o.frac, z(o.model), color=col, lw=1.2)
            e = q[q.stim == stim].sort_values('frac')
            ax.errorbar(e.frac, z(e.observed),
                        yerr=[z(e.observed) - z(e.observed - e['wsem']),
                              z(e.observed + e['wsem']) - z(e.observed)],
                        fmt='o', ms=3.2, color=col, lw=0, elinewidth=.9,
                        capsize=0, zorder=4)
        ax.axhline(0, color='0.88', lw=.6, ls='--', zorder=0)
        ax.set_ylim(-1.65, 1.85)
        ax.set_yticks(z(np.array([.1, .3, .5, .7, .9])))
        ax.set_yticklabels(['.1', '.3', '.5', '.7', '.9'])
        logx(ax)
        key = order == 'Risky second' and sb == 0
        ax.set_title(f'{NAMES[sb]} · {order.lower()}', fontsize=7.5, pad=3,
                     color='#8c2d04' if key else '0.15')
        if k // 2 == 0:
            ax.set_ylabel('P(chose risky), probit axis')
        else:
            ax.set_yticklabels([])
        if k % 2 == 1:
            ax.set_xlabel('Risky/safe payoff ratio')
        else:
            ax.set_xticklabels([])
    fig.text(.005, .985, 'a', fontsize=8, fontweight='bold', family='Arial',
             va='top')
    ax0 = fig.axes[0]
    if diff:
        ax0.text(.04, .95, 'Data ± paired SEM', color=DATA,
                 transform=ax0.transAxes, va='top', fontsize=6.8)
        ax0.text(.04, .84, 'Model', color=MODEL, transform=ax0.transAxes,
                 va='top', fontsize=6.8)
    else:
        ax0.text(.05, .96, 'IPS', color=IPS, transform=ax0.transAxes, va='top',
                 fontsize=7)
        ax0.text(.05, .83, 'Vertex', color=VERTEX, transform=ax0.transAxes,
                 va='top', fontsize=7)

    # -- b: the two probit parameters, model beside data ------------------
    y = np.arange(len(CELLS))
    for j, (par, lab) in enumerate([('slope', 'Δ probit slope\n(consistency)'),
                                    ('rnp', 'Δ risk-neutral probability\n(risk attitude)')]):
        ax = fig.add_subplot(gs[:, 2 + j])
        ax.axvline(0, color='0.75', lw=.7, ls='--', zorder=0)
        for i, (order, sb) in enumerate(CELLS):
            for off, src, col in [(-.17, mod[(order, sb)][par], MODEL),
                                  (.17, obs[(par, order, sb)], DATA)]:
                lo, md, hi = np.quantile(src, [.025, .5, .975])
                ax.plot([lo, hi], [y[i] + off] * 2, color=col, lw=1.3,
                        solid_capstyle='butt', zorder=2)
                ax.plot(md, y[i] + off, 'o', ms=4, color=col, zorder=3)
            if order == 'Risky second' and sb == 0:
                ax.axhspan(y[i] - .45, y[i] + .45, color='#fdf0e3', zorder=0)
        ax.set_yticks(y)
        ax.set_yticklabels([f'{NAMES[sb]}\n{o.lower()}' for o, sb in CELLS],
                           fontsize=6.2)
        if j == 1:
            ax.set_yticklabels([])
        ax.set_ylim(-.6, len(CELLS) - .4)
        ax.set_xlabel(lab, fontsize=7)
        ax.invert_yaxis()
        if j == 0:
            ax.text(.03, .04, 'Model', color=MODEL, transform=ax.transAxes,
                    fontsize=7)
            ax.text(.03, .11, 'Data', color=DATA, transform=ax.transAxes,
                    fontsize=7)
    fig.text(.615, .985, 'b', fontsize=8, fontweight='bold', family='Arial',
             va='top')
    fig.suptitle(f'{label} · a: probit axis; points data ± within-subject SEM, '
                 f'bands 95% PPI.   '
                 f'b: model derived in closed form from the PMC parameters; '
                 f'data from the hierarchical probit.  Bars 95% CrI.',
                 fontsize=6.4, color='0.4', y=1.045)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/fig3_ppc_anchor'))
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--diff', action='store_true',
                    help='panel a shows the cTBS DIFFERENCE rather than levels')
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label, a.bids_folder, a.diff)
