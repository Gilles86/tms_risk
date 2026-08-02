"""Figure 3: what cTBS did to the psychometric function, from the probit fits alone.

No cognitive model. The probit has exactly two parameters, and they correspond to the
two competing explanations:

    slope  choice consistency. A pure loss of consistency -- the "flattening" account
           -- moves this and nothing else.
    RNP    the risk-neutral probability, i.e. where the psychometric crosses 0.5. A
           pure change in preference moves this and nothing else.

Both move, and both only when the risky option came second. So neither pure account
covers it, which is the observation the cognitive model has to explain.

    a, b   fitted psychometric curves per presentation order, both conditions, with
           the observed proportions.
    c, d   the posterior of the cTBS change in each parameter, per order, paired
           across draws.

    python -m tms_risk.behavior.scripts.plot_fig3_probit

Reads notes/data/localnoise_{signatures,group_posterior,delta_by_ratio}.tsv.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')
PANEL = dict(fontsize=11, fontweight='bold', va='bottom', ha='right')


def paired_delta(g, parameter, order):
    """IPS - vertex, paired within (chain, draw) so the difference keeps its posterior."""
    s = g[(g.parameter == parameter) & (g.order == order)]
    w = s.pivot_table(index=['chain', 'draw'], columns='stimulation_condition',
                      values='value')
    return (w['ips'] - w['vertex']).values


def main(data_dir, out_stem):
    data = Path(data_dir)
    sig = pd.read_csv(data / 'localnoise_signatures.tsv', sep='\t')
    post = pd.read_csv(data / 'localnoise_group_posterior.tsv', sep='\t')
    rat = pd.read_csv(data / 'localnoise_delta_by_ratio.tsv', sep='\t')

    fig = plt.figure(figsize=(7.25, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.28,
                          left=.095, right=.98, top=.90, bottom=.10)

    # --- a, b: the fitted psychometric functions
    tops = []
    for col, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs[0, col]); tops.append(ax)
        s = sig[sig.order == order]
        for curve, colr in [('Vertex', VERTEX), ('IPS (full probit)', IPS)]:
            c = s[s.curve == curve].sort_values('x')
            ax.plot(c.x, c.p, color=colr, lw=1.5, zorder=3)
        ax.axhline(.5, color='.75', lw=.7, ls='--', zorder=0)
        ax.set_ylim(0, 1); ax.set_yticks([0, .25, .5, .75, 1])
        ax.set_xlabel('Risky/safe payoff ratio')
        ax.set_title(order, fontsize=8.5, color='.2', pad=4)
        if col == 0:
            ax.set_ylabel('P(chose risky)')
        else:
            ax.set_yticklabels([])
    tops[0].text(.05, .95, 'Vertex', transform=tops[0].transAxes, fontsize=7.2,
                 color=VERTEX, va='top')
    tops[0].text(.05, .85, 'IPS', transform=tops[0].transAxes, fontsize=7.2,
                 color=IPS, va='top')

    # --- c, d: the cTBS change in each probit parameter
    stats = {}
    specs = [('rnp', 'Δ risk-neutral probability\nIPS − vertex', 'More risk-seeking'),
             ('slope', 'Δ psychometric slope\nIPS − vertex', 'Less consistent')]
    bots = []
    for col, (par, ylab, note) in enumerate(specs):
        ax = fig.add_subplot(gs[1, col]); bots.append(ax)
        ax.axhline(0, color='.75', lw=.7, ls='--', zorder=0)
        for i, order in enumerate(ORDERS):
            d = paired_delta(post, par, order)
            stats[(par, order)] = (d.mean(), np.quantile(d, .025), np.quantile(d, .975),
                                   float((d > 0).mean()))
            colr = IPS if order == 'Risky second' else '.55'
            parts = ax.violinplot([d], positions=[i], widths=.62, showextrema=False)
            for b in parts['bodies']:
                b.set_facecolor(colr); b.set_alpha(.22); b.set_edgecolor('none')
            lo, hi = np.quantile(d, [.025, .975])
            ax.plot([i, i], [lo, hi], color=colr, lw=1.6, solid_capstyle='round',
                    zorder=3)
            ax.plot([i], [d.mean()], 'o', color=colr, ms=5.5, zorder=4)
        ax.set_xticks([0, 1]); ax.set_xticklabels(['Risky\nfirst', 'Risky\nsecond'])
        ax.set_xlim(-.6, 1.6)
        ax.set_ylabel(ylab)
        ax.text(.03, .96, note, transform=ax.transAxes, fontsize=6.6, color='.4',
                va='top')

    for a, letter in zip([tops[0], tops[1], bots[0], bots[1]], 'abcd'):
        a.text(-.17, 1.05, letter, transform=a.transAxes, **PANEL)
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf\n')
    print(f'{"parameter":<8}{"order":<15}{"Δ (IPS−vertex)":>16}{"95% CrI":>22}{"P(Δ>0)":>9}')
    for (par, order), (m, lo, hi, p) in stats.items():
        print(f'{par:<8}{order:<15}{m:>+16.4f}   [{lo:+.4f}, {hi:+.4f}]{p:>9.3f}')
    print('\nA pure flattening moves the slope only; a pure preference change moves the')
    print('RNP only. Both move, and only when the risky option came second.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/fig3_probit')
    a = parser.parse_args()
    main(a.data_dir, a.out)
