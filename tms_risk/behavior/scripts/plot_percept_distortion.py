"""How each option's perceived value is distorted, by presentation order and cTBS.

Four panels, read as a 2x2:

    columns  presentation order (risky first / risky second)
    row 1    perceived expected value against objective expected value. Distance
             below the identity line is prior attraction; the gap between the two
             stimulation conditions is the cTBS effect.
    row 2    that cTBS effect on its own, with 95% credible intervals.

The point of the layout is the column contrast in row 2: the safe option's curve
drops noticeably further in the right-hand column, because that is where the safe
option is the one held in memory. The risky option's curve barely moves between
columns. That difference is the whole order effect.

    python -m tms_risk.behavior.scripts.plot_percept_distortion --label flexible2nf

Reads notes/data/pmc_percepts_by_order.<label>.tsv. No trace needed.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
SAFE, RISKY = '#4d4d4d', '#b2182b'
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
    'lines.linewidth': 1.3, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')
PANEL = dict(fontsize=11, fontweight='bold', va='bottom', ha='right')


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    d = pd.read_csv(data / f'pmc_percepts_by_order.{label}.tsv', sep='\t')

    fig = plt.figure(figsize=(7.0, 4.9))
    gs = fig.add_gridspec(2, 2, hspace=.46, wspace=.30,
                          left=.095, right=.90, top=.86, bottom=.10)

    lim_hi = max(d.objective_ev.max(), d.vertex.max()) * 1.06
    dlo = min(d.lo.min(), d.delta.min()) * 1.12
    dhi = max(0.02, d.hi.max() * 1.12)

    axes_top, axes_bot = [], []
    for col, order in enumerate(ORDERS):
        o = d[d.order == order]

        # ---- row 1: perceived vs objective expected value
        ax = fig.add_subplot(gs[0, col]); axes_top.append(ax)
        ax.plot([0, lim_hi], [0, lim_hi], color='.78', lw=.7, ls=':', zorder=0)
        ax.text(lim_hi * .98, lim_hi * .98, 'Veridical', fontsize=5.8, color='.55',
                ha='right', va='bottom', rotation=45, rotation_mode='anchor')
        for opt, base in [('safe', SAFE), ('risky', RISKY)]:
            s = o[o.option == opt].sort_values('objective_ev')
            pos = s.position.iloc[0]
            ax.plot(s.objective_ev, s.vertex, color=VERTEX, lw=1.3, marker='o', ms=3.4,
                    zorder=3)
            ax.plot(s.objective_ev, s.ips, color=IPS, lw=1.3, marker='s', ms=3.4,
                    mfc='white', zorder=3)
            ax.annotate(f'{opt.capitalize()} ({pos})',
                        xy=(s.objective_ev.iloc[-1], s.vertex.iloc[-1]),
                        xytext=(4, 6 if opt == 'safe' else -12),
                        textcoords='offset points', fontsize=6.6, color=base)
        ax.set_xlim(0, lim_hi); ax.set_ylim(0, lim_hi)
        ax.set_title(order, fontsize=8.5, color='.2', pad=4)
        ax.set_xlabel('Objective expected value (CHF)')
        if col == 0:
            ax.set_ylabel('Perceived expected\nvalue (CHF)')

        # ---- row 2: the cTBS effect
        ax = fig.add_subplot(gs[1, col]); axes_bot.append(ax)
        ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
        for opt, colr in [('safe', SAFE), ('risky', RISKY)]:
            s = o[o.option == opt].sort_values('n_safe')
            ax.fill_between(s.n_safe, s.lo, s.hi, color=colr, alpha=.20, lw=0, zorder=1)
            ax.plot(s.n_safe, s.delta, color=colr, marker='o', ms=3.4, zorder=2)
            ax.annotate(f'{opt.capitalize()}', xy=(s.n_safe.iloc[-1], s.delta.iloc[-1]),
                        xytext=(5, 0), textcoords='offset points', fontsize=7,
                        color=colr, va='center')
        ax.set_xticks([7, 10, 14, 20, 28])
        ax.set_xlim(6, 33)
        ax.set_ylim(dlo, dhi)
        ax.set_xlabel('Safe payoff (CHF)')
        if col == 0:
            ax.set_ylabel('Δ perceived value\nIPS − vertex (CHF)')
        else:
            ax.set_yticklabels([])

    axes_top[0].plot([], [], color=VERTEX, marker='o', ms=3.4, label='Vertex')
    axes_top[0].plot([], [], color=IPS, marker='s', ms=3.4, mfc='white', label='IPS')
    leg = axes_top[0].legend(loc='upper left', fontsize=6.6, handlelength=1.5,
                             borderpad=.3, labelspacing=.25)
    leg.set_zorder(5)

    g = d.groupby(['option', 'position']).delta.mean()
    axes_bot[1].annotate('Safe option falls further\nwhen it is presented first',
                         xy=(20, float(d[(d.order == 'Risky second') &
                                         (d.option == 'safe')].delta.iloc[-2])),
                         xytext=(8.5, dlo * .55), fontsize=6.8, color='.3',
                         ha='left', va='center',
                         arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-.25',
                                         color='.45', lw=.6))

    for ax, letter in [(axes_top[0], 'a'), (axes_bot[0], 'b')]:
        ax.text(-.21, 1.06, letter, transform=ax.transAxes, **PANEL)
    fig.suptitle('cTBS pulls both options toward the prior — but the safe option '
                 'further, and most when it comes first', fontsize=9, y=.955, color='.15')
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf')
    print('  mean Δ perceived value (CHF), by option and position:')
    print('   ' + g.round(3).to_string().replace('\n', '\n   '))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    main(args.data_dir, args.label,
         args.out or f'/Users/gdehol/git/tms_risk/notes/figures/percept_distortion.{args.label}')
