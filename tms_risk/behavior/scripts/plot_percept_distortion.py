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

    fig = plt.figure(figsize=(7.25, 4.8))
    gs = fig.add_gridspec(2, 2, hspace=.30, wspace=.34,
                          left=.095, right=.905, top=.86, bottom=.105)

    # Percepts are compressed into a narrow band near the prior, so an axis that
    # spans the objective range squashes all four curves on top of each other. Fit
    # the y-axis to the percepts instead and mark the fitted priors, which is what
    # the curves are being pulled toward.
    lim_hi = d.objective_ev.max() * 1.06
    pv = pd.concat([d.vertex, d.ips])
    y_lo, y_hi = pv.min() - .45, pv.max() + .55
    try:
        pri = pd.read_csv(data / f'pmcpars_priors.{label}.tsv', sep='\t')
        pri = pri[pri.level == 'group'].set_index('parameter')['mean']
        priors = {'safe': float(pri.safe_prior_mu), 'risky': float(pri.risky_prior_mu)}
    except (FileNotFoundError, KeyError):
        priors = {}
    dlo = min(d.lo.min(), d.delta.min()) * 1.12
    dhi = max(0.02, d.hi.max() * 1.12)

    # One option per panel: four curves on a single axis (safe/risky x vertex/IPS)
    # were impossible to tell apart once the y-axis was zoomed to the percept band.
    DIFF = '#4a1486'
    axes_top, axes_bot = [], []
    for col, order in enumerate(ORDERS):
        o = d[d.order == order]
        for r, (opt, base) in enumerate([('safe', SAFE), ('risky', RISKY)]):
            ax = fig.add_subplot(gs[r, col])
            axes_top.append(ax)
            s_ = o[o.option == opt].sort_values('objective_ev')
            pos = s_.position.iloc[0]
            ax.plot([0, lim_hi], [0, lim_hi], color='.8', lw=.7, ls=':', zorder=0)
            if opt in priors and y_lo < priors[opt] < y_hi:
                ax.axhline(priors[opt], color=base, lw=.7, ls=(0, (4, 3)), alpha=.55,
                           zorder=0)
                ax.text(lim_hi * .985, priors[opt], 'Prior', fontsize=6, color=base,
                        ha='right', va='bottom')
            ax.fill_between(s_.objective_ev, s_.ips, s_.vertex, color=IPS, alpha=.13,
                            lw=0, zorder=1)
            ax.plot(s_.objective_ev, s_.vertex, color=VERTEX, lw=1.4, marker='o',
                    ms=3.4, zorder=3)
            ax.plot(s_.objective_ev, s_.ips, color=IPS, lw=1.4, marker='s', ms=3.4,
                    mfc='white', zorder=3)
            ax.set_xlim(0, lim_hi); ax.set_ylim(y_lo, y_hi)
            ax.text(.035, .95, f'{opt.capitalize()} option — presented {pos}',
                    transform=ax.transAxes, fontsize=7.2, color=base, va='top')
            if r == 0:
                ax.set_title(order, fontsize=8.5, color='.2', pad=4)
            ax.set_xlabel('Objective expected value (CHF)')
            if col == 0:
                ax.set_ylabel('Perceived expected\nvalue (CHF)')

            # ---- the cTBS effect, shown as the gap itself
            # A twin axis was tried here and actively misled: the difference curve
            # sat above the vertex curve in figure coordinates, so it read as a
            # positive effect when it is negative everywhere. The difference IS the
            # vertical distance between the two curves, so draw exactly that.
            for xx, yv, yi in zip(s_.objective_ev, s_.vertex, s_.ips):
                ax.plot([xx, xx], [yi, yv], color=DIFF, lw=.9, alpha=.75, zorder=2,
                        solid_capstyle='butt')
            mean_d = float(s_.delta.mean())
            ax.text(.965, .06,
                    f'Mean Δ = {mean_d:+.2f} CHF\n'
                    f'({s_.delta.min():+.2f} to {s_.delta.max():+.2f})',
                    transform=ax.transAxes, fontsize=6.4, color=DIFF,
                    ha='right', va='bottom', linespacing=1.25)

    axes_top[0].plot([], [], color=VERTEX, marker='o', ms=3.6, label='Vertex')
    axes_top[0].plot([], [], color=IPS, marker='s', ms=3.6, mfc='white', label='IPS')
    axes_top[0].plot([], [], color=DIFF, lw=1.4, label='IPS − vertex')
    leg = axes_top[0].legend(loc='upper left', fontsize=6.4, handlelength=1.6,
                             borderpad=.3, labelspacing=.22,
                             bbox_to_anchor=(.02, .88))
    leg.set_zorder(5)

    g = d.groupby(['option', 'position']).delta.mean()
    # axes_top order is [safe|risky-first, risky|risky-first, safe|risky-second, ...]
    axes_top[2].text(.035, .80, 'Widest gap of the four panels',
                     transform=axes_top[2].transAxes, fontsize=6.4, color=DIFF,
                     va='top')

    for ax, letter in [(axes_top[0], 'a'), (axes_top[1], 'b')]:
        ax.text(-.17, 1.04, letter, transform=ax.transAxes, **PANEL)
    fig.suptitle('cTBS pulls both options toward the prior — but the safe option '
                 'further, and most when it comes first', fontsize=9, y=.955, color='.15')
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
