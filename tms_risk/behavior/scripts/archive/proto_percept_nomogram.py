"""PROTOTYPE. The compression drawn as a MAPPING, and the ratio as a merge.

One nomogram per stake. Each cell has three vertical scales, read left to right:

    OBJECTIVE   the two expected values the trial actually offers
    PERCEIVED   where the model puts them, same log-CHF scale, so the length of
                the tie-line IS the compression
    RATIO       the two perceived values merged into the one number the choice
                rule sees, on its own zoomed scale

The tie-lines are the argument. They all point down (every payoff sits above the
fitted prior), the objective pair is far apart while the perceived pair is nearly
on top of each other, and under cTBS (open markers) the safe tie-line lengthens
more than the risky one. That difference in lengthening is the whole ratio shift.

    python -m tms_risk.behavior.scripts.proto_percept_nomogram

Reads notes/data/pmc_percepts_by_order.<label>.tsv and pmcpars_priors.<label>.tsv.
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
DIFF = '#1a1a1a'
ORDERS = ['Risky first', 'Risky second']
STAKES = [7.0, 14.0, 28.0]

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
PANEL = dict(fontsize=10, fontweight='bold', va='bottom', ha='right')

X_OBJ, X_PER = 0.0, 1.0


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    d = pd.read_csv(data / f'pmc_percepts_by_order.{label}.tsv', sep='\t')
    pri = pd.read_csv(data / f'pmcpars_priors.{label}.tsv', sep='\t')
    pri = pri[pri.level == 'group'].set_index('parameter')['mean']
    prior = {'safe': float(pri.safe_prior_mu), 'risky': 0.55 * float(pri.risky_prior_mu)}

    fig = plt.figure(figsize=(7.25, 4.7))
    outer = fig.add_gridspec(2, 3, hspace=.22, wspace=.50,
                             left=.125, right=.985, top=.855, bottom=.075)

    for r, order in enumerate(ORDERS):
        for k, stake in enumerate(STAKES):
            inner = outer[r, k].subgridspec(1, 2, width_ratios=[2.3, 1], wspace=1.05)
            ax = fig.add_subplot(inner[0])
            axr = fig.add_subplot(inner[1])
            sub = d[(d.order == order) & (d.n_safe == stake)]

            ax.set_yscale('log')
            ax.set_xlim(-.20, 1.34)
            ax.set_ylim(3.1, 42)
            ax.set_yticks([4, 6, 10, 16, 25, 40])
            ax.set_yticklabels(['4', '6', '10', '16', '25', '40'])
            ax.minorticks_off()
            ax.set_xticks([X_OBJ, X_PER])
            ax.set_xticklabels(['Objective', 'Perceived'] if r == 1 else ['', ''],
                               fontsize=7, color='0.3')
            ax.tick_params(axis='x', length=0)

            # the two vertical scales of the nomogram
            for xpos in (X_OBJ, X_PER):
                ax.plot([xpos, xpos], [3.3, 40], color='0.86', lw=.8, zorder=0)

            perc = {}
            for _, row in sub.iterrows():
                col = SAFE if row.option == 'safe' else RISKY
                obj = row.objective_ev
                perc[row.option] = (row.vertex, row.ips)
                # the tie-line: objective -> percept. Its length is the compression.
                ax.plot([X_OBJ, X_PER], [obj, row.vertex], '-', color=col,
                        lw=1.4, zorder=3, solid_capstyle='round')
                ax.plot([X_OBJ, X_PER], [obj, row.ips], '-', color=col,
                        lw=.8, alpha=.5, zorder=2)
                ax.plot([X_OBJ], [obj], 'o', color=col, mfc=col, mec=col, ms=4.5,
                        zorder=4)
                ax.plot([X_PER], [row.vertex], 'o', color=col, mfc=col, mec=col,
                        ms=4.5, zorder=4)
                ax.plot([X_PER], [row.ips], 'o', color=col, mfc='white', mec=col,
                        mew=1.1, ms=4.5, zorder=5)
                # the prior each option is being pulled toward
                ax.plot([X_PER + .12, X_PER + .30], [prior[row.option]] * 2, '-',
                        color=col, lw=1.4, zorder=4, alpha=.8)

            # ---- the merge: two perceived values -> one ratio --------------------
            rv = perc['risky'][0] / perc['safe'][0]
            ri = perc['risky'][1] / perc['safe'][1]
            axr.set_ylim(.955, 1.105)
            axr.set_xlim(-.1, 1.05)
            axr.set_yticks([0.96, 1.00, 1.04, 1.08])
            axr.set_yticklabels(['0.96', '1.00', '1.04', '1.08'])
            axr.set_xticks([])
            axr.axhline(1.0, color='0.85', lw=.6, ls='--', zorder=0)
            axr.annotate('', xy=(.55, ri), xytext=(.55, rv),
                         arrowprops=dict(arrowstyle='->', color=DIFF, lw=1.4,
                                         shrinkA=0, shrinkB=0, mutation_scale=8))
            axr.plot([.55], [rv], 'o', color=VERTEX, mfc=VERTEX, ms=5, zorder=5)
            axr.plot([.55], [ri], 'o', color=IPS, mfc=IPS, ms=5, zorder=5)
            # the merge, drawn as a funnel rather than as lines that could be
            # mistaken for data: the two perceived values feed the one ratio
            inv = fig.transFigure.inverted()
            pa = inv.transform(ax.transData.transform((X_PER + .05, perc['risky'][0])))
            pb = inv.transform(ax.transData.transform((X_PER + .05, perc['safe'][0])))
            pc = inv.transform(axr.transData.transform((.30, rv)))
            fig.patches.append(mpl.patches.Polygon(
                [pa, pb, pc], closed=True, transform=fig.transFigure,
                facecolor='0.55', edgecolor='none', alpha=.16, zorder=0))

            if k == 0:
                ax.set_ylabel('Expected value (CHF)')
                pos = outer[r, 0].get_position(fig)
                fig.text(.018, pos.y0 + pos.height / 2, order, rotation=90,
                         va='center', ha='left', fontsize=8.5)
            if r == 0:
                ax.set_title(f'{stake:.0f} CHF safe', fontsize=8, pad=4)
            if r == 0:
                axr.set_title('Ratio', fontsize=7.5, pad=4, color='0.3')
            if k > 0:
                ax.set_yticklabels([])
            if r == 0 and k == 0:
                ax.text(X_PER + .21, 3.25, 'Priors', fontsize=7,
                        ha='center', va='bottom', color='0.35')
                ax.text(X_OBJ, sub[sub.option == 'risky'].objective_ev.iloc[0] * 1.10,
                        'Risky', color=RISKY, fontsize=7.5, ha='center', va='bottom')
                ax.text(X_OBJ, stake * .88, 'Safe', color=SAFE,
                        fontsize=7.5, ha='center', va='top')
            if r == 1 and k == 0:
                axr.text(.42, ri + .004, 'cTBS', fontsize=7.5, color=IPS,
                         ha='right', va='bottom')
                axr.text(.42, rv - .004, 'Vertex', fontsize=7.5, color=VERTEX,
                         ha='right', va='top')
            sns.despine(ax=ax, offset=3, trim=False, bottom=True)
            sns.despine(ax=axr, offset=3, trim=False, bottom=True)
            axr.spines['bottom'].set_visible(False)

    fig.suptitle('Every payoff collapses onto the prior; the safe option falls '
                 'further, so the ratio rises', fontsize=9, y=.965)
    for k, letter in zip(range(3), 'abc'):
        pos = outer[0, k].get_position(fig)
        fig.text(pos.x0 - .022, pos.y1 + .045, letter, **PANEL)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    print(f'wrote {out.with_suffix(".pdf")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--label', default='flexible2nf')
    p.add_argument('--out', default='notes/figures/prototypes/percept_nomogram')
    a = p.parse_args()
    main(a.data_dir, a.label, a.out)
