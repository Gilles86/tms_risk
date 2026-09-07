"""Figure 3A, with the cognitive model in place of the probit fit.

The published Figure 3A shows observed choice proportions over a *probit* fit --
a curve that assumes nothing beyond "there is a psychophysical function". This
draws the identical panel, same axes, same colours, same markers, but the curve
and band are the PMC's posterior predictive. So the model is judged on the
display the paper already uses, rather than on a new one invented for it.

Everything else is deliberately unchanged from `plot_fig3_probit`: presentation
order as the row variable, log ratio axis over the observed range, the
risk-neutral ratio marked, and a legend rather than direct labels because the
two curves run too close together to label in place.

    python -m tms_risk.behavior.scripts.plot_fig3a_anchor --model_label log-power-n1n2
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
READ = dict(sep='\t', keep_default_na=False, na_values=[''])
ORDERS = ['Risky first', 'Risky second']
XT = [1.5, 2, 2.5, 3]
RISK_NEUTRAL = 1 / .55

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main(data_dir, out_stem, label, width):
    d = pd.read_csv(Path(data_dir) / f'ppc_anchor/ppc_anchor.rung.{label}.tsv',
                    **READ)
    fig = plt.figure(figsize=(width, 3.4))
    gs = fig.add_gridspec(2, 1, hspace=.14, left=.20, right=.97, top=.90,
                          bottom=.16)
    axes = []
    for row, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs[row])
        axes.append(ax)
        ax.axhline(.5, color='.75', lw=.7, ls='--', zorder=0)
        ax.axvline(RISK_NEUTRAL, color='.75', lw=.7, ls='--', zorder=0)
        o = d[d.order == order]
        for stim, colr, mk in [('vertex', VERTEX, 'o'), ('ips', IPS, 's')]:
            g = o[o.stim == stim].sort_values('frac')
            ax.fill_between(g.frac, g.lo, g.hi, color=colr, alpha=.22, lw=0,
                            zorder=1)
            ax.plot(g.frac, g.model, color=colr, lw=1.4, zorder=2)
            ax.plot(g.frac, g.observed, mk, color=colr, ms=3.7, lw=0, zorder=4)
        ax.set_xscale('log')
        ax.set_xticks(XT)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_xlim(1.5, 3.25)
        ax.set_ylim(.20, .90)
        ax.set_yticks([.25, .5, .75])
        ax.set_ylabel('P(chose risky)')
        ax.text(.02, .96, order, transform=ax.transAxes, fontsize=8,
                color='.2', va='top')
        if row == 0:
            ax.set_xticklabels([])
            ax.text(RISK_NEUTRAL * 1.04, .89, 'Risk-neutral', fontsize=7.5,
                    color='.45', ha='left', va='top', style='italic')
        else:
            ax.set_xlabel('Risky/safe payoff ratio')
    sns.despine(ax=axes[0], offset=3, bottom=True)
    axes[0].tick_params(axis='x', length=0)
    sns.despine(ax=axes[1], offset=3)
    axes[0].plot([], [], color=VERTEX, marker='o', ms=3.7, lw=1.4, label='Vertex')
    axes[0].plot([], [], color=IPS, marker='s', ms=3.7, lw=1.4, label='IPS')
    axes[0].legend(loc='lower right', fontsize=8, handlelength=1.5, borderpad=.2,
                   labelspacing=.2, borderaxespad=.2)
    fig.text(.20, .965, f'{label} · curves and bands are the model, '
                        f'not a probit fit', fontsize=6.6, color='.45')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/fig3a_anchor'))
    ap.add_argument('--width', default=2.6, type=float)
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label, a.width)
