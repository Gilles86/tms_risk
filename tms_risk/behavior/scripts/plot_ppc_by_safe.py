"""Posterior predictive check, broken down by safe payoff.

The pooled version shows that the model reproduces the order-specific cTBS effect. This
splits the same check by the size of the safe payoff, which is where the mechanism says
the effect should live: cTBS raises perceptual noise by a roughly constant absolute
amount, so the proportional degradation -- and therefore the behavioural effect -- is
largest where payoffs are smallest.

Rows are presentation order, columns are safe payoff. Ratio bins are Vincentized, formed
within each (participant, safe payoff), so all 35 participants contribute to every cell
and the plotted x is the across-subject mean of each participant's own bin mean.

    python -m tms_risk.behavior.scripts.plot_ppc_by_safe --label flexible2nf_perception

Reads notes/data/ppc_by_safe.<label>.tsv. No trace needed.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
# EV_risky = 0.55 * n_risky equals EV_safe = n_safe at n_risky/n_safe = 1/0.55, so left
# of this line choosing risky is risk-seeking and right of it choosing safe is risk-averse.
RISK_NEUTRAL = 1 / 0.55
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def main(data_dir, label, out_stem):
    d = pd.read_csv(Path(data_dir) / f'ppc_by_safe.{label}.tsv', sep='\t')
    safes = np.sort(d.n_safe.unique())
    n = len(safes)

    fig, axes = plt.subplots(2, n, figsize=(7.25, 3.5), sharex=True, sharey=True,
                             squeeze=False)
    for r, order in enumerate(ORDERS):
        for c, safe in enumerate(safes):
            ax = axes[r][c]
            g0 = d[(d.order == order) & (d.n_safe == safe)]
            ax.axhline(.5, color='.85', lw=.6, ls='--', zorder=0)
            ax.axvline(RISK_NEUTRAL, color='.7', lw=.6, ls=':', zorder=0)
            for stim, colr, mk in [('vertex', VERTEX, 'o'), ('ips', IPS, 's')]:
                g = g0[g0.stim == stim].sort_values('frac')
                ax.fill_between(g.frac, g.lo, g.hi, color=colr, alpha=.22, lw=0,
                                zorder=1)
                ax.plot(g.frac, g['mean'], color=colr, lw=1.2, zorder=2)
                ax.plot(g.frac, g.observed, mk, color=colr, ms=3.6, lw=0,
                        mfc='white' if stim == 'ips' else colr, mew=1, zorder=4)
            ax.set_xlim(1.45, 3.15)
            ax.set_xticks([1.5, 2, 2.5, 3])
            ax.set_ylim(.12, .95)
            ax.set_yticks([.25, .5, .75])
            if r == 0:
                ax.set_title(f'{safe:.0f} CHF', fontsize=8, color='.2', pad=4)
            if c == 0:
                ax.set_ylabel(f'{order}\n\nP(chose risky)', fontsize=8.5)
            if r == 1:
                ax.set_xlabel('Risky/safe ratio' if c == n // 2 else '')

    axes[0][0].plot([], [], color=VERTEX, marker='o', ms=3.6, lw=1.2, label='Vertex')
    axes[0][0].plot([], [], color=IPS, marker='s', ms=3.6, mfc='white', mew=1,
                    lw=1.2, label='IPS')
    axes[0][0].legend(loc='upper left', fontsize=6.4, handlelength=1.5, borderpad=.25,
                      labelspacing=.2)
    axes[0][n - 1].text(.97, .06, 'Dotted: risk neutral', transform=axes[0][n - 1].transAxes,
                        fontsize=6, color='.45', ha='right')
    fig.text(.5, .965, 'Safe payoff', ha='center', fontsize=8.5, color='.2')
    sns.despine(fig=fig, offset=3)
    fig.tight_layout(rect=[0, 0, 1, .945])
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf')
    w = d.pivot_table(index=['order', 'n_safe'], columns='stim', values='observed')
    w['delta'] = w['ips'] - w['vertex']
    print('\nobserved cTBS effect on P(chose risky), by safe payoff:')
    print(w['delta'].unstack('order').round(3).to_string())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf_perception')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    a = parser.parse_args()
    main(a.data_dir, a.label,
         a.out or f'/Users/gdehol/git/tms_risk/notes/figures/ppc_by_safe.{a.label}')
