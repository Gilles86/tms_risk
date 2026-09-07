"""Posterior predictive psychometric curves, in the idiom of Figure 3a.

Same axes, same palette, same reading as the model-free panel the paper opens
the behavioural section with: proportion of risky choices against the
risky/safe payoff ratio, IPS (red) against vertex (green), one row per
presentation order. The difference is that the shaded band is now the model's
95% posterior predictive interval rather than a probit fit, so the panel asks
whether the cognitive model can PRODUCE the psychophysics, not just whether a
descriptive curve passes through it.

Two versions from the same data:

    --split none    one column, collapsed over stake (the Figure 3a layout)
    --split stake   three columns, one per within-participant stake tercile

The stake split is the discriminating one: the cTBS effect is concentrated at
small payoffs, so a model that reproduces the collapsed curve can still miss
the tercile pattern.

    python -m tms_risk.behavior.scripts.plot_ppc_psychometric_supp --split stake
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
IPS, VERTEX = '#d62728', '#2ca02c'
ORDERS = ['Risky first', 'Risky second']
P_RISKY = .55

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})


def main(data_dir, out_stem, label, split):
    dd = Path(data_dir) / 'ppc_anchor'
    if split == 'stake':
        d = pd.read_csv(dd / f'ppc_anchor.stake3rung.{label}.tsv', **READ)
        cols = sorted(d.stake_grp.unique())
        titles = [f'Stake ≈ {v:.0f} CHF' for v in
                  d.groupby('stake_grp')['stake_chf'].mean().round(0)]
    else:
        d = pd.read_csv(dd / f'ppc_anchor.rung.{label}.tsv', **READ)
        d['stake_grp'] = 0
        cols, titles = [0], ['All trials']

    n = len(cols)
    fig, AX = plt.subplots(2, n, figsize=(2.1 * n + .9, 4.2), sharex=True,
                           sharey=True, constrained_layout=True, squeeze=False)
    for r, order in enumerate(ORDERS):
        for c, sg in enumerate(cols):
            ax = AX[r, c]
            o = d[(d.order == order) & (d.stake_grp == sg)]
            for stim, col in (('vertex', VERTEX), ('ips', IPS)):
                q = o[o.stim == stim].sort_values('frac')
                ax.fill_between(q.frac, q.lo, q.hi, color=col, alpha=.22, lw=0,
                                zorder=1)
                ax.plot(q.frac, q.model, color=col, lw=1.6, zorder=2)
                ax.plot(q.frac, q.observed,
                        'o' if stim == 'vertex' else 's', ms=4.4, color=col,
                        lw=0, zorder=4)
            ax.axhline(.5, color='.85', lw=.7, ls='--', zorder=0)
            ax.axvline(1 / P_RISKY, color='.85', lw=.7, ls='--', zorder=0)
            ax.set_xscale('log')
            ax.set_xticks([1.5, 2, 3])
            ax.set_xticklabels(['1.5', '2', '3'])
            ax.minorticks_off()
            ax.set_ylim(.08, .97)
            ax.set_yticks([.25, .5, .75])
            if r == 0:
                ax.set_title(titles[c], fontsize=9.5)
            if r == 1:
                ax.set_xlabel('Risky / safe payoff ratio')
            if c == 0:
                ax.set_ylabel(f'{order}\nP(chose risky)')
    # direct labels, on the row where the two conditions separate
    ax = AX[1, 0]
    o = d[(d.order == ORDERS[1]) & (d.stake_grp == cols[0])]
    for stim, col, dy in (('ips', IPS, 15), ('vertex', VERTEX, -15)):
        q = o[o.stim == stim].sort_values('frac')
        k = 1
        ax.annotate('IPS' if stim == 'ips' else 'Vertex',
                    (q.frac.iloc[k], q.observed.iloc[k]),
                    xytext=(5, dy) if stim == 'ips' else (-2, dy),
                    textcoords='offset points', color=col, fontsize=8,
                    ha='left' if stim == 'ips' else 'center', va='center',
                    fontweight='bold')
    AX[0, 0].text(.04, .96, 'Risk-neutral →', transform=AX[0, 0].transAxes,
                  fontsize=6.5, color='.5', va='top')
    for letter, ax in zip('ab', AX[:, 0]):
        ax.text(-.30, 1.04, letter, transform=ax.transAxes, fontsize=9,
                fontweight='bold', family='Arial', va='bottom', ha='right')
    sns.despine(fig=fig, offset=4, trim=False)
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--split', default='stake', choices=['none', 'stake'])
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    stem = a.out_stem or str(REPO / f'notes/figures/supp_ppc_psychometric_{a.split}')
    main(a.data_dir, stem, a.model_label, a.split)
