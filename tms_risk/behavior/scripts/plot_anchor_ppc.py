"""Posterior predictive check for the anchor fits.

a  Choice curves against the per-subject payoff ladder, split by presentation
   order: the panel that has to reproduce the order-specific cTBS effect.
b  The same choices collapsed onto stake terciles -- the order x stake x
   stimulation interaction the model comparison is meant to quantify.
c  How well each fitted form does, as RMSE of model minus observed.

Bands are 95% posterior predictive intervals built from SIMULATED CHOICES, so
they carry trial-level binomial noise and the check can actually fail; points
are observed means +/- 1 SEM across subjects.

Reads notes/data/ppc_anchor/*.tsv from `extract_anchor_ppc.py`.
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IPS, VERTEX = '#d62728', '#2ca02c'
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def band(ax, d, xcol):
    for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
        s = d[d.stim == stim].sort_values(xcol)
        ax.fill_between(s[xcol], s.lo, s.hi, color=col, alpha=.20, lw=0, zorder=1)
        ax.plot(s[xcol], s.model, color=col, lw=1.1, zorder=2)
        ax.errorbar(s[xcol], s.observed, yerr=s.observed_sem, fmt='o', color=col,
                    ms=3.6, lw=0, elinewidth=.9, capsize=0, zorder=4)


def main(data_dir, out_stem, model_label):
    data_dir = Path(data_dir)
    rung = pd.read_csv(data_dir / f'ppc_anchor.rung.{model_label}.tsv', sep='\t', keep_default_na=False, na_values=[''])
    stake = pd.read_csv(data_dir / f'ppc_anchor.stake.{model_label}.tsv', sep='\t', keep_default_na=False, na_values=[''])

    allr = pd.concat([pd.read_csv(f, sep='\t', keep_default_na=False, na_values=[''])
                      for f in glob(str(data_dir / 'ppc_anchor.rung.*.tsv'))])
    rmse = (allr.assign(e=(allr.model - allr.observed) ** 2)
                .groupby(['label', 'order'])['e'].mean().pow(.5).reset_index())

    fig = plt.figure(figsize=(7.25, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.15])
    ax_a = [fig.add_subplot(gs[i, 0]) for i in range(2)]
    ax_b = [fig.add_subplot(gs[i, 1]) for i in range(2)]
    ax_c = fig.add_subplot(gs[:, 2])

    for i, order in enumerate(ORDERS):
        ax = ax_a[i]
        band(ax, rung[rung.order == order], 'frac')
        ax.axhline(.5, color='0.8', lw=.6, ls='--', zorder=0)
        ax.set_ylim(.15, .95)
        ax.set_yticks([.2, .4, .6, .8])
        ax.set_ylabel('P(chose risky)')
        ax.text(.03, .96, order, transform=ax.transAxes, fontsize=7.5,
                color='0.2', va='top')
        ax.set_xticks([1.5, 2.0, 2.5, 3.0])
        if i == 1:
            ax.set_xlabel('Risky/safe payoff ratio')

        ax = ax_b[i]
        s = stake[stake.order == order]
        band(ax, s, 'stake_chf')
        ax.set_ylim(.28, .76)
        ax.set_yticks([.3, .4, .5, .6, .7])
        ax.set_xscale('log')
        ax.set_xticks(sorted(s.stake_chf.unique()))
        ax.set_xticklabels([f'{v:.0f}' for v in sorted(s.stake_chf.unique())])
        ax.minorticks_off()
        ax.text(.03, .96, order, transform=ax.transAxes, fontsize=7.5,
                color='0.2', va='top')
        if i == 1:
            ax.set_xlabel('Stake (CHF, terciles)')

    ax_a[0].text(.97, .06, 'Points: data ± SEM\nBands: 95% PPI',
                 transform=ax_a[0].transAxes, ha='right', va='bottom',
                 fontsize=6.5, color='0.35', linespacing=1.5)
    ax_a[0].text(.34, .70, 'IPS', color=IPS, transform=ax_a[0].transAxes, fontsize=7.5)
    ax_a[0].text(.34, .57, 'Vertex', color=VERTEX, transform=ax_a[0].transAxes,
                 fontsize=7.5)

    # -- c: how well each form does ---------------------------------------
    piv = rmse.pivot(index='label', columns='order', values='e').sort_values(
        'Risky second', ascending=False)
    y = np.arange(len(piv))
    ax_c.barh(y - .19, piv['Risky first'], height=.36, color='0.62', lw=0)
    ax_c.barh(y + .19, piv['Risky second'], height=.36, color='0.15', lw=0)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(piv.index, fontsize=6.5)
    ax_c.set_xlabel('RMSE, model − observed')
    ax_c.set_ylim(-.6, len(piv) - .15)
    # legend above the top pair, in the strip left free by set_ylim
    ax_c.text(.02, len(piv) - .30, 'Risky second', color='0.15', fontsize=7,
              va='center')
    ax_c.text(.02 + .0115, len(piv) - .30, '   ·  Risky first', color='0.62',
              fontsize=7, va='center')

    for ax, letter in [(ax_a[0], 'a'), (ax_b[0], 'b'), (ax_c, 'c')]:
        ax.text(-.22, 1.06, letter, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left', family='Arial')

    fig.suptitle(f'Posterior predictive check · {model_label} · 35 subjects',
                 fontsize=7.5, color='0.35', y=1.02)
    sns.despine(fig=fig, offset=4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data/ppc_anchor'))
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/anchor_ppc'))
    ap.add_argument('--model_label', default='log-affine-n1n2')
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
