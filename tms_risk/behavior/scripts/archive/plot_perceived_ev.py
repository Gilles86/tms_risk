"""Perceived expected value against objective expected value.

The distortion the whole model is about, on one pair of axes. The identity line
is a veridical observer; everything the Bayesian observer does is the departure
from it. Because the percept is shrunk toward a prior, the curve is compressed:
small values are overvalued, large ones undervalued, and the crossing point is
the prior mean.

Rows are presentation order, colour is stimulation, line style is the option.
Both matter: the noise depends on which POSITION an option occupied (the first
is recalled from memory) and the prior on which ROLE it played (risky or safe).

Bottom row shows the same thing as a percentage departure from veridical, which
is where a cTBS effect of a few percent is actually legible.

    python -m tms_risk.behavior.scripts.plot_perceived_ev
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
IPS, VERTEX = '#d62728', '#2ca02c'
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def logaxis(ax, ticks):
    ax.set_xscale('log')
    ax.set_xticks(ticks)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def main(data_dir, out_stem, label):
    d = pd.read_csv(Path(data_dir) /
                    f'perceived_ev/perceived_ev.{label}.tsv', **READ)
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 4.6), constrained_layout=True,
                             sharex=True)
    xt = [4, 7, 14, 28, 62]
    lim = (d.objective_ev.min() * .9, d.objective_ev.max() * 1.1)

    for c, order in enumerate(ORDERS):
        ax = axes[0, c]
        ax.plot(lim, lim, color='0.75', lw=.8, ls=':', zorder=1)
        for role, ls in [('risky', '-'), ('safe', (0, (3, 1.5)))]:
            for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
                s = d[(d.order == order) & (d.role == role)
                      & (d.stimulation_condition == cond)].sort_values('objective_ev')
                ax.fill_between(s.objective_ev, s.lo, s.hi, color=col, alpha=.15,
                                lw=0)
                ax.plot(s.objective_ev, s.perceived_ev, color=col, ls=ls, lw=1.3)
        ax.set_yscale('log')
        ax.set_yticks(xt)
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        logaxis(ax, xt)
        ax.set_xlim(*lim)
        ax.set_ylim(*lim)
        ax.set_title(order, fontsize=8, color='0.15', pad=3)
        if c == 0:
            ax.set_ylabel('Perceived EV (CHF)')

        ax = axes[1, c]
        ax.axhline(0, color='0.75', lw=.8, ls=':', zorder=1)
        for role, ls in [('risky', '-'), ('safe', (0, (3, 1.5)))]:
            for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
                s = d[(d.order == order) & (d.role == role)
                      & (d.stimulation_condition == cond)].sort_values('objective_ev')
                rel = 100 * (s.perceived_ev / s.objective_ev - 1)
                ax.fill_between(s.objective_ev,
                                100 * (s.lo / s.objective_ev - 1),
                                100 * (s.hi / s.objective_ev - 1),
                                color=col, alpha=.15, lw=0)
                ax.plot(s.objective_ev, rel, color=col, ls=ls, lw=1.3)
        logaxis(ax, xt)
        ax.set_xlim(*lim)
        ax.set_xlabel('Objective EV (CHF)')
        if c == 0:
            ax.set_ylabel('Departure from veridical (%)')

    a = axes[0, 0]
    a.text(.04, .96, 'IPS', color=IPS, transform=a.transAxes, va='top',
           fontsize=7)
    a.text(.04, .86, 'Vertex', color=VERTEX, transform=a.transAxes, va='top',
           fontsize=7)
    a.text(.97, .06, 'Solid: risky option\nDashed: safe option\nDotted: veridical',
           transform=a.transAxes, ha='right', va='bottom', fontsize=6.2,
           color='0.4', linespacing=1.5)
    for ax, s in zip(axes[:, 0], 'ab'):
        ax.text(-.26, 1.04, s, transform=ax.transAxes, fontsize=8,
                fontweight='bold', family='Arial', va='bottom')
    fig.suptitle(f'{label} · group posterior, subjects averaged within draw',
                 fontsize=6.6, color='0.4', y=1.03)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/perceived_ev'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
