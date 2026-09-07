"""Implied log-space noise curves per subject: one panel per option, per fit.

sigma(x) = c * x**p reconstructed from each subject's fitted parameters, drawn on
the log-payoff axis where Weber's law is a HORIZONTAL LINE. Thin lines are the 35
subjects, thick lines their median.

Rows are the two options, columns the two parameterizations, so each column is
one model's complete account of the noise and the two columns are the same
observable quantity estimated two ways.

    python -m tms_risk.behavior.scripts.plot_implied_noise_curves
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

IPS, VERTEX = '#d62728', '#2ca02c'
X = np.geomspace(7, 112, 60)
ROWS = [('n1 (first)', 'First-presented option\n(held in memory)'),
        ('n2 (second)', 'Second-presented option\n(seen)')]
# Third column: the shared fit in its OWN coordinates. These are softplus of one
# term of a sum, so they are not any option's noise SD -- but they are what that
# model's parameters actually are, so worth seeing next to what they imply.
COLS = [('logflex1', 'Independent fit\nr\u0302 = 1.00'),
        ('logflex2', 'Shared fit\nr\u0302 = 2.41 — DID NOT CONVERGE'),
        ('logflex2', 'Shared fit, own coordinates')]
CHANNEL_ROWS = ['memory', 'perceptual']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='notes/data/per_subject_powerlaw.tsv')
    ap.add_argument('--out', default='notes/figures/implied_noise_curves')
    args = ap.parse_args()
    d = pd.read_csv(args.tsv, sep='\t')

    fig, axes = plt.subplots(2, 3, figsize=(8.6, 4.8), constrained_layout=True,
                             sharex=True, sharey=True)
    for row, (param, rlab) in enumerate(ROWS):
        for col, (model, clab) in enumerate(COLS):
            if col == 2:
                param = CHANNEL_ROWS[row]
            ax = axes[row, col]
            for stim, c in [('vertex', VERTEX), ('ips', IPS)]:
                g = d[(d.model == model) & (d.param == param) & (d.stim == stim)]
                cur = np.array([r.c * X ** r.exponent for r in g.itertuples()])
                for y in cur:
                    ax.plot(X, y, color=c, lw=.3, alpha=.16, zorder=1)
                # These are multiplicative quantities on a log axis, so the
                # geometric mean and a multiplicative SEM band are the right
                # summary: mean +- SEM computed on log(sigma), then exponentiated.
                lg = np.log(cur)
                m, sem = lg.mean(0), lg.std(0, ddof=1) / np.sqrt(len(lg))
                ax.fill_between(X, np.exp(m - sem), np.exp(m + sem), color=c,
                                alpha=.35, lw=0, zorder=2)
                ax.plot(X, np.exp(m), color=c, lw=2.0, zorder=3)
                if col == 0:
                    e = g.exponent.values
                    ax.text(116, np.exp(m[-1]),
                            f'{e.mean():+.2f} ± {e.std(ddof=1)/np.sqrt(len(e)):.2f}',
                            color=c, fontsize=6.2, va='center', ha='left')
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xticks([7, 14, 28, 56, 112])
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.set_yticks([0.05, 0.1, 0.2, 0.4, 0.8])
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            if row == 1:
                ax.set_xlabel('Payoff (CHF)')
            if col == 0:
                ax.set_ylabel(f'{rlab}\n\nNoise SD (log units)', fontsize=7.5,
                              linespacing=1.25)
            if col == 2:
                ax.text(.5, .93, param.capitalize() + ' channel',
                        transform=ax.transAxes, fontsize=7, ha='center',
                        va='top', color='.25')
            if row == 0:
                ax.set_title(clab, fontsize=8)
    axes[0, 0].text(.03, .06, 'IPS', transform=axes[0, 0].transAxes, color=IPS,
                    fontsize=7, va='bottom')
    axes[0, 0].text(.03, .17, 'Vertex', transform=axes[0, 0].transAxes,
                    color=VERTEX, fontsize=7, va='bottom')
    axes[0, 1].text(.97, .06, 'Flat = Weber', transform=axes[0, 1].transAxes,
                    fontsize=6.3, color='.45', ha='right', va='bottom')
    axes[1, 1].text(.97, .04,
                    'Bands: ±1 SEM across subjects of the per-subject point\n'
                    'estimates — each subject\u2019s own posterior width is not in it',
                    transform=axes[1, 1].transAxes, fontsize=5.9, color='.45',
                    ha='right', va='bottom', linespacing=1.3)

    sns.despine(fig=fig, offset=3)
    for ax, letter in zip(axes.ravel(), 'abcd'):
        ax.text(-0.17, 1.04, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    for model, _ in COLS[:2]:
        for param, _ in ROWS:
            g = d[(d.model == model) & (d.param == param)]
            w = g.pivot_table(index='subject', columns='stim', values='exponent')
            print(f'{model:9s} {param:12s} exponent  vertex {w["vertex"].median():+.3f}'
                  f'  ips {w["ips"].median():+.3f}'
                  f'  delta {(w["ips"] - w["vertex"]).mean():+.3f}')
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
