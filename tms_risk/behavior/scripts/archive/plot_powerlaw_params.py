"""Both power-law parameters, per subject, for both curves and both fits.

sigma(x) = c * x**p has two parameters. Reporting c itself is a trap: it is
sigma at x = 1 CHF, an extrapolation far below the 7-112 CHF range, and it is
strongly anticorrelated with p (raise the exponent, lower the intercept, same
curve through the data). The decorrelated version is sigma at the GEOMETRIC MEAN
of the payoff range -- sqrt(7 * 112) = 28 CHF, where the two parameters are
close to orthogonal. Both are printed; only the latter is plotted.

Rows: level (sigma at 28 CHF) and slope (exponent p).
Columns: the two parameterizations. Within each panel, n1 (first-presented,
remembered) then n2 (second-presented, seen); one grey line per subject.

    python -m tms_risk.behavior.scripts.plot_powerlaw_params
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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
CURVES = ['n1 (first)', 'n2 (second)']
FITS = [('logflex1', 'Independent fit'), ('logflex2', 'Shared fit')]
ROWS = [('sigma_at_28', 'Level: noise SD at 28 CHF\n(log units)', False),
        ('exponent', 'Slope: power-law exponent', True)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='notes/data/per_subject_powerlaw.tsv')
    ap.add_argument('--out', default='notes/figures/powerlaw_params')
    args = ap.parse_args()
    d = pd.read_csv(args.tsv, sep='\t')

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.8), constrained_layout=True)
    for row, (metric, ylab, zero_line) in enumerate(ROWS):
        for col, (model, title) in enumerate(FITS):
            ax = axes[row, col]
            for k, param in enumerate(CURVES):
                w = (d[(d.model == model) & (d.param == param)]
                     .pivot_table(index='subject', columns='stim', values=metric))
                x0, x1 = 3 * k, 3 * k + 1
                for _, r in w.iterrows():
                    ax.plot([x0, x1], [r['vertex'], r['ips']], color='.65',
                            lw=.4, zorder=1)
                ax.plot(np.full(len(w), x0), w['vertex'], 'o', ms=2.6,
                        color=VERTEX, zorder=3)
                ax.plot(np.full(len(w), x1), w['ips'], 'o', ms=2.6, color=IPS,
                        zorder=3)
                ax.plot([x0, x1], [w['vertex'].median(), w['ips'].median()],
                        color='.1', lw=2.2, zorder=4)
                diff = (w['ips'] - w['vertex']).dropna()
                ax.text((x0 + x1) / 2, -.26,
                        f'{param.split()[0]}   Δ={diff.mean():+.3f}\n'
                        f'{int((diff < 0).sum())}/{len(diff)} down',
                        transform=ax.get_xaxis_transform(), fontsize=6.3,
                        ha='center', va='top', linespacing=1.3)
            if zero_line:
                ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
                ax.text(4.65, .012, 'Weber', fontsize=6.2, color='.5',
                        ha='right', va='bottom')
            ax.set_xlim(-.7, 4.7)
            ax.set_xticks([0, 1, 3, 4])
            ax.set_xticklabels(['Vtx', 'IPS', 'Vtx', 'IPS'], fontsize=6.5)
            ax.set_ylabel(ylab if col == 0 else '', linespacing=1.3)
            if row == 0:
                ax.set_title(title, fontsize=8)
    axes[0, 0].text(.03, .97, 'IPS', transform=axes[0, 0].transAxes, color=IPS,
                    fontsize=7, va='top')
    axes[0, 0].text(.03, .86, 'Vertex', transform=axes[0, 0].transAxes,
                    color=VERTEX, fontsize=7, va='top')

    sns.despine(fig=fig, offset=3)
    for ax, letter in zip(axes.ravel(), 'abcd'):
        ax.text(-0.20, 1.06, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    print('=== medians ===')
    t = d.groupby(['model', 'param', 'stim'])[['c', 'sigma_at_28', 'exponent']].median()
    print(t.round(4).to_string())
    print('\n=== across-subject correlation of the two parameters ===')
    for (m, pa, st), g in d.groupby(['model', 'param', 'stim']):
        r_raw = np.corrcoef(np.log(g.c), g.exponent)[0, 1]
        r_dec = np.corrcoef(np.log(g.sigma_at_28), g.exponent)[0, 1]
        print(f'{m:9s} {pa:12s} {st:7s}  r(log c, p) = {r_raw:+.3f}   '
              f'r(log sigma@28, p) = {r_dec:+.3f}')
    print(f'\nwrote {args.out}.pdf')


if __name__ == '__main__':
    main()
