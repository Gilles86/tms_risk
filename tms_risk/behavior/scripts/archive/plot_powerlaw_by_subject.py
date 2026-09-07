"""Both option-noise curves, two parameterizations, log and natural space.

Each subject's fitted noise curve was reduced to a power law sigma = c * x**p by
OLS of log(sigma) on log(payoff). Curves are subject medians; the exponent panels
show every subject.

Encoding follows the repo convention: stimulation gets the hue (IPS red, vertex
green) and presentation ORDER never does -- n1 (first-presented, remembered) is
solid, n2 (second-presented, seen) dashed.

The comparison the figure exists for: the independent fit lets n1 and n2 have
their own noise functions; the shared fit forces n2 to be a component of n1
(n1 = softplus(memory + perceptual), n2 = softplus(perceptual)). Same data, same
log-space observer -- and they disagree about the vertex baseline, which is where
their disagreement about the cTBS effect comes from.

    python -m tms_risk.behavior.scripts.plot_powerlaw_by_subject
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
CURVES = [('n1 (first)', '-'), ('n2 (second)', '--')]
FITS = [('logflex1', 'Independent fit\nn1 and n2 estimated separately'),
        ('logflex2', 'Shared fit\nn2 forced to be part of n1')]


def lognormal_sd(x, sigma):
    return x * np.exp(sigma ** 2 / 2) * np.sqrt(np.exp(sigma ** 2) - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='notes/data/per_subject_powerlaw.tsv')
    ap.add_argument('--out', default='notes/figures/powerlaw_by_subject')
    args = ap.parse_args()
    d = pd.read_csv(args.tsv, sep='\t')

    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.8), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1, 1.15]))

    for col, (model, title) in enumerate(FITS):
        for row, natural in enumerate([False, True]):
            ax = axes[row, col]
            for param, ls in CURVES:
                for stim, c in [('vertex', VERTEX), ('ips', IPS)]:
                    g = d[(d.model == model) & (d.param == param) & (d.stim == stim)]
                    cur = np.array([r.c * X ** r.exponent for r in g.itertuples()])
                    if natural:
                        cur = lognormal_sd(X[None, :], cur)
                    ax.plot(X, np.median(cur, 0), color=c, lw=1.9, ls=ls)
            ax.set_xscale('log'); ax.set_yscale('log')
            ax.set_xticks([7, 14, 28, 56, 112])
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.xaxis.set_minor_locator(mticker.NullLocator())
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            ax.set_xlabel('Payoff (CHF)')
            if natural:
                ax.set_yticks([1, 3, 10, 30])
                ax.set_ylabel('NATURAL SPACE\nAbsolute noise SD (CHF)'
                              if col == 0 else '')
            else:
                ax.set_yticks([0.1, 0.2, 0.4, 0.8])
                ax.set_ylabel('LOG SPACE\nRelative noise SD (log units)'
                              if col == 0 else '')
            if row == 0:
                ax.set_title(title, fontsize=7.5, linespacing=1.3)

    a = axes[0, 0]
    a.text(.03, .97, 'IPS', transform=a.transAxes, color=IPS, fontsize=7, va='top')
    a.text(.03, .86, 'Vertex', transform=a.transAxes, color=VERTEX, fontsize=7,
           va='top')
    a.plot([], [], color='.35', ls='-', lw=1.6, label='n1 first (remembered)')
    a.plot([], [], color='.35', ls='--', lw=1.6, label='n2 second (seen)')
    a.legend(loc='lower right', fontsize=6.2, handlelength=2.0)

    # -- right column: per-subject exponents, both curves --------------------
    for row, (model, title) in enumerate(FITS):
        ax = axes[row, 2]
        for k, (param, ls) in enumerate(CURVES):
            w = (d[(d.model == model) & (d.param == param)]
                 .pivot_table(index='subject', columns='stim', values='exponent'))
            x0, x1 = 3 * k, 3 * k + 1
            for _, r in w.iterrows():
                ax.plot([x0, x1], [r['vertex'], r['ips']], color='.65', lw=.4,
                        zorder=1)
            ax.plot(np.full(len(w), x0), w['vertex'], 'o', ms=2.6, color=VERTEX,
                    zorder=3)
            ax.plot(np.full(len(w), x1), w['ips'], 'o', ms=2.6, color=IPS, zorder=3)
            ax.plot([x0, x1], [w['vertex'].median(), w['ips'].median()],
                    color='.1', lw=2.0, ls=ls, zorder=4)
            diff = (w['ips'] - w['vertex']).dropna()
            ax.text((x0 + x1) / 2, ax.get_ylim()[1], '', fontsize=6)
            ax.text((x0 + x1) / 2, -0.30, f'{param.split()[0]}\nΔ={diff.mean():+.3f}\n'
                    f'{int((diff < 0).sum())}/{len(diff)} down',
                    fontsize=6.3, ha='center', va='top', linespacing=1.3,
                    transform=ax.get_xaxis_transform())
        ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
        ax.set_xlim(-.7, 4.7)
        ax.set_xticks([0, 1, 3, 4])
        ax.set_xticklabels(['Vtx', 'IPS', 'Vtx', 'IPS'], fontsize=6.5)
        ax.set_ylabel('Power-law exponent')
        ax.set_title(title.split('\n')[0], fontsize=7.5)
        ax.text(4.6, .01, 'Weber', fontsize=6.2, color='.5', ha='right',
                va='bottom')

    sns.despine(fig=fig, offset=3)
    for ax, letter in zip(axes.ravel(), 'abcdef'):
        ax.text(-0.26, 1.14, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    t = (d.groupby(['model', 'param', 'stim'])['exponent'].median().unstack('stim'))
    t['ips_minus_vertex'] = t['ips'] - t['vertex']
    print(t.round(3).to_string())
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
