"""Why the choice effect is smaller than the percept shift suggests.

cTBS moves both parts of the choice index — it shifts the numerator (percepts
pulled toward the prior) AND inflates the denominator (the decision function
flattens). On choice probability these oppose each other, which no plot of the
noise function or the perceived ratio can reveal.

a  The decision functions themselves, at the safe payoff where the effect is
   largest. Vertex and IPS, plus the two counterfactuals: what cTBS would do if
   it ONLY shifted the percepts, and if it ONLY flattened the function.
b  The same decomposition as a function of the ratio: ΔP split into its bias and
   scaling parts, which have opposite signs over most of the range.
c  Maps of the two parts over the decision space, on one colour scale.

    python -m tms_risk.behavior.scripts.plot_decision_function
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
BIAS, SCALE, FULL = '#3B5BA5', '#C97B2E', '0.15'

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


def series(d, q, order, n_safe):
    s = d[(d.quantity == q) & (d.order == order) & (d.n_safe == n_safe)]
    return s.sort_values('ratio')


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([1.5, 2, 2.5, 3])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())


def main(data_dir, out_stem, label, order, n_safe):
    d = pd.read_csv(Path(data_dir) /
                    f'decision_function/decision_function.{label}.tsv', **READ)
    fig = plt.figure(figsize=(7.25, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])

    ax = fig.add_subplot(gs[0, 0])
    for q, col, ls, nm in [('p_vertex', VERTEX, '-', 'Vertex'),
                           ('p_full', IPS, '-', 'IPS'),
                           ('p_bias_only', BIAS, (0, (3, 1.4)), 'Bias only'),
                           ('p_scale_only', SCALE, (0, (1.6, 1.4)), 'Scaling only')]:
        s = series(d, q, order, n_safe)
        ax.plot(s.ratio, s['mid'], color=col, ls=ls, lw=1.3, label=nm)
    ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
    logx(ax)
    ax.set_ylim(.1, .95)
    ax.set_xlabel('Risky/safe ratio')
    ax.set_ylabel('P(chose risky)')
    ax.set_title(f'a  Decision functions · {n_safe:.0f} CHF safe', loc='left',
                 fontsize=8)
    ax.legend(loc='lower right', fontsize=6.4, handlelength=1.6,
              labelspacing=.25, borderaxespad=.2)

    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color='0.8', lw=.7, ls='--', zorder=0)
    for q, col, nm in [('dp_full', FULL, 'Full cTBS effect'),
                       ('dp_bias', BIAS, 'Bias part'),
                       ('dp_scale', SCALE, 'Scaling part')]:
        s = series(d, q, order, n_safe)
        ax.fill_between(s.ratio, s.lo, s.hi, color=col, alpha=.15, lw=0)
        ax.plot(s.ratio, s['mid'], color=col, lw=1.4, label=nm)
    logx(ax)
    ax.set_xlabel('Risky/safe ratio')
    ax.set_ylabel('ΔP(chose risky)')
    ax.set_title('b  Decomposition', loc='left', fontsize=8)
    ax.legend(loc='upper right', fontsize=6.4, handlelength=1.4,
              labelspacing=.25, borderaxespad=.2)

    ax = fig.add_subplot(gs[0, 2])
    ax.axhline(1, color='0.8', lw=.7, ls='--', zorder=0)
    s = series(d, 'den_ratio', order, n_safe)
    ax.fill_between(s.ratio, s.lo, s.hi, color=SCALE, alpha=.18, lw=0)
    ax.plot(s.ratio, s['mid'], color=SCALE, lw=1.4)
    ax2 = ax.twinx()
    s = series(d, 'num_shift', order, n_safe)
    ax2.plot(s.ratio, s['mid'], color=BIAS, lw=1.4)
    ax2.axhline(0, color=BIAS, lw=.5, ls=':', alpha=.5)
    ax2.set_ylabel('Numerator shift (log units)', color=BIAS, fontsize=7)
    ax2.tick_params(axis='y', colors=BIAS, labelsize=6.5)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(BIAS)
    logx(ax)
    ax.set_xlabel('Risky/safe ratio')
    ax.set_ylabel('Denominator ratio\nIPS / vertex', color=SCALE, fontsize=7)
    ax.tick_params(axis='y', colors=SCALE)
    ax.set_title('c  What cTBS moves', loc='left', fontsize=8)

    for k, (q, nm) in enumerate([('dp_bias', 'Bias part'),
                                 ('dp_scale', 'Scaling part'),
                                 ('dp_full', 'Net effect')]):
        ax = fig.add_subplot(gs[1, k])
        s = d[(d.quantity == q) & (d.order == order)]
        p = s.pivot_table(index='ratio', columns='n_safe', values='mid')
        m = max(np.abs(d[d.quantity.isin(['dp_bias', 'dp_scale', 'dp_full'])
                         & (d.order == order)]['mid']).max(), 1e-6)
        im = ax.pcolormesh(np.arange(p.shape[1]), p.index.values, p.values,
                           cmap='RdBu_r', vmin=-m, vmax=m, shading='gouraud',
                           rasterized=True)
        ax.set_xticks(np.arange(p.shape[1]))
        ax.set_xticklabels([f'{v:.0f}' for v in p.columns])
        ax.set_yscale('log')
        ax.set_yticks([1.5, 2, 2.5, 3])
        ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
        ax.set_xlabel('Safe payoff (CHF)')
        if k == 0:
            ax.set_ylabel('Risky/safe ratio')
        ax.set_title(f'{"def"[k]}  {nm}', loc='left', fontsize=8)
        if k == 2:
            cb = fig.colorbar(im, ax=ax, fraction=.05, pad=.02)
            cb.set_label('ΔP(chose risky)', fontsize=6.5)
            cb.ax.tick_params(labelsize=5.5)

    fig.suptitle(f'{label} · {order.lower()} · closed form from the posterior',
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
    ap.add_argument('--order', default='Risky second')
    ap.add_argument('--n_safe', default=7.0, type=float)
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/decision_function'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label, a.order, a.n_safe)
