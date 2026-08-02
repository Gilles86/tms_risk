"""Figure 5, as 1D curves rather than heatmaps.

Same argument as the preprint's Figure 5, same quantities, but plotted against the
risky/safe ratio -- the psychometric axis -- with one curve per safe payoff. Read it
down a column:

    row 1   P(chose risky) under vertex. Where a curve crosses 0.5 the participant is
            indifferent; those crossings are the 1D form of the preprint's
            indifference contour, and they are marked and carried down the column.
    row 2   Leverage, |dP/dm|: how far a given distortion of the decision variable
            moves choice. It peaks at indifference by construction, which is why the
            crossings from row 1 line up with the peaks here.
    row 3   The resulting cTBS effect on P(chose risky). It is large only where a
            distortion exists AND leverage is high.

Columns are presentation order, and both share every axis, so the row-3 contrast
between columns is the order effect.

    python -m tms_risk.behavior.scripts.plot_fig5_1d --label flexible2nf

Reads notes/data/decision_space.<label>.tsv. No trace needed.
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
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')

ORDERS = ['Risky first', 'Risky second']
SAFES = [7., 10., 14., 20., 28.]          # the levels the design actually sampled
PANEL = dict(fontsize=11, fontweight='bold', va='bottom', ha='right')


def indifference_ratio(g):
    """Ratio at which vertex P(risky) crosses 0.5, by linear interpolation."""
    g = g.sort_values('ratio')
    p, r = g.p_vertex.values, g.ratio.values
    if p.min() > .5 or p.max() < .5:
        return np.nan
    return float(np.interp(.5, p, r))


def main(data_dir, label, out_stem):
    d = pd.read_csv(Path(data_dir) / f'decision_space.{label}.tsv', sep='\t')
    # snap the model grid onto the five sampled safe payoffs
    grid = np.sort(d.n_safe.unique())
    keep = {s: grid[np.abs(grid - s).argmin()] for s in SAFES}
    d = d[d.n_safe.isin(keep.values())].copy()
    d['safe'] = d.n_safe.map({v: k for k, v in keep.items()})

    cmap = sns.color_palette('mako', as_cmap=True)
    cols = {s: cmap(t) for s, t in zip(SAFES, np.linspace(.78, .18, len(SAFES)))}

    fig = plt.figure(figsize=(7.25, 6.2))
    gs = fig.add_gridspec(3, 2, hspace=.22, wspace=.10,
                          left=.10, right=.86, top=.93, bottom=.075)

    ind = {(o, s): indifference_ratio(g)
           for (o, s), g in d.groupby(['order', 'safe'])}

    ROWS = [('p_vertex', 'P(chose risky)\nvertex stimulation'),
            ('leverage', 'Leverage\n|dP/dm| at vertex'),
            ('effect', 'Δ P(chose risky)\nIPS − vertex')]
    axes = np.empty((3, 2), dtype=object)
    for r, (key, ylab) in enumerate(ROWS):
        for c, order in enumerate(ORDERS):
            ax = fig.add_subplot(gs[r, c]); axes[r, c] = ax
            o = d[d.order == order]
            if key == 'p_vertex':
                ax.axhline(.5, color='.6', lw=.7, ls='--', zorder=0)
            if key == 'effect':
                ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
            for s in SAFES:
                g = o[o.safe == s].sort_values('ratio')
                ax.plot(g.ratio, g[key], color=cols[s], lw=1.4, zorder=3)
                # the indifference ratio, carried down every row
                x0 = ind.get((order, s), np.nan)
                if np.isfinite(x0):
                    y0 = float(np.interp(x0, g.ratio, g[key]))
                    ax.plot([x0], [y0], 'o', color=cols[s], ms=3.6, mfc='white',
                            mew=1.1, zorder=4)
            ax.set_xlim(1, 4)
            ax.set_xticks([1, 2, 3, 4])
            if c == 0:
                ax.set_ylabel(ylab)
            else:
                ax.set_yticklabels([])
            if r == 0:
                ax.set_title(order, fontsize=8.5, color='.2', pad=4)
            if r < 2:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Risky / safe payoff ratio')
        lo = min(axes[r, c].get_ylim()[0] for c in range(2))
        hi = max(axes[r, c].get_ylim()[1] for c in range(2))
        for c in range(2):
            axes[r, c].set_ylim(lo, hi)

    axes[0, 0].text(.045, .93, 'Open circles: indifference', transform=axes[0, 0].transAxes,
                    fontsize=6.6, color='.35', va='top')

    # colour key for safe payoff, as a direct-labelled strip rather than a legend
    cax = fig.add_axes([.875, .40, .016, .30])
    for i, s in enumerate(SAFES):
        cax.add_patch(plt.Rectangle((0, i), 1, 1, color=cols[s], lw=0))
        cax.text(1.5, i + .5, f'{s:.0f}', fontsize=7, va='center', ha='left',
                 color='.2')
    cax.set_xlim(0, 1); cax.set_ylim(0, len(SAFES))
    cax.axis('off')
    cax.text(.5, len(SAFES) + .55, 'Safe\npayoff\n(CHF)', fontsize=7, ha='center',
             va='bottom', color='.2', linespacing=1.25)

    for r, letter in enumerate('abc'):
        axes[r, 0].text(-.17, 1.02, letter, transform=axes[r, 0].transAxes, **PANEL)
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf')
    for order in ORDERS:
        o = d[d.order == order]
        print(f'  {order:14s} peak Δ P(risky) = {o.effect.max():+.4f}  '
              f'at safe = {o.loc[o.effect.idxmax(), "safe"]:.0f} CHF, '
              f'ratio = {o.loc[o.effect.idxmax(), "ratio"]:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    a = parser.parse_args()
    main(a.data_dir, a.label,
         a.out or f'/Users/gdehol/git/tms_risk/notes/figures/fig5_1d.{a.label}')
