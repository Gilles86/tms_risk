"""Figure 5: where in the decision space cTBS actually changes behaviour.

The argument runs left to right, and the reader should be able to multiply the first
two columns to get the third:

    distortion  x  leverage  =  behavioural effect
      (cause)      (|dP/dm|)     (Delta P(risky))

Rows are presentation order. The point of the figure is the contrast BETWEEN the rows,
so every column shares one colour scale across both rows -- the preprint version gave
each of its twelve panels its own autoscaled colourbar, which makes exactly that
comparison impossible to make by eye.

The indifference contour (vertex P(risky) = 0.5) is drawn on all six panels. It is the
ridge of the leverage map by construction: a distortion of the decision variable only
moves choices where the psychometric function is steep, and the psychometric function
is steepest at indifference. Distortions far from that contour are invisible in
behaviour however large they are.

Open circles mark the 30 (safe payoff x ratio) cells the design actually sampled;
everything off them is model extrapolation.

    python -m tms_risk.behavior.scripts.plot_fig5 --label flexible2nf

Reads notes/data/decision_space.<label>.tsv and paradigm_payoffs.tsv. No trace needed.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')

ORDERS = ['Risky first', 'Risky second']
SPECS = [
    ('cause', 'Perceived risky/safe ratio\nIPS / vertex', 'RdBu_r', 1.0),
    ('leverage', 'Leverage\n(how far a distortion moves choice)', 'mako', None),
    ('effect', 'Δ P(chose risky)\nIPS − vertex', 'RdBu_r', 0.0),
]


def grid(d, key):
    """Long-format rows -> (ratio x safe payoff) matrix plus its axes."""
    piv = d.pivot_table(index='ratio', columns='n_safe', values=key)
    return piv.columns.values, piv.index.values, piv.values


def design_cells(data):
    """The 30 (safe payoff, ratio) combinations the participants actually saw."""
    p = pd.read_csv(data / 'paradigm_payoffs.tsv', sep='\t')
    p['ratio'] = p.n_risky / p.n_safe
    p['bin'] = pd.qcut(p.ratio, 6, labels=False)
    cy = p.groupby('bin').ratio.mean().values
    cx = np.sort(p.n_safe.unique())
    return np.tile(cx, len(cy)), np.repeat(cy, len(cx))


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    d = pd.read_csv(data / f'decision_space.{label}.tsv', sep='\t')
    cells_x, cells_y = design_cells(data)

    fig = plt.figure(figsize=(7.25, 5.15))
    gs = fig.add_gridspec(2, 3, hspace=.13, wspace=.16,
                          left=.085, right=.985, top=.855, bottom=.20)

    # one colour scale per column, so the two rows are directly comparable
    norms = {}
    for key, _, cmap, centre in SPECS:
        z = np.concatenate([d[d.order == o][key].values for o in ORDERS])
        if centre is None:
            norms[key] = (np.nanmin(z), np.nanmax(z))
        else:
            c = np.nanmax(np.abs(z - centre))
            norms[key] = (centre - c, centre + c)

    ims = {}
    for row, order in enumerate(ORDERS):
        o = d[d.order == order]
        for col, (key, title, cmap, _) in enumerate(SPECS):
            ax = fig.add_subplot(gs[row, col])
            x, y, z = grid(o, key)
            vmin, vmax = norms[key]
            ims[key] = ax.pcolormesh(x, y, z, cmap=cmap, shading='gouraud',
                                     vmin=vmin, vmax=vmax, rasterized=True)
            _, _, pv = grid(o, 'p_vertex')
            # matplotlib >= 3.8 removed ContourSet.collections; the set itself is the
            # artist now, so set the outline stroke on it directly.
            cs = ax.contour(x, y, pv, levels=[.5], colors='w', linewidths=1.4)
            cs.set(path_effects=[pe.withStroke(linewidth=2.6, foreground='0.15')])
            ax.scatter(cells_x, cells_y, s=5.5, facecolor='none', edgecolor='w',
                       linewidth=.55, zorder=4, alpha=.85)
            ax.set_xticks([7, 14, 20, 28])
            ax.set_yticks([1, 2, 3, 4])
            if row == 0:
                ax.set_title(title, fontsize=7.8, color='.2', pad=4)
                ax.set_xticklabels([])
            else:
                ax.set_xlabel('Safe payoff (CHF)')
            if col == 0:
                ax.set_ylabel(f'{order}\n\nRisky/safe payoff ratio', fontsize=8.5)
            else:
                ax.set_yticklabels([])
            if row == 1 and col == 0:
                ax.text(8.2, 3.55, 'Indifference', color='w', fontsize=6.2,
                        path_effects=[pe.withStroke(linewidth=2.2, foreground='.15')])
            if row == 1 and col == 2:
                ax.annotate('Effect concentrates below the\ncontour, at small safe payoffs',
                            xy=(8.6, 1.5), xytext=(12.5, 3.45), fontsize=6.6, color='w',
                            ha='left', va='center',
                            path_effects=[pe.withStroke(linewidth=2.2, foreground='.15')],
                            arrowprops=dict(arrowstyle='-', color='w', lw=.9,
                                            connectionstyle='arc3,rad=.25',
                                            path_effects=[pe.withStroke(linewidth=2.2,
                                                                        foreground='.15')]))

    for col, (key, _, _, _) in enumerate(SPECS):
        ref = fig.axes[3 + col]                      # bottom-row axis for this column
        box = ref.get_position()
        cax = fig.add_axes([box.x0, .075, box.width, .022])
        cb = fig.colorbar(ims[key], cax=cax, orientation='horizontal')
        cb.outline.set_linewidth(.6)
        cax.tick_params(labelsize=6.5, length=2)

    for col, letter in enumerate('abc'):
        ax = fig.axes[col]
        ax.text(-.06 if col else -.30, 1.16, letter, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', ha='right')
    fig.suptitle('A distortion only changes behaviour where the psychometric function '
                 'is steep', fontsize=9, y=.955, color='.15')
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf')
    for order in ORDERS:
        o = d[d.order == order]
        lo = o[(o.n_safe <= 10) & (o.ratio <= 1.6)]
        print(f'  {order:14s} peak Δ P(risky) = {o.effect.max():+.4f}; '
              f'mean in the low corner = {lo.effect.mean():+.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    main(args.data_dir, args.label,
         args.out or f'/Users/gdehol/git/tms_risk/notes/figures/fig5.{args.label}')
