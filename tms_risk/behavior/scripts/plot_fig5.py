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
# (key, title, cmap, colour-centre, ink) -- `ink` is the contour/marker colour, chosen
# to read against that column's colormap so no outline stroke is needed.
SPECS = [
    ('cause', 'Perceived risky/safe ratio\nIPS / vertex', 'RdBu_r', 1.0, '0.15'),
    ('leverage', 'Leverage\n(how far a distortion moves choice)', 'mako', None, 'w'),
    ('effect', 'Δ P(chose risky)\nIPS − vertex', 'RdBu_r', 0.0, '0.15'),
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

    obs = pd.read_csv(data / 'behavior_effect_by_safe.tsv', sep='\t')
    fig = plt.figure(figsize=(7.25, 4.3))
    gs = fig.add_gridspec(2, 4, hspace=.13, wspace=.14,
                          left=.075, right=.99, top=.845, bottom=.235)

    # one colour scale per column, so the two rows are directly comparable
    norms = {}
    for key, _, cmap, centre, _ink in SPECS:
        z = np.concatenate([d[d.order == o][key].values for o in ORDERS])
        if centre is None:
            norms[key] = (np.nanmin(z), np.nanmax(z))
        else:
            c = np.nanmax(np.abs(z - centre))
            norms[key] = (centre - c, centre + c)

    ims, d_axes = {}, []
    IPS = '#d62728'
    for row, order in enumerate(ORDERS):
        o = d[d.order == order]
        for col, (key, title, cmap, _, ink) in enumerate(SPECS):
            ax = fig.add_subplot(gs[row, col])
            x, y, z = grid(o, key)
            vmin, vmax = norms[key]
            ims[key] = ax.pcolormesh(x, y, z, cmap=cmap, shading='gouraud',
                                     vmin=vmin, vmax=vmax, rasterized=True)
            _, _, pv = grid(o, 'p_vertex')
            cs = ax.contour(x, y, pv, levels=[.5], colors=ink, linewidths=1.1)
            if row == 0 and col == 0:
                ax.clabel(cs, fmt={.5: 'Indifference'}, fontsize=6, inline=True,
                          inline_spacing=3)
            ax.scatter(cells_x, cells_y, s=5.5, facecolor='none', edgecolor=ink,
                       linewidth=.5, zorder=4, alpha=.7)
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

        # --- fourth column: model prediction against what participants actually did.
        # A 2D grid of the observed effect was too thin per cell (~25 subjects) to read
        # as evidence; collapsing over the ratio gives 35 subjects per point and an
        # error bar, which is what makes this a usable posterior predictive check.
        ax = fig.add_subplot(gs[row, 3])
        ax.axhline(0, color='.75', lw=.7, ls='--', zorder=0)
        mo = o.groupby('n_safe').effect.mean()
        ax.plot(mo.index.values, mo.values, color='.35', lw=1.6, zorder=2)
        ob = obs[obs.order == order].sort_values('n_safe')
        ax.errorbar(ob.n_safe, ob.delta, yerr=ob['sem'], fmt='o', color=IPS, ms=4.2,
                    lw=0, elinewidth=1.1, capsize=0, zorder=3)
        ax.set_xticks([7, 14, 20, 28])
        ax.set_xlim(5, 30)
        ax.yaxis.tick_right(); ax.yaxis.set_label_position('right')
        ax.spines['right'].set_visible(True); ax.spines['left'].set_visible(False)
        if row == 0:
            ax.set_title('Model vs observed\nΔ P(chose risky)', fontsize=7.8,
                         color='.2', pad=4)
            ax.set_xticklabels([])
            ax.text(.05, .06, 'Model', transform=ax.transAxes, fontsize=7, color='.35')
            ax.text(.05, .19, 'Observed', transform=ax.transAxes, fontsize=7, color=IPS)
        else:
            ax.set_xlabel('Safe payoff (CHF)')
        ax.set_ylabel('Δ P(chose risky)', fontsize=8)
        d_axes.append(ax)

    # one colourbar per column; columns 3 and 4 share a scale, so one bar spans both
    lo = min(a.get_ylim()[0] for a in d_axes)
    hi = max(a.get_ylim()[1] for a in d_axes)
    for a in d_axes:
        a.set_ylim(lo, hi)
    bars = [(0, 'cause', 0), (1, 'leverage', 1), (2, 'effect', 2)]
    for col, key, last_col in bars:
        b0 = fig.axes[4 + col].get_position()
        b1 = fig.axes[4 + last_col].get_position()
        cax = fig.add_axes([b0.x0, .105, b1.x1 - b0.x0, .022])
        cb = fig.colorbar(ims[key], cax=cax, orientation='horizontal')
        cb.outline.set_linewidth(.6)
        cax.tick_params(labelsize=6.5, length=2)

    for col, letter in enumerate('abcd'):
        ax = fig.axes[col]
        ax.text(-.06 if col else -.34, 1.17, letter, transform=ax.transAxes,
                fontsize=11, fontweight='bold', va='bottom', ha='right')
    fig.suptitle('A distortion moves choice only where the psychometric function is '
                 'steep — and the data agree', fontsize=9, y=.95, color='.15')
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf')
    for order in ORDERS:
        o = d[d.order == order]
        ob = obs[obs.order == order]
        print(f'  {order:14s} model mean Δ = {o.effect.mean():+.4f}   '
              f'observed mean Δ = {ob.delta.mean():+.4f} '
              f'(peak {ob.delta.max():+.3f} ± {ob.loc[ob.delta.idxmax(), "sem"]:.3f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    main(args.data_dir, args.label,
         args.out or f'/Users/gdehol/git/tms_risk/notes/figures/fig5.{args.label}')
