"""Figure 5: where in the decision space cTBS actually changes behaviour.

The argument runs left to right: a distortion of perceived value (column a) matters
only where choice is sensitive to it (column b), and the behavioural effect (column c)
is large only where both hold. Column d checks that prediction against the data.

The columns are NOT literally multiplicable. `cause` is a dimensionless ratio
(perceived risky/safe EV, IPS over vertex) while `leverage` is |dP/dm| in units of
1/CHF, so their product is not `effect`; regressing one on the other gives a slope of
0.74, not 1. Each is also averaged over subjects and draws independently, so by
Jensen's inequality none of them is a function of the others' displayed means --
`leverage` runs about 0.70x the value implied by the plotted `p_vertex` and
`noise_vertex`, and peaks at p = 0.44-0.47 rather than exactly 0.5. The columns are
three separate views of one mechanism, not an arithmetic chain.

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
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8,
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
# The risk-neutral ratio: EV_risky = 0.55 * n_risky equals EV_safe = n_safe at
# n_risky/n_safe = 1/0.55. Left of it choosing risky is risk-seeking, right of it
# choosing safe is risk-averse, so it turns the axis from arbitrary into
# interpretable. The fitted indifference points (1/RNP) straddle it.
RISK_NEUTRAL = 1 / 0.55
# A difference between the two stimulation conditions is a third quantity: it must not
# borrow the IPS red or the vertex green, so everything derived from it takes
# near-black. The model prediction it is checked against takes mid-grey.
DIFF = '#1a1a1a'
MODEL = '.5'

# Titles are hard-wrapped so that no line is wider than its panel -- an overflowing
# title runs straight into the neighbouring column's panel letter.
# (key, title, cmap, colour-centre, ink, colourbar ticks) -- `ink` is the colour of
# every line and marker drawn over that column's map (indifference contour, design
# cells, risk-neutral reference), chosen to read against its colormap so no outline
# stroke is needed.
SPECS = [
    ('cause', 'Perceived risky/safe\nratio, IPS / vertex', 'RdBu_r', 1.0, '.15',
     [.95, 1.00, 1.05]),
    ('leverage', 'Leverage\n|∂P/∂m| (1/CHF)', 'mako', None, 'w', [.1, .2, .3, .4]),
    ('effect', 'Δ P(chose risky)\nIPS − vertex', 'RdBu_r', 0.0, '.15', [-.1, 0., .1]),
]

XTICKS = [7, 14, 20, 28]
XPAD = .9  # CHF of margin, so the design-cell markers at 7 and 28 are not clipped

# Panel geometry in figure coordinates. The block of 2D maps and the 1D panel get a
# wide gutter between them so that column d's y-axis never crowds column c; the left
# margin has to hold the rotated row name as well as the y-axis label.
MAPS_L, MAPS_R = .088, .700
D_L, D_R = .766, .990
TOP, BOTTOM = .855, .315
CBAR_Y, CBAR_H = .120, .020


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

    fig = plt.figure(figsize=(7.25, 4.4))
    gs = fig.add_gridspec(2, 3, left=MAPS_L, right=MAPS_R, top=TOP, bottom=BOTTOM,
                          wspace=.17, hspace=.17)
    gsd = fig.add_gridspec(2, 1, left=D_L, right=D_R, top=TOP, bottom=BOTTOM,
                           hspace=.17)

    # one colour scale per column, so the two rows are directly comparable
    norms = {}
    for key, _t, _cmap, centre, _ink, _ct in SPECS:
        z = np.concatenate([d[d.order == o][key].values for o in ORDERS])
        if centre is None:
            norms[key] = (np.nanmin(z), np.nanmax(z))
        else:
            c = np.nanmax(np.abs(z - centre))
            norms[key] = (centre - c, centre + c)

    ims, map_axes, d_axes = {}, [], []
    for row, order in enumerate(ORDERS):
        o = d[d.order == order]
        for col, (key, title, cmap, _c, ink, _ct) in enumerate(SPECS):
            ax = fig.add_subplot(gs[row, col])
            map_axes.append(ax)
            x, y, z = grid(o, key)
            vmin, vmax = norms[key]
            ims[key] = ax.pcolormesh(x, y, z, cmap=cmap, shading='gouraud',
                                     vmin=vmin, vmax=vmax, rasterized=True)
            _, _, pv = grid(o, 'p_vertex')
            ax.contour(x, y, pv, levels=[.5], colors=ink, linewidths=1.1)
            # Dotted and thinner than the contour so the reference line and the model
            # output stay distinguishable even where they run close together.
            ax.axhline(RISK_NEUTRAL, color=ink, lw=.7, ls=':', alpha=.75, zorder=3)
            ax.scatter(cells_x, cells_y, s=5.5, facecolor='none', edgecolor=ink,
                       linewidth=.5, zorder=4, alpha=.7)
            # The contour is named once, in the one band of the panel that carries no
            # design cells (between the ratio-2.84 and ratio-3.63 rows), with a leader
            # down to the line: an inline contour label sits on top of the cells.
            if row == 0 and col == 0:
                ax.annotate('Indifference', xy=(13.4, 2.07), xytext=(9.2, 3.25),
                            fontsize=6.5, color='.15', ha='left', va='center',
                            arrowprops=dict(arrowstyle='-', color='.35', lw=.6,
                                            connectionstyle='arc3,rad=-.25',
                                            shrinkA=2, shrinkB=1))
            ax.set_xticks(XTICKS)
            ax.set_yticks([1, 2, 3, 4])
            ax.set_xlim(x.min() - XPAD, x.max() + XPAD)
            ax.set_ylim(y.min(), y.max())
            if row == 0:
                ax.set_title(title, color='.2', pad=4)
                ax.set_xticklabels([])
            if col == 0:
                # short enough to fit inside the panel height -- a longer label
                # overhangs the axes and runs into the other row's copy
                ax.set_ylabel('Risky/safe ratio')
            else:
                ax.set_yticklabels([])
            sns.despine(ax=ax, offset=3, trim=True)

        # --- fourth column: model prediction against what participants actually did.
        # A 2D grid of the observed effect was too thin per cell (~25 subjects) to read
        # as evidence; collapsing over the ratio gives 35 subjects per point and an
        # error bar, which is what makes this a usable posterior predictive check.
        ax = fig.add_subplot(gsd[row, 0])
        ax.axhline(0, color='.75', lw=.7, ls='--', zorder=0)
        mo = o.groupby('n_safe').effect.mean()
        ax.plot(mo.index.values, mo.values, color=MODEL, lw=1.6, zorder=2)
        ob = obs[obs.order == order].sort_values('n_safe')
        ax.errorbar(ob.n_safe, ob.delta, yerr=ob['sem'], fmt='o', color=DIFF, ms=4.2,
                    lw=0, elinewidth=1.1, capsize=0, zorder=3)
        ax.set_xticks(XTICKS)
        ax.set_xlim(7 - XPAD * 1.6, 28 + XPAD * 1.6)
        ax.set_yticks([-.05, 0, .05, .10])
        ax.set_ylim(-.09, .16)
        ax.set_ylabel('Δ P(chose risky)')
        if row == 0:
            ax.set_title('Model vs observed', color='.2', pad=4)
            ax.set_xticklabels([])
            # Direct labels rather than a legend, in the headroom above the row-0 data
            # (which peaks at 0.085, at 28 CHF, on the far side of the panel).
            ax.text(.03, .97, 'Observed', transform=ax.transAxes, fontsize=7,
                    color=DIFF, va='top')
            ax.text(.03, .855, 'Model', transform=ax.transAxes, fontsize=7,
                    color=MODEL, va='top')
        sns.despine(ax=ax, offset=4, trim=True)
        d_axes.append(ax)

    # one x-label for the three maps and one for column d, instead of four copies
    for x in [(MAPS_L + MAPS_R) / 2, (D_L + D_R) / 2]:
        fig.text(x, BOTTOM - .108, 'Safe payoff (CHF)', ha='center', va='baseline',
                 fontsize=8.5)

    # one colourbar per column, aligned to that column and set well below the x-label
    for col, (key, _t, _c, _ce, _i, ticks) in enumerate(SPECS):
        b = map_axes[3 + col].get_position()
        cax = fig.add_axes([b.x0, CBAR_Y, b.width, CBAR_H])
        cb = fig.colorbar(ims[key], cax=cax, orientation='horizontal', ticks=ticks)
        cb.outline.set_linewidth(.6)
        cax.tick_params(labelsize=7, length=2, pad=2)

    # Panel letters: one per column, identical offset from the column's left edge and
    # all on one baseline. They sit above the axes, so column a's letter clears its
    # y-axis label (which is vertically centred inside the axes) without extra room.
    letter_y = TOP + .082
    for letter, ax in zip('abcd', map_axes[:3] + d_axes[:1]):
        fig.text(ax.get_position().x0 - .016, letter_y, letter, fontsize=11,
                 fontweight='bold', va='bottom', ha='right')

    # Row names, placed from the measured extent of column a's y-axis label rather
    # than a guessed offset, so they can never ride up against it.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()
    x_row = min(ax.yaxis.label.get_window_extent(rend).transformed(inv).x0
                for ax in map_axes[::3]) - .011
    for row, order in enumerate(ORDERS):
        b = map_axes[3 * row].get_position()
        t = fig.text(x_row, b.y0 + b.height / 2, order, rotation=90, ha='center',
                     va='center', fontsize=9, color='.15')
        # ha/va on rotated text is unreliable; nudge by the measured overhang instead
        fig.canvas.draw()
        t.set_x(x_row - (t.get_window_extent(rend).transformed(inv).x1 - x_row))

    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)

    print(f'wrote {out_stem}.pdf')
    for order in ORDERS:
        o = d[d.order == order]
        ob = obs[obs.order == order]
        print(f'  {order:14s} model mean Δ = {o.effect.mean():+.4f}   '
              f'observed mean Δ = {ob.delta.mean():+.4f} '
              f'(peak {ob.delta.max():+.3f} ± {ob.loc[ob.delta.idxmax(), "sem"]:.3f})')
    return fig


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    plt.close(main(args.data_dir, args.label,
                   args.out or
                   f'/Users/gdehol/git/tms_risk/notes/figures/fig5.{args.label}'))
