"""Supplementary figure: exactly which posterior predictive checks, and how models do on them.

The paper originally leaned on eight targeted statistics -- a mean over
risky-second trials, an order contrast, a three-way interaction. Each is
defensible, but they are contrasts someone CHOSE, and eleven of the twenty
fitted models pass all eight, including a model in which cTBS changes nothing
but the magnitude priors. A criterion half the model set passes is not a
criterion.

So this figure reports the design's OWN cells instead. The task used five safe
payoffs; every trial is risky-first or risky-second and IPS or vertex. That
fixes a grid before any data existed, and it is read four ways -- the same
choices, four projections:

    Safe payoff   P(risky) against the five safe payoffs -- WHERE in payoff space
    Ratio         P(risky) against the risky/safe ratio -- WHERE on the
                  psychometric function (the axis of Figure 3a)
    Stake         P(risky) against stake terciles -- how much is at issue
    Slope         the psychometric SLOPE against stake -- the direct signature
                  of a noise change, because noise is what flattens a
                  psychometric function. The three P(risky) views show the
                  effect on choice; only this one shows it is a change in
                  DISCRIMINABILITY rather than a shift in preference.

Panels a-h plot the cells themselves: the reported model's IPS (red) and
vertex (green) curves, each with its own 95% posterior predictive interval,
against the observed proportion (or slope), one row per presentation order.
Nothing is differenced -- a cTBS effect has to show up as the red curve sitting
above or below the green one at the same x, the way it would in Figure 3a, not
be inferred from a contrast plot. Panel i is the one difference that remains:
it pools the paired IPS - vertex contrast (computed per posterior draw, so its
interval is a genuine predictive interval for the contrast, not the overlap of
two marginals) over all 34 design cells and shows how far the reported model
and its alternatives get -- the part that actually separates them.

    python -m tms_risk.behavior.scripts.plot_supp_ppc
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tms_risk.behavior.scripts.ppc_design_grid import VIEWS, load, covered

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
ORDERS = ['Risky first', 'Risky second']
#: the one hard-coded palette rule in this repo: IPS (stimulated) is red,
#: vertex (sham) is green, always -- never inverted, never used for order.
#: This knowingly overrides the usual "avoid pure red/green" guidance for
#: deuteranopia: every IPS-vs-vertex figure in the paper uses this pair, and
#: that cross-figure consistency matters more here than any one figure's
#: colourblind-safety -- both colours are also distinguishable from the
#: gray/black ink used elsewhere (coverage bars, text) by luminance alone.
IPS, VERTEX = '#d62728', '#2ca02c'
STIM_COLOR = {'ips': IPS, 'vertex': VERTEX}
STIM_LABEL = {'ips': 'IPS', 'vertex': 'Vertex'}

#: local view metadata for the LEVEL panels (a-h). `group` is the categorical
#: column shared by both stimulation arms within one order (so the two curves
#: line up at the same x); `tick` is the column averaged for the tick label
#: text -- continuous (frac, stake_chf) rather than the bin index itself.
VIEW_META = {
    'safe':  dict(group='n_safe',    tick='n_safe',    yq='model',
                  xlabel='Safe payoff (CHF)', title='By safe payoff',
                  ylabel='P(risky)', chance=True),
    'rung':  dict(group='rung',      tick='frac',       yq='model',
                  xlabel='Risky / safe payoff', title='By risky / safe ratio',
                  ylabel='P(risky)', chance=True),
    'stake': dict(group='stake_bin', tick='stake_chf',  yq='model',
                  xlabel='Stake (CHF)', title='By stake',
                  ylabel='P(risky)', chance=True),
    'slope': dict(group='stake_bin', tick='stake_chf',  yq='model',
                  xlabel='Stake (CHF)', title='Psychometric slope, by stake',
                  ylabel='Slope', chance=False),
}
VIEW_ORDER = ['safe', 'rung', 'stake', 'slope']

#: the placement ladder, richest first. Names are the paper's wording.
LADDER = [
    ('log-power-percmempmu.mapjitter.klw', 'Perceptual + memory noise,\nprior means'),
    ('log-power-percpmu.mapjitter.klw',    'Perceptual noise + prior means'),
    ('log-power-percmem.mapjitter.klw',    'Perceptual + memory noise'),
    ('log-power-perc.mapjitter.klw',       'Perceptual noise only'),
    ('log-power-spmu.mapjitter.klw',       'Prior means only,\nno noise change'),
    ('log-power-mem.mapjitter.klw',        'Memory noise only'),
    ('log-power-null.klw',                 'No cTBS effect'),
]
REPORTED = 'log-power-percpmu.mapjitter.klw'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})


def fmt_tick(v):
    return f'{v:.0f}' if v >= 5 else f'{v:.1f}'.rstrip('0').rstrip('.')


def read_level(dd, label, view):
    """The per-cell table for one view, both stimulation arms, level (not contrast).

    `ppc_design_grid.load()` exists for the *delta* tables -- for `slope` it
    actively collapses the two stimulation arms into their contrast, which is
    exactly what these panels must NOT show. So this reads the raw per-view
    TSV directly and, for `slope` only, renames its model column (`slope`) to
    `model` so the two branches share one plotting function.
    """
    f = dd / f'ppc_anchor.{view}.{label}.tsv'
    if not f.exists():
        raise SystemExit(f'no {view} table for {label}\n  expected {f}')
    d = pd.read_csv(f, **READ)
    return d.rename(columns={'slope': 'model'}) if view == 'slope' else d


def level_panel(ax, d, view, order, show_ylab, show_xlab, show_title):
    """One (order, view) cell: IPS and vertex curves, band, and observed data."""
    meta = VIEW_META[view]
    sub = d[d.order == order]
    cats = sorted(sub[meta['group']].unique())
    x = np.arange(len(cats))
    ok = n_cell = 0
    if meta['chance']:
        ax.axhline(.5, color='.82', lw=.6, ls='--', zorder=0)
    for stim in ('vertex', 'ips'):          # vertex first, IPS drawn on top
        s = sub[sub.stim == stim].set_index(meta['group']).reindex(cats)
        col = STIM_COLOR[stim]
        ax.fill_between(x, s.lo, s.hi, color=col, alpha=.20, lw=0, zorder=1)
        ax.plot(x, s[meta['yq']], '-', color=col, lw=1.3, zorder=2)
        # NO error bar on the observed point. Every uncertainty in this
        # project is a posterior or posterior-predictive interval, never an
        # s.e.m. on data -- and here an s.e.m. would answer a question nobody
        # asked: the check is whether the observed proportion falls inside the
        # MODEL's predictive band, which the band already shows.
        #
        # A point OUTSIDE its band is ringed, because a reader will count the
        # misses whether or not the figure invites them to, and a miss that
        # has to be judged by eye against a translucent band is a miss the
        # figure has left ambiguous.
        inside = ((s.observed >= s.lo) & (s.observed <= s.hi)).values
        ok += int(inside.sum())
        n_cell += int(len(inside))
        ax.plot(x[inside], s.observed.values[inside], 'o', color=col, ms=4.0,
                mec='0.15', mew=.5, lw=0, zorder=4)
        ax.plot(x[~inside], s.observed.values[~inside], 'o', color=col,
                ms=4.7, mec='0.15', mew=.9, lw=0, zorder=5)
    ticklabs = sub.groupby(meta['group'])[meta['tick']].mean().reindex(cats)
    ax.set_xticks(x)
    ax.set_xlim(x[0] - .55, x[-1] + .55)
    if show_xlab:
        ax.set_xticklabels([fmt_tick(v) for v in ticklabs])
        ax.set_xlabel(meta['xlabel'])
    else:
        ax.set_xticklabels([])
    # per-panel LEVEL coverage. Panel i counts the paired CONTRAST over the
    # same choices, which is a different quantity with a different
    # denominator, and without both numbers on the page a reader who counts
    # the rings here and compares them with panel i is simply misled.
    ax.text(.97, .04, f'{ok}/{n_cell}', transform=ax.transAxes, fontsize=6.3,
            va='bottom', ha='right',
            color='0.4' if ok == n_cell else '#8c2d2d')
    if show_title:
        ax.set_title(meta['title'], fontsize=8, loc='left', x=0)
    if show_ylab:
        # order is never a hue -- it is already the row -- so it is named in
        # words here, folded into the one y-label the row shows, rather than
        # as a second rotated text next to the first (the two collided)
        ax.set_ylabel(f'{order}\n{meta["ylabel"]}')


def level_coverage(dd, lab):
    """How many of the 68 LEVEL cells this model covers."""
    ok = n = 0
    for v in VIEWS:
        f = dd / f'ppc_anchor.{v}.{lab}.tsv'
        if not f.exists():
            continue
        d = pd.read_csv(f, **READ)
        ok += int(((d.observed >= d.lo) & (d.observed <= d.hi)).sum())
        n += len(d)
    return ok, n


def coverage_panel(ax, dd, ladder):
    """Coverage per model, on the contrast and on the levels.

    Both are shown because the comparison between them IS the result. The
    levels are what panels a-h plot and the intuitive thing to count, but they
    are dominated by the psychometric curve itself, which every model in the
    ladder fits: they span 53 to 58 of 68, and the model in which cTBS changes
    nothing but the priors comes FIRST. Ranking by them would actively
    mislead. The paired IPS - vertex contrast removes the part every model
    gets right and leaves the part that separates them, and that is what the
    ordering here uses.
    """
    rows = []
    for lab, nm in ladder:
        ok = n = 0
        for v in VIEWS:
            try:
                _, d = load(dd, lab, v)
            except SystemExit:
                continue
            ok += int(covered(d).sum())
            n += len(d)
        if n:
            rows.append((nm, ok, n) + level_coverage(dd, lab)
                        + (lab == REPORTED,))
    if not rows:
        raise SystemExit('no design-grid tables for the ladder')
    y = np.arange(len(rows))[::-1]
    total, ltotal = rows[0][2], rows[0][4]
    # both on ONE scale, as a percentage: the two denominators differ (34 and
    # 68), so counts could not share an axis, and the point of putting them
    # side by side is that they are directly comparable
    for yy, (nm, ok, n, lok, ln, is_ref) in zip(y, rows):
        c = '0.15' if is_ref else '0.55'
        ax.barh(yy + .21, 100 * ok / n, height=.40, color=c, zorder=2)
        ax.barh(yy - .21, 100 * lok / ln, height=.40, color=c, alpha=.32,
                zorder=2)
        ax.text(2.0, yy + .21, f'{ok}/{n}', ha='left', va='center',
                fontsize=6.0, color='w', zorder=3)
        ax.text(2.0, yy - .21, f'{lok}/{ln}', ha='left', va='center',
                fontsize=6.0, color='0.25', zorder=3)
        ax.text(103, yy, nm, fontsize=6.8, va='center', color=c,
                linespacing=1.25)
    ax.set_yticks([])
    ax.axvline(100, color='0.75', lw=.8, ls=(0, (3, 2)), zorder=1)
    ax.set_xlim(0, 138)
    ax.set_xticks([0, 50, 100])
    ax.set_ylim(-.8, len(rows) - .2)
    ax.set_xlabel('Cells covered (%)')
    ax.set_title('The contrast separates the models; the levels do not',
                 fontsize=8, loc='left', x=.0)
    return [(nm, ok, n, is_ref) for nm, ok, n, lok, ln, is_ref in rows]


def main(data_dir, out_stem, label):
    dd = Path(data_dir) / 'ppc_anchor'

    tabs = {v: read_level(dd, label, v) for v in VIEW_ORDER}
    ylims = {}
    for v, d in tabs.items():
        lo, hi = float(d.lo.min()), float(d.hi.max())
        pad = .09 * (hi - lo)
        lo2, hi2 = max(0., lo - pad), hi + pad
        if v != 'slope':
            hi2 = min(1., hi2)
        ylims[v] = (lo2, hi2)

    fig = plt.figure(figsize=(7.2, 5.4))
    # two independent gridspecs, not one three-row grid: row 1 needs x-tick
    # labels underneath it and row 2 (coverage + key) needs its own title, so
    # the gap below row 1 has to be much bigger than the gap between rows 0
    # and 1 (which carries no labels at all). One `hspace` cannot do both.
    gs_top = fig.add_gridspec(2, 4, left=.095, right=.985, top=.945, bottom=.445,
                              hspace=.13, wspace=.5)
    gs_bot = fig.add_gridspec(1, 4, left=.095, right=.985, top=.315, bottom=.075,
                              wspace=.5)

    grid_axes = []
    for r, order in enumerate(ORDERS):
        row_axes = []
        for c, v in enumerate(VIEW_ORDER):
            ax = fig.add_subplot(gs_top[r, c])
            level_panel(ax, tabs[v], v, order, show_ylab=(c == 0),
                       show_xlab=(r == 1), show_title=(r == 0))
            ax.set_ylim(*ylims[v])
            row_axes.append(ax)
        grid_axes.append(row_axes)

    # direct colour label, once -- every other panel reuses the same mapping,
    # per house style (never re-explain a hue that already means something)
    a0 = grid_axes[0][0]
    a0.text(.97, .93, 'IPS', color=IPS, fontsize=7.2, ha='right', va='top',
           fontweight='bold', transform=a0.transAxes)
    a0.text(.97, .80, 'Vertex', color=VERTEX, fontsize=7.2, ha='right', va='top',
           fontweight='bold', transform=a0.transAxes)

    axe = fig.add_subplot(gs_bot[0, :2])
    rows = coverage_panel(axe, dd, LADDER)
    for nm, ok, n, is_ref in rows:
        print(f'  {ok:>2}/{n}  {nm.replace(chr(10), " ")}')

    # inline key, drawn as the marks themselves rather than described in words
    axk = fig.add_subplot(gs_bot[0, 2:])
    axk.axis('off')
    entries = [
        ('bar_full', 'Panel i: IPS − vertex contrast (of 34)'),
        ('bar_pale', 'Panel i: levels (of 68)'),
        ('line', 'Model median'),
        ('band', '95% posterior predictive interval'),
        ('marker', 'Observed, inside the interval'),
        ('miss', 'Observed, outside it'),
    ]
    y0, dy = .97, .118
    for i, (kind, nm) in enumerate(entries):
        yy = y0 - i * dy
        if kind in ('bar_full', 'bar_pale'):
            axk.add_patch(plt.Rectangle((.03, yy - .026), .09, .052,
                          transform=axk.transAxes, facecolor='0.35',
                          alpha=1.0 if kind == 'bar_full' else .32,
                          lw=0, clip_on=False))
        elif kind == 'band':
            axk.add_patch(plt.Rectangle((.03, yy - .038), .09, .076,
                          transform=axk.transAxes, facecolor='0.4', alpha=.20,
                          lw=0, clip_on=False))
        elif kind == 'miss':
            axk.plot([.075], [yy], 'o', color='0.35', ms=4.7, mec='0.15',
                     mew=.9, lw=0, transform=axk.transAxes, clip_on=False)
        elif kind == 'marker':
            # the glyph must match the mark it names, and the observed points
            # no longer carry an error bar
            axk.plot([.075], [yy], 'o', color='0.35', ms=4.0, mec='0.15',
                     mew=.5, lw=0, transform=axk.transAxes, clip_on=False)
        else:
            axk.plot([.03, .12], [yy, yy], transform=axk.transAxes, color='0.35',
                    lw=1.3, solid_capstyle='butt', clip_on=False)
        axk.text(.16, yy, nm, transform=axk.transAxes, fontsize=6.8,
                 va='center', color='0.3')
    yy = y0 - len(entries) * dy - .04
    axk.text(.02, yy,
             'Panels a-h show LEVELS: the reported model\u2019s IPS and vertex\n'
             'curves against the design\u2019s own cells, and the count in each\n'
             'panel is how many of those 68 observed proportions fall inside\n'
             'their own interval.\n'
             'Panel i counts something else \u2014 the paired IPS \u2212 vertex\n'
             'CONTRAST, computed per posterior draw, over the 34 cells that\n'
             'contrast defines. The two denominators are not the same.\n'
             'No cell is binned, ranked or chosen post hoc.',
             transform=axk.transAxes, fontsize=6.5, va='top', color='0.45',
             linespacing=1.5)

    for axrow in grid_axes:
        for ax in axrow:
            sns.despine(ax=ax, offset=4, trim=True)
    sns.despine(ax=axe, offset=4, left=True)

    fig.canvas.draw()
    letters = 'abcdefghi'
    all_lettered = grid_axes[0] + grid_axes[1] + [axe]
    for letter, ax in zip(letters, all_lettered):
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.transFigure.inverted())
        fig.text(bb.x0 - .014, min(bb.y1 + .018, .997), letter, fontsize=9,
                 fontweight='bold', family='Arial', va='top', ha='left')
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--label', default=REPORTED)
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/SUPP_ppc_grid'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.label)
