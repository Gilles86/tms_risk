"""Figure 5: where in the decision space cTBS actually changes behaviour.

The argument runs left to right, and it now starts where the mechanism starts.

    A   What cTBS does to each option, as a fraction of that option's perceived
        value. Both options lose value -- the percept is pulled toward the prior --
        but that is not what changes a choice. A choice changes only if one option
        loses MORE than the other, and column A is where that is legible: under
        "risky first" the two curves lie on top of each other (-4.3% vs -4.0% on
        average, no net pressure on the choice), under "risky second" the safe option
        loses about two points more than the risky one (-5.9% vs -4.0%), because the
        safe option is then the one held in memory.
    B   The consequence of A across the whole decision space: the perceived risky/safe
        expected-value ratio, IPS over vertex. Column A is one option at a time in
        percent; column B is what their difference does to the ratio that drives
        choice.
    C   Where choice is actually sensitive to that ratio.
    D   The behavioural effect, large only where B and C both hold.
    E   That prediction against the data.

Column A is deliberately relative, not absolute. In CHF the shift grows with payoff
(-0.14 CHF at 7, -0.58 CHF at 28) purely because the percepts themselves grow, which
makes an absolute panel say "the effect is biggest at large payoffs" when the underlying
proportional pull is what changed. Every magnitude claim in this project is made on the
relative scale for the same reason -- see notes/reanalysis_handoff.md 2.2, where the
apparent contradiction between the flat CHF noise curve and the probit's low-stake
specificity turns out to be exactly this units artefact. The credible intervals are the
stored CHF intervals divided by the posterior-mean vertex percept of the same cell, so
they are intervals on "fraction of the vertex percept", which is what the axis says.

The map columns are NOT literally multiplicable. `cause` is a dimensionless ratio while
`leverage` is |dP/dm|, a probability per CHF, so their product is not `effect`;
regressing one on the other gives a slope of 0.74, not 1. Each is also averaged over
subjects and draws independently, so by Jensen's inequality none of them is a function
of the others' displayed means -- `leverage` runs about 0.70x the value implied by the
plotted `p_vertex` and `noise_vertex`, and peaks at p = 0.44-0.47 rather than exactly
0.5. The columns are separate views of one mechanism, not an arithmetic chain.

Rows are presentation order. The point of the figure is the contrast BETWEEN the rows,
so every column shares one scale across both rows -- the preprint version gave each of
its twelve panels its own autoscaled colourbar, which makes exactly that comparison
impossible to make by eye.

The indifference contour (vertex P(risky) = 0.5) is drawn on all six maps, flanked by
the 0.20 and 0.80 contours. The 0.5 line is the ridge of the leverage map by
construction: a distortion of the decision variable only moves choices where the
psychometric function is steep, and the psychometric function is steepest at
indifference. Distortions far from that contour are invisible in behaviour however
large they are. The spacing of the three contours is that steepness made visible --
bunched where choice is sensitive, spread where it is not -- which is why the leverage
column and the contour spacing tell the same story.

There is no lattice of "design cells" to mark. The safe payoff is a real 5-level
design factor (7, 10, 14, 20, 28 CHF), but the risky amount was titrated per subject,
so the risky/safe ratio is continuous -- 121 distinct values, roughly uniform over the
plotted range of 1 to 4. Earlier versions drew a 5 x 6 grid of open circles whose
y positions were `pd.qcut` bin means of that continuous cloud; those rows were an
artefact of the binning, not levels anyone was shown. The x ticks now sit on the five
payoffs that were sampled, and the y axis is sampled throughout. Columns A and E use
that same 5-level x, so four of the five columns share one x quantity.

    python -m tms_risk.behavior.scripts.plot_fig5 --label flexible2nf

Reads notes/data/decision_space.<label>.tsv, pmc_percepts_by_order.<label>.tsv,
ppc_delta_by_safe.<label>.tsv and paradigm_payoffs.tsv. No trace needed.

Column E comes from the posterior predictive rather than from a marginal of column D's
map: `plot_ppc_fig3a` simulates choices for the real trials at 200 posterior draws and
aggregates them exactly as the observed statistic is aggregated (within subject, then
the paired IPS - vertex difference, then across subjects). Column D's map is evaluated
on a synthetic grid uniform in the risky/safe ratio, so its marginal answers a slightly
different question and is not what the data should be judged against.

The absolute-CHF companion to column A -- perceived against objective expected value,
both stimulation conditions drawn -- stays available as its own figure:
`plot_percept_distortion`, notes/figures/percepts/.
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
# One quantity, one ink. The indifference contour meant the same thing in every panel
# but was drawn white on mako and dark on RdBu_r, which reads as two different things.
# A single dark grey works on all three maps because the p = 0.5 contour runs through
# the pale middle of the diverging maps and along mako's light high-leverage ridge --
# which is not a coincidence: leverage peaks at indifference.
CONTOUR = '0.15'   # the p = 0.5 indifference contour
FLANK = '0.45'     # its 0.25 / 0.75 flanks, lighter so the hierarchy is obvious
GUIDE = '0.35'     # the risk-neutral reference, dotted so it cannot be confused
# A difference between the two stimulation conditions is a third quantity: it must not
# borrow the IPS red or the vertex green, so everything derived from it takes
# near-black. The model prediction it is checked against takes mid-grey -- and takes it
# literally. An earlier version drew the model line as a value-coloured LineCollection
# on column D's colormap over a dark casing, which (i) read as a filled ribbon with an
# outline, i.e. as a credible interval that the collapsed marginal does not have and
# the stored draws cannot supply, and (ii) spanned only the pale salmon third of a
# colourmap scaled to +-0.13, so it encoded nothing legible anyway. Worse, its direct
# label said "Model" in a grey that appeared nowhere in the panel.
DIFF = '#1a1a1a'
MODEL = '.45'
# Column A separates the two OPTIONS, which is a distinct contrast from the two
# stimulation conditions -- the panel already shows a cTBS difference, so it must not
# reach for the IPS red or the vertex green. Grey and purple carry no other meaning
# anywhere in this figure.
SAFE, RISKY = '#4d4d4d', '#6a3d9a'
DODGE = .6    # CHF, so the two options' credible intervals do not sit on one another

# Titles are hard-wrapped so that no line is wider than its panel -- an overflowing
# title runs straight into the neighbouring column's panel letter.
#
# The two diverging columns take a FIXED half-range rather than one scaled to the
# largest cell. Scaling to the extreme let a single corner set the scale for both rows,
# which pushed the bulk of column D into visible red and made the risky-first map read
# as though large effects were scattered all over it -- its true maximum is +0.079, in
# one corner, against +0.108 for risky second. The effect column's half-range is 0.15,
# the same as column E's y axis, so Δ P(chose risky) is on ONE scale across the figure
# whether it is drawn as colour or as position.
# (key, title, cmap, colour-centre, colourbar ticks, half-range)
SPECS = [
    ('cause', 'Perceived risky/safe\nratio, IPS / vertex', 'RdBu_r', 1.0,
     [.95, 1.00, 1.05], .07),
    # |dP/dm| is a probability over a CHF difference in expected value, so its unit is
    # a probability per CHF, not a bare reciprocal CHF.
    ('leverage', 'Leverage\n(ΔP per CHF)', 'mako', None, [.1, .2, .3, .4], None),
    ('effect', 'Δ P(chose risky)\nIPS − vertex', 'RdBu_r', 0.0, [-.15, 0., .15], .15),
]

XPAD = .9  # CHF of margin, so the end ticks are not flush against the panel edge

# Panel geometry in figure coordinates, laid out so all five panels come out the same
# width (1.09 in). Each 1D panel gets a gutter wide enough for its own y tick labels
# plus the neighbouring column's; the left margin holds the rotated row name as well.
A_L, A_R = .0800, .2305
MAPS_L, MAPS_R = .2884, .7839
E_L, E_R = .8446, .9950
WSPACE = .147               # 0.16 in between neighbouring maps
TOP, BOTTOM = .855, .315
CBAR_Y, CBAR_H = .148, .020
YLAB_X = -.215  # axes-fraction x of the maps' y-label; matplotlib's automatic
#                 placement leaves ~0.3 inch of dead space we cannot afford here
# The two 1D columns key themselves with a real legend, inside the axes, in the corner
# their own data leaves empty. What the shaded and barred regions ARE goes in the
# caption, not in the panel: column A's bars are 95% posterior credible intervals on a
# model quantity, column E's band is a 95% posterior PREDICTIVE interval -- built from
# simulated choices at the real trials, so it carries the trial-level binomial noise
# the observed points are subject to. That last point is why the observed points in
# column E are bare markers: their sampling error is already inside the band, and
# drawing an s.e.m. on the marker too would count the same variability twice.
ATICKS = [-12, -8, -4, 0]
ALIM = (-12.6, 1.0)     # the widest credible interval reaches -12.3%
AKEY = dict(loc='lower left', bbox_to_anchor=(.03, .04))
# Column E: the band reaches 0.127 (risky second, 10 CHF), so the axis has to run past
# 0.10 or it is clipped by the end of the spine.
ETICKS = [-.05, 0., .05, .10, .15]
ELIM = (-.09, .165)
EKEY = dict(loc='upper left', bbox_to_anchor=(.03, .985))


def panel_key(ax, spec):
    """A compact in-axes legend, sized so two entries fit a 1.1 inch panel."""
    leg = ax.legend(frameon=False, fontsize=7, handlelength=1.3, handletextpad=.5,
                    labelspacing=.3, borderpad=0, borderaxespad=0, **spec)
    leg._legend_box.align = 'left'
    return leg


def grid(d, key):
    """Long-format rows -> (ratio x safe payoff) matrix plus its axes."""
    piv = d.pivot_table(index='ratio', columns='n_safe', values=key)
    return piv.columns.values, piv.index.values, piv.values


def design_payoffs(data):
    """The five safe payoffs the design actually used -- the figure's x ticks.

    The risky amount was titrated per subject, so the ratio on the y axis has no
    discrete levels to mark; see the module docstring.
    """
    p = pd.read_csv(data / 'paradigm_payoffs.tsv', sep='\t')
    return np.sort(p.n_safe.unique())


def relative_percepts(data, label):
    """Perceived-value shift per option and safe payoff, in % of the vertex percept.

    Each cell of `pmc_percepts_by_order` stores the shift and its 95% credible interval
    in CHF alongside the vertex percept it is a shift away from; dividing by the latter
    converts all three to a fraction of that percept. See the module docstring for why
    the figure shows the fraction rather than the CHF.
    """
    d = pd.read_csv(data / f'pmc_percepts_by_order.{label}.tsv', sep='\t')
    for c in ['delta', 'lo', 'hi']:
        d['rel_' + c] = 100 * d[c] / d.vertex
    return d.sort_values('n_safe')


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    d = pd.read_csv(data / f'decision_space.{label}.tsv', sep='\t')
    xticks = design_payoffs(data)
    ppc = pd.read_csv(data / f'ppc_delta_by_safe.{label}.tsv', sep='\t')
    perc = relative_percepts(data, label)
    xlim = (xticks.min() - XPAD * 1.6, xticks.max() + XPAD * 1.6)

    fig = plt.figure(figsize=(7.25, 4.4))
    gsa = fig.add_gridspec(2, 1, left=A_L, right=A_R, top=TOP, bottom=BOTTOM,
                           hspace=.17)
    gs = fig.add_gridspec(2, 3, left=MAPS_L, right=MAPS_R, top=TOP, bottom=BOTTOM,
                          wspace=WSPACE, hspace=.17)
    gse = fig.add_gridspec(2, 1, left=E_L, right=E_R, top=TOP, bottom=BOTTOM,
                           hspace=.17)

    # one colour scale per column, so the two rows are directly comparable
    norms = {}
    for key, _t, _cmap, centre, _ct, half in SPECS:
        z = np.concatenate([d[d.order == o][key].values for o in ORDERS])
        if centre is None:
            norms[key] = (np.nanmin(z), np.nanmax(z))
        else:
            c = half if half is not None else np.nanmax(np.abs(z - centre))
            assert np.nanmax(np.abs(z - centre)) <= c, (
                f'{key} runs past its fixed half-range of {c}; widen it rather than '
                f'letting the map clip silently')
            norms[key] = (centre - c, centre + c)

    ims, map_axes, a_axes, e_axes = {}, [], [], []
    for row, order in enumerate(ORDERS):
        o = d[d.order == order]

        # --- column A: what cTBS costs each option, relative to that option's own
        # perceived value. Five design payoffs, so points with credible intervals
        # rather than a continuous band; the two options are dodged apart because
        # under "risky first" they coincide, and coinciding is the finding.
        axa = fig.add_subplot(gsa[row, 0])
        a_axes.append(axa)
        axa.axhline(0, color='.75', lw=.7, ls='--', zorder=0)
        pe_ = perc[perc.order == order]
        opts = {k: g for k, g in pe_.groupby('option')}
        # The gap between the two curves is the whole quantity of interest -- neither
        # curve on its own predicts anything about choice -- so it gets filled in. It
        # is drawn on the true payoffs, not the dodged ones.
        axa.fill_between(opts['safe'].n_safe, opts['safe'].rel_delta,
                         opts['risky'].rel_delta, color='.55', alpha=.14, lw=0,
                         zorder=1)
        for opt, colour, dodge in [('safe', SAFE, -DODGE), ('risky', RISKY, DODGE)]:
            g = opts[opt]
            yerr = np.vstack([g.rel_delta - g.rel_lo, g.rel_hi - g.rel_delta])
            axa.errorbar(g.n_safe.values + dodge, g.rel_delta, yerr=yerr, fmt='-o',
                         color=colour, lw=1.3, ms=3.2, elinewidth=1.,
                         ecolor=mpl.colors.to_rgba(colour, .55), capsize=0, zorder=3,
                         label=f'{opt.capitalize()} option')
        axa.set_xticks(xticks)
        axa.set_xlim(*xlim)
        axa.set_yticks(ATICKS)
        axa.set_ylim(*ALIM)

        for col, (key, title, cmap, _c, _ct, _h) in enumerate(SPECS):
            ax = fig.add_subplot(gs[row, col])
            map_axes.append(ax)
            x, y, z = grid(o, key)
            vmin, vmax = norms[key]
            ims[key] = ax.pcolormesh(x, y, z, cmap=cmap, shading='gouraud',
                                     vmin=vmin, vmax=vmax, rasterized=True)
            _, _, pv = grid(o, 'p_vertex')
            # 0.20 and 0.80 flank the indifference line. Where the three bunch up the
            # psychometric function is steep, where they spread apart it is shallow --
            # so the leverage panel's message becomes visible in every column, without
            # a word of text. Thin and solid rather than dashed, to stay clearly
            # distinct from the dotted risk-neutral line they run near.
            cf = ax.contour(x, y, pv, levels=[.20, .80], colors=FLANK,
                            linewidths=.6, alpha=.9)
            cs = ax.contour(x, y, pv, levels=[.5], colors=CONTOUR, linewidths=1.1)
            # label every contour in every panel; with the wider panels there is room,
            # and it saves the reader carrying the meaning across from one panel
            ax.clabel(cs, fmt={.5: '50%'}, fontsize=5.8, inline=True, inline_spacing=3)
            ax.clabel(cf, fmt={.20: '20%', .80: '80%'}, fontsize=5.4, inline=True,
                      inline_spacing=2)
            # Dotted and thinner than the contour so the reference line and the model
            # output stay distinguishable even where they run close together.
            ax.axhline(RISK_NEUTRAL, color=GUIDE, lw=.7, ls=':', alpha=.9, zorder=3)
            ax.set_xticks(xticks)
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
                ax.yaxis.set_label_coords(YLAB_X, .5)
            else:
                ax.set_yticklabels([])
            sns.despine(ax=ax, offset=3, trim=True)

        # --- column E: model prediction against what participants actually did.
        # A 2D grid of the observed effect was too thin per cell (~25 subjects) to read
        # as evidence; collapsing over the ratio gives 35 subjects per point and an
        # error bar, which is what makes this a usable posterior predictive check.
        axe = fig.add_subplot(gse[row, 0])
        e_axes.append(axe)
        axe.axhline(0, color='.75', lw=.7, ls='--', zorder=0)
        pp = ppc[ppc.order == order].sort_values('n_safe')
        axe.fill_between(pp.n_safe, pp.lo, pp.hi, color=MODEL, alpha=.20, lw=0,
                         zorder=1)
        # created observed-first so it leads the legend; zorder fixes the drawing
        axe.plot(pp.n_safe, pp.observed, 'o', color=DIFF, ms=4., ls='none', zorder=3,
                 label='Observed')
        axe.plot(pp.n_safe, pp['mean'], color=MODEL, lw=1.6, solid_capstyle='round',
                 zorder=2, label='Model')
        axe.set_xticks(xticks)
        axe.set_xlim(*xlim)
        axe.set_yticks(ETICKS)
        axe.set_ylim(*ELIM)

        if row == 0:
            axa.set_title('Perceived value lost\nto cTBS (%)', color='.2', pad=6)
            axe.set_title('Model vs observed\nΔ P(chose risky)', color='.2', pad=6)
            axa.set_xticklabels([])
            axe.set_xticklabels([])
            # Keyed once, on row 0; row 1 inherits by position.
            panel_key(axa, AKEY)
            panel_key(axe, EKEY)
        for a in (axa, axe):
            sns.despine(ax=a, offset=4, trim=True)

    # Every column is a function of the safe payoff, and every column says so. One
    # label centred under the block of maps sits under column C and reads as C's.
    for ax in a_axes[1:] + map_axes[3:6] + e_axes[1:]:
        b = ax.get_position()
        fig.text(b.x0 + b.width / 2, BOTTOM - .108, 'Safe payoff (CHF)', ha='center',
                 va='baseline', fontsize=8.5)

    # one colourbar per map column, aligned to that column and set well below the
    # x-label. Columns A and E need none -- their scale is their y axis.
    for col, (key, _t, _c, _ce, ticks, _h) in enumerate(SPECS):
        b = map_axes[3 + col].get_position()
        cax = fig.add_axes([b.x0, CBAR_Y, b.width, CBAR_H])
        cb = fig.colorbar(ims[key], cax=cax, orientation='horizontal', ticks=ticks)
        cb.outline.set_linewidth(.6)
        cax.tick_params(labelsize=7, length=2, pad=2)

    # Panel letters: one per column, identical offset from the column's left edge and
    # all on one baseline. They sit above the axes, so column A's letter clears its
    # y-axis label (which is vertically centred inside the axes) without extra room.
    letter_y = TOP + .082
    for letter, ax in zip('ABCDE', a_axes[:1] + map_axes[:3] + e_axes[:1]):
        fig.text(ax.get_position().x0 - .014, letter_y, letter, fontsize=11,
                 fontweight='bold', va='bottom', ha='right')

    # Row names, placed from the measured extent of column A's y tick labels rather
    # than a guessed offset, so they can never ride up against them.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    rend = fig.canvas.get_renderer()
    x_row = min(t.get_window_extent(rend).transformed(inv).x0
                for ax in a_axes for t in ax.get_yticklabels() if t.get_text()) - .013
    for row, order in enumerate(ORDERS):
        b = a_axes[row].get_position()
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
        pp = ppc[ppc.order == order]
        pe_ = perc[perc.order == order]
        rel = pe_.groupby('option').rel_delta.mean()
        print(f'  {order:14s} perceived value lost: safe {rel["safe"]:+.2f}%, '
              f'risky {rel["risky"]:+.2f}%  (gap {rel["risky"] - rel["safe"]:+.2f} pts '
              f'in favour of risky)')
        inside = ((pp.observed >= pp.lo) & (pp.observed <= pp.hi)).sum()
        print(f'  {"":14s} model mean Δ P = {pp["mean"].mean():+.4f}, '
              f'observed {pp.observed.mean():+.4f} (peak {pp.observed.max():+.3f}); '
              f'{inside}/{len(pp)} observed points inside the 95% PPC band')
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
