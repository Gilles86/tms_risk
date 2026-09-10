"""Figure 3: what cTBS did to the psychometric function, from the probit fits alone.

This IS the model-free figure. A probit assumes nothing beyond "there is a
psychophysical curve"; it is not a cognitive model, so nothing in this figure
depends on the PMC, on bauer, or on any trace.

Laid out like the published Figure 3: two blocks, the psychometric functions on the
left (A) and the psychophysical parameters on the right (B), presentation order as
the row variable throughout, so the cTBS effect reads as the same vertical
displacement in both blocks.

The probit has exactly two parameters, and they correspond to the two competing
explanations of the effect:

    slope  choice consistency. A pure loss of consistency -- the "flattening" account
           -- moves this and nothing else.
    RNP    the risk-neutral probability, where the psychometric crosses 0.5. A higher
           RNP means the subject is indifferent at a *smaller* risky/safe ratio, i.e.
           more risk-seeking. A pure change in preference moves this and nothing else.

Both move, and both only when the risky option came second, so neither pure account
covers it. That is the observation the cognitive model has to explain.

Block B draws the *paired* (chain, draw) difference posterior, IPS - vertex, and not
the two per-condition marginals the published figure mirrored against each other.
The marginals are strongly correlated across draws, so their overlap badly overstates
the uncertainty of the difference: for RNP with the risky option second the marginals
overlap over most of their range while the difference excludes zero at p < 0.001. A
reader of the published panel would have to take that on trust. Here the drawn
quantity, the credible interval and the p-value are all the same object, and zero is
on the axis. The cost is that the parameters' absolute values leave the figure -- the
per-condition means are printed to stdout for the caption.

The probit is fitted on log(risky/safe), which is why the x-axis of A is logarithmic
-- and why a constant absolute noise increase reads as magnitude-specific on that
scale.

    python -m tms_risk.behavior.scripts.plot_fig3_probit

Reads notes/data/localnoise_{group_posterior,delta_by_ratio}.tsv for the model and
ppc_fig3a.<label>.tsv for the observed proportions.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as ss

VERTEX, IPS = '#2ca02c', '#d62728'
# House rule: a *difference* gets its own ink, never one of the condition colours.
# Weight carries the inference -- near-black where the interval excludes zero, muted
# grey where it does not, so the two rows that matter are the two the eye lands on.
DIFF, DIFF_NS = '#2b2b2b', '#b6b6b6'
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    # Nothing here goes below 7.5 pt: at 7.25 in wide these are read at print size,
    # and 6-point annotations are unreadable on paper however clean they look on screen.
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
# NO `sns.set_context` here. It does not scale the block above, it REPLACES it:
# calling it re-set font.size 8.5 -> 9.6, axes.linewidth 0.8 -> 1.0 and tick
# size 3 -> 4.8, which is why this figure printed visibly heavier than the
# other four. Removed 2026-09-10.

# macOS ships Helvetica as a .ttc whose faces matplotlib cannot index, so every weight
# resolves to Regular and `fontweight='bold'` is silently a no-op. Arial Bold is a
# separate file and is metric-compatible, so route only the bold text through it.
BOLD = dict(fontname='Arial', fontweight='bold')

XT = [1.5, 2, 2.5, 3]
# The risk-neutral ratio: EV_risky = 0.55 * n_risky equals EV_safe = n_safe at
# n_risky/n_safe = 1/0.55. Left of it choosing risky is risk-seeking, right of it
# choosing safe is risk-averse, so it turns the axis from arbitrary into
# interpretable. The fitted indifference points (1/RNP) straddle it.
P_RISKY = .55
RISK_NEUTRAL = 1 / P_RISKY

# One column of block B per probit parameter. The anchors say which way the cTBS
# effect points, so the direction of the shift needs no caption.
SPECS = [
    dict(par='slope', xlabel='Δ Psychometric slope', ticks=[-1., -.5, 0., .5],
         abs_xlabel='Psychometric slope',
         col_title='Choice consistency',
         anchors=('Less consistent', 'More consistent')),
    # "More risk-averse"/"More risk-seeking" would be the exact reading of a Δ axis,
    # but the pair does not fit the column at a legible size; the published figure's
    # shorter wording is unambiguous next to a marked zero.
    dict(par='rnp', xlabel='Δ Risk-neutral probability',
         ticks=[-.05, 0., .05, .10, .15],
         abs_xlabel='Risk-neutral probability',
         col_title='Risk attitude',
         anchors=('Risk-averse', 'Risk-seeking')),
]


# NOTE on what the model line in A is.
# Evaluating Phi at the MEAN posterior parameters gives a curve that is systematically
# too steep: the observed statistic is a mean over subjects of sigmoids with different
# indifference points and slopes, and by Jensen that average is flatter than the
# sigmoid at the average parameters. Measured against these data, the group-parameter
# curve has RMSE 0.069 while the properly aggregated prediction has RMSE 0.019.
# So the model line here is `probit_<stim>` from analyze_localized_noise, which is the
# prediction of the binned statistic, with its own credible interval.


def paired_delta(g, parameter, order):
    """IPS - vertex, paired within (chain, draw) so the difference keeps its posterior."""
    s = g[(g.parameter == parameter) & (g.order == order)]
    w = s.pivot_table(index=['chain', 'draw'], columns='stimulation_condition',
                      values='value')
    return (w['ips'] - w['vertex']).values


def pfmt(d):
    """One-sided posterior tail probability, in the paper's pBayesian convention."""
    p = min(float((d > 0).mean()), float((d < 0).mean()))
    return 'p < 0.001' if p < .001 else f'p = {p:.3f}'


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    post = pd.read_csv(data / 'localnoise_group_posterior.tsv', sep='\t')
    rat = pd.read_csv(data / 'localnoise_delta_by_ratio.tsv', sep='\t')
    obs = pd.read_csv(data / f'ppc_fig3a.{label}.tsv', sep='\t')


    # Presentation order is the ROW variable in BOTH blocks: "Risky first" is always
    # the top row, "Risky second" always the bottom, labelled once at the far left.
    # So the cTBS effect reads as a displacement away from zero in B in the same rows
    # where the two psychometric functions separate in A.
    # narrower than the published version: block B no longer needs room for a
    # density, only for two intervals on a common axis
    fig = plt.figure(figsize=(7.25, 3.9))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.05, 1.35], wspace=.30,
                             left=.135, right=.985, top=.875, bottom=.155)
    gsA = outer[0].subgridspec(2, 1, hspace=.14)
    gsB = outer[1].subgridspec(2, 2, hspace=.20, wspace=.60)

    # --- A: psychometric functions, observed proportions over the probit fit
    lefts = []
    for row, order in enumerate(ORDERS):
        ax = fig.add_subplot(gsA[row]); lefts.append(ax)
        ax.axhline(.5, color='.75', lw=.7, ls='--', zorder=0)
        ax.axvline(RISK_NEUTRAL, color='.75', lw=.7, ls='--', zorder=0)
        o = obs[obs.order == order]
        r = rat[rat.order == order].set_index('bin')
        for stim, colr, mk in [('vertex', VERTEX, 'o'), ('ips', IPS, 'o')]:
            g = o[o.stim == stim].sort_values('frac')
            ax.fill_between(g.frac, r.loc[g.bin, f'probit_{stim}_lo'].values,
                            r.loc[g.bin, f'probit_{stim}_hi'].values,
                            color=colr, alpha=.22, lw=0, zorder=1)
            ax.plot(g.frac, r.loc[g.bin, f'probit_{stim}'].values, color=colr,
                    lw=2.0, zorder=2)
            ax.plot(g.frac, g.observed, mk, color=colr, ms=4.4, lw=0, zorder=4)
        ax.set_xscale('log')
        ax.set_xticks(XT)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        # Snug limits: the observed bins run 1.57-3.14 and 0.26-0.84, so anything
        # wider is dead space. No `trim` here -- it would clip the spines back to the
        # last round tick (3.0, 0.75) and leave real data sitting outside the axes.
        ax.set_xlim(1.5, 3.25)
        ax.set_ylim(.20, .90); ax.set_yticks([.25, .5, .75])
        if row == 0:
            ax.set_xticklabels([])
            ax.text(RISK_NEUTRAL * 1.012, .89, 'Risk-neutral', fontsize=7.5,
                    color='.45', ha='left', va='top', style='italic')
        else:
            ax.set_xlabel('Risky/safe payoff ratio')
    sns.despine(ax=lefts[0], offset=3, bottom=True)
    lefts[0].tick_params(axis='x', length=0)
    sns.despine(ax=lefts[1], offset=3)

    # The two curves run too close together to direct-label without collision, so
    # direct labels on the curves themselves: a legend in the corner makes the
    # reader carry a colour->condition mapping across the panel (skill rule 4).
    # Placed on the RISKY-SECOND row, where the two curves separate.
    _lab_ax = lefts[1] if len(lefts) > 1 else lefts[0]
    for cond, ccol, dy in [('ips', IPS, +.055), ('vertex', VERTEX, -.055)]:
        gg = obs[(obs.order == ORDERS[-1]) & (obs.stim == cond)]
        gg = gg.sort_values('frac')
        if not len(gg):
            continue
        # IPS above its own first point; vertex BELOW the curve further along,
        # where the empty wedge under the sigmoid is widest. Anchoring vertex
        # at the first point puts it on the bottom spine.
        # IPS above its own curve and LEFT of the risk-neutral rule; anchoring
        # it at the first point put the label on the P = 0.5 gridline. Vertex
        # below the curve, in the empty wedge under the sigmoid.
        # IPS just RIGHT of the risk-neutral rule and just above its own curve;
        # vertex below the green curve, in the empty wedge under the sigmoid
        k = 1 if cond == 'ips' else 2
        _lab_ax.annotate(cond.upper() if cond == 'ips' else 'Vertex',
                         (gg.frac.iloc[k], gg.observed.iloc[k]),
                         xytext=(4, 13) if cond == 'ips' else (-2, -17),
                         textcoords='offset points', color=ccol, fontsize=8,
                         ha='left' if cond == 'ips' else 'center',
                         va='center', fontweight='bold')

    # --- B: the parameters themselves, as a paired slope graph.
    # Condition on x, parameter value on y, one line per order. This is the
    # layout a within-subject two-level contrast wants: the effect is the
    # DIRECTION of the segment, read the same way as "goes down", and the
    # absolute values stay on the axis. The earlier version drew horizontal
    # intervals with an arrow between them, which put the contrast on the same
    # axis as the values and made neither easy to read.
    stats, rights = {}, {}
    for col, spec in enumerate(SPECS):
        par = spec['par']
        cell, deltas = {}, {}
        for order in ORDERS:
            deltas[order] = paired_delta(post, par, order)
            for cond in ('vertex', 'ips'):
                v = post[(post.parameter == par) & (post.order == order)
                         & (post.stimulation_condition == cond)]['value'].values
                cell[(order, cond)] = np.quantile(v, [.025, .5, .975])
        allv = np.concatenate([cell[k] for k in cell])
        pad = .16 * np.ptp(allv)
        ylim = (allv.min() - pad, allv.max() + pad)

        for row, order in enumerate(ORDERS):
            ax = fig.add_subplot(gsB[row, col]); rights[(par, order)] = ax
            d = deltas[order]
            lo, hi = np.quantile(d, [.025, .975])
            pv = pfmt(d)
            sig = pv.startswith('p <') or float(pv.split('=')[-1]) < .05
            stats[(par, order)] = (d.mean(), lo, hi, pv)

            xs = {'vertex': 0, 'ips': 1}
            ax.plot([0, 1], [cell[(order, 'vertex')][1], cell[(order, 'ips')][1]],
                    color='.2' if sig else '.65', lw=1.4, zorder=2)
            for cond, ccol, mk in (('vertex', VERTEX, 'o'), ('ips', IPS, 'o')):
                q = cell[(order, cond)]
                ax.plot([xs[cond]] * 2, q[[0, 2]], color=ccol, lw=1.5,
                        solid_capstyle='round', zorder=3)
                ax.plot(xs[cond], q[1], mk, ms=6, color=ccol, mec='white',
                        mew=.9, zorder=4)
            fmt = '{:+.2f}' if par == 'slope' else '{:+.3f}'
            ax.text(.5, .965, f'Δ {fmt.format(d.mean())}   {pv}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=7,
                    color='.15' if sig else '.5', **(BOLD if sig else {}))
            ax.set_xlim(-.55, 1.55)
            ax.set_ylim(*ylim)
            ax.set_xticks([0, 1])
            if row == 0:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', length=0)
            else:
                ax.set_xticklabels(['Vertex', 'IPS'])
            if spec['par'] == 'rnp':
                # RNP = 0.55 IS risk neutrality; without the line the reader
                # cannot tell risk-averse from risk-seeking on this axis
                ax.axhline(P_RISKY, color='.75', lw=.7, ls='--', zorder=0)
                # No in-panel label: the panel is ~1.5 in wide, both edges
                # carry a credible interval and the word does not fit between
                # them at any placement. Panel A names the same reference, and
                # what a reference line marks is caption bookkeeping anyway.
            ax.set_ylabel(spec['abs_xlabel'], fontsize=7)
            sns.despine(ax=ax, offset={'left': 3, 'bottom': 3})

    # --- row labels once at the far left, and one title per block
    for ax, order in zip(lefts, ORDERS):
        pos = ax.get_position()
        fig.text(.004, (pos.y0 + pos.y1) / 2, order.replace(' ', '\n'), fontsize=9,
                 ha='left', va='center', linespacing=1.25)

    y = lefts[0].get_position().y1 + .045
    # lowercase panel letters: Nature / Nature Comms house style, and what
    # every other figure in this paper uses
    blocks = [('a', lefts[0], lefts[0], 'Proportion of risky choices'),
              ('b', rights[(SPECS[0]['par'], ORDERS[0])],
               rights[(SPECS[0]['par'], ORDERS[0])], SPECS[0]['col_title']),
              ('c', rights[(SPECS[1]['par'], ORDERS[0])],
               rights[(SPECS[1]['par'], ORDERS[0])], SPECS[1]['col_title'])]
    for letter, first, last, title in blocks:
        x0, x1 = first.get_position().x0, last.get_position().x1
        fig.text(x0 - .040, y, letter, fontsize=9, va='baseline', ha='left',
                 **BOLD)
        fig.text((x0 + x1) / 2, y, title, fontsize=8.5, ha='center',
                 va='baseline')

    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf\n')
    print(f'{"parameter":<8}{"order":<15}{"Δ (IPS−vertex)":>16}{"95% CrI":>24}{"":>4}')
    for (par, order), (m, lo, hi, p) in stats.items():
        print(f'{par:<8}{order:<15}{m:>+16.4f}   [{lo:+.4f}, {hi:+.4f}]   {p}')
    # absolute levels leave the figure with the difference framing; they belong in
    # the caption, so print them here
    print('\nPer-condition posterior means (for the caption):')
    means = post.groupby(['parameter', 'order', 'stimulation_condition']).value.mean()
    for k, v in means.items():
        print(f'  {k[0]:<6}{k[1]:<15}{k[2]:<8}{v:.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/fig3_probit')
    a = parser.parse_args()
    main(a.data_dir, a.label, a.out)
