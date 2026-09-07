"""Companion figure for the Slack update: log vs natural space, and where cTBS lands.

Four panels, one claim each:

  a  On a log axis, taking the expected value of a gamble is a single rigid
     shift of log p -- the same at every payoff. On a linear axis it is a
     different move each time. If people are roughly consistent in the RATIO of
     their risk preferences, log space is where that consistency is a constant.
  b  What the two observers' fitted priors look like against the payoffs they
     actually saw. The linear-scale model needs a prior sitting below the range.
  c  The cTBS effect on noise, by payoff, with the within-draw credible band.
     Positive at small magnitudes, shrinking (and flipping) at large ones -- and
     not resolved at either end.
  d  Model comparison, where the same effect is unambiguous: dropping the cTBS
     regressor costs ~100 nats in every family.

Everything reads from TSVs already extracted from the traces.

    python -m tms_risk.behavior.scripts.plot_for_christian
"""
import argparse
import sys
from pathlib import Path

from matplotlib.gridspec import GridSpec

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
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.6,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

P_WIN = 0.55
LO, HI = 7.0, 112.0
C_LOG, C_LIN = '#3B5BA5', '#D8801F'
IPS, VERTEX = '#d62728', '#2ca02c'
C_DATA = '#1A1A1A'


def logx(ax, label='Payoff (CHF)', ticks=(7, 14, 28, 56, 112)):
    ax.set_xscale('log')
    ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel(label)


def letter(ax, s, dx=-0.20, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8.5, family='Arial',
            fontweight='bold', va='bottom', ha='left')


def gauss(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    ap.add_argument('--out', default='notes/figures/for_christian')
    args = ap.parse_args()
    sys.path.insert(0, 'libs/bauer')

    fig = plt.figure(figsize=(10.8, 15.6))
    gs = GridSpec(7, 10, figure=fig,
                  height_ratios=[1.00, 0.74, 0.80, 0.80, 0.72, 0.72, 0.72],
                  hspace=0.62, wspace=1.05, left=0.058, right=0.985,
                  top=0.965, bottom=0.042)


    # -- a: x p is a rigid shift in log space, a shear in linear space --------
    ax = ax_a = fig.add_subplot(gs[0, 0:5])
    anchors = [7.0, 28.0, 112.0]
    for i, a in enumerate(anchors):
        y = i * 1.25
        for val, col, alpha in [(a, C_DATA, .16), (P_WIN * a, C_LOG, .30)]:
            g = np.geomspace(1.5, 260, 500)
            d = gauss(np.log(g), np.log(val), 0.22)
            ax.fill_between(g, y, y + 0.95 * d, color=col, alpha=alpha, lw=0)
            ax.plot(g, y + 0.95 * d, color=col, lw=0.9)
        ax.annotate('', xy=(a * P_WIN, y + 1.03), xytext=(a, y + 1.03),
                    arrowprops=dict(arrowstyle='-|>', color='.25', lw=1.0,
                                    mutation_scale=7))
        ax.text(a * P_WIN, y + 1.07, f'−{-np.log(P_WIN):.2f}', fontsize=6.4,
                color='.25', ha='left', va='bottom')
        # what the same operation costs on a linear axis
        ax.text(1.7, y + 0.45, f'{a:.0f} to {P_WIN*a:.0f} CHF\n'
                               f'(a {a - P_WIN*a:.0f} CHF move)',
                fontsize=6.2, color='.45', ha='left', va='center',
                linespacing=1.25)
    logx(ax, 'Payoff (CHF), log axis', ticks=(2, 7, 28, 112))
    ax.set_xlim(1.5, 260)
    ax.set_ylim(-0.05, 2 * 1.25 + 1.85)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Belief')
    ax.set_title('Expected value = one rigid shift of log p', fontsize=7.8)
    ax.text(.02, .97, 'Same shift at every payoff. On a\n'
                      'CHF axis (grey) it is a different\nmove each time.',
            transform=ax.transAxes, fontsize=6.4, color='.3', ha='left',
            va='top', linespacing=1.3)

    # -- b: the prior each observer needs ------------------------------------
    ax = ax_b = fig.add_subplot(gs[0, 5:10])
    from scipy import stats
    from tms_risk.utils.data import get_all_behavior
    d = get_all_behavior(bids_folder=args.bids_folder)
    safe = d['n_safe'].dropna().values
    bins = np.geomspace(6, 32, 14)
    h, _ = np.histogram(safe, bins=bins, density=True)
    ax.bar(bins[:-1], h / h.max(), width=np.diff(bins), align='edge',
           color='.85', edgecolor='none', zorder=0)

    g = np.geomspace(1.2, 60, 3000)
    nat = gauss(g, 4.90, 1.06)
    lg = gauss(np.log(g), np.log(9.23), np.log(1.556))
    # the impossible region: below the smallest payoff ever shown
    ax.fill_between(g, 0, nat, where=g <= LO, color=C_LIN, alpha=.35, lw=0,
                    zorder=2)
    ax.plot(g, nat, color=C_LIN, lw=2.0, zorder=3)
    ax.plot(g, lg, color=C_LOG, lw=2.0, zorder=3)
    ax.axvline(LO, color='.25', lw=1.1, ls='--', zorder=4)

    frac = stats.norm.cdf(LO, 4.90, 1.06)
    ax.annotate(f'{100*frac:.0f}% of the NATURAL prior\nlies below the smallest\n'
                f'payoff anyone ever saw',
                xy=(4.6, 0.55), xytext=(1.45, 1.38), fontsize=6.8, color=C_LIN,
                ha='left', va='top', linespacing=1.3,
                arrowprops=dict(arrowstyle='-|>', color=C_LIN, lw=1.1,
                                mutation_scale=8, shrinkA=2, shrinkB=4))
    ax.text(30, 1.38, 'LOG prior\nsits in the range', color=C_LOG,
            fontsize=6.8, ha='right', va='top', linespacing=1.3)
    ax.text(LO * 1.06, 0.03, '7 CHF', color='.25', fontsize=6.4,
            ha='left', va='bottom')
    ax.text(16, 0.05, 'safe payoffs', color='.5', fontsize=6.4, ha='center')
    logx(ax, ticks=(2, 4, 7, 14, 28))
    ax.set_xlim(1.4, 40)
    ax.set_ylim(0, 1.55)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_ylabel('Density (peak-scaled)')
    ax.set_title('The natural-space prior is impossible', fontsize=7.8)

    # -- c: one panel per noise function, schematic + what it fitted ---------
    import bauer.models as bm

    def sigma(form, anc, sa, x):
        m = bm.LogAnchorNoiseRiskModel.__new__(bm.LogAnchorNoiseRiskModel)
        m.noise_form = form
        m.n_anchors, m.anchor_link = bm.NOISE_FORMS[form]
        m._anchors = np.asarray(anc, float)
        B = m.interp_matrix(x)
        th = np.log(np.asarray(sa, float))
        return np.exp(B @ th) if m.anchor_link == 'log' else B @ np.exp(th)

    XG = np.geomspace(LO, HI, 300)
    FORMS = [
        ('weber', 'Weber', '#8A8A8A', (LO,), [0.19], 1,
         'lfx2-bs3-w-dp-bm'),
        ('affine', 'Affine', '#D8801F', (LO, HI), [0.13, 0.29], 2, None),
        ('power', 'Power law', '#5D8C3F', (LO, HI), [0.13, 0.29], 2,
         'lfx2-pl-m2-dp-m-p2-i'),
        ('genweber', 'Gen. Weber', '#C44E52', (LO, HI), [0.13, 0.29], 2,
         'lfx2-gw-m2-dp-b-p2-fp'),
        ('spl5', 'Spline', '#3B5BA5', (7, 13, 20, 30, 112),
         [0.13, 0.16, 0.24, 0.22, 0.29], 5, 'lfx2-bs2-m2-dp-b'),
    ]
    form_axes = []
    for i, (form, nm, col, anc, sa, k, lab) in enumerate(FORMS):
        ax = fig.add_subplot(gs[1, 2 * i:2 * i + 2])
        ax.plot(XG, sigma(form, anc, sa, XG), color=col, lw=1.6)
        ax.plot(anc, sa, 'o', color=col, ms=4, zorder=5)
        if lab is not None:
            f = Path(f'notes/data/delta/curves.{lab}.tsv')
            if not f.exists():
                f = Path(f'notes/data/cards/curves.{lab}.tsv')
            cv = pd.read_csv(f, sep='\t')
            gg = cv[(cv.curve == 'n2 (second)') & (cv.stim.isin(['vertex', 'all']))]
            if len(gg):
                gg = gg.sort_values('payoff')
                ax.plot(gg.payoff, gg['median'], color='.25', lw=1.3, ls='--')
        logx(ax, '' if i else 'Payoff (CHF)', ticks=(7, 28, 112))
        ax.set_ylim(0.10, 0.33)
        ax.set_yticks([0.1, 0.2, 0.3])
        if i:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel('Noise SD (log units)')
        ax.set_title(f'{nm}  ({k} par.)', fontsize=7.4, color=col)
        form_axes.append(ax)
    form_axes[0].text(.05, .95, 'colour: the form\ndashed: what it fitted',
                      transform=form_axes[0].transAxes, fontsize=6.0,
                      color='.35', ha='left', va='top', linespacing=1.3)
    form_axes[0].axhline(0.19, color='.65', lw=0.8, ls=':', zorder=0)
    form_axes[1].text(.05, .95, 'not fitted yet\nunder the new\nparameterisation',
                      transform=form_axes[1].transAxes, fontsize=6.0,
                      color='.55', ha='left', va='top', linespacing=1.3)

    # -- d/e: one row per option, one panel per model, IPS vs vertex --------
    MODELS = [('lfx2-bs3-w-dp-bm', 'Weber'),
              ('lfx2-gw-m2-dp-b-p2-fp', 'Gen. Weber'),
              ('lfx2-pl-m2-dp-m-p2-i', 'Power law'),
              ('lfx2-bs2-m2-dp-b', 'Spline, log'),
              ('flexible1_noisefix.head', 'Spline, linear')]
    grid = {}
    for row, (which, tag) in enumerate([('n1 (first)', 'n1 — remembered'),
                                        ('n2 (second)', 'n2 — on screen')]):
        for j, (lab, nm) in enumerate(MODELS):
            ax = fig.add_subplot(gs[2 + row, 2 * j:2 * j + 2])
            f = Path(f'notes/data/delta/curves.{lab}.tsv')
            if not f.exists():
                f = Path(f'notes/data/cards/curves.{lab}.tsv')
            cv = pd.read_csv(f, sep='\t')
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                gg = cv[(cv.curve == which) & (cv.stim == stim)].sort_values('payoff')
                if not len(gg):
                    continue
                ax.fill_between(gg.payoff, gg.lo, gg.hi, color=col, alpha=.15,
                                lw=0)
                ax.plot(gg.payoff, gg['median'], color=col, lw=1.5)
            # Weber's law is a flat line on this axis; every panel gets the
            # same reference so the DIRECTION of departure is comparable
            ax.axhline(0.19, color='.65', lw=0.8, ls=':', zorder=0)
            logx(ax, 'Payoff (CHF)' if row else '', ticks=(7, 28, 112))
            if not row:
                ax.set_xticklabels([])
            ax.set_yscale('log')
            ax.set_ylim(0.035, 0.55)
            ax.set_yticks([0.05, 0.1, 0.2, 0.4])
            ax.yaxis.set_major_formatter(
                mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.yaxis.set_minor_locator(mticker.NullLocator())
            if j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f'{tag}\nnoise SD (log units)')
            if not row:
                # the linear-space model is flagged wherever it appears: its
                # noise slope has the wrong sign, which is not a matter of fit
                # quality but of what the observer would have to be like
                ax.set_title(nm, fontsize=7.6,
                             color=C_LIN if lab.startswith('flexible') else '.12')
            if lab.startswith('flexible'):
                ax.set_facecolor('#FDF1E3')
                ax.text(.5, .93, 'IMPLAUSIBLE', transform=ax.transAxes,
                        fontsize=6.6, family='Arial', color=C_LIN,
                        ha='center', va='top')
            grid[(row, j)] = ax
    ax_d, ax_e = grid[(0, 0)], grid[(1, 0)]

    grid[(0, 0)].text(.05, .95, 'IPS', color=IPS, fontsize=7.0,
                      transform=grid[(0, 0)].transAxes, ha='left', va='top')
    grid[(0, 0)].text(.05, .80, 'Vertex', color=VERTEX, fontsize=7.0,
                      transform=grid[(0, 0)].transAxes, ha='left', va='top')
    ROW_NOTES = [
        (0, 'n1 is nearly flat — WEBER HOLDS for the remembered option '
            '(x1.00 gen-Weber, x1.25 spline). Dotted line = Weber.', '.25'),
        (1, 'n2 rises steeply — WEBER FAILS for the option on screen '
            '(x1.37 to x2.12): less precise about larger amounts. '
            'The linear-space model goes the wrong way in BOTH (x0.22).',
         '.25'),
    ]

    # -- contrasts: the same cells as a within-draw difference ---------------
    ax_g = None
    for row, (which, tag) in enumerate([('n1 (first)', 'n1'),
                                        ('n2 (second)', 'n2')]):
        for j, (lab, nm) in enumerate(MODELS):
            ax = fig.add_subplot(gs[4 + row, 2 * j:2 * j + 2])
            f = Path(f'notes/data/delta/delta.{lab}.tsv')
            if not f.exists():
                ax.text(.5, .5, 'no contrast\nextracted', ha='center',
                        va='center', transform=ax.transAxes, fontsize=6.4,
                        color='.6', linespacing=1.3)
                ax.set_xticks([]); ax.set_yticks([])
                for sp_ in ax.spines.values():
                    sp_.set_visible(False)
                continue
            dl = pd.read_csv(f, sep='\t')
            g2 = dl[dl.curve == which].sort_values('payoff')
            ax.axhline(0, color='.7', lw=0.8, ls=':', zorder=0)
            ax.fill_between(g2.payoff, g2.lo, g2.hi, color=IPS, alpha=.16, lw=0)
            ax.plot(g2.payoff, g2['median'], color=IPS, lw=1.5)
            logx(ax, 'Payoff (CHF)' if row else '', ticks=(7, 28, 112))
            if not row:
                ax.set_xticklabels([])
            ax.set_ylim(-0.075, 0.075)
            ax.set_yticks([-0.05, 0, 0.05])
            if j:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f'cTBS on {tag}\n(IPS − vertex)')
            if row == 0 and j == 0:
                ax_g = ax
                ax.text(.05, .95, 'Within-draw difference,\n95% CrI',
                        transform=ax.transAxes, fontsize=6.2, color='.4',
                        ha='left', va='top', linespacing=1.3)
            rg = Path(f'notes/data/delta/regions.{lab}.tsv')
            if rg.exists():
                rr = pd.read_csv(rg, sep='\t')
                rr = rr[rr.curve == which]
                if len(rr):
                    ax.text(.95, .05, '\n'.join(
                        f"{r.region.split()[0]:<7s}P {r.p_gt0:.2f}"
                        for _, r in rr.iterrows()),
                        transform=ax.transAxes, fontsize=5.8, color='.35',
                        family='monospace', ha='right', va='bottom',
                        linespacing=1.3)

    # -- f: model comparison, where the effect is unambiguous ----------------
    ax = ax_f = fig.add_subplot(gs[6, 0:5])
    L = pd.read_csv('notes/data/ladder_v12.tsv', sep='\t').set_index('label')
    rows = [('Spline, log', 'lfx2-bs2-m2-dp-b', 'lfx2-bs3-m2-dp-null', C_LOG),
            ('Spline, linear', 'flexible1_noisefix.head',
             'flexible1_noisefix_null.head', C_LIN),
            ('Weber, log', 'lfx2-bs3-w-dp-bm', 'lfx2-bs3-w-dp-null', '#8A8A8A')]
    y = np.arange(len(rows))[::-1]
    for yi, (nm, a, b, col) in zip(y, rows):
        gain = L.loc[a, 'elpd_loo'] - L.loc[b, 'elpd_loo']
        ax.barh(yi, gain, height=.55, color=col, alpha=.85, lw=0)
        ax.text(gain + 2.5, yi, f'{gain:.0f}', fontsize=6.8, va='center',
                color='.2')
        ax.text(2.5, yi + 0.42, nm, fontsize=6.8, va='center', color='.1')
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlim(0, 128)
    ax.set_ylim(-0.7, len(rows) - 0.15)
    ax.set_xlabel('ELPD gained by adding the cTBS regressor (nats)')
    ax.set_title('Model comparison is unambiguous', fontsize=7.8)
    ax.text(.98, .06, 'Same effect the parameters\ncannot pin down',
            transform=ax.transAxes, fontsize=6.4, color='.35', ha='right',
            va='bottom', linespacing=1.25)

    # captions go in the GAP between rows, measured rather than guessed: an
    # offset that clears row d's (label-free) axis lands on row e's tick labels
    fig.canvas.draw()
    for row, note, col in ROW_NOTES:
        boxes = [grid[(row, j)].get_position() for j in range(len(MODELS))]
        top = min(b.y0 for b in boxes)
        below = [a.get_position().y1 for a in fig.axes
                 if a.get_position().y1 < top - 0.005]
        bottom = max(below) if below else top - 0.05
        fig.text(0.5 * (min(b.x0 for b in boxes) + max(b.x1 for b in boxes)),
                 bottom + 0.35 * (top - bottom), note, ha='center',
                 va='center', fontsize=6.8, color=col)

    sns.despine(fig=fig, offset=3)
    for ax in (ax_a, ax_b, ax_f):
        ax.spines['left'].set_visible(False)
    for ax, lab, dx in [(ax_a, 'a', -0.08), (ax_b, 'b', -0.08),
                        (form_axes[0], 'c', -0.44), (ax_d, 'd', -0.44),
                        (ax_e, 'e', -0.44)] + \
                       ([(ax_g, 'f', -0.44)] if ax_g is not None else []) + \
                       [(ax_f, 'g' if ax_g is not None else 'f', -0.09)]:
        letter(ax, lab, dx=dx)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.03)
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
