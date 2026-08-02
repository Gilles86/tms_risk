"""PROTOTYPE. The ratio shift as two opposing contributions that mostly cancel.

cTBS devalues BOTH options, and the two devaluations push the perceived
risky/safe ratio in OPPOSITE directions:

    the safe option losing value  raises  the ratio
    the risky option losing value lowers  the ratio

so the effect on choice is a residual, not a main effect. On the log scale that
statement is exact and additive:

    dlog(ratio) = -dlog(safe)  +  dlog(risky)
                = (bar up)     +  (bar down)   = (dot)

Each stake gets one pair of bars and one net dot. The order effect then has a
one-line reading: when the risky option comes first the two bars are the same
height and cancel to nothing; when it comes second the safe bar is roughly twice
the risky one and a residual survives.

The right-hand panel converts the residual into the quantity the reader cares
about, using the model's own local slope of P(chose risky) against the perceived
ratio, taken from decision_space.<label>.tsv.

    python -m tms_risk.behavior.scripts.proto_ratio_waterfall

Reads notes/data/pmc_percepts_by_order.<label>.tsv and decision_space.<label>.tsv.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
SAFE, RISKY = '#4d4d4d', '#b2182b'
DIFF = '#1a1a1a'
ORDERS = ['Risky first', 'Risky second']
LEVELS = [7.0, 10.0, 14.0, 20.0, 28.0]

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
    'lines.linewidth': 1.3, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')
PANEL = dict(fontsize=10, fontweight='bold', va='bottom', ha='right')


def contributions(d):
    """Per (order, stake): the two signed contributions to dlog(ratio), in %."""
    out = []
    for (order, ns), sub in d.groupby(['order', 'n_safe']):
        r = {'order': order, 'n_safe': ns}
        for opt in ['safe', 'risky']:
            row = sub[sub.option == opt].iloc[0]
            r[f'{opt}_dlog'] = 100 * np.log(row.ips / row.vertex)
        # a loss on the safe option RAISES the ratio, hence the sign flip
        r['from_safe'] = -r['safe_dlog']
        r['from_risky'] = r['risky_dlog']
        r['net'] = r['from_safe'] + r['from_risky']
        out.append(r)
    return pd.DataFrame(out)


def main(data_dir, label, out_stem):
    data = Path(data_dir)
    d = pd.read_csv(data / f'pmc_percepts_by_order.{label}.tsv', sep='\t')
    w = contributions(d)

    # local slope of P(chose risky) on the perceived ratio, from the model's own
    # decision-space grid: dP / dlog(ratio), evaluated per (order, stake) by
    # regressing p_vertex on log(ratio_vertex) across the ratio grid.
    ds = pd.read_csv(data / f'decision_space.{label}.tsv', sep='\t')
    slopes = {}
    for (order, ns), g in ds.groupby(['order', 'n_safe']):
        g = g.sort_values('ratio')
        lr = np.log(g.ratio_vertex.values)
        p = g.p_vertex.values
        slopes[(order, ns)] = np.polyfit(lr, p, 1)[0]
    grid_ns = np.array(sorted(ds.n_safe.unique()))

    fig = plt.figure(figsize=(7.25, 3.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.02], wspace=.36,
                          left=.085, right=.985, top=.79, bottom=.145)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bw = .34

    for k, order in enumerate(ORDERS):
        ax = axes[k]
        s = w[w.order == order].set_index('n_safe').loc[LEVELS]
        xi = np.arange(len(LEVELS))
        ax.bar(xi - bw / 2, s.from_safe, width=bw, color=SAFE, lw=0, zorder=2)
        ax.bar(xi + bw / 2, s.from_risky, width=bw, color=RISKY, lw=0, zorder=2)
        ax.plot(xi, s.net, 'D', color=DIFF, mfc=DIFF, ms=5, zorder=4)
        ax.hlines(s.net, xi - .46, xi + .46, color=DIFF, lw=.7, ls=':', zorder=3)
        ax.axhline(0, color='0.4', lw=.8, zorder=1)
        ax.set_xticks(xi)
        ax.set_xticklabels([f'{v:.0f}' for v in LEVELS])
        ax.set_xlim(-.62, len(LEVELS) - .38)
        ax.set_ylim(-7.3, 8.6)
        ax.set_yticks([-6, -4, -2, 0, 2, 4, 6, 8])
        ax.set_xlabel('Safe payoff (CHF)')
        ax.set_title(order, fontsize=8.5, pad=4)
        if k == 0:
            ax.set_ylabel('Contribution to Δ perceived ratio (%)')
            ax.text(xi[3] - bw / 2, s.from_safe.iloc[3] + .5, 'Safe option\nloses value',
                    color=SAFE, fontsize=7.5, ha='center', va='bottom')
            ax.text(xi[3] + bw / 2, s.from_risky.iloc[3] - .5, 'Risky option\nloses value',
                    color=RISKY, fontsize=7.5, ha='center', va='top')
        else:
            ax.set_yticklabels([])
            ax.annotate('Net', xy=(xi[1], s.net.iloc[1]), xytext=(xi[0] + .1, 6.4),
                        fontsize=7.5, color=DIFF, ha='left', va='bottom',
                        arrowprops=dict(arrowstyle='-', color='0.45', lw=.6,
                                        connectionstyle='arc3,rad=-.25'))

    # ---- what the residual buys, in choice probability -------------------------
    ax = axes[2]
    for order, col in [('Risky first', '0.62'), ('Risky second', DIFF)]:
        s = w[w.order == order].set_index('n_safe').loc[LEVELS]
        sl = np.array([slopes[(order, grid_ns[np.argmin(abs(grid_ns - v))])]
                       for v in LEVELS])
        ax.plot(np.arange(len(LEVELS)), sl * s.net.values / 100, '-D',
                color=col, mfc=col, ms=5,
                lw=1.6 if order == 'Risky second' else 1.1, zorder=3)
    ax.axhline(0, color='0.4', lw=.8, zorder=1)
    ax.set_xticks(np.arange(len(LEVELS)))
    ax.set_xticklabels([f'{v:.0f}' for v in LEVELS])
    ax.set_xlim(-.62, len(LEVELS) - .38)
    ax.set_ylim(-.012, .085)
    ax.set_yticks([0, .02, .04, .06, .08])
    ax.set_xlabel('Safe payoff (CHF)')
    ax.set_ylabel('Δ P(chose risky)')
    ax.text(len(LEVELS) - .55, .062, 'Risky second', color=DIFF, fontsize=7.5,
            ha='right', va='bottom')
    ax.text(len(LEVELS) - .55, .004, 'Risky first', color='0.55', fontsize=7.5,
            ha='right', va='bottom')

    for ax, letter in zip(axes, 'abc'):
        ax.text(-.13 if ax is not axes[0] else -.28, 1.055, letter,
                transform=ax.transAxes, **PANEL)

    fig.suptitle('Both options lose value, and the two losses fight each other; '
                 'choice sees only what is left', fontsize=9, y=.965)
    sns.despine(fig=fig, offset=4, trim=False)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    print(f'wrote {out.with_suffix(".pdf")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--label', default='flexible2nf')
    p.add_argument('--out', default='notes/figures/prototypes/ratio_waterfall')
    a = p.parse_args()
    main(a.data_dir, a.label, a.out)
