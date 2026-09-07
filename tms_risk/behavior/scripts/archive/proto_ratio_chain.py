"""PROTOTYPE. Two compressed payoff curves -> a shifted ratio, as an additive chain.

The decision variable is a RATIO, and a ratio is a DIFFERENCE on a log axis. That
single change of scale makes the whole mechanism additive and lets three panels be
read left to right as one arithmetic statement:

    col a   perceived expected value of each option, log axis.
            The vertical gap between the two curves IS log(risky/safe).
    col b   what cTBS does to each option, in % of its own perceived value.
            The near-black stick between the two curves is how much the gap changed.
    col c   the same sticks, replanted on a zero baseline: the change in the
            perceived risky/safe ratio.

Exactly, with no approximation:  dlog(ratio) = dlog(risky) - dlog(safe).

The order effect is then a purely visual fact: in the top row the two curves in
col b lie on top of each other (no stick, no ratio shift), in the bottom row they
separate.

    python -m tms_risk.behavior.scripts.proto_ratio_chain

Reads notes/data/pmc_percepts_by_order.<label>.tsv only. No trace needed.
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
LEVELS = [7, 10, 14, 20, 28]

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


def prepare(d):
    """Wide table per (order, safe payoff): percepts, % shifts, ratio shift.

    The percent shifts are 100 * log(ips / vertex); the credible band on them is
    the stored credible interval on the CHF difference rescaled by the vertex
    percept, which is a deterministic monotone rescaling, not a new inference.
    """
    out = []
    for (order, ns), sub in d.groupby(['order', 'n_safe']):
        row = {'order': order, 'n_safe': ns}
        for opt in ['safe', 'risky']:
            r = sub[sub.option == opt].iloc[0]
            row[f'{opt}_v'] = r.vertex
            row[f'{opt}_i'] = r.ips
            row[f'{opt}_pct'] = 100 * np.log(r.ips / r.vertex)
            row[f'{opt}_pct_lo'] = 100 * r.lo / r.vertex
            row[f'{opt}_pct_hi'] = 100 * r.hi / r.vertex
        row['ratio_v'] = row['risky_v'] / row['safe_v']
        row['ratio_i'] = row['risky_i'] / row['safe_i']
        row['ratio_pct'] = row['risky_pct'] - row['safe_pct']
        out.append(row)
    return pd.DataFrame(out).sort_values(['order', 'n_safe'])


def main(data_dir, label, out_stem):
    d = pd.read_csv(Path(data_dir) / f'pmc_percepts_by_order.{label}.tsv', sep='\t')
    w = prepare(d)

    fig = plt.figure(figsize=(7.25, 4.5))
    gs = fig.add_gridspec(2, 3, hspace=.28, wspace=.44,
                          left=.085, right=.965, top=.875, bottom=.115)
    axes = np.array([[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)])

    pct_lo = min(w.safe_pct.min(), w.risky_pct.min()) - 1.6
    obj_ratio = (d[d.option == 'risky'].set_index(['order', 'n_safe']).objective_ev
                 / d[d.option == 'safe'].set_index(['order', 'n_safe']).objective_ev)

    for r, order in enumerate(ORDERS):
        s = w[w.order == order]
        x = s.n_safe.values
        obj_s = x.astype(float)
        obj_r = d[(d.order == order) & (d.option == 'risky')].objective_ev.values
        a, b, c = axes[r]

        # ---- col a: objective values fan apart, percepts collapse together -------
        a.set_yscale('log')
        a.set_xscale('log')
        a.plot(x, obj_s, ':', color=SAFE, lw=1.0, zorder=1)
        a.plot(x, obj_r, ':', color=RISKY, lw=1.0, zorder=1)
        for opt, col in [('safe', SAFE), ('risky', RISKY)]:
            a.plot(x, s[f'{opt}_v'], '-o', color=col, mfc=col, mec=col, zorder=3)
            a.plot(x, s[f'{opt}_i'], '-o', color=col, mfc='white', mec=col,
                   mew=1.0, lw=1.0, zorder=2)
        a.set_ylim(5.2, 45)
        a.set_yticks([6, 10, 20, 40])
        a.set_yticklabels(['6', '10', '20', '40'])
        a.set_xticks(LEVELS)
        a.set_xticklabels([str(v) for v in LEVELS])
        a.minorticks_off()
        a.text(.02, .97, order, transform=a.transAxes, fontsize=8, va='top')
        if r == 0:
            a.annotate('Objective', xy=(x[2], obj_r[2] * 1.06),
                       xytext=(x[1] * .98, 30), fontsize=7.5, color='0.3',
                       ha='center', va='bottom',
                       arrowprops=dict(arrowstyle='-', color='0.45', lw=.6,
                                       connectionstyle='arc3,rad=-.25'))
            a.annotate('Perceived', xy=(x[3], s.risky_v.iloc[3] * 1.04),
                       xytext=(x[4] * 1.02, 16), fontsize=7.5, color='0.3',
                       ha='right', va='bottom',
                       arrowprops=dict(arrowstyle='-', color='0.45', lw=.6,
                                       connectionstyle='arc3,rad=.25'))
            a.text(x[-1] * 1.03, obj_r[-1] * 1.06, 'Risky', color=RISKY,
                   fontsize=7.5, ha='right', va='bottom')
            a.text(x[-1] * 1.03, obj_s[-1] * .80, 'Safe', color=SAFE,
                   fontsize=7.5, ha='right', va='top')

        # ---- col b: what cTBS costs each option, proportionally -----------------
        # No marginal credible bands here: the stored intervals are on each
        # option's own CHF shift, and drawing them swamps the only thing this
        # panel is for, which is the vertical distance between the two curves.
        for opt, col in [('safe', SAFE), ('risky', RISKY)]:
            b.plot(x, s[f'{opt}_pct'], '-o', color=col, zorder=3)
        # the stick between the curves is exactly the ratio shift plotted in col c
        b.vlines(x, s.safe_pct, s.risky_pct, color=DIFF, lw=2.0, zorder=4)
        b.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
        b.set_xscale('log')
        b.set_xticks(LEVELS)
        b.set_xticklabels([str(v) for v in LEVELS])
        b.minorticks_off()
        b.set_ylim(-8.6, 1.4)
        b.set_yticks([-8, -6, -4, -2, 0])
        if r == 1:
            b.annotate('Ratio shift', xy=(x[1], (s.safe_pct.iloc[1]
                                                 + s.risky_pct.iloc[1]) / 2),
                       xytext=(x[2] * 1.05, -1.4), fontsize=7.5, color=DIFF,
                       ha='left', va='center',
                       arrowprops=dict(arrowstyle='-', color='0.45', lw=.6,
                                       connectionstyle='arc3,rad=.25'))
        if r == 0:
            b.text(x[-1] * 1.03, s.safe_pct.iloc[-1] - .3, 'Safe', color=SAFE,
                   fontsize=7.5, ha='right', va='top')
            b.text(x[-1] * 1.03, s.risky_pct.iloc[-1] + .4, 'Risky', color=RISKY,
                   fontsize=7.5, ha='right', va='bottom')

        # ---- col c: the perceived ratio, and the same stick moving it ------------
        c.plot(x, obj_ratio.loc[order].values, ':', color='0.45', lw=1.0, zorder=1)
        c.plot(x, s.ratio_v, '-o', color=VERTEX, mfc=VERTEX, zorder=3)
        c.plot(x, s.ratio_i, '-o', color=IPS, mfc=IPS, zorder=3)
        c.vlines(x, s.ratio_v, s.ratio_i, color=DIFF, lw=2.0, zorder=4)
        c.set_xscale('log')
        c.set_xticks(LEVELS)
        c.set_xticklabels([str(v) for v in LEVELS])
        c.minorticks_off()
        c.set_ylim(.945, 1.30)
        c.set_yticks([0.95, 1.0, 1.05, 1.25])
        c.set_yticklabels(['0.95', '1.00', '1.05', '1.25'])
        if r == 0:
            c.text(x[0] * .97, 1.265, 'Objective', color='0.35', fontsize=7.5,
                   ha='left', va='bottom')
            c.text(x[-1] * 1.03, s.ratio_i.iloc[-1] + .012, 'cTBS', color=IPS,
                   fontsize=7.5, ha='right', va='bottom')
            c.text(x[-1] * 1.03, s.ratio_v.iloc[-1] - .012, 'Vertex', color=VERTEX,
                   fontsize=7.5, ha='right', va='top')

        axes[r, 0].set_ylabel('Expected value (CHF)')
        axes[r, 1].set_ylabel('Δ Perceived EV (%)')
        axes[r, 2].set_ylabel('Perceived risky/safe')

    for c in range(3):
        axes[1, c].set_xlabel('Safe payoff (CHF)')
    for c, letter in zip(range(3), 'abc'):
        axes[0, c].text(-.22, 1.05, letter, transform=axes[0, c].transAxes, **PANEL)

    fig.suptitle('cTBS costs the safe option more, and the ratio is what is left over',
                 fontsize=9, y=.972)
    sns.despine(fig=fig, offset=4, trim=False)
    for ax in axes.ravel():
        ax.tick_params(which='minor', bottom=False, left=False)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    print(f'wrote {out.with_suffix(".pdf")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--label', default='flexible2nf')
    p.add_argument('--out', default='notes/figures/prototypes/ratio_chain')
    a = p.parse_args()
    main(a.data_dir, a.label, a.out)
