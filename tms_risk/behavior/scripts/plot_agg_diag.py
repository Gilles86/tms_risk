"""Diagnostic: how much does the AGGREGATION choice move Figure 5's panels?

Not a paper figure -- a provenance check.  Panels f/g of `plot_fig4_big.py`
rebuild the mechanism from the per-subject MEDIAN parameter tables (a plug-in
estimate), while panels h/i show a posterior predictive that INTEGRATES over
draws; and the earlier ratio-shift prototype evaluated the same algebra on a
uniform GRID of ratios rather than on the trials people actually saw.  This
plots all of those side by side against the model's own PPC, so the cost of
each choice is a number rather than an argument.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd

mpl.use('Agg')
import matplotlib.pyplot as plt

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 6.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 300, 'savefig.bbox': 'tight', 'savefig.pad_inches': .02,
})

# black = the estimator that reproduces the model; colours = the shortcuts
V = {'integrate/mean':   dict(c='k',       ls='-',  lw=1.6, label='Integrate over draws (mean)'),
     'integrate/median': dict(c='0.45',    ls='--', lw=1.1, label='Integrate over draws (median)'),
     'plugin_mean':      dict(c='#3B5BA5', ls='-',  lw=1.1, label='Plug in posterior mean'),
     'plugin_median':    dict(c='#C44E52', ls='-',  lw=1.1, label='Plug in posterior median')}
ORDERS = ('Risky first', 'Risky second')


def main(label, data_dir, ppc_tsv, out_pdf):
    d = pd.read_csv(Path(data_dir) / f'agg_variants.{label}.tsv', sep='\t')
    d = d[d.over_subjects == 'mean']
    p = pd.read_csv(ppc_tsv, sep='\t')
    w = p.pivot_table(index=['order', 'stake_grp', 'rung'], columns='stim',
                      values=['model', 'observed'])
    ppc = pd.DataFrame({'ppc': w[('model', 'ips')] - w[('model', 'vertex')],
                        'obs': w[('observed', 'ips')] - w[('observed', 'vertex')]})

    fig, AX = plt.subplots(2, 3, figsize=(7.25, 4.6), constrained_layout=True)

    # -- row 1: the mechanism panels' own x-axis (safe payoff) --------------
    for j, order in enumerate(ORDERS):
        ax = AX[0, j]
        for wh, alpha, ls_over in (('trials_safe', 1.0, None), ('grid', .35, ':')):
            for v, st in V.items():
                s = d[(d['where'] == wh) & (d.variant == v)
                      & (d.quantity == 'dp') & (d.order == order)]
                if not len(s):
                    continue
                s = s.sort_values('n_safe')
                ax.plot(s.n_safe, s.value, color=st['c'],
                        ls=ls_over or st['ls'], lw=st['lw'], alpha=alpha)
        ax.axhline(0, color='0.8', lw=.6, zorder=0)
        ax.set_xscale('log')
        ax.set_xticks([7, 10, 14, 20, 28])
        ax.set_xticklabels(['7', '10', '14', '20', '28'])
        ax.minorticks_off()
        ax.set_title(order, fontsize=7.5)
        ax.set_xlabel('Safe payoff (CHF)')
        if j == 0:
            ax.set_ylabel('ΔP(risky), IPS − vertex')
    AX[0, 0].text(.03, .95, 'Solid: real trials\nDotted: uniform ratio grid',
                  transform=AX[0, 0].transAxes, va='top', fontsize=6, color='0.35')

    # -- a: legend panel doubles as the estimator key -----------------------
    ax = AX[0, 2]
    for i, (v, st) in enumerate(V.items()):
        y = 1 - i * .13
        ax.plot([.04, .18], [y, y], color=st['c'], ls=st['ls'], lw=st['lw'],
                transform=ax.transAxes, clip_on=False)
        ax.text(.22, y, st['label'], transform=ax.transAxes, va='center',
                fontsize=6.5, color=st['c'])
    # the number that matters: agreement with the model's own PPC
    rows = []
    s = d[(d['where'] == 'trials') & (d.quantity == 'dp')]
    piv = s.pivot_table(index=['order', 'stake_grp', 'rung'], columns='variant',
                        values='value')
    m = ppc.join(piv)
    for v in V:
        e = m.ppc - m[v]
        rows.append((v, np.corrcoef(m.ppc, m[v])[0, 1], e.abs().max(),
                     np.sqrt((e ** 2).mean())))
    ax.text(.04, .48, 'Agreement with the model\'s own PPC\n(24 order × stake × rung cells)',
            transform=ax.transAxes, va='top', fontsize=6.5, color='0.25')
    for i, (v, r, mx, rms) in enumerate(rows):
        ax.text(.04, .33 - i * .09, f'r = {r:.3f}   max |Δ| = {mx:.4f}',
                transform=ax.transAxes, va='top', fontsize=6.5, color=V[v]['c'])
    ax.axis('off')

    # -- row 2: the PPC panels' own x-axis (ladder rung), per stake half ----
    for j, sg in enumerate([0, 1]):
        ax = AX[1, j]
        for order, mk, lsty in (('Risky first', 'o', '--'),
                                ('Risky second', 's', '-')):
            sel = (slice(None), sg, slice(None))
            sub = m.xs(order, level='order').xs(sg, level='stake_grp')
            ax.plot(sub.index, sub.ppc, color='#2E7D32', ls=lsty, lw=1.8,
                    alpha=.85, zorder=1)
            for v, st in V.items():
                ax.plot(sub.index, sub[v], color=st['c'], ls=st['ls'],
                        lw=st['lw'], alpha=.9, zorder=2)
        ax.axhline(0, color='0.8', lw=.6, zorder=0)
        ax.set_title(f'{"Low" if sg == 0 else "High"} stakes', fontsize=7.5)
        ax.set_xlabel('Ladder rung (risky/safe ratio)')
        if j == 0:
            ax.set_ylabel('ΔP(risky), IPS − vertex')
    AX[1, 0].text(.97, .97, 'Green: the model\'s own simulated PPC\n'
                  'Dashed / solid: risky first / second',
                  transform=AX[1, 0].transAxes, ha='right', va='top',
                  fontsize=6, color='0.35', linespacing=1.4)

    # -- f: the mechanism quantities themselves ----------------------------
    ax = AX[1, 2]
    for q, mkr, lab in (('num_shift', 'o', 'Perceived log-ratio shift'),
                        ('den_ratio', 's', 'Decision SD ratio − 1')):
        for v, st in V.items():
            s = d[(d['where'] == 'trials_safe') & (d.variant == v)
                  & (d.quantity == q) & (d.order == 'Risky second')]
            if not len(s):
                continue
            s = s.sort_values('n_safe')
            y = s.value - (1 if q == 'den_ratio' else 0)
            ax.plot(s.n_safe, y, color=st['c'], ls=st['ls'], lw=st['lw'],
                    marker=mkr, ms=2.5)
    ax.axhline(0, color='0.8', lw=.6, zorder=0)
    ax.set_xscale('log')
    ax.set_xticks([7, 10, 14, 20, 28])
    ax.set_xticklabels(['7', '10', '14', '20', '28'])
    ax.minorticks_off()
    ax.set_title('Mechanism, risky second', fontsize=7.5)
    ax.set_xlabel('Safe payoff (CHF)')
    ax.set_ylabel('IPS − vertex')
    ax.text(.97, .97, 'Circles: perceived log-ratio shift\n'
            'Squares: decision SD ratio − 1',
            transform=ax.transAxes, ha='right', va='top', fontsize=6,
            color='0.35', linespacing=1.4)

    # headroom first, so the in-panel notes below never sit on top of data
    for ax in (AX[0, 0], AX[0, 1], AX[1, 0], AX[1, 1], AX[1, 2]):
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + .30 * (hi - lo))
    for k, ax in zip('abcdef', AX.ravel()):
        ax.text(-.16, 1.12, k, transform=ax.transAxes, fontsize=9,
                fontname='Arial', fontweight='bold', va='top')
    fig.suptitle(f'Aggregation diagnostic · {label}', fontsize=8.5)
    import seaborn as sns
    sns.despine(fig=fig, offset=4, trim=False)
    AX[0, 2].set_axis_off()
    Path(out_pdf).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf)
    fig.savefig(str(out_pdf).replace('.pdf', '.png'), dpi=200)
    print('wrote', out_pdf)
    print(m.round(4).to_string())


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('label')
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--ppc_tsv', required=True)
    ap.add_argument('--out_pdf', required=True)
    a = ap.parse_args()
    main(a.label, a.data_dir, a.ppc_tsv, a.out_pdf)
