"""The cTBS effect itself, model by model: does the model PRODUCE the thing the
paper is about?

Plotting P(risky) for each stimulation arm and hoping the reader spots a gap
wastes most of the panel on the psychometric function, which every model gets
right. The quantity under test is the DIFFERENCE, IPS minus vertex, and at this
size it is a few percentage points sitting on a curve that spans 50. So plot the
difference directly: observed markers against the model's own 95% posterior
predictive band, one row per presentation order, one column per model.

Read it as: does the band contain the markers, and does the band separate from
zero in the row where the data do (risky second) and not in the row where they
do not (risky first)?

No error bars on the observed points. The band already carries the uncertainty
that matters -- what the fitted model predicts for a dataset this size -- and
putting an s.e.m. beside it invites reading two incompatible intervals as one
comparison.

    python -m tms_risk.behavior.scripts.plot_ppc_delta_compare \\
        --labels log-power-n1n2.mapjitter.klw log-power-n1n2pmu.klw \\
        --names 'Noise only (reported)' 'Noise + prior shift'
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
ORDERS = ['Risky first', 'Risky second']
#: the effect lives on risky-second trials, so that row is the black one and
#: risky-first is the muted control row -- position and weight encode order,
#: never hue (hue is reserved for stimulation across every figure in the paper)
ROWC = {'Risky first': '0.62', 'Risky second': '0.15'}
P_RISKY = .55

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


def key(ax, entries, x=.04, y=.06, dy=.10, seg=.10, fs=6.5):
    """Inline legend drawn as the real marks, never as prose."""
    for i, (lab, col, kind, o) in enumerate(entries):
        yy = y + i * dy
        tf = ax.transAxes
        if kind == 'band':
            ax.add_patch(plt.Rectangle((x, yy - .022), seg, .044, transform=tf,
                                       facecolor=col, alpha=o.get('alpha', .22),
                                       lw=0, clip_on=False))
        elif kind == 'marker':
            ax.plot(x + seg / 2, yy, o.get('marker', 'o'), transform=tf,
                    ms=o.get('ms', 3.8), color=col, clip_on=False, lw=0)
        else:
            ax.plot([x, x + seg], [yy, yy], transform=tf, color=col,
                    lw=o.get('lw', 1.4), ls=o.get('ls', '-'), clip_on=False)
        ax.text(x + seg + .03, yy, lab, transform=tf, color=o.get('tc', col),
                fontsize=fs, va='center')


def main(labels, names, data_dir, out_stem, stat):
    dd = Path(data_dir) / 'ppc_anchor'
    d = {}
    for lab in labels:
        f = dd / f'ppc_anchor.delta_rung.{lab}.tsv'
        if not f.exists():
            raise SystemExit(f'no PPC extraction for {lab}\n  expected {f}')
        d[lab] = pd.read_csv(f, **READ)
    stats = {}
    for lab in labels:
        f = dd / f'ppc_stats.{lab}.tsv'
        if f.exists():
            stats[lab] = pd.read_csv(f, **READ).set_index('statistic')

    n = len(labels)
    fig, AX = plt.subplots(2, n, figsize=(1.95 * n + .8, 3.9), sharex=True,
                           sharey=True, constrained_layout=True, squeeze=False)
    for c, lab in enumerate(labels):
        for r, order in enumerate(ORDERS):
            ax, col = AX[r, c], ROWC[order]
            o = d[lab][d[lab].order == order].sort_values('frac')
            ax.axhline(0, color='.8', lw=.7, ls='--', zorder=0)
            ax.axvline(1 / P_RISKY, color='.9', lw=.7, zorder=0)
            ax.fill_between(o.frac, o.lo, o.hi, color=col, alpha=.20, lw=0,
                            zorder=1)
            ax.plot(o.frac, o.model, color=col, lw=1.5, zorder=2)
            ax.plot(o.frac, o.observed, 'o', ms=4.2, color=col, lw=0, zorder=4)
            ax.set_xscale('log')
            ax.set_xticks([1.6, 2, 2.5, 3])
            ax.set_xticklabels(['1.6', '2', '2.5', '3'])
            ax.minorticks_off()
            # headroom: the risky-second rung at ratio 1.78 reaches +0.13,
            # and the claim line sits above everything
            ax.set_ylim(-0.10, 0.21)
            if r == 0:
                ax.set_title(names[c], fontsize=8)
            if r == 1:
                ax.set_xlabel('Risky / safe payoff ratio')
            if c == 0:
                ax.set_ylabel(f'{order}\nΔ P(chose risky), IPS − vertex')
            if lab in stats and stat in stats[lab].index:
                s = stats[lab].loc[stat]
                if order == 'Risky second':
                    inside = 'inside' if s.covered else 'OUTSIDE'
                    ax.text(.04, .97,
                            f'Mean Δ {s.observed:+.3f} observed\n'
                            f'{s.model_median:+.3f} predicted, {inside} the band',
                            transform=ax.transAxes, fontsize=6.5,
                            color='0.15' if s.covered else '#b2182b',
                            va='top', ha='left')
    key(AX[0, 0], [('Observed', ROWC['Risky first'], 'marker', {}),
                   ('Model median', ROWC['Risky first'], 'line', {}),
                   ('95% predictive', ROWC['Risky first'], 'band', {})])
    sns.despine(fig=fig, offset=4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}')
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', default=None)
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--stat', default='dp_second_mean')
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/ppc_delta_compare'))
    a = ap.parse_args()
    main(a.labels, a.names or a.labels, a.data_dir, a.out_stem, a.stat)
