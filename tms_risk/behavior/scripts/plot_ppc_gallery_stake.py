"""Supplementary: the psychometric posterior predictive for every key model.

One row per model, four columns: presentation order crossed with a
within-participant median split on stake. The median split rather than terciles
so each point still rests on about three ladder rungs per participant, which
keeps the columns legible down a long page.

Read it column by column. Every model tracks the overall rise of P(risky) with
the payoff ratio, so the rows look alike at a glance; what separates them is
whether red sits above green in the RISKY-SECOND columns, and whether that gap
is larger at low than at high stakes. Models that drop the cTBS effect, or put
it on the wrong option, lose the separation while still tracking the curve.
This is why choice proportions rank the models so weakly and why the paper
leans on the targeted slope check instead.

    python -m tms_risk.behavior.scripts.plot_ppc_gallery_stake
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
IPS, VERTEX = '#d62728', '#2ca02c'
ORDERS = ['Risky first', 'Risky second']
DEAD = '0.62'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': .7, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})

MODELS = [
    ('log-power-n1n2',     'Power · noise on both options (reported)'),
    ('log-power-percmem',  'Power · perceptual + memory noise'),
    ('log-power-perc',     'Power · perceptual noise only'),
    ('log-power-n2',       'Power · noise on 2nd-presented option'),
    ('log-power-n1',       'Power · noise on 1st-presented option'),
    ('log-power-mem',      'Power · memory noise only'),
    ('log-power-nullind',  'Power · no cTBS effect'),
    ('log-spl5-n1n2',      'Spline, 5 knots · noise on both options'),
    ('log-spl3-n1n2',      'Spline, 3 knots · noise on both options'),
    ('log-genweber-n1n2',  'Generalised Weber · noise on both options'),
    ('log-weber-n1n2',     'Weber, constant ν · noise on both options'),
    ('log-weber-nullind',  'Weber, constant ν · no cTBS effect'),
]


def main(data_dir, out_stem):
    dd = Path(data_dir)
    chk = pd.read_csv(dd / 'all_anchor_check.tsv', **READ).set_index('trace')
    have = [(l, n) for l, n in MODELS
            if (dd / 'ppc_anchor' / f'ppc_anchor.stakerung.{l}.tsv').exists()]
    n = len(have)
    fig, AX = plt.subplots(n, 4, figsize=(7.2, 1.18 * n), sharex=True,
                           sharey=True, constrained_layout=True, squeeze=False)
    slab = None
    for r, (lab, nm) in enumerate(have):
        d = pd.read_csv(dd / 'ppc_anchor' / f'ppc_anchor.stakerung.{lab}.tsv', **READ)
        # an older extraction wrote the median-split column as `stake2`
        d = d.rename(columns={'stake2': 'stake_grp'})
        if slab is None:
            slab = d.groupby('stake_grp')['stake_chf'].mean().round(0).astype(int)
        rmse = float(np.sqrt(((d.model - d.observed) ** 2).mean()))
        cov = float(((d.observed >= d.lo) & (d.observed <= d.hi)).mean())
        ok = bool(chk.loc[lab, 'ok']) if lab in chk.index else True
        c = 0
        for order in ORDERS:
            for sg in sorted(d.stake_grp.unique()):
                ax = AX[r, c]
                o = d[(d.order == order) & (d.stake_grp == sg)]
                for stim, col in (('vertex', VERTEX), ('ips', IPS)):
                    q = o[o.stim == stim].sort_values('frac')
                    ax.fill_between(q.frac, q.lo, q.hi, color=col, alpha=.20,
                                    lw=0, zorder=1)
                    ax.plot(q.frac, q.model, color=col, lw=1.1, zorder=2)
                    ax.plot(q.frac, q.observed,
                            'o' if stim == 'vertex' else 's', ms=2.6,
                            color=col, lw=0, zorder=4)
                ax.axhline(.5, color='.88', lw=.6, ls='--', zorder=0)
                ax.set_xscale('log')
                ax.set_xticks([1.5, 2, 3])
                ax.set_xticklabels(['1.5', '2', '3'])
                ax.minorticks_off()
                ax.set_ylim(.10, .95)
                ax.set_yticks([.25, .5, .75])
                if r == 0:
                    ax.set_title(f'{order}\n{"Low" if sg == 0 else "High"} stake'
                                 f'  ({slab[sg]} CHF)', fontsize=7.2)
                if r == n - 1:
                    ax.set_xlabel('Risky / safe payoff ratio')
                c += 1
        AX[r, 0].set_ylabel('P(chose risky)', fontsize=7)
        AX[r, 0].text(.0, 1.30 if r == 0 else 1.03, nm + ('' if ok else '   (did not converge)'),
                      transform=AX[r, 0].transAxes, ha='left', va='bottom',
                      fontsize=7.8, color='0.15' if ok else DEAD)
        AX[r, 3].text(.97, .06, f'RMSE {rmse:.3f}   in band {cov:.0%}',
                      transform=AX[r, 3].transAxes, ha='right', fontsize=6,
                      color='0.45')
    # one key, on the first panel
    ax = AX[0, 0]
    for stim, col, mk, y in (('IPS', IPS, 's', .93), ('Vertex', VERTEX, 'o', .80)):
        ax.plot(.07, y, mk, ms=2.8, transform=ax.transAxes, color=col)
        ax.plot([.12, .20], [y, y], transform=ax.transAxes, color=col, lw=1.1)
        ax.text(.24, y, stim, transform=ax.transAxes, fontsize=6.8,
                va='center', color=col)
    sns.despine(fig=fig, offset=2, trim=False)
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=170)
    print(f'wrote {out_stem}.pdf / .png  ({n} models)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/supp_ppc_gallery_stake'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem)
