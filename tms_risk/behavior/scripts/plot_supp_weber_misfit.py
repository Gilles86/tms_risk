"""Supplementary: Weber's law cannot produce the stake-dependence of choice.

Under Weber's law noise is constant in log space, so the psychometric function
has the same slope whatever the payoffs are worth. The data say otherwise. This
figure puts the two models side by side on the same psychometric curves, split
by stake tercile, in the style of Figure 4: observed proportions as points, the
model's 95% POSTERIOR PREDICTIVE band behind them. No error bars on the data --
the band is what carries the uncertainty, and a point outside it is a misfit.

Rows are models, columns are presentation order. The bands are pooled over
stimulation condition (this figure is about magnitude, not cTBS), simulated once
pooled rather than averaged from the two per-condition bands.

    python -m tms_risk.behavior.scripts.plot_supp_weber_misfit
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
#: low -> high stake, a sequential ramp so the ORDER of the terciles is visible
STAKE = ['#9EC5E8', '#3B7DBF', '#123A63']
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})



def probit_slope(frac, p):
    x = np.log(np.asarray(frac, float)); x = x - x.mean()
    y = norm.ppf(np.clip(np.asarray(p, float), 1e-3, 1 - 1e-3))
    return float(x @ (y - y.mean()) / (x @ x))


def main(data_dir, out_stem, weber_label, power_label):
    dd = Path(data_dir) / 'ppc_pooled'
    D = {nm: pd.read_csv(dd / f'ppc_stake_pooled.{lab}.tsv', **READ)
         for nm, lab in (("Weber's law", weber_label), ('Power law', power_label))}
    any_ = next(iter(D.values()))
    stakes = sorted(any_.stake_grp.unique())
    slab = any_.groupby('stake_grp')['stake'].mean().round(0).astype(int)

    fig, AX = plt.subplots(2, 3, figsize=(7.2, 4.6), constrained_layout=True,
                           gridspec_kw=dict(width_ratios=[1, 1, .85]))

    for r, (nm, d) in enumerate(D.items()):
        for c, order in enumerate(ORDERS):
            ax = AX[r, c]
            o = d[d.order == order]
            for sg, col in zip(stakes, STAKE):
                q = o[o.stake_grp == sg].sort_values('frac')
                ax.fill_between(q.frac, q.lo, q.hi, color=col, alpha=.25, lw=0,
                                zorder=1)
                ax.plot(q.frac, q.model, color=col, lw=1.6, zorder=2)
                ax.plot(q.frac, q.observed, 'o', ms=4.2, color=col, zorder=4)
            ax.axhline(.5, color='0.85', lw=.7, ls='--', zorder=0)
            ax.set_xscale('log')
            ax.set_xticks([1.5, 2, 3])
            ax.set_xticklabels(['1.5', '2', '3'])
            ax.minorticks_off()
            ax.set_ylim(.08, .95)
            ax.set_yticks([.25, .5, .75])
            if r == 1:
                ax.set_xlabel('Risky / safe payoff ratio')
            else:
                ax.tick_params(labelbottom=False)
            if c == 0:
                ax.set_ylabel(f'{nm}\nP(chose risky)')
            else:
                ax.tick_params(labelleft=False)
            if r == 0:
                ax.set_title(order, fontsize=9.5)

        # -- right column: the slope each model implies, against stake -------
        ax = AX[r, 2]
        for c, (order, mk, ls) in enumerate([('Risky first', 'o', (0, (2.5, 1.5))),
                                             ('Risky second', 's', '-')]):
            o = d[d.order == order]
            xs, ym, yo = [], [], []
            for sg in stakes:
                q = o[o.stake_grp == sg]
                xs.append(q.stake.mean())
                ym.append(probit_slope(q.frac, q.model))
                yo.append(probit_slope(q.frac, q.observed))
            ax.plot(xs, yo, ls=ls, color='0.15', lw=1.9, marker=mk, ms=4.5,
                    zorder=4)
            ax.plot(xs, ym, ls=ls, color=STAKE[1], lw=1.5, marker=mk, ms=4,
                    zorder=3)
        ax.set_xscale('log')
        ax.set_xticks(list(slab.values))
        ax.set_xticklabels([f'{v:.0f}' for v in slab.values])
        ax.minorticks_off()
        ax.set_ylim(1.35, 2.95)
        if r == 1:
            ax.set_xlabel('Stake (CHF)')
        else:
            ax.tick_params(labelbottom=False)
            ax.set_title('Implied consistency', fontsize=9.5)
        ax.set_ylabel('Probit slope')

    # keys, drawn as the marks they name
    ax = AX[0, 0]
    for i, (sg, col) in enumerate(zip(stakes, STAKE)):
        y = .95 - .085 * i
        ax.plot(.06, y, 'o', ms=4.2, transform=ax.transAxes, color=col)
        ax.add_patch(plt.Rectangle((.10, y - .017), .06, .034,
                                   transform=ax.transAxes, facecolor=col,
                                   alpha=.25, lw=0))
        ax.text(.185, y, f'Stake ≈ {slab[sg]} CHF', transform=ax.transAxes,
                fontsize=7, va='center', color=col)
    ax.text(.06, .95 - .085 * 3.1, 'Dots: observed · band: 95% predictive',
            transform=ax.transAxes, fontsize=6.4, va='center', color='0.45')
    ax = AX[0, 2]
    ax.plot([.06, .16], [.95, .95], transform=ax.transAxes, color='0.15', lw=1.9)
    ax.text(.20, .95, 'Observed', transform=ax.transAxes, fontsize=7,
            va='center', color='0.15')
    ax.plot([.06, .16], [.86, .86], transform=ax.transAxes, color=STAKE[1], lw=1.5)
    ax.text(.20, .86, 'Model', transform=ax.transAxes, fontsize=7,
            va='center', color=STAKE[1])
    ax.text(.06, .77, 'Solid: risky second', transform=ax.transAxes,
            fontsize=6.4, va='center', color='0.45')
    ax.text(.06, .70, 'Dashed: risky first', transform=ax.transAxes,
            fontsize=6.4, va='center', color='0.45')

    for letter, ax in zip('abcdef', AX.ravel()):
        ax.text(-.19, 1.05, letter, transform=ax.transAxes, fontsize=10.5,
                fontweight='bold', family='Arial', va='bottom')
    sns.despine(fig=fig, offset=4, trim=False)
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--weber_label', default='log-weber-n1n2.mapjitter.klw')
    ap.add_argument('--power_label', default='log-power-n1n2.mapjitter.klw')
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/supp_weber_misfit'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.weber_label, a.power_label)
