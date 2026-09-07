"""Psychometric curves in the four stake x order cells, model against data.

The bias statistics — where the curve sits — can be matched by a model that gets
the SLOPE wrong, and the slope is what a change in choice consistency shows up
as. So keep the whole curve inside each cell instead of collapsing it: rows are
the four low/high stake x risky-first/second cells, columns are models, colour
is stimulation.

What to look for: the observed IPS curve is FLATTER than vertex in the
risky-second cells. A model reproduces that only if its red band is flatter than
its green one there — a vertical offset between the two is a bias shift, not a
consistency change.

    python -m tms_risk.behavior.scripts.plot_ppc_psychometric \\
        log-power-nullind log-power-n1n2 log-power-psd log-power-n1n2psd
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IPS, VERTEX = '#d62728', '#2ca02c'
READ = dict(sep='\t', keep_default_na=False, na_values=[''])
CELLS = [('Risky first', 0), ('Risky second', 0),
         ('Risky first', 1), ('Risky second', 1)]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
PRETTY = {'nullind': 'Null', 'n1n2': 'Noise only', 'psd': 'Prior width only',
          'n1n2psd': 'Noise + prior width', 'pmusd': 'Prior mean + width',
          'pmu': 'Prior mean only', 'n1n2pmu': 'Noise + prior mean'}


def slope(x, y):
    x = np.log(np.asarray(x, float))
    x = x - x.mean()
    return float((x @ np.asarray(y, float)) / (x @ x))


def main(data_dir, out_stem, labels):
    dd = Path(data_dir)
    fig, axes = plt.subplots(4, len(labels), figsize=(1.75 * len(labels) + .8, 7.4),
                             sharex=True, sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes.reshape(4, -1))

    for j, lbl in enumerate(labels):
        f = dd / f'ppc_anchor.stakerung.{lbl}.tsv'
        if not f.exists():
            for r in range(4):
                axes[r, j].text(.5, .5, 'not extracted', ha='center',
                                transform=axes[r, j].transAxes, color='0.6')
            continue
        d = pd.read_csv(f, **READ)
        for r, (order, sb) in enumerate(CELLS):
            ax = axes[r, j]
            s = d[(d.order == order) & (d.stake2 == sb)]
            sl = {}
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                o = s[s.stim == stim].sort_values('frac')
                ax.fill_between(o.frac, o.lo, o.hi, color=col, alpha=.20, lw=0)
                ax.plot(o.frac, o.model, color=col, lw=1.1)
                ax.errorbar(o.frac, o.observed, yerr=o.observed_sem, fmt='o',
                            ms=3.0, color=col, lw=0, elinewidth=.8, capsize=0,
                            zorder=4)
                sl[stim] = (slope(o.frac, o.observed), slope(o.frac, o.model))
            ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
            ax.set_ylim(.10, 1.0)
            ax.set_yticks([.2, .4, .6, .8])
            # the number the eye cannot read off a pair of curves
            d_obs = sl['ips'][0] - sl['vertex'][0]
            d_mod = sl['ips'][1] - sl['vertex'][1]
            ax.text(.97, .05, f'Δslope\ndata {d_obs:+.2f}\nmodel {d_mod:+.2f}',
                    transform=ax.transAxes, ha='right', va='bottom',
                    fontsize=5.8, color='0.3', linespacing=1.5)
            if j == 0:
                stake = 'Low stake' if sb == 0 else 'High stake'
                ax.set_ylabel(f'{stake}\n{order.lower()}\n\nP(chose risky)',
                              fontsize=7)
            if r == 3:
                ax.set_xlabel('Risky/safe ratio')
            ax.set_xticks([1.5, 2.0, 2.5, 3.0])
        axes[0, j].set_title(PRETTY.get(lbl.split('-')[2].replace('.klw', ''),
                                        lbl), fontsize=7.5, color='0.15', pad=4)

    axes[0, 0].text(.05, .96, 'IPS', color=IPS, transform=axes[0, 0].transAxes,
                    va='top', fontsize=7)
    axes[0, 0].text(.05, .84, 'Vertex', color=VERTEX,
                    transform=axes[0, 0].transAxes, va='top', fontsize=7)
    rule = 'KLW-consistent choice noise' if labels[0].endswith('.klw') else \
           'raw-evidence-SD choice noise'
    fig.suptitle(f'Psychometric curves by stake and order · {rule} · '
                 f'points data ± SEM, bands 95% PPI · Δslope is IPS − vertex',
                 fontsize=6.8, color='0.4', y=1.015)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+')
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data/ppc_anchor'))
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/ppc_psychometric'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.labels)
