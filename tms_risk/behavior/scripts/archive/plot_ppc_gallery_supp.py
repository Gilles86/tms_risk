"""Supplementary: the posterior predictive check for every candidate model.

One row per model, one column per presentation order. Observed choice
proportions against the safe payoff (dots) and the model's 95% posterior
predictive band, for IPS and vertex. The point of the page is that models which
lose the cTBS effect on the second-presented option lose the red-above-green
separation in the right-hand column while still tracking the overall curve --
which is why choice proportions alone rank them so weakly, and why the
slope-based check in the companion figure is the one that discriminates.

RMSE and the fraction of points inside the band are printed per model.

    python -m tms_risk.behavior.scripts.plot_ppc_gallery_supp
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
IPS, VERTEX = '#d62728', '#2ca02c'
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

#: readable name per label, in the order the page should read
PRETTY = [('log-power-n1n2', 'Noise on both options (reported model)'),
          ('log-power-n1n2psd', '+ cTBS on prior width'),
          ('log-power-n1psd', 'Noise on 1st option + prior width'),
          ('log-power-n2psd', 'Noise on 2nd option + prior width'),
          ('log-power-psd', 'Prior width only, no noise term'),
          ('log-weber-psd', "Weber noise (constant ν) + prior width"),
          ('log-weber-n1n2', "Weber noise (constant ν) on both options"),
          ('log-power-nullind', 'No cTBS effect anywhere')]


def main(data_dir, out_stem):
    dd = Path(data_dir) / 'ppc_anchor'
    have = {re.match(r'ppc_anchor\.safe\.(.+)\.tsv', Path(f).name).group(1)
            for f in glob.glob(str(dd / 'ppc_anchor.safe.*.tsv'))}
    models = [(l, n) for l, n in PRETTY if l in have]
    missing = [l for l, _ in PRETTY if l not in have]
    if missing:
        print('not extracted yet: ' + ', '.join(missing))
    if not models:
        raise SystemExit('no PPC files found')

    n = len(models)
    fig, axes = plt.subplots(n, 2, figsize=(5.4, 1.62 * n), sharex=True,
                             sharey=True, constrained_layout=True)
    axes = np.atleast_2d(axes)
    for r, (lab, nm) in enumerate(models):
        d = pd.read_csv(dd / f'ppc_anchor.safe.{lab}.tsv', **READ)
        rmse = float(np.sqrt(((d.model - d.observed) ** 2).mean()))
        cov = float(((d.observed >= d.lo) & (d.observed <= d.hi)).mean())
        for cc, order in enumerate(ORDERS):
            ax = axes[r, cc]
            o = d[d.order == order]
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                q = o[o.stim == stim].sort_values('n_safe')
                ax.fill_between(q.n_safe, q.lo, q.hi, color=col, alpha=.20, lw=0)
                ax.plot(q.n_safe, q.model, color=col, lw=1.1)
                ax.plot(q.n_safe, q.observed, 'o', ms=3, color=col, zorder=4)
            ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
            ax.set_xscale('log')
            ax.set_xticks(sorted(o.n_safe.unique()))
            ax.set_xticklabels([f'{v:.0f}' for v in sorted(o.n_safe.unique())])
            ax.minorticks_off()
            ax.set_ylim(.38, .76)
            ax.set_yticks([.45, .55, .65])
            if r == 0:
                # column titles sit ABOVE the row title, which is added below
                ax.text(.5, 1.34, order, transform=ax.transAxes, ha='center',
                        va='bottom', fontsize=8)
            if r == n - 1:
                ax.set_xlabel('Safe payoff (CHF)')
            if cc == 0:
                ax.set_ylabel('P(chose risky)', fontsize=7)
                # a rotated 6.2 pt label is unreadable at a glance; the model
                # is what the reader is scanning for, so give it a real title
                ax.text(.0, 1.04, nm, transform=ax.transAxes, ha='left',
                        va='bottom', fontsize=8, color='0.15')
        axes[r, 1].text(.98, .05, f'RMSE {rmse:.3f}   in band {cov:.0%}',
                        transform=axes[r, 1].transAxes, ha='right', fontsize=5.6,
                        color='0.45')
    axes[0, 0].text(.04, .95, 'IPS', color=IPS, transform=axes[0, 0].transAxes,
                    va='top', fontsize=6.5)
    axes[0, 0].text(.04, .78, 'Vertex', color=VERTEX,
                    transform=axes[0, 0].transAxes, va='top', fontsize=6.5)
    fig.supylabel('P(chose risky)', fontsize=7.5)
    sns.despine(fig=fig, offset=2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png  ({n} models)')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/supp_ppc_gallery'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem)
