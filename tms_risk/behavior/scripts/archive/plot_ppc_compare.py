"""Side-by-side posterior predictive checks for a handful of named models.

Built to answer one question: when a model wins on ELPD, can you SEE where it
wins? Rows are the two presentation orders plus the cTBS difference, which is
where the paper's claim lives; columns are models.

Row 3 is the one to read. The observed cTBS effect on choices is a curve, not a
number, and a model that gains ELPD without tracking that curve is gaining it
somewhere the paper does not care about.

    python -m tms_risk.behavior.scripts.plot_ppc_compare \\
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
ORDERS = ['Risky first', 'Risky second']

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


def main(data_dir, out_stem, labels, pretty):
    dd = Path(data_dir)
    n = len(labels)
    fig, axes = plt.subplots(3, n, figsize=(1.85 * n + .7, 5.6), sharex=True,
                             constrained_layout=True)
    axes = np.atleast_2d(axes)

    for j, lbl in enumerate(labels):
        f = dd / f'ppc_anchor.rung.{lbl}.tsv'
        fd = dd / f'ppc_anchor.delta_rung.{lbl}.tsv'
        if not f.exists():
            for r in range(3):
                axes[r, j].text(.5, .5, 'no PPC', transform=axes[r, j].transAxes,
                                ha='center', color='0.6')
            continue
        d = pd.read_csv(f, **READ)
        for r, order in enumerate(ORDERS):
            ax = axes[r, j]
            s = d[d.order == order]
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                o = s[s.stim == stim].sort_values('frac')
                ax.fill_between(o.frac, o.lo, o.hi, color=col, alpha=.20, lw=0)
                ax.plot(o.frac, o.model, color=col, lw=1.1)
                ax.errorbar(o.frac, o.observed, yerr=o.observed_sem, fmt='o',
                            ms=3.2, color=col, lw=0, elinewidth=.85, capsize=0,
                            zorder=4)
            ax.axhline(.5, color='0.88', lw=.6, ls='--', zorder=0)
            ax.set_ylim(.15, .95)
            ax.set_yticks([.2, .4, .6, .8])
            if j == 0:
                ax.set_ylabel(f'P(chose risky)\n{order.lower()}')
            else:
                ax.set_yticklabels([])

        ax = axes[2, j]
        if fd.exists():
            e = pd.read_csv(fd, **READ)
            ax.axhline(0, color='0.75', lw=.6, ls='--', zorder=0)
            for order, col, mfc in [('Risky first', '0.62', 'white'),
                                    ('Risky second', '0.15', '0.15')]:
                o = e[e.order == order].sort_values('frac')
                ax.fill_between(o.frac, o.lo, o.hi, color=col, alpha=.20, lw=0)
                ax.plot(o.frac, o.model, color=col, lw=1.1)
                ax.errorbar(o.frac, o.observed, yerr=o.observed_sem, fmt='o',
                            ms=3.2, color=col, mfc=mfc, mew=.7, lw=0,
                            elinewidth=.85, capsize=0, zorder=4)
            ax.set_ylim(-.16, .22)
            ax.set_yticks([-.1, 0, .1, .2])
            if j == 0:
                ax.set_ylabel('ΔP(chose risky)\nIPS − vertex')
            else:
                ax.set_yticklabels([])
        ax.set_xticks([1.5, 2.0, 2.5, 3.0])
        ax.set_xlabel('Risky/safe ratio')
        axes[0, j].set_title(pretty.get(lbl, lbl), fontsize=7.5, color='0.15',
                             pad=4)

    axes[0, 0].text(.05, .96, 'IPS', color=IPS, transform=axes[0, 0].transAxes,
                    va='top', fontsize=7)
    axes[0, 0].text(.05, .82, 'Vertex', color=VERTEX,
                    transform=axes[0, 0].transAxes, va='top', fontsize=7)
    axes[2, 0].text(.05, .96, 'Risky second', color='0.15',
                    transform=axes[2, 0].transAxes, va='top', fontsize=6.8)
    axes[2, 0].text(.05, .82, 'Risky first', color='0.62',
                    transform=axes[2, 0].transAxes, va='top', fontsize=6.8)
    fig.suptitle('Points: data ± SEM across 35 subjects.   Bands: 95% posterior '
                 'predictive interval from simulated choices.',
                 fontsize=6.8, color='0.4', y=1.02)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='+')
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data/ppc_anchor'))
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/ppc_compare'))
    a = ap.parse_args()
    PRETTY = {l + suf: n + ('' if not suf else '')
              for suf in ('', '.klw')
              for l, n in [('log-power-nullind', 'Null\nno cTBS effect'),
                           ('log-power-n1n2', 'Noise only\nn1 + n2'),
                           ('log-power-psd', 'Prior width only\npsd'),
                           ('log-power-n1n2psd', 'Noise + prior width\nn1n2psd'),
                           ('log-power-pmu', 'Prior mean only\npmu'),
                           ('log-power-pmusd', 'Prior mean + width\npmusd')]}
    _unused = {'log-power-nullind': 'Null\nno cTBS effect',
              'log-power-n1n2': 'Noise only\nn1 + n2',
              'log-power-psd': 'Prior width only\npsd',
              'log-power-n1n2psd': 'Noise + prior width\nn1n2psd',
              'log-power-pmu': 'Prior mean only\npmu'}
    main(a.data_dir, a.out_stem, a.labels, PRETTY)
