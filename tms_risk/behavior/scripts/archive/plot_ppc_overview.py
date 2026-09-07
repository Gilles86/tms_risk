"""One PDF with the posterior predictive check for every fitted model, for skimming.

Page 1  risky-first trials, one small panel per model
Page 2  risky-second trials, same layout
Page 3  goodness-of-fit summary: RMSE of model minus observed, per model, sorted,
        split by presentation order -- the page to read first

Each panel: observed choice proportions with +/-1 SEM across subjects (points), and the
95% posterior predictive interval (band), for IPS and vertex. A model that fits has the
band covering the points in both conditions; a model that fails is obvious at this size
because the band misses the points systematically in one condition or at one end.

Reads every notes/data/ppc_fig3a.<label>.tsv, which
`behavior/scripts/plot_ppc_fig3a.py` writes on the node holding each trace.

    python -m tms_risk.behavior.scripts.plot_ppc_overview
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
from matplotlib.backends.backend_pdf import PdfPages

IPS, VERTEX = '#d62728', '#2ca02c'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.7, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
    'lines.linewidth': 1.0, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 130, 'savefig.dpi': 300,
})

# Readable names, and the order they should appear in (best-motivated first).
PRETTY = {
    'flexible2nf_perception': 'Flexible 2 · perceptual noise only',
    'flexible2nf': 'Flexible 2 · perceptual + memory',
    'flexible2nf_memory': 'Flexible 2 · memory noise only',
    'flexible2nf_null': 'Flexible 2 · null',
    'flexible1nf': 'Flexible 1 · both options',
    'flexible1nf_first': 'Flexible 1 · first option only',
    'flexible1nf_second': 'Flexible 1 · second option only',
    'flexible1nf_null': 'Flexible 1 · null',
    'flexible1': 'Flexible 1 · PUBLISHED (ecc6454)',
    'flexible2.9nf': 'Flexible 2 · df = 9 spline',
    'weber2nf_perception': 'Weber · perceptual noise only',
    'weber2nf': 'Weber · perceptual + memory',
    'weber2nf_memory': 'Weber · memory noise only',
    'weber2nf_null': 'Weber · null',
    'objprior_perception': 'Objective prior · perceptual noise',
}


def load(data_dir):
    out = {}
    for fn in sorted(glob.glob(str(Path(data_dir) / 'ppc_fig3a.*.tsv'))):
        label = re.match(r'ppc_fig3a\.(.+)\.tsv', Path(fn).name).group(1)
        d = pd.read_csv(fn, sep='\t')
        if {'order', 'frac', 'stim', 'mean', 'lo', 'hi', 'observed'} <= set(d.columns):
            out[label] = d
    return out


def panel(ax, d, title):
    for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
        g = d[d.stim == stim].sort_values('frac')
        ax.fill_between(g.frac, g.lo, g.hi, color=col, alpha=.22, lw=0, zorder=1)
        ax.errorbar(g.frac, g.observed, yerr=g.observed_sem, fmt='o', ms=2.6,
                    color=col, lw=0, elinewidth=.7, capsize=0, zorder=3)
    ax.axhline(.5, color='.75', lw=.5, ls=':', zorder=0)
    ax.set_xscale('log')
    ax.set_ylim(-.02, 1.02)
    ax.set_yticks([0, .5, 1])
    ax.set_title(title, fontsize=6.8, pad=2)


def main(data_dir, out_pdf, ncol):
    data = load(data_dir)
    labels = [l for l in PRETTY if l in data] + [l for l in data if l not in PRETTY]
    print(f'{len(labels)} models: {", ".join(labels)}')
    nrow = int(np.ceil(len(labels) / ncol))

    with PdfPages(out_pdf) as pdf:
        # ------------------------------------------------- pages 1-2: the PPCs
        for order in ['Risky first', 'Risky second']:
            fig, axes = plt.subplots(nrow, ncol, figsize=(9.5, 2.0 * nrow),
                                     sharex=True, sharey=True,
                                     constrained_layout=True)
            axes = np.atleast_1d(axes).ravel()
            for ax, label in zip(axes, labels):
                panel(ax, data[label][data[label].order == order],
                      PRETTY.get(label, label))
            for ax in axes[len(labels):]:
                ax.set_visible(False)
            for ax in axes[:len(labels)]:
                sns.despine(ax=ax, offset=2)
            fig.suptitle(f'Posterior predictive checks — {order}', fontsize=10,
                         fontweight='bold')
            fig.supxlabel('Risky / safe payoff ratio', fontsize=8)
            fig.supylabel('P(chose risky)', fontsize=8)
            # one legend for the whole page
            axes[0].plot([], [], 'o', color=IPS, ms=3, label='IPS')
            axes[0].plot([], [], 'o', color=VERTEX, ms=3, label='Vertex')
            axes[0].legend(loc='upper left', fontsize=6, handletextpad=.3)
            pdf.savefig(fig)
            plt.close(fig)

        # ------------------------------------------------ page 3: fit summary
        rows = []
        for label, d in data.items():
            for order, g in d.groupby('order'):
                rows.append(dict(model=PRETTY.get(label, label), order=order,
                                 rmse=np.sqrt(((g['mean'] - g.observed) ** 2).mean()),
                                 max_abs=np.abs(g['mean'] - g.observed).max(),
                                 covered=((g.observed >= g.lo) & (g.observed <= g.hi)).mean()))
        S = pd.DataFrame(rows)
        order_by = (S.groupby('model').rmse.mean().sort_values().index.tolist())
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 0.30 * len(order_by) + 1.6),
                                 constrained_layout=True, sharey=True)
        for ax, (metric, lab) in zip(axes, [('rmse', 'RMSE, model − observed'),
                                            ('covered', 'Fraction of bins inside the 95% band')]):
            for o, mk, c in [('Risky first', 'o', '#4d4d4d'), ('Risky second', 's', '#b2182b')]:
                g = S[S.order == o].set_index('model').reindex(order_by)
                ax.plot(g[metric], np.arange(len(order_by)), mk, ms=4, lw=0, color=c,
                        label=o, alpha=.85)
            ax.set_yticks(np.arange(len(order_by)))
            ax.set_yticklabels(order_by, fontsize=6.5)
            ax.set_xlabel(lab)
            ax.invert_yaxis()
            sns.despine(ax=ax, offset=2)
        axes[1].axvline(0.95, color='.7', lw=.6, ls='--', zorder=0)
        axes[0].legend(loc='lower right', fontsize=7)
        fig.suptitle('Goodness of fit, sorted by mean RMSE (best at top)',
                     fontsize=10, fontweight='bold')
        pdf.savefig(fig)
        plt.close(fig)
        S.to_csv(Path(data_dir) / 'ppc_fit_summary.tsv', sep='\t', index=False)

    print(f'wrote {out_pdf} (3 pages) and ppc_fit_summary.tsv')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--out', default='notes/figures/ppc_overview.pdf')
    p.add_argument('--ncol', default=4, type=int)
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.data_dir, a.out, a.ncol)
