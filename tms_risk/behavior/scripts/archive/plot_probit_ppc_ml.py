"""Figure 3B against the model: observed cTBS effect vs its predictive distribution.

Figure 3B reports what cTBS did to the two parameters of the psychophysical
model -- the SLOPE (choice consistency) and the risk-neutral probability at
indifference (risk attitude). This asks whether the PMC model predicts those two
numbers, by running the paper's own estimator on choices simulated from the
posterior (see `probit_ppc_ml`).

Bars are the 95% predictive interval over 200 posterior draws with the
predictive median; the marker is the observed value. A marker outside its bar is
a failed check, and the posterior predictive p-value is printed beside it.

    python -m tms_risk.behavior.scripts.plot_probit_ppc_ml --model_label log-power-n2psd
"""
import argparse
import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
OK, BAD = '#3B5BA5', '#C44E52'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

PARAMS = [('slope', 'cTBS effect on\nprobit slope'),
          ('rnp', 'cTBS effect on\nrisk-neutral probability')]
ORDERS = ['Risky second', 'Risky first']
STAKES = ['low', 'high']


def main(data_dir, out_stem, labels):
    dd = Path(data_dir) / 'probit_ppc_ml'
    d = pd.concat([pd.read_csv(f, **READ)
                   for f in sorted(glob.glob(str(dd / '*.tsv')))],
                  ignore_index=True)
    d['short'] = d.label.str.replace('log-power-', '', regex=False)
    labels = labels or ['n2psd']
    d = d[d.short.isin(labels)]
    if not len(d):
        raise SystemExit(f'no rows for {labels}')

    n = len(labels)
    fig, axes = plt.subplots(2, 2, figsize=(5.0, 4.0), constrained_layout=True,
                             sharex=True)
    for r, (par, ylab) in enumerate(PARAMS):
        for c, order in enumerate(ORDERS):
            ax = axes[r, c]
            xs = []
            for si, stake in enumerate(STAKES):
                for li, lab in enumerate(labels):
                    s = d[(d.parameter == par) & (d.order == order)
                          & (d.stake == stake) & (d.short == lab)]
                    if not len(s):
                        continue
                    s = s.iloc[0]
                    x = si + (li - (n - 1) / 2) * .22
                    xs.append(x)
                    fail = s.ppp > .95 or s.ppp < .05
                    col = BAD if fail else OK
                    ax.plot([x, x], [s.lo, s.hi], color=col, lw=5, alpha=.28,
                            solid_capstyle='butt', zorder=1)
                    ax.plot([x - .09, x + .09], [s.model] * 2, color=col, lw=1.4,
                            zorder=2)
                    ax.plot(x, s.observed, 'o', ms=5, color='0.12', zorder=4)
                    ax.text(x + .12, s.observed, f'p={s.ppp:.2f}', fontsize=5.6,
                            va='center', color=col if fail else '0.45')
            ax.axhline(0, color='0.75', lw=.7, ls='--', zorder=0)
            ax.set_xticks(range(len(STAKES)))
            ax.set_xticklabels([s.capitalize() for s in STAKES])
            ax.set_xlim(-.5, len(STAKES) - .25)
            if r == 0:
                ax.set_title(order, fontsize=8)
            else:
                ax.set_xlabel('Stake')
            if c == 0:
                ax.set_ylabel(ylab)
            else:
                ax.tick_params(labelleft=False)
        lo = min(axes[r, i].get_ylim()[0] for i in range(2))
        hi = max(axes[r, i].get_ylim()[1] for i in range(2))
        for i in range(2):
            axes[r, i].set_ylim(lo, hi)

    axes[0, 0].plot([], [], 'o', ms=5, color='0.12', label='Observed')
    axes[0, 0].plot([], [], lw=5, alpha=.28, color=OK, label='95% predictive')
    axes[0, 0].legend(loc='lower left', fontsize=6, handlelength=1.2,
                      labelspacing=.25)
    for ax, s in [(axes[0, 0], 'a'), (axes[1, 0], 'b')]:
        ax.text(-.30, 1.03, s, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', family='Arial', va='bottom')
    fig.suptitle(f'Model: {", ".join(labels)}   ·   red = observed value outside '
                 'the predictive interval', fontsize=6.8, color='0.35', y=1.02)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', nargs='+', default=None,
                    help='short labels, e.g. n2psd n1n2')
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    labs = a.model_label or ['n2psd']
    stem = a.out_stem or str(REPO / f'notes/figures/probit_ppc_{"_".join(labs)}')
    main(a.data_dir, stem, labs)
