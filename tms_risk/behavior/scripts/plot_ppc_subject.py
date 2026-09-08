"""Does the model track INDIVIDUAL participants, or only the group mean?

The group-level panels of Figure 5 can look right while every participant is
predicted badly, and the psychometric PPCs pool participants inside a cell, so
neither answers this. Both panels here work at the level of one participant in
one cell -- (participant x order x stake tercile x stimulation), 420 cells, a
median of 20 trials each.

a  Observed against predicted choice proportion, one point per cell, with the
   model's 95% predictive interval as a vertical whisker. The identity line is
   the reference, not a fit. Coverage -- the fraction of cells whose observed
   value lies inside its own interval -- is printed: 95% is what a calibrated
   model should give, and much more than that means the intervals are wider
   than they need to be.

b  The cTBS contrast (IPS - vertex) per participant, observed against
   predicted. This is the harder test and the one that matters for the
   brain-behaviour analysis: a model can reproduce every participant's choice
   level and still put the individual differences in the STIMULATION EFFECT
   entirely in the wrong place. The dashed line is identity; the regression is
   deliberately not drawn, because the question is calibration, not
   association.

Note b is expected to be compressed toward zero. Partial pooling shrinks each
participant's contrast toward the group, so sd(model) < sd(observed) by
construction, and part of the observed spread is measurement error rather than
real individual differences (see anchor_subject_reliability).

    python -m tms_risk.behavior.scripts.plot_ppc_subject \\
        --model_label log-power-n1n2.mapjitter.klw.ti0.1
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

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})


def main(data_dir, out_stem, label):
    d = pd.read_csv(Path(data_dir) / 'ppc_anchor' / f'ppc_subject.{label}.tsv',
                    **READ)
    fig, AX = plt.subplots(1, 2, figsize=(6.6, 3.1), constrained_layout=True)

    ax = AX[0]
    ax.plot([0, 1], [0, 1], color='0.6', lw=.9, ls='--', zorder=1)
    for stim, col in (('vertex', VERTEX), ('ips', IPS)):
        q = d[d.stim == stim]
        ax.vlines(q.model, q.lo, q.hi, color=col, lw=.5, alpha=.30, zorder=2)
        ax.plot(q.model, q.observed, 'o', ms=2.6, color=col, alpha=.75,
                mew=0, zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Predicted P(chose risky)')
    ax.set_ylabel('Observed P(chose risky)')
    ax.set_title('Each participant, each cell', fontsize=9)
    ax.text(.04, .96, f'{len(d)} cells\n{d.covered.mean():.0%} inside the 95% interval\n'
                      f'r = {np.corrcoef(d.observed, d.model)[0, 1]:.2f}',
            transform=ax.transAxes, va='top', fontsize=7, color='0.25')
    ax.text(.62, .12, 'IPS', transform=ax.transAxes, color=IPS, fontsize=8)
    ax.text(.62, .04, 'Vertex', transform=ax.transAxes, color=VERTEX, fontsize=8)

    ax = AX[1]
    w = d.pivot_table(index=['subject', 'order', 'stake_bin'], columns='stim',
                      values=['observed', 'model'])
    x = (w[('model', 'ips')] - w[('model', 'vertex')])
    y = (w[('observed', 'ips')] - w[('observed', 'vertex')])
    lim = 1.05 * max(np.abs(x).max(), np.abs(y).max())
    ax.plot([-lim, lim], [-lim, lim], color='0.6', lw=.9, ls='--', zorder=1)
    ax.axhline(0, color='0.85', lw=.7, zorder=0)
    ax.axvline(0, color='0.85', lw=.7, zorder=0)
    ax.plot(x, y, 'o', ms=3.0, color='0.25', alpha=.65, mew=0, zorder=3)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_xlabel('Predicted cTBS effect (IPS − vertex)')
    ax.set_ylabel('Observed cTBS effect (IPS − vertex)')
    ax.set_title('The stimulation effect, per participant', fontsize=9)
    ax.text(.04, .96,
            f'r = {np.corrcoef(x, y)[0, 1]:.2f}\n'
            f'SD observed {y.std():.2f}\nSD predicted {x.std():.2f}',
            transform=ax.transAxes, va='top', fontsize=7, color='0.25')

    for letter, ax in zip('ab', AX):
        ax.text(-.16, 1.04, letter, transform=ax.transAxes, fontsize=9,
                fontweight='bold', family='Arial', va='bottom', ha='right')
    sns.despine(fig=fig, offset=4, trim=False)
    fig.savefig(f'{out_stem}.pdf')
    fig.savefig(f'{out_stem}.png', dpi=200)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2.mapjitter.klw')
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    main(a.data_dir, a.out_stem or
         str(REPO / f'notes/figures/supp_ppc_subject_{a.model_label}'),
         a.model_label)
