"""Per-subject cTBS shift in sigma_n1 against sigma_n2, at two payoffs.

The group curves say the second-presented option's noise moves and the first's
does not. A hierarchical model shrinks, so this asks whether the same holds one
participant at a time, and whether the two channels move together.

Points on the identity line would mean cTBS raises both channels equally (what a
purely perceptual account predicts); points along the horizontal axis mean only
n2 moved.

    python -m tms_risk.behavior.scripts.plot_subject_shifts --model_label log-spl3-n1n2
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
ACC = '#3B5BA5'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main(data_dir, out_stem, label):
    d = pd.read_csv(Path(data_dir) / f'subject_shifts.{label}.tsv', **READ)
    payoffs = sorted(d.x.unique())
    fig, axes = plt.subplots(1, len(payoffs) + 1, figsize=(7.25, 2.7),
                             constrained_layout=True,
                             gridspec_kw={'width_ratios': [1] * len(payoffs) + [1.1]})

    lim = 0
    wide = {}
    for xv in payoffs:
        w = d[d.x == xv].pivot(index='subject', columns='channel',
                              values='shift_pct')
        wide[xv] = w
        lim = max(lim, np.abs(w.values).max())
    lim = 1.08 * lim

    for k, xv in enumerate(payoffs):
        ax, w = axes[k], wide[xv]
        ax.axhline(0, color='0.85', lw=.6, zorder=0)
        ax.axvline(0, color='0.85', lw=.6, zorder=0)
        ax.plot([-lim, lim], [-lim, lim], color='0.7', lw=.8, ls=':', zorder=1)
        ax.scatter(w['n2'], w['n1'], s=16, facecolor=ACC, alpha=.65,
                   edgecolor='white', lw=.5, zorder=3)
        r, p = stats.pearsonr(w['n2'], w['n1'])
        med = w.median()
        ax.plot(med['n2'], med['n1'], marker='D', ms=8, mfc='#d62728',
                mec='0.15', mew=1.4, zorder=5)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal')
        ax.set_xlabel('Δσ$_{n2}$, second-presented (%)')
        if k == 0:
            ax.set_ylabel('Δσ$_{n1}$, first-presented (%)')
        ax.set_title(f'{xv:.0f} CHF', fontsize=8, color='0.15', pad=3)
        ax.text(.03, .97, f'r = {r:.2f}\nMedian n1 {med["n1"]:+.1f}%\n'
                          f'Median n2 {med["n2"]:+.1f}%',
                transform=ax.transAxes, va='top', fontsize=6.4, color='0.25',
                linespacing=1.6)
    axes[0].text(.97, .03, 'Dotted: equal shift\n(purely perceptual)',
                 transform=axes[0].transAxes, ha='right', va='bottom',
                 fontsize=6.2, color='0.5', linespacing=1.5)

    # paired distribution of the n2 - n1 difference, the quantity in question
    ax = axes[-1]
    ax.axvline(0, color='0.85', lw=.6, zorder=0)
    for k, xv in enumerate(payoffs):
        w = wide[xv]
        diff = w['n2'] - w['n1']
        y = k + np.random.default_rng(0).uniform(-.13, .13, len(diff))
        ax.scatter(diff, y, s=13, facecolor=ACC, alpha=.55, edgecolor='white',
                   lw=.4, zorder=3)
        m = diff.median()
        q = np.quantile(diff, [.25, .75])
        ax.plot(q, [k, k], color='0.15', lw=1.6, zorder=4,
                solid_capstyle='butt')
        ax.plot([m], [k], marker='D', ms=7, mfc='#d62728', mec='0.15', mew=1.2,
                zorder=5)
        n_pos = int((diff > 0).sum())
        ax.text(ax.get_xlim()[1], k + .30, f'{n_pos}/{len(diff)} subjects > 0',
                fontsize=6.2, color='0.35', ha='right', va='bottom')
    ax.set_yticks(range(len(payoffs)))
    ax.set_yticklabels([f'{v:.0f} CHF' for v in payoffs])
    ax.set_ylim(-.5, len(payoffs) - .3)
    ax.set_xlabel('Δσ$_{n2}$ − Δσ$_{n1}$ (%)')
    ax.set_title('Per-subject asymmetry', fontsize=8, color='0.15', pad=3)

    for ax, s in zip(axes, 'abc'):
        ax.text(-.20, 1.06, s, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left', family='Arial')
    fig.suptitle(f'{label} · posterior median per subject, 35 participants',
                 fontsize=7, color='0.4', y=1.04)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data/subject_shifts'))
    ap.add_argument('--model_label', default='log-spl3-n1n2')
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    out = a.out_stem or str(REPO / f'notes/figures/subject_shifts.{a.model_label}')
    main(a.data_dir, out, a.model_label)
