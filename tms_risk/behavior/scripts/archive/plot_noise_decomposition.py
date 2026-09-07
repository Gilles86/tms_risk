"""Memory / perceptual decomposition of the fitted noise, with curve-level inference.

Reads noisecurve_reparam.<label>.tsv from noise_curve_inference.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS, DIFF = '#2ca02c', '#d62728', '#1f6fb4'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')

TITLE = {'perceptual': 'Perceptual noise\n(both options)',
         'memory': 'Memory noise\n(first-presented option only)'}


def logx(ax, hi):
    ax.set_xscale('log')
    ticks = [t for t in [10, 20, 40, 80] if t <= hi]
    ax.set_xticks(ticks)
    ax.set_xlim(7, hi)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())


def main(data_dir, label, regional, out_stem):
    c = pd.read_csv(Path(data_dir) / f'noisecurve_reparam.{label}.tsv', sep='\t')
    reg = pd.read_csv(Path(data_dir) / regional, sep='\t') if regional else None
    hi = c.payoff.max()

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.2), sharex=True,
                             constrained_layout=True)
    for col, term in enumerate(['perceptual', 'memory']):
        ax = axes[0, col]
        for cond, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = c[(c.term == term) & (c.stimulation == cond)]
            ax.fill_between(s.payoff, s.lo, s.hi, color=colr, alpha=.18, lw=0)
            ax.plot(s.payoff, s.nu, color=colr, lw=1.5)
        ax.set_title(TITLE[term], fontsize=8.5, color='0.15')
        ax.set_ylim(0, 5.6)
        if col == 0:
            ax.set_ylabel('Representational noise ν (CHF)')
            ax.text(-.21, 1.10, 'a', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right')
        logx(ax, hi)

        ax = axes[1, col]
        s = c[(c.term == term) & (c.stimulation == 'ips - vertex')]
        ax.axhline(0, color='0.75', lw=.6, ls='--', zorder=1)
        # widest band first: simultaneous (whole-curve corrected), then pointwise
        if 'sim_lo' in s:
            ax.fill_between(s.payoff, s.sim_lo, s.sim_hi, color=DIFF, alpha=.12,
                            lw=0, zorder=2)
        ax.fill_between(s.payoff, s.lo, s.hi, color=DIFF, alpha=.28, lw=0, zorder=3)
        ax.plot(s.payoff, s.nu, color=DIFF, lw=1.6, zorder=4)
        pw = s[s.lo > 0]
        if len(pw):
            ax.plot([pw.payoff.min(), pw.payoff.max()], [.86, .86], color=DIFF,
                    lw=2.4, solid_capstyle='butt', zorder=5)
            ax.text(np.sqrt(pw.payoff.min() * pw.payoff.max()), .90,
                    'pointwise 95% CrI > 0', fontsize=6, color=DIFF,
                    ha='center', va='bottom')
        ax.set_ylim(-1.0, 1.05)
        if col == 0:
            ax.set_ylabel('Δ ν, IPS − vertex (CHF)')
            ax.text(-.21, 1.06, 'b', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right')
        ax.set_xlabel('Payoff magnitude (CHF)')
        logx(ax, hi)
        if reg is not None:
            r = reg[(reg.lo == 7) & (reg.hi == 20)]
            if len(r):
                ax.text(.97, .04, f'P(Δν>0 over 7–20 CHF) = {r.iloc[0][f"p_{term}"]:.3f}',
                        transform=ax.transAxes, fontsize=6.5, color='0.25', ha='right')

    axes[0, 0].text(7.6, 5.2, 'Vertex', color=VERTEX, fontsize=8)
    axes[0, 0].text(7.6, 4.6, 'IPS', color=IPS, fontsize=8)
    axes[1, 1].text(.97, .93, 'Dark: pointwise   Pale: simultaneous',
                    transform=axes[1, 1].transAxes, fontsize=6, color=DIFF, ha='right')
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible1')
    parser.add_argument('--regional', default='noisecurve_regional.flexible1.tsv')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/noise_decomposition')
    args = parser.parse_args()
    main(args.data_dir, args.label, args.regional, args.out)
