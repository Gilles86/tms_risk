"""Figure 4B/4C of the preprint, redrawn from a refitted trace.

Deliberately close to the published panel: memory noise (first option only) above
perceptual noise (both options), linear payoff axis 5-50, "sigma noise" on y,
green = vertex / red = parietal, grey band for the cTBS difference.

Input is `noisecurve_reparam.<label>.tsv` from `noise_curve_inference`, which puts
a family-1 fit (first / second option) into the family-2 coordinates the figure
uses: perceptual = nu_2, memory = nu_1 - nu_2.

    python -m tms_risk.behavior.scripts.plot_fig4bc_style --label flexible1nf
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, PARIETAL, DIFF = '#2ca02c', '#d62728', '0.35'
ROWS = [('memory', 'Memory noise (only option 1)'),
        ('perceptual', 'Perceptual noise (both options)')]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'legend.frameon': True,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def main(data_dir, label, out_stem, x_hi, y_hi):
    c = pd.read_csv(Path(data_dir) / f'noisecurve_reparam.{label}.tsv', sep='\t')

    fig = plt.figure(figsize=(5.2, 3.3))
    gs = fig.add_gridspec(2, 2, hspace=.55, wspace=.42,
                          left=.10, right=.985, top=.83, bottom=.13)
    axes = np.empty((2, 2), dtype=object)

    vis = c[c.payoff <= x_hi]

    def ylim(sel, floor_at_zero):
        """Row limits from the data, so a negative memory curve is not clipped."""
        lo, hi = float(sel.lo.min()), float(sel.hi.max())
        lo = 0. if (floor_at_zero and lo >= 0) else lo - .1 * (hi - lo)
        return lo, hi + .1 * (hi - lo)

    for r, (term, title) in enumerate(ROWS):
        # ------------------------------------------------ B: noise per condition
        ax = fig.add_subplot(gs[r, 0]); axes[r, 0] = ax
        for cond, col, nm in [('vertex', VERTEX, 'Vertex'), ('ips', PARIETAL, 'Parietal')]:
            s = vis[(vis.term == term) & (vis.stimulation == cond)].sort_values('payoff')
            ax.fill_between(s.payoff, s.lo, s.hi, color=col, alpha=.22, lw=0)
            ax.plot(s.payoff, s.nu, color=col, lw=1.2, label=nm)
        lo, hi = ylim(vis[(vis.term == term) & (vis.stimulation != 'ips - vertex')], True)
        if y_hi:                             # honour an explicit override
            lo, hi = 0., y_hi
        if lo < 0:
            ax.axhline(0, color='0.6', lw=.6, ls=':', zorder=1)
        ax.set_ylim(lo, hi)
        ax.set_yticks(np.round([lo, (lo + hi) / 2, hi], 1))
        ax.set_ylabel('σ noise')
        ax.set_title(title, fontsize=7.5, color='0.15', pad=3)
        if r == 1:
            leg = ax.legend(title='Stimulation condition', fontsize=6,
                            title_fontsize=6, loc='lower right', handlelength=1.2,
                            borderpad=.35, labelspacing=.25)
            leg.get_frame().set_linewidth(.5)
            leg.get_frame().set_edgecolor('0.6')

        # ------------------------------------------------------ C: cTBS contrast
        ax = fig.add_subplot(gs[r, 1]); axes[r, 1] = ax
        s = vis[(vis.term == term) & (vis.stimulation == 'ips - vertex')].sort_values('payoff')
        ax.axhline(0, color='0.3', lw=.7, ls='--', zorder=1)
        ax.fill_between(s.payoff, s.lo, s.hi, color=DIFF, alpha=.30, lw=0, zorder=2)
        ax.plot(s.payoff, s.nu, color='0.1', lw=1.1, zorder=3)
        d = max(.35, 1.1 * max(abs(s.lo.min()), abs(s.hi.max())))
        ax.set_ylim(-d, d)
        ax.set_yticks(np.round([-d / 2, 0., d / 2], 1))
        ax.set_ylabel('σ noise')
        ax.set_title(title, fontsize=7.5, color='0.15', pad=3)

    for ax in axes.ravel():
        ax.set_xlim(5, x_hi)
        ax.set_xticks([5, 15, 25, 35, 45] if x_hi <= 55 else [5, 25, 50, 75, 100])
    for ax in axes[0]:
        ax.set_xticklabels([])
    for ax in axes[1]:
        ax.set_xlabel('Payoff magnitude')

    fig.text(.30, .955, 'Noise as a function of\nmagnitude', ha='center', va='center',
             fontsize=8, fontweight='bold', linespacing=1.25)
    fig.text(.78, .955, 'Effect of cTBS\non noise', ha='center', va='center',
             fontsize=8, fontweight='bold', linespacing=1.25)
    fig.text(.012, .965, 'B', fontsize=12, fontweight='bold', va='center')
    fig.text(.545, .965, 'C', fontsize=12, fontweight='bold', va='center')

    sns.despine(fig=fig, offset=2)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible1nf')
    parser.add_argument('--x_hi', default=50., type=float)
    parser.add_argument('--y_hi', default=0., type=float,
                        help='force the noise-level y-max; 0 = from the data')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    main(args.data_dir, args.label, args.out or
         f'/Users/gdehol/git/tms_risk/notes/figures/fig4bc_style.{args.label}',
         args.x_hi, args.y_hi)
