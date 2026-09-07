"""Prototype: Fig 4B replotted in NATURAL space, with Weber and Weber+floor references.

Motivation (2026-08-19): Weber's law strictly means proportionality (a line through the
origin), so the log-log framing of Fig 4B ("slope 0.48 < 1") conflates two different
departures: a compressive power law, and an additive noise floor on top of Weber
scaling. Against the fitted flexible2nf curve (posterior mean): power law
nu = 0.57*x^0.48 fits almost exactly (rms 0.05 CHF); affine "Weber + floor"
nu = 1.70 + 0.036*x misses the mean curve (rms 0.17) but stays largely inside the 95%
CrI; pure Weber through the origin fails (rms 0.86). Natural axes show all of this
directly; log-log axes hide the offset question.

    python -m tms_risk.behavior.scripts.plot_fig4b_natural
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IPS, VERTEX = '#d62728', '#2ca02c'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main():
    d = pd.read_csv('notes/data/noisecurve_reparam.flexible2nf.tsv', sep='\t')
    fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
    for cond, col, lab in [('vertex', VERTEX, 'Vertex'), ('ips', IPS, 'IPS')]:
        g = d[(d.term == 'perceptual') & (d.stimulation == cond)].sort_values('payoff')
        ax.plot(g.payoff, g.nu, color=col)
        ax.fill_between(g.payoff, g.lo, g.hi, color=col, alpha=.18, lw=0)
        ax.text(113, g.nu.iloc[-1] + (0.28 if cond == 'ips' else -0.28), lab,
                color=col, fontsize=8.5, va='center')

    # references fitted to the vertex posterior-mean curve (CrI-precision weighted)
    x = np.linspace(0, 112, 200)
    ax.plot(x, 0.062 * x, ls=':', color='0.55', lw=1.0)
    ax.text(58, 0.062 * 58 + 0.25, 'Weber (k·payoff)', color='0.45', fontsize=7.5,
            rotation=38, rotation_mode='anchor')
    ax.plot(x, 1.699 + 0.036 * x, ls='--', color='0.25', lw=1.0)
    ax.text(80, 1.699 + 0.036 * 80 - 0.5, 'Weber + floor', color='0.25', fontsize=7.5,
            rotation=15, rotation_mode='anchor')

    ax.set_xlim(0, 126)
    ax.set_ylim(0, 7)
    ax.set_xticks([0, 7, 28, 56, 112])
    ax.set_yticks([0, 2, 4, 6])
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Perceptual noise ν (CHF)')
    sns.despine(fig=fig, offset=3, trim=True)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'notes/figures/prototypes/fig4b_natural.{ext}',
                    bbox_inches='tight', pad_inches=0.02)
    print('wrote notes/figures/prototypes/fig4b_natural.pdf')


if __name__ == '__main__':
    main()
