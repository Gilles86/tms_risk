"""Inferred noisiness of the magnitude representation (manuscript Fig 4B/4C style).

Rebuilt from `flexible1`, which is bug-free and converged. In that parameterisation
the two fitted functions are the noise on the *first-* and *second-presented* option,
so the mapping to risky/safe depends on presentation order:

    risky first   -> nu1 is the risky option, nu2 the safe one
    risky second  -> nu1 is the safe option,  nu2 the risky one

Reads pmcpars_curves.<label>.tsv from extract_pmc_parameters.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
LABEL = {'n1_evidence_sd': 'First-presented option',
         'n2_evidence_sd': 'Second-presented option',
         'memory_noise_sd': 'Memory noise', 'perceptual_noise_sd': 'Perceptual noise'}

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([10, 20, 40, 80])
    ax.set_xlim(7, 112)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax.set_xlabel('Payoff magnitude (CHF)')


def main(data_dir, label, out_stem):
    c = pd.read_csv(Path(data_dir) / f'pmcpars_curves.{label}.tsv', sep='\t')
    terms = sorted(c.term.unique())
    ct = pd.read_csv(Path(data_dir) / f'pmcpars_contrast.{label}.tsv', sep='\t')

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 5.0), sharex=True,
                             constrained_layout=True)
    for col, term in enumerate(terms):
        # --- top row: the fitted noise function per stimulation condition
        ax = axes[0, col]
        for cond, colr, nm in [('vertex', VERTEX, 'Vertex'), ('ips', IPS, 'IPS')]:
            s = c[(c.term == term) & (c.stimulation == cond)]
            ax.fill_between(s.payoff, s.lo, s.hi, color=colr, alpha=.18, lw=0)
            ax.plot(s.payoff, s.nu, color=colr, lw=1.5)
        ax.set_title(LABEL.get(term, term), fontsize=9, color='0.15')
        if col == 0:
            ax.set_ylabel('Representational noise ν (CHF)')
            ax.text(-.20, 1.06, 'a', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right')
        ax.set_ylim(0, 6.2)
        logx(ax); ax.set_xlabel('')

        # --- bottom row: the cTBS contrast, with the payoff range shaded
        ax = axes[1, col]
        s = c[(c.term == term) & (c.stimulation == 'ips - vertex')]
        ax.axvspan(7, 56, color='0.93', zorder=0, lw=0)
        ax.axhline(0, color='0.75', lw=.6, ls='--', zorder=1)
        ax.fill_between(s.payoff, s.lo, s.hi, color='#1f6fb4', alpha=.20, lw=0, zorder=2)
        ax.plot(s.payoff, s.nu, color='#1f6fb4', lw=1.6, zorder=3)
        # mark the payoffs where the 95% CrI excludes zero
        cc = ct[ct.term == term]
        sig = cc[(cc.lo > 0) | (cc.hi < 0)]
        if len(sig):
            ax.scatter(sig.payoff, np.full(len(sig), 0.92), marker='*', s=34,
                       color='#1f6fb4', zorder=5, clip_on=False)
        ax.set_ylim(-1.0, 1.0)
        if col == 0:
            ax.set_ylabel('Δ ν, IPS − vertex (CHF)')
            ax.text(-.20, 1.06, 'b', transform=ax.transAxes, fontsize=11,
                    fontweight='bold', va='bottom', ha='right')
        logx(ax)

    axes[0, 0].text(8, 5.7, 'Vertex', color=VERTEX, fontsize=8)
    axes[0, 0].text(8, 5.0, 'IPS', color=IPS, fontsize=8)
    axes[1, 0].text(20, -0.90, '92% of option presentations', fontsize=6,
                    color='0.45', ha='center')
    axes[1, 1].text(0.97, 0.06, 'Stars: 95% CrI excludes 0', transform=axes[1, 1].transAxes,
                    fontsize=6.5, color='#1f6fb4', ha='right')
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible1')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/noise_functions')
    args = parser.parse_args()
    main(args.data_dir, args.label, args.out)
