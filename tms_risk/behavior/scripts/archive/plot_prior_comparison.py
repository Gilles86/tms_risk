"""Free versus pinned prior: does the compressive prior earn its keep?

The fitted priors sit far below the payoff range (safe mu ~ 3.8 CHF against payoffs of
7-112), which is hard to read as a belief about payoffs and looks instead like a
compressive value function. This compares that fit against one where the prior is pinned
to the objective payoff distribution.

    a   the fitted prior means against the range of payoffs actually presented
    b   the perceptual noise functions, both fits, on log-log axes
    c   the cTBS contrast in each fit
    d   posterior predictive checks, both fits, risky-second trials

The pinned fit samples far better (0 divergences, ESS 10676, versus 86 and 745) because
pinning removes a flat direction. It is the parameters that fail, not the sampler.

    python -m tms_risk.behavior.scripts.plot_prior_comparison
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
FREE, PINNED = '#3B5BA5', '#b8860b'
XT = [7, 14, 28, 56, 112]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')
PANEL = dict(fontsize=11, fontweight='bold', va='bottom', ha='right')

FITS = [('flexible2nf_perception', 'Fitted prior', FREE),
        ('objprior_perception', 'Prior pinned to payoffs', PINNED)]


def main(data_dir, out_stem):
    data = Path(data_dir)
    par = pd.read_csv(data / 'paradigm_payoffs.tsv', sep='\t')

    fig = plt.figure(figsize=(7.25, 4.6))
    gs = fig.add_gridspec(2, 2, hspace=.48, wspace=.34,
                          left=.095, right=.98, top=.90, bottom=.11)

    # --- a: where the priors sit relative to the payoffs presented
    ax = fig.add_subplot(gs[0, 0])
    for k, (opt, colr) in enumerate([('n_safe', '#4d4d4d'), ('n_risky', '#b2182b')]):
        v = par[opt].values
        ax.plot([np.percentile(v, 5), np.percentile(v, 95)], [k, k], color=colr,
                lw=5, alpha=.28, solid_capstyle='butt', zorder=1)
        ax.plot([np.median(v)], [k], '|', color=colr, ms=11, mew=1.6, zorder=2)
    for tag, lab, colr in FITS:
        p = pd.read_csv(data / f'pmcpars_priors.{tag}.tsv', sep='\t')
        g = p[p.level == 'group'].set_index('parameter')['mean']
        ax.plot([g.safe_prior_mu], [0], 'o', color=colr, ms=6, zorder=4)
        ax.plot([g.risky_prior_mu], [1], 'o', color=colr, ms=6, zorder=4)
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Safe', 'Risky'])
    ax.set_xscale('log'); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('CHF')
    ax.set_ylim(-.6, 1.6)
    ax.set_title('Prior mean vs the payoffs presented', fontsize=8, color='.2', pad=4)
    ax.text(.03, .93, 'Bars: 5th-95th percentile of payoffs', transform=ax.transAxes,
            fontsize=6.3, color='.4', va='top')

    # --- b: the perceptual noise functions
    ax = fig.add_subplot(gs[0, 1])
    for tag, lab, colr in FITS:
        c = pd.read_csv(data / f'pmcpars_curves.{tag}.tsv', sep='\t')
        v = c[(c.term == 'perceptual_noise_sd') & (c.stimulation == 'vertex')]
        ax.plot(v.payoff, v.nu, color=colr, lw=1.5)
    ax.plot(XT, XT, color='.75', lw=.7, ls=':', zorder=0)
    ax.text(60, 78, 'ν = payoff', fontsize=6.2, color='.5', rotation=34)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(XT); ax.set_yticks([1, 4, 16, 64])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)'); ax.set_ylabel('Perceptual noise ν (CHF)')
    ax.set_title('Vertex noise function', fontsize=8, color='.2', pad=4)

    # --- c: the cTBS contrast
    ax = fig.add_subplot(gs[1, 0])
    ax.axhline(0, color='.75', lw=.7, ls='--', zorder=0)
    for tag, lab, colr in FITS:
        c = pd.read_csv(data / f'pmcpars_curves.{tag}.tsv', sep='\t')
        d = c[(c.term == 'perceptual_noise_sd') & (c.stimulation == 'ips - vertex')]
        ax.plot(d.payoff, d.nu, color=colr, lw=1.5)
    ax.set_xscale('log'); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δ perceptual noise\nIPS − vertex (CHF)')
    ax.set_title('cTBS effect', fontsize=8, color='.2', pad=4)
    for tag, lab, colr in FITS:
        ax.plot([], [], color=colr, lw=1.5, label=lab)
    ax.legend(loc='lower left', fontsize=6.6, handlelength=1.6, borderpad=.3)

    # --- d: posterior predictive, risky-second
    ax = fig.add_subplot(gs[1, 1])
    ax.axhline(.5, color='.85', lw=.6, ls='--', zorder=0)
    for tag, lab, colr in FITS:
        p = pd.read_csv(data / f'ppc_fig3a.{tag}.tsv', sep='\t')
        p = p[p.order == 'Risky second']
        for stim, ls in [('vertex', '-'), ('ips', '--')]:
            g = p[p.stim == stim].sort_values('frac')
            ax.plot(g.frac, g['mean'], color=colr, ls=ls, lw=1.3, zorder=2)
    p = pd.read_csv(data / f'ppc_fig3a.{FITS[0][0]}.tsv', sep='\t')
    p = p[p.order == 'Risky second']
    for stim, mk in [('vertex', 'o'), ('ips', 's')]:
        g = p[p.stim == stim].sort_values('frac')
        ax.plot(g.frac, g.observed, mk, color='.2', ms=4, lw=0,
                mfc='white' if stim == 'ips' else '.2', mew=1, zorder=4)
    ax.set_xlabel('Risky/safe payoff ratio'); ax.set_ylabel('P(chose risky)')
    ax.set_title('Posterior predictive, risky second', fontsize=8, color='.2', pad=4)
    ax.text(.04, .95, 'Points: data\nSolid vertex, dashed IPS', transform=ax.transAxes,
            fontsize=6.3, color='.4', va='top', linespacing=1.25)

    for a, letter in zip(fig.axes, 'abcd'):
        a.text(-.17, 1.06, letter, transform=a.transAxes, **PANEL)
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf')
    for tag, lab, _ in FITS:
        c = pd.read_csv(data / f'pmcpars_curves.{tag}.tsv', sep='\t')
        v = c[(c.term == 'perceptual_noise_sd') & (c.stimulation == 'vertex')]
        print(f'  {lab:24s} ν at 112 CHF = {v.nu.iloc[-1]:6.2f} CHF '
              f'({100 * v.nu.iloc[-1] / 112:.0f}% of the payoff)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/prior_comparison')
    a = parser.parse_args()
    main(a.data_dir, a.out)
