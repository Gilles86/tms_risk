"""How big is the cTBS shift the PRIOR-SHIFT model needs, and is it plausible?

The `_prior` variant is the main alternative to the paper's account: cTBS moved the
observer's beliefs rather than adding representational noise. It fits 21.5 nats worse
than the perceptual-noise model. These panels show *why*.

    a  Where the fitted priors sit, against the payoffs they are supposed to describe.
       The risky prior runs off to ~13 000 CHF with a credible interval spanning three
       orders of magnitude -- it is barely identified.
    b  The cTBS shift each prior mean would have to carry. Neither is credible.
    c  The decisive panel. Both accounts have to move the same thing -- the percept.
       A prior shift moves it by (1 - w) * d(mu_prior), and (1 - w) grows steeply with
       payoff, so the prior account predicts a percept shift that GROWS with magnitude
       (-0.5 CHF at 7, -24 CHF at 112). The noise account predicts a roughly constant
       shift of a few tenths of a CHF. The data prefer the latter.

    python -m tms_risk.behavior.scripts.plot_prior_shift

Reads notes/data/prior_shift.priorshift.tsv, prior_percept_shift.priorshift.tsv
(written by extract_prior_shift.py on the node holding the trace) and
pmc_percepts_by_order.flexible2nf.tsv for the noise model's comparison.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PRIOR_C, NOISE_C = '#8172B2', '#3B5BA5'      # prior account, noise account
RISKY, SAFE = '#b2182b', '#4d4d4d'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def main(data_dir, out_stem):
    dd = Path(data_dir)
    P = pd.read_csv(dd / 'prior_shift.priorshift.tsv', sep='\t')
    PC = pd.read_csv(dd / 'prior_percept_shift.priorshift.tsv', sep='\t')
    NP = pd.read_csv(dd / 'pmc_percepts_by_order.flexible2nf.tsv', sep='\t')

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), constrained_layout=True)

    # ------------------------------------------------- a: where the priors sit
    ax = axes[0]
    ax.axhspan(7, 112, color='0.85', zorder=0)
    ax.text(1.5, 112 * 1.25, 'Presented payoffs', fontsize=7, color='0.35', va='bottom')
    xs, labs = [], []
    for i, (opt, col) in enumerate([('safe', SAFE), ('risky', RISKY)]):
        for j, cond in enumerate(['vertex', 'ips']):
            r = P[(P.quantity == f'{opt}_prior_mu') & (P.condition == cond)].iloc[0]
            x = i * 2 + j
            ax.plot([x, x], [r.chf_lo, r.chf_hi], color=col, lw=1.4,
                    alpha=.45, solid_capstyle='butt')
            ax.plot(x, r.chf_mean, 'o', ms=6, color=col,
                    markerfacecolor=col if cond == 'ips' else 'white',
                    markeredgewidth=1.4)
            xs.append(x); labs.append(f'{opt}\n{cond}')
    ax.set_yscale('log'); ax.set_xticks(xs); ax.set_xticklabels(labs, fontsize=7)
    ax.set_ylabel('Prior mean (CHF)')
    ax.set_xlim(-0.7, 3.7)
    ax.annotate('Risky prior runs off\nto ~13 000 CHF', xy=(3, 13400),
                xytext=(1.1, 4000), fontsize=7, color='0.3', ha='left',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.25',
                                color='0.45', lw=0.6))

    # ------------------------------------------------------- b: the cTBS shift
    ax = axes[1]
    ax.axvline(0, color='.75', lw=.6, ls=':', zorder=0)
    for i, (opt, col) in enumerate([('safe', SAFE), ('risky', RISKY)]):
        r = P[(P.quantity == f'{opt}_prior_mu') & (P.condition == 'ips - vertex')].iloc[0]
        ax.plot([r.log_lo, r.log_hi], [i, i], color=col, lw=1.6, solid_capstyle='butt')
        ax.plot(r.log_mean, i, 'o', ms=6, color=col)
        ax.text(r.log_hi + .06, i, f'P(shift < 0) = {r.p_lt0:.2f}', fontsize=7,
                color='0.35', va='center')
    ax.set_yticks([0, 1]); ax.set_yticklabels(['Safe prior', 'Risky prior'])
    ax.set_xlabel('cTBS shift in prior mean (log CHF)')
    ax.set_ylim(-.6, 1.6); ax.set_xlim(-1.35, 1.35)

    # ------------------------------- c: implied percept shift, prior vs noise
    ax = axes[2]
    ax.axhline(0, color='.75', lw=.6, ls=':', zorder=0)
    g = PC[(PC.option == 'safe') & (PC.position == 'n2 (second-presented)')]
    ax.fill_between(g.payoff, g.d_percept_chf_lo, g.d_percept_chf_hi,
                    color=PRIOR_C, alpha=.20, lw=0)
    ax.plot(g.payoff, g.d_percept_chf, color=PRIOR_C, lw=1.6,
            label='Prior-shift model')
    n = NP[NP.option == 'safe']
    ax.errorbar(n.objective_ev, n.delta, yerr=[n.delta - n.lo, n.hi - n.delta],
                fmt='o', ms=3.4, color=NOISE_C, lw=0, elinewidth=.8, capsize=0,
                label='Noise model (fitted)', zorder=4)
    ax.set_xlabel('Safe payoff (CHF)')
    ax.set_ylabel('Percept shift, IPS − vertex (CHF)')
    ax.legend(loc='lower left', fontsize=7)
    ax.annotate('Prior account needs a shift\nthat grows with payoff',
                xy=(90, g[g.payoff > 85].d_percept_chf.mean()), xytext=(20, -60),
                fontsize=7, color='0.3', ha='left',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.25',
                                color='0.45', lw=0.6))

    for a, letter in zip(axes, 'abc'):
        a.text(-0.16, 1.04, letter, transform=a.transAxes, fontsize=12,
               fontweight='bold', va='bottom', ha='right')
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--out', default='notes/figures/prior_shift')
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.data_dir, a.out)
