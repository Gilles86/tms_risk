"""Illustration: how a magnitude-local noise increase becomes risk-seeking choice.

Everything plotted is measured from the fitted `flexible2` posterior, not schematic:
percepts come from pmc_percepts.<label>.tsv, priors from pmcpars_priors.<label>.tsv.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
RISKY, SAFE = '#7b4f9d', '#2b7a8c'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def panel_letter(ax, letter, x=-0.20, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='right')


def main(data_dir, label, out_stem):
    dd = Path(data_dir)
    pc = pd.read_csv(dd / f'pmc_percepts.{label}.tsv', sep='\t')
    pr = pd.read_csv(dd / f'pmcpars_priors.{label}.tsv', sep='\t')
    grp = pr[pr.level == 'group'].set_index('parameter')['mean']
    curves = pd.read_csv(dd / f'pmcpars_curves.{label}.tsv', sep='\t')

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(7.25, 2.65),
                                           constrained_layout=True)

    # ------------------------------------------------------- a  noise increase
    c = curves[(curves.term == 'perceptual_noise_sd') &
               (curves.stimulation == 'ips - vertex')]
    ax_a.fill_between(c.payoff, c.lo, c.hi, color='#1f6fb4', alpha=.18, lw=0)
    ax_a.plot(c.payoff, c.nu, color='#1f6fb4', lw=1.5)
    ax_a.axhline(0, color='0.75', lw=0.6, ls='--', zorder=0)
    ax_a.set_xscale('log'); ax_a.set_xlim(7, 112)
    ax_a.minorticks_off()
    ax_a.set_xticks([10, 20, 40, 80])
    ax_a.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_a.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax_a.set_xlabel('Payoff (CHF)')
    ax_a.set_ylabel('Δ noise, IPS − vertex (CHF)')
    ax_a.set_title('cTBS adds noise\nto small payoffs', fontsize=8.5, color='0.25')
    panel_letter(ax_a, 'a')

    # ---------------------------------------------- b  percept transfer function
    lim = (5, 38)
    ax_b.plot(lim, lim, color='0.75', lw=0.7, ls='--', zorder=0)
    for opt, col, lab in [('safe', SAFE, 'Safe'), ('risky', RISKY, 'Risky')]:
        s = pc[pc.option == opt].sort_values('objective_ev')
        ax_b.plot(s.objective_ev, s.vertex, 'o-', color=col, ms=4.5, lw=1.4, zorder=3)
    for p, col, nm in [(grp['safe_prior_mu'], SAFE, 'μ safe'),
                       (grp['risky_prior_mu'], RISKY, 'μ risky')]:
        ax_b.axhline(p, color=col, lw=0.7, ls=':', zorder=1)
        ax_b.text(lim[1], p, f' {nm}', color=col, fontsize=6.5, va='center')
    ax_b.set_xlim(*lim); ax_b.set_ylim(6, 24)
    ax_b.set_xlabel('Objective expected value (CHF)')
    ax_b.set_ylabel('Perceived expected value (CHF)')
    ax_b.set_title('Percepts regress toward priors\nthat sit below most payoffs',
                   fontsize=8.5, color='0.25')
    ax_b.text(6.3, 22.6, 'Safe', color=SAFE, fontsize=8)
    ax_b.text(6.3, 21.2, 'Risky', color=RISKY, fontsize=8)
    # where the safe curve crosses identity == the fitted safe prior mean: below it
    # payoffs are overestimated, above it underestimated
    sm = grp['safe_prior_mu']
    ax_b.plot([sm], [sm], marker='o', ms=9, mfc='none', mec='0.2', mew=1.1, zorder=5)
    ax_b.annotate('Crossover at μ safe:\nsmaller payoffs over-,\nlarger under-estimated',
                  xy=(sm + 0.4, sm - 0.3), xytext=(15.5, 8.2), fontsize=6.8,
                  color='0.3', ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.25',
                                  color='0.4', lw=0.6))
    ax_b.annotate('Identity', xy=(21, 21), xytext=(24, 15.5), fontsize=6.5,
                  color='0.55', ha='left',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.2',
                                  color='0.6', lw=0.5))
    panel_letter(ax_b, 'b')

    # ------------------------------------------ c  net shift, per safe payoff
    w = pc.pivot(index='n_safe', columns='option', values='delta')
    lo = pc.pivot(index='n_safe', columns='option', values='lo')
    hi = pc.pivot(index='n_safe', columns='option', values='hi')
    xs = np.arange(len(w))
    ax_c.axhline(0, color='0.75', lw=0.6, ls='--', zorder=0)
    for i, (opt, col) in enumerate([('safe', SAFE), ('risky', RISKY)]):
        off = (i - .5) * .22
        ax_c.errorbar(xs + off, w[opt], yerr=[w[opt] - lo[opt], hi[opt] - w[opt]],
                      fmt='o', color=col, ms=4.5, lw=0, elinewidth=1.1,
                      capsize=0, zorder=3)
    ax_c.set_xticks(xs); ax_c.set_xticklabels([f'{v:.0f}' for v in w.index])
    ax_c.set_xlabel('Safe payoff (CHF)')
    ax_c.set_ylabel('Δ perceived EV, IPS − vertex (CHF)')
    ax_c.set_title('Safe pulled down, risky up:\nboth favour the risky option',
                   fontsize=8.5, color='0.25')
    ax_c.text(-0.35, ax_c.get_ylim()[1] * .92, 'Risky', color=RISKY, fontsize=8)
    ax_c.text(-0.35, ax_c.get_ylim()[0] * .80, 'Safe', color=SAFE, fontsize=8)
    panel_letter(ax_c, 'c')

    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible2')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/pmc_mechanism')
    args = parser.parse_args()
    main(args.data_dir, args.label, args.out)
