"""Supplementary figure: all group-level parameters of the published Flexible PMC.

Reads the TSVs written by tms_risk.behavior.scripts.extract_pmc_parameters.
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
    'mathtext.fontset': 'stixsans',
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

PRIOR_LABEL = {'safe_prior_mu': 'μ safe', 'safe_prior_sd': 'σ safe',
               'risky_prior_mu': 'μ risky', 'risky_prior_sd': 'σ risky'}
TERM_LABEL = {'perceptual_noise_sd': 'Perceptual', 'memory_noise_sd': 'Memory',
              'n1_evidence_sd': 'First option', 'n2_evidence_sd': 'Second option'}


def panel_letter(ax, letter, x=-0.16, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='right')


def main(data_dir, label, out_stem):
    dd = Path(data_dir)
    priors = pd.read_csv(dd / f'pmcpars_priors.{label}.tsv', sep='\t')
    splines = pd.read_csv(dd / f'pmcpars_splines.{label}.tsv', sep='\t')
    curves = pd.read_csv(dd / f'pmcpars_curves.{label}.tsv', sep='\t')

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.6), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # ------------------------------------------------------------------ a priors
    order = ['μ safe', 'σ safe', 'μ risky', 'σ risky']
    pr = priors.copy()
    pr['name'] = pr['parameter'].map(PRIOR_LABEL)
    grp = pr[pr.level == 'group'].set_index('name').loc[order]
    subj = pr[pr.level != 'group']
    rng = np.random.default_rng(0)
    for i, nm in enumerate(order):
        col = SAFE if 'safe' in nm else RISKY
        s = subj[subj.name == nm]['mean'].values
        ax_a.scatter(i + rng.uniform(-.16, .16, len(s)), s, s=7, color=col,
                     alpha=.35, lw=0, zorder=2)
        r = grp.loc[nm]
        ax_a.errorbar(i, r['mean'], yerr=[[r['mean'] - r.lo], [r.hi - r['mean']]],
                      fmt='D', color=col, ms=7, mec='0.15', mew=1.4,
                      elinewidth=1.6, capsize=0, zorder=4)
    ax_a.axhline(15.82, color=SAFE, lw=0.7, ls=':', zorder=1)
    ax_a.axhline(36.15, color=RISKY, lw=0.7, ls=':', zorder=1)
    ax_a.text(3.45, 15.82, 'Mean safe payoff', fontsize=6.5, color=SAFE,
              ha='right', va='bottom')
    ax_a.text(3.45, 36.15, 'Mean risky payoff', fontsize=6.5, color=RISKY,
              ha='right', va='bottom')
    ax_a.set_xticks(range(4)); ax_a.set_xticklabels(order)
    ax_a.set_xlim(-.5, 3.5)
    ax_a.set_ylabel('CHF')
    ax_a.annotate('Both priors sit below\nthe payoffs they describe',
                  xy=(2, 21.5), xytext=(0.55, 30), fontsize=7.5, color='0.3',
                  ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.25',
                                  color='0.4', lw=0.6))
    panel_letter(ax_a, 'a')

    # ------------------------------------------------------- b fitted noise curves
    terms = sorted(curves.term.unique())          # n1/n2 or memory/perceptual
    for term, ls in [(terms[1], '-'), (terms[0], '--')]:
        for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
            c = curves[(curves.term == term) & (curves.stimulation == cond)]
            ax_b.plot(c.payoff, c.nu, color=col, ls=ls, lw=1.3)
            if ls == '-':
                ax_b.fill_between(c.payoff, c.lo, c.hi, color=col, alpha=.13, lw=0)
    ax_b.set_xscale('log')
    ax_b.set_xticks([10, 20, 40, 80]); ax_b.set_xlim(7, 112)
    ax_b.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_b.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax_b.set_xlabel('Payoff (CHF)')
    ax_b.set_ylabel('Representational noise ν (CHF)')
    ax_b.text(7.6, ax_b.get_ylim()[1] * .97,
              f'Solid: {TERM_LABEL[terms[1]].lower()}   Dashed: {TERM_LABEL[terms[0]].lower()}',
              fontsize=7, color='0.35', va='top')
    ax_b.text(95, ax_b.get_ylim()[1] * .55, 'IPS', color=IPS, fontsize=8, ha='right')
    ax_b.text(95, ax_b.get_ylim()[1] * .45, 'Vertex', color=VERTEX, fontsize=8, ha='right')
    panel_letter(ax_b, 'b')

    # -------------------------------------------------------- c cTBS noise increase
    for term, col, ls in [(terms[1], '#1f6fb4', '-'), (terms[0], '0.45', '--')]:
        c = curves[(curves.term == term) & (curves.stimulation == 'ips - vertex')]
        ax_c.fill_between(c.payoff, c.lo, c.hi, color=col, alpha=.18, lw=0)
        ax_c.plot(c.payoff, c.nu, color=col, ls=ls, lw=1.4)
    ax_c.axhline(0, color='0.75', lw=0.6, ls='--', zorder=0)
    ax_c.axvspan(7, 28, color='0.9', zorder=0, lw=0)
    ax_c.set_xscale('log')
    ax_c.set_xticks([10, 20, 40, 80]); ax_c.set_xlim(7, 112)
    ax_c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_c.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
    ax_c.set_xlabel('Payoff (CHF)')
    ax_c.set_ylabel('Δ ν, IPS − vertex (CHF)')
    ax_c.text(7.6, ax_c.get_ylim()[1] * .95, TERM_LABEL[terms[1]], color='#1f6fb4',
              fontsize=8, va='top')
    ax_c.text(7.6, ax_c.get_ylim()[1] * .78, TERM_LABEL[terms[0]], color='0.45',
              fontsize=8, va='top')
    ax_c.text(13.5, ax_c.get_ylim()[0] * .78, 'Presented\npayoff range',
              fontsize=6.5, color='0.4', ha='center', va='bottom')
    panel_letter(ax_c, 'c')

    # ------------------------------------------------- d all spline coefficients
    sp = splines.copy()
    sp['is_stim'] = sp.regressor != 'Intercept'
    sp = sp.sort_values(['is_stim', 'term', 'spline'], ascending=[True, True, False])
    sp = sp.reset_index(drop=True)
    for i, r in sp.iterrows():
        cred = (r.p_gt0 > .975) or (r.p_gt0 < .025)
        col = ('#1f6fb4' if r.term == terms[1] else '0.45')
        ax_d.plot([r.lo, r.hi], [i, i], color=col, lw=1.1,
                  alpha=1.0 if cred else 0.45, zorder=2)
        ax_d.scatter(r['mean'], i, s=18 if cred else 10, color=col,
                     edgecolor='0.15' if cred else 'none',
                     linewidth=0.8 if cred else 0, zorder=3)
    ax_d.axvline(0, color='0.75', lw=0.6, ls='--', zorder=0)
    n_int = (~sp.is_stim).sum()
    ax_d.axhline(n_int - 0.5, color='0.8', lw=0.6, zorder=1)
    ax_d.set_yticks([(n_int - 1) / 2, n_int + (len(sp) - n_int - 1) / 2])
    ax_d.set_yticklabels(['Intercept\n(IPS)', 'Vertex −\nIPS'], fontsize=7.5)
    ax_d.set_ylim(-1, len(sp))
    ax_d.set_xlabel('Spline coefficient (a.u.)')
    ax_d.text(0.98, 0.03, 'Filled: 95% CrI excludes 0', transform=ax_d.transAxes,
              fontsize=6.5, color='0.4', ha='right')
    panel_letter(ax_d, 'd')

    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible2')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/pmc_parameters')
    args = parser.parse_args()
    main(args.data_dir, args.label, args.out)
