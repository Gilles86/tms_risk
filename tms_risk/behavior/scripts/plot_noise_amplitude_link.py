"""The cTBS noise increase is localised at low payoffs *per subject*, and how
localised it is tracks how much nPRF amplitude that subject's stimulated voxels lost.

A) Scatter: per-subject cTBS amplitude change vs the localisation slope
   Δν_perceptual(7 CHF) − Δν_perceptual(28 CHF).
B) The same data as curves: Δν_perceptual(payoff) averaged within a median split on
   amplitude change. Subjects who lost amplitude show the low-payoff-weighted
   increase the group mean washes out.

    python -m tms_risk.behavior.scripts.plot_noise_amplitude_link --label flexible2nf
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

LOSS, KEEP = '#b2182b', '#4d4d4d'      # lost amplitude / kept it

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'mathtext.fontset': 'stixsans',
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def main(data_dir, label, out_stem):
    data_dir = Path(data_dir)
    d = pd.read_csv(data_dir / f'subject_noise_shift.{label}.tsv', sep='\t')
    d = d[d.term == 'perceptual']
    amp = pd.read_csv(data_dir / 'm2_tms_param_shifts.tsv', sep='\t',
                      header=[0, 1], index_col=0)[('amplitude', 'diff')]
    amp.name = 'damp'

    slope = d[d.payoff == -1].set_index('subject').d_nu.rename('slope')
    j = pd.concat([slope, amp], axis=1).dropna()
    r, p = stats.pearsonr(j.damp, j.slope)
    rho, pp = stats.spearmanr(j.damp, j.slope)

    fig = plt.figure(figsize=(6.4, 2.9))
    gs = fig.add_gridspec(1, 2, wspace=.42, left=.10, right=.985, top=.83, bottom=.19)

    # ------------------------------------------------------------------ A: scatter
    ax = fig.add_subplot(gs[0, 0])
    ax.axhline(0, color='.75', lw=.6, ls=':', zorder=1)
    ax.axvline(0, color='.75', lw=.6, ls=':', zorder=1)
    lost = j.damp < j.damp.median()
    ax.scatter(j.damp[lost], j.slope[lost], s=17, c=LOSS, lw=0, alpha=.85, zorder=3)
    ax.scatter(j.damp[~lost], j.slope[~lost], s=17, c=KEEP, lw=0, alpha=.85, zorder=3)
    xs = np.linspace(j.damp.min(), j.damp.max(), 50)
    b, a = np.polyfit(j.damp, j.slope, 1)
    ax.plot(xs, a + b * xs, color='0.15', lw=1.2, zorder=4)
    # bootstrap band on the fit
    boot = np.array([np.polyval(np.polyfit(*j.sample(len(j), replace=True,
                                                     random_state=k)[['damp', 'slope']]
                                           .values.T, 1), xs) for k in range(2000)])
    ax.fill_between(xs, *np.quantile(boot, [.025, .975], axis=0),
                    color='0.5', alpha=.18, lw=0, zorder=2)
    ax.set_xlabel('Δ nPRF amplitude (IPS − vertex)')
    ax.set_ylabel('Localisation of the noise increase\nΔν(7) − Δν(28), CHF')
    ax.set_title('More amplitude loss,\nmore low-payoff-specific noise',
                 fontsize=7.5, color='0.15', pad=3, linespacing=1.25)
    ax.text(.03, .04, f'r = {r:+.2f}, p = {p:.3f}\nρ = {rho:+.2f}, p = {pp:.3f}',
            transform=ax.transAxes, fontsize=6.2, va='bottom', ha='left',
            linespacing=1.35)

    # ------------------------------------------------------------- B: split curves
    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color='.3', lw=.7, ls='--', zorder=1)
    curves = d[d.payoff > 0].pivot_table(index='subject', columns='payoff', values='d_nu')
    curves = curves.loc[j.index]
    for mask, col, nm in [(lost, LOSS, 'Lost amplitude'),
                          (~lost, KEEP, 'Kept amplitude')]:
        g = curves[mask.values]
        m, se = g.mean(0), g.std(0) / np.sqrt(len(g))
        ax.fill_between(g.columns, m - se, m + se, color=col, alpha=.20, lw=0)
        ax.plot(g.columns, m, color=col, lw=1.4, marker='o', ms=3, label=nm)
    ax.set_xlabel('Payoff magnitude')
    ax.set_ylabel('Δν perceptual (IPS − vertex), CHF')
    ax.set_xticks([7, 10, 14, 20, 28])
    ax.set_title('The group mean averages\nthe localisation away',
                 fontsize=7.5, color='0.15', pad=3, linespacing=1.25)
    leg = ax.legend(title='Median split on Δ amplitude', fontsize=6,
                    title_fontsize=6, loc='upper right', handlelength=1.3,
                    borderpad=.35, labelspacing=.25)
    leg.get_frame().set_linewidth(.5)
    leg.get_frame().set_edgecolor('0.6')

    fig.text(.012, .955, 'A', fontsize=12, fontweight='bold', va='center')
    fig.text(.535, .955, 'B', fontsize=12, fontweight='bold', va='center')
    sns.despine(fig=fig, offset=2)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    print(f'wrote {out_stem}.pdf   (n = {len(j)}, r = {r:+.3f}, p = {p:.4f})')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    main(args.data_dir, args.label, args.out or
         f'/Users/gdehol/git/tms_risk/notes/figures/noise_amplitude_link.{args.label}')
