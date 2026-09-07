"""Figure: the cTBS effect is a *local* increase in randomness, not a preference shift.

Reads the TSVs written by tms_risk.behavior.scripts.analyze_localized_noise.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tms_risk.utils.data import get_all_behavior

VERTEX, IPS = '#2ca02c', '#d62728'      # canonical palette: red = stimulated site
NRISKY_LABELS = ['7-17', '18-26', '27-36', '37-53', '54-112']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False, 'legend.handlelength': 1.4,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def panel_letter(ax, letter, x=-0.19, y=1.04):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='right')


def load(data_dir, bids_folder):
    dd = Path(data_dir)
    ratio = pd.read_csv(dd / 'localnoise_delta_by_ratio.tsv', sep='\t')
    nrisky = pd.read_csv(dd / 'localnoise_delta_by_nrisky.tsv', sep='\t')
    sig = pd.read_csv(dd / 'localnoise_signatures.tsv', sep='\t')
    rec = pd.read_csv(dd / 'localnoise_recovery.tsv', sep='\t')
    gen = pd.read_csv(dd / 'localnoise_generator_profile.tsv', sep='\t')

    b = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True)
    b = b.drop('baseline', level='stimulation_condition').reset_index()
    b['bin'] = b['bin(risky/safe)'].astype(str)
    b['order'] = b['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    b['stim'] = b['stimulation_condition']
    b['n_risky_bin'] = pd.cut(b['n_risky'], [0, 17, 26, 36, 53, np.inf], labels=NRISKY_LABELS)

    ratio_x = b.groupby(['order', 'bin'])['frac'].mean().rename('ratio')
    ratio = ratio.join(ratio_x, on=['order', 'bin'])

    curves = (b.groupby(['subject', 'order', 'bin', 'stim'])['chose_risky'].mean()
                .groupby(['order', 'bin', 'stim']).agg(['mean', 'sem']).reset_index()
                .merge(ratio_x.reset_index(), on=['order', 'bin']))

    payoff_x = (b.groupby('n_risky_bin', observed=True)['n_risky']
                  .apply(lambda s: np.exp(np.log(s).mean())).rename('payoff'))
    nrisky = nrisky.join(payoff_x, on='n_risky_bin')
    gen = gen.merge(ratio_x.xs('Risky second').reset_index(), on='bin')
    return ratio, nrisky, sig, rec, curves, gen


def main(data_dir, bids_folder, out_stem):
    ratio, nrisky, sig, rec, curves, gen = load(data_dir, bids_folder)

    fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.4), constrained_layout=True)
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    r2 = ratio[ratio.order == 'Risky second'].sort_values('ratio')
    r1 = ratio[ratio.order == 'Risky first'].sort_values('ratio')
    s2 = sig[sig.order == 'Risky second']

    # -------------------------------------------------------------------- a
    # Observed psychometric functions with the fitted probit, risky-second trials.
    c2 = curves[curves.order == 'Risky second']
    for stim, col, lab in [('vertex', VERTEX, 'Vertex'), ('ips', IPS, 'IPS')]:
        sub = c2[c2.stim == stim].sort_values('ratio')
        ax_a.plot(r2['ratio'], r2[f'probit_{stim}'], color=col, lw=1.1, alpha=0.55, zorder=2)
        ax_a.fill_between(r2['ratio'], r2[f'probit_{stim}_lo'], r2[f'probit_{stim}_hi'],
                          color=col, alpha=0.15, lw=0, zorder=1)
        ax_a.errorbar(sub['ratio'], sub['mean'], yerr=sub['sem'], fmt='o',
                      color=col, ms=4.5, lw=0, elinewidth=1.0, capsize=0, zorder=3)
    ax_a.axhline(.5, color='0.78', lw=0.6, ls='--', zorder=0)
    ax_a.set_xlabel('Risky/safe payoff ratio')
    ax_a.set_ylabel('P(chose risky)')
    ax_a.set_xticks([1.5, 2.0, 2.5, 3.0])
    ax_a.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax_a.set_ylim(0.16, 0.95)
    ax_a.text(1.47, 0.92, 'Risky option presented second · n = 35', fontsize=8, color='0.25')
    ax_a.text(3.22, 0.845, 'IPS', color=IPS, fontsize=8, ha='left', va='center')
    ax_a.text(3.22, 0.775, 'Vertex', color=VERTEX, fontsize=8, ha='left', va='center')
    ax_a.annotate('Only the left arm moves', xy=(1.82, 0.44), xytext=(2.05, 0.28),
                  fontsize=8, color='0.25', ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.3',
                                  color='0.35', lw=0.6))
    ax_a.set_xlim(1.35, 3.5)
    panel_letter(ax_a, 'a')

    # -------------------------------------------------------------------- b
    # The two things a probit *can* do, against what the data actually did.
    styles = {'Consistency only': dict(ls='--', color='0.2'),
              'Preference only': dict(ls=':', color='0.2'),
              'IPS (full probit)': dict(ls='-', color='0.68')}
    for key, st in styles.items():
        cur = s2[s2.curve == key]
        ax_b.plot(np.exp(cur['x']), cur['delta'], lw=1.1, zorder=1, **st)
    ax_b.plot(gen['ratio'], gen['delta_sim'], ls='-.', color='#1f6fb4', lw=1.2, zorder=2)
    ax_b.axhline(0, color='0.78', lw=0.6, ls='--', zorder=0)
    ax_b.errorbar(r1['ratio'], r1['delta'],
                  yerr=[r1['delta'] - r1['ci_lo'], r1['ci_hi'] - r1['delta']],
                  fmt='o', mfc='white', mec='0.6', ecolor='0.85', ms=3.2,
                  lw=0, elinewidth=0.7, capsize=0, zorder=2)
    ax_b.errorbar(r2['ratio'], r2['delta'],
                  yerr=[r2['delta'] - r2['ci_lo'], r2['ci_hi'] - r2['delta']],
                  fmt='o', color=IPS, ms=4.5, lw=0, elinewidth=1.0, capsize=0, zorder=4)
    ax_b.set_xlabel('Risky/safe payoff ratio')
    ax_b.set_ylabel('Δ P(chose risky), IPS − vertex')
    ax_b.set_xticks([1.5, 2.0, 2.5, 3.0])
    ax_b.set_xlim(1.35, 3.5)
    ax_b.set_ylim(-0.16, 0.40)
    ax_b.set_yticks([-0.1, 0.0, 0.1, 0.2])
    for i, (txt, st, col) in enumerate([
            ('Consistency change only', '--', '0.2'),
            ('Local noise only, indifference fixed', '-.', '#1f6fb4'),
            ('Preference change only', ':', '0.2'),
            ('Fitted probit (both knobs)', '-', '0.68')]):
        y = 0.385 - i * 0.032
        ax_b.plot([1.40, 1.53], [y, y], ls=st, color=col, lw=1.1)
        ax_b.text(1.57, y, txt, fontsize=7, color=col, va='center')
    ax_b.annotate('Pure flattening predicts a negative\nright arm; not observed',
                  xy=(2.90, -0.045), xytext=(1.62, -0.128),
                  fontsize=7.5, color='0.3', ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.25',
                                  color='0.4', lw=0.6))
    ax_b.text(3.47, 0.195, 'Open symbols:\nrisky option first', fontsize=7.5,
              color='0.62', ha='right', va='top')
    panel_letter(ax_b, 'b')

    # -------------------------------------------------------------------- c
    n2 = nrisky[nrisky.order == 'Risky second']
    ax_c.axhline(0, color='0.78', lw=0.6, ls='--', zorder=0)
    ax_c.axvspan(6, 10, color='0.88', zorder=0, lw=0)
    ax_c.fill_between(n2['payoff'], n2['probit_lo'], n2['probit_hi'],
                      color='0.6', alpha=0.30, lw=0, zorder=1)
    ax_c.plot(n2['payoff'], n2['probit'], color='0.5', lw=1.1, zorder=2)
    ax_c.errorbar(n2['payoff'], n2['delta'],
                  yerr=[n2['delta'] - n2['ci_lo'], n2['ci_hi'] - n2['delta']],
                  fmt='o', color=IPS, ms=4.5, lw=0, elinewidth=1.0, capsize=0, zorder=3)
    ax_c.set_xscale('log')
    ax_c.set_xticks([10, 20, 40, 80])
    ax_c.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax_c.set_xlim(5.2, 100)
    ax_c.set_xlabel('Risky payoff (CHF)')
    ax_c.set_ylabel('Δ P(chose risky), IPS − vertex')
    ax_c.set_ylim(-0.09, 0.33)
    ax_c.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax_c.text(5.6, 0.315, 'nPRF preferred\nnumerosities', fontsize=7, color='0.35',
              ha='left', va='top')
    ax_c.annotate('0.35 to 0.51:\nchoices drop to chance', xy=(13.5, 0.175), xytext=(19.5, 0.295),
                  fontsize=7.5, color='0.25', ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.25',
                                  color='0.35', lw=0.6))
    ax_c.annotate('Fitted probit', xy=(46, 0.049), xytext=(31, -0.058),
                  fontsize=7.5, color='0.45', ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.25',
                                  color='0.5', lw=0.6))
    panel_letter(ax_c, 'c')

    # -------------------------------------------------------------------- d
    rec['drnp'] = rec['rnp_ips'] - rec['rnp_vertex']
    real = rec[rec.source == 'Real data']
    simd = rec[rec.source != 'Real data']
    ax_d.axvline(0, color='0.4', lw=0.8, ls='--', zorder=0)
    for _, g in simd.groupby('sim'):
        sns.kdeplot(x=g['drnp'], ax=ax_d, color='0.35', lw=0.9, cut=0, alpha=0.85, zorder=2)
    sns.kdeplot(x=real['drnp'], ax=ax_d, color=IPS, fill=True, alpha=0.22,
                lw=1.5, cut=0, zorder=3)
    ax_d.set_xlabel('Δ risk-neutral probability, IPS − vertex')
    ax_d.set_ylabel('Posterior density')
    ax_d.set_yticks([])
    ymax = ax_d.get_ylim()[1]
    ax_d.set_ylim(0, ymax * 1.45)
    xlo = ax_d.get_xlim()[0]
    ax_d.text(simd['drnp'].mean(), ymax * 1.42, 'Simulations:\nlocal noise only,\nno preference change',
              color='0.3', fontsize=8, ha='center', va='top')
    ax_d.text(real['drnp'].mean(), ymax * 1.10, 'Real data', color=IPS,
              fontsize=8.5, ha='center', va='top')
    ax_d.annotate('Simulated truth', xy=(0, ymax * 0.18), xytext=(xlo * 0.97, ymax * 0.42),
                  fontsize=7.5, color='0.35', ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=-0.25',
                                  color='0.45', lw=0.6))
    panel_letter(ax_d, 'd')

    sns.despine(fig=fig, offset=4, trim=False)
    ax_d.spines['left'].set_visible(False)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/localized_noise')
    args = parser.parse_args()
    main(args.data_dir, args.bids_folder, args.out)
