"""Figure: inside the fitted Flexible PMC, the bias channel carries the cTBS effect.

Reads the TSVs written by tms_risk.behavior.scripts.decompose_pmc_channels
(model) and analyze_localized_noise (observed).
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tms_risk.utils.data import get_all_behavior

FULL, BIAS, NOISE = '0.15', '#c8781a', '#1f6fb4'
RATIO_BINS = ['20%', '32%', '44%', '56%', '68%', '80%']
NRISKY_LABELS = ['7-17', '18-26', '27-36', '37-53', '54-112']

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


def panel_letter(ax, letter, x=-0.17, y=1.05):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='right')


def main(data_dir, bids_folder, out_stem, label='flexible2'):
    dd = Path(data_dir)
    mod_r = pd.read_csv(dd / f'pmc_channels_by_ratio.{label}.tsv', sep='\t')
    mod_n = pd.read_csv(dd / f'pmc_channels_by_nrisky.{label}.tsv', sep='\t')
    obs_r = pd.read_csv(dd / 'localnoise_delta_by_ratio.tsv', sep='\t')
    obs_n = pd.read_csv(dd / 'localnoise_delta_by_nrisky.tsv', sep='\t')

    b = get_all_behavior(bids_folder=bids_folder, all_tms_conditions=True)
    b = b.drop('baseline', level='stimulation_condition').reset_index()
    b['bin'] = b['bin(risky/safe)'].astype(str)
    b['order'] = b['risky_first'].map({True: 'Risky first', False: 'Risky second'})
    b['n_risky_bin'] = pd.cut(b['n_risky'], [0, 17, 26, 36, 53, np.inf], labels=NRISKY_LABELS)
    ratio_x = b[~b.risky_first].groupby('bin')['frac'].mean().rename('ratio')
    payoff_x = (b.groupby('n_risky_bin', observed=True)['n_risky']
                  .apply(lambda s: np.exp(np.log(s).mean())).rename('payoff'))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7.25, 2.9), constrained_layout=True)

    def draw(ax, mod, obs, xmap, key, xlabel, logx=False):
        mod = mod[mod.order == 'Risky second'].join(xmap, on=key)
        obs = obs[obs.order == 'Risky second'].join(xmap, on=key)
        xc = xmap.name
        ax.axhline(0, color='0.78', lw=0.6, ls='--', zorder=0)
        ax.errorbar(obs[xc], obs['delta'],
                    yerr=[obs['delta'] - obs['ci_lo'], obs['ci_hi'] - obs['delta']],
                    fmt='o', color='0.15', ms=4.5, lw=0, elinewidth=1.0,
                    capsize=0, zorder=4)
        for name, col, lab in [('full', FULL, 'Full model'),
                               ('bias_only', BIAS, 'Bias channel only'),
                               ('noise_only', NOISE, 'Randomness channel only')]:
            m = mod[mod.channel == name].sort_values(xc)
            ax.fill_between(m[xc], m['lo'], m['hi'], color=col, alpha=0.16, lw=0, zorder=1)
            ax.plot(m[xc], m['delta'], color=col, lw=1.4,
                    ls='--' if name == 'full' else '-', zorder=3 if name != 'full' else 2)
        if logx:
            ax.set_xscale('log')
            ax.set_xticks([10, 20, 40, 80])
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())
            ax.set_xlim(11, 90)
        ax.set_xlabel(xlabel)
        ax.set_ylim(-0.075, 0.29)
        ax.set_yticks([0.0, 0.1, 0.2])

    draw(ax_a, mod_r, obs_r, ratio_x, 'bin', 'Risky/safe payoff ratio')
    ax_a.set_ylabel('Δ P(chose risky), IPS − vertex')
    ax_a.set_xticks([1.5, 2.0, 2.5, 3.0])
    ax_a.set_xlim(1.48, 3.35)
    for i, (txt, col, w) in enumerate([('Observed', '0.15', 'bold'),
                                       ('Full model', FULL, 'normal'),
                                       ('Bias channel only', BIAS, 'normal'),
                                       ('Randomness channel only', NOISE, 'normal')]):
        ax_a.text(3.32, 0.278 - i * 0.030, txt, color=col, fontsize=7.5,
                  fontweight=w, ha='right', va='center')
    panel_letter(ax_a, 'a')

    draw(ax_b, mod_n, obs_n, payoff_x, 'n_risky_bin', 'Risky payoff (CHF)', logx=True)
    ax_b.set_ylabel('')
    ax_b.annotate('The randomness channel\ncontributes almost nothing',
                  xy=(62, -0.010), xytext=(12.5, -0.062),
                  fontsize=7.5, color=NOISE, ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.22',
                                  color=NOISE, lw=0.6))
    ax_b.annotate('Bias channel reproduces\nthe full model exactly', xy=(24, 0.026),
                  xytext=(26, 0.20), fontsize=7.5, color=BIAS, ha='left', va='center',
                  arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=0.25',
                                  color=BIAS, lw=0.6))
    panel_letter(ax_b, 'b')

    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    parser.add_argument('--label', default='flexible2',
                        help="the paper's model is flexible2 (5 splines)")
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/pmc_channels')
    args = parser.parse_args()
    main(args.data_dir, args.bids_folder, args.out, args.label)
