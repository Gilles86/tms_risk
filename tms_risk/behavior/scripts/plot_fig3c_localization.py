"""Candidate Figure 3C: the cTBS choice effect is confined to the smallest risky payoffs.

Model-free companion panel for Fig 3 (matched height/style to plot_fig3_probit.py):
observed Delta P(chose risky), IPS - vertex, per risky-payoff quintile
(localnoise_delta_by_nrisky.tsv), with the hierarchical probit's own posterior
prediction for risky-second trials as a band -- the same probit as panels A/B, so the
panel also shows that the model's slope+intercept account is left-weighted by itself.
The bin containing the nPRF preferred-numerosity IQR [6, 10] is shaded.

Headline: in the 7-17 CHF band cTBS moves risky-second choices from 35% to 51%
risky -- i.e. to chance -- (+0.160 [0.067, 0.252], p = 0.001), with +0.03..0.05
everywhere above and nothing on risky-first trials.

    python -m tms_risk.behavior.scripts.plot_fig3c_localization
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

DARK, LIGHT = '.15', '.62'          # fig3's convention: dark = risky second

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'mathtext.fontset': 'stixsans',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main(tsv, out_stem):
    d = pd.read_csv(tsv, sep='\t')
    bins = list(d[d.order == 'Risky second'].n_risky_bin)
    x = np.arange(len(bins))

    fig, ax = plt.subplots(figsize=(2.9, 3.2), constrained_layout=True)
    ax.axvspan(-.45, .45, color='.93', zorder=0)
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=1)

    # the probit's own prediction for risky-second trials, as a band
    rs = d[d.order == 'Risky second'].reset_index(drop=True)
    ax.fill_between(x, rs.probit_lo, rs.probit_hi, color=DARK, alpha=.13, lw=0,
                    zorder=2)
    ax.plot(x, rs.probit, color=DARK, lw=1.0, alpha=.55, zorder=3)

    for order, colr, mfc, dx in [('Risky first', LIGHT, 'white', -.12),
                                 ('Risky second', DARK, DARK, .12)]:
        g = d[d.order == order].reset_index(drop=True)
        ax.errorbar(x + dx, g.delta, yerr=[g.delta - g.ci_lo, g.ci_hi - g.delta],
                    fmt='o', color=colr, mfc=mfc, ms=4.4, lw=0, elinewidth=1.1,
                    capsize=0, zorder=4)

    ax.annotate('35% to 51% risky\np = 0.001',
                xy=(x[0] + .12, rs.delta.iloc[0]), xytext=(x[0] + .75, .225),
                fontsize=7, color='.2', ha='left', va='center', linespacing=1.3,
                arrowprops=dict(arrowstyle='-|>', color='.2', lw=1.0,
                                mutation_scale=8, shrinkA=3, shrinkB=6,
                                relpos=(0., 0.5)))
    ax.text(-.38, .96, 'nPRF-preferred\npayoffs', transform=ax.get_xaxis_transform(),
            fontsize=7, color='.42', ha='left', va='top', linespacing=1.25)
    ax.text(.97, .955, 'Risky second', transform=ax.transAxes, fontsize=7.5,
            color=DARK, ha='right')
    ax.text(.97, .875, 'Risky first', transform=ax.transAxes, fontsize=7.5,
            color=LIGHT, ha='right')

    ax.set_xticks(x)
    ax.set_xticklabels(bins, rotation=35, ha='right', rotation_mode='anchor')
    ax.set_xlim(-.55, len(bins) - .45)
    ax.set_ylim(-.12, .28)
    ax.set_yticks([-.1, 0, .1, .2])
    ax.set_xlabel('Risky payoff (CHF)')
    ax.set_ylabel('Δ P(chose risky), IPS − vertex')
    sns.despine(fig=fig, offset=4, trim=True)

    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')
    print(rs[['n_risky_bin', 'delta', 'ci_lo', 'ci_hi', 'p']].to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tsv', default='notes/data/localnoise_delta_by_nrisky.tsv')
    p.add_argument('--out', default='notes/figures/fig3c_localization')
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.tsv, a.out)
