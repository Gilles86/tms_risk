"""Figure 4: which model the data prefer, and what its noise function looks like.

    a   Every fitted model on one ELPD axis, as a difference from the best with its
        dSE. The models that carry an explicit PRESENTATION-POSITION parameter sit far
        below the one that has none, which is the quantitative answer to the objection
        that the order effect was fitted rather than emergent.
    b   The winning model's perceptual noise function on log-log axes, both stimulation
        conditions, against a slope-1 (Weber) reference. The fitted slope is ~0.5:
        noise grows with the square root of payoff, not in proportion to it.
    c   The cTBS increase as a percentage of baseline, with its credible interval. This
        is the scale on which the psychophysical analyses operate, and the scale on
        which the effect is magnitude-specific.

    python -m tms_risk.behavior.scripts.plot_fig4_model

Reads notes/data/table1_flexible8.tsv (or --table), noisecurve_reparam.<label>.tsv and
pmcpars_relative.<label>.tsv.
"""
import argparse
import re
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
FLEX, WEBER = '#3B5BA5', '#9c9c9c'
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

SHORT = [
    (r'.*perceptual noise only\)$',              'Perceptual noise'),
    (r'.*perceptual and memory noise\)$',        'Perceptual + memory'),
    (r'.*memory noise only\)$',                  'Memory noise'),
    (r'.*first- and second-option noise\)$',     'First + second option'),
    (r'.*first-presented option only\)$',        'First option'),
    (r'.*second-presented option only\)$',       'Second option'),
    (r'.*null model$',                           'No cTBS effect'),
]
# models whose cTBS effect is tied to presentation position rather than to a
# representational component -- these are the ones the argument turns on
POSITIONAL = {'First + second option', 'First option', 'Second option'}


def shorten(name):
    for pat, s in SHORT:
        if re.fullmatch(pat, name):
            return s
    return name


def main(data_dir, table, label, out_stem):
    data = Path(data_dir)
    t = pd.read_csv(table, sep='\t', index_col=0).sort_values('elpd_diff')
    t['short'] = t.name.map(shorten)
    t['weber'] = t.index.str.startswith('weber')
    t['family'] = t.index.str.extract(r'(?:flexible|weber)([12])')[0].values
    # loo_table renders both nulls as "Flexible PMC null model"; they are different
    # models (different prior coordinates) and need distinguishing on the axis
    dup = t.short.duplicated(keep=False)
    t.loc[dup, 'short'] = [f'{r.short} (fam. {r.family})' for _, r in t[dup].iterrows()]

    fig = plt.figure(figsize=(7.25, 2.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1, 1], wspace=.58,
                          left=.20, right=.985, top=.84, bottom=.19)

    # --- a: ELPD, as a cost relative to the best model
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(t))[::-1]
    for yi, (_, r) in zip(y, t.iterrows()):
        colr = WEBER if r.weber else FLEX
        mark = 'D' if r.short in POSITIONAL else 'o'
        ax.errorbar(r.elpd_diff, yi, xerr=r.dse, fmt=mark, color=colr, ms=4.4,
                    lw=0, elinewidth=1.1, capsize=0, zorder=3)
    ax.axvline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_yticks(y)
    lbl = [f'{r.short}' + ('' if r.weber else '') for _, r in t.iterrows()]
    ax.set_yticklabels(lbl, fontsize=7)
    for tick, (_, r) in zip(ax.get_yticklabels(), t.iterrows()):
        if r.short in POSITIONAL:
            tick.set_color('.15')
            tick.set_fontweight('bold')
    ax.set_xlabel('ELPD cost vs the best model (nats)')
    ax.set_ylim(-.8, len(t) - .2)
    ax.invert_xaxis()
    ax.text(.98, .06, 'Diamonds: models with an\nexplicit position parameter',
            transform=ax.transAxes, fontsize=6.3, color='.3', ha='right', va='bottom',
            linespacing=1.25)

    # --- b: the noise function, log-log
    ax = fig.add_subplot(gs[0, 1])
    c = pd.read_csv(data / f'pmcpars_curves.{label}.tsv', sep='\t')
    TERM = 'perceptual_noise_sd'
    p = c[(c.term == TERM) & (c.stimulation == 'vertex')].sort_values('payoff')
    for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
        s = c[(c.term == TERM) & (c.stimulation == stim)].sort_values('payoff')
        ax.fill_between(s.payoff, s.lo, s.hi, color=colr, alpha=.16, lw=0, zorder=1)
        ax.plot(s.payoff, s.nu, color=colr, zorder=2)
    x0, y0 = p.payoff.iloc[0], p.nu.iloc[0]
    xs = np.array([x0, p.payoff.iloc[-1]])
    ax.plot(xs, y0 * xs / x0, color='.6', lw=.7, ls=':', zorder=0)
    slope = np.polyfit(np.log(p.payoff.values), np.log(p.nu.values), 1)[0]
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks(XT); ax.set_yticks([1, 2, 4, 8])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Perceptual noise ν (CHF)')
    ax.text(.96, .10, 'IPS', transform=ax.transAxes, fontsize=7.2, color=IPS, ha='right')
    ax.text(.96, .02, 'Vertex', transform=ax.transAxes, fontsize=7.2, color=VERTEX,
            ha='right')
    ax.text(.03, .96, f'Slope {slope:.2f}', transform=ax.transAxes, fontsize=7,
            color='.25', va='top')
    ax.text(.30, .70, 'Weber (1)', transform=ax.transAxes, fontsize=6.4, color='.5',
            rotation=32)

    # --- c: the increase as a percentage, with its credible interval
    ax = fig.add_subplot(gs[0, 2])
    rel = pd.read_csv(data / f'pmcpars_relative.{label}.tsv', sep='\t')
    rel = rel[rel.term == 'perceptual_noise_sd'].sort_values('payoff')
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax.fill_between(rel.payoff, rel.lo, rel.hi, color=IPS, alpha=.18, lw=0, zorder=1)
    ax.plot(rel.payoff, rel.pct, color=IPS, zorder=2)
    ax.set_xscale('log'); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δ perceptual noise (%)\nIPS − vertex')
    lo7 = rel.iloc[(rel.payoff - 7).abs().argmin()]
    hi7 = rel.iloc[(rel.payoff - 112).abs().argmin()]

    for a, letter in zip(fig.axes, 'abc'):
        a.text(-.14 if letter == 'a' else -.30, 1.05, letter,
               transform=a.transAxes, **PANEL)
    sns.despine(fig=fig, offset=4)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf   ({len(t)} models)')
    print(f'  log-log slope {slope:.3f} (Weber = 1)')
    print(f'  relative effect: {lo7.pct:+.1f}% at 7 CHF [{lo7.lo:+.1f}, {lo7.hi:+.1f}], '
          f'{hi7.pct:+.1f}% at 112 [{hi7.lo:+.1f}, {hi7.hi:+.1f}]')
    pos = t[t.short.isin(POSITIONAL)]
    print('  positional models cost: '
          + ', '.join(f'{r.short} +{r.elpd_diff:.0f}' for _, r in pos.iterrows()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--table', default='/Users/gdehol/git/tms_risk/notes/data/table1_flexible8.tsv')
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/fig4_model')
    a = parser.parse_args()
    main(a.data_dir, a.table, a.label, a.out)
