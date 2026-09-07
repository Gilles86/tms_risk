"""The fitted noise curves by stimulation condition, in log AND natural space.

Same posterior, three views. Only the units change, but the units decide what a
reader concludes:

  a  log space, shared coordinates -- "memory" and "perceptual". NOTE these are
     softplus of ONE TERM of a sum; the composition is
     n1 = softplus(memory + perceptual), so neither curve is the noise SD of any
     option. At 7 CHF the memory curve reads 0.79 while n1 is 0.22. Coordinates,
     not observables.
  b  log space, per option (independent fit) -- sigma on the log scale, i.e.
     RELATIVE noise. Weber's law is a flat line here.
  c  natural space, per option -- the SD in CHF. For log-SD sigma at payoff x,

         SD(x) = x * exp(sigma**2 / 2) * sqrt(exp(sigma**2) - 1)

     which is NOT x * sigma once sigma is large: at sigma = 0.9 the factor is
     1.67, not 0.9. So this is not a rescaled copy of panel b.

Reads notes/data/noise_curves.tsv (per subject x draw, averaged over subjects
within draw, then summarized across draws).

    python -m tms_risk.behavior.scripts.plot_ctbs_prediction
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

IPS, VERTEX = '#d62728', '#2ca02c'      # house palette: stimulated red, sham green


def lognormal_sd(payoff, sigma):
    """SD in CHF of a lognormal whose log has mean log(payoff) and SD sigma."""
    return payoff * np.exp(sigma ** 2 / 2) * np.sqrt(np.exp(sigma ** 2) - 1)


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


def draw(ax, sub, natural=False):
    for param in sorted(sub.param.unique()):
        for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
            s = sub[(sub.param == param) & (sub.stim == stim)].sort_values('payoff')
            y, lo, hi = s['median'].values, s.lo.values, s.hi.values
            if natural:
                y = lognormal_sd(s.payoff.values, y)
                lo = lognormal_sd(s.payoff.values, lo)
                hi = lognormal_sd(s.payoff.values, hi)
            ax.fill_between(s.payoff, lo, hi, color=col, alpha=.14, lw=0)
            ax.plot(s.payoff, y, color=col, lw=1.5)
        s = sub[(sub.param == param) & (sub.stim == 'vertex')].sort_values('payoff')
        y0 = s['median'].iloc[0]
        if natural:
            y0 = lognormal_sd(s.payoff.iloc[0], y0)
        ax.text(7.35, y0 * 1.14, param, fontsize=6.4, color='.2',
                ha='left', va='bottom')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='notes/data/noise_curves.tsv')
    ap.add_argument('--out', default='notes/figures/ctbs_prediction')
    args = ap.parse_args()
    d = pd.read_csv(args.tsv, sep='\t')

    shared = d[(d.model == 'logflex2') & (d.quantity == 'channel')]
    indep = d[(d.model == 'logflex1') & (d.quantity == 'option')]

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), constrained_layout=True)

    draw(axes[0], shared)
    axes[0].set_title('Log space, shared coordinates\n(not option noise)',
                      fontsize=7.5, linespacing=1.3)
    axes[0].set_ylabel('Noise SD (log units)')

    draw(axes[1], indep)
    axes[1].set_title('Log space, per option\n(relative noise)',
                      fontsize=7.5, linespacing=1.3)
    axes[1].set_ylabel('Noise SD (log units)')
    axes[1].text(.96, .04, 'Weber = flat', transform=axes[1].transAxes,
                 fontsize=6.3, color='.45', ha='right')

    draw(axes[2], indep, natural=True)
    axes[2].set_title('Natural space, per option\n(absolute noise)',
                      fontsize=7.5, linespacing=1.3)
    axes[2].set_ylabel('Noise SD (CHF)')

    for ax in axes:
        ax.set_yscale('log')
        logx(ax)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
    for ax in axes[:2]:
        ax.set_yticks([0.1, 0.3, 1.0])
    axes[2].set_yticks([1, 3, 10, 30, 100])

    axes[0].text(.04, .10, 'IPS', transform=axes[0].transAxes, color=IPS,
                 fontsize=7, va='bottom')
    axes[0].text(.04, .01, 'Vertex', transform=axes[0].transAxes, color=VERTEX,
                 fontsize=7, va='bottom')

    sns.despine(fig=fig, offset=3)
    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.22, 1.11, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    print('Independent fit (logflex1), IPS - vertex:')
    for param in sorted(indep.param.unique()):
        for target in [7.0, 28.0, 112.0]:
            near = indep.payoff.iloc[(indep.payoff - target).abs().argsort().iloc[0]]
            s = indep[(indep.param == param) & (indep.payoff == near)]
            i = float(s[s.stim == 'ips']['median'].iloc[0])
            v = float(s[s.stim == 'vertex']['median'].iloc[0])
            print(f'  {param:12s} {near:6.1f} CHF   log {i - v:+.3f}   '
                  f'natural {lognormal_sd(near, i) - lognormal_sd(near, v):+7.2f} CHF')
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
