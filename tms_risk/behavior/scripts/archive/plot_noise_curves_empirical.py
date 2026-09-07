"""The fitted noise curves, both parameterizations, by stimulation condition.

Left: the shared decomposition (logflex2) in its own terms -- memory and
perceptual. Middle: the SAME fit re-expressed as the two presented options,
n1 = softplus(memory + perceptual) and n2 = softplus(perceptual), which is
what the choice rule actually sees. Right: the independent fit (logflex1),
which estimates those two option noises directly.

Middle and right are the same observable quantity from two parameterizations,
so the comparison says whether the decomposition changes the answer.

Reads notes/data/noise_curves.tsv (per-subject-per-draw curves, averaged over
subjects within draw, then summarized across draws).
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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

# House palette: IPS (stimulated) red, vertex (sham) green.
IPS, VERTEX = '#d62728', '#2ca02c'

PANELS = [
    ('logflex2', 'channel', 'Shared decomposition\n(memory / perceptual)'),
    ('logflex2', 'option', 'Same fit, per option\n(n1 = mem + perc)'),
    ('logflex1', 'option', 'Independent fit\n(n1, n2 estimated directly)'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tsv', default='notes/data/noise_curves.tsv')
    ap.add_argument('--out', default='notes/figures/noise_curves_empirical')
    args = ap.parse_args()
    d = pd.read_csv(args.tsv, sep='\t')

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), constrained_layout=True,
                             sharey=True)
    for ax, (model, quantity, title) in zip(axes, PANELS):
        sub = d[(d.model == model) & (d.quantity == quantity)]
        for param in sorted(sub.param.unique()):
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                s = sub[(sub.param == param) & (sub.stim == stim)].sort_values('payoff')
                ax.fill_between(s.payoff, s.lo, s.hi, color=col, alpha=.14, lw=0)
                ax.plot(s.payoff, s['median'], color=col, lw=1.5)
            s = sub[(sub.param == param) & (sub.stim == 'vertex')].sort_values('payoff')
            ax.text(7.4, s['median'].iloc[0] * 1.10, param, fontsize=6.4,
                    color='.2', ha='left', va='bottom')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xticks([7, 14, 28, 56, 112])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yticks([0.1, 0.3, 1.0])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlabel('Payoff (CHF)')
        ax.set_title(title, fontsize=7.5, linespacing=1.3)
    axes[0].set_ylabel('Noise SD (log units)')
    axes[0].text(.04, .04, 'IPS', transform=axes[0].transAxes, color=IPS,
                 fontsize=7, va='bottom')
    axes[0].text(.04, .14, 'Vertex', transform=axes[0].transAxes, color=VERTEX,
                 fontsize=7, va='bottom')

    sns.despine(fig=fig, offset=3)
    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.16, 1.10, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{args.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    piv = (d[d.payoff.isin([d.payoff.min(), d.payoff.max()])]
           .pivot_table(index=['model', 'param', 'payoff'], columns='stim',
                        values='median'))
    piv['ips_minus_vertex'] = piv['ips'] - piv['vertex']
    print(piv.round(3).to_string())
    print(f'wrote {args.out}.pdf')


if __name__ == '__main__':
    main()
