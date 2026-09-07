"""Figure 5, rebuilt on the anchor grid: the causal chain from percept to choice.

Rows are presentation order; the argument runs left to right.

a  What cTBS does to each option's perceived value, as a percentage of that
   option's own value. Both options lose value -- the percept is pulled toward
   the prior -- but a choice only changes if one loses MORE than the other.
b  The consequence: the perceived risky/safe ratio, IPS over vertex.
c  The behavioural effect that follows, model band against the observed cTBS
   difference (paired within subject).

The point of the figure is the contrast between the rows: the chain only closes
when the risky option came second.

Panel a is deliberately relative, not absolute -- in CHF the shift grows with
payoff purely because the percepts do, which would make an absolute panel say
'the effect is biggest at large payoffs' when the proportional pull is flat.

    python -m tms_risk.behavior.scripts.plot_fig5_anchor --model_label log-spl3-percmem
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

IPS = '#d62728'
RISKY, SAFE = '#3B5BA5', '#C97B2E'
READ = dict(sep='\t', keep_default_na=False, na_values=[''])
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def logx(ax, vals):
    ax.set_xscale('log')
    ax.set_xticks(sorted(vals))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.minorticks_off()
    ax.set_xlim(min(vals) * .88, max(vals) * 1.14)


def band(ax, d, color, ls='-'):
    d = d.sort_values('n_safe')
    ax.fill_between(d.n_safe, d.lo, d.hi, color=color, alpha=.18, lw=0, zorder=1)
    ax.plot(d.n_safe, d['mid'], color=color, ls=ls, lw=1.3, zorder=3)


def main(data_dir, out_stem, label):
    f = Path(data_dir) / f'decision_space.{label}.tsv'
    if not f.exists():
        raise SystemExit(f'{f} not found — run extract_anchor_decision_space first')
    d = pd.read_csv(f, **READ)
    safes = sorted(d.n_safe.unique())

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 4.5), sharex=True,
                             constrained_layout=True)
    for r, order in enumerate(ORDERS):
        dd = d[d.order == order]

        ax = axes[r, 0]
        ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
        band(ax, dd[dd.quantity == 'rel_risky'], RISKY)
        band(ax, dd[dd.quantity == 'rel_safe'], SAFE, ls=(0, (3, 1.5)))
        ax.set_ylabel(f'Perceived value,\ncTBS effect (%)\n{order.lower()}')

        ax = axes[r, 1]
        ax.axhline(1, color='0.8', lw=.6, ls='--', zorder=0)
        band(ax, dd[dd.quantity == 'ratio_shift'], '0.15')
        ax.set_ylabel('Perceived risky/safe\nratio, IPS / vertex')

        ax = axes[r, 2]
        ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
        band(ax, dd[dd.quantity == 'dp'], IPS)
        o = dd[dd.quantity == 'dp_observed'].sort_values('n_safe')
        if len(o):
            ax.errorbar(o.n_safe, o.observed, yerr=o.observed_sem, fmt='o',
                        ms=3.6, color='0.15', lw=0, elinewidth=.9, capsize=0,
                        zorder=4)
        ax.set_ylabel('ΔP(chose risky)\nIPS − vertex')

        for c in range(3):
            logx(axes[r, c], safes)
            if r == 1:
                axes[r, c].set_xlabel('Safe payoff (CHF)')

    # shared y-scales per column, so the two rows are actually comparable
    for c in range(3):
        lo = min(axes[r, c].get_ylim()[0] for r in range(2))
        hi = max(axes[r, c].get_ylim()[1] for r in range(2))
        for r in range(2):
            axes[r, c].set_ylim(lo, hi)

    axes[0, 0].text(.04, .10, 'Risky option', color=RISKY,
                    transform=axes[0, 0].transAxes, fontsize=7)
    axes[0, 0].text(.04, .02, 'Safe option', color=SAFE,
                    transform=axes[0, 0].transAxes, fontsize=7)
    # The band is the posterior of the model's MEAN effect (parameter
    # uncertainty), not a predictive interval: dP is a deterministic contrast
    # between two counterfactuals on the same trials, so there is no trial-level
    # noise in it. The observed points carry that noise; say so rather than
    # letting the widths be read as comparable.
    axes[0, 2].text(.96, .04, 'Points: data ± SEM\nBand: model mean effect, 95% CrI',
                    transform=axes[0, 2].transAxes, ha='right', fontsize=6.2,
                    color='0.4', linespacing=1.5, va='bottom')
    for ax, s in zip(axes[0], 'abc'):
        ax.text(-.34, 1.07, s, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left', family='Arial')
    fig.suptitle(f'{label} · group posterior, subjects averaged within draw',
                 fontsize=7, color='0.4', y=1.02)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data/decision_space'))
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/fig5_anchor'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
