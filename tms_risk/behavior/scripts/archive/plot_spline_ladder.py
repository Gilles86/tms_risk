"""How the fitted noise function changes with the flexibility of the spline basis.

One column per basis. Top row: the vertex noise function with its 95% credible band.
Bottom row: the cTBS contrast. Each column is annotated with the sampler diagnostics,
because that is what separates a usable fit from an unusable one here -- an
over-parameterised basis produces a curve that looks entirely reasonable while the
chains have not mixed at all.

    python -m tms_risk.behavior.scripts.plot_spline_ladder

Reads notes/data/pmcpars_curves.<label>.tsv for each rung present, plus
notes/data/ladder_convergence.tsv (label, rhat, ess, divergences) if it exists.
Rungs that have not been fitted yet are skipped.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
# Colour is semantic across the paper: green = vertex, red = IPS. A DIFFERENCE
# between them is a third quantity, not one of the conditions, so it gets its own
# near-black ink rather than borrowing the IPS red.
DIFF = '#1a1a1a'
XT = [7, 14, 28, 56, 112]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
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

# (label, heading, interior knots)
RUNGS = [
    ('flexible2.2nf', 'df 2, linear\n0 interior knots', 0),
    ('flexible2.3nf', 'df 3, quadratic\n0 interior knots', 0),
    ('flexible2.4nf', 'df 4, cubic\n0 interior knots', 0),
    ('flexible2nf', 'df 5, cubic\n1 interior knot', 1),
    ('flexible2.6nf', 'df 6, linear\n4 interior knots', 4),
    ('flexible2.9nf', 'df 9, cubic\n5 interior knots', 5),
]


def main(data_dir, out_stem):
    data = Path(data_dir)
    conv = {}
    f = data / 'ladder_convergence.tsv'
    if f.exists():
        c = pd.read_csv(f, sep='\t').set_index('label')
        conv = c.to_dict('index')

    have = [(lab, head, k) for lab, head, k in RUNGS
            if (data / f'pmcpars_curves.{lab}.tsv').exists()]
    if not have:
        raise SystemExit('no ladder rungs extracted yet')
    n = len(have)
    fig, axes = plt.subplots(2, n, figsize=(1.42 * n + .9, 4.2), squeeze=False,
                             sharex=True, sharey='row')

    ylim_c = [0, 0]
    for j, (lab, head, knots) in enumerate(have):
        c = pd.read_csv(data / f'pmcpars_curves.{lab}.tsv', sep='\t')
        top, bot = axes[0][j], axes[1][j]
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = c[(c.term == 'perceptual_noise_sd')
                  & (c.stimulation == stim)].sort_values('payoff')
            top.fill_between(s.payoff, s.lo, s.hi, color=colr, alpha=.16, lw=0)
            top.plot(s.payoff, s.nu, color=colr, lw=1.3)
        d = c[(c.term == 'perceptual_noise_sd')
              & (c.stimulation == 'ips - vertex')].sort_values('payoff')
        bot.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
        bot.fill_between(d.payoff, d.lo, d.hi, color=DIFF, alpha=.16, lw=0)
        bot.plot(d.payoff, d.nu, color=DIFF, lw=1.3)
        ylim_c = [min(ylim_c[0], d.lo.min()), max(ylim_c[1], d.hi.max())]

        m = conv.get(lab)
        diag = (f"\nr̂ {m['rhat']:.2f}, ESS {m['ess']:.0f}" if m else '')
        top.set_title(head + diag, fontsize=7, color='.2', pad=4, linespacing=1.35)
        for ax in (top, bot):
            ax.set_xscale('log'); ax.set_xticks(XT)
            ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        top.set_yscale('log'); top.set_yticks([1, 2, 4, 8])
        top.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        top.get_yaxis().set_minor_formatter(mpl.ticker.NullFormatter())
        bot.set_xlabel('Payoff (CHF)')

        # Frame only fits whose chains genuinely did not mix. ESS a little under 400
        # is a precision warning; r_hat well above 1.01 means the posterior summary is
        # not to be trusted at all, which is a different kind of failure.
        if m and (m['rhat'] > 1.05 or m['ess'] < 100):
            for ax in (top, bot):
                for sp in ax.spines.values():
                    sp.set_color('#b2182b')
                    sp.set_linewidth(1.2)
            top.title.set_color('#b2182b')
    for j in range(n):
        axes[1][j].set_ylim(ylim_c[0] * 1.05, ylim_c[1] * 1.05)
    axes[0][0].set_ylabel('Perceptual noise\nν (CHF)')
    axes[1][0].set_ylabel('Δ noise\nIPS − vertex (CHF)')
    axes[0][0].text(.06, .94, 'IPS', transform=axes[0][0].transAxes, fontsize=7,
                    color=IPS, va='top')
    axes[0][0].text(.06, .80, 'Vertex', transform=axes[0][0].transAxes, fontsize=7,
                    color=VERTEX, va='top')
    fig.suptitle('Red: chains did not mix, so the posterior summary is not usable',
                 fontsize=7.2, y=.015, color='.35')
    sns.despine(fig=fig, offset=3)
    fig.tight_layout(rect=[0, .055, 1, 1])
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf   ({n} rungs: '
          + ', '.join(lab for lab, _, _ in have) + ')')
    for lab, head, _ in have:
        m = conv.get(lab)
        if m:
            print(f'  {lab:16s} rhat {m["rhat"]:.3f}  ESS {m["ess"]:6.0f}  '
                  f'div {m["divergences"]:.0f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out',
                        default='/Users/gdehol/git/tms_risk/notes/figures/spline_ladder')
    a = parser.parse_args()
    main(a.data_dir, a.out)
