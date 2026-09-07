"""The fitted noise functions: the winning model on its own, and every variant compared.

Two figures.

`--mode winner` (default)  Three panels for one model: the two noise functions per
    stimulation condition, the cTBS contrast in CHF, and the same contrast as a
    percentage. Markers along the top of the contrast panels flag payoffs where the
    95% credible interval excludes zero.

`--mode variants`  One panel per fitted model, all on a shared scale, showing the
    cTBS contrast on the FIRST- and SECOND-presented option. Those two curves exist
    in both parameterisations -- family 1 fits them directly, family 2 composes them
    from the memory and perceptual terms -- so this is the one representation in
    which every variant can be compared like for like.

    python -m tms_risk.behavior.scripts.plot_noise_summary --label flexible2nf_perception
    python -m tms_risk.behavior.scripts.plot_noise_summary --mode variants

Reads notes/data/pmcpars_curves.<label>.tsv only.
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
FIRST, SECOND = '#3B5BA5', '#7b3294'

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
XT = [7, 14, 28, 56, 112]

# label -> (row title, whether it is the model the paper reports)
VARIANTS = [
    ('flexible2nf_perception', 'Fam. 2 — perceptual only\nBEST BY ELPD', True),
    ('flexible2nf', 'Fam. 2 — both terms\n+2.0 nats', False),
    ('flexible2nf_memory', 'Fam. 2 — memory only\n+59.9 nats', False),
    ('flexible2nf_null', 'Fam. 2 — null\n+115.6 nats', False),
    ('flexible1nf', 'Fam. 1 — both options\n+27.0 nats', False),
    ('flexible1nf_first', 'Fam. 1 — first option only\n+64.3 nats', False),
    ('flexible1nf_second', 'Fam. 1 — second option only\n+69.9 nats', False),
    ('flexible2.9nf', 'Fam. 2, 9 splines\ndid not converge', False),
]


def cred(ax, s, y, colr):
    """Tick marks where the 95% credible interval excludes zero."""
    sig = s[(s.lo > 0) | (s.hi < 0)]
    if len(sig):
        ax.plot(sig.payoff, np.full(len(sig), y), '|', color=colr, ms=3.5,
                mew=.9, clip_on=False)
    return len(sig)


def winner(data, label, out_stem):
    c = pd.read_csv(data / f'pmcpars_curves.{label}.tsv', sep='\t')
    fig = plt.figure(figsize=(7.25, 2.55))
    gs = fig.add_gridspec(1, 3, wspace=.42, left=.075, right=.985, top=.80, bottom=.20)

    # --- a: the two noise functions
    ax = fig.add_subplot(gs[0, 0])
    for term, ls in [('perceptual_noise_sd', '-'), ('memory_noise_sd', '--')]:
        dd = c[(c.term == term) & (c.stimulation == 'ips - vertex')]
        flat = bool(len(dd)) and float(dd.nu.abs().max()) < 1e-9
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = c[(c.term == term) & (c.stimulation == stim)].sort_values('payoff')
            if flat:                      # no cTBS regressor on this term
                if stim == 'ips':
                    continue
                colr = '.45'
            if ls == '-':
                ax.fill_between(s.payoff, s.lo, s.hi, color=colr, alpha=.15, lw=0)
            ax.plot(s.payoff, s.nu, color=colr, ls=ls, lw=1.3 if ls == '-' else 1.0)
    ax.set_xscale('log'); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Representational noise ν (CHF)')
    ax.text(.04, .96, 'Perceptual', transform=ax.transAxes, fontsize=7.2,
            color='.25', va='top')
    ax.text(.04, .16, 'Memory (dashed, no cTBS\nterm in this model)',
            transform=ax.transAxes, fontsize=6.4, color='.45', va='top',
            linespacing=1.25)
    ax.text(.97, .06, 'Vertex', transform=ax.transAxes, fontsize=7.2, color=VERTEX,
            ha='right')
    ax.text(.97, .19, 'IPS', transform=ax.transAxes, fontsize=7.2, color=IPS,
            ha='right')

    # --- b: cTBS contrast, absolute
    ax = fig.add_subplot(gs[0, 1])
    d = c[(c.term == 'perceptual_noise_sd') & (c.stimulation == 'ips - vertex')]
    d = d.sort_values('payoff')
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    ax.fill_between(d.payoff, d.lo, d.hi, color=DIFF, alpha=.16, lw=0)
    ax.plot(d.payoff, d.nu, color=DIFF)
    n_sig = cred(ax, d, d.hi.max() * 1.06, DIFF)
    ax.set_xscale('log'); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δ perceptual noise\nIPS − vertex (CHF)')
    ax.set_ylim(min(0, d.lo.min() * 1.1), d.hi.max() * 1.18)

    # --- c: the same, proportionally
    ax = fig.add_subplot(gs[0, 2])
    i = c[(c.term == 'perceptual_noise_sd') & (c.stimulation == 'ips')].set_index('payoff').nu
    v = c[(c.term == 'perceptual_noise_sd') & (c.stimulation == 'vertex')].set_index('payoff').nu
    rel = 100 * (i / v - 1)
    ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
    ax.plot(rel.index.values, rel.values, color=DIFF)
    ax.set_xscale('log'); ax.set_xticks(XT)
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δ perceptual noise\nIPS − vertex (%)')
    ax.set_ylim(0, rel.max() * 1.3)
    lo = rel.index[np.abs(rel.index - 7).argmin()]
    ax.annotate(f'{rel[lo]:.0f}% at 7 CHF\nvs {rel.iloc[-1]:.0f}% at 112',
                xy=(lo, rel[lo]), xytext=(16, rel.max() * 1.13), fontsize=6.8,
                color='.3', ha='left', va='center',
                arrowprops=dict(arrowstyle='-', connectionstyle='arc3,rad=.25',
                                color='.45', lw=.6))

    for i_, letter in enumerate('abc'):
        fig.axes[i_].text(-.26, 1.09, letter, transform=fig.axes[i_].transAxes, **PANEL)
    fig.suptitle(f'Noise functions of the best-fitting model ({label})',
                 fontsize=9, y=.95, color='.15')
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf   ({n_sig}/{len(d)} grid points with 95% CrI excluding 0)')


def variants(data, out_stem):
    have = [(lab, ttl, star) for lab, ttl, star in VARIANTS
            if (data / f'pmcpars_curves.{lab}.tsv').exists()]
    n = len(have)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.25, 2.15 * nrow + .7),
                             sharex=True, sharey=True, squeeze=False)
    lim = 0.
    store = {}
    for lab, _, _ in have:
        c = pd.read_csv(data / f'pmcpars_curves.{lab}.tsv', sep='\t')
        store[lab] = c
        for term in ['n1_evidence_sd', 'n2_evidence_sd']:
            d = c[(c.term == term) & (c.stimulation == 'ips - vertex')]
            if len(d):
                lim = max(lim, np.abs(d.lo).max(), np.abs(d.hi).max())
    for k, (lab, ttl, star) in enumerate(have):
        ax = axes[k // ncol][k % ncol]
        c = store[lab]
        ax.axhline(0, color='.75', lw=.6, ls='--', zorder=0)
        for term, colr, nm in [('n1_evidence_sd', FIRST, 'First'),
                               ('n2_evidence_sd', SECOND, 'Second')]:
            d = c[(c.term == term) & (c.stimulation == 'ips - vertex')].sort_values('payoff')
            if not len(d):
                continue
            ax.fill_between(d.payoff, d.lo, d.hi, color=colr, alpha=.16, lw=0)
            ax.plot(d.payoff, d.nu, color=colr, lw=1.2)
        ax.set_xscale('log'); ax.set_xticks(XT)
        ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
        ax.set_ylim(-lim * 1.1, lim * 1.1)
        ax.set_title(ttl, fontsize=6.6, color='.05' if star else '.4', pad=3,
                     fontweight='bold' if star else 'normal', linespacing=1.3)
        if k % ncol == 0:
            ax.set_ylabel('Δ ν, IPS − vertex (CHF)', fontsize=7.5)
        if k // ncol == nrow - 1:
            ax.set_xlabel('Payoff (CHF)')
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis('off')
    axes[0][0].text(.05, .93, 'First-presented', transform=axes[0][0].transAxes,
                    fontsize=6.4, color=FIRST, va='top')
    axes[0][0].text(.05, .78, 'Second-presented', transform=axes[0][0].transAxes,
                    fontsize=6.4, color=SECOND, va='top')
    fig.suptitle('cTBS effect on each option\'s noise, across model variants '
                 '(ELPD cost relative to the best)', fontsize=9, y=.995, color='.15')
    fig.tight_layout(rect=[0, 0, 1, .955])
    sns.despine(fig=fig, offset=2)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)
    print(f'wrote {out_stem}.pdf  ({n} variants)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', default='winner', choices=['winner', 'variants'])
    parser.add_argument('--label', default='flexible2nf_perception')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--out', default=None)
    a = parser.parse_args()
    data = Path(a.data_dir)
    fig_dir = Path('/Users/gdehol/git/tms_risk/notes/figures')
    if a.mode == 'winner':
        winner(data, a.label, a.out or str(fig_dir / f'noise_winner.{a.label}'))
    else:
        variants(data, a.out or str(fig_dir / 'noise_variants'))
