"""The fitted noise functions of the anchor grid, and the cTBS effect on them.

Top block: sigma(payoff) for the first- and second-presented option under each
noise form, IPS vs vertex. Bottom block: the cTBS effect itself, IPS - vertex,
differenced within draw so the credible interval is on the difference.

Reads notes/data/anchor_curves.tsv, written by
`behavior/scripts/extract_anchor_curves.py` on the node holding the traces.

    python -m tms_risk.behavior.scripts.plot_anchor_noise --placement percmem
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

IPS, VERTEX = '#d62728', '#2ca02c'
FORMS = ['weber', 'affine', 'power', 'genweber', 'spl3', 'spl5']
PRETTY = {'weber': 'Weber\nσ constant', 'affine': 'Affine\nσ linear in log x',
          'power': 'Power\nlog σ linear in log x',
          'genweber': 'Generalized Weber\nσ = k + c/x',
          'spl3': 'Spline, 3 anchors', 'spl5': 'Spline, 5 anchors'}

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


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def main(tsv, out_stem, space, placement):
    d = pd.read_csv(tsv, sep='\t', keep_default_na=False, na_values=[''])
    d = d[(d.space == space) & (d.placement == placement)]
    forms = [f for f in FORMS if f in set(d.form)]
    if not forms:
        raise SystemExit(f'nothing for {space}/{placement} in {tsv}')

    fig, axes = plt.subplots(4, 3, figsize=(7.25, 7.6), constrained_layout=True,
                             sharex=True)
    top, bot = axes[:2].ravel(), axes[2:].ravel()

    ylim_top = (0, 1.05 * d[d.condition.isin(['ips', 'vertex'])
                            & d.channel.isin(['n1', 'n2'])].hi.max())
    dd = d[d.condition == 'delta']
    m = 1.08 * max(abs(dd.lo.min()), abs(dd.hi.max()))

    for k, form in enumerate(forms):
        f = d[d.form == form]
        # -- sigma(x) -----------------------------------------------------
        ax = top[k]
        for chan, ls in [('n1', '-'), ('n2', (0, (3, 1.6)))]:
            for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
                s = f[(f.channel == chan) & (f.condition == cond)].sort_values('x')
                ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.16, lw=0, zorder=1)
                ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.2, zorder=3)
        logx(ax)
        ax.set_ylim(*ylim_top)
        ax.set_title(PRETTY[form], fontsize=7.5, color='0.15', pad=3)
        if k % 3 == 0:
            ax.set_ylabel('σ (log CHF)' if space == 'log' else 'σ (CHF)')

        # -- the cTBS effect ----------------------------------------------
        ax = bot[k]
        ax.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
        for chan, ls, col in [('n1', '-', '#3B5BA5'), ('n2', (0, (3, 1.6)), '#8172B2')]:
            s = f[(f.channel == chan) & (f.condition == 'delta')].sort_values('x')
            ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.18, lw=0, zorder=1)
            ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.2, zorder=3)
        logx(ax)
        ax.set_ylim(-m, m)
        if k % 3 == 0:
            ax.set_ylabel('Δσ, IPS − vertex')
        if k >= 3:
            ax.set_xlabel('Payoff (CHF)')

    for ax in list(top[len(forms):]) + list(bot[len(forms):]):
        ax.set_visible(False)

    # direct labels, first panel of each block
    a = top[0]
    a.text(.06, .95, 'IPS', color=IPS, transform=a.transAxes, va='top', fontsize=7.5)
    a.text(.06, .82, 'Vertex', color=VERTEX, transform=a.transAxes, va='top',
           fontsize=7.5)
    a.text(.97, .12, 'Solid: first-presented (σ$_{n1}$)\nDashed: second (σ$_{n2}$)',
           transform=a.transAxes, ha='right', va='bottom', fontsize=6.5,
           color='0.35', linespacing=1.5)
    b = bot[0]
    b.text(.06, .95, 'First-presented', color='#3B5BA5', transform=b.transAxes,
           va='top', fontsize=7.5)
    b.text(.06, .82, 'Second-presented', color='#8172B2', transform=b.transAxes,
           va='top', fontsize=7.5)

    for ax, letter in [(top[0], 'a'), (bot[0], 'b')]:
        ax.text(-.28, 1.12, letter, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left', family='Arial')

    fig.suptitle(f'Anchor grid, {space} space · cTBS on {placement} · '
                 f'group posterior median, 95% CrI',
                 fontsize=7.5, color='0.35', y=1.015)
    sns.despine(fig=fig, offset=4, trim=False)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    REPO = Path(__file__).resolve().parents[2]
    ap.add_argument('--tsv', default=str(REPO.parent / 'notes/data/anchor_curves.tsv'))
    ap.add_argument('--out_stem',
                    default=str(REPO.parent / 'notes/figures/anchor_noise'))
    ap.add_argument('--space', default='log')
    ap.add_argument('--placement', default='percmem')
    args = ap.parse_args()
    Path(args.out_stem).parent.mkdir(parents=True, exist_ok=True)
    main(args.tsv, args.out_stem, args.space, args.placement)
