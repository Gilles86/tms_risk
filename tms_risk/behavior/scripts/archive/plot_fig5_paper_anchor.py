"""Figure 5 in the published layout, driven by the anchor model.

Rows are presentation order; the argument runs left to right, one column per
step of the causal chain, exactly as in `plot_fig5_paper`:

    a  What cTBS does to each option's perceived value, as a percentage of that
       option's own value. Both options lose (or gain) value because the percept
       is pulled toward the prior -- but a choice only changes if one moves MORE
       than the other.
    b  The consequence: the perceived risky/safe ratio, IPS over vertex, across
       the decision space.
    c  Where choice is actually sensitive to that ratio -- dP/d log(ratio) at
       vertex. A shift only matters where leverage is high.
    d  The behavioural effect that follows, large only where b and c both hold.
    e  That prediction against the observed cTBS effect.

Columns b-d are maps over the whole (safe payoff x ratio) plane rather than over
the trials that happened to be presented, which is what makes the chain legible.
All of it is closed form from the posterior -- see extract_anchor_decision_map.

    python -m tms_risk.behavior.scripts.plot_fig5_paper_anchor
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
RISKY, SAFE = '#3B5BA5', '#C97B2E'
IPS, DATA = '#d62728', '0.15'
ORDERS = ['Risky first', 'Risky second']

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 7.5,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def grid(d, q, order):
    """(ratio x n_safe) matrix, matching the published Figure 5 orientation:
    safe payoff on x, risky/safe ratio on y."""
    s = d[(d.quantity == q) & (d.order == order)]
    p = s.pivot_table(index='ratio', columns='n_safe', values='mid')
    return p.values, p.columns.values, p.index.values


def heat(ax, M, safes, ratios, cmap, vmin, vmax, center=None):
    im = ax.pcolormesh(np.arange(len(safes)), ratios, M, cmap=cmap, vmin=vmin,
                       vmax=vmax, shading='gouraud', rasterized=True)
    ax.set_xticks(np.arange(len(safes)))
    ax.set_xticklabels([f'{v:.0f}' for v in safes])
    ax.set_yscale('log')
    ax.set_yticks([1.5, 2, 2.5, 3])
    ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    ax.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    return im


def main(data_dir, out_stem, label):
    dd = Path(data_dir)
    d = pd.read_csv(dd / f'decision_map/decision_map.{label}.tsv', **READ)
    ds = pd.read_csv(dd / f'decision_space/decision_space.{label}.tsv', **READ) \
        if (dd / f'decision_space/decision_space.{label}.tsv').exists() else None

    fig, axes = plt.subplots(2, 5, figsize=(7.25, 3.5), constrained_layout=True)
    lims = {q: np.abs(d[d.quantity == q]['mid']).max() for q in
            ('ratio_shift', 'dp')}
    lev_max = d[d.quantity == 'leverage']['mid'].max()

    for r, order in enumerate(ORDERS):
        # -- a: per-option perceived value change ------------------------
        ax = axes[r, 0]
        ax.axhline(0, color='0.85', lw=.6, ls='--', zorder=0)
        for q, col, ls in [('rel_risky', RISKY, '-'),
                           ('rel_safe', SAFE, (0, (3, 1.5)))]:
            s = (d[(d.quantity == q) & (d.order == order)]
                 .groupby('n_safe')['mid'].mean())
            ax.plot(np.arange(len(s)), s.values, color=col, ls=ls, lw=1.3)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['7', '10', '14', '20', '28'])
        ax.set_ylabel(f'{order}\n\ncTBS effect on\nperceived value (%)',
                      fontsize=7)

        # -- b, c, d: maps -----------------------------------------------
        for k, (q, cmap, lab) in enumerate([
                ('ratio_shift', 'RdBu_r', 'Perceived risky/safe\nratio, IPS / vertex'),
                ('leverage', 'mako_r', 'Choice sensitivity\ndP / d log(ratio)'),
                ('dp', 'RdBu_r', 'ΔP(chose risky)\nIPS − vertex')]):
            M, safes, ratios = grid(d, q, order)
            if q == 'ratio_shift':
                m = np.abs(M - 1).max()
                im = heat(axes[r, 1 + k], M, safes, ratios, cmap, 1 - m, 1 + m)
            elif q == 'leverage':
                im = heat(axes[r, 1 + k], M, safes, ratios, cmap, 0, lev_max)
            else:
                m = lims['dp']
                im = heat(axes[r, 1 + k], M, safes, ratios, cmap, -m, m)
            if r == 1:
                cb = fig.colorbar(im, ax=axes[:, 1 + k], fraction=.06, pad=.02,
                                  location='bottom')
                cb.ax.tick_params(labelsize=5.5)
                cb.set_label(lab, fontsize=6)
            if k == 0:
                axes[r, 1].set_ylabel('Risky/safe ratio', fontsize=7)

        # -- e: model against the observed effect -------------------------
        ax = axes[r, 4]
        ax.axhline(0, color='0.85', lw=.6, ls='--', zorder=0)
        s = (d[(d.quantity == 'dp') & (d.order == order)]
             .groupby('n_safe')[['lo', 'mid', 'hi']].mean())
        o = (ds[(ds.quantity == 'dp_observed') & (ds.order == order)]
             .sort_values('n_safe') if ds is not None else None)
        if o is not None:
            # widen the parameter band by the observed sampling SE, so the band
            # is a predictive interval for a measurement of this size
            se = o.observed_sem.values
            half = (s.hi.values - s.lo.values) / 2
            wide = np.sqrt(half ** 2 + (1.96 * se) ** 2)
            ax.fill_between(np.arange(len(s)), s['mid'] - wide, s['mid'] + wide,
                            color=IPS, alpha=.13, lw=0)
        ax.fill_between(np.arange(len(s)), s.lo, s.hi, color=IPS, alpha=.22, lw=0)
        ax.plot(np.arange(len(s)), s['mid'], color=IPS, lw=1.3)
        if o is not None:
            ax.errorbar(np.arange(len(o)), o.observed, yerr=o.observed_sem,
                        fmt='o', ms=3.2, color=DATA, lw=0, elinewidth=.9,
                        capsize=0, zorder=4)
        ax.set_xticks(np.arange(5))
        ax.set_xticklabels(['7', '10', '14', '20', '28'])
        ax.set_ylabel('ΔP(chose risky)', fontsize=7)
        ax.set_ylim(-.06, .16)

    for c in range(5):
        axes[1, c].set_xlabel('Safe payoff (CHF)', fontsize=7)
    axes[0, 0].text(.04, .95, 'Risky option', color=RISKY,
                    transform=axes[0, 0].transAxes, va='top', fontsize=6.5)
    axes[0, 0].text(.04, .80, 'Safe option', color=SAFE,
                    transform=axes[0, 0].transAxes, va='top', fontsize=6.5)
    axes[0, 4].text(.04, .95, 'Model', color=IPS,
                    transform=axes[0, 4].transAxes, va='top', fontsize=6.5)
    axes[0, 4].text(.04, .80, 'Data ± SEM', color=DATA,
                    transform=axes[0, 4].transAxes, va='top', fontsize=6.5)
    for c, letter in enumerate('abcde'):
        axes[0, c].text(-.05, 1.14, letter, transform=axes[0, c].transAxes,
                        fontsize=8, fontweight='bold', family='Arial',
                        va='bottom', ha='left')
    fig.suptitle(f'{label} · all quantities derived in closed form from the '
                 f'posterior, subjects averaged within draw',
                 fontsize=6.4, color='0.4', y=1.06)
    sns.despine(fig=fig, offset=2)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/fig5_paper_anchor'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
