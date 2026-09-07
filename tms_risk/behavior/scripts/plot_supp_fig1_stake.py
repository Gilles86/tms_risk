"""Supplementary Figure 1, regenerated from the published by-stake probit trace.

The v10 draft's Supp Fig 1 raster (from the old notebook) has the panel-D p-values
transposed between the two presentation-order columns: it prints p = 0.204 on the
risky-first/low-stake cell and p = 0.111 on the risky-second/high-stake cell, while
the trace gives risky-first/low = 0.111 and risky-second/high = 0.204 (see
`reproduce_stake_probit.py` and `notes/data/probit_stake_cells_published.tsv`; the
manuscript text quotes the correct values). This script replaces that figure,
computing every number directly from the stored trace so annotation placement
cannot drift from the data again.

Model (fit 2024-11, `fit_probit.py` label `probit_average_n_full`):
`chose_risky ~ x*risky_first*stimulation_condition*C(average_n_bin) + (1|subject)`,
probit link, x = log(risky/safe), reference levels IPS / low stake / safe first.
Group-level cell parameters are rebuilt per posterior draw by summing the named
fixed effects; RNP = exp(beta0 / beta1) (verified against the published panel-A
values, e.g. risky-second low 0.505 -> 0.554).

Layout follows the caption's A-D structure as columns, with one row per
(order x stake) cell as in `plot_fig3_probit_stake.py`: A) indifference points
(RNP) with 95% CrI, B) their IPS - vertex difference posterior, C) choice
consistency (psychometric slope), D) its difference posterior. p-values use the
min(P(d>0), P(d<0)) convention of the main Fig 3, so the risky-first/high slope
cell prints 0.248 (the slope goes the other way there) where the old figure
printed 0.752 -- same posterior mass, consistent convention.

    python -m tms_risk.behavior.scripts.plot_supp_fig1_stake

Cross-checks its slope cells against notes/data/probit_stake_cells_published.tsv
and aborts on disagreement.
"""
import argparse
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats as ss

VERTEX, IPS = '#2ca02c', '#d62728'
DIFF = '#2b2b2b'
CELLS = [('Risky first', 'Low stake'), ('Risky first', 'High stake'),
         ('Risky second', 'Low stake'), ('Risky second', 'High stake')]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8.5, 'axes.labelsize': 9.5, 'axes.titlesize': 9.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

# macOS ships Helvetica as a .ttc whose faces matplotlib cannot index, so every weight
# resolves to Regular and `fontweight='bold'` is silently a no-op. Arial Bold is a
# separate file and is metric-compatible, so route only the bold text through it.
BOLD = dict(fontname='Arial', fontweight='bold')


def cell_draws(post):
    """Per-draw group-level (beta0, beta1) for every (order, stake, stim) cell."""
    def term(name, level=None):
        v = post[name]
        if level is not None:
            dim = [d for d in v.dims if d != 'sample'][0]
            v = v.sel({dim: level})
        return v.values

    i = {k: term(n, l) for k, (n, l) in {
        '0': ('Intercept', None), 'rf': ('risky_first', None),
        'v': ('stimulation_condition', 'vertex'),
        'h': ('C(average_n_bin)', 'high'),
        'rfv': ('risky_first:stimulation_condition', 'vertex'),
        'rfh': ('risky_first:C(average_n_bin)', 'high'),
        'vh': ('stimulation_condition:C(average_n_bin)', 'vertex, high'),
        'rfvh': ('risky_first:stimulation_condition:C(average_n_bin)',
                 'vertex, high')}.items()}
    x = {k: term(n, l) for k, (n, l) in {
        '0': ('x', None), 'rf': ('x:risky_first', None),
        'v': ('x:stimulation_condition', 'vertex'),
        'h': ('x:C(average_n_bin)', 'high'),
        'rfv': ('x:risky_first:stimulation_condition', 'vertex'),
        'rfh': ('x:risky_first:C(average_n_bin)', 'high'),
        'vh': ('x:stimulation_condition:C(average_n_bin)', 'vertex, high'),
        'rfvh': ('x:risky_first:stimulation_condition:C(average_n_bin)',
                 'vertex, high')}.items()}

    def combine(t, rf, v, h):
        return (t['0'] + rf * t['rf'] + v * t['v'] + h * t['h']
                + rf * v * t['rfv'] + rf * h * t['rfh'] + v * h * t['vh']
                + rf * v * h * t['rfvh'])

    out = {}
    for order, rf in [('Risky first', 1), ('Risky second', 0)]:
        for stake, h in [('Low stake', 0), ('High stake', 1)]:
            for stim, v in [('vertex', 1), ('ips', 0)]:
                out[(order, stake, stim)] = (combine(i, rf, v, h),
                                             combine(x, rf, v, h))
    return out


def pfmt(d):
    p = min(float((d > 0).mean()), float((d < 0).mean()))
    return 'p < 0.001' if p < .001 else f'p = {p:.3f}'


def crosscheck(slopes, tsv_path):
    """The slope cells must match the numbers the manuscript text quotes."""
    ref = pd.read_csv(tsv_path, sep='\t')
    name = {'Risky second (safe first)': 'Risky second', 'Risky first': 'Risky first'}
    for _, r in ref.iterrows():
        cell = (name[r['order']], f"{r['stake'].capitalize()} stake")
        for stim in ['vertex', 'ips']:
            got = slopes[cell + (stim,)].mean()
            assert abs(got - r[f'{stim}_mean']) < 1e-6, (cell, stim, got)
    print(f'cross-check against {tsv_path}: OK')


def forest(ax, draws, xlim, ticks, ref=None, label=False):
    if ref is not None:
        ax.axvline(ref, color='.75', lw=.7, ls='--', zorder=0)
    for y, stim, colr, name in [(.68, 'vertex', VERTEX, 'Vertex'),
                                (.30, 'ips', IPS, 'IPS')]:
        d = draws[stim]
        lo, hi = np.quantile(d, [.025, .975])
        ax.plot([lo, hi], [y, y], color=colr, lw=1.7, solid_capstyle='round',
                zorder=3)
        ax.plot([d.mean()], [y], 'o', ms=4.2, color=colr, mec='white', mew=.8,
                zorder=4)
        if label:
            ax.text(.99, y, name, transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=7.5, color=colr)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    ax.spines['left'].set_visible(False)


def density(ax, d, grid, scale, ticks):
    lo, hi = np.quantile(d, [.025, .975])
    # the zero reference stops at the density ceiling so the p-value strip above
    # stays clear of ink
    ax.plot([0, 0], [-.34, 1.0], color='.6', lw=.7, ls='--', zorder=1)
    dens = ss.gaussian_kde(d)(grid) / scale
    ax.fill_between(grid, 0, dens, color=DIFF, alpha=.3, lw=0, zorder=2)
    ax.plot(grid, np.where(dens > .01, dens, np.nan), color=DIFF, lw=.9, zorder=3)
    ax.plot([lo, hi], [-.22, -.22], color=DIFF, lw=1.7, solid_capstyle='round',
            zorder=4)
    ax.plot([d.mean()], [-.22], 'o', ms=4.2, color=DIFF, mec='white', mew=.8,
            zorder=5)
    ax.axhline(0, color='.75', lw=.6, zorder=1)
    # p-value on whichever side the posterior mass leaves empty, in the strip
    # above the density/reference-line ceiling
    frac = (d.mean() - grid[0]) / (grid[-1] - grid[0])
    x, ha = (.02, 'left') if frac > .55 else (.98, 'right')
    ax.text(x, .99, pfmt(d), transform=ax.transAxes, ha=ha, va='top',
            fontsize=8, color='.15')
    ax.set_xlim(grid[0], grid[-1])
    ax.set_ylim(-.60, 1.34)
    ax.set_yticks([])
    ax.set_xticks(ticks)
    if ticks == [-.05, 0., .05, .10]:
        ax.set_xticklabels(['−.05', '0', '.05', '.10'])
    for side in ('left', 'top', 'right'):
        ax.spines[side].set_visible(False)


def main(trace_path, out_stem, check_tsv):
    post = az.from_netcdf(trace_path).posterior.stack(sample=('chain', 'draw'))
    beta = cell_draws(post)
    rnp = {k: np.exp(b0 / b1) for k, (b0, b1) in beta.items()}
    slope = {k: b1 for k, (b0, b1) in beta.items()}
    if check_tsv:
        crosscheck(slope, check_tsv)

    COLS = [
        dict(kind='forest', vals=rnp, title='Indifference point',
             xlim=(.36, .66), ticks=[.4, .5, .6], ref=.55,
             xlabel='Risk-neutral probability'),
        dict(kind='density', vals=rnp, title='Δ RNP',
             ticks=[-.05, 0., .05, .10], xlabel='Δ RNP (IPS − vertex)',
             anchors=('Risk-averse', 'Risk-seeking')),
        dict(kind='forest', vals=slope, title='Consistency',
             xlim=(1.15, 3.5), ticks=[1.5, 2.5, 3.5], ref=None,
             xlabel='Psychometric slope'),
        dict(kind='density', vals=slope, title='Δ Slope',
             ticks=[-1., -.5, 0., .5], xlabel='Δ Slope (IPS − vertex)',
             anchors=('Less consistent', 'More consistent')),
    ]

    H = 4.4
    fig = plt.figure(figsize=(7.25, H))
    # densities get the wider columns: their tick labels and anchor texts need
    # the room, the forests do not
    outer = fig.add_gridspec(1, 4, width_ratios=[.85, 1.15, .85, 1.15],
                             wspace=.24, left=.135, right=.985, top=1 - .30 / H,
                             bottom=.52 / H)
    axs = {}
    for ci in range(4):
        # a larger gap between the two presentation orders than between the two
        # stake bands inside each -- the grouping is the point
        gs = outer[ci].subgridspec(2, 1, hspace=.13)
        for gi, order in enumerate(['Risky first', 'Risky second']):
            sub = gs[gi].subgridspec(2, 1, hspace=.16)
            for si, stake in enumerate(['Low stake', 'High stake']):
                axs[(ci, order, stake)] = fig.add_subplot(sub[si])

    stats = []
    for ci, col in enumerate(COLS):
        if col['kind'] == 'density':
            deltas = {c: col['vals'][c + ('ips',)] - col['vals'][c + ('vertex',)]
                      for c in CELLS}
            allv = np.concatenate(list(deltas.values()))
            pad = .07 * np.ptp(allv)
            grid = np.linspace(allv.min() - pad, allv.max() + pad, 512)
            scale = max(ss.gaussian_kde(d)(grid).max() for d in deltas.values())

        for cell in CELLS:
            ax = axs[(ci,) + cell]
            if col['kind'] == 'forest':
                draws = {s: col['vals'][cell + (s,)] for s in ['vertex', 'ips']}
                forest(ax, draws, col['xlim'], col['ticks'], ref=col['ref'],
                       label=(ci == 0 and cell == CELLS[0]))
            else:
                d = deltas[cell]
                density(ax, d, grid, scale, col['ticks'])
                stats.append(dict(column=col['title'], order=cell[0],
                                  stake=cell[1], mean=d.mean(),
                                  lo=np.quantile(d, .025),
                                  hi=np.quantile(d, .975), p=pfmt(d)))
            if cell == CELLS[-1]:
                sns.despine(ax=ax, left=True, offset={'bottom': 3})
                ax.set_xlabel(col['xlabel'], fontsize=8.5)
                if col['kind'] == 'density':
                    for x, lab, ha in zip((.02, .98), col['anchors'],
                                          ('left', 'right')):
                        ax.text(x, .02, lab, transform=ax.transAxes, ha=ha,
                                va='bottom', fontsize=7, color='.35',
                                style='italic')
            else:
                ax.spines['bottom'].set_visible(False)
                ax.set_xticklabels([])
                ax.tick_params(axis='x', length=0)
        if COLS[ci]['kind'] == 'forest' and COLS[ci]['ref'] is not None:
            ax0 = axs[(ci,) + CELLS[0]]
            ax0.text(col['ref'], 1.02, 'Risk-neutral', fontsize=7, color='.45',
                     ha='center', va='bottom', style='italic')

    for cell in CELLS:
        pos = axs[(0,) + cell].get_position()
        fig.text(.004, (pos.y0 + pos.y1) / 2, f'{cell[0]}\n{cell[1]}',
                 fontsize=9, ha='left', va='center', linespacing=1.3,
                 color='.15')

    y = axs[(0,) + CELLS[0]].get_position().y1 + .035
    for letter, ci in zip('ABCD', range(4)):
        pos = axs[(ci,) + CELLS[0]].get_position()
        fig.text(pos.x0 - .022, y, letter, fontsize=11.5, va='baseline',
                 ha='right', **BOLD)
        fig.text((pos.x0 + pos.x1) / 2, y, COLS[ci]['title'],
                 fontsize=9, ha='center', va='baseline', **BOLD)

    fig.text(.004, -.028, 'Random effects: intercept only, as in the published '
             'fit · dots and lines: posterior mean and 95% CrI',
             fontsize=7.5, color='.45', style='italic', ha='left', va='bottom')

    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)

    print(f'wrote {out_stem}.pdf\n')
    print(f'{"column":<26}{"order":<14}{"stake":<12}{"Δ":>9}{"95% CrI":>20}')
    for s in stats:
        print(f'{s["column"]:<26}{s["order"]:<14}{s["stake"]:<12}'
              f'{s["mean"]:>+9.4f}   [{s["lo"]:+.4f}, {s["hi"]:+.4f}]   {s["p"]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--trace', default='/data/ds-tmsrisk/derivatives/cogmodels/'
                        'model-probit_average_n_full_trace.netcdf')
    parser.add_argument('--out', default='/Users/gdehol/git/tms_risk/notes/figures/'
                        'supp_fig1_stake')
    parser.add_argument('--check_tsv', default='/Users/gdehol/git/tms_risk/notes/'
                        'data/probit_stake_cells_published.tsv')
    a = parser.parse_args()
    main(a.trace, a.out, a.check_tsv)
