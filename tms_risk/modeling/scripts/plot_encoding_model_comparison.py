"""Diagnostic figure 1: which encoding model actually works best?

Not a paper figure -- everything relevant is on the page, including the things a paper
figure would hide: the null line, per-subject spread, and how often each model wins
voxel-by-voxel.

    a  cvR2 per model, per subject, as a DIFFERENCE FROM THE NULL. cvR2 = 0 is not the
       null: braincoder's get_rsq uses the held-out fold's own mean, so 0 assumes
       knowledge a real null predictor lacks (see null_cvr2.py -- the null sits near
       -0.018). Plotting the difference makes "does this model do anything at all"
       readable, which raw negative cvR2 does not.
    b  Fraction of voxels where each model beats the null, per subject.
    c  Paired within-subject difference against m1, the incumbent. The models are fitted
       on identical data and folds, so the paired contrast is the honest comparison --
       between-subject variance in cvR2 dwarfs the differences between models.
    d  Voxel-level winner: of the voxels where ANY model beats the null, the fraction
       won by each. Answers "is the average hiding a mixture?".

    python -m tms_risk.modeling.scripts.plot_encoding_model_comparison --roi NPCr2cm-cluster

Reads notes/data/cvr2_model_grid.tsv, written by `extract_cvr2_model_grid.py`.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

NAMES = {0: 'm0\nnothing', 1: 'm1\namplitude', 2: 'm2\nall four',
         3: 'm3\namp + sd', 4: 'm4\nmu + sd\n(tuning)',
         5: 'm5\namp + base\n(response)'}
# tuning models cool, response-magnitude models warm, extremes neutral
COLORS = {0: '#9C9C9C', 1: '#C44E52', 2: '#4d4d4d', 3: '#8172B2',
          4: '#3B5BA5', 5: '#D1885C'}

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def strip_mean(ax, x, vals, color, width=.28):
    """Per-subject points plus a fat mean marker with SEM -- the house idiom."""
    jit = (np.random.RandomState(0).rand(len(vals)) - .5) * width
    ax.scatter(x + jit, vals, s=11, color=color, alpha=.38, lw=0, zorder=2)
    m, se = np.nanmean(vals), stats.sem(vals, nan_policy='omit')
    ax.errorbar(x, m, yerr=se, fmt='D', ms=7, color=color, mec='0.15', mew=1.4,
                elinewidth=1.4, capsize=0, zorder=4)
    return m


def main(data_dir, roi, out_stem):
    d = pd.read_csv(Path(data_dir) / 'cvr2_model_grid.tsv', sep='\t')
    d = d[d.roi == roi]
    if not len(d):
        raise SystemExit(f'no rows for roi {roi}')
    models = sorted(d.model.unique())
    n = d.subject.nunique()

    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1), constrained_layout=True)

    # ---------------------------------------------------------- a: cvR2 minus null
    ax = axes[0]
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    for i, m in enumerate(models):
        g = d[d.model == m]
        strip_mean(ax, i, (g.cvr2 - g.null).values, COLORS.get(m, '.4'))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES.get(m, f'm{m}') for m in models])
    ax.set_ylabel('cvR² − null')
    ax.text(0.02, 0.97, 'Null', transform=ax.transAxes, fontsize=7, color='.45',
            va='top')

    # ------------------------------------------------- b: fraction beating the null
    ax = axes[1]
    ax.axhline(.5, color='.6', lw=.8, ls=':', zorder=0)
    for i, m in enumerate(models):
        strip_mean(ax, i, d[d.model == m].frac_beats_null.values, COLORS.get(m, '.4'))
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES.get(m, f'm{m}') for m in models])
    ax.set_ylabel('Fraction of voxels beating null')

    # --------------------------------------------------- c: paired contrast vs m1
    ax = axes[2]
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    w = d.pivot_table(index='subject', columns='model', values='cvr2')
    ref = 1 if 1 in w.columns else models[0]
    others = [m for m in models if m != ref]
    for i, m in enumerate(others):
        diff = (w[m] - w[ref]).dropna().values
        strip_mean(ax, i, diff, COLORS.get(m, '.4'))
        t, p = stats.ttest_1samp(diff, 0)
        ax.text(i, ax.get_ylim()[1], f'p={p:.3f}', ha='center', va='bottom',
                fontsize=6.5, color='.35')
    ax.set_xticks(range(len(others)))
    ax.set_xticklabels([NAMES.get(m, f'm{m}') for m in others])
    ax.set_ylabel(f'cvR² − cvR²(m{ref}),  paired')

    # ------------------------------------------------------- d: voxel-level winner
    ax = axes[3]
    if 'frac_wins' in d.columns:
        for i, m in enumerate(models):
            strip_mean(ax, i, d[d.model == m].frac_wins.values, COLORS.get(m, '.4'))
        ax.axhline(1 / len(models), color='.6', lw=.8, ls=':', zorder=0)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([NAMES.get(m, f'm{m}') for m in models])
        ax.set_ylabel('Fraction of signal voxels won')
    else:
        ax.set_visible(False)

    for a, letter in zip(axes, 'abcd'):
        if a.get_visible():
            a.text(-0.18, 1.04, letter, transform=a.transAxes, fontsize=12,
                   fontweight='bold', va='bottom', ha='right')
    fig.suptitle(f'{roi} · n = {n} subjects · points are subjects, diamonds mean ± SEM',
                 fontsize=9)
    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')

    print(f'\n{roi}, n = {n}')
    for m in models:
        g = d[d.model == m]
        print(f'  m{m}: cvR2 {g.cvr2.mean():+.5f}  (null {g.null.mean():+.5f}, '
              f'diff {(g.cvr2-g.null).mean():+.5f})  beats null in '
              f'{100*g.frac_beats_null.mean():.1f}% of voxels')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--roi', default='NPCr2cm-cluster')
    p.add_argument('--out', default='notes/figures/encoding_model_comparison')
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.data_dir, a.roi, a.out)
