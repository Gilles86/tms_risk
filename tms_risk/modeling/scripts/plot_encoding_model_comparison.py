"""Encoding-model comparison across the m0-m5 set (candidate supplementary figure).

Everything relevant is on the page, including what a naive plot would hide: the properly
computed null (cvR2 = 0 is NOT the null -- braincoder's get_rsq uses the held-out fold's
own mean, so the real null sits near -0.018; see cvr2_vs_null.py), per-subject spread,
and the paired contrast against the canonical model.

    A  cvR2 per model, per subject, as a difference from the null.
    B  Fraction of ROI voxels where each model beats the null, per subject.
    C  Paired within-subject difference against m1, the canonical model. Models are
       fitted on identical data and folds, so the paired contrast is the honest
       comparison -- between-subject variance in cvR2 dwarfs the model differences.
    D  (only if `frac_wins` present) Voxel-level winner among signal voxels.

    python -m tms_risk.modeling.scripts.plot_encoding_model_comparison --roi NPCr2cm-cluster

Reads notes/data/cvr2_model_grid.tsv (long format: roi, model, subject, cvr2, null,
frac_beats_null[, frac_wins]), currently derived from notes/data/cvr2_vs_null_m0-5.tsv.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

NAMES = {0: 'None', 1: 'Amplitude', 2: 'All four', 3: 'Amplitude + σ',
         4: 'μ + σ (tuning)', 5: 'Amplitude + baseline'}
# tuning models cool, response-magnitude models warm, extremes neutral
COLORS = {0: '#9C9C9C', 1: '#C44E52', 2: '#4d4d4d', 3: '#8172B2',
          4: '#3B5BA5', 5: '#D1885C'}
CANONICAL = 1

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 8, 'legend.fontsize': 7.5,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def strip_mean(ax, x, vals, color, width=.28):
    """Per-subject points plus a fat mean marker with SEM -- the house idiom."""
    jit = (np.random.RandomState(0).rand(len(vals)) - .5) * width
    ax.scatter(x + jit, vals, s=11, color=color, alpha=.38, lw=0, zorder=2)
    m, se = np.nanmean(vals), stats.sem(vals, nan_policy='omit')
    ax.errorbar(x, m, yerr=se, fmt='D', ms=7, color=color, mec='0.15', mew=1.4,
                elinewidth=1.4, capsize=0, zorder=4)
    return m


def model_ticks(ax, models):
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([NAMES.get(m, f'm{m}') for m in models],
                       rotation=35, ha='right', rotation_mode='anchor')
    for lab, m in zip(ax.get_xticklabels(), models):
        if m == CANONICAL:
            lab.set_color(COLORS[CANONICAL])


def main(data_dir, roi, out_stem):
    d = pd.read_csv(Path(data_dir) / 'cvr2_model_grid.tsv', sep='\t')
    d = d[d.roi == roi]
    if not len(d):
        raise SystemExit(f'no rows for roi {roi}')
    models = sorted(d.model.unique())
    n = d.subject.nunique()
    has_wins = 'frac_wins' in d.columns

    n_panels = 4 if has_wins else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(2.42 * n_panels, 2.9),
                             constrained_layout=True)

    # ---------------------------------------------------------- A: cvR2 minus null
    ax = axes[0]
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    for i, m in enumerate(models):
        g = d[d.model == m]
        strip_mean(ax, i, (g.cvr2 - g.null).values, COLORS.get(m, '.4'))
    tick_jobs = [(ax, models)]
    ax.set_title('Out-of-sample fit', fontsize=9.5)
    ax.set_ylabel('cvR² − null')
    ax.text(0.01, 0.0, 'Null', transform=ax.get_yaxis_transform(), fontsize=7,
            color='.45', va='bottom', ha='left')

    # ------------------------------------------------- B: fraction beating the null
    ax = axes[1]
    ax.axhline(.5, color='.6', lw=.8, ls=':', zorder=0)
    for i, m in enumerate(models):
        strip_mean(ax, i, d[d.model == m].frac_beats_null.values, COLORS.get(m, '.4'))
    tick_jobs.append((ax, models))
    ax.set_title('Voxels beating the null', fontsize=9.5)
    ax.set_ylabel('Fraction of voxels')

    # --------------------------------------------------- C: paired contrast vs m1
    ax = axes[2]
    ax.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    w = d.pivot_table(index='subject', columns='model', values='cvr2')
    ref = CANONICAL if CANONICAL in w.columns else models[0]
    others = [m for m in models if m != ref]
    for i, m in enumerate(others):
        diff = (w[m] - w[ref]).dropna().values
        strip_mean(ax, i, diff, COLORS.get(m, '.4'))
        t, p = stats.ttest_1samp(diff, 0)
        ptxt = f'{p:.3f}'.lstrip('0') if p >= .001 else '<.001'
        ax.text(i, 1.01, f'p {ptxt}', transform=ax.get_xaxis_transform(),
                ha='center', va='bottom', fontsize=6.5, color='.35')
    tick_jobs.append((ax, others))
    ax.set_title('Versus the canonical model', fontsize=9.5, pad=16)
    ax.set_ylabel('Δ cvR² (model − m1)')

    # ------------------------------------------------------- D: voxel-level winner
    if has_wins:
        ax = axes[3]
        for i, m in enumerate(models):
            strip_mean(ax, i, d[d.model == m].frac_wins.values, COLORS.get(m, '.4'))
        ax.axhline(1 / len(models), color='.6', lw=.8, ls=':', zorder=0)
        tick_jobs.append((ax, models))
        ax.set_title('Voxel-level winner', fontsize=9.5)
        ax.set_ylabel('Fraction of signal voxels won')

    for a, letter in zip(axes, 'ABCD'):
        a.text(-0.22, 1.04, letter, transform=a.transAxes, fontsize=12,
               fontweight='bold', va='bottom', ha='right', family='Arial')
    fig.supxlabel('Parameters free per session (IPS vs vertex)', fontsize=9)
    # despine BEFORE tick styling: spine.set_position() resets tick objects, which
    # keeps the label text (formatter) but silently drops rotation/color overrides
    sns.despine(fig=fig, offset=3)
    for a, ms in tick_jobs:
        model_ticks(a, ms)
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
