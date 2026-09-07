"""Pooled 'proportion of voxels won' for the cvR² model comparison.

Companion to ``compare_cvr2_models.py``. That script shows the *per-subject*
win-proportion swarm; here we pool every voxel across subjects and report
what fraction of the ROI voxel pool each model wins — a stacked bar per ROI
whose segments sum to 100%.

Two panels:
- **Primary (left)** — null voxels dropped first, then the *relative*
  proportions of the three real encoding models {m0, m1, m2} among the
  signal voxels (renormalised to sum to 100%). This is the comparison we
  actually care about: given a voxel that *some* model explains, which
  variant explains it best.
- **Secondary (right)** — the full pool including null, so the signal-vs-
  noise fraction stays visible.

Pooled (count-weighted) rather than mean-of-per-subject-proportions: a
subject with more voxels contributes proportionally more, so the bar reads
as "of all voxels in this ROI, X% are best fit by m2". Reads the TSV that
``compare_cvr2_models.py`` already dumped — no raw-data reload.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from tms_risk.modeling.scripts.plot_spherical_expected_uncertainty import apply_style


ORDER = ['null', 'm0', 'm1', 'm2']
PALETTE = {'null': '#9C9C9C', 'm0': '#7F7F7F', 'm1': '#3B5BA5', 'm2': '#C44E52'}
LABELS = {'null': 'Null (train mean)', 'm0': 'm0 (pooled)',
          'm1': 'm1 (amp per ses)', 'm2': 'm2 (full per ses)'}


def pooled_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Count-weighted win proportion per (roi, model).

    ``win_prop`` is per-subject won/total, so ``win_prop * n_total`` recovers
    the won-voxel count; summing those over subjects and dividing by the
    total voxel count gives the pooled proportion.
    """
    df = df.copy()
    df['n_won'] = df['win_prop'] * df['n_total']
    agg = df.groupby(['roi', 'model'])[['n_won', 'n_total']].sum()
    agg['pooled_prop'] = agg['n_won'] / agg['n_total']
    return agg.reset_index()


def _stacked(ax, wide, models, title, ylabel):
    """Draw one stacked-bar panel: rows of `wide` = ROIs, `models` = ordered
    list of columns to stack."""
    import seaborn as sns
    bottom = pd.Series(0.0, index=wide.index)
    x = range(len(wide.index))
    for model in models:
        vals = wide[model]
        ax.bar(x, vals, bottom=bottom, width=0.7, color=PALETTE[model],
               edgecolor='white', linewidth=0.6, label=LABELS[model])
        for xi, (v, b) in enumerate(zip(vals, bottom)):
            if v >= 0.04:
                ax.text(xi, b + v / 2, f'{v*100:.0f}%', ha='center',
                        va='center', fontsize=7.5,
                        color='white' if model != 'null' else '0.2')
        bottom += vals
    ax.set_xticks(list(x))
    ax.set_xticklabels(wide.index, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    sns.despine(ax=ax, offset=3, trim=False)


def main(tsv='notes/data/cvr2_model_comparison.tsv',
         out='notes/figures/cvr2_voxels_won'):
    apply_style()
    # keep_default_na=False: otherwise the literal model label "null" is
    # parsed as NaN and dropped from every groupby/pivot.
    df = pd.read_csv(tsv, sep='\t', keep_default_na=False)
    df['win_prop'] = df['win_prop'].astype(float)
    df['n_total'] = df['n_total'].astype(int)
    pooled = pooled_proportions(df)

    rois = [r for r in df['roi'].unique()]
    # Full pool incl. null (cols sum to 1 over {null, m0, m1, m2})
    wide_all = (pooled.pivot(index='roi', columns='model', values='pooled_prop')
                      .reindex(index=rois, columns=ORDER))
    # Signal-only: drop null, renormalise the three real models to sum to 1
    real = ['m0', 'm1', 'm2']
    wide_signal = wide_all[real].div(wide_all[real].sum(axis=1), axis=0)

    print('\nPooled proportion of voxels won — full pool incl. null (%):')
    print((wide_all * 100).round(1))
    print('\nSignal voxels only (null dropped), relative m0/m1/m2 (%):')
    print((wide_signal * 100).round(1))

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(2.0 * len(rois) + 3.0, 3.8), constrained_layout=True)

    # Primary: signal-only relative proportions
    _stacked(axL, wide_signal, real,
             'Signal voxels only — relative win share\n(null dropped, m0/m1/m2 renormalised)',
             'Relative proportion among signal voxels')
    # Secondary: full pool incl. null
    _stacked(axR, wide_all, ORDER,
             'All voxels — incl. null\n(null share = noise fraction)',
             'Proportion of all voxels')

    handles, labels = axR.get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5),
               frameon=False, fontsize=8, title='Winner', title_fontsize=8)

    n_sub = df['subject'].nunique()
    fig.suptitle('Proportion of voxels won by each nPRF model (pooled across subjects)',
                 fontsize=11, y=1.05)
    fig.text(0.5, -0.04,
             f'n_subjects = {n_sub}  ·  voxels pooled (count-weighted) across subjects within each ROI  '
             f'·  null = per-voxel training-mean predictor (cvR² = 0 by construction)',
             ha='center', va='top', fontsize=7.5, color='0.4')

    out_root = Path(out)
    out_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_root.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out_root.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'wrote {out_root}.{{pdf,png}}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--tsv', default='notes/data/cvr2_model_comparison.tsv')
    p.add_argument('--out', default='notes/figures/cvr2_voxels_won')
    args = p.parse_args()
    main(tsv=args.tsv, out=args.out)
