"""cvR² model comparison for the three nPRF encoding-model variants
(m0 / m1 / m2). See notes/analyses/cvr2_model_comparison.md for the
analysis design.

For every subject and every voxel in the requested ROI we load cvR²
from `encoding_model2.model-{0,1,2}.smoothed.cv/`. Voxels where none of
the three variants beats the null (cvR² > 0) are dropped. On the
surviving "non-noise" pool we identify which variant wins per voxel and
report per-subject win proportions.

Headline figure: per-subject swarm + group mean ± SEM of win
proportions, faceted by ROI. Also dumps the underlying TSV so the
proportions can be re-aggregated however we like later.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tms_risk.utils.data import Subject
from tms_risk.modeling.scripts.plot_spherical_expected_uncertainty import apply_style


# Subject set: TMS cohort ∩ PRF-fits-on-disk
SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31,
            34, 35, 36, 37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
# Subject.get_volume_mask only supports NPC* / NF* / NTO* masks (the
# parietal/numerosity-related ROIs hand-defined for this study).
# V1 / V2 / control ROIs aren't on disk here, so we restrict to those.
ROIS = ['NPC12r', 'NPCl', 'NPCr', 'NF1', 'NTO']
MODELS = (0, 1, 2)
# The null model = "predict the per-voxel training-set mean for every test
# trial". Its cvR² on a held-out fold is ≈ 0 by definition. We carry it as
# an explicit 4th column so the winner-counting is between
# {null, m0, m1, m2}: voxels where null is highest = noise voxels (no
# encoding model beats the constant predictor).
NULL_CVR2 = 0.0


def load_cvr2(subject: int, model_label: int, roi: str,
              session: int | None = None,
              bids_folder: str = '/data/ds-tmsrisk') -> pd.Series:
    """cvR² per voxel for one (subject, model, roi[, session]).
    Returns a Series indexed by voxel; NaNs propagated.

    `session=None` → use the global per-voxel cvR² that
    `Subject.get_prf_parameters` returns with `NaN` session label
    (it's session-agnostic — computed across all available data).
    `session=<int>` → use that session's cvR² directly.
    """
    sub = Subject(subject, bids_folder=bids_folder)
    pars = sub.get_prf_parameters(model_label=model_label, session=session, roi=roi)
    if session is None:
        # MultiIndex columns: ('cvr2', NaN) is the global one. The
        # per-session columns share the same cvR² (it's session-agnostic
        # in the regression PRF), so picking the first is also fine.
        cvr2_block = pars['cvr2']
        if isinstance(cvr2_block, pd.DataFrame):
            cvr2_series = cvr2_block.iloc[:, 0]
        else:
            cvr2_series = cvr2_block
    else:
        cvr2_series = pars['cvr2']
    return pd.Series(cvr2_series.values, index=cvr2_series.index,
                     name=f'm{model_label}')


def main(rois=ROIS, subjects=SUBJECTS, bids_folder='/data/ds-tmsrisk',
         session_label: str = 'global'):
    """For every voxel, compute the winner among {null, m0, m1, m2} by
    cvR². null has cvR² = 0 by construction (predict per-voxel
    training-mean). Per subject, aggregate win proportions over ALL
    voxels in the ROI (no pre-filtering — the 'null wins' fraction is
    the noise-voxel fraction)."""
    apply_style()
    sess = None if session_label == 'global' else int(session_label)

    rows = []
    per_subj_voxels = []
    for roi in rois:
        for sid in subjects:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    cv = pd.concat([load_cvr2(sid, m, roi, session=sess,
                                              bids_folder=bids_folder)
                                    for m in MODELS], axis=1)
            except FileNotFoundError as e:
                print(f'  skip sub-{sid:02d} {roi} ({e})')
                continue
            # Explicit null column = 0 everywhere. Winner per voxel includes
            # null as a candidate; voxels where null wins ≡ noise voxels.
            cv = cv.assign(null=NULL_CVR2)
            cv = cv[['null', 'm0', 'm1', 'm2']]   # column order for plotting
            winner = cv.idxmax(axis=1)
            counts = winner.value_counts() / len(winner)
            for label in ['null', 'm0', 'm1', 'm2']:
                rows.append({
                    'roi':      roi,
                    'subject':  sid,
                    'model':    label,
                    'win_prop': float(counts.get(label, 0.0)),
                    'n_total':  len(cv),
                })
            per_subj_voxels.append({
                'roi':            roi,
                'subject':        sid,
                'n_voxels':       len(cv),
                'frac_signal':    float(1 - counts.get('null', 0.0)),
            })

    df = pd.DataFrame(rows)
    diag = pd.DataFrame(per_subj_voxels)
    print('\nVoxel pool per (subject, ROI):')
    print(diag.groupby('roi')[['n_voxels', 'frac_signal']]
              .agg(['mean', 'median', 'min', 'max']).round(2))

    print('\nGroup-mean win proportions by ROI:')
    print(df.groupby(['roi', 'model'])['win_prop'].agg(['mean', 'sem', 'count']).round(3))

    # ── Plot ────────────────────────────────────────────────────────────────
    n_roi = len(rois)
    fig, axes = plt.subplots(1, n_roi, figsize=(2.4 * n_roi + 0.5, 3.0),
                              sharey=True, constrained_layout=True)
    if n_roi == 1:
        axes = [axes]

    order = ['null', 'm0', 'm1', 'm2']
    palette = {'null': '#9C9C9C', 'm0': '#7F7F7F', 'm1': '#3B5BA5', 'm2': '#C44E52'}
    model_label_map = {'null': 'null\n(train mean)',
                        'm0':   'm0\n(pooled)',
                        'm1':   'm1\n(amp per ses)',
                        'm2':   'm2\n(full per ses)'}

    for ax, roi in zip(axes, rois):
        sub = df[df.roi == roi]
        if len(sub) == 0:
            ax.set_title(f'{roi}\n(no data)')
            continue
        sns.stripplot(
            data=sub, x='model', y='win_prop', ax=ax,
            order=order, palette=palette, size=3.5, alpha=0.6,
            jitter=0.18, edgecolor='none', zorder=2,
        )
        grp = sub.groupby('model')['win_prop'].agg(['mean', 'sem']).reset_index()
        for _, row in grp.iterrows():
            xpos = order.index(row['model'])
            ax.errorbar(xpos, row['mean'], yerr=row['sem'],
                         marker='D', markersize=8, mew=1.5,
                         color='black', mfc=palette[row['model']],
                         capsize=3, zorder=5, lw=1.5)
        ax.axhline(0.25, ls=':', color='0.6', lw=0.6, zorder=0)
        ax.set_xticklabels([model_label_map[m] for m in order], fontsize=7.5)
        ax.set_xlabel('')
        ax.set_ylabel('Voxel-win proportion'
                       if ax is axes[0] else '')
        # Per-ROI signal fraction = 1 − P(null wins)
        signal_frac = 1 - grp[grp.model == 'null']['mean'].iloc[0]
        ax.set_title(f"{roi}\nsignal voxels: {signal_frac*100:.0f}%",
                      fontsize=9)
        ax.set_ylim(0, 1)
        sns.despine(ax=ax, offset=4, trim=True)

    fig.suptitle('cvR² model comparison vs null (training-mean predictor)',
                  fontsize=11, y=1.04)
    fig.text(0.5, -0.04,
              f'session = {session_label}  ·  n_subjects = {df["subject"].nunique()}  '
              f'·  null = predict per-voxel training-set mean (cvR² ≡ 0)  '
              f'·  diamonds = group mean ± SEM',
              ha='center', va='top', fontsize=7.5, color='0.4')

    out_root = Path('notes/figures/cvr2_model_comparison')
    out_root.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_root.with_suffix('.pdf'))
    fig.savefig(out_root.with_suffix('.png'), dpi=200)
    print(f'wrote {out_root}.{{pdf,png}}')

    out_tsv = Path('notes/data/cvr2_model_comparison.tsv')
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--rois', nargs='+', default=ROIS)
    p.add_argument('--session', default='global',
                    help='global | 1 | 2 | 3')
    args = p.parse_args()
    main(rois=args.rois, session_label=args.session)
