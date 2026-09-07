"""Plot per-session PRF-parameter distributions for the three encoding-model
variants, restricted to signal voxels.

For each (subject, ROI, model, session) we load μ, σ, amplitude, baseline
within the ROI, then keep only "signal voxels" — those where this
model's cvR² > 0 (i.e. beats the per-voxel training-mean null on this
session). For each model, plot the per-subject across-voxel mean of each
parameter, with session on x. The visual hierarchy of what each model
*allows* to change is:

  m0 = pooled — parameters are forced identical across sessions (so the
       per-session means should be approximately equal across sessions,
       reflecting only signal-voxel selection differences)
  m1 = amplitude varies per session — μ, σ, baseline are pooled;
       amplitude is the only one that's free to shift session-to-session
  m2 = full per-session — every parameter can shift across sessions

If the per-session μ / σ / baseline distributions are visibly different
for m2 but not for m1, that's direct evidence that m2 is fitting real
across-session retuning that m1 misses.
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


SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31,
            34, 35, 36, 37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
ROI = 'NPC12r'
PARAMS = ['mu', 'sd', 'amplitude', 'baseline']
MODELS = (0, 1, 2)
# Sessions 2 and 3 are the TMS sessions with per-session (μ, σ, amp, base)
# refits; session 1 is the PRF-mapping baseline and only has r²/cvR² on
# disk (no per-session parameters to plot).
SESSIONS = (2, 3)


def collect(rois=(ROI,), bids_folder='/data/ds-tmsrisk'):
    rows = []
    for roi in rois:
        for sid in SUBJECTS:
            sub = Subject(sid, bids_folder=bids_folder)
            for m in MODELS:
                for ses in SESSIONS:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter('ignore')
                            pars = sub.get_prf_parameters(
                                model_label=m, session=ses, roi=roi
                            )
                    except FileNotFoundError:
                        continue
                    # Need all four PRF parameter columns. Session 1 only
                    # has r²/cvR² on disk, so skip if any param is missing.
                    if any(p not in pars.columns for p in PARAMS):
                        continue
                    signal = pars[pars['cvr2'] > 0]
                    if len(signal) == 0:
                        continue
                    rows.append({
                        'subject': sid, 'roi': roi, 'model': f'm{m}',
                        'session': ses, 'n_signal': len(signal),
                        'n_total': len(pars),
                        **{p: float(signal[p].mean()) for p in PARAMS},
                    })
    return pd.DataFrame(rows)


def main(bids_folder='/data/ds-tmsrisk'):
    apply_style()
    df = collect(bids_folder=bids_folder)
    print(f'collected {len(df)} (subject × model × session) rows')
    print(df.groupby(['model', 'session'])['n_signal']
            .agg(['mean', 'median', 'count']).round(1))

    # Figure: 4 rows (parameters) × 3 columns (models)
    fig, axes = plt.subplots(len(PARAMS), len(MODELS),
                              figsize=(8.0, 8.0),
                              sharex='col', constrained_layout=True)

    color_per_model = {'m0': '#7F7F7F', 'm1': '#3B5BA5', 'm2': '#C44E52'}

    # Per-parameter y-limits: shared across all three model columns so they
    # are directly comparable, scaled to the group-mean ± SEM band rather
    # than the across-subject spread (which can dominate and squash the
    # signal). Add a healthy margin so most subject lines stay on-panel.
    par_ylims = {}
    for par in PARAMS:
        grp_all = (df.groupby(['model', 'session'])[par]
                     .agg(['mean', 'sem']).reset_index())
        lo = (grp_all['mean'] - 5 * grp_all['sem']).min()
        hi = (grp_all['mean'] + 5 * grp_all['sem']).max()
        # Pad ±10% of the band
        pad = (hi - lo) * 0.1
        par_ylims[par] = (lo - pad, hi + pad)

    for col, m in enumerate(MODELS):
        mtag = f'm{m}'
        for row, par in enumerate(PARAMS):
            ax = axes[row, col]
            sub = df[df['model'] == mtag]
            # Per-subject line connecting sessions
            for sid, sd in sub.groupby('subject'):
                sd = sd.sort_values('session')
                ax.plot(sd['session'], sd[par], color=color_per_model[mtag],
                         lw=0.5, alpha=0.30, zorder=2)
            # Group mean ± SEM
            grp = sub.groupby('session')[par].agg(['mean', 'sem']).reset_index()
            ax.errorbar(grp['session'], grp['mean'], yerr=grp['sem'],
                         color='black', mfc=color_per_model[mtag],
                         marker='D', markersize=7, mew=1.4, capsize=3,
                         lw=1.3, zorder=5)
            if row == 0:
                ax.set_title({'m0': 'm0 (pooled)',
                              'm1': 'm1 (amp per ses)',
                              'm2': 'm2 (full per ses)'}[mtag],
                              fontsize=10, color=color_per_model[mtag])
            if col == 0:
                ax.set_ylabel({'mu': 'μ (log n)',
                                'sd': 'σ',
                                'amplitude': 'Amplitude',
                                'baseline': 'Baseline'}[par])
            if row == len(PARAMS) - 1:
                ax.set_xlabel('Session')
                ax.set_xticks(list(SESSIONS))
            ax.set_ylim(*par_ylims[par])
            sns.despine(ax=ax, offset=3, trim=True)

    fig.suptitle(f'Per-session PRF parameter means in {ROI} '
                  f'(signal voxels, cvR² > 0)',
                  fontsize=11, y=1.02)
    fig.text(0.5, -0.02,
              f'each thin line = one subject  ·  diamonds = group mean ± SEM  ·  '
              f'n_subjects = {df["subject"].nunique()}',
              ha='center', va='top', fontsize=8, color='0.4')

    out_root = Path('notes/figures/prf_params_per_session')
    fig.savefig(out_root.with_suffix('.pdf'))
    fig.savefig(out_root.with_suffix('.png'), dpi=200)
    print(f'wrote {out_root}.{{pdf,png}}')

    out_tsv = Path('notes/data/prf_params_per_session.tsv')
    df.to_csv(out_tsv, sep='\t', index=False)
    print(f'wrote {out_tsv}')


if __name__ == '__main__':
    main()
