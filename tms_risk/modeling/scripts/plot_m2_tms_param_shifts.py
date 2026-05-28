"""Did IPS-TMS shift m2's per-session PRF parameters relative to vertex sham?

For every TMS subject we have an IPS-stimulation session and a vertex
(sham) session — either (ses-2, ses-3) or (ses-3, ses-2) depending on
per-subject randomization. m2 fits (μ, σ, amplitude, baseline) per
session. The natural question: do per-subject (IPS − Vertex)
differences in those four parameters depart from zero?

The within-subject paired difference removes the session-2 vs session-3
order confound. The group test is whether the parameter shift is
non-zero across subjects.

Restricted to NPC12r signal voxels (cvR² > 0 in both sessions under m2).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from tms_risk.utils.data import Subject, get_all_behavior
from tms_risk.modeling.scripts.plot_spherical_expected_uncertainty import apply_style


SUBJECTS = [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 18, 19, 21, 25, 26, 29, 30, 31,
            34, 35, 36, 37, 45, 46, 47, 50, 53, 56, 59, 62, 63, 67, 69, 72, 74]
ROI = 'NPC12r'
PARAMS = ['mu', 'sd', 'amplitude', 'baseline']
COLOR = {'ips': '#d62728', 'vertex': '#2ca02c'}


def collect(bids_folder='/data/ds-tmsrisk', model_label=2):
    """For each subject × session, per-voxel mean of each param in signal
    voxels. Joined with stimulation_condition."""
    beh = get_all_behavior(bids_folder=bids_folder)
    cond_map = (beh.reset_index()[['subject', 'session', 'stimulation_condition']]
                    .drop_duplicates())

    rows = []
    for sid in SUBJECTS:
        sub = Subject(sid, bids_folder=bids_folder)
        for ses in (2, 3):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    pars = sub.get_prf_parameters(model_label=model_label,
                                                   session=ses, roi=ROI)
            except FileNotFoundError:
                continue
            if any(p not in pars.columns for p in PARAMS):
                continue
            signal = pars[pars['cvr2'] > 0]
            if len(signal) == 0:
                continue
            rows.append({
                'subject': sid, 'session': ses, 'n_signal': len(signal),
                **{p: float(signal[p].mean()) for p in PARAMS},
            })
    df = pd.DataFrame(rows).merge(cond_map, on=['subject', 'session'], how='left')
    df = df[df['stimulation_condition'].isin(['ips', 'vertex'])].copy()
    return df


def main(model_label=2, bids_folder='/data/ds-tmsrisk'):
    apply_style()
    df = collect(bids_folder=bids_folder, model_label=model_label)
    print(f'collected {len(df)} (subject × condition) rows in {ROI}')
    print(df.groupby('stimulation_condition')['subject'].nunique())

    # Pivot to one row per subject with (ips, vertex) columns for each param
    pivots = {}
    for par in PARAMS:
        wide = df.pivot_table(index='subject', columns='stimulation_condition',
                                values=par, aggfunc='mean')
        wide = wide.dropna(subset=['ips', 'vertex'])
        wide['diff'] = wide['ips'] - wide['vertex']
        pivots[par] = wide
    n_paired = pivots[PARAMS[0]].shape[0]
    print(f'paired subjects (IPS + Vertex both present): {n_paired}')

    # ── Figure: two-panel row — per-condition swarm (left), paired diff (right) ──
    fig, axes = plt.subplots(1, len(PARAMS), figsize=(2.6 * len(PARAMS), 3.6),
                              constrained_layout=True)

    for ax, par in zip(axes, PARAMS):
        wide = pivots[par]
        # Per-subject paired difference, plus 0-line
        x_ips, x_vtx = 0, 1
        for _, row in wide.iterrows():
            ax.plot([x_ips, x_vtx], [row['ips'], row['vertex']],
                     color='0.6', lw=0.5, alpha=0.5, zorder=2)
            ax.scatter(x_ips, row['ips'], s=18, color=COLOR['ips'],
                        alpha=0.6, edgecolor='none', zorder=3)
            ax.scatter(x_vtx, row['vertex'], s=18, color=COLOR['vertex'],
                        alpha=0.6, edgecolor='none', zorder=3)
        # Group means with SEM bars
        m_ips, s_ips = wide['ips'].mean(), wide['ips'].sem()
        m_vtx, s_vtx = wide['vertex'].mean(), wide['vertex'].sem()
        ax.errorbar([x_ips], [m_ips], yerr=[s_ips], color='black',
                     mfc=COLOR['ips'], marker='D', markersize=9, mew=1.5,
                     capsize=4, lw=1.5, zorder=5)
        ax.errorbar([x_vtx], [m_vtx], yerr=[s_vtx], color='black',
                     mfc=COLOR['vertex'], marker='D', markersize=9, mew=1.5,
                     capsize=4, lw=1.5, zorder=5)
        ax.set_xticks([x_ips, x_vtx])
        ax.set_xticklabels(['IPS', 'Vertex'])
        ax.set_xlim(-0.5, 1.5)

        # Paired t-test annotation
        t, p = stats.ttest_rel(wide['ips'], wide['vertex'])
        ax.set_title(
            {'mu': 'μ (log n)', 'sd': 'σ',
             'amplitude': 'Amplitude', 'baseline': 'Baseline'}[par]
            + f"\nΔ = {wide['diff'].mean():+.3f}, t({n_paired-1}) = {t:+.2f}, p = {p:.3f}",
            fontsize=9,
        )
        sns.despine(ax=ax, offset=3, trim=True)

    fig.suptitle(f'm{model_label} per-session PRF parameter shifts: IPS-TMS vs Vertex sham '
                  f'({ROI}, n={n_paired} paired subjects)',
                  fontsize=10, y=1.04)
    fig.text(0.5, -0.04,
              'each gray line = one subject  ·  diamonds = group mean ± SEM  ·  '
              'paired t-test p-values in titles',
              ha='center', va='top', fontsize=8, color='0.4')

    out_root = Path(f'notes/figures/m{model_label}_tms_param_shifts')
    fig.savefig(out_root.with_suffix('.pdf'))
    fig.savefig(out_root.with_suffix('.png'), dpi=200)
    print(f'wrote {out_root}.{{pdf,png}}')

    out_tsv = Path(f'notes/data/m{model_label}_tms_param_shifts.tsv')
    full = pd.concat({par: pivots[par] for par in PARAMS}, axis=1)
    full.to_csv(out_tsv, sep='\t')
    print(f'wrote {out_tsv}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model_label', type=int, default=2)
    args = p.parse_args()
    main(model_label=args.model_label)
