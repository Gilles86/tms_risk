"""Robustness of the spherical-Ω expected-uncertainty effect across voxel
selections.

Reruns the same decode (spherical noise covariance, expected SD of the decoded
estimate vs true magnitude, IPS vs Vertex) under four voxel-selection rules and
lays them out side by side, so the IPS−Vertex precision-loss curve can be judged
*independent of how voxels were picked*:

    top100      — top 100 voxels by in-session R²  (the original default; biased
                  because cTBS lowers R² so each session selects different voxels)
    top500      — top 500 by in-session R²
    ses1cvr2    — session-1 cvR² > 0  (the paper's rule: independent targeting
                  session, identical voxel set for IPS and Vertex)
    mixture     — R²-mixture P(signal) >= 0.5 on session-1 cvR² (principled,
                  per-subject signal/noise threshold)

Each column: top row = expected SD per condition; bottom row = IPS−Vertex paired
difference with a Maris-Oostenveld cluster permutation test. Reuses the loaders,
cluster test, palette and style from ``plot_spherical_expected_uncertainty``.
"""
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tms_risk.modeling.scripts.plot_spherical_expected_uncertainty import (
    SPHERICAL_ROOT, COLOR, N_MIN, N_MAX, X_TICKS, TEST_WINDOW,
    apply_style, join_stimulation_condition, cluster_perm_test, _style_x,
    windowed_paired_test,
)

# Plotted metric: realised mean |decoded − true|, NOT √var_E (which is fooled by
# grid-mean collapse and reverses the cTBS effect — see plot_spherical_expected_uncertainty).
METRIC_COL = 'mean_abs_error'

# Selection tokens to show, in order, with display labels.
SELECTIONS = [
    ('100',      'Top 100 (in-session R²)'),
    ('500',      'Top 500 (in-session R²)'),
    ('ses1cvr2', 'cvR² > 0  (session 1)'),
    ('mixture',  'R² mixture  (session 1)'),
]


def load_all_selections(root: Path = SPHERICAL_ROOT) -> pd.DataFrame:
    """Load every spherical mc_decode TSV, parsing the selection token (which
    may be an int like 100/500 or a string like ses1cvr2/mixture)."""
    rows = []
    for tsv in root.glob('sub-*/ses-*/func/*_mc_decode.tsv'):
        m = re.match(
            r'sub-(\d+)_ses-(\d)_roi-([^_]+)_nvoxels-([A-Za-z0-9]+)_mc_decode',
            tsv.stem,
        )
        if not m:
            continue
        df = pd.read_csv(tsv, sep='\t')
        df['subject']   = int(m.group(1))
        df['session']   = int(m.group(2))
        df['roi']       = m.group(3)
        df['selection'] = m.group(4)
        rows.append(df)
    if not rows:
        raise SystemExit(f'No TSVs found under {root}')
    out = pd.concat(rows, ignore_index=True)
    out['expected_sd'] = np.sqrt(out['var_E'])
    out['bias']        = out['mean_error']
    return out


def paired_for_selection(mc_sel: pd.DataFrame) -> pd.DataFrame:
    paired = mc_sel.pivot_table(
        index=['subject', 'value'], columns='stimulation_condition',
        values=METRIC_COL, aggfunc='mean',
    ).reset_index()
    if 'ips' not in paired or 'vertex' not in paired:
        return pd.DataFrame(columns=['subject', 'value', 'ips', 'vertex', 'diff'])
    paired['diff'] = paired['ips'] - paired['vertex']
    return paired


def main(n_perm: int = 1000, log_prior: bool = False):
    apply_style()
    root = Path(str(SPHERICAL_ROOT) + '.logprior') if log_prior else SPHERICAL_ROOT
    mc = load_all_selections(root=root)
    mc = join_stimulation_condition(mc)
    mc = mc[(mc['value'] >= N_MIN) & (mc['value'] <= N_MAX)].copy()

    present = set(mc['selection'].unique())
    sels = [(tok, lab) for tok, lab in SELECTIONS if tok in present]
    missing = [tok for tok, _ in SELECTIONS if tok not in present]
    if missing:
        print(f'NOTE: selections not on disk yet, skipping: {missing}')
    ncol = len(sels)

    fig, axes = plt.subplots(2, ncol, figsize=(2.5 * ncol, 5.0),
                             constrained_layout=True, squeeze=False)

    print('\nCluster-permutation results per selection (expected SD, IPS−Vertex):')
    for j, (tok, lab) in enumerate(sels):
        sub = mc[mc['selection'] == tok]
        n_pair = (sub.pivot_table(index=['subject', 'value'],
                                  columns='stimulation_condition',
                                  values=METRIC_COL, aggfunc='mean')
                     .dropna().reset_index()['subject'].nunique())

        # Top row: expected SD per condition
        axT = axes[0][j]
        for cond in ('ips', 'vertex'):
            agg = (sub[sub.stimulation_condition == cond]
                   .groupby('value')[METRIC_COL].agg(['mean', 'sem']).reset_index())
            axT.fill_between(agg['value'], agg['mean'] - agg['sem'], agg['mean'] + agg['sem'],
                             alpha=0.18, color=COLOR[cond], linewidth=0)
            axT.plot(agg['value'], agg['mean'], color=COLOR[cond], lw=1.3, zorder=3)
        axT.set_title(f'{lab}\n(n={n_pair} paired)', fontsize=8.5)
        if j == 0:
            axT.set_ylabel('Expected absolute\ndecoding error (n)')
        _style_x(axT)

        # Bottom row: IPS − Vertex with cluster test
        axB = axes[1][j]
        paired = paired_for_selection(sub).dropna(subset=['ips', 'vertex'])
        axB.axhline(0, ls='--', color='0.6', lw=0.6, zorder=0)
        if len(paired):
            dagg = paired.groupby('value')['diff'].agg(['mean', 'sem']).reset_index()
            axB.fill_between(dagg['value'], dagg['mean'] - dagg['sem'], dagg['mean'] + dagg['sem'],
                             alpha=0.25, color='0.4', linewidth=0)
            axB.plot(dagg['value'], dagg['mean'], color='black', lw=1.3, zorder=3)
            perm = cluster_perm_test(paired, n_perm=n_perm)
            sig = []
            if perm is not None:
                for cl in perm['clusters']:
                    x0, x1 = perm['stimuli'][cl['start']], perm['stimuli'][cl['end']]
                    inside = (dagg['value'] >= x0) & (dagg['value'] <= x1)
                    color = '#d62728' if cl['mass'] > 0 else '#3B5BA5'
                    axB.fill_between(dagg.loc[inside, 'value'],
                                     dagg.loc[inside, 'mean'] - dagg.loc[inside, 'sem'],
                                     dagg.loc[inside, 'mean'] + dagg.loc[inside, 'sem'],
                                     alpha=0.55, color=color, linewidth=0, zorder=2)
                    if cl['p'] < 0.05:
                        ytop = (dagg.loc[inside, 'mean'] + dagg.loc[inside, 'sem']).max()
                        axB.annotate(f"p={cl['p']:.3f}", xy=(np.sqrt(x0 * x1), ytop),
                                     xytext=(0, 6), textcoords='offset points',
                                     ha='center', fontsize=7, color=color, fontweight='bold')
                    sig.append(f"n={int(x0)}..{int(x1)} mass={cl['mass']:+.1f} p={cl['p']:.3f}")
            # Targeted small-numerosity paired test
            lo, hi = TEST_WINDOW
            axB.axvspan(lo, hi, color='0.92', zorder=0)
            wt = windowed_paired_test(paired, TEST_WINDOW)
            wtxt = (f"n {lo}–{hi}: p={wt['p_one']:.3f}" if wt else 'n.s.')
            axB.text(0.97, 0.05, wtxt, transform=axB.transAxes, ha='right',
                     va='bottom', fontsize=7, color='black')
            print(f'  {lab:32s} (n={n_pair}): '
                  + ('; '.join(sig) if sig else 'no clusters')
                  + (f'  | window {lo}-{hi} Δ={wt["mean"]:+.2f} p1s={wt["p_one"]:.3f}' if wt else ''))
        if j == 0:
            axB.set_ylabel('Abs. decoding error\n(IPS − Vertex)')
        axB.set_xlabel('True magnitude (n)')
        _style_x(axB)

    fig.suptitle('Spherical-Ω expected uncertainty: IPS vs Vertex across voxel selections',
                 fontsize=11, y=1.04)
    fig.text(0.5, -0.03,
             'Top row: expected SD per condition (IPS red, Vertex green).  '
             'Bottom row: IPS−Vertex paired diff, cluster-perm test (1000 perms).  '
             'ROI = NPC12r, spherical noise covariance.',
             ha='center', va='top', fontsize=7.5, color='0.4')

    out = Path('notes/figures/spherical_selection_comparison'
               + ('_logprior' if log_prior else ''))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=200, bbox_inches='tight')
    print(f'\nwrote {out}.{{pdf,png}}')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--n_perm', type=int, default=1000)
    p.add_argument('--log_prior', action='store_true',
                   help='Load the log-prior (geometric-grid) decode.')
    args = p.parse_args()
    main(n_perm=args.n_perm, log_prior=args.log_prior)
