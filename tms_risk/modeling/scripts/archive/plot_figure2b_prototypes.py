"""Prototype: Figure 2B rebuilt from canonical m1 parameters, under three voxel selections.

Columns: (1) ALL voxels of the 2 cm stimulation cluster (the strongest amplitude test,
see notes/v9_plan.md B2); (2) voxels where m1 beats the training-mean null (proxy:
cvR2_m1 > -0.0178, the ROI-mean null -- the exact per-voxel null lives cluster-side);
(3) voxels with cvR2_m1 > 0 (the legacy criterion). Top row: m1 amplitude by preferred
numerosity (m1's mu/sd are session-invariant, so the binning axis is one value per
voxel). Bottom row: preferred-numerosity density vs presented stimuli.

Unlike the published 2B (old log-space tree), everything here is the canonical
encoding_model2.model-1 fit, so the annotated test is the two-sided one from
notes/amplitude_effect_voxel_selection.md.

    python -m tms_risk.modeling.scripts.plot_figure2b_prototypes
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

IPS, VERTEX = '#d62728', '#2ca02c'
NULL = -0.0178

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 8,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main(voxel_tsv, bids_folder, out_stem):
    d = pd.read_csv(voxel_tsv, sep='\t')
    d = d[d['in_NPCr2cm-cluster']].copy()

    from tms_risk.utils.data import get_all_behavior
    beh = get_all_behavior(bids_folder=bids_folder).reset_index()
    stim = pd.concat([beh.n1, beh.n2]).dropna()

    variants = [
        ('All voxels', d),
        ('m1 beats null', d[d.cvr2_m1 > NULL]),
        ('cvR² > 0', d[d.cvr2_m1 > 0]),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(7.25, 3.4), sharex=True, sharey='row',
                             height_ratios=[2.4, 1.1], constrained_layout=True)

    long = None
    for col, (name, sel) in enumerate(variants):
        ax, ax2 = axes[0, col], axes[1, col]
        sub = sel.melt(id_vars=['subject', 'pref_n_m1'],
                       value_vars=['amplitude_m1_vertex', 'amplitude_m1_ips'],
                       var_name='arm', value_name='amplitude')
        sub['arm'] = sub.arm.str.replace('amplitude_m1_', '')
        sub['x'] = pd.cut(sub.pref_n_m1, bins=np.arange(0, 50, 5.)).apply(
            lambda b: b.mid if b == b else np.nan).astype(float)
        src = (sub.dropna(subset=['x'])
                  .groupby(['subject', 'arm', 'x'], observed=True)
                  .amplitude.mean().reset_index())
        src = src[src.x < 30]
        for a_, c in [('vertex', VERTEX), ('ips', IPS)]:
            sns.lineplot(data=src[src.arm == a_], x='x', y='amplitude', ax=ax,
                         color=c, errorbar=('se', 1), marker='o',
                         markeredgecolor='none', markersize=3,
                         err_kws=dict(alpha=.2, linewidth=0), zorder=3, legend=False)

        ps = (sel.groupby('subject')[['amplitude_m1_ips', 'amplitude_m1_vertex']]
                 .mean().dropna())
        t, p = stats.ttest_rel(ps.amplitude_m1_ips, ps.amplitude_m1_vertex)
        w = stats.wilcoxon(ps.amplitude_m1_ips - ps.amplitude_m1_vertex).pvalue
        nv = len(sel)
        print(f'{name:14s} n_vox={nv:6d} ({nv / len(d):.0%})  '
              f'vertex {ps.amplitude_m1_vertex.mean():.3f} -> '
              f'ips {ps.amplitude_m1_ips.mean():.3f}  t(34)={t:+.2f} p2={p:.4f} '
              f'wilcoxon={w:.4f}')
        ax.set_title(f'{name} ({nv / len(d):.0%})')
        ax.text(0.03, 0.97, f't(34) = {abs(t):.2f}\np = {p:.3f} (two-sided)',
                transform=ax.transAxes, fontsize=7, color='0.3', va='top', ha='left')
        ax.set_xlim(0, 32)
        ax.set_xticks([0, 10, 20, 30])
        ax.set_ylabel('m1 amplitude (psc)' if col == 0 else None)
        ax.set_xlabel(None)

        sns.histplot(sel, x='pref_n_m1', element='step', ax=ax2, color='0.55',
                     bins=np.arange(0, 100), stat='density', lw=0.9, alpha=.25,
                     legend=False)
        sns.kdeplot(x=stim, color='k', lw=1.1, ls='--', ax=ax2, legend=False)
        ax2.set_xlabel('Preferred numerosity')
        ax2.set_ylabel('Density' if col == 0 else None)

    # direct condition labels once, in the first panel
    axes[0, 0].text(0.97, 0.85, 'Vertex', color=VERTEX, fontsize=8, ha='right',
                    transform=axes[0, 0].transAxes)
    axes[0, 0].text(0.97, 0.73, 'IPS', color=IPS, fontsize=8, ha='right',
                    transform=axes[0, 0].transAxes)

    sns.despine(fig=fig, offset=3)
    for ext in ['pdf', 'png']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--voxel_tsv', default='notes/data/prf_voxel_table.tsv')
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out', default='notes/figures/prototypes/figure2b_m1_selections')
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.voxel_tsv, a.bids_folder, a.out)
