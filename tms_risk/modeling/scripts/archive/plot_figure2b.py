"""Figure 2B, cleaned up: nPRF amplitude by preferred numerosity + numerosity distributions.

Same verified data source as `plot_figure2.py` (the OLD log-space tree cached to
`notes/data/prf_voxels_oldtree.tsv`, notebook-cell-7 mask: cvR2 > 0 in either arm), same
subject-level +/-1 SEM band. Only the presentation changes: direct labels instead of
legends, in-panel stat annotation, explicit ticks, single paper-panel size.

    python -m tms_risk.modeling.scripts.plot_figure2b
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

# House convention (CLAUDE.md): IPS (stimulated) red, vertex (sham) green.
IPS, VERTEX = '#d62728', '#2ca02c'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 4,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def main(voxel_tsv, out_stem, bids_folder, xmax):
    d = pd.read_csv(voxel_tsv, sep='\t')
    thr = d[d.in_mask].copy()
    thr['mu_natural'] = np.exp(thr.mu)

    from tms_risk.utils.data import get_all_behavior
    beh = get_all_behavior(bids_folder=bids_folder).reset_index()

    fig = plt.figure(figsize=(3.4, 3.2), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[3.0, 1.6])

    # ------------------------------------------- top: amplitude vs preferred numerosity
    ax = fig.add_subplot(gs[0, 0])
    thr['mu_bin'] = pd.cut(thr.mu_natural, bins=np.arange(0, 50, 5.))
    thr['x'] = thr.mu_bin.apply(lambda b: b.mid if b == b else np.nan).astype(float)
    src = (thr.dropna(subset=['x'])
              .groupby(['subject', 'arm', 'x'], observed=True)
              .amplitude.mean().reset_index())
    src = src[src.x < xmax]
    for a_, col in [('vertex', VERTEX), ('ips', IPS)]:
        sns.lineplot(data=src[src.arm == a_], x='x', y='amplitude', ax=ax,
                     color=col, errorbar=('se', 1), marker='o',
                     markeredgecolor='none', markersize=3.5,
                     err_kws=dict(alpha=.2, linewidth=0), zorder=3, legend=False)

    # Direct labels at the right end of each line, in the line's color
    ends = (src.groupby(['arm', 'x'], observed=True).amplitude.mean()
               .groupby('arm').last())
    up, down = ('vertex', 'ips') if ends['vertex'] >= ends['ips'] else ('ips', 'vertex')
    names = {'vertex': ('Vertex', VERTEX), 'ips': ('IPS', IPS)}
    ax.text(28.6, ends[up] + 0.28, names[up][0], color=names[up][1], fontsize=8.5,
            va='center', ha='left')
    ax.text(28.6, ends[down] - 0.28, names[down][0], color=names[down][1],
            fontsize=8.5, va='center', ha='left')

    ps = thr.pivot_table(index='subject', columns='arm', values='amplitude',
                         aggfunc='mean').dropna()
    t, p = stats.ttest_rel(ps['ips'], ps['vertex'])
    print(f'amplitude: vertex {ps["vertex"].median():.4f} -> ips {ps["ips"].median():.4f}'
          f'  t({len(ps) - 1})={t:+.4f}  p1={p / 2:.4f}')
    ax.text(0.03, 0.97, f'cTBS lowers amplitude\nt({len(ps) - 1}) = {abs(t):.2f}, '
            f'p = {p / 2:.3f} (one-sided)',
            transform=ax.transAxes, fontsize=7.5, color='0.3', va='top', ha='left')

    ax.set_ylabel('Amplitude (% signal change)')
    ax.set_xlabel(None)
    ax.set_xlim(0, 34)
    ax.set_xticks([0, 10, 20, 30])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylim(0, 3.2)
    ax.tick_params(labelbottom=False)

    # ------------------------------- bottom: preferred vs presented numerosity densities
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    sns.histplot(thr, x='mu_natural', element='step', ax=ax2, color='0.55',
                 bins=np.arange(1, 100), stat='density', lw=0.9, alpha=.25,
                 legend=False)
    sns.kdeplot(x=pd.concat([beh.n1, beh.n2]).dropna(), color='k', lw=1.3, ls='--',
                ax=ax2, legend=False)
    ax2.text(3.0, 0.135, 'Preferred\n(voxels)', color='0.35', fontsize=7.5,
             va='top', ha='left')
    ax2.text(24.5, 0.055, 'Presented\n(stimuli)', color='k', fontsize=7.5,
             va='bottom', ha='left')
    ax2.set_xlabel('Numerosity')
    ax2.set_ylabel('Density')
    ax2.set_yticks([0, 0.1])

    sns.despine(fig=fig, offset=4, trim=True)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--voxel_tsv', default='notes/data/prf_voxels_oldtree.tsv')
    p.add_argument('--xmax', default=30., type=float)
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out', default='notes/figures/figure2b_clean')
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.voxel_tsv, a.out, a.bids_folder, a.xmax)
