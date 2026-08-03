"""Figure 2: nPRF amplitude against preferred numerosity, and where the code sits.

As close a recreation as the data allow of cell 17 of
`modeling/notebooks/analyze_encoding_model.ipynb` (saved there as
`derivatives/figures/amplitude_vs_preferred_numerosity.pdf`).

**Same data source as the published figure**: the OLD log-space tree
`encoding_model.denoise.smoothed` (+ `.cv.` for cvR2), ROI `NPCr2cm-cluster`, cached to
`notes/data/prf_voxels_oldtree.tsv` by
`modeling/scripts/reproduce_figure2_stats.py --dump_voxels`. That script reproduces all
five statistics of the Figure-2 paragraph to 3-4 decimals, so the numbers behind this
figure are verified. `encoding_model2.model-1` is NOT used here: under it `mu`, `sd`,
`r2` and `cvr2` are session-invariant by construction, so the per-session tuning the
figure shows does not exist there.

Replicated from the original:
    voxel mask   cvR2 > 0 in EITHER arm (notebook cell 7), preserving the pairing
    amplitude    binned by pd.cut(mu_natural, np.arange(0, 50, 5)), plotted at midpoints
    histogram    grey step, unit bins np.arange(1, 100), stat='density'
    stimulus     dashed black KDE over n1
    layout       GridSpec 2 rows, height_ratios 5:3, linear x, xlim(0, 30)
    palette      vertex green, IPS red (hue_order ['vertex', 'ips'] in the original)

ONE deliberate deviation, flagged: the original passed voxel-level rows straight to
`sns.lineplot`, so its band is a bootstrap over ~10^3-10^4 voxels rather than over 35
subjects -- pseudoreplication, and roughly an order of magnitude too narrow. Here the
mean is taken within subject x bin first and the band is +/-1 SEM across subjects, which
also matches how the paragraph's statistics are computed. `--voxel_level_ci` restores
the original band exactly as published.

    python -m tms_risk.modeling.scripts.plot_figure2
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

# The original used stimulation_palette with hue_order ['vertex', 'ips'], i.e. vertex
# green, IPS red -- the house convention (CLAUDE.md), kept here.
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
sns.set_context('paper')


def main(voxel_tsv, out_stem, bids_folder, xmax, voxel_level_ci):
    d = pd.read_csv(voxel_tsv, sep='\t')
    thr = d[d.in_mask].copy()                       # notebook cell 7 mask
    thr['mu_natural'] = np.exp(thr.mu)
    print(f'{len(d)} voxel-sessions, {d.subject.nunique()} subjects; '
          f'mask keeps {len(thr)}')

    from tms_risk.utils.data import get_all_behavior
    beh = get_all_behavior(bids_folder=bids_folder).reset_index()

    q = np.percentile(thr.mu_natural, [25, 50, 75])
    qp = np.percentile(pd.concat([beh.n1, beh.n2]).dropna(), [25, 50, 75])
    print(f'  preferred numerosity IQR [{q[0]:.2f}, {q[2]:.2f}], median {q[1]:.2f}')
    print(f'  presented  numerosity IQR [{qp[0]:.0f}, {qp[2]:.0f}], median {qp[1]:.0f}')

    fig = plt.figure(figsize=(6.0, 5.0), constrained_layout=True)
    gs = GridSpec(2, 1, figure=fig, height_ratios=[5, 3])

    # ---------------------------------------------- a: amplitude vs preferred numerosity
    ax = fig.add_subplot(gs[0, 0])
    thr['mu_bin'] = pd.cut(thr.mu_natural, bins=np.arange(0, 50, 5.))
    thr['x'] = thr.mu_bin.apply(lambda b: b.mid if b == b else np.nan)
    src = thr.dropna(subset=['x'])
    if voxel_level_ci:
        plot_src, err = src, ('ci', 95)      # as published: bootstrap over voxels
    else:
        plot_src = (src.groupby(['subject', 'arm', 'x'], observed=True)
                       .amplitude.mean().reset_index())
        err = ('se', 1)                       # +/-1 SEM across subjects
    for a_, col, lab in [('vertex', VERTEX, 'Vertex'), ('ips', IPS, 'IPS')]:
        sns.lineplot(data=plot_src[plot_src.arm == a_], x='x', y='amplitude', ax=ax,
                     color=col, errorbar=err, marker='o', markeredgecolor='none',
                     err_kws=dict(alpha=.22, linewidth=0), label=lab, zorder=3)
    ax.set_ylabel('Amplitude')
    ax.set_xlabel(None)
    ax.legend(loc='upper right', frameon=False)
    ax.tick_params(labelbottom=False)

    ps = thr.pivot_table(index='subject', columns='arm', values='amplitude',
                         aggfunc='mean').dropna()
    t, p = stats.ttest_rel(ps['ips'], ps['vertex'])
    print(f'  amplitude: vertex {ps["vertex"].median():.4f} -> ips {ps["ips"].median():.4f}'
          f'  t({len(ps)-1})={t:+.4f}  p2={p:.4f}  p1={p/2:.4f}')
    ax.text(0.02, 0.05,
            f'cTBS lowers amplitude\nt({len(ps)-1}) = {abs(t):.2f}, p = {p/2:.3f} (one-sided)',
            transform=ax.transAxes, fontsize=7.5, color='0.35', va='bottom', ha='left')

    # --------------------------------------- b: preferred vs presented numerosity
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax)
    sns.histplot(thr, x='mu_natural', element='step', ax=ax2, color='gray',
                 bins=np.arange(1, 100), stat='density', lw=1.0,
                 label='Preferred numerosities')
    sns.kdeplot(x=beh.n1.dropna(), color='k', fill=False, alpha=1., lw=1.6, ls='--',
                ax=ax2, label='Stimulus distribution')
    ax2.set_xlabel('Preferred numerosity')
    ax2.set_ylabel('Density')
    ax2.legend(loc='upper right', frameon=False)
    ax2.set_xlim(0, xmax)

    for a_, letter in [(ax, 'a'), (ax2, 'b')]:
        a_.text(-0.09, 1.02, letter, transform=a_.transAxes, fontsize=12,
                fontweight='bold', va='bottom', ha='right')
    sns.despine(fig=fig, offset=4)

    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--voxel_tsv', default='notes/data/prf_voxels_oldtree.tsv')
    p.add_argument('--xmax', default=30., type=float)
    p.add_argument('--voxel_level_ci', action='store_true',
                   help='reproduce the original (pseudoreplicated) voxel-level band')
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out', default='notes/figures/figure2_recreated')
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.voxel_tsv, a.out, a.bids_folder, a.xmax, a.voxel_level_ci)
