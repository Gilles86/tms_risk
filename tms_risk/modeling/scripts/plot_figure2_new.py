"""Candidate new Figure 2: neural cTBS effects (top) + encoding-model comparison (bottom).

    A  nPRF surface map, example subject (notes/figures/imaging/figure2a_surface.png,
       extracted from the v9 manuscript's Figure 2 -- no clean vector asset survives
       locally; NOTE notes/figures/{paper,imaging}/figure2a.pdf are an OLDER behavioral
       figure despite the name)
    B  nPRF amplitude by preferred numerosity (old-tree data, subject-level SEM band)
       + preferred vs presented numerosity densities
    C  Trial-wise decoding accuracy, vertex vs IPS (bb_decoding.tsv, NPCr2cm-cluster;
       reproduces the published 0.142 vs 0.092, p = 0.032)
    D  Held-out cvR2 per encoding model vs the training-mean null
    E  Fraction of ROI voxels beating the null
    F  Paired per-subject contrast against the amplitude model (m1, the canonical fit)

Color semantics: red = IPS (stimulated), green = vertex, everywhere. The bottom-row
models therefore do NOT use red: canonical near-black, tuning (mu+sd) blue, response
magnitude (amplitude+baseline) orange, remaining models gray.

    python -m tms_risk.modeling.scripts.plot_figure2_new
"""
import argparse
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy import stats

IPS, VERTEX = '#d62728', '#2ca02c'          # house convention: red = stimulated
MNAMES = {0: 'None', 1: 'Amplitude', 2: 'All four', 3: 'Amplitude + σ',
          4: 'μ + σ (tuning)', 5: 'Amplitude + baseline'}
MCOLORS = {0: '#C4C4C4', 1: '#2F2F2F', 2: '#9A9A9A', 3: '#9A9A9A',
           4: '#3B5BA5', 5: '#D1885C'}
CANONICAL = 1

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 8, 'legend.fontsize': 7.5,
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


def strip_mean(ax, x, vals, color, width=.28, ms=6.5, dot=10):
    jit = (np.random.RandomState(0).rand(len(vals)) - .5) * width
    ax.scatter(x + jit, vals, s=dot, color=color, alpha=.38, lw=0, zorder=2)
    m, se = np.nanmean(vals), stats.sem(vals, nan_policy='omit')
    ax.errorbar(x, m, yerr=se, fmt='D', ms=ms, color=color, mec='0.15', mew=1.3,
                elinewidth=1.3, capsize=0, zorder=4)


def model_ticks(ax, models):
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([MNAMES.get(m, f'm{m}') for m in models],
                       rotation=35, ha='right', rotation_mode='anchor')
    for lab, m in zip(ax.get_xticklabels(), models):
        lab.set_color('.45' if MCOLORS[m] == '#C4C4C4' else MCOLORS[m])


def load_panel_image(path, dpi=300):
    path = Path(path)
    if path.suffix == '.pdf':
        out = path.with_suffix('')
        subprocess.run(['pdftoppm', '-r', str(dpi), '-png', '-singlefile',
                        str(path), str(out)], check=True)
        path = Path(f'{out}.png')
    return plt.imread(path)


def main(voxel_tsv, decoding_tsv, grid_tsv, panel_a, bids_folder, out_stem):
    # ------------------------------------------------------------------- data
    vox = pd.read_csv(voxel_tsv, sep='\t')
    thr = vox[vox.in_mask].copy()
    thr['mu_natural'] = np.exp(thr.mu)

    from tms_risk.utils.data import get_all_behavior
    beh = get_all_behavior(bids_folder=bids_folder).reset_index()

    dec = pd.read_csv(decoding_tsv, sep='\t')
    dec = (dec[(dec['mask'] == 'NPCr2cm-cluster')
               & dec.stimulation_condition.isin(['ips', 'vertex'])]
           .pivot_table(index='subject', columns='stimulation_condition',
                        values='r_En1').dropna())

    grid = pd.read_csv(grid_tsv, sep='\t')
    grid = grid[grid.roi == 'NPCr2cm-cluster']
    models = sorted(grid.model.unique())

    # ------------------------------------------------------------------ layout
    fig = plt.figure(figsize=(7.25, 5.35), constrained_layout=True)
    outer = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0])
    top = outer[0].subgridspec(1, 3, width_ratios=[1.3, 1.35, 0.75])
    bot = outer[1].subgridspec(1, 3)
    letters = {}

    # ------------------------------------------------- A: surface map (image)
    axA = fig.add_subplot(top[0, 0])
    axA.imshow(load_panel_image(panel_a))
    axA.set_axis_off()
    axA.set_title('nPRF map, example subject', pad=16)
    letters['A'] = axA

    # ------------------------- B: amplitude by preferred numerosity + densities
    sub = top[0, 1].subgridspec(2, 1, height_ratios=[2.6, 1.15])
    axB = fig.add_subplot(sub[0, 0])
    thr['x'] = pd.cut(thr.mu_natural, bins=np.arange(0, 50, 5.)).apply(
        lambda b: b.mid if b == b else np.nan).astype(float)
    src = (thr.dropna(subset=['x'])
              .groupby(['subject', 'arm', 'x'], observed=True)
              .amplitude.mean().reset_index())
    src = src[src.x < 30]
    for a_, col in [('vertex', VERTEX), ('ips', IPS)]:
        sns.lineplot(data=src[src.arm == a_], x='x', y='amplitude', ax=axB,
                     color=col, errorbar=('se', 1), marker='o',
                     markeredgecolor='none', markersize=3.2,
                     err_kws=dict(alpha=.2, linewidth=0), zorder=3, legend=False)
    ends = (src.groupby(['arm', 'x'], observed=True).amplitude.mean()
               .groupby('arm').last())
    axB.text(28.6, ends['vertex'] + 0.30, 'Vertex', color=VERTEX, fontsize=8,
             va='center', ha='left')
    axB.text(28.6, ends['ips'] - 0.30, 'IPS', color=IPS, fontsize=8,
             va='center', ha='left')
    ps = thr.pivot_table(index='subject', columns='arm', values='amplitude',
                         aggfunc='mean').dropna()
    t, p = stats.ttest_rel(ps['ips'], ps['vertex'])
    print(f'B amplitude: vertex {ps["vertex"].median():.4f} -> ips '
          f'{ps["ips"].median():.4f}  t({len(ps)-1})={t:+.3f}  p1={p/2:.4f}')
    axB.text(0.03, 0.97, f'cTBS lowers amplitude\nt({len(ps)-1}) = {abs(t):.2f}, '
             f'p = {p/2:.3f} (one-sided)',
             transform=axB.transAxes, fontsize=7, color='0.3', va='top', ha='left')
    axB.set_title('nPRF amplitude after cTBS', pad=16)
    axB.set_ylabel('Amplitude (% signal change)')
    axB.set_xlabel(None)
    axB.set_xlim(0, 34)
    axB.set_xticks([0, 10, 20, 30])
    axB.set_yticks([0, 1, 2, 3])
    axB.set_ylim(0, 3.2)
    axB.tick_params(labelbottom=False)
    letters['B'] = axB

    axB2 = fig.add_subplot(sub[1, 0], sharex=axB)
    sns.histplot(thr, x='mu_natural', element='step', ax=axB2, color='0.55',
                 bins=np.arange(1, 100), stat='density', lw=0.9, alpha=.25,
                 legend=False)
    sns.kdeplot(x=pd.concat([beh.n1, beh.n2]).dropna(), color='k', lw=1.2,
                ls='--', ax=axB2, legend=False)
    axB2.text(2.6, 0.155, 'Preferred\n(voxels)', color='0.35', fontsize=7,
              va='top', ha='left')
    axB2.text(24.0, 0.055, 'Presented\n(stimuli)', color='k', fontsize=7,
              va='bottom', ha='left')
    axB2.set_xlabel('Numerosity')
    axB2.set_ylabel('Density')
    axB2.set_yticks([0, 0.1])

    # ------------------------------------------------- C: decoding accuracy
    subC = top[0, 2].subgridspec(2, 1, height_ratios=[2.6, 1.15])
    axC = fig.add_subplot(subC[0, 0])
    axC.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    strip_mean(axC, 0, dec.vertex.values, VERTEX)
    strip_mean(axC, 1, dec.ips.values, IPS)
    t, p = stats.ttest_rel(dec.ips, dec.vertex)
    print(f'C decoding: vertex {dec.vertex.mean():.4f} -> ips {dec.ips.mean():.4f}'
          f'  t({len(dec)-1})={t:+.3f}  p2={p:.4f}')
    axC.text(0.5, 1.005, f'p = {p:.3f}', transform=axC.get_xaxis_transform(),
             ha='center', va='bottom', fontsize=7, color='.35')
    axC.set_title('Decoding accuracy', pad=16)
    axC.set_ylabel('Decoding accuracy (r)')
    axC.set_xlim(-0.6, 1.6)
    letters['C'] = axC

    # ------------------------------------- D/E/F: encoding-model comparison
    axD = fig.add_subplot(bot[0, 0])
    axD.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    for i, m in enumerate(models):
        g = grid[grid.model == m]
        strip_mean(axD, i, (g.cvr2 - g.null).values, MCOLORS.get(m, '.4'))
    axD.set_title('Out-of-sample fit', pad=16)
    axD.set_ylabel('cvR² − null')
    axD.text(0.01, 0.0, 'Null', transform=axD.get_yaxis_transform(), fontsize=7,
             color='.45', va='bottom', ha='left')
    tD, pD = stats.ttest_rel(
        grid[grid.model == CANONICAL].set_index('subject').cvr2,
        grid[grid.model == CANONICAL].set_index('subject')['null'])
    axD.set_ylim(top=0.13)
    axD.text(0.97, 0.97, f'Amplitude model best\nt(34) = {tD:.2f}, p = {pD:.3f}',
             transform=axD.transAxes, fontsize=7, color='0.3', va='top', ha='right')
    letters['D'] = axD

    axE = fig.add_subplot(bot[0, 1])
    axE.axhline(.5, color='.6', lw=.8, ls=':', zorder=0)
    for i, m in enumerate(models):
        strip_mean(axE, i, grid[grid.model == m].frac_beats_null.values,
                   MCOLORS.get(m, '.4'))
    axE.set_title('Voxels beating the null', pad=16)
    axE.set_ylabel('Fraction of voxels')
    letters['E'] = axE

    axF = fig.add_subplot(bot[0, 2])
    axF.axhline(0, color='.6', lw=.8, ls='--', zorder=0)
    w = grid.pivot_table(index='subject', columns='model', values='cvr2')
    others = [m for m in models if m != CANONICAL]
    for i, m in enumerate(others):
        diff = (w[m] - w[CANONICAL]).dropna().values
        strip_mean(axF, i, diff, MCOLORS.get(m, '.4'))
        t, p = stats.ttest_1samp(diff, 0)
        ptxt = f'{p:.3f}'.lstrip('0') if p >= .001 else '<.001'
        axF.text(i, 1.01, f'p {ptxt}', transform=axF.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=6.5, color='.35')
    axF.set_title('Versus the amplitude model', pad=16)
    axF.set_ylabel('Δ cvR² (model − amplitude)')
    letters['F'] = axF

    # ------------------------------------------------------------- finishing
    # despine BEFORE tick styling: spine.set_position() resets tick objects,
    # dropping rotation/color overrides while keeping the label text
    sns.despine(fig=fig, offset=3)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(['Vertex', 'IPS'])
    for lab, col in zip(axC.get_xticklabels(), [VERTEX, IPS]):
        lab.set_color(col)
    model_ticks(axD, models)
    model_ticks(axE, models)
    model_ticks(axF, others)

    # lowercase panel letters, 8 pt bold: Nature / Nature Comms house style and
    # what every other figure in this paper uses. 12 pt is out of spec.
    for letter, a in letters.items():
        a.text(-0.16 if letter != 'A' else -0.02, 1.06, letter.lower(),
               transform=a.transAxes, fontsize=9, fontweight='bold',
               va='bottom', ha='right', family='Arial')

    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(f'wrote {out_stem}.pdf')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--voxel_tsv', default='notes/data/prf_voxels_oldtree.tsv')
    p.add_argument('--decoding_tsv', default='notes/data/bb_decoding.tsv')
    p.add_argument('--grid_tsv', default='notes/data/cvr2_model_grid.tsv')
    p.add_argument('--panel_a', default='notes/figures/imaging/figure2a_surface.png')
    p.add_argument('--bids_folder', default='/data/ds-tmsrisk')
    p.add_argument('--out', default='notes/figures/figure2_new')
    a = p.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    main(a.voxel_tsv, a.decoding_tsv, a.grid_tsv, a.panel_a, a.bids_folder, a.out)
