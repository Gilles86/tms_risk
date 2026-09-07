"""How the per-subject cTBS shifts relate to each other, and to brain and behaviour.

The group-level figure shows what cTBS did on average. This one asks what it did
to individual participants, and whether the pieces line up the way the mechanism
says they should:

a  the per-subject shift in each noise channel, at the low and the high anchor.
   Paired within participant, so the n1-vs-n2 contrast is visible per person
   rather than only in the group mean.
b  do the two channels move together? If cTBS raised a single shared perceptual
   noise, sigma_n1 and sigma_n2 shifts would lie on the identity line; if it hit
   the second-presented option specifically, the cloud sits below it.
c  correlation matrix over every per-subject shift -- the two noise channels,
   the nPRF amplitude loss, and the behavioural changes in consistency,
   indifference point and choice proportion.
d  the two scatters the mechanism predicts: more noise on the second-presented
   option should mean less consistent choices, and a bigger nPRF amplitude loss
   should mean more noise.

    python -m tms_risk.behavior.scripts.plot_shift_mechanism --model_label log-power-n1n2
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
N1C, N2C = '#3B5BA5', '#C44E52'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 2.5, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

#: the per-subject columns worth relating, and how to name them on the axes
LINK = [('d_amp_rel_median', 'Δ nPRF amplitude'),
        ('d_consistency_rsecond', 'Δ consistency (risky 2nd)'),
        ('d_indifference_rsecond', 'Δ indifference (risky 2nd)'),
        ('d_p_risky_rsecond', 'ΔP(risky) (risky 2nd)')]


def scatter(ax, x, y, xl, yl, col='0.25'):
    m = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[m], np.asarray(y)[m]
    ax.scatter(x, y, s=13, color=col, alpha=.75, lw=0)
    if len(x) > 3:
        r, p = stats.pearsonr(x, y)
        b = np.polyfit(x, y, 1)
        xx = np.linspace(x.min(), x.max(), 20)
        ax.plot(xx, np.polyval(b, xx), color=col, lw=1.1, alpha=.8)
        ax.text(.04, .95, f'r = {r:+.2f}, p = {p:.3f}  (n = {len(x)})',
                transform=ax.transAxes, va='top', fontsize=6, color='0.3')
    ax.axhline(0, color='0.85', lw=.6, zorder=0)
    ax.axvline(0, color='0.85', lw=.6, zorder=0)
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)


def main(data_dir, out_stem, label):
    dd = Path(data_dir)
    sh = pd.read_csv(dd / f'subject_shifts/subject_shifts.{label}.tsv', **READ)
    W = sh.pivot_table(index='subject', columns=['channel', 'x'],
                       values='shift_pct')
    W.columns = [f'd{c}_{int(x)}' for c, x in W.columns]
    link = pd.read_csv(dd / 'bb_link_master.tsv', **READ).set_index('subject')
    D = W.join(link, how='inner')
    print(f'{len(D)} subjects with both model shifts and link measures')

    fig = plt.figure(figsize=(7.25, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1.3])

    # -- a: the shifts themselves ----------------------------------------
    LETTERED = []
    ax = fig.add_subplot(gs[0, 0]); LETTERED.append(ax)
    cols = [c for c in ['dn1_7', 'dn2_7', 'dn1_28', 'dn2_28'] if c in D]
    for i, c in enumerate(cols):
        v = D[c].values
        col = N1C if c.startswith('dn1') else N2C
        ax.scatter(np.full(len(v), i) + np.random.default_rng(0).uniform(
            -.13, .13, len(v)), v, s=9, color=col, alpha=.45, lw=0)
        ax.plot([i - .28, i + .28], [np.mean(v)] * 2, color=col, lw=2.2,
                solid_capstyle='butt', zorder=4)
        se = stats.sem(v)
        ax.plot([i, i], [np.mean(v) - se, np.mean(v) + se], color=col, lw=1.4,
                zorder=4)
    ax.axhline(0, color='0.75', lw=.7, ls='--', zorder=0)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(['σ$_{n1}$\n7 CHF', 'σ$_{n2}$\n7 CHF',
                        'σ$_{n1}$\n28 CHF', 'σ$_{n2}$\n28 CHF'][:len(cols)])
    ax.set_ylabel('cTBS effect on noise (%)')
    ax.set_title('Per-subject noise shifts', fontsize=8)

    # -- b: do the two channels move together? ---------------------------
    ax = fig.add_subplot(gs[0, 1]); LETTERED.append(ax)
    if {'dn1_7', 'dn2_7'} <= set(D):
        scatter(ax, D.dn1_7, D.dn2_7, 'Δσ$_{n1}$ at 7 CHF (%)',
                'Δσ$_{n2}$ at 7 CHF (%)')
        lim = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
               max(ax.get_xlim()[1], ax.get_ylim()[1])]
        ax.plot(lim, lim, color='0.7', lw=.8, ls=':', zorder=0)
        ax.text(.96, .05, 'Dotted: equal shift\n(a shared perceptual effect)',
                transform=ax.transAxes, ha='right', fontsize=5.6, color='0.5',
                linespacing=1.4)
    ax.set_title('Are the channels yoked?', fontsize=8)

    # -- c: everything against everything --------------------------------
    ax = fig.add_subplot(gs[:, 2]); LETTERED.append(ax)
    keep = [c for c in cols] + [c for c, _ in LINK if c in D]
    lab = ([c.replace('dn1_', 'Δσ n1 ').replace('dn2_', 'Δσ n2 ') + ' CHF'
            for c in cols] + [n for c, n in LINK if c in D])
    C = D[keep].corr()
    im = ax.imshow(C.values, cmap='vlag', vmin=-1, vmax=1)
    ax.set_xticks(range(len(keep)))
    ax.set_xticklabels(lab, fontsize=6)
    plt.setp(ax.get_xticklabels(), rotation=90, ha='center', va='top')
    ax.set_yticks(range(len(keep)))
    ax.set_yticklabels(lab, fontsize=6)
    for i in range(len(keep)):
        for j in range(len(keep)):
            if i == j:
                continue
            ax.text(j, i, f'{C.values[i, j]:.2f}', ha='center', va='center',
                    fontsize=5.4,
                    color='white' if abs(C.values[i, j]) > .55 else '0.2')
    ax.set_title('Correlation of per-subject shifts', fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=.035, pad=.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label('Pearson r', fontsize=6.5)

    # -- d: the two the mechanism predicts -------------------------------
    for k, (xc, yc, xl, yl) in enumerate([
            ('dn2_7', 'd_consistency_rsecond', 'Δσ$_{n2}$ at 7 CHF (%)',
             'Δ consistency (risky 2nd)'),
            ('d_amp_rel_median', 'dn2_7', 'Δ nPRF amplitude (rel.)',
             'Δσ$_{n2}$ at 7 CHF (%)')]):
        ax = fig.add_subplot(gs[1, k]); LETTERED.append(ax)
        if xc in D and yc in D:
            scatter(ax, D[xc], D[yc], xl, yl, col=N2C)
        ax.set_title(['Noise vs consistency', 'Brain vs noise'][k], fontsize=8)

    # label the panels explicitly; fig.axes also contains the colorbar
    for ax, s in zip(LETTERED, 'abcde'):
        ax.text(-.22, 1.06, s, transform=ax.transAxes, fontsize=8.5,
                fontweight='bold', family='Arial', va='bottom')
    sns.despine(fig=fig, offset=3)
    for a in fig.axes:
        if a.get_title() == 'Correlation of per-subject shifts':
            a.spines[:].set_visible(False)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')

    print('\nkey correlations:')
    for a, b in [('dn1_7', 'dn2_7'), ('dn2_7', 'd_consistency_rsecond'),
                 ('dn2_7', 'd_p_risky_rsecond'), ('d_amp_rel_median', 'dn2_7'),
                 ('d_amp_rel_median', 'd_consistency_rsecond')]:
        if a in D and b in D:
            m = np.isfinite(D[a]) & np.isfinite(D[b])
            r, p = stats.pearsonr(D[a][m], D[b][m])
            print(f'  {a:22s} x {b:26s} r = {r:+.3f}  p = {p:.4f}  n = {m.sum()}')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    stem = a.out_stem or str(REPO / f'notes/figures/shift_mechanism_{a.model_label}')
    main(a.data_dir, stem, a.model_label)
