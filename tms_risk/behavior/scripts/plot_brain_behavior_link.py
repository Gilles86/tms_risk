"""Figure: the per-subject link between the neural and the behavioural cTBS effect.

    a  Subjects who lost more nPRF gain at the stimulation site lost more choice
       consistency, on trials where the safe option came first.
    b  The same correlation across ROIs and both presentation orders: it is specific
       to the stimulated right-parietal cortex and to one presentation order.
    c  The within-subject version, which needs no session pairing: trials with a more
       precisely decoded first option had a steeper psychometric slope.

Rebuilds from `notes/data/*.tsv` alone.

    python -m tms_risk.behavior.scripts.plot_brain_behavior_link
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

REPO = Path(__file__).resolve().parents[3]
DATA = REPO / 'notes' / 'data'
FIGS = REPO / 'notes' / 'figures'

# A cTBS difference gets its own ink, never one of the condition colours.
INK = '#1a1a1a'
CONTROL = '#9c9c9c'

ROI_ORDER = ['NPCr2cm-cluster', 'NPC12r', 'NPCl', 'NF1', 'NTO']
ROI_LABEL = {'NPCr2cm-cluster': 'Stim.\nsite', 'NPC12r': 'Right\npariet.',
             'NPCl': 'Left\npariet.', 'NF1': 'Frontal', 'NTO': 'Occip.\ntemp.'}


def set_style():
    mpl.rcParams.update({
        'font.family': 'Helvetica',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9,
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


def load():
    neu = pd.read_csv(DATA / 'bb_neural.tsv', sep='\t')
    neu = neu[neu['selection'] == 'cvr2pos']
    amp = neu.pivot_table(index='subject', columns='mask', values='d_amp_median')

    beh = pd.read_csv(DATA / 'bb_behavior.tsv', sep='\t')
    w = beh.pivot_table(index='subject', columns='stimulation_condition',
                        values=['consistency_rsecond', 'consistency_rfirst'])
    con = pd.DataFrame({c: w[(c, 'ips')] - w[(c, 'vertex')]
                        for c in ['consistency_rsecond', 'consistency_rfirst']})
    return amp.join(con)


def boot_ci(x, y, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    b = np.array([np.corrcoef(x[i], y[i])[0, 1]
                  for i in (rng.integers(0, len(x), (n_boot, len(x))))])
    return np.nanpercentile(b, [2.5, 97.5])


def panel_a(ax, d):
    x = d['NPCr2cm-cluster'].values
    y = d['consistency_rsecond'].values
    r, p = stats.pearsonr(x, y)

    ax.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax.axvline(0, color='0.8', lw=0.6, zorder=0)
    ax.scatter(x, y, s=18, facecolor=INK, edgecolor='white', linewidth=0.4,
               alpha=0.85, zorder=3)

    fit = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(fit, xs), color=INK, lw=1.2, zorder=2)

    ax.set_xlabel('Δ nPRF gain (IPS − vertex)')
    ax.set_ylabel('Δ Choice consistency\n(IPS − vertex)')
    ax.set_xticks([-1, -0.5, 0, 0.5])
    ax.set_yticks([-6, -3, 0, 3, 6])
    ax.text(0.03, 0.95, f'r({len(x)-2}) = {r:.2f}, p = {p:.3f}',
            transform=ax.transAxes, ha='left', va='top', fontsize=8)
    ax.annotate('Safe option first', xy=(0.03, 0.85), xycoords='axes fraction',
                ha='left', va='top', fontsize=8, color='0.35')


def panel_b(ax, d):
    for order, col, dx, lab in [('consistency_rsecond', INK, -0.13, 'Safe first'),
                                ('consistency_rfirst', CONTROL, +0.13, 'Risky first')]:
        rs, los, his = [], [], []
        for roi in ROI_ORDER:
            x, y = d[roi].values, d[order].values
            rs.append(stats.pearsonr(x, y)[0])
            lo, hi = boot_ci(x, y)
            los.append(lo)
            his.append(hi)
        pos = np.arange(len(ROI_ORDER)) + dx
        ax.errorbar(pos, rs, yerr=[np.array(rs) - los, np.array(his) - np.array(rs)],
                    fmt='o', color=col, ms=4.5, lw=0.9, capsize=0, zorder=3)
        ax.text(len(ROI_ORDER) - 0.55, rs[-1] + (0.13 if col == INK else -0.13), lab,
                color=col, fontsize=8, ha='left',
                va='bottom' if col == INK else 'top')

    ax.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax.set_xticks(range(len(ROI_ORDER)))
    ax.set_xticklabels([ROI_LABEL[r] for r in ROI_ORDER], fontsize=7.5)
    ax.set_ylabel('Correlation with Δ gain')
    ax.set_ylim(-0.62, 0.95)
    ax.set_yticks([-0.4, 0, 0.4, 0.8])
    ax.set_xlim(-0.55, len(ROI_ORDER) + 0.35)


def panel_c(ax, quality='log_abs_err'):
    t = pd.read_csv(DATA / f'bb_trialwise_byorder_{quality}.tsv', sep='\t')
    labels = {'risky second (safe first)': 'Safe\nfirst', 'risky first': 'Risky\nfirst'}
    t['x'] = t['order'].map({'risky second (safe first)': 0, 'risky first': 1})
    cols = {0: INK, 1: CONTROL}

    ax.axhline(0, color='0.8', lw=0.6, zorder=0)
    ax.set_ylim(-3.2, 5.2)
    rng = np.random.default_rng(0)
    for xi, g in t.groupby('x'):
        jit = xi + rng.uniform(-0.11, 0.11, len(g))
        ax.scatter(jit, g['b_lr_x_q'], s=9, color=cols[xi], alpha=0.35,
                   linewidth=0, zorder=2)
        m = g['b_lr_x_q'].mean()
        se = g['b_lr_x_q'].sem()
        ax.errorbar(xi, m, yerr=se, fmt='D', color=cols[xi], ms=7,
                    markeredgecolor='white', markeredgewidth=1.2, lw=1.6,
                    capsize=0, zorder=4)
        p = stats.ttest_1samp(g['b_lr_x_q'], 0)[1]
        ax.text(xi, 4.6, f'p = {p:.3f}', ha='center', va='bottom',
                fontsize=7.5, color=cols[xi])

    ax.set_xticks([0, 1])
    ax.set_xticklabels([labels[k] for k in
                        ['risky second (safe first)', 'risky first']])
    ax.set_ylabel('Decoding × ratio\ninteraction (β)')
    ax.set_xlim(-0.45, 1.45)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=str(FIGS / 'brain_behavior_link.pdf'))
    args = ap.parse_args()

    set_style()
    d = load().dropna(subset=ROI_ORDER + ['consistency_rsecond',
                                          'consistency_rfirst'])

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), constrained_layout=True,
                             gridspec_kw={'width_ratios': [1.05, 1.45, 0.7]})
    panel_a(axes[0], d)
    panel_b(axes[1], d)
    panel_c(axes[2])

    for ax, letter in zip(axes, 'abc'):
        ax.text(-0.28, 1.06, letter, transform=ax.transAxes, fontsize=11,
                fontweight='bold', va='bottom', ha='left')
    sns.despine(fig=fig, offset=4, trim=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(str(args.out).replace('.pdf', '.png'), dpi=200)
    print(f'wrote {args.out}  (n = {len(d)} subjects)')


if __name__ == '__main__':
    main()
