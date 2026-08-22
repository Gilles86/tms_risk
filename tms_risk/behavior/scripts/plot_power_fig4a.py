"""Fig-4A-style PPC panel row (exploratory, NOT paper): Weber | Flexible |
Power-law PMC, P(chose risky) per stake tercile, split by presentation order
and stimulation. Exact replica of plot_fig4_model.ppc_panel's design, extended
to three models. Reads notes/data/ppc_by_stake.{weber2nf,flexible2nf,power2full}.tsv.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'mathtext.fontset': 'stixsans',
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': .8, 'ytick.major.width': .8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

IPS, VERTEX = '#d62728', '#2ca02c'
BOLD = dict(fontweight='bold', fontfamily='Arial')
ORDERS = ['Risky first', 'Risky second']

MODELS = [('weber2nf', 'Weber PMC'),
          ('flexible2nf', 'Flexible PMC'),
          ('power2full', 'Power-law PMC')]
frames = {i: pd.read_csv(DATA / f'ppc_by_stake.{lbl}.tsv', sep='\t')
          for i, (lbl, _) in enumerate(MODELS)}

allv = pd.concat(frames.values())
ylo = min(allv.lo.min(), allv.observed.min()) - .012
yhi = max(allv.hi.max(), allv.observed.max()) + .012
stakes = np.sort(allv.stake.unique())
x = np.arange(len(stakes))

H_A, PAD_TOP, PAD_BOT = 1.30, .55, .52
H = PAD_TOP + H_A + PAD_BOT
fig = plt.figure(figsize=(7.25, H))
row = dict(top=1 - PAD_TOP / H, bottom=PAD_BOT / H)
PAIRS = [(.065, .345), (.39, .67), (.715, .995)]
axes = []
for l, r in PAIRS:
    gs = fig.add_gridspec(1, 2, left=l, right=r, wspace=.12, **row)
    axes += [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]

misses = {}
for m, (lbl, name) in enumerate(MODELS):
    d = frames[m]
    for o, order in enumerate(ORDERS):
        ax = axes[2 * m + o]
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = d[(d.order == order) & (d.stim == stim)].sort_values('stake')
            ax.fill_between(x, s.lo, s.hi, color=colr, alpha=.20, lw=0, zorder=1)
            ax.plot(x, s['mean'], color=colr, lw=1.2, zorder=2)
            dx = .06 if stim == 'ips' else -.06
            ax.plot(x + dx, s.observed, 'o', color=colr, ms=3.8, lw=0, zorder=4)
            for _, r in s.iterrows():
                if r.lo <= r.observed <= r.hi:
                    continue
                up = r.observed > r.hi
                xi = float(x[np.argmin(np.abs(stakes - r.stake))]) + dx
                ax.annotate('', xy=(xi, r.observed), xycoords='data',
                            xytext=(-15, 13 if up else -13),
                            textcoords='offset points', zorder=6,
                            arrowprops=dict(arrowstyle='-|>', color='.1', lw=.9,
                                            shrinkA=0, shrinkB=3.5,
                                            mutation_scale=6))
                misses[lbl] = misses.get(lbl, 0) + 1
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-.42, len(stakes) - .58)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_yticks([.5, .55, .6, .65])
        ax.set_title(order, fontsize=7.2, color='.3', pad=4, style='italic')
        if 2 * m + o == 0:
            ax.set_ylabel('P(chose risky)')
        else:
            ax.set_yticklabels([])

axes[0].text(.06, .96, 'IPS', transform=axes[0].transAxes, fontsize=7,
             color=IPS, va='top')
axes[0].text(.06, .85, 'Vertex', transform=axes[0].transAxes, fontsize=7,
             color=VERTEX, va='top')
for (l, r), (_, name) in zip(PAIRS, MODELS):
    fig.text((l + r) / 2, row['top'] + .17 / H, name, ha='center', va='bottom',
             fontsize=9.5, color='.1', **BOLD)
    fig.text((l + r) / 2, row['bottom'] - .46 / H, 'Stake (CHF)', ha='center',
             va='bottom', fontsize=8.5)

sns.despine(fig=fig, offset=4)
for ext in ['pdf', 'png']:
    fig.savefig(OUT / f'power_fig4a.{ext}', bbox_inches='tight', pad_inches=.03)
print('wrote', OUT / 'power_fig4a.pdf')
print('PPC misses: ' + (', '.join(f'{k} {v}' for k, v in misses.items()) or 'none'))
