"""Posterior-predictive checks for the power-law PMC set (exploratory, NOT paper).

Reads notes/data/power_ppc_{model,data}.tsv (built on sciencecloud by
compute_ppc.py: simulated choices per posterior draw, aggregated per subject x
cell within draw, then across subjects; 95% HDI across draws).

Top row: Bayesian (power2_full) vs no-prior (power2_flat_full) bands over the
same observed points, pooled over stimulation. Bottom row: power2_full by
stimulation condition. Points: observed across-subject means. Bands:
posterior-predictive 95% HDI; line: median.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_IPS, C_VERTEX = '#d62728', '#2ca02c'
C_BAYES, C_FLAT = '#3B5BA5', '#E08214'

model = pd.read_csv(DATA / 'power_ppc_model.tsv', sep='\t')
data = pd.read_csv(DATA / 'power_ppc_data.tsv', sep='\t')

ORDERS = [(True, 'Risky first'), (False, 'Risky second')]
RN = 1 / 0.55


def style_ax(ax):
    ax.set_xscale('log')
    ax.set_xlim(1.15, 4.1)
    ax.set_xticks([1.25, 1.82, 2.5, 3.5])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.axhline(.5, color='0.8', lw=.6, ls='--', zorder=0)
    ax.vlines(RN, 0, .92, color='0.7', lw=.7, ls=':', zorder=0)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0, .5, 1])
    ax.set_xlabel('Risky / safe payoff ratio')


def draw(ax, m, color):
    m = m.sort_values('ratio')
    ax.fill_between(m['ratio'], m['lo'], m['hi'], color=color, alpha=.22, lw=0, zorder=1)
    ax.plot(m['ratio'], m['median'], color=color, lw=1.3, zorder=2)


fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.4), constrained_layout=True,
                         sharex=True, sharey=True)

for col, (rf, name) in enumerate(ORDERS):
    # top: Bayes vs no-prior, pooled over stimulation
    ax = axes[0, col]
    for mlab, c in [('power2_full', C_BAYES), ('power2_flat_full', C_FLAT)]:
        draw(ax, model.query('model == @mlab and agg == "pooled" and risky_first == @rf'), c)
    d = data.query('agg == "pooled" and risky_first == @rf').sort_values('ratio')
    ax.plot(d['ratio'], d['observed'], 'o', color='.15', ms=4.5, mec='white',
            mew=.5, zorder=3)
    style_ax(ax)
    ax.set_title(name, fontsize=9)
    if col == 0:
        ax.set_ylabel('P(choose risky)')
        ax.text(1.22, .92, 'Bayesian', color=C_BAYES, fontsize=8)
        ax.text(1.22, .80, 'No prior', color=C_FLAT, fontsize=8)
        ax.text(1.22, .68, 'Data', color='.15', fontsize=8)
    if col == 1:
        ax.text(RN, .96, 'Risk-neutral', color='0.5', fontsize=7.5, ha='center')

    # bottom: power2_full by stimulation
    ax = axes[1, col]
    for stim, c in [('ips', C_IPS), ('vertex', C_VERTEX)]:
        draw(ax, model.query('model == "power2_full" and agg == "by_stim" and '
                             'risky_first == @rf and stimulation_condition == @stim'), c)
        d = data.query('agg == "by_stim" and risky_first == @rf and '
                       'stimulation_condition == @stim').sort_values('ratio')
        ax.plot(d['ratio'], d['observed'], 'o', color=c, ms=4, mec='white',
                mew=.5, zorder=3)
    style_ax(ax)
    if col == 0:
        ax.set_ylabel('P(choose risky)')
        ax.text(1.22, .92, 'IPS', color=C_IPS, fontsize=8)
        ax.text(1.22, .80, 'Vertex', color=C_VERTEX, fontsize=8)

for ax, letter in zip(axes.ravel(), 'abcd'):
    ax.text(-0.13, 1.05, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'power_ppc.pdf')
fig.savefig(OUT / 'power_ppc.png', dpi=150)
print('wrote', OUT / 'power_ppc.pdf')
