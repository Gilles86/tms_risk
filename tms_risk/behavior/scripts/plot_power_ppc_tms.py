"""Paper-style PPC with the TMS effect front and center (exploratory, NOT paper).

Top: psychometrics by stimulation (power2_full posterior-predictive bands, data
points ± across-subject SEM). Bottom: the ips − vertex difference computed
within subject, with both models' posterior-predictive bands (power law vs
flexible splines). Reads notes/data/ppc2_{power,flexible,data}.tsv.
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
C_POWER, C_FLEX = '#3B5BA5', '#8172B2'

power = pd.read_csv(DATA / 'ppc2_power.tsv', sep='\t')
flex = pd.read_csv(DATA / 'ppc2_flexible.tsv', sep='\t')
data = pd.read_csv(DATA / 'ppc2_data.tsv', sep='\t')

ORDERS = [(True, 'Risky first'), (False, 'Risky second')]
RN = 1 / 0.55

fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.6), constrained_layout=True,
                         sharex=True)
for ax in axes.ravel():
    ax.set_xlim(1.15, 3.9)

for col, (rf, name) in enumerate(ORDERS):
    # top: psychometric, power2_full bands + data ± SEM
    ax = axes[0, col]
    ax.axhline(.5, color='0.85', lw=.6, ls='--', zorder=0)
    ax.vlines(RN, 0.05, .92, color='0.75', lw=.7, ls=':', zorder=0)
    for stim, c in [('ips', C_IPS), ('vertex', C_VERTEX)]:
        m = power.query('kind == "psychometric" and risky_first == @rf and '
                        'stimulation_condition == @stim').sort_values('ratio')
        ax.fill_between(m['ratio'], m['lo'], m['hi'], color=c, alpha=.20, lw=0,
                        zorder=1)
        ax.plot(m['ratio'], m['median'], color=c, lw=1.3, zorder=2)
        d = data.query('kind == "psychometric" and risky_first == @rf and '
                       'stimulation_condition == @stim').sort_values('ratio')
        ax.errorbar(d['ratio'], d['mean'], yerr=d['sem'], fmt='o', color=c,
                    ms=4, mec='white', mew=.5, elinewidth=.8, capsize=0,
                    zorder=3)
    ax.set_xscale('log')
    ax.set_xticks([1.25, 1.82, 2.5, 3.5])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(0.05, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_title(name, fontsize=9)
    if col == 0:
        ax.set_ylabel('P(choose risky)')
        ax.text(1.22, .90, 'IPS', color=C_IPS, fontsize=8)
        ax.text(1.22, .80, 'Vertex', color=C_VERTEX, fontsize=8)
    else:
        ax.text(RN, .96, 'Risk-neutral', color='0.5', fontsize=7.5, ha='center')

    # bottom: within-subject ips − vertex difference
    ax = axes[1, col]
    ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
    for m, c in [(power, C_POWER), (flex, C_FLEX)]:
        md = m.query('kind == "delta" and risky_first == @rf').sort_values('ratio')
        ax.fill_between(md['ratio'], md['lo'], md['hi'], color=c, alpha=.20,
                        lw=0, zorder=1)
        ax.plot(md['ratio'], md['median'], color=c, lw=1.3, zorder=2)
    d = data.query('kind == "delta" and risky_first == @rf').sort_values('ratio')
    ax.errorbar(d['ratio'], d['mean'], yerr=d['sem'], fmt='o', color='.15',
                ms=4.5, mec='white', mew=.5, elinewidth=.8, capsize=0, zorder=3)
    ax.set_ylim(-0.14, 0.19)
    ax.set_yticks([-0.1, 0, 0.1])
    ax.set_xlabel('Risky / safe payoff ratio')
    if col == 0:
        ax.set_ylabel('ΔP(risky), IPS − Vertex')
        ax.text(1.22, .165, 'Power law', color=C_POWER, fontsize=8)
        ax.text(1.22, .135, 'Splines', color=C_FLEX, fontsize=8)
        ax.text(1.22, .105, 'Data ± SEM', color='.15', fontsize=8)

for ax, letter in zip(axes.ravel(), 'abcd'):
    ax.text(-0.13, 1.03, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'power_ppc_tms.pdf')
fig.savefig(OUT / 'power_ppc_tms.png', dpi=150)
print('wrote', OUT / 'power_ppc_tms.pdf')
