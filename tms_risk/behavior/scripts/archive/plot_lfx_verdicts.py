"""Verdict map of the log-space grid: ELPD ladder with the Δ-band PPC
verdict encoded in marker fill and convergence in marker shape. Reads
notes/data/lfx_verdicts.tsv (cluster cells) + logflex_family_verdicts.tsv
(VM logflex2/logflexm2 family)."""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 9,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
})
PASS, FAIL, MID = '#2ca02c', '#d62728', '#e8a33d'
BOLD = dict(fontweight='bold', fontfamily='Arial')

cl = pd.read_csv(DATA / 'lfx_verdicts.tsv', sep='\t')
vm = pd.read_csv(DATA / 'logflex_family_verdicts.tsv', sep='\t')
vm['min_ess'] = float('nan')

NAME = {
    'lfx2-bs3-m2-dp-bm': 'Linear mem. — TMS: both (bs3-m2-bm)',
    'lfx2-bs2-m2-dp-b': 'Linear mem. — TMS: perc. (bs2-m2-b)',
    'lfx2-bs2-m3-dp-b': 'Quadr. mem. — TMS: perc. (bs2-m3-b)',
    'lfx2-bs2-m3-dp-bm': 'Quadr. mem. — TMS: both (bs2-m3-bm)',
    'lfx2-bs3-m3-dp-b': 'Quadr. mem. — TMS: perc. (bs3-m3-b)',
    'lfx2-cr3-m3-dp-b': 'Quadr. mem. — TMS: perc. (cr3-m3-b)',
    'logflex2': '5-df mem. — TMS: both (logflex2)',
    'logflex1': 'n1/n2 noise — TMS: both (logflex1)',
    'lfx2-bs2-m2-dp-bm': 'Linear mem. — TMS: both (bs2-m2-bm)',
    'lfx2-bs3-m2-dp-b': 'Linear mem. — TMS: perc. (bs3-m2-b)',
    'logflex2b': '5-df mem. — TMS: perc. (logflex2b)',
    'lfx2-bs3-m3-dp-bm': 'Quadr. mem. — TMS: both (bs3-m3-bm)',
    'logflex1a': 'n1/n2 — TMS: first opt. (logflex1a)',
    'lfx2-cr3-m3-dp-bm': 'Quadr. mem. — TMS: both (cr3-m3-bm)',
    'logflex2a': '5-df mem. — TMS: mem. (logflex2a)',
    'logflex1b': 'n1/n2 — TMS: second opt. (logflex1b)',
    'logflexm2b': 'Scalar mem. — TMS: perc. (logflexm2b)',
    'logflexm2a': 'Scalar mem. — TMS: mem. (logflexm2a)',
}
NULLS = {'lfx2-bs3-m2-dp-null', 'lfx2-bs3-m3-dp-null', 'lfx2-bs2-m2-dp-null',
         'lfx2-bs2-m3-dp-null', 'lfx2-cr3-m3-dp-null', 'logflex1_null',
         'logflex2_null', 'logflexm2_null'}

d = pd.concat([cl, vm], ignore_index=True)
d = d[~d.label.isin(NULLS)].sort_values('elpd_loo', ascending=True)
d['name'] = d.label.map(NAME).fillna(d.label)
d['conv'] = d.max_rhat <= 1.01
best = d.elpd_loo.max()
null_band = (cl[cl.label.isin(NULLS)].elpd_loo.max() - best,
             vm[vm.label.isin(NULLS)].elpd_loo.min() - best)

fig, ax = plt.subplots(figsize=(5.6, 4.6))
for y, (_, r) in enumerate(d.iterrows()):
    x = r.elpd_loo - best
    col = PASS if r.delta_cov >= .99 else (MID if r.delta_cov > .7 else FAIL)
    if r.conv:
        ax.plot(x, y, 'o', color=col, ms=5.5, zorder=3)
    else:
        ax.plot(x, y, 'o', mfc='white', mec=col, mew=1.2, ms=5.5, zorder=3)
    grey = '.45' if r.conv else '.65'
    ax.text(x - 1.5, y, r['name'], ha='right', va='center', fontsize=6.5,
            color=('.15' if r.conv else '.6'))
ax.axvline(0, color='.85', lw=.6, ls='--', zorder=0)
ax.axvspan(null_band[0], null_band[1], color='.93', zorder=0)
ax.text((null_band[0] + null_band[1]) / 2, .2, 'No-TMS nulls',
        ha='center', fontsize=6.5, color='.5', style='italic', rotation=90,
        va='bottom')
ax.set_xlim(-128, 6)
ax.set_ylim(-.8, len(d) - .2)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('ELPD − best (nats)')
ax.set_title('Log-space PMC grid: ELPD × convergence × Δ-band PPC',
             fontsize=9, **BOLD)

handles = [
    mlines.Line2D([], [], marker='o', ls='', color=PASS, ms=5.5,
                  label='Δ-PPC pass (6/6 cells)'),
    mlines.Line2D([], [], marker='o', ls='', color=MID, ms=5.5,
                  label='Partial (5/6)'),
    mlines.Line2D([], [], marker='o', ls='', color=FAIL, ms=5.5,
                  label='Fail (≤4/6)'),
    mlines.Line2D([], [], marker='o', ls='', mfc='white', mec='.4', mew=1.2,
                  ms=5.5, label='Open = unconverged (r̂ > 1.01)'),
]
ax.legend(handles=handles, loc='upper left', fontsize=6.5,
          handletextpad=.3, borderaxespad=0)

sns.despine(fig=fig, offset=4)
fig.tight_layout()
fig.savefig(OUT / 'lfx_verdict_ladder.pdf')
fig.savefig(OUT / 'lfx_verdict_ladder.png', dpi=150)
print('wrote', OUT / 'lfx_verdict_ladder.pdf')
