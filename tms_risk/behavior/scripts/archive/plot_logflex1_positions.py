"""What the position-based (n1/n2) log-space model says: noise curves for
the first- vs second-presented option by stimulation, plus the cTBS
difference curves per position. logflex1 = independent 5-df splines per
position, TMS on both. Subject-averaged; log-unit differences with
50%/95% HDIs."""
from pathlib import Path
import sys

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'libs' / 'bauer'))
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'mathtext.fontset': 'stixsans',
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .03,
})
IPS, VERTEX = '#d62728', '#2ca02c'
PURPLE = '#8172B2'
BOLD = dict(fontweight='bold', fontfamily='Arial')
VC = 'stimulation_condition[T.vertex]'
softplus = lambda x: np.logaddexp(0, x)
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 80))

from tms_risk.behavior.fit_model import build_model, get_data

df = get_data('/data/ds-tmsrisk', model_label='logflex1')
model = build_model('logflex1', df)
subj = pd.read_csv(DATA / 'logflex1_subject_draws.tsv.gz', sep='\t')
SUBS = sorted(subj['subject'].unique())


def s_get(sub, var, reg):
    q = subj.query('subject == @sub and var == @var and regressor == @reg')
    return q.sort_values('draw')['value'].values


def curves(noise, vertex):
    bas = np.asarray(model.make_dm(N_GRID, variable=noise))
    regs = ['Intercept'] + ([VC] if vertex else [])
    coef = np.stack([np.stack(
        [sum(s_get(sub, f'{noise}_spline{j}', rg) for rg in regs)
         for j in range(1, 6)], 1) for sub in SUBS], 0)
    return softplus(np.einsum('sdj,gj->sdg', coef, bas)).mean(0)


POS = [('n1_evidence_sd', 'First-presented option (n1, from memory)'),
       ('n2_evidence_sd', 'Second-presented option (n2, on screen)')]
store = {(n, v): curves(n, v) for n, _ in POS for v in (False, True)}

fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.6), sharex=True,
                         sharey='col')
for c, (noise, cname) in enumerate(POS):
    ax = axes[0, c]
    for vertex, colr in [(True, VERTEX), (False, IPS)]:
        y = store[(noise, vertex)]
        med = np.median(y, 0)
        h95 = np.array([az.hdi(y[:, i], hdi_prob=.95)
                        for i in range(y.shape[1])])
        h50 = np.array([az.hdi(y[:, i], hdi_prob=.50)
                        for i in range(y.shape[1])])
        ax.fill_between(N_GRID, h95[:, 0], h95[:, 1], color=colr, alpha=.10,
                        lw=0)
        ax.fill_between(N_GRID, h50[:, 0], h50[:, 1], color=colr, alpha=.22,
                        lw=0)
        ax.plot(N_GRID, med, color=colr)
    ax.set_title(cname, fontsize=8, **BOLD)
    if c == 0:
        ax.set_ylabel('Evidence noise SD (log units)')
        ax.text(.97, .97, 'IPS', transform=ax.transAxes, fontsize=7,
                color=IPS, ha='right', va='top')
        ax.text(.97, .87, 'Vertex', transform=ax.transAxes, fontsize=7,
                color=VERTEX, ha='right', va='top')

    ax = axes[1, c]
    rel = store[(noise, False)] - store[(noise, True)]
    med = np.median(rel, 0)
    h95 = np.array([az.hdi(rel[:, i], hdi_prob=.95)
                    for i in range(rel.shape[1])])
    h50 = np.array([az.hdi(rel[:, i], hdi_prob=.50)
                    for i in range(rel.shape[1])])
    ax.fill_between(N_GRID, h95[:, 0], h95[:, 1], color=PURPLE, alpha=.14,
                    lw=0)
    ax.fill_between(N_GRID, h50[:, 0], h50[:, 1], color=PURPLE, alpha=.30,
                    lw=0)
    ax.plot(N_GRID, med, color=PURPLE)
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')
    if c == 0:
        ax.set_ylabel('Δ noise SD, IPS − vertex\n(log units)', fontsize=7.5)
    p_pos = float((rel.mean(1) > 0).mean())
    ax.text(.03, .97, f'P(mean Δ > 0) = {p_pos:.2f}',
            transform=ax.transAxes, fontsize=6.5, va='top', color='.25')

sns.despine(fig=fig, offset=4)
fig.tight_layout()
fig.savefig(OUT / 'logflex1_positions.pdf')
fig.savefig(OUT / 'logflex1_positions.png', dpi=150)
print('wrote', OUT / 'logflex1_positions.pdf')
