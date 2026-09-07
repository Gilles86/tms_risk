"""Figure 4 draft for logflex2 in the fig4_new layout (exploratory).

A: PPC by stake, three model pairs (Weber | Flexible | Log-flexible2), the
paper's miss-arrow convention. B: perceptual noise by stimulation
(subject-averaged). C: relative cTBS contrast with CrI. D: condensed ladder
with the delta-band PPC verdicts annotated.
"""
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
BOLD = dict(fontweight='bold', fontfamily='Arial')
ORDERS = ['Risky first', 'Risky second']
VC = 'stimulation_condition[T.vertex]'

# ---------------------------------------------------------------- panel A --
MODELS = [('weber2nf', 'Weber PMC'),
          ('flexible2nf', 'Flexible PMC'),
          ('logflex2', 'Log-flexible PMC')]
frames = {i: pd.read_csv(DATA / f'ppc_by_stake.{lbl}.tsv', sep='\t')
          for i, (lbl, _) in enumerate(MODELS)}
allv = pd.concat(frames.values())
ylo = min(allv.lo.min(), allv.observed.min()) - .012
yhi = max(allv.hi.max(), allv.observed.max()) + .012
stakes = np.sort(allv.stake.unique())
x = np.arange(len(stakes))

fig = plt.figure(figsize=(7.25, 4.9))
gs_a = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.12, top=.90,
                         bottom=.62)
        for l, r in [(.065, .345), (.39, .67), (.715, .995)]]
axes_a = []
for m in range(3):
    axes_a += [fig.add_subplot(gs_a[m][0, 0]), fig.add_subplot(gs_a[m][0, 1])]

misses = {}
for m, (lbl, name) in enumerate(MODELS):
    d = frames[m]
    for o, order in enumerate(ORDERS):
        ax = axes_a[2 * m + o]
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = d[(d.order == order) & (d.stim == stim)].sort_values('stake')
            ax.fill_between(x, s.lo, s.hi, color=colr, alpha=.20, lw=0, zorder=1)
            ax.plot(x, s['mean'], color=colr, lw=1.2, zorder=2)
            dx = .06 if stim == 'ips' else -.06
            ax.plot(x + dx, s.observed, 'o', color=colr, ms=3.6, lw=0, zorder=4)
            for _, r in s.iterrows():
                if r.lo <= r.observed <= r.hi:
                    continue
                up = r.observed > r.hi
                xi = float(x[np.argmin(np.abs(stakes - r.stake))]) + dx
                ax.annotate('', xy=(xi, r.observed), xycoords='data',
                            xytext=(-14, 12 if up else -12),
                            textcoords='offset points', zorder=6,
                            arrowprops=dict(arrowstyle='-|>', color='.1',
                                            lw=.9, shrinkA=0, shrinkB=3.5,
                                            mutation_scale=6))
                misses[lbl] = misses.get(lbl, 0) + 1
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-.42, len(stakes) - .58)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_yticks([.5, .55, .6, .65])
        ax.set_title(order, fontsize=7, color='.3', pad=3, style='italic')
        if 2 * m + o == 0:
            ax.set_ylabel('P(chose risky)')
        else:
            ax.set_yticklabels([])
for gsx, (lbl, name) in zip(gs_a, MODELS):
    l, r = gsx.left, gsx.right
    fig.text((l + r) / 2, .955, name + f'  ({misses.get(lbl, 0)} miss'
             + ('es' if misses.get(lbl, 0) != 1 else '') + ')',
             ha='center', fontsize=8.5, color='.1', **BOLD)
    fig.text((l + r) / 2, .535, 'Stake (CHF)', ha='center', fontsize=8)
axes_a[0].text(.06, .96, 'IPS', transform=axes_a[0].transAxes, fontsize=7,
               color=IPS, va='top')
axes_a[0].text(.06, .84, 'Vertex', transform=axes_a[0].transAxes, fontsize=7,
               color=VERTEX, va='top')

# ------------------------------------------------- panels B, C, D ----------
from tms_risk.behavior.fit_model import build_model, get_data
df = get_data('/data/ds-tmsrisk', model_label='logflex2')
model = build_model('logflex2', df)
subj = pd.read_csv(DATA / 'logflex2_subject_draws.tsv.gz', sep='\t')
softplus = lambda x: np.logaddexp(0, x)
SUBS = sorted(subj['subject'].unique())
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 80))


def s_get(sub, var, reg='Intercept'):
    q = subj.query('subject == @sub and var == @var and regressor == @reg')
    return q.sort_values('draw')['value'].values


SC = {r: np.stack([np.stack([s_get(sub, f'perceptual_noise_sd_spline{j}', r)
                             for j in range(1, 6)], 1) for sub in SUBS], 0)
      for r in ('Intercept', VC)}
BAS = model.make_dm(N_GRID, variable='perceptual_noise_sd')


def nu_subj(vertex):
    c = SC['Intercept'] + (SC[VC] if vertex else 0)
    return softplus(np.einsum('sdj,gj->sdg', c, BAS)).mean(0)


gs_b = fig.add_gridspec(1, 3, left=.075, right=.985, top=.42, bottom=.10,
                        wspace=.42)


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


ax = fig.add_subplot(gs_b[0, 0])
for vertex, c in [(False, IPS), (True, VERTEX)]:
    y = nu_subj(vertex)
    med = np.median(y, 0)
    h = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    ax.fill_between(N_GRID, h[:, 0], h[:, 1], color=c, alpha=.18, lw=0)
    ax.plot(N_GRID, med, color=c, lw=1.3)
logx(ax)
ax.set_ylabel('Perceptual noise SD (log units)')
ax.set_title('Noise as a function of magnitude', fontsize=8, **BOLD)

ax = fig.add_subplot(gs_b[0, 1])
y_i, y_v = nu_subj(False), nu_subj(True)
rel = (y_i - y_v) / y_v * 100
med = np.median(rel, 0)
h = np.array([az.hdi(rel[:, i], hdi_prob=.95) for i in range(rel.shape[1])])
ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
ax.fill_between(N_GRID, h[:, 0], h[:, 1], color='#8172B2', alpha=.18, lw=0)
ax.plot(N_GRID, med, color='#8172B2', lw=1.3)
logx(ax)
ax.set_ylabel('Δ noise, IPS − vertex (%)')
ax.set_title('Effect of cTBS on noise', fontsize=8, **BOLD)

# ELPDs: power_vs_flexible_ladder.tsv, lfxgrid_loo.tsv, VM logflex ladder
ax = fig.add_subplot(gs_b[0, 2])
rows = [('Flexible PMC (TMS: perception)', -4157.5, None),
        ('Flexible PMC (TMS: both)', -4161.0, None),
        ('Log-flexible, scalar mem. (TMS: perc.)', -4162.7, 'FAIL 0.67'),
        ('Log-flexible2 (TMS: both)', -4175.4, 'PASS 1.0'),
        ('Log-flexible, scalar mem. (null)', -4260.4, None),
        ('Flexible PMC (null)', -4271.4, None)]
best = max(r[1] for r in rows)
for y, (name, e, ppc) in enumerate(rows[::-1]):
    c = '#33619E' if 'Log' in name else '.45'
    ax.plot(e - best, y, 'o', color=c, ms=4)
    ax.text(e - best - 5, y + .22, name, fontsize=6.5, ha='right',
            va='center', color=c)
    if ppc:
        ax.text(e - best + 6, y, f'Δ-PPC {ppc}', fontsize=6, va='center',
                color='#2ca02c' if 'PASS' in ppc else '#d62728')
ax.axvline(0, color='.8', lw=.6, ls='--', zorder=0)
ax.set_xlim(-260, 95)
ax.set_ylim(-.6, len(rows) - .2)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('ELPD − best (nats)')
ax.set_title('Model comparison', fontsize=8, **BOLD)

for yf, letter in [(.965, 'A'), (.46, 'B')]:
    fig.text(.008, yf, letter, fontsize=11, va='bottom', ha='left', **BOLD)
fig.text(.40, .46, 'C', fontsize=11, va='bottom', ha='left', **BOLD)
fig.text(.70, .46, 'D', fontsize=11, va='bottom', ha='left', **BOLD)

sns.despine(fig=fig, offset=4)
fig.savefig(OUT / 'fig4_logflex2.pdf')
fig.savefig(OUT / 'fig4_logflex2.png', dpi=150)
print('wrote', OUT / 'fig4_logflex2.pdf', '| misses:', misses)
