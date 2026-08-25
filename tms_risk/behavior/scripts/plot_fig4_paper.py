"""Figure 4 for the primary log-space PMC (exploratory draft).

A: by-stake choice PPC for three models — Weber (constant noise, the null for
magnitude dependence) | Flexible PMC in natural space (the incumbent) | the
log-space primary — with the paper's miss-arrow convention.
B: the primary's two noise channels by stimulation. C: the cTBS contrast on
both channels, in log units. D: the ELPD ladder with convergence and Δ-PPC
verdicts attached.

All curves subject-averaged: per-subject-per-draw, averaged over subjects
within draw, then summarized over draws.
"""
from pathlib import Path
import argparse
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

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--label', default='lfx2-bs3-m2-dp-bm')
p.add_argument('--draws', default='m2bm_subject_draws.tsv.gz')
p.add_argument('--ppc', default='ppc_by_stake.lfx2-bs3-m2-dp-bm.tsv')
p.add_argument('--name', default='Log-flexible PMC (primary)')
p.add_argument('--out', default='fig4_paper_m2bm')
ARGS = p.parse_args()

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
C_MEM, C_PERC = '#33619E', '#8172B2'
BOLD = dict(fontweight='bold', fontfamily='Arial')
ORDERS = ['Risky first', 'Risky second']
VC = 'stimulation_condition[T.vertex]'
softplus = lambda x: np.logaddexp(0, x)
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 80))

# ---------------------------------------------------------------- panel A --
MODELS = [('ppc_by_stake.weber2nf.tsv', 'Weber PMC\n(constant noise)'),
          ('ppc_by_stake.flexible2nf.tsv', 'Flexible PMC\n(natural space)'),
          (ARGS.ppc, ARGS.name.replace(' (', '\n('))]
frames = {i: pd.read_csv(DATA / f, sep='\t') for i, (f, _) in enumerate(MODELS)}
allv = pd.concat(frames.values())
ylo = min(allv.lo.min(), allv.observed.min()) - .012
yhi = max(allv.hi.max(), allv.observed.max()) + .012
stakes = np.sort(allv.stake.unique())
x = np.arange(len(stakes))

fig = plt.figure(figsize=(7.25, 5.1))
gs_a = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.12, top=.88, bottom=.60)
        for l, r in [(.065, .345), (.39, .67), (.715, .995)]]
axes_a = []
for m in range(3):
    axes_a += [fig.add_subplot(gs_a[m][0, 0]), fig.add_subplot(gs_a[m][0, 1])]

misses = {}
for m, (fname, name) in enumerate(MODELS):
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
                xi = float(x[np.argmin(np.abs(stakes - r.stake))]) + dx
                ax.annotate('', xy=(xi, r.observed), xycoords='data',
                            xytext=(-14, 12 if r.observed > r.hi else -12),
                            textcoords='offset points', zorder=6,
                            arrowprops=dict(arrowstyle='-|>', color='.1', lw=.9,
                                            shrinkA=0, shrinkB=3.5,
                                            mutation_scale=6))
                misses[m] = misses.get(m, 0) + 1
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-.42, len(stakes) - .58)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_yticks([.5, .55, .6, .65])
        ax.set_title(order, fontsize=7, color='.3', pad=3, style='italic')
        if 2 * m + o == 0:
            ax.set_ylabel('P(chose risky)')
            ax.text(.06, .96, 'IPS', transform=ax.transAxes, fontsize=7,
                    color=IPS, va='top')
            ax.text(.06, .84, 'Vertex', transform=ax.transAxes, fontsize=7,
                    color=VERTEX, va='top')
        else:
            ax.set_yticklabels([])
for m, (gsx, (fname, name)) in enumerate(zip(gs_a, MODELS)):
    l, r = gsx.left, gsx.right
    n = misses.get(m, 0)
    fig.text((l + r) / 2, .935, name.replace('\n', ' '), ha='center',
             fontsize=8.5, color='.1', **BOLD)
    fig.text((l + r) / 2, .905, f'{n} miss' + ('es' if n != 1 else '')
             + ' of 12', ha='center', fontsize=7,
             color=('#b0453b' if n else '#2c7a52'))
    fig.text((l + r) / 2, .525, 'Stake (CHF)', ha='center', fontsize=8)

# ------------------------------------------------- panels B, C, D ----------
from tms_risk.behavior.fit_model import build_model, get_data
df = get_data('/data/ds-tmsrisk', model_label=ARGS.label)
model = build_model(ARGS.label, df)
subj = pd.read_csv(DATA / ARGS.draws, sep='\t')
SUBS = sorted(subj['subject'].unique())


def s_get(sub, var, reg):
    q = subj.query('subject == @sub and var == @var and regressor == @reg')
    return q.sort_values('draw')['value'].values


def curves(noise, vertex):
    spl = sorted(v for v in subj['var'].unique()
                 if v.startswith(noise + '_spline') and not v.endswith('_offset'))
    bas = np.asarray(model.make_dm(N_GRID, variable=noise))[:, :len(spl)]
    regs = ['Intercept'] + ([VC] if vertex else [])
    coef = np.stack([np.stack([sum(s_get(sub, v, rg) for rg in regs)
                               for v in spl], 1) for sub in SUBS], 0)
    return softplus(np.einsum('sdj,gj->sdg', coef, bas)).mean(0)


CH = [('memory_noise_sd', 'Memory', C_MEM, '-'),
      ('perceptual_noise_sd', 'Perceptual', C_PERC, '--')]
store = {(n, v): curves(n, v) for n, _, _, _ in CH for v in (False, True)}

gs_b = fig.add_gridspec(1, 3, left=.075, right=.985, top=.40, bottom=.09,
                        wspace=.46)


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


def band(ax, y, color, alpha=.13, ls='-'):
    med = np.median(y, 0)
    h95 = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    h50 = np.array([az.hdi(y[:, i], hdi_prob=.50) for i in range(y.shape[1])])
    ax.fill_between(N_GRID, h95[:, 0], h95[:, 1], color=color, alpha=alpha, lw=0)
    ax.fill_between(N_GRID, h50[:, 0], h50[:, 1], color=color, alpha=alpha * 2,
                    lw=0)
    ax.plot(N_GRID, med, color=color, ls=ls, lw=1.3)
    return med


ax = fig.add_subplot(gs_b[0, 0])
for noise, nm, _, ls in CH:
    for vertex, colr in [(True, VERTEX), (False, IPS)]:
        med = band(ax, store[(noise, vertex)], colr, alpha=.10, ls=ls)
    ax.text(N_GRID[3], med[3] + (.07 if 'memory' in noise else .05), nm,
            fontsize=6.5, color='.25')
logx(ax)
ax.set_ylabel('Noise SD (log units)')
ax.set_title('Noise by magnitude', fontsize=8.5, **BOLD)
ax.text(.97, .97, 'IPS', transform=ax.transAxes, fontsize=7, color=IPS,
        ha='right', va='top')
ax.text(.97, .86, 'Vertex', transform=ax.transAxes, fontsize=7, color=VERTEX,
        ha='right', va='top')

ax = fig.add_subplot(gs_b[0, 1])
ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
for noise, nm, colr, ls in CH:
    rel = store[(noise, False)] - store[(noise, True)]
    band(ax, rel, colr, alpha=.13, ls=ls)
    ax.plot([], [], color=colr, ls=ls, lw=1.3, label=nm)
ax.legend(loc='lower left', fontsize=6.5, handlelength=1.6)
logx(ax)
ax.set_ylabel('Δ noise SD, IPS − vertex\n(log units)', fontsize=7.5)
ax.set_title('Effect of cTBS', fontsize=8.5, **BOLD)

# ELPDs verified against notes/data/{lfx_verdicts,power_vs_flexible_ladder}.tsv
ax = fig.add_subplot(gs_b[0, 2])
rows = [('Log-flex, TMS perc. (bs2-m2-b)', -4148.6, 'fail', False),
        ('Log-flex, TMS both — PRIMARY', -4149.0, 'pass', True),
        ('Log-flex, TMS both (bs2-m3-bm)', -4155.0, 'pass', False),
        ('Flexible PMC, TMS perc.', -4157.5, None, False),
        ('Flexible PMC, TMS both', -4161.0, None, False),
        ('Log-flex, no TMS', -4245.3, None, False),
        ('Flexible PMC, no TMS', -4271.4, None, False)]
best = max(r[1] for r in rows)
for y, (name, e, verdict, is_primary) in enumerate(rows[::-1]):
    c = '#2c7a52' if verdict == 'pass' else ('#b0453b' if verdict == 'fail'
                                             else '.45')
    ax.plot(e - best, y, 'o', color=c, ms=5.5 if is_primary else 4)
    ax.text(e - best - 6, y + .3, name, fontsize=6.2, ha='right', va='center',
            color=('.1' if is_primary else '.35'),
            **(BOLD if is_primary else {}))
    if verdict:
        ax.text(e - best - 6, y - .28,
                'Δ-PPC 6/6' if verdict == 'pass' else 'Δ-PPC 4/6',
                fontsize=5.8, ha='right', va='center', color=c)
ax.axvline(0, color='.8', lw=.6, ls='--', zorder=0)
ax.set_xlim(-215, 30)
ax.set_ylim(-.7, len(rows) - .2)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('ELPD − best (nats)')
ax.set_title('Model comparison', fontsize=8.5, **BOLD)

for xf, yf, letter in [(.008, .955, 'A'), (.008, .44, 'B'), (.345, .44, 'C'),
                       (.665, .44, 'D')]:
    fig.text(xf, yf, letter, fontsize=11, va='bottom', ha='left', **BOLD)

sns.despine(fig=fig, offset=4)
fig.savefig(OUT / f'{ARGS.out}.pdf')
fig.savefig(OUT / f'{ARGS.out}.png', dpi=150)
print('wrote', OUT / f'{ARGS.out}.pdf', '| misses by model:', misses)
