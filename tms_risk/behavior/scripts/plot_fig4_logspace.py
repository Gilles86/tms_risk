"""Figure 4 draft comparing the two log-space PMC candidates (exploratory).

A: PPC by stake — Weber | Log-flexible with scalar memory (grid winner
lfx2-bs3-sm-dp-b) | Log-flexible2 (both noises flexible, TMS on both) —
paper miss-arrow convention. B/C: subject-averaged noise functions of the
two log-space models (perceptual by stimulation, memory in gray). D:
relative cTBS effect on perceptual noise, both models overlaid. E: ladder
with the delta-band PPC verdicts.
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
C_WIN, C_LF2 = '#33619E', '#8172B2'
BOLD = dict(fontweight='bold', fontfamily='Arial')
ORDERS = ['Risky first', 'Risky second']
VC = 'stimulation_condition[T.vertex]'

# ---------------------------------------------------------------- panel A --
MODELS = [('weber2nf', 'Weber PMC'),
          ('lfx2winner', 'Log-flexible, scalar mem.'),
          ('logflex2', 'Log-flexible, both flex.')]
frames = {i: pd.read_csv(DATA / f'ppc_by_stake.{lbl}.tsv', sep='\t')
          for i, (lbl, _) in enumerate(MODELS)}
allv = pd.concat(frames.values())
ylo = min(allv.lo.min(), allv.observed.min()) - .012
yhi = max(allv.hi.max(), allv.observed.max()) + .012
stakes = np.sort(allv.stake.unique())
x = np.arange(len(stakes))

fig = plt.figure(figsize=(7.25, 5.0))
gs_a = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.12, top=.90,
                         bottom=.64)
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
    fig.text((l + r) / 2, .555, 'Stake (CHF)', ha='center', fontsize=8)
axes_a[0].text(.06, .96, 'IPS', transform=axes_a[0].transAxes, fontsize=7,
               color=IPS, va='top')
axes_a[0].text(.06, .84, 'Vertex', transform=axes_a[0].transAxes, fontsize=7,
               color=VERTEX, va='top')

# ------------------------------------------- panels B, C, D, E -------------
from tms_risk.behavior.fit_model import build_model, get_data

softplus = lambda x: np.logaddexp(0, x)
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 80))


def load_curves(fit_label, draws_file):
    """Subject-averaged (draws, grid) noise curves for one log-space model."""
    df = get_data('/data/ds-tmsrisk', model_label=fit_label)
    model = build_model(fit_label, df)
    subj = pd.read_csv(DATA / draws_file, sep='\t')
    subs = sorted(subj['subject'].unique())

    def s_get(sub, var, reg):
        q = subj.query('subject == @sub and var == @var and regressor == @reg')
        return q.sort_values('draw')['value'].values

    out = {}
    for noise in ('perceptual_noise_sd', 'memory_noise_sd'):
        spl_vars = sorted(v for v in subj['var'].unique()
                          if v.startswith(noise + '_spline')
                          and not v.endswith('_offset'))
        n_spl = len(spl_vars)
        bas = np.asarray(model.make_dm(N_GRID, variable=noise))
        regs = set(subj.loc[subj['var'] == spl_vars[0], 'regressor'])
        for reg_set, key in [(['Intercept'], 'ips'),
                             (['Intercept', VC], 'vertex')]:
            if key == 'vertex' and VC not in regs:
                out[(noise, 'vertex')] = out[(noise, 'ips')]
                continue
            coef = np.stack([np.stack(
                [sum(s_get(sub, f'{noise}_spline{j}', r) for r in reg_set)
                 for j in range(1, n_spl + 1)], 1) for sub in subs], 0)
            out[(noise, key)] = softplus(
                np.einsum('sdj,gj->sdg', coef, bas[:, :n_spl])).mean(0)
    return out


CURVES = {'lfx2winner': load_curves('lfx2-bs3-sm-dp-b',
                                    'winner_subject_draws.tsv.gz'),
          'logflex2': load_curves('logflex2', 'logflex2_subject_draws.tsv.gz')}

gs_b = fig.add_gridspec(1, 4, left=.075, right=.985, top=.44, bottom=.10,
                        wspace=.55)


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


def band(ax, y, color, alpha=.18, ls='-', lw=1.3):
    med = np.median(y, 0)
    h = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    ax.fill_between(N_GRID, h[:, 0], h[:, 1], color=color, alpha=alpha, lw=0)
    ax.plot(N_GRID, med, color=color, lw=lw, ls=ls)
    return med


NOISE_TITLES = {'lfx2winner': 'Scalar-memory model:\nnoise functions',
                'logflex2': 'Both-flexible model:\nnoise functions'}
ylim_bc = [np.inf, -np.inf]
axes_bc = {}
for col, key in enumerate(['lfx2winner', 'logflex2']):
    ax = fig.add_subplot(gs_b[0, col])
    axes_bc[key] = ax
    cv = CURVES[key]
    for stim, colr in [(('perceptual_noise_sd', 'vertex'), VERTEX),
                       (('perceptual_noise_sd', 'ips'), IPS)]:
        band(ax, cv[stim], colr)
    mem_med = band(ax, cv[('memory_noise_sd', 'vertex')], '.5', alpha=.12,
                   ls='--', lw=1.1)
    ax.text(N_GRID[4], mem_med[4] + .02, 'Memory', fontsize=6.5, color='.4')
    logx(ax)
    ax.set_title(NOISE_TITLES[key], fontsize=8, **BOLD)
    if col == 0:
        ax.set_ylabel('Noise SD (log units)')
    ylim_bc = [min(ylim_bc[0], ax.get_ylim()[0]),
               max(ylim_bc[1], ax.get_ylim()[1])]
for ax in axes_bc.values():
    ax.set_ylim(*ylim_bc)

ax = fig.add_subplot(gs_b[0, 2])
ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
for key, colr, name in [('lfx2winner', C_WIN, 'Scalar mem.'),
                        ('logflex2', C_LF2, 'Both flex.')]:
    cv = CURVES[key]
    y_i = cv[('perceptual_noise_sd', 'ips')]
    y_v = cv[('perceptual_noise_sd', 'vertex')]
    band(ax, (y_i - y_v) / y_v * 100, colr, alpha=.14)
    ax.plot([], [], color=colr, lw=1.3, label=name)
ax.legend(loc='upper left', fontsize=6.5, handlelength=1.2)
logx(ax)
ax.set_ylabel('Δ perceptual noise,\nIPS − vertex (%)')
ax.set_title('Effect of cTBS\non noise', fontsize=8, **BOLD)

# ELPDs: power_vs_flexible_ladder.tsv, lfxgrid_loo.tsv, VM logflex ladder
ax = fig.add_subplot(gs_b[0, 3])
rows = [('Flexible PMC (TMS: perception)', -4157.5, None),
        ('Flexible PMC (TMS: both)', -4161.0, None),
        ('Log-flex, scalar mem. (TMS: perc.)', -4162.7, 'FAIL 0.67'),
        ('Log-flex, scalar mem. (TMS: both)', -4163.5, 'FAIL 0.67'),
        ('Log-flex, both flex. (TMS: both)', -4175.4, 'PASS 1.0'),
        ('Log-flex, scalar mem. (null)', -4260.4, None),
        ('Flexible PMC (null)', -4271.4, None)]
best = max(r[1] for r in rows)
for y, (name, e, ppc) in enumerate(rows[::-1]):
    c = C_WIN if 'scalar' in name else C_LF2 if 'both flex' in name else '.45'
    ax.plot(e - best, y, 'o', color=c, ms=4)
    ax.text(e - best - 8, y + .26, name, fontsize=6, ha='right',
            va='center', color=c)
    if ppc:
        ax.text(e - best - 8, y - .26, f'Δ-PPC {ppc}', fontsize=5.8,
                ha='right', va='center',
                color='#2ca02c' if 'PASS' in ppc else '#d62728')
ax.axvline(0, color='.8', lw=.6, ls='--', zorder=0)
ax.set_xlim(-290, 40)
ax.set_ylim(-.6, len(rows) - .2)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('ELPD − best (nats)')
ax.set_title('Model\ncomparison', fontsize=8, **BOLD)

for xf, yf, letter in [(.008, .965, 'A'), (.008, .48, 'B'), (.255, .48, 'C'),
                       (.505, .48, 'D'), (.745, .48, 'E')]:
    fig.text(xf, yf, letter, fontsize=11, va='bottom', ha='left', **BOLD)

sns.despine(fig=fig, offset=4)
fig.savefig(OUT / 'fig4_logspace.pdf')
fig.savefig(OUT / 'fig4_logspace.png', dpi=150)
print('wrote', OUT / 'fig4_logspace.pdf', '| misses:', misses)
