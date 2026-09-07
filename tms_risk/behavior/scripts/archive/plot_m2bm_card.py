"""Card figure for lfx2-bs3-m2-dp-bm — the minimal log-space PMC currently
leading the program: linear (2-df) memory noise, 5-df perceptual spline,
TMS on both channels. A: by-stake PPC (fig-4A format). B: delta-band PPC on
the IPS-vertex contrast by stake tercile (the asymmetry criterion). C: the
two noise channels by stimulation. D/E: cTBS contrasts in log units.
All curves subject-averaged (mean over subject curves within draw)."""
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
LABEL = 'lfx2-bs3-m2-dp-bm'

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
softplus = lambda x: np.logaddexp(0, x)
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 80))

fig = plt.figure(figsize=(7.25, 5.0))

# ---------------------------------------------------------------- panel A --
d = pd.read_csv(DATA / f'ppc_by_stake.{LABEL}.tsv', sep='\t')
stakes = np.sort(d.stake.unique())
x = np.arange(len(stakes))
ylo = min(d.lo.min(), d.observed.min()) - .012
yhi = max(d.hi.max(), d.observed.max()) + .012
gs_a = fig.add_gridspec(1, 2, left=.065, right=.46, wspace=.12, top=.90,
                        bottom=.62)
for o, order in enumerate(ORDERS):
    ax = fig.add_subplot(gs_a[0, o])
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
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(-.42, len(stakes) - .58)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:.0f}' for v in stakes])
    ax.set_yticks([.5, .55, .6, .65])
    ax.set_title(order, fontsize=7, color='.3', pad=3, style='italic')
    ax.set_xlabel('Stake (CHF)')
    if o == 0:
        ax.set_ylabel('P(chose risky)')
        ax.text(.06, .96, 'IPS', transform=ax.transAxes, fontsize=7,
                color=IPS, va='top')
        ax.text(.06, .84, 'Vertex', transform=ax.transAxes, fontsize=7,
                color=VERTEX, va='top')
    else:
        ax.set_yticklabels([])
fig.text(.2625, .955, 'Choice PPC by stake', ha='center', fontsize=8.5,
         **BOLD)

# ---------------------------------------------------------------- panel B --
e = pd.read_csv(DATA / f'ppc_delta_by_stake.{LABEL}.tsv', sep='\t')
gs_b = fig.add_gridspec(1, 2, left=.56, right=.985, wspace=.12, top=.90,
                        bottom=.62)
elo = min(e.lo.min(), (e.obs - e.obs_sem).min()) - .01
ehi = max(e.hi.max(), (e.obs + e.obs_sem).max()) + .01
for o, order in enumerate(ORDERS):
    ax = fig.add_subplot(gs_b[0, o])
    s = e[e.order == order].sort_values('stake')
    ax.fill_between(x, s.lo, s.hi, color='.85', lw=0)
    ax.plot(x, s['median'], color='.35', lw=1.4)
    ax.errorbar(x, s.obs, yerr=s.obs_sem, fmt='o', color='.1', ms=3.6,
                elinewidth=.8, capsize=0, zorder=4)
    ax.axhline(0, color='.8', lw=.5, ls='--', zorder=0)
    ax.set_ylim(elo, ehi)
    ax.set_xlim(-.42, len(stakes) - .58)
    ax.set_xticks(x)
    ax.set_xticklabels([f'{v:.0f}' for v in stakes])
    ax.set_title(order, fontsize=7, color='.3', pad=3, style='italic')
    ax.set_xlabel('Stake (CHF)')
    if o == 0:
        ax.set_ylabel('Δ P(chose risky)\nIPS − vertex')
    else:
        ax.set_yticklabels([])
fig.text(.7725, .955, 'cTBS contrast PPC (coverage 6/6)', ha='center',
         fontsize=8.5, **BOLD)

# --------------------------------------------- panels C, D, E --------------
from tms_risk.behavior.fit_model import build_model, get_data
df = get_data('/data/ds-tmsrisk', model_label=LABEL)
model = build_model(LABEL, df)
subj = pd.read_csv(DATA / 'm2bm_subject_draws.tsv.gz', sep='\t')
SUBS = sorted(subj['subject'].unique())


def s_get(sub, var, reg):
    q = subj.query('subject == @sub and var == @var and regressor == @reg')
    return q.sort_values('draw')['value'].values


def curves(noise, vertex):
    spl = sorted(v for v in subj['var'].unique()
                 if v.startswith(noise + '_spline')
                 and not v.endswith('_offset'))
    bas = np.asarray(model.make_dm(N_GRID, variable=noise))[:, :len(spl)]
    regs = ['Intercept'] + ([VC] if vertex else [])
    coef = np.stack([np.stack(
        [sum(s_get(sub, v, rg) for rg in regs) for v in spl], 1)
        for sub in SUBS], 0)
    return softplus(np.einsum('sdj,gj->sdg', coef, bas)).mean(0)


store = {(n, v): curves(n, v)
         for n in ('memory_noise_sd', 'perceptual_noise_sd')
         for v in (False, True)}


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


def band(ax, y, color, alpha=.14, ls='-'):
    med = np.median(y, 0)
    h95 = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    h50 = np.array([az.hdi(y[:, i], hdi_prob=.50) for i in range(y.shape[1])])
    ax.fill_between(N_GRID, h95[:, 0], h95[:, 1], color=color, alpha=alpha,
                    lw=0)
    ax.fill_between(N_GRID, h50[:, 0], h50[:, 1], color=color,
                    alpha=alpha * 2, lw=0)
    ax.plot(N_GRID, med, color=color, ls=ls)


gs_c = fig.add_gridspec(1, 3, left=.075, right=.985, top=.44, bottom=.10,
                        wspace=.48)
ax = fig.add_subplot(gs_c[0, 0])
for noise, ls in [('memory_noise_sd', '-'), ('perceptual_noise_sd', '--')]:
    for vertex, colr in [(True, VERTEX), (False, IPS)]:
        band(ax, store[(noise, vertex)], colr, alpha=.10, ls=ls)
ax.text(N_GRID[4], 1.13, 'Memory (linear)', fontsize=6.5, color='.25')
ax.text(N_GRID[4], 0.06, 'Perceptual (5-df spline)', fontsize=6.5,
        color='.25')
logx(ax)
ax.set_ylabel('Noise SD (log units)')
ax.set_title('The two noise channels', fontsize=8.5, **BOLD)
ax.text(.97, .97, 'IPS', transform=ax.transAxes, fontsize=7, color=IPS,
        ha='right', va='top')
ax.text(.97, .87, 'Vertex', transform=ax.transAxes, fontsize=7,
        color=VERTEX, ha='right', va='top')

for k, (noise, ttl) in enumerate([('memory_noise_sd', 'memory'),
                                  ('perceptual_noise_sd', 'perceptual')]):
    ax = fig.add_subplot(gs_c[0, 1 + k])
    rel = store[(noise, False)] - store[(noise, True)]
    band(ax, rel, '#8172B2')
    ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
    logx(ax)
    ax.set_ylabel('Δ noise SD, IPS − vertex\n(log units)', fontsize=7.5)
    ax.set_title(f'cTBS effect: {ttl}', fontsize=8.5, **BOLD)
    p_pos = float((rel.mean(1) > 0).mean())
    ax.text(.03, .97, f'P(mean Δ > 0) = {p_pos:.2f}',
            transform=ax.transAxes, fontsize=7, va='top', color='.25')

for xf, yf, letter in [(.008, .965, 'A'), (.505, .965, 'B'),
                       (.008, .48, 'C'), (.375, .48, 'D'), (.69, .48, 'E')]:
    fig.text(xf, yf, letter, fontsize=11, va='bottom', ha='left', **BOLD)
fig.text(.008, .005,
         'Linear-memory log-flexible PMC (m2-bm): ELPD −4149.0 (best; '
         'Flexible PMC −4157.5), r̂ 1.002, ESS 2462, Δ-PPC coverage '
         '6/6, 1/12 by-stake miss', fontsize=6.5, color='.35')

sns.despine(fig=fig, offset=4)
fig.savefig(OUT / 'm2bm_card.pdf')
fig.savefig(OUT / 'm2bm_card.png', dpi=150)
print('wrote', OUT / 'm2bm_card.pdf')
