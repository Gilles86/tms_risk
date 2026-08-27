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
# Panel A shows the cTBS CONTRAST, not the two conditions separately. Overlapping
# P(chose risky) curves hide a 0.06 effect inside two 0.1-wide bands; the difference
# against zero is the quantity the claim is actually about, and it is what separates
# the models. All three are log-space, so the comparison isolates what the flexible
# noise function and the cTBS term each buy, rather than confounding it with scale.
MODELS = [('ppc_delta_by_stake.lfx2-bs3-w-dp-bm.tsv',
           'Log-Weber PMC\n(constant noise)'),
          ('ppc_delta_by_stake.lfx2-bs3-m2-dp-null.tsv',
           'Log-flexible PMC\n(no cTBS effect)'),
          (f'ppc_delta_by_stake.{ARGS.label}.tsv',
           ARGS.name.replace(' (', '\n('))]
frames = {i: pd.read_csv(DATA / f, sep='\t') for i, (f, _) in enumerate(MODELS)}
allv = pd.concat(frames.values())
ylo = min(allv.lo.min(), allv.obs.min()) - .012
yhi = max(allv.hi.max(), allv.obs.max()) + .012
stakes = np.sort(allv.stake.unique())
x = np.arange(len(stakes))

fig = plt.figure(figsize=(7.25, 5.1))
gs_a = [fig.add_gridspec(1, 2, left=l, right=r, wspace=.12, top=.855, bottom=.60)
        for l, r in [(.065, .345), (.39, .67), (.715, .995)]]
axes_a = []
for m in range(3):
    axes_a += [fig.add_subplot(gs_a[m][0, 0]), fig.add_subplot(gs_a[m][0, 1])]

misses = {}
for m, (fname, name) in enumerate(MODELS):
    d = frames[m]
    for o, order in enumerate(ORDERS):
        ax = axes_a[2 * m + o]
        s_ = d[d.order == order].sort_values('stake')
        ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
        ax.fill_between(x, s_.lo, s_.hi, color='.45', alpha=.22, lw=0, zorder=1)
        ax.plot(x, s_['median'], color='.25', lw=1.3, zorder=2)
        ax.plot(x, s_.obs, 'o', color=IPS, ms=4.2, lw=0, zorder=4)
        for xi, (_, r) in zip(x, s_.iterrows()):
            if r.lo <= r.obs <= r.hi:
                continue
            ax.annotate('', xy=(xi, r.obs), xycoords='data',
                        xytext=(-13, 11 if r.obs > r.hi else -11),
                        textcoords='offset points', zorder=6,
                        arrowprops=dict(arrowstyle='-|>', color='.1', lw=.9,
                                        shrinkA=0, shrinkB=3.5, mutation_scale=6))
            misses[m] = misses.get(m, 0) + 1
        ax.set_ylim(ylo, yhi)
        ax.set_xlim(-.42, len(stakes) - .58)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_title(order, fontsize=7, color='.3', pad=3, style='italic')
        if 2 * m + o == 0:
            ax.set_ylabel('\u0394 P(chose risky)\nIPS \u2212 vertex', fontsize=7.5)
            ax.text(.05, .96, 'Observed', transform=ax.transAxes, fontsize=6.5,
                    color=IPS, va='top')
            ax.text(.05, .84, 'Model', transform=ax.transAxes, fontsize=6.5,
                    color='.25', va='top')
        else:
            ax.set_yticklabels([])
for m, (gsx, (fname, name)) in enumerate(zip(gs_a, MODELS)):
    l, r = gsx.left, gsx.right
    n = misses.get(m, 0)
    fig.text((l + r) / 2, .985, name, ha='center', va='top', linespacing=1.35,
             fontsize=8, color='.1', **BOLD)
    fig.text((l + r) / 2, .878, f'{n} miss' + ('es' if n != 1 else '') + ' of 6',
             ha='center', fontsize=7,
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


def _pre(noise, vertex):
    """Pre-softplus spline contribution, (subject, draw, grid)."""
    spl = sorted(v for v in subj['var'].unique()
                 if v.startswith(noise + '_spline') and not v.endswith('_offset'))
    bas = np.asarray(model.make_dm(N_GRID, variable=noise))[:, :len(spl)]
    regs = ['Intercept'] + ([VC] if vertex else [])
    coef = np.stack([np.stack([sum(s_get(sub, v, rg) for rg in regs)
                               for v in spl], 1) for sub in SUBS], 0)
    return np.einsum('sdj,gj->sdg', coef, bas)


def curves(noise, vertex):
    return softplus(_pre(noise, vertex)).mean(0)


def nu1(vertex):
    """Total noise on the FIRST-presented option: softplus(perceptual +
    memory), matching bauer's _get_trialwise_evidence_sd -- NOT the sum of
    the two softplus curves."""
    return softplus(_pre('perceptual_noise_sd', vertex)
                    + _pre('memory_noise_sd', vertex)).mean(0)


CH = [('memory_noise_sd', 'Memory', C_MEM, '-'),
      ('perceptual_noise_sd', 'Perceptual', C_PERC, '--')]
store = {(n, v): curves(n, v) for n, _, _, _ in CH for v in (False, True)}

gs_b = fig.add_gridspec(1, 4, left=.075, right=.985, top=.40, bottom=.09,
                        wspace=.55)


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


# Memory and perceptual noise live an order of magnitude apart (memory ~0.7-1.1,
# perceptual ~0.15-0.3 log units), so a shared axis flattens the perceptual curve
# into a line at the floor and hides its cTBS effect entirely. One panel each.
for col, (noise, nm, _c, _ls) in enumerate(CH):
    ax = fig.add_subplot(gs_b[0, col])
    for vertex, colr in [(True, VERTEX), (False, IPS)]:
        band(ax, store[(noise, vertex)], colr, alpha=.13, ls='-')
    logx(ax)
    ax.set_ylabel('Noise SD (log units)')
    ax.set_title(f'{nm} noise', fontsize=8.5, **BOLD)
    if col == 0:
        ax.text(.97, .97, 'IPS', transform=ax.transAxes, fontsize=7, color=IPS,
                ha='right', va='top')
        ax.text(.97, .85, 'Vertex', transform=ax.transAxes, fontsize=7,
                color=VERTEX, ha='right', va='top')

# The cTBS panel shows the TOTAL only. Per-channel differences are ten times
# wider than the effect being claimed -- the memory channel has 2 df and is
# identified only through the first-presented option -- so plotting them here
# reads as "no effect" regardless of what the integrated posterior says.
ax = fig.add_subplot(gs_b[0, 2])
ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
rel_tot = nu1(False) - nu1(True)
band(ax, rel_tot, '.15', alpha=.16, ls='-')
_lo, _hi = 7.001, 28.0
_sel = (N_GRID >= _lo) & (N_GRID <= _hi)
_m = rel_tot[:, _sel].mean(1)
_med, _q = float(np.median(_m)), np.percentile(_m, [2.5, 97.5])
ax.axvspan(_lo, _hi, color='.93', zorder=0)
ax.text(.03, .97,
        f'7-28 CHF: {_med:+.3f}\n[{_q[0]:+.3f}, {_q[1]:+.3f}]\nP(>0) = {float((_m > 0).mean()):.2f}',
        transform=ax.transAxes, fontsize=6.4, va='top', color='.15',
        linespacing=1.35)
logx(ax)
ax.set_ylabel('\u0394 noise SD (log units)', fontsize=7.5)
ax.set_title('cTBS effect, total noise', fontsize=8.5, **BOLD)

# ELPDs from notes/data/ladder_v11.tsv (paired dSE, identical 8335 trials).
ax = fig.add_subplot(gs_b[0, 3])
rows = [('Log-flex, TMS perc. (bs2-m2-b)', -4148.6, 'fail', False),
        ('Log-flex, TMS both \u2014 PRIMARY', -4149.0, 'pass', True),
        ('Log-flex, TMS both (bs2-m3-bm)', -4155.0, 'pass', False),
        ('Flexible PMC, natural space', -4184.6, None, False),
        ('Log-flex, no TMS', -4245.3, None, False),
        ('Natural space, no TMS', -4271.8, None, False)]
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
                '\u0394-PPC 6/6' if verdict == 'pass' else '\u0394-PPC 4/6',
                fontsize=5.8, ha='right', va='center', color=c)
ax.axvline(0, color='.8', lw=.6, ls='--', zorder=0)
ax.set_xlim(-215, 30)
ax.set_ylim(-.7, len(rows) - .2)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('ELPD \u2212 best (nats)')
ax.set_title('Model comparison', fontsize=8.5, **BOLD)

for xf, yf, letter in [(.008, .955, 'A'), (.008, .44, 'B'), (.255, .44, 'C'),
                       (.505, .44, 'D'), (.755, .44, 'E')]:
    fig.text(xf, yf, letter, fontsize=11, va='bottom', ha='left', **BOLD)

sns.despine(fig=fig, offset=4)
fig.savefig(OUT / f'{ARGS.out}.pdf')
fig.savefig(OUT / f'{ARGS.out}.png', dpi=150)
print('wrote', OUT / f'{ARGS.out}.pdf', '| misses by model:', misses)
