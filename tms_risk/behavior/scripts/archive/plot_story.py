"""One-figure summary of the cTBS behavioural result, as a causal chain.

1  cTBS raises noise in the magnitude representation (fitted noise functions).
2  More noise means the first-presented option is pulled harder toward the
   prior, so the risky option gains relative value.
3  That produces more gambling — but only when the risky option came second.
4  Model comparison supports each link.

Primary model: lfx2-bs3-m2-dp-bm (log-space PMC, magnitude-dependent
perceptual + memory noise, cTBS on both). All model quantities are computed
per subject per draw and averaged across the 35 subjects within draw.
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
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'libs' / 'bauer'))
DATA, OUT = ROOT / 'notes' / 'data', ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': .9, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3.5, 'ytick.major.size': 3.5,
    'lines.linewidth': 1.5, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .05,
})
IPS, VERTEX = '#d62728', '#2ca02c'
C_SAFE, C_RISKY = '#4A4A4A', '#8172B2'
PASSC, FAILC = '#2c7a52', '#b0453b'
STEP = '#33619e'
B = dict(fontweight='bold', fontfamily='Arial')
VC = 'stimulation_condition[T.vertex]'
sp = lambda x: np.logaddexp(0, x)
LAB = 'lfx2-bs3-m2-dp-bm'
THR = np.log(1 / .55)
N_DR = 200

from tms_risk.behavior.fit_model import build_model, get_data
df = get_data('/data/ds-tmsrisk', model_label=LAB)
model = build_model(LAB, df)
subj = pd.read_csv(DATA / 'm2bm_subject_draws.tsv.gz', sep='\t')
SUBS = sorted(subj['subject'].unique())


def sg(sub, v, r='Intercept'):
    q = subj.query('subject == @sub and var == @v and regressor == @r')
    return q.sort_values('draw')['value'].values[:N_DR]


def pre(noise, n, vertex):
    spl = sorted(v for v in subj['var'].unique()
                 if v.startswith(noise + '_spline') and not v.endswith('_offset'))
    bas = np.asarray(model.make_dm(n, variable=noise))[:, :len(spl)]
    regs = ['Intercept'] + ([VC] if vertex else [])
    c = np.stack([np.stack([sum(sg(s, v, r) for r in regs) for v in spl], 1)
                  for s in SUBS], 0)
    return np.einsum('sdj,gj->sdg', c, bas)


PRIOR = {v: np.stack([sg(s, v) for s in SUBS], 0)
         for v in ('risky_prior_mu', 'risky_prior_sd',
                   'safe_prior_mu', 'safe_prior_sd')}


def nu(n, first, vertex):
    p = pre('perceptual_noise_sd', n, vertex)
    return sp(p + pre('memory_noise_sd', n, vertex)) if first else sp(p)


def percept(n, first, role, vertex):
    mu = PRIOR[f'{role}_prior_mu'][:, :, None]
    sd = sp(PRIOR[f'{role}_prior_sd'][:, :, None])
    v = nu(n, first, vertex)
    w = sd ** 2 / (sd ** 2 + v ** 2)
    return w * np.log(n)[None, None, :] + (1 - w) * mu


fig = plt.figure(figsize=(7.4, 6.4))
gs = fig.add_gridspec(2, 2, left=.085, right=.985, top=.845, bottom=.075,
                      hspace=.62, wspace=.34)

# ---- 1 : noise -----------------------------------------------------------
G = np.exp(np.linspace(np.log(7.001), np.log(27.999), 60))
ax = fig.add_subplot(gs[0, 0])
for vtx, c, nm in [(True, VERTEX, 'Vertex (control)'), (False, IPS, 'IPS (stimulated)')]:
    y = nu(G, True, vtx).mean(0)
    m = np.median(y, 0)
    h = np.array([az.hdi(y[:, i], hdi_prob=.5) for i in range(y.shape[1])])
    ax.fill_between(G, h[:, 0], h[:, 1], color=c, alpha=.22, lw=0)
    ax.plot(G, m, color=c)
d = (nu(G, True, False) - nu(G, True, True)).mean(0)
ax.set_xscale('log'); ax.set_xticks([7, 10, 14, 20, 28])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlim(6.8, 29.5)
ax.text(.62, .93, 'IPS (stimulated)', transform=ax.transAxes, fontsize=8,
        color=IPS, ha='right', **B)
ax.text(.62, .83, 'Vertex (control)', transform=ax.transAxes, fontsize=8,
        color=VERTEX, ha='right', **B)
ax.set_xlabel('Payoff (CHF)'); ax.set_ylabel('Noise in the remembered\namount (log units)')
ax.text(.03, .04, f'cTBS effect  +{np.median(d.mean(1)):.3f} log units\n'
        f'P(increase) = {float((d.mean(1) > 0).mean()):.2f}',
        transform=ax.transAxes, fontsize=7.4, color=IPS, va='bottom')

# ---- 2 : perceived value -------------------------------------------------
S5 = np.array([7., 10., 14., 20., 28.])
ax = fig.add_subplot(gs[0, 1])
for role, first, c, nm in [('safe', True, C_SAFE, 'Safe option\n(shown first)'),
                           ('risky', False, C_RISKY, 'Risky option\n(shown second)')]:
    n = S5 * 2.0 if role == 'risky' else S5
    pct = ((percept(n, first, role, False) - percept(n, first, role, True)) * 100).mean(0)
    m = np.median(pct, 0)
    lo, hi = np.percentile(pct, [25, 75], axis=0)
    ax.errorbar(S5, m, yerr=[m - lo, hi - m], fmt='o-', color=c, ms=4,
                elinewidth=1, capsize=0)
    ax.text(S5[-1] * 1.06, m[-1], nm, color=c, fontsize=7.4, va='center', **B)
ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
ax.set_xscale('log'); ax.set_xticks(S5)
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlim(6.4, 47)
ax.set_xlabel('Safe payoff (CHF)')
ax.set_ylabel('Change in perceived value\nunder cTBS (log points)')

# ---- 3 : choice ----------------------------------------------------------
e = pd.read_csv(DATA / f'ppc_delta_by_stake.{LAB}.tsv', sep='\t')
ax = fig.add_subplot(gs[1, 0])
xs = np.sort(e.stake.unique()); x = np.arange(len(xs))
for order, c, ls, nm in [('Risky second', '#1a1a1a', '-', 'Risky option second'),
                         ('Risky first', '#9a9a9a', '--', 'Risky option first')]:
    s = e[e.order == order].sort_values('stake')
    ax.fill_between(x, s.lo, s.hi, color=c, alpha=.13, lw=0)
    ax.plot(x, s['median'], color=c, ls=ls, lw=1.4)
    off = .08 if order == 'Risky second' else -.08
    ax.errorbar(x + off, s.obs, yerr=s.obs_sem, fmt='o', color=c, ms=5,
                elinewidth=1.1, capsize=0, zorder=4)
    ax.text(x[-1] + .17, s.obs.values[-1], nm, color=c, fontsize=7.4,
            va='center', **B)
ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
ax.set_xticks(x); ax.set_xticklabels([f'{v:.0f}' for v in xs])
ax.set_xlim(-.4, len(xs) + .55)
ax.set_xlabel('Stake (CHF)'); ax.set_ylabel('More gambling under cTBS\nΔ P(chose risky)')
ax.set_ylim(-.062, .112)
ax.text(.97, .04, 'Dots = data (±1 SEM)   Bands = model',
        transform=ax.transAxes, fontsize=7.2, va='bottom', ha='right',
        color='.35')

# ---- 4 : evidence --------------------------------------------------------
ax = fig.add_subplot(gs[1, 1])
ROWS = [('cTBS changes nothing', -4245.3, FAILC),
        ('Noise is the same at\nevery payoff (Weber)', -4195.1, FAILC),
        ('cTBS changes payoff-\ndependent noise', -4149.0, PASSC)]
best = max(r[1] for r in ROWS)
ys = np.arange(len(ROWS))
for y, (nm, el, c) in zip(ys, ROWS):
    dd = el - best
    ax.barh(y, dd, height=.46, color=c, alpha=.85)
    ax.text(4, y, nm, va='center', ha='left', fontsize=7.8, color='.1')
    if dd < 0:
        ax.text(dd - 4, y, f'{dd:.0f}', va='center', ha='right', fontsize=7.8,
                color=c, **B)
    else:
        ax.text(-4, y, 'best', va='center', ha='right', fontsize=7.8,
                color=c, **B)
ax.set_xlim(-132, 108); ax.set_ylim(-.55, 2.6)
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xticks([-100, -50, 0])
ax.axvline(0, color='.75', lw=.7)
ax.set_xlabel('Predictive accuracy\n(ELPD relative to best, nats)')

# ---- captions ------------------------------------------------------------
STEPS = [('1', 'cTBS raises memory noise',
          'The first-shown option is held in memory;\n'
          'stimulating IPS makes it noisier.'),
         ('2', 'The gamble gains relative value',
          'A noisier memory is pulled toward the prior,\n'
          'costing the safe option perceived value.'),
         ('3', 'More gambling — in one order only',
          'Present when the risky option comes second.\n'
          'The model reproduces all six cells.'),
         ('4', 'Model comparison backs each link',
          'Noise must depend on payoff, and cTBS must\n'
          'be allowed to change it.')]
POS = [(.105, .905), (.565, .905), (.105, .445), (.565, .445)]
for (num, head, sub), (px, py) in zip(STEPS, POS):
    fig.text(px - .050, py + .004, num, fontsize=15, color=STEP, va='bottom', **B)
    fig.text(px, py + .022, head, fontsize=9.4, va='bottom', **B)
    fig.text(px, py + .012, sub, fontsize=7.3, color='.35', va='top')

fig.text(.085, .975, 'cTBS over parietal cortex makes people gamble more '
         'by adding noise to remembered amounts', fontsize=11, va='bottom', **B)
fig.text(.085, .955, '35 subjects · within-subject IPS vs vertex · log-space '
         'Perceptual-and-Memory-based Choice model (lfx2-bs3-m2-dp-bm)',
         fontsize=7.6, color='.4', va='bottom')

sns.despine(fig=fig, offset=4)
fig.savefig(OUT / 'story.pdf')
fig.savefig(OUT / 'story.png', dpi=200)
print('wrote', OUT / 'story.pdf')
