"""Complete one-page account of the cTBS behavioural result.

Row 1  the two fitted noise terms, and what cTBS does to each.
Row 2  estimated parameters: where the priors sit, the noise/effect forest,
       and the per-subject cTBS effect.
Row 3  the consequences: perceived value, choices, model comparison.

Primary model lfx2-bs3-m2-dp-bm. Every model quantity is computed per
subject per draw and averaged across the 35 subjects within draw.

NOTE on the two noise terms: the model composes them as
sigma_1 = softplus(perceptual + memory) for the first-shown option and
sigma_2 = softplus(perceptual) for the second. The per-term curves in A/B
are softplus of each term alone and are therefore NOT additive; the
composed sigma_1 is what panel C's "total" line and panel F use.
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
DATA, OUT = ROOT / 'notes' / 'data', ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
    'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.4, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 400,
    'savefig.bbox': 'tight', 'savefig.pad_inches': .05,
})
IPS, VERTEX = '#d62728', '#2ca02c'
C_MEM, C_PER, C_TOT = '#33619e', '#8172B2', '#1a1a1a'
C_SAFE, C_RISKY = '#4A4A4A', '#8172B2'
PASSC, FAILC, STEP = '#2c7a52', '#b0453b', '#33619e'
B = dict(fontweight='bold', fontfamily='Arial')
VC = 'stimulation_condition[T.vertex]'
sp = lambda x: np.logaddexp(0, x)
LAB, N_DR = 'lfx2-bs3-m2-dp-bm', 200

from tms_risk.behavior.fit_model import build_model, get_data
df = get_data('/data/ds-tmsrisk', model_label=LAB)
model = build_model(LAB, df)
subj = pd.read_csv(DATA / 'm2bm_subject_draws.tsv.gz', sep='\t')
SUBS = sorted(subj['subject'].unique())
LO, HI = 7.001, 27.999
G = np.exp(np.linspace(np.log(LO), np.log(HI), 50))
IN = np.ones_like(G, bool)


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


PRE = {(ch, v): pre(ch, G, v) for ch in ('memory_noise_sd', 'perceptual_noise_sd')
       for v in (False, True)}
PRIOR = {v: np.stack([sg(s, v) for s in SUBS], 0)
         for v in ('risky_prior_mu', 'risky_prior_sd',
                   'safe_prior_mu', 'safe_prior_sd')}
nu1 = {v: sp(PRE[('perceptual_noise_sd', v)] + PRE[('memory_noise_sd', v)])
       for v in (False, True)}

fig = plt.figure(figsize=(7.4, 9.2))
gs = fig.add_gridspec(3, 3, left=.085, right=.985, top=.905, bottom=.055,
                      hspace=.62, wspace=.52)


def logx(ax, ticks=(7, 10, 14, 20, 28)):
    ax.set_xscale('log'); ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(LO * .96, HI * 1.04)


def band(ax, y, c, ls='-', a=.13):
    m = np.median(y, 0)
    h95 = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    h50 = np.array([az.hdi(y[:, i], hdi_prob=.50) for i in range(y.shape[1])])
    ax.fill_between(G, h95[:, 0], h95[:, 1], color=c, alpha=a, lw=0)
    ax.fill_between(G, h50[:, 0], h50[:, 1], color=c, alpha=a * 2, lw=0)
    ax.plot(G, m, color=c, ls=ls)
    return m


# ---- A / B : the two noise terms ----------------------------------------
for k, (ch, nm) in enumerate([('memory_noise_sd', 'Memory term'),
                              ('perceptual_noise_sd', 'Perceptual term')]):
    ax = fig.add_subplot(gs[0, k])
    for v, c in [(True, VERTEX), (False, IPS)]:
        band(ax, sp(PRE[(ch, v)]).mean(0), c)
    logx(ax); ax.set_ylabel('Noise SD (log units)')
    ax.set_xlabel('Payoff (CHF)')
    ax.set_title(nm, fontsize=8.5, pad=6, **B)
    if k == 0:
        ax.text(.04, .06, 'IPS', transform=ax.transAxes, color=IPS,
                fontsize=7.5, **B)
        ax.text(.04, .16, 'Vertex', transform=ax.transAxes, color=VERTEX,
                fontsize=7.5, **B)

# ---- C : the cTBS difference curves -------------------------------------
ax = fig.add_subplot(gs[0, 2])
DIFFS = [('memory_noise_sd', 'Memory', C_MEM, '-'),
         ('perceptual_noise_sd', 'Perceptual', C_PER, '--')]
for ch, nm, c, ls in DIFFS:
    band(ax, (sp(PRE[(ch, False)]) - sp(PRE[(ch, True)])).mean(0), c, ls=ls)
    ax.plot([], [], color=c, ls=ls, label=nm)
d_tot = (nu1[False] - nu1[True]).mean(0)
band(ax, d_tot, C_TOT, a=.16)
ax.plot([], [], color=C_TOT, label='Total (1st option)')
ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
ax.legend(loc='lower left', fontsize=6.6, handlelength=1.5)
logx(ax); ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Δ noise, IPS − vertex\n(log units)')
ax.set_title('Effect of cTBS on noise', fontsize=8.5, pad=6, **B)

# ---- D : where the priors sit -------------------------------------------
ax = fig.add_subplot(gs[1, 0])
pay = {'risky': np.where(df['p1'] == .55, df['n1'], df['n2']).astype(float),
       'safe': np.where(df['p1'] == .55, df['n2'], df['n1']).astype(float)}
for i, (role, c) in enumerate([('risky', C_RISKY), ('safe', C_SAFE)]):
    y = 1 - i
    ax.plot([pay[role].min(), pay[role].max()], [y + .22, y + .22], color=c,
            lw=3, alpha=.25, solid_capstyle='butt')
    mu = np.exp(PRIOR[f'{role}_prior_mu'].mean(0))
    h = az.hdi(mu, hdi_prob=.95)
    ax.plot([h[0], h[1]], [y, y], color=c, lw=2)
    ax.plot(np.median(mu), y, 'o', color=c, ms=6)
    w = np.exp(sp(PRIOR[f'{role}_prior_sd'].mean(0)))
    ax.text(6.0, y + .40, f'{role.capitalize()} option', ha='left',
            fontsize=7.2, color=c, **B)
    ax.text(6.0, y - .34, f'prior {np.median(mu):.0f} CHF, width ×/ '
            f'{np.median(w):.2f}', ha='left', fontsize=6.6, color=c)
ax.text(.98, .03, 'pale bar = payoffs shown', transform=ax.transAxes,
        fontsize=6.3, color='.5', va='bottom', ha='right')
ax.set_xscale('log'); ax.set_xticks([7, 14, 28, 56, 112])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlim(5.5, 190); ax.set_ylim(-.95, 1.95)
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('Payoff (CHF)')
ax.set_title('Where the priors sit', fontsize=8.5, pad=6, **B)

# ---- E : parameter forest (noise + effects) -----------------------------
ax = fig.add_subplot(gs[1, 1])


def at(ch, x, v):
    j = int(np.argmin(np.abs(G - x)))
    return sp(PRE[(ch, v)]).mean(0)[:, j]


ROWS_E = [
    ('Memory @ 7', at('memory_noise_sd', 7, True), C_MEM, False),
    ('Memory @ 28', at('memory_noise_sd', 28, True), C_MEM, False),
    ('Perceptual @ 7', at('perceptual_noise_sd', 7, True), C_PER, False),
    ('Perceptual @ 28', at('perceptual_noise_sd', 28, True), C_PER, False),
    ('Δ memory', (sp(PRE[('memory_noise_sd', False)])
                        - sp(PRE[('memory_noise_sd', True)])).mean(0).mean(1),
     C_MEM, True),
    ('Δ perceptual', (sp(PRE[('perceptual_noise_sd', False)])
                            - sp(PRE[('perceptual_noise_sd', True)])).mean(0).mean(1),
     C_PER, True),
    ('Δ total (1st opt.)', d_tot.mean(1), C_TOT, True),
]
for i, (nm, v, c, is_eff) in enumerate(ROWS_E):
    y = len(ROWS_E) - 1 - i
    h95, h50 = az.hdi(v, hdi_prob=.95), az.hdi(v, hdi_prob=.50)
    ax.plot(h95, [y, y], color=c, lw=1, alpha=.6)
    ax.plot(h50, [y, y], color=c, lw=2.6)
    ax.plot(np.median(v), y, 'o', color=c, ms=4.5)
    ax.text(-.22, y, nm, ha='right', va='center', fontsize=6.6, color='.15')
    if is_eff:
        ax.text(np.median(v) + .08, y, f'P>0 {float((v > 0).mean()):.2f}',
                ha='left', va='center', fontsize=6.4, color=c)
ax.axvline(0, color='.6', lw=.7, ls='--', zorder=0)
ax.axhline(2.5, color='.85', lw=.8)
ax.set_xlim(-1.15, 1.45)
ax.set_ylim(-.6, len(ROWS_E) - .3); ax.set_yticks([])
ax.set_xticks([0, .5, 1.])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Noise SD (log units)')
ax.set_title('Estimated parameters (group)', fontsize=8.5, pad=6, **B)

# ---- F : per-subject cTBS effect ----------------------------------------
ax = fig.add_subplot(gs[1, 2])
per_sub = (nu1[False] - nu1[True]).mean(2)          # (subject, draw)
med = np.median(per_sub, 1)
order = np.argsort(med)
for rank, si in enumerate(order):
    h = az.hdi(per_sub[si], hdi_prob=.5)
    c = IPS if med[si] > 0 else '.55'
    ax.plot([rank, rank], h, color=c, lw=1.1, alpha=.8)
    ax.plot(rank, med[si], 'o', color=c, ms=2.6)
gm = np.median(per_sub.mean(0))
ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
ax.axhline(gm, color=C_TOT, lw=1.2)
ax.text(.98, .96, f'{int((med > 0).sum())} of {len(SUBS)} subjects > 0',
        transform=ax.transAxes, ha='right', va='top', fontsize=7, **B)
ax.text(.98, .87, 'line = group mean', transform=ax.transAxes, ha='right',
        va='top', fontsize=6.5, color=C_TOT)
ax.set_xlabel('Subject (ranked)')
ax.set_ylabel('Δ total noise, IPS − vertex\n(log units)')
ax.set_title('Estimated parameters (individual)', fontsize=8.5, pad=6, **B)

# ---- G : perceived value -------------------------------------------------
from scipy.stats import norm
S5 = np.array([7., 10., 14., 20., 28.])


def percept(n, first, role, vertex):
    mu = PRIOR[f'{role}_prior_mu'][:, :, None]
    sd = sp(PRIOR[f'{role}_prior_sd'][:, :, None])
    p = pre('perceptual_noise_sd', n, vertex)
    v = sp(p + pre('memory_noise_sd', n, vertex)) if first else sp(p)
    w = sd ** 2 / (sd ** 2 + v ** 2)
    return w * np.log(n)[None, None, :] + (1 - w) * mu


ax = fig.add_subplot(gs[2, 0])
for role, first, c, nm in [('safe', True, C_SAFE, 'Safe (1st)'),
                           ('risky', False, C_RISKY, 'Risky (2nd)')]:
    n = S5 * 2.0 if role == 'risky' else S5
    pct = ((percept(n, first, role, False) - percept(n, first, role, True)) * 100).mean(0)
    m = np.median(pct, 0); lo, hi = np.percentile(pct, [25, 75], axis=0)
    ax.errorbar(S5, m, yerr=[m - lo, hi - m], fmt='o-', color=c, ms=3.5,
                elinewidth=.9, capsize=0)
    ax.text(.03, .93 if role == 'risky' else .07, nm, color=c, fontsize=7.2,
            transform=ax.transAxes, va='center', **B)
ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
logx(ax); ax.set_xlim(6.6, 30)
ax.set_xlabel('Safe payoff (CHF)')
ax.set_ylabel('Δ perceived value\nunder cTBS (log points)')
ax.set_title('Consequence for value', fontsize=8.5, pad=6, **B)

# ---- H : the choice effect ----------------------------------------------
e = pd.read_csv(DATA / f'ppc_delta_by_stake.{LAB}.tsv', sep='\t')
ax = fig.add_subplot(gs[2, 1])
xs = np.sort(e.stake.unique()); xi = np.arange(len(xs))
for order_, c, ls, nm in [('Risky second', '#1a1a1a', '-', 'Risky 2nd'),
                          ('Risky first', '#9a9a9a', '--', 'Risky 1st')]:
    s = e[e.order == order_].sort_values('stake')
    ax.fill_between(xi, s.lo, s.hi, color=c, alpha=.13, lw=0)
    ax.plot(xi, s['median'], color=c, ls=ls)
    off = .07 if order_ == 'Risky second' else -.07
    ax.errorbar(xi + off, s.obs, yerr=s.obs_sem, fmt='o', color=c, ms=4,
                elinewidth=1, capsize=0, zorder=4)
    ax.text(xi[-1] + .16, s.obs.values[-1], nm, color=c, fontsize=7,
            va='center', **B)
ax.axhline(0, color='.6', lw=.7, ls='--', zorder=0)
ax.set_xticks(xi); ax.set_xticklabels([f'{v:.0f}' for v in xs])
ax.set_xlim(-.4, len(xs) + .35); ax.set_ylim(-.062, .112)
ax.set_xlabel('Stake (CHF)')
ax.set_ylabel('Δ P(chose risky)\nIPS − vertex')
ax.set_title('Consequence for choice', fontsize=8.5, pad=6, **B)
ax.text(.5, .04, 'dots = data (±1 SEM), bands = model',
        transform=ax.transAxes, fontsize=6.4, ha='center', color='.4')

# ---- I : model comparison ------------------------------------------------
ax = fig.add_subplot(gs[2, 2])
ROWS = [('cTBS changes nothing', -4245.3, FAILC),
        ('Noise same at every\npayoff (Weber)', -4195.1, FAILC),
        ('cTBS changes payoff-\ndependent noise', -4149.0, PASSC)]
best = max(r[1] for r in ROWS)
for y, (nm, el, c) in enumerate(ROWS):
    dd = el - best
    ax.barh(y, dd, height=.45, color=c, alpha=.85)
    ax.text(4, y, nm, va='center', ha='left', fontsize=6.8, color='.1')
    ax.text(dd - 4, y, f'{dd:.0f}' if dd else 'best', va='center', ha='right',
            fontsize=7, color=c, **B)
ax.axvline(0, color='.75', lw=.7)
ax.set_xlim(-142, 118); ax.set_ylim(-.55, 2.6)
ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xticks([-100, -50, 0])
ax.set_xlabel('ELPD relative to best (nats)')
ax.set_title('Model comparison', fontsize=8.5, pad=6, **B)

for xf, yf, t in [(.012, .918, 'A'), (.325, .918, 'B'), (.645, .918, 'C'),
                  (.012, .607, 'D'), (.325, .607, 'E'), (.645, .607, 'F'),
                  (.012, .296, 'G'), (.325, .296, 'H'), (.645, .296, 'I')]:
    fig.text(xf, yf, t, fontsize=11, va='bottom', ha='left', **B)
fig.text(.085, .960, 'cTBS over parietal cortex adds noise to remembered '
         'amounts, and people gamble more', fontsize=11, va='bottom', **B)
fig.text(.085, .938, '35 subjects · within-subject IPS vs vertex · log-space '
         'PMC (lfx2-bs3-m2-dp-bm) · bands are 50% and 95% posterior intervals',
         fontsize=7.2, color='.4', va='bottom')

sns.despine(fig=fig, offset=3)
fig.savefig(OUT / 'story_full.pdf')
fig.savefig(OUT / 'story_full.png', dpi=180)
print('wrote', OUT / 'story_full.pdf')
