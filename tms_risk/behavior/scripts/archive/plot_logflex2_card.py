"""Model card for logflex2 (both noise functions flexible, TMS on both).

Same layout family as winner_lfx2: priors, noise curves by condition (now
both terms carry the TMS regressor), contrasts, forests, per-subject panels.
CAVEAT baked into the title: this fit has r-hat ~2 on the prior means (the
fm-family wandering direction), so prior-location bands are inflated.
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
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 9,
    'mathtext.fontset': 'stixsans',
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_IPS, C_VERTEX = '#d62728', '#2ca02c'
VC = 'stimulation_condition[T.vertex]'
LABEL = 'logflex2'

from tms_risk.behavior.fit_model import build_model, get_data

df = get_data('/data/ds-tmsrisk', model_label=LABEL)
model = build_model(LABEL, df)
grp = pd.read_csv(DATA / 'logflex_group_draws.tsv', sep='\t').query('label == @LABEL')
subj = pd.read_csv(DATA / 'logflex2_subject_draws.tsv.gz', sep='\t')
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 100))
softplus = lambda x: np.logaddexp(0, x)


def g(var, reg='Intercept'):
    s = grp.query('var == @var and regressor == @reg')
    return s.sort_values('draw')['value'].values


def g_coefs(term, reg):
    return np.stack([g(f'{term}_noise_sd_spline{j}', reg) for j in range(1, 6)], 1)


BASIS = {t: model.make_dm(N_GRID, variable=f'{t}_noise_sd')
         for t in ('perceptual', 'memory')}


def nu_grp(term, vertex):
    c = g_coefs(term, 'Intercept')
    if vertex:
        c = c + g_coefs(term, VC)
    return softplus(c @ BASIS[term].T)


SUBS = sorted(subj['subject'].unique())


def _s_get(sub, var, reg='Intercept'):
    q = subj.query('subject == @sub and var == @var and regressor == @reg')
    return q.sort_values('draw')['value'].values


def _s_coefs_all(term, reg):
    """array (subjects, draws, 5) of subject-level spline coefficients."""
    return np.stack([np.stack([_s_get(sub, f'{term}_noise_sd_spline{j}', reg)
                               for j in range(1, 6)], 1) for sub in SUBS], 0)


_SC = {(t, r): _s_coefs_all(t, r) for t in ('perceptual', 'memory')
       for r in ('Intercept', VC)}
_SPRIOR = {v: np.stack([_s_get(sub, v) for sub in SUBS], 0)
           for v in ('risky_prior_mu', 'risky_prior_sd',
                     'safe_prior_mu', 'safe_prior_sd')}


def nu_subj(term, vertex):
    """Across-subject mean noise curve per draw: (draws, grid)."""
    c = _SC[(term, 'Intercept')]
    if vertex:
        c = c + _SC[(term, VC)]
    return softplus(np.einsum('sdj,gj->sdg', c, BASIS[term])).mean(0)


def band(ax, x, y, color, ls='-', alpha=.2):
    med = np.median(y, 0)
    h = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    ax.fill_between(x, h[:, 0], h[:, 1], color=color, alpha=alpha, lw=0)
    ax.plot(x, med, color=color, lw=1.4, ls=ls)


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


fig = plt.figure(figsize=(7.25, 6.6), constrained_layout=True)
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.05])

# a — priors over payoff distributions
ax = fig.add_subplot(gs[0, 0])
risky_n = np.where(df['p1'] != 1.0, df['n1'], df['n2'])
safe_n = np.where(df['p2'] != 1.0, df['n2'], df['n1'])
PG = np.linspace(0.5, 130, 400)
for emp, mu_v, sd_v, c, name, ytxt, hcol in [
        (risky_n, 'risky_prior_mu', 'risky_prior_sd', '.15', 'Risky', .96, '.90'),
        (safe_n, 'safe_prior_mu', 'safe_prior_sd', '.55', 'Safe', .85, '.82')]:
    ax.hist(emp, bins=26, density=True, color=hcol, histtype='stepfilled',
            alpha=.7, zorder=0)
    mus = _SPRIOR[mu_v][:, :, None]                       # subj, draw, 1
    sds = softplus(_SPRIOR[sd_v])[:, :, None]
    x = PG[None, None, :]
    dens = (np.exp(-(np.log(x) - mus) ** 2 / (2 * sds ** 2))
            / (x * sds * np.sqrt(2 * np.pi))).mean(0)     # draw, grid
    ax.fill_between(PG, np.percentile(dens, 2.5, 0),
                    np.percentile(dens, 97.5, 0), color=c, alpha=.2, lw=0)
    ax.plot(PG, np.median(dens, 0), color=c, lw=1.4)
    ax.text(0.97, ytxt, f'{name} prior', color=c, fontsize=8,
            transform=ax.transAxes, ha='right', va='top')
ax.set_xlim(0, 130)
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Density')
ax.set_title('Subjective priors (subject-averaged)', fontsize=9)

# b — perceptual noise by condition
ax = fig.add_subplot(gs[0, 1])
band(ax, N_GRID, nu_subj('perceptual', False), C_IPS)
band(ax, N_GRID, nu_subj('perceptual', True), C_VERTEX)
logx(ax)
ax.set_ylabel('Noise SD (log units)')
ax.set_title('Perceptual noise (TMS regressor)', fontsize=9)
ax.text(0.05, 0.96, 'IPS', color=C_IPS, fontsize=8, transform=ax.transAxes,
        va='top')
ax.text(0.05, 0.87, 'Vertex', color=C_VERTEX, fontsize=8,
        transform=ax.transAxes, va='top')

# c — memory noise by condition (the order-asymmetric lever)
ax = fig.add_subplot(gs[0, 2])
band(ax, N_GRID, nu_subj('memory', False), C_IPS)
band(ax, N_GRID, nu_subj('memory', True), C_VERTEX)
logx(ax)
ax.set_ylabel('Noise SD (log units)')
ax.set_title('Memory noise (TMS regressor)', fontsize=9)

# d — contrasts, both terms
ax = fig.add_subplot(gs[1, 0])
ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
d_p = nu_subj('perceptual', False) - nu_subj('perceptual', True)
d_m = nu_subj('memory', False) - nu_subj('memory', True)
band(ax, N_GRID, d_p, '.15')
band(ax, N_GRID, d_m, '.6')
logx(ax)
ax.set_ylabel('Δ noise, IPS − Vertex (log units)')
ax.set_title('cTBS contrasts', fontsize=9)
ax.text(0.04, 0.96, 'Perceptual', color='.15', fontsize=8,
        transform=ax.transAxes, va='top')
ax.text(0.04, 0.87, 'Memory', color='.6', fontsize=8,
        transform=ax.transAxes, va='top')
for term, d_ in [('perc', d_p), ('mem', d_m)]:
    m730 = (N_GRID >= 7) & (N_GRID <= 30)
    mm = d_[:, m730].mean(1)
    print(f'{term} mean Δσ 7-30: {np.median(mm):+.4f} '
          f'[{np.percentile(mm, 2.5):+.4f}, {np.percentile(mm, 97.5):+.4f}] '
          f'P(>0)={float((mm > 0).mean()):.3f}')
ax.set_ylim(-0.12, 0.15)

# e — prior locations (CHF)
ax = fig.add_subplot(gs[1, 1])
y = 0
for var, name in [('risky_prior_mu', 'Risky prior median'),
                  ('safe_prior_mu', 'Safe prior median')]:
    s = np.exp(_SPRIOR[var]).mean(0)
    lo, hi = az.hdi(s, hdi_prob=.95)
    ax.hlines(y, lo, hi, color='.2', lw=1.2)
    ax.plot(np.median(s), y, 'o', color='.2', ms=4.5)
    ax.text(min(np.median(s), 38), y + .18, name, fontsize=7.5, ha='center')
    y -= 1
ax.vlines([31, 15], -1.6, 0.6, color='0.75', lw=.7, ls=':')
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlim(0, 60)
ax.set_xlabel('CHF (dotted: payoff geomeans)')
ax.set_title('Prior locations', fontsize=9)

# f — TMS knot coefficients, both terms
ax = fig.add_subplot(gs[1, 2])
y = 0
for term, c in [('perceptual', '.15'), ('memory', '.6')]:
    for j in range(1, 6):
        s = g(f'{term}_noise_sd_spline{j}', VC) * -1
        lo, hi = az.hdi(s, hdi_prob=.95)
        ax.hlines(y, lo, hi, color=c, lw=1.1)
        ax.plot(np.median(s), y, 'o', color=c, ms=3.5)
        y -= 1
    y -= 0.7
ax.axvline(0, color='0.8', lw=.6, ls='--', zorder=0)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('TMS coefficient, IPS − Vertex\n(knots low→high; perc dark, mem light)')
ax.set_title('Where cTBS loads', fontsize=9)

# g/h — per-subject integrated effects, both terms
subs = sorted(subj['subject'].unique())


def s_coefs(sub, term, reg):
    out = []
    for j in range(1, 6):
        v = f'{term}_noise_sd_spline{j}'
        s = subj.query('subject == @sub and var == @v and regressor == @reg')
        out.append(s.sort_values('draw')['value'].values)
    return np.stack(out, 1)


b730 = {t: model.make_dm(np.exp(np.linspace(np.log(7 + 1e-6), np.log(30), 40)),
                         variable=f'{t}_noise_sd') for t in ('perceptual', 'memory')}
for gi, (term, tname) in enumerate([('perceptual', 'perceptual'),
                                    ('memory', 'memory')]):
    ax = fig.add_subplot(gs[2, gi])
    eff = []
    for sub in subs:
        ci = s_coefs(sub, term, 'Intercept')
        cv = s_coefs(sub, term, VC)
        d_s = (softplus(ci @ b730[term].T)
               - softplus((ci + cv) @ b730[term].T)).mean(1)
        eff.append(d_s)
    eff = np.array(eff)
    med = np.median(eff, 1)
    order = np.argsort(med)
    h = np.array([az.hdi(eff[i], hdi_prob=.95) for i in order])
    x = np.arange(len(subs))
    ax.vlines(x, h[:, 0], h[:, 1], color='.55', lw=.9, alpha=.8)
    ax.plot(x, med[order], 'o', color='.15', ms=3, zorder=3)
    ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
    frac = float((med > 0).mean())
    ax.text(0.02, 0.95, f'{frac:.0%} median increase', transform=ax.transAxes,
            fontsize=8, va='top', color='.35')
    ax.set_xticks([])
    ax.set_xlabel('Subjects (sorted)')
    ax.set_ylabel(f'Δ {tname} noise 7–30 CHF\nIPS − Vertex (log units)')
    ax.set_title(f'Individual cTBS effects: {tname}', fontsize=9)

ax = fig.add_subplot(gs[2, 2])
ax.axis('off')
ax.text(0, .95, 'Model: logflex2\n(both noise splines flexible,\nTMS on both '
        'terms)\n\nELPD −4175.4\n(−12.8 vs lfx2-bs3-sm-b)\n\nΔ-band PPC: running\n\n'
        'Diagnostics: r̂ ≈ 2.0 on prior\nmeans (wandering direction);\n'
        'noise/TMS params clean', fontsize=8.5, va='top', color='.25')

for ax_, letter in zip(fig.axes, 'abcdefgh'):
    ax_.text(-0.16, 1.06, letter, transform=ax_.transAxes, fontsize=12,
             fontweight='bold', fontfamily='Arial', va='bottom', ha='right')
fig.suptitle('Model card: logflex2 — flexible perceptual AND memory noise, '
             'TMS on both', fontsize=9, color='.3', y=1.02)

sns.despine(fig=fig, offset=4, trim=True)
fig.savefig(OUT / 'logflex2_card.pdf')
fig.savefig(OUT / 'logflex2_card.png', dpi=150)
print('wrote', OUT / 'logflex2_card.pdf')
