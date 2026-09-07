"""The winning model of the lfx2 grid, on one page (exploratory, NOT paper).

lfx2-bs3-sm-dp-b: log-space observer, lognormal risky/safe priors, cubic
B-spline perceptual noise over log payoff, scalar memory noise, cTBS on
perceptual noise. 5000+5000 draws; group bands 95% HDI over 500 thinned
draws; subject panels 400 draws each.
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
LABEL = 'lfx2-bs3-sm-dp-b'

from tms_risk.behavior.fit_model import build_model, get_data

df = get_data('/data/ds-tmsrisk', model_label=LABEL)
model = build_model(LABEL, df)
grp = pd.read_csv(DATA / 'lfxgrid_draws.tsv', sep='\t').query('label == @LABEL')
subj = pd.read_csv(DATA / 'winner_subject_draws.tsv.gz', sep='\t')
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 100))
softplus = lambda x: np.logaddexp(0, x)


def g(var, reg='Intercept'):
    s = grp.query('var == @var and regressor == @reg')
    return s.sort_values('draw')['value'].values


def g_coefs(reg):
    return np.stack([g(f'perceptual_noise_sd_spline{j}', reg)
                     for j in range(1, 6)], 1)


BASIS = model.make_dm(N_GRID, variable='perceptual_noise_sd')


def nu_grp(vertex):
    c = g_coefs('Intercept')
    if vertex:
        c = c + g_coefs(VC)
    return softplus(c @ BASIS.T)


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

# --- a: priors over payoff distributions -----------------------------------
ax = fig.add_subplot(gs[0, 0])
risky_n = np.where(df['p1'] != 1.0, df['n1'], df['n2'])
safe_n = np.where(df['p2'] != 1.0, df['n2'], df['n1'])
PG = np.linspace(0.5, 130, 400)
for emp, mu_v, sd_v, c, name, ytxt, hcol in [
        (risky_n, 'risky_prior_mu', 'risky_prior_sd', '.15', 'Risky', .96, '.90'),
        (safe_n, 'safe_prior_mu', 'safe_prior_sd', '.55', 'Safe', .85, '.82')]:
    ax.hist(emp, bins=26, density=True, color=hcol, histtype='stepfilled',
            alpha=.7, zorder=0)
    mus, sds = g(mu_v)[None, :], softplus(g(sd_v))[None, :]
    x = PG[:, None]
    dens = np.exp(-(np.log(x) - mus) ** 2 / (2 * sds ** 2)) / (
        x * sds * np.sqrt(2 * np.pi))
    ax.fill_between(PG, *np.percentile(dens, [2.5, 97.5], axis=1), color=c,
                    alpha=.2, lw=0)
    ax.plot(PG, np.median(dens, 1), color=c, lw=1.4)
    ax.text(0.97, ytxt, f'{name} prior', color=c, fontsize=8,
            transform=ax.transAxes, ha='right', va='top')
ax.text(0.97, 0.72, 'Grey: presented payoffs', color='.6', fontsize=7,
        transform=ax.transAxes, ha='right', va='top')
ax.set_xlim(0, 130)
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Density')
ax.set_title('Subjective priors (lognormal)', fontsize=9)

# --- b: perceptual noise by condition --------------------------------------
ax = fig.add_subplot(gs[0, 1])
band(ax, N_GRID, nu_grp(False), C_IPS)
band(ax, N_GRID, nu_grp(True), C_VERTEX)
logx(ax)
ax.set_ylabel('Perceptual noise SD (log units)')
ax.set_title('Noise function by stimulation', fontsize=9)
ax.text(0.05, 0.96, 'IPS', color=C_IPS, fontsize=8, transform=ax.transAxes,
        va='top')
ax.text(0.05, 0.87, 'Vertex', color=C_VERTEX, fontsize=8,
        transform=ax.transAxes, va='top')

# --- c: cTBS contrast + integrated stat ------------------------------------
ax = fig.add_subplot(gs[0, 2])
ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
d = nu_grp(False) - nu_grp(True)
band(ax, N_GRID, d, '.2')
logx(ax)
ax.set_ylabel('Δ noise, IPS − Vertex (log units)')
ax.set_title('cTBS contrast', fontsize=9)
m730 = (N_GRID >= 7) & (N_GRID <= 30)
mm = d[:, m730].mean(1)
ax.text(0.04, 0.96,
        f'Mean over 7–30 CHF:\n{np.median(mm):+.3f} '
        f'[{np.percentile(mm, 2.5):+.3f}, {np.percentile(mm, 97.5):+.3f}]\n'
        f'P(increase) = {float((mm > 0).mean()):.2f}',
        transform=ax.transAxes, fontsize=7.5, va='top')
ax.set_ylim(-0.06, 0.09)

# --- d: group-parameter forest (two unit systems) --------------------------
ax = fig.add_subplot(gs[1, 0])
rows_chf = [('risky_prior_mu', 'Risky prior median'),
            ('safe_prior_mu', 'Safe prior median')]
y = 0
for var, name in rows_chf:
    s = np.exp(g(var))
    lo, hi = az.hdi(s, hdi_prob=.95)
    ax.hlines(y, lo, hi, color='.2', lw=1.2)
    ax.plot(np.median(s), y, 'o', color='.2', ms=4.5)
    ax.text(np.median(s), y + .18, name, fontsize=7.5, ha='center')
    y -= 1
ax.vlines([31, 15], -1.6, 0.6, color='0.75', lw=.7, ls=':')
ax.text(31, .55, 'Risky geomean', fontsize=6.5, color='0.5', ha='center')
ax.text(15, -1.55, 'Safe geomean', fontsize=6.5, color='0.5', ha='center',
        va='top')
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlim(0, 40)
ax.set_xlabel('CHF')
ax.set_title('Prior locations', fontsize=9)

ax = fig.add_subplot(gs[1, 1])
rows_log = [('risky_prior_sd', 'Risky prior width', True),
            ('safe_prior_sd', 'Safe prior width', True),
            ('memory_noise_sd_spline1', 'Memory noise (scalar)', True)]
y = 0
for var, name, sp_ in rows_log:
    s = softplus(g(var)) if sp_ else g(var)
    lo, hi = az.hdi(s, hdi_prob=.95)
    ax.hlines(y, lo, hi, color='.2', lw=1.2)
    ax.plot(np.median(s), y, 'o', color='.2', ms=4.5)
    ax.text(np.median(s), y + .18, name, fontsize=7.5, ha='center')
    y -= 1
ax.vlines([0.55, 0.44], -2.6, 0.6, color='0.75', lw=.7, ls=':')
ax.text(0.50, .55, 'Payoff log-SDs', fontsize=6.5, color='0.5', ha='center')
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlim(0, 1.2)
ax.set_xlabel('Log-payoff units')
ax.set_title('Widths and memory noise', fontsize=9)

# --- e: TMS coefficient (group + spread) -----------------------------------
ax = fig.add_subplot(gs[1, 2])
labels_e, y = [], 0
for j in range(1, 6):
    s = g(f'perceptual_noise_sd_spline{j}', VC) * -1     # IPS - vertex
    lo, hi = az.hdi(s, hdi_prob=.95)
    ax.hlines(y, lo, hi, color='.2', lw=1.1)
    ax.plot(np.median(s), y, 'o', color='.2', ms=4)
    labels_e.append(f'Knot {j}')
    y -= 1
ax.axvline(0, color='0.8', lw=.6, ls='--', zorder=0)
ax.set_yticks([0, -1, -2, -3, -4])
ax.set_yticklabels(labels_e, fontsize=7.5)
ax.set_xlabel('TMS coefficient, IPS − Vertex\n(pre-softplus)')
ax.set_title('Where cTBS loads (knots low to high payoff)', fontsize=9)

# --- f: per-subject integrated cTBS effect ---------------------------------
ax = fig.add_subplot(gs[2, :2])
subs = sorted(subj['subject'].unique())


def s_coefs(sub, reg):
    out = []
    for j in range(1, 6):
        v = f'perceptual_noise_sd_spline{j}'
        s = subj.query('subject == @sub and var == @v and regressor == @reg')
        out.append(s.sort_values('draw')['value'].values)
    return np.stack(out, 1)


b730 = model.make_dm(np.exp(np.linspace(np.log(7 + 1e-6), np.log(30), 40)),
                     variable='perceptual_noise_sd')
eff = []
for sub in subs:
    ci = s_coefs(sub, 'Intercept')
    cv = s_coefs(sub, VC)
    d_s = (softplus(ci @ b730.T) - softplus((ci + cv) @ b730.T)).mean(1)
    eff.append(d_s)
eff = np.array(eff)                       # subjects x draws
med = np.median(eff, 1)
order = np.argsort(med)
h = np.array([az.hdi(eff[i], hdi_prob=.95) for i in order])
x = np.arange(len(subs))
ax.vlines(x, h[:, 0], h[:, 1], color='.55', lw=.9, alpha=.8)
ax.plot(x, med[order], 'o', color='.15', ms=3, zorder=3)
ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
frac = float((med > 0).mean())
ax.text(0.02, 0.95, f'{frac:.0%} of subjects with median increase '
        f'(mean Δσ over 7–30 CHF)', transform=ax.transAxes, fontsize=8,
        va='top', color='.35')
ax.set_xticks([])
ax.set_xlabel('Subjects (sorted)')
ax.set_ylabel('Δ noise 7–30 CHF\nIPS − Vertex (log units)')
ax.set_title('Individual cTBS effects', fontsize=9)

# --- g: per-subject memory noise -------------------------------------------
ax = fig.add_subplot(gs[2, 2])
mem = []
for sub in subs:
    s = subj.query('subject == @sub and var == "memory_noise_sd_spline1" and '
                   'regressor == "Intercept"')['value'].values
    mem.append(softplus(s))
mem = np.array(mem)
medm = np.median(mem, 1)
om = np.argsort(medm)
hm = np.array([az.hdi(mem[i], hdi_prob=.95) for i in om])
x = np.arange(len(subs))
ax.vlines(x, hm[:, 0], hm[:, 1], color='.55', lw=.9, alpha=.8)
ax.plot(x, medm[om], 'o', color='.15', ms=3, zorder=3)
ax.set_xticks([])
ax.set_xlabel('Subjects (sorted)')
ax.set_ylabel('Memory noise (log units)')
ax.set_title('Individual memory noise', fontsize=9)

for ax_, letter in zip(fig.axes, 'abcdefgh'):
    ax_.text(-0.16, 1.06, letter, transform=ax_.transAxes, fontsize=12,
             fontweight='bold', fontfamily='Arial', va='bottom', ha='right')
fig.suptitle('Winning model: log-space PMC, cubic spline perceptual noise, '
             'scalar memory (lfx2-bs3-sm-dp-b)', fontsize=9, color='.3',
             y=1.02)

sns.despine(fig=fig, offset=4, trim=True)
fig.savefig(OUT / 'winner_lfx2.pdf')
fig.savefig(OUT / 'winner_lfx2.png', dpi=150)
print('wrote', OUT / 'winner_lfx2.pdf')
