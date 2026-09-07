"""Exploratory figures for the power-law-noise PMC set (NOT paper figures).

Reads the small TSVs pulled from the sciencecloud sweep:
    notes/data/power_group_draws.tsv   (800 thinned group-level posterior draws)
    notes/data/power_ladder.tsv        (LOO ladder incl. diagnostics)

Writes to notes/figures/:
    power_mechanism.pdf   how the fitted model produces compression
    power_tms.pdf         where the cTBS effect lands in parameter space
    power_ladder.pdf      LOO model comparison

Run locally:
    ~/mambaforge/envs/tms_risk/bin/python -m tms_risk.behavior.scripts.plot_power_summary
"""
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'
OUT.mkdir(exist_ok=True)

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 4,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False, 'legend.handlelength': 1.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

# Canonical stimulation palette (repo-wide, non-negotiable)
C_IPS, C_VERTEX = '#d62728', '#2ca02c'
# Model-contrast palette (blue/orange reserved for model contrasts repo-wide)
C_BAYES, C_FLAT = '#3B5BA5', '#E08214'
# Presentation order never gets a hue: second = near-black, first = light gray
C_SECOND, C_FIRST = '.15', '.55'

VERTEX_COEF = 'stimulation_condition[T.vertex]'

draws = pd.read_csv(DATA / 'power_group_draws.tsv', sep='\t')
ladder = pd.read_csv(DATA / 'power_ladder.tsv', sep='\t').set_index('label')

N_GRID = np.exp(np.linspace(np.log(2), np.log(112), 120))
PAYOFF_TICKS = [2, 5, 10, 28, 112]  # 5-28 safe range, 112 = max risky


def get(label, var, regressor='Intercept'):
    """Return the 800 draws of one group-level parameter as an array."""
    sel = draws.query('label == @label and var == @var and regressor == @regressor')
    return sel.sort_values('draw')['value'].values


def band(ax, x, y_draws, color, label_median=True, alpha=.22, lw=1.3, ls='-'):
    """Median line + 95% HDI band from (n_draws, len(x)) matrix."""
    hdi = az.hdi(y_draws[np.newaxis], hdi_prob=.95)[:, 0] if y_draws.ndim == 1 else \
        np.array([az.hdi(y_draws[:, i], hdi_prob=.95) for i in range(y_draws.shape[1])])
    ax.fill_between(x, hdi[:, 0], hdi[:, 1], color=color, alpha=alpha, lw=0, zorder=1)
    med = np.median(y_draws, axis=0)
    ax.plot(x, med, color=color, lw=lw, ls=ls, zorder=2)
    return med


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks(PAYOFF_TICKS)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def panel_letter(ax, letter):
    ax.text(-0.28, 1.05, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='left')


# ===========================================================================
# Figure 1 — mechanism of the fitted Bayesian power-law model (power1_null)
# ===========================================================================
lab = 'power1_null'
i1, i2 = get(lab, 'n1_log_sd_intercept'), get(lab, 'n2_log_sd_intercept')
beta = get(lab, 'noise_exponent')
rp_mu, rp_sd = get(lab, 'risky_prior_mu'), get(lab, 'risky_prior_sd')
sp_mu, sp_sd = get(lab, 'safe_prior_mu'), get(lab, 'safe_prior_sd')

sd1 = np.exp(i1[:, None] + beta[:, None] * np.log(N_GRID))   # first presented
sd2 = np.exp(i2[:, None] + beta[:, None] * np.log(N_GRID))   # second presented
w_risky = rp_sd[:, None] ** 2 / (rp_sd[:, None] ** 2 + sd1 ** 2)   # risky shown first
w_safe = sp_sd[:, None] ** 2 / (sp_sd[:, None] ** 2 + sd2 ** 2)    # safe shown second
xhat_risky = rp_mu[:, None] + w_risky * (N_GRID - rp_mu[:, None])

# effective Stevens exponent of the mean mapping over the used payoff range
m = (N_GRID >= 5) & (N_GRID <= 112)
alpha_eff = np.array([np.polyfit(np.log(N_GRID[m]), np.log(np.maximum(x[m], .1)), 1)[0]
                      for x in xhat_risky])

fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), constrained_layout=True)

ax = axes[0]
band(ax, N_GRID, sd1, C_FIRST)
band(ax, N_GRID, sd2, C_SECOND)
logx(ax)
ax.set_yscale('log')
ax.set_yticks([1, 3, 10, 30])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Noise SD (CHF)')
ax.text(14, np.median(np.exp(i1)) * 14 ** np.median(beta) * 1.8, 'First presented',
        color=C_FIRST, fontsize=8, ha='center', rotation=33)
ax.text(14, np.median(np.exp(i2)) * 14 ** np.median(beta) * 0.45, 'Second presented',
        color=C_SECOND, fontsize=8, ha='center', rotation=33)
ax.text(0.03, 0.97, f'SD ~ nᵝ,  β = {np.median(beta):.2f}',
        transform=ax.transAxes, fontsize=9, va='top')
panel_letter(ax, 'a')

ax = axes[1]
band(ax, N_GRID, w_risky, C_FIRST)
band(ax, N_GRID, w_safe, C_SECOND)
logx(ax)
ax.set_ylim(0, 1.02)
ax.set_yticks([0, .5, 1])
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Weight on evidence  w(n)')
ax.text(3, .78, 'Safe, second', color=C_SECOND, fontsize=8)
ax.text(3, .60, 'Risky, first', color=C_FIRST, fontsize=8)
ax.annotate('Prior takes over', xy=(90, np.median(w_risky[:, N_GRID > 85], axis=0).min()),
            xytext=(9, .18), fontsize=9, ha='left', va='center',
            arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3,angleA=0,angleB=70',
                            color='0.2', lw=1.1, mutation_scale=9,
                            shrinkA=3, shrinkB=8, relpos=(1.0, 0.5)))
panel_letter(ax, 'b')

ax = axes[2]
ax.plot([0, 115], [0, 115], color='0.7', lw=0.8, ls='--', zorder=0)
ax.text(72, 86, 'Veridical', color='0.55', fontsize=8, rotation=38, ha='center')
band(ax, N_GRID, xhat_risky, C_BAYES)
ax.set_xlim(0, 115)
ax.set_ylim(0, 115)
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_yticks([0, 28, 56, 84, 112])
ax.set_xlabel('True payoff (CHF)')
ax.set_ylabel('Perceived payoff (CHF)')
lo, hi = np.percentile(alpha_eff, [2.5, 97.5])
ax.text(0.98, 0.06, f'Effective exponent {np.median(alpha_eff):.2f} [{lo:.2f}, {hi:.2f}]',
        transform=ax.transAxes, fontsize=8, ha='right', color=C_BAYES)
ax.annotate('Shrinkage = compression', xy=(103, np.median(xhat_risky[:, -8])),
            xytext=(6, 96), fontsize=9, ha='left', va='center',
            arrowprops=dict(arrowstyle='-|>', connectionstyle='angle3,angleA=0,angleB=-60',
                            color='0.2', lw=1.1, mutation_scale=9,
                            shrinkA=3, shrinkB=8, relpos=(1.0, 0.5)))
panel_letter(ax, 'c')

sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'power_mechanism.pdf')
fig.savefig(OUT / 'power_mechanism.png', dpi=150)
plt.close(fig)

# ===========================================================================
# Figure 2 — where the cTBS effect lands (power2_full + forest over models)
# ===========================================================================
lab = 'power2_full'
perc_i = get(lab, 'perceptual_log_sd_intercept')
perc_c = get(lab, 'perceptual_log_sd_intercept', VERTEX_COEF)
mem_i = get(lab, 'memory_log_sd_intercept')
mem_c = get(lab, 'memory_log_sd_intercept', VERTEX_COEF)
b_i = get(lab, 'noise_exponent')
b_c = get(lab, 'noise_exponent', VERTEX_COEF)


def sd_curves(is_vertex):
    perc = perc_i + is_vertex * perc_c
    mem = mem_i + is_vertex * mem_c
    b = b_i + is_vertex * b_c
    ln = np.log(N_GRID)
    sd_first = (np.exp(perc[:, None] + b[:, None] * ln) +
                np.exp(mem[:, None] + b[:, None] * ln))
    sd_second = np.exp(perc[:, None] + b[:, None] * ln)
    return sd_first, sd_second


sd1_ips, sd2_ips = sd_curves(0)
sd1_ver, sd2_ver = sd_curves(1)

fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.2), constrained_layout=True)

for ax, (d_ips, d_ver), title, letter in [
        (axes[0, 0], (sd1_ips, sd1_ver), 'First presented (perceptual + memory)', 'a'),
        (axes[0, 1], (sd2_ips, sd2_ver), 'Second presented (perceptual)', 'b')]:
    band(ax, N_GRID, d_ips, C_IPS)
    band(ax, N_GRID, d_ver, C_VERTEX)
    logx(ax)
    ax.set_yscale('log')
    ax.set_yticks([1, 3, 10, 30])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Noise SD (CHF)')
    ax.set_title(title, fontsize=9)
    ax.set_ylim(float(np.median(d_ver[:, 0])) * 0.45, None)
    ax.text(2.2, float(np.median(d_ips[:, 5])) * 1.7, 'IPS', color=C_IPS, fontsize=9)
    ax.text(3.6, float(np.median(d_ver[:, 12])) * 0.55, 'Vertex', color=C_VERTEX, fontsize=9)
    panel_letter(ax, letter)

axes[0, 0].annotate('cTBS raises noise\nat small payoffs',
                    xy=(3.1, float(np.median(sd1_ips[:, 20]))), xytext=(11, 1.35),
                    fontsize=9, ha='left', va='center',
                    arrowprops=dict(arrowstyle='-|>',
                                    connectionstyle='angle3,angleA=180,angleB=-40',
                                    color='0.2', lw=1.1, mutation_scale=9,
                                    shrinkA=4, shrinkB=8, relpos=(0.0, 0.5)))

# c — exponent posterior by condition
ax = axes[1, 0]
for s, color in [(b_i, C_IPS), (b_i + b_c, C_VERTEX)]:
    sns.kdeplot(x=s, ax=ax, color=color, fill=True, alpha=.35, lw=1.2, cut=0)
ax.set_xlabel('Noise exponent β')
ax.set_ylabel('Posterior density')
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.text(float(np.median(b_i)), 0.68, 'IPS', color=C_IPS, fontsize=9,
        ha='center', transform=ax.get_xaxis_transform())
ax.text(float(np.median(b_i + b_c)), 0.68, 'Vertex', color=C_VERTEX, fontsize=9,
        ha='center', transform=ax.get_xaxis_transform())
p_dir = float((b_c > 0).mean())
ax.text(0.02, 0.97, f'P(β lower under IPS) = {p_dir:.2f}',
        transform=ax.transAxes, fontsize=8, va='top')
panel_letter(ax, 'c')

# d — forest of the stimulation effect (IPS − Vertex) across the four _full models
ax = axes[1, 1]
rows = []
model_meta = [
    ('power2_full', 'Bayes · perc/mem', C_BAYES,
     [('perceptual_log_sd_intercept', 'Perceptual noise'),
      ('memory_log_sd_intercept', 'Memory noise'),
      ('noise_exponent', 'Exponent β')]),
    ('power1_full', 'Bayes · indep', C_BAYES,
     [('n1_log_sd_intercept', 'First-option noise'),
      ('n2_log_sd_intercept', 'Second-option noise'),
      ('noise_exponent', 'Exponent β')]),
    ('power2_flat_full', 'No prior · perc/mem', C_FLAT,
     [('perceptual_log_sd_intercept', 'Perceptual noise'),
      ('memory_log_sd_intercept', 'Memory noise'),
      ('noise_exponent', 'Exponent β')]),
    ('power1_flat_full', 'No prior · indep', C_FLAT,
     [('n1_log_sd_intercept', 'First-option noise'),
      ('n2_log_sd_intercept', 'Second-option noise'),
      ('noise_exponent', 'Exponent β')]),
]
y = 0
for mlab, mname, color, params in model_meta:
    ax.text(-0.03, y + 0.9, mname, fontsize=8, color=color,
            transform=ax.get_yaxis_transform(), ha='right', va='center',
            fontweight='bold', fontfamily='Arial')
    for var, pname in params:
        s = -get(mlab, var, VERTEX_COEF)          # IPS − Vertex
        lo, hi = az.hdi(s, hdi_prob=.95)
        ax.hlines(y, lo, hi, color=color, lw=1.2)
        ax.plot(np.median(s), y, 'o', color=color, ms=4.5,
                mfc=color, mec=color)
        ax.text(-0.03, y, pname, fontsize=7.5, color='0.2',
                transform=ax.get_yaxis_transform(), ha='right', va='center')
        y -= 1
    y -= 1.1
ax.axvline(0, color='0.7', lw=0.6, ls='--', zorder=0)
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Stimulation effect, IPS − Vertex (log units)')
ax.set_xlim(-0.35, 1.05)
ax.set_xticks([-0.25, 0, 0.25, 0.5, 0.75, 1.0])
panel_letter(ax, 'd')

sns.despine(fig=fig, offset=5, trim=True)
axes[1, 0].spines['left'].set_visible(False)
fig.savefig(OUT / 'power_tms.pdf')
fig.savefig(OUT / 'power_tms.png', dpi=150)
plt.close(fig)

# ===========================================================================
# Figure 3 — LOO ladder
# ===========================================================================
PRETTY = {}
for f, fam in [('1', 'indep'), ('2', 'perc/mem')]:
    for flat, prior in [('', 'Bayes'), ('_flat', 'No prior')]:
        for suf, tms in [('_null', 'null'), ('', 'TMS noise'),
                         ('_exp', 'TMS β'), ('_full', 'TMS noise+β')]:
            PRETTY[f'power{f}{flat}{suf}'] = f'{prior} · {fam} · {tms}'

lad = ladder.copy()
lad['shaky'] = (lad.max_rhat > 1.02) | (lad.divergences > 500)
lad['color'] = [C_FLAT if '_flat' in l else C_BAYES for l in lad.index]
lad = lad.sort_values('elpd_loo')

fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.4), width_ratios=[1, 1],
                         constrained_layout=True)

for ax, sub, letter in [(axes[0], lad, 'a'),
                        (axes[1], lad[lad.color == C_BAYES], 'b')]:
    ys = np.arange(len(sub))
    for yi, (l, r) in zip(ys, sub.iterrows()):
        ax.hlines(yi, r.delta_best - r.dse_best, r.delta_best + r.dse_best,
                  color=r.color, lw=1.1)
        ax.plot(r.delta_best, yi, 'o', ms=4.5, mec=r.color,
                mfc='white' if r.shaky else r.color, mew=1.1)
    ax.set_yticks(ys)
    ax.set_yticklabels([PRETTY[l] + (' †' if sub.loc[l, 'shaky'] else '')
                        for l in sub.index], fontsize=7.5)
    ax.set_xlabel('ΔELPD vs best model')
    panel_letter(ax, letter)

axes[0].set_xticks([-800, -600, -400, -200, 0])
axes[0].text(-420, 7.5, 'Removing the prior\ncosts ≈730 ELPD',
             fontsize=9, ha='center', va='center')
axes[1].set_xticks([-100, -75, -50, -25, 0])
axes[1].annotate('TMS regressors\nadd ≈80 ELPD',
                 xy=(-60, 1.6), xytext=(-60, 3.6), fontsize=9,
                 ha='center', va='center',
                 arrowprops=dict(arrowstyle='-|>', color='0.2', lw=1.1,
                                 mutation_scale=9, shrinkA=4, shrinkB=6,
                                 relpos=(0.5, 0.0)))
axes[1].set_title('Bayesian models only', fontsize=9)

sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'power_ladder.pdf')
fig.savefig(OUT / 'power_ladder.png', dpi=150)
plt.close(fig)

# ===========================================================================
# Figure 4 — tutorial: what the model entails. All parameters named; noise
# formula; payoff-space (not log/sensory-space) prior integration; sensory-
# space equivalence of the exponent. Group medians of power1_null; example
# trial: risky 56 CHF @ p=.55 presented FIRST vs safe 20 CHF.
# ===========================================================================
from scipy.stats import norm

med = {v: float(np.median(get('power1_null', v)))
       for v in ['n1_log_sd_intercept', 'n2_log_sd_intercept', 'noise_exponent',
                 'risky_prior_mu', 'risky_prior_sd', 'safe_prior_mu', 'safe_prior_sd']}
BETA = med['noise_exponent']
ALPHA = 1 - BETA          # Stevens exponent of the equivalent sensory transform
C_RISKY, C_SAFE = C_BAYES, '.45'

def sd_of(n, which):          # which: 'first' (n1) or 'second' (n2)
    ic = med['n1_log_sd_intercept'] if which == 'first' else med['n2_log_sd_intercept']
    return np.exp(ic) * n ** BETA

def observer(n, which, role):
    """Posterior mean/sd + decision-variable noise for one option."""
    p_mu = med[f'{role}_prior_mu']
    p_sd = med[f'{role}_prior_sd']
    e_sd = sd_of(n, which)
    w = p_sd ** 2 / (p_sd ** 2 + e_sd ** 2)
    post_mu = p_mu + w * (n - p_mu)
    post_sd = np.sqrt(p_sd ** 2 * e_sd ** 2 / (p_sd ** 2 + e_sd ** 2))
    return post_mu, post_sd, w * e_sd          # last = SD of posterior mean

N_RISKY, N_SAFE, P_RISKY = 56., 20., .55

fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), constrained_layout=True)

# a — the noise formula: intercepts set the level, the exponent the slope
ax = axes[0, 0]
sd1c = np.exp(med['n1_log_sd_intercept']) * N_GRID ** BETA
sd2c = np.exp(med['n2_log_sd_intercept']) * N_GRID ** BETA
ax.plot(N_GRID, sd1c, color=C_FIRST, lw=1.4)
ax.plot(N_GRID, sd2c, color=C_SECOND, lw=1.4)
logx(ax)
ax.set_yscale('log')
ax.set_yticks([1, 3, 10, 30])
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.yaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel('Payoff n (CHF)')
ax.set_ylabel('Noise SD (CHF)')
ax.text(0.03, 1.0, r'$\sigma_k(n) = e^{c_k} \cdot n^{\beta}$', transform=ax.transAxes,
        fontsize=10, va='top')
ax.annotate('', xy=(2.2, sd2c[0]), xytext=(2.2, sd1c[0] * 1.05),
            arrowprops=dict(arrowstyle='<->', color='0.2', lw=1.0, shrinkA=0, shrinkB=0))
ax.set_ylim(0.55, None)
ax.text(9, 1.30, 'Intercepts:\n'
        f'$e^{{c_1}}$ = {np.exp(med["n1_log_sd_intercept"]):.2f} (first)\n'
        f'$e^{{c_2}}$ = {np.exp(med["n2_log_sd_intercept"]):.2f} (second)',
        fontsize=7, va='top')
ax.text(38, np.exp(med['n1_log_sd_intercept']) * 38 ** BETA * 1.75,
        f'Slope = β = {BETA:.2f}', fontsize=8, rotation=33, ha='center')
panel_letter(ax, 'a')

# b — what the exponent means: equivalent Stevens transform with constant
#     sensory noise (α = 1 − β)
ax = axes[0, 1]
nn = np.linspace(0.01, 112, 300)
m_of = lambda n: n ** ALPHA
ax.plot(nn, m_of(nn), color='0.2', lw=1.4)
sd_m = ALPHA * np.exp(med['n1_log_sd_intercept'])      # constant in sensory space
for ni in [15, 55, 100]:
    mi = m_of(ni)
    ax.plot([ni, ni], [0, mi], color='0.8', lw=0.6, ls=':', zorder=0)
    ax.plot([0, ni], [mi, mi], color='0.8', lw=0.6, ls=':', zorder=0)
    # sensory-space bump (constant width, along y at x=0)
    yy = np.linspace(mi - 3.5 * sd_m, mi + 3.5 * sd_m, 60)
    ax.fill_betweenx(yy, 0, norm.pdf(yy, mi, sd_m) * sd_m * 14,
                     color=C_BAYES, alpha=.45, lw=0)
    # payoff-space bump (widening, along x at y=0)
    sd_n = np.exp(med['n1_log_sd_intercept']) * ni ** BETA
    xx = np.linspace(max(ni - 3.5 * sd_n, 0), ni + 3.5 * sd_n, 60)
    ax.fill_between(xx, 0, norm.pdf(xx, ni, sd_n) * sd_n * 0.9,
                    color='0.55', alpha=.5, lw=0)
ax.set_xlim(0, 118)
ax.set_ylim(0, 4.4)
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_yticks([0, 2, 4])
ax.set_xlabel('Payoff n (CHF)')
ax.set_ylabel('Sensory magnitude m')
ax.text(113, 2.45, r'$m = n^{\alpha}$' + f'\nα = 1 − β = {ALPHA:.2f}',
        fontsize=8, ha='right', va='top')
ax.text(0.03, 0.97, 'Constant sensory noise ≈ power-law\npayoff noise (1st-order equivalence)',
        transform=ax.transAxes, fontsize=7.5, va='top', color=C_BAYES)
panel_letter(ax, 'b')

# c — Bayesian shrinkage of the (noisier) risky percept, in payoff space
ax = axes[0, 2]
x = np.linspace(0, 95, 400)
sd_r = sd_of(N_RISKY, 'first')
post_mu, post_sd, _ = observer(N_RISKY, 'first', 'risky')
prior = norm.pdf(x, med['risky_prior_mu'], med['risky_prior_sd'])
lik = norm.pdf(x, N_RISKY, sd_r)
post = norm.pdf(x, post_mu, post_sd)
ax.fill_between(x, prior / prior.max() * .55, color='.8', alpha=.6, lw=0)
ax.plot(x, prior / prior.max() * .55, color='.55', lw=1.0)
ax.plot(x, lik / lik.max(), color=C_RISKY, lw=1.0, ls='--')
ax.plot(x, post / post.max(), color=C_RISKY, lw=1.6)
ax.text(med['risky_prior_mu'] - 24, .50, 'Risky prior', color='.4', fontsize=8)
ax.text(76, .80, 'Likelihood', color=C_RISKY, fontsize=8, alpha=.8)
ax.text(30.5, .97, 'Posterior', color=C_RISKY, fontsize=9, ha='right')
ax.annotate('', xy=(post_mu + 1.5, 1.03), xytext=(N_RISKY, 1.03),
            arrowprops=dict(arrowstyle='-|>', color='0.2', lw=1.1, mutation_scale=9))
ax.text((N_RISKY + post_mu) / 2, 1.07, f'Shrunk: 56 to {post_mu:.0f}',
        fontsize=8, ha='center')
ax.text(0.99, 0.99, 'Inference in payoff\nspace (CHF), not log\nor sensory space',
        transform=ax.transAxes, fontsize=7, va='top', ha='right', color='0.35')
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Density, scaled')
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xticks([0, 20, int(round(post_mu)), 56, 90])
ax.set_ylim(0, 1.22)
panel_letter(ax, 'c')

# d — decision stage: probability-weighted posterior means + comparison noise
ax = axes[1, 0]
r_mu, _, r_dv_sd = observer(N_RISKY, 'first', 'risky')
s_mu, _, s_dv_sd = observer(N_SAFE, 'second', 'safe')
dv_r, dv_r_sd = P_RISKY * r_mu, P_RISKY * r_dv_sd
dv_s, dv_s_sd = 1.0 * s_mu, 1.0 * s_dv_sd
xd = np.linspace(5, 45, 400)
for mu, sd, c, name, xt, ha in [
        (dv_s, dv_s_sd, C_SAFE, 'Safe: 1.0 × 20', dv_s - 2.5, 'right'),
        (dv_r, dv_r_sd, C_RISKY, f'Risky: 0.55 × {r_mu:.0f}', dv_r + 2.5, 'left')]:
    d = norm.pdf(xd, mu, sd)
    ax.fill_between(xd, d / d.max(), color=c, alpha=.25, lw=0)
    ax.plot(xd, d / d.max(), color=c, lw=1.3)
    ax.text(xt, 1.05, name, color=c, fontsize=8, ha=ha)
ev_obj = P_RISKY * N_RISKY
ax.vlines(ev_obj, 0, .8, color=C_RISKY, lw=0.9, ls=':')
ax.text(ev_obj + .8, .70, 'Objective EV\n30.8', color=C_RISKY, fontsize=7.5, va='top')
p_choose = norm.cdf((dv_r - dv_s) / np.hypot(dv_r_sd, dv_s_sd))
ax.text(0.03, 0.97, f'P(choose risky) = Φ(Δ/σΔ) = {p_choose:.2f}',
        transform=ax.transAxes, fontsize=8, va='top')
ax.set_xlabel('Decision variable p · n̂ (CHF)')
ax.set_ylabel('Density, scaled')
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xticks([5, 15, 25, 35, 45])
ax.set_ylim(0, 1.22)
panel_letter(ax, 'd')

# e — the psychometric this produces, by presentation order
ax = axes[1, 1]
ratios = np.exp(np.linspace(np.log(.4), np.log(4), 200))
def p_risky_curve(risky_first):
    n_r = N_SAFE * ratios
    which_r, which_s = ('first', 'second') if risky_first else ('second', 'first')
    r_mu, _, r_sd = np.vectorize(lambda n: observer(n, which_r, 'risky'))(n_r)
    s_mu, _, s_sd = np.vectorize(lambda n: observer(n, which_s, 'safe'))(np.full_like(n_r, N_SAFE))
    return norm.cdf((P_RISKY * r_mu - s_mu) / np.hypot(P_RISKY * r_sd, s_sd))

for rf, c, name, ytxt in [(False, C_SECOND, 'Risky second', .72),
                          (True, C_FIRST, 'Risky first', .40)]:
    ax.plot(ratios, p_risky_curve(rf), color=c, lw=1.4)
    ax.text(3.9, ytxt, name, color=c, fontsize=8, ha='right')
ax.axhline(.5, color='0.75', lw=.6, ls='--', zorder=0)
rn = 1 / P_RISKY
ax.vlines(rn, 0, .92, color='0.6', lw=.8, ls=':')
ax.text(rn, .96, 'Risk-neutral', color='0.4', fontsize=8, ha='center')
ax.set_xscale('log')
ax.set_xticks([.5, 1, 1.82, 4])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_ylim(0, 1.0)
ax.set_yticks([0, .5, 1])
ax.set_xlabel('Risky / safe payoff ratio (safe = 20 CHF)')
ax.set_ylabel('P(choose risky)')
panel_letter(ax, 'e')

# f — every free parameter, with fitted group medians
ax = axes[1, 2]
ax.axis('off')
e1 = np.exp(med['n1_log_sd_intercept'])
e2 = np.exp(med['n2_log_sd_intercept'])
lines = [
    ('Free parameters (group medians)', True),
    (r'Noise:  $\sigma_k(n) = e^{c_k} \cdot n^{\beta}$', False),
    (r'   $c_1$ = %.2f  ($e^{c_1}$ = %.2f, first option)' % (med['n1_log_sd_intercept'], e1), False),
    (r'   $c_2$ = %.2f  ($e^{c_2}$ = %.2f, second option)' % (med['n2_log_sd_intercept'], e2), False),
    (f'   β = {BETA:.2f}  (noise exponent)', False),
    ('Priors — separate risky & safe, payoff space:', False),
    (r'   Risky:  $\mu_r$ = %.1f,  $\sigma_r$ = %.1f CHF' % (med['risky_prior_mu'], med['risky_prior_sd']), False),
    (r'   Safe:  $\mu_s$ = %.1f,  $\sigma_s$ = %.1f CHF' % (med['safe_prior_mu'], med['safe_prior_sd']), False),
    ('Choice:  P(risky) = Φ(Δ / σΔ)', False),
    (r'   Δ = $p \cdot \hat{n}_r - \hat{n}_s$;  DV noise = p · w · σ(n)', False),
    ('All parameters hierarchical (35 subjects);', False),
    (r'TMS variants: condition regressors on $c_1$, $c_2$, β', False),
]
y = 0.98
for txt, bold in lines:
    ax.text(0.0, y, txt, transform=ax.transAxes, fontsize=7.5, va='top',
            fontweight='bold' if bold else 'normal',
            fontfamily='Arial' if bold else 'Helvetica')
    y -= 0.085
panel_letter(ax, 'f')

sns.despine(fig=fig, offset=5, trim=True)
for ax in [axes[0, 2], axes[1, 0]]:
    ax.spines['left'].set_visible(False)
fig.savefig(OUT / 'power_tutorial.pdf')
fig.savefig(OUT / 'power_tutorial.png', dpi=150)
plt.close(fig)

print('Wrote:')
for f in ['power_mechanism', 'power_tms', 'power_ladder', 'power_tutorial']:
    print('  ', OUT / f'{f}.pdf')
