"""Group-parameter posteriors: power-law vs flexible-spline PMC (exploratory).

Where do the priors end up in the two candidate models, and how similar are
the noise functions / implied compression? Reads:
    notes/data/power_group_draws.tsv     (power2_full and friends)
    notes/data/flexible_group_draws.tsv  (flexible2 head refits + power2_null)
    notes/data/flexible_spline_dm.tsv    (spline bases anchored at fit time)

Writes notes/figures/power_vs_flexible.pdf. Bands: 95% HDI over 800 thinned
group-level posterior draws; lines: posterior medians.
"""
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / 'notes' / 'data'
OUT = ROOT / 'notes' / 'figures'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 9, 'axes.labelsize': 10, 'axes.titlesize': 10,
    'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'mathtext.fontset': 'stixsans',
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.2, 'lines.markersize': 4,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_POWER, C_FLEX = '#3B5BA5', '#8172B2'   # power-law blue, spline purple

draws = pd.concat([pd.read_csv(DATA / 'power_group_draws.tsv', sep='\t'),
                   pd.read_csv(DATA / 'flexible_group_draws.tsv', sep='\t')])
dm = pd.read_csv(DATA / 'flexible_spline_dm.tsv', sep='\t')
XG = dm['x'].values


def get(label, var, regressor='Intercept'):
    sel = draws.query('label == @label and var == @var and regressor == @regressor')
    return sel.sort_values('draw')['value'].values


def softplus(x):
    return np.logaddexp(0, x)


def hdi_band(curves):
    med = np.median(curves, axis=0)
    h = np.array([az.hdi(curves[:, i], hdi_prob=.95) for i in range(curves.shape[1])])
    return med, h[:, 0], h[:, 1]


def flex_nu(label, which):
    """Spline noise curves per draw: 'memory', 'perceptual' or 'first' (sum)."""
    def one(var):
        coefs = np.stack([get(label, f'{var}_spline{j}') for j in range(1, 6)], axis=1)
        basis = dm[[f'{var}_dm{j}' for j in range(1, 6)]].values
        return softplus(coefs @ basis.T)
    if which == 'first':
        return one('memory_noise_sd') + one('perceptual_noise_sd')
    return one(f'{which}_noise_sd')


def power_nu(label, which):
    b = get(label, 'noise_exponent')
    perc = np.exp(get(label, 'perceptual_log_sd_intercept')[:, None]
                  + b[:, None] * np.log(XG))
    if which == 'perceptual':
        return perc
    mem = np.exp(get(label, 'memory_log_sd_intercept')[:, None]
                 + b[:, None] * np.log(XG))
    return perc + mem


# empirical payoff distributions
import sys
sys.path.insert(0, str(ROOT / 'libs' / 'bauer'))
from tms_risk.behavior.fit_model import get_data
df = get_data('/data/ds-tmsrisk', model_label='power2_full')
risky_n = np.where(df['p1'] != 1.0, df['n1'], df['n2'])
safe_n = np.where(df['p2'] != 1.0, df['n2'], df['n1'])

fig, axes = plt.subplots(2, 3, figsize=(7.6, 5.0), constrained_layout=True)
PGRID = np.linspace(-20, 130, 300)

# a/b — fitted priors over the actual payoff distributions
for ax, role, emp, letter in [(axes[0, 0], 'risky', risky_n, 'a'),
                              (axes[0, 1], 'safe', safe_n, 'b')]:
    ax.hist(emp, bins=24, density=True, color='0.88', zorder=0)
    for lab, c in [('power2_full', C_POWER), ('flexible2_tmsboth', C_FLEX)]:
        mus = get(lab, f'{role}_prior_mu')
        sds = get(lab, f'{role}_prior_sd')
        curves = norm.pdf(PGRID[None, :], mus[:, None], sds[:, None])
        med, lo, hi = hdi_band(curves)
        ax.fill_between(PGRID, lo, hi, color=c, alpha=.22, lw=0, zorder=1)
        ax.plot(PGRID, med, color=c, lw=1.3, zorder=2)
    ax.axvline(0, color='0.75', lw=.6, ls=':', zorder=0)
    ax.set_ylim(0, 0.062)
    ax.set_xlim(-22, 132)
    ax.set_xticks([0, 28, 56, 84, 112])
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_title(f'{role.capitalize()} prior', fontsize=9)
    if letter == 'a':
        ax.set_ylabel('Density')
        ax.text(0.97, 0.95, 'Power law', color=C_POWER, fontsize=8,
                transform=ax.transAxes, ha='right', va='top')
        ax.text(0.97, 0.84, 'Splines', color=C_FLEX, fontsize=8,
                transform=ax.transAxes, ha='right', va='top')
        ax.text(0.97, 0.73, 'Actual payoffs', color='0.55', fontsize=8,
                transform=ax.transAxes, ha='right', va='top')
        ax.text(16, 0.040, 'Peak off-scale\n(σ ≈ 1.5 CHF)', color=C_FLEX,
                fontsize=7, ha='left', va='top')
    ax.text(-0.1, 1.05, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

# c — forest of the four prior parameters, both models, TMS + null variants
ax = axes[0, 2]
PARAMS = [('risky_prior_mu', 'Risky μ'), ('risky_prior_sd', 'Risky σ'),
          ('safe_prior_mu', 'Safe μ'), ('safe_prior_sd', 'Safe σ')]
MODELS = [('power2_full', C_POWER, True), ('power2_null', C_POWER, False),
          ('flexible2_tmsboth', C_FLEX, True), ('flexible2_null', C_FLEX, False)]
y = 0
for var, pname in PARAMS:
    ax.text(-0.03, y + 0.75, pname, fontsize=8, color='0.2', fontweight='bold',
            fontfamily='Arial', transform=ax.get_yaxis_transform(),
            ha='right', va='center')
    for lab, c, filled in MODELS:
        s = get(lab, var)
        lo, hi = az.hdi(s, hdi_prob=.95)
        ax.hlines(y, lo, hi, color=c, lw=1.1)
        ax.plot(np.median(s), y, 'o', ms=4, mec=c, mfc=c if filled else 'white',
                mew=1.0)
        y -= 1
    y -= 0.9
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('CHF')
ax.set_xticks([0, 20, 40, 60])
ax.text(0.97, 1.0, 'Open: null variants', fontsize=7, color='0.4',
        transform=ax.transAxes, va='top', ha='right')
ax.text(-0.1, 1.05, 'c', transform=ax.transAxes, fontsize=12,
        fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

# d/e — noise functions, null models: spline shape vs power law
for ax, which, title, letter in [
        (axes[1, 0], 'first', 'First presented (perc + mem)', 'd'),
        (axes[1, 1], 'perceptual', 'Second presented (perceptual)', 'e')]:
    for curves, c in [(power_nu('power2_null', which), C_POWER),
                      (flex_nu('flexible2_null', which), C_FLEX)]:
        med, lo, hi = hdi_band(curves)
        ax.fill_between(XG, lo, hi, color=c, alpha=.22, lw=0)
        ax.plot(XG, med, color=c, lw=1.3)
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_yscale('log')
    ax.set_yticks([1, 3, 10, 30])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Noise SD (CHF)')
    ax.set_title(title, fontsize=9)
    ax.text(-0.22, 1.05, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

# f — implied compression: perceived risky payoff (first position), both models
ax = axes[1, 2]
ax.plot([0, 118], [0, 118], color='0.7', lw=0.8, ls='--', zorder=0)
ax.text(80, 93, 'Veridical', color='0.55', fontsize=8, rotation=38, ha='center')
for lab_p, nu, c in [('power2_null', power_nu('power2_null', 'first'), C_POWER),
                     ('flexible2_null', flex_nu('flexible2_null', 'first'), C_FLEX)]:
    mus = get(lab_p, 'risky_prior_mu')[:, None]
    sds = get(lab_p, 'risky_prior_sd')[:, None]
    w = sds ** 2 / (sds ** 2 + nu ** 2)
    xhat = mus + w * (XG[None, :] - mus)
    med, lo, hi = hdi_band(xhat)
    ax.fill_between(XG, lo, hi, color=c, alpha=.22, lw=0)
    ax.plot(XG, med, color=c, lw=1.3)
ax.set_xlim(0, 118)
ax.set_ylim(0, 118)
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_yticks([0, 28, 56, 84, 112])
ax.set_xlabel('True payoff (CHF)')
ax.set_ylabel('Perceived payoff (CHF)')
ax.set_title('Implied compression', fontsize=9)
ax.text(100, 72, 'Power law', color=C_POWER, fontsize=8, ha='right')
ax.text(100, 22, 'Splines', color=C_FLEX, fontsize=8, ha='right')
ax.text(-0.22, 1.05, 'f', transform=ax.transAxes, fontsize=12,
        fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

sns.despine(fig=fig, offset=5, trim=True)
for ax in axes[0, :2]:
    ax.spines['left'].set_visible(False)
axes[0, 2].spines['left'].set_visible(False)
fig.savefig(OUT / 'power_vs_flexible.pdf')
fig.savefig(OUT / 'power_vs_flexible.png', dpi=150)
print('wrote', OUT / 'power_vs_flexible.pdf')
