"""Noise curves + priors + cTBS contrast from the FULL logflex fits.

logflex2b = TMS on perceptual noise only (memory noise shared across
conditions); 5000+5000 draws, 0 divergences. Bands: 95% HDI over 800 thinned
group-level draws. Basis rebuilt with the identical model construction.
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
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_IPS, C_VERTEX = '#d62728', '#2ca02c'
VC = 'stimulation_condition[T.vertex]'
LABEL = 'logflex2b'

from tms_risk.behavior.fit_model import build_model, get_data

df = get_data('/data/ds-tmsrisk', model_label=LABEL)
model = build_model(LABEL, df)
draws = pd.read_csv(DATA / 'logflex_group_draws.tsv', sep='\t')
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 100))


def softplus(x):
    return np.logaddexp(0, x)


def get(var, regressor='Intercept', label=LABEL):
    s = draws.query('label == @label and var == @var and regressor == @regressor')
    return s.sort_values('draw')['value'].values


def coefs(term, regressor):
    out = []
    for j in range(1, 6):
        v = f'{term}_noise_sd_spline{j}'
        s = draws.query('label == @LABEL and var == @v and regressor == @regressor')
        out.append(s.sort_values('draw')['value'].values if len(s)
                   else np.zeros(800))
    return np.stack(out, 1)


def nu(term, vertex):
    c = coefs(term, 'Intercept')
    if vertex:
        c = c + coefs(term, VC)
    return softplus(c @ model.make_dm(N_GRID, variable=f'{term}_noise_sd').T)


def band(ax, y, color, ls='-'):
    med = np.median(y, 0)
    h = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    ax.fill_between(N_GRID, h[:, 0], h[:, 1], color=color, alpha=.20, lw=0)
    ax.plot(N_GRID, med, color=color, lw=1.4, ls=ls)


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


fig, axes = plt.subplots(2, 2, figsize=(7.25, 4.4), constrained_layout=True)

# a — priors over payoff distributions
ax = axes[0, 0]
risky_n = np.where(df['p1'] != 1.0, df['n1'], df['n2'])
safe_n = np.where(df['p2'] != 1.0, df['n2'], df['n1'])
PG = np.linspace(0.5, 130, 400)
for emp, mu_v, sd_v, c, name, ytxt, hcol in [
        (risky_n, 'risky_prior_mu', 'risky_prior_sd', '.15', 'Risky', .95, '.90'),
        (safe_n, 'safe_prior_mu', 'safe_prior_sd', '.55', 'Safe', .85, '.82')]:
    ax.hist(emp, bins=28, density=True, color=hcol, histtype='stepfilled',
            alpha=.7, zorder=0)
    mus = get(mu_v)[None, :]
    sds = softplus(get(sd_v))[None, :]
    g = PG[:, None]
    dens = np.exp(-(np.log(g) - mus) ** 2 / (2 * sds ** 2)) / (
        g * sds * np.sqrt(2 * np.pi))
    med = np.median(dens, 1)
    lo, hi = np.percentile(dens, [2.5, 97.5], axis=1)
    ax.fill_between(PG, lo, hi, color=c, alpha=.20, lw=0)
    ax.plot(PG, med, color=c, lw=1.4)
    ax.text(0.97, ytxt, f'{name} prior', color=c, fontsize=8,
            transform=ax.transAxes, ha='right', va='top')
ax.text(0.97, 0.73, 'Histograms: actual payoffs', color='.6', fontsize=7.5,
        transform=ax.transAxes, ha='right', va='top')
ax.set_xlim(0, 130)
ax.set_xticks([0, 28, 56, 84, 112])
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Density')
ax.set_title('Subjective priors (group level): curve = prior at\nposterior-median parameters, band = 95% CrI over shapes', fontsize=8)

# b — perceptual noise by condition
ax = axes[0, 1]
for vertex, c in [(False, C_IPS), (True, C_VERTEX)]:
    band(ax, nu('perceptual', vertex), c)
logx(ax)
ax.set_ylabel('Noise SD (log-payoff units)')
ax.set_title('Perceptual noise (TMS regressor here)', fontsize=9)
ax.text(0.05, 0.95, 'IPS', color=C_IPS, fontsize=8, transform=ax.transAxes,
        va='top')
ax.text(0.05, 0.86, 'Vertex', color=C_VERTEX, fontsize=8,
        transform=ax.transAxes, va='top')

# c — memory noise (shared across conditions in logflex2b)
ax = axes[1, 0]
band(ax, nu('memory', False), '.3')
logx(ax)
ax.set_ylabel('Noise SD (log-payoff units)')
ax.set_title('Memory noise (shared across conditions)', fontsize=9)

# d — the cTBS contrast on perceptual noise, ABSOLUTE in log units
ax = axes[1, 1]
ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
d = nu('perceptual', False) - nu('perceptual', True)
band(ax, d, '.2')
logx(ax)
ax.set_ylabel('Δ noise, IPS − Vertex (log-payoff units)')
ax.set_title('cTBS contrast (absolute)', fontsize=9)
for ni in [7, 15, 30, 60]:
    gi = np.argmin(np.abs(N_GRID - ni))
    p = float((d[:, gi] > 0).mean())
    print(f'perceptual Δσ at {ni} CHF: {np.median(d[:, gi]):+.3f} '
          f'[{np.percentile(d[:, gi], 2.5):+.3f}, {np.percentile(d[:, gi], 97.5):+.3f}] '
          f'P(ips noisier)={p:.3f}')

for ax, letter in zip(axes.ravel(), 'abcd'):
    ax.text(-0.2, 1.04, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='right')
fig.suptitle('Full logflex2b fit — 5000+5000 draws', fontsize=8, color='.35',
             y=1.03)

sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'logflex_full.pdf')
fig.savefig(OUT / 'logflex_full.png', dpi=150)
print('wrote', OUT / 'logflex_full.pdf')
