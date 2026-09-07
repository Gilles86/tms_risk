"""Preview: cTBS effect on the noise functions in the QUICK logflex2 fit.

750+750 draws only (max rhat 2.4 on prior means) — treat as a preview, not a
result. Curves: softplus(spline · basis) per posterior draw, group level;
basis rebuilt with the identical model construction (knots over log payoff).
"""
from pathlib import Path
import sys

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'libs' / 'bauer'))
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

from tms_risk.behavior.fit_model import build_model, get_data

df = get_data('/data/ds-tmsrisk', model_label='logflex2')
model = build_model('logflex2', df)
tr = az.from_netcdf('/data/ds-tmsrisk/derivatives/cogmodels.logflex-quick/'
                    'model-logflex2_trace.netcdf')
post = tr.posterior
# Chains 1+2 only: internally coherent and mutually consistent; chain 3
# wandered along the weakly-identified prior-mean direction, chain 0 sits
# between (see per-chain medians in the analysis log).
CHAINS = [1, 2]
post = post.isel(chain=CHAINS)
rng = np.random.default_rng(0)
n_draw = post.sizes['chain'] * post.sizes['draw']
idx = rng.choice(n_draw, 800, replace=False)

N_GRID = np.exp(np.linspace(np.log(7.0 + 1e-6), np.log(112.0 - 1e-6), 100))


def softplus(x):
    return np.logaddexp(0, x)


def coefs(term, regressor):
    out = []
    for j in range(1, 6):
        da = post[f'{term}_spline{j}_mu'].stack(sample=('chain', 'draw'))
        rd = [d for d in da.dims if d.endswith('regressors')][0]
        out.append(da.sel({rd: regressor}).values[idx])
    return np.stack(out, axis=1)          # draws x 5


def nu(term, vertex):
    c = coefs(term, 'Intercept')
    if vertex:
        c = c + coefs(term, VC)
    basis = model.make_dm(N_GRID, variable=term)   # knots over log payoff
    return softplus(c @ basis.T)                   # draws x grid


def band(ax, y, color, ls='-'):
    med = np.median(y, 0)
    h = np.array([az.hdi(y[:, i], hdi_prob=.95) for i in range(y.shape[1])])
    ax.fill_between(N_GRID, h[:, 0], h[:, 1], color=color, alpha=.20, lw=0)
    ax.plot(N_GRID, med, color=color, lw=1.3, ls=ls)


fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.4), constrained_layout=True)
axes = axes.ravel()

# --- a: estimated priors (lognormal, CHF) over the actual payoff distributions
ax = axes[0]
risky_n = np.where(df['p1'] != 1.0, df['n1'], df['n2'])
safe_n = np.where(df['p2'] != 1.0, df['n2'], df['n1'])
PGRID = np.linspace(0.5, 130, 400)
for emp, mu_v, sd_v, c, name, ytxt in [
        (risky_n, 'risky_prior_mu_mu', 'risky_prior_sd_mu', '.15', 'Risky', .95),
        (safe_n, 'safe_prior_mu_mu', 'safe_prior_sd_mu', '.55', 'Safe', .85)]:
    ax.hist(emp, bins=28, density=True, color='.90' if name == 'Risky' else '.82',
            histtype='stepfilled', alpha=.7, zorder=0)
    mus = post[mu_v].values.ravel()[None, :]
    sds = softplus(post[sd_v].values.ravel())[None, :]
    g = PGRID[:, None]
    dens = np.exp(-(np.log(g) - mus) ** 2 / (2 * sds ** 2)) / (g * sds * np.sqrt(2 * np.pi))
    med = np.median(dens, 1)
    lo = np.percentile(dens, 2.5, 1)
    hi = np.percentile(dens, 97.5, 1)
    ax.fill_between(PGRID, lo, hi, color=c, alpha=.20, lw=0)
    ax.plot(PGRID, med, color=c, lw=1.4)
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
ax.set_title('Estimated priors: lognormal, in payoff territory', fontsize=9)

for ax, term, title in [(axes[1], 'perceptual_noise_sd', 'Perceptual noise'),
                        (axes[2], 'memory_noise_sd', 'Memory noise')]:
    for vertex, c in [(False, C_IPS), (True, C_VERTEX)]:
        band(ax, nu(term, vertex), c)
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Noise SD (log-payoff units)')
    ax.set_title(title, fontsize=9)
axes[1].text(0.05, 0.95, 'IPS', color=C_IPS, fontsize=8,
             transform=axes[1].transAxes, va='top')
axes[1].text(0.05, 0.85, 'Vertex', color=C_VERTEX, fontsize=8,
             transform=axes[1].transAxes, va='top')

ax = axes[3]
ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
for term, c, name, ytxt in [('perceptual_noise_sd', '.2', 'Perceptual', .92),
                            ('memory_noise_sd', '.6', 'Memory', .82)]:
    d = (nu(term, False) - nu(term, True)) / nu(term, True) * 100
    band(ax, d, c)
    ax.text(0.05, ytxt, name, color=c, fontsize=8, transform=ax.transAxes,
            va='top')
    p_low = float((d[:, 0] > 0).mean())
    print(f'{term}: Δ%% at 7 CHF median {np.median(d[:, 0]):+.1f}%%, '
          f'P(ips noisier) = {p_low:.2f}; at 30 CHF '
          f'{np.median(d[:, np.argmin(np.abs(N_GRID - 30))]):+.1f}%%')
ax.set_xscale('log')
ax.set_xticks([7, 15, 30, 60, 112])
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
ax.xaxis.set_minor_locator(mticker.NullLocator())
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Δ noise, IPS − Vertex (%)')
ax.set_title('cTBS contrast (preview fit)', fontsize=9)

for ax, letter in zip(axes, 'abcd'):
    ax.text(-0.24, 1.04, letter, transform=ax.transAxes, fontsize=12,
            fontweight='bold', fontfamily='Arial', va='bottom', ha='right')
fig.suptitle('Quick logflex2 fit — chains 1+2 only (mutually consistent); preview',
             fontsize=8, color='.35', x=0.5, y=1.04)

sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'logflex_quick_tms_chains12.pdf')
fig.savefig(OUT / 'logflex_quick_tms_chains12.png', dpi=150)
print('wrote', OUT / 'logflex_quick_tms_chains12.pdf')
for v in ['risky_prior_mu_mu', 'safe_prior_mu_mu']:
    x = post[v].values.ravel()
    print(f'{v}: median {np.exp(np.median(x)):.1f} CHF '
          f'[{np.exp(np.percentile(x, 2.5)):.1f}, {np.exp(np.percentile(x, 97.5)):.1f}]')
for v in ['risky_prior_sd_mu', 'safe_prior_sd_mu']:
    x = softplus(post[v].values.ravel())
    print(f'{v}: median {np.median(x):.2f} log-units '
          f'[{np.percentile(x, 2.5):.2f}, {np.percentile(x, 97.5):.2f}]')
