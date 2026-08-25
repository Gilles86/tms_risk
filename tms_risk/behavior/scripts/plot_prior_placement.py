"""Where do the fitted observer priors actually sit, relative to the payoffs?

The natural-space Flexible PMC passes every predictive check but places its
prior outside the range of the stimuli it is a prior over (the safe-option
prior mean is negative). The log-space model, fitted to the same choices,
puts both priors on the payoff distributions. This is the argument for the
log-space front-end in one figure.

Priors are subject-averaged within draw, then summarized over draws.
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

C_RISKY, C_SAFE = '#8172B2', '#4A4A4A'
BOLD = dict(fontweight='bold', fontfamily='Arial')
softplus = lambda x: np.logaddexp(0, x)

from tms_risk.behavior.fit_model import get_data

df = get_data('/data/ds-tmsrisk', model_label='flexible2')
n_risky = np.where(df['p1'] == 0.55, df['n1'], df['n2']).astype(float)
n_safe = np.where(df['p1'] == 0.55, df['n2'], df['n1']).astype(float)


def prior_draws(fname, log_space):
    s = pd.read_csv(DATA / fname, sep='\t')
    out = {}
    for role in ('risky', 'safe'):
        for kind in ('mu', 'sd'):
            q = s[(s['var'] == f'{role}_prior_{kind}') & (s.regressor == 'Intercept')]
            w = q.pivot_table(index='draw', columns='subject',
                              values='value').mean(1).values
            out[(role, kind)] = softplus(w) if kind == 'sd' else w
    return out


NAT = prior_draws('flex2a_subject_draws.tsv.gz', False)
LOG = prior_draws('m2bm_subject_draws.tsv.gz', True)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

# ---- natural space --------------------------------------------------------
ax = axes[0]
grid = np.linspace(-16, 120, 600)
for role, n, c in [('risky', n_risky, C_RISKY), ('safe', n_safe, C_SAFE)]:
    ax.hist(n, bins=40, range=(0, 120), density=True, color=c, alpha=.22,
            lw=0)
    mu = np.median(NAT[(role, 'mu')])
    sd = np.median(NAT[(role, 'sd')])
    ax.plot(grid, norm.pdf(grid, mu, sd), color=c, lw=1.5)
    h = az.hdi(NAT[(role, 'mu')], hdi_prob=.95)
    ax.plot([h[0], h[1]], [-.004, -.004], color=c, lw=2.2,
            solid_capstyle='butt', clip_on=False)
    ax.plot([mu], [-.004], 'o', color=c, ms=4, clip_on=False)
ax.axvline(0, color='.35', lw=.8, ls=':')
ax.text(-14.5, .148, 'Safe prior mean\nis NEGATIVE', fontsize=6.8,
        color='#b0453b', va='top', **BOLD)
ax.annotate('', xy=(-4.6, .10), xytext=(-9.5, .132),
            arrowprops=dict(arrowstyle='-|>', color='#b0453b', lw=1))
ax.set_xlim(-16, 120)
ax.set_ylim(-.008, .16)
ax.set_xlabel('Payoff (CHF)')
ax.set_ylabel('Density')
ax.set_title('Flexible PMC, natural space\n(passes every PPC)', fontsize=8.5,
             **BOLD)
ax.text(.97, .96, 'Risky', transform=ax.transAxes, ha='right', va='top',
        fontsize=7, color=C_RISKY)
ax.text(.97, .86, 'Safe', transform=ax.transAxes, ha='right', va='top',
        fontsize=7, color=C_SAFE)
ax.text(.97, .70, 'shaded = actual payoffs\nline = fitted prior',
        transform=ax.transAxes, ha='right', va='top', fontsize=6.3,
        color='.45')

# ---- log space ------------------------------------------------------------
ax = axes[1]
lgrid = np.linspace(np.log(3), np.log(200), 600)
for role, n, c in [('risky', n_risky, C_RISKY), ('safe', n_safe, C_SAFE)]:
    ax.hist(np.log(n), bins=40, range=(np.log(6), np.log(120)), density=True,
            color=c, alpha=.22, lw=0)
    mu = np.median(LOG[(role, 'mu')])
    sd = np.median(LOG[(role, 'sd')])
    ax.plot(lgrid, norm.pdf(lgrid, mu, sd), color=c, lw=1.5)
    h = az.hdi(LOG[(role, 'mu')], hdi_prob=.95)
    ax.plot([h[0], h[1]], [-.05, -.05], color=c, lw=2.2,
            solid_capstyle='butt', clip_on=False)
    ax.plot([mu], [-.05], 'o', color=c, ms=4, clip_on=False)
    ax.text(mu, norm.pdf(mu, mu, sd) + .06, f'{np.exp(mu):.0f} CHF',
            ha='center', fontsize=6.8, color=c, **BOLD)
ax.set_xlim(np.log(4), np.log(160))
ax.set_ylim(-.1, 1.85)
ticks = [7, 14, 28, 56, 112]
ax.set_xticks(np.log(ticks))
ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel('Payoff (CHF, log scale)')
ax.set_ylabel('Density (per log unit)')
ax.set_title('Log-space PMC\n(priors land on the payoffs)', fontsize=8.5,
             **BOLD)

sns.despine(fig=fig, offset=4)
fig.tight_layout()
fig.savefig(OUT / 'prior_placement.pdf')
fig.savefig(OUT / 'prior_placement.png', dpi=150)

print('natural space:')
for role in ('risky', 'safe'):
    mu, sd = NAT[(role, 'mu')], NAT[(role, 'sd')]
    h = az.hdi(mu, hdi_prob=.95)
    obs = n_risky if role == 'risky' else n_safe
    z = (obs.mean() - np.median(mu)) / np.median(sd)
    print(f'  {role:6s} prior {np.median(mu):+7.2f} CHF [{h[0]:+.2f},{h[1]:+.2f}] '
          f'sd {np.median(sd):.2f}  -> mean payoff sits {z:.1f} prior-SDs above it')
print('log space:')
for role in ('risky', 'safe'):
    mu, sd = LOG[(role, 'mu')], LOG[(role, 'sd')]
    obs = n_risky if role == 'risky' else n_safe
    z = (np.log(obs).mean() - np.median(mu)) / np.median(sd)
    print(f'  {role:6s} prior {np.exp(np.median(mu)):7.2f} CHF x/ '
          f'{np.exp(np.median(sd)):.2f}  -> mean log payoff sits {z:+.1f} prior-SDs away')
print('wrote', OUT / 'prior_placement.pdf')
