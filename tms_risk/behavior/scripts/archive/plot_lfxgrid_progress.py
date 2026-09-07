"""First-look dashboard for whatever lfx2 grid cells have completed.

Reads notes/data/lfxgrid_{loo,draws}.tsv (extracted on the cluster).
Panels: (a) ELPD forest of completed cells; (b) perceptual noise curves by
basis (null cells) — the tail-behavior comparison; (c) cTBS contrast for
completed b-cells; (d) risky-prior medians (realism check). Data-driven:
renders whichever cells exist.
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

C_BASIS = {'bs3': '#33619E', 'bs2': '#B3873B', 'cr3': '#4E7D52'}
BASIS_NAME = {'bs3': 'Cubic B-spline', 'bs2': 'Quadratic B-spline',
              'cr3': 'Natural cubic'}
VC = 'stimulation_condition[T.vertex]'

loo = pd.read_csv(DATA / 'lfxgrid_loo.tsv', sep='\t')
draws = pd.read_csv(DATA / 'lfxgrid_draws.tsv', sep='\t')
parts = loo['label'].str.extract(r'lfx2-(?P<basis>\w+)-(?P<mem>\w+)-(?P<hp>\w+)-(?P<tms>\w+)')
loo = pd.concat([loo, parts], axis=1)
print(loo[['label', 'elpd_loo', 'se', 'divergences', 'max_rhat']].round(2).to_string(index=False))

from tms_risk.behavior.fit_model import build_model, get_data
df = get_data('/data/ds-tmsrisk', model_label='lfx2-bs3-fm-dp-null')
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 100))
softplus = lambda x: np.logaddexp(0, x)
_models = {}


def model_for(label):
    if label not in _models:
        _models[label] = build_model(label, df)
    return _models[label]


def coefs(label, term, reg):
    out = []
    for j in range(1, 6):
        v = f'{term}_noise_sd_spline{j}'
        s = draws.query('label == @label and var == @v and regressor == @reg')
        out.append(s.sort_values('draw')['value'].values if len(s)
                   else np.zeros(500))
    return np.stack(out, 1)


def nu(label, term, vertex=False):
    c = coefs(label, term, 'Intercept')
    if vertex:
        c = c + coefs(label, term, VC)
    basis = model_for(label).make_dm(N_GRID, variable=f'{term}_noise_sd')
    k = basis.shape[1]
    return softplus(c[:, :k] @ basis.T)


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks([7, 15, 30, 60, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


fig, axes = plt.subplots(2, 2, figsize=(7.25, 5.6), constrained_layout=True)

# a — ELPD forest of completed cells
ax = axes[0, 0]
lo_sorted = loo.sort_values('elpd_loo')
best = lo_sorted['elpd_loo'].max()
for y, (_, r) in enumerate(lo_sorted.iterrows()):
    c = C_BASIS[r.basis]
    ax.errorbar(r.elpd_loo - best, y, xerr=r.se, fmt='o', color=c, ms=4.5,
                mfc=c if r.hp == 'dp' else 'white', mew=1.1, elinewidth=1.0,
                capsize=0)
ax.set_yticks(range(len(lo_sorted)))
ax.set_yticklabels([f'{r.basis}·{r.mem}·{r.hp}·{r.tms}'
                    for _, r in lo_sorted.iterrows()], fontsize=7)
ax.axvline(0, color='0.8', lw=.6, ls='--', zorder=0)
ax.set_xlabel('ELPD − best completed (±SE)')
ax.set_title(f'{len(loo)} of 24 cells completed', fontsize=9)
ax.text(0.03, 0.97, 'Open markers:\ntightened priors', transform=ax.transAxes,
        fontsize=7, va='top', color='.4')

# b — perceptual noise curves, null cells: the tail-behavior comparison
ax = axes[0, 1]
null_cells = loo.query('tms == "null"')
for _, r in null_cells.iterrows():
    y = nu(r.label, 'perceptual')
    med = np.median(y, 0)
    ax.plot(N_GRID, med, color=C_BASIS[r.basis], lw=1.4,
            ls='-' if r.hp == 'dp' else (0, (4, 2)),
            alpha=1.0 if r.mem == 'sm' else 0.55)
logx(ax)
ax.set_ylabel('Perceptual noise SD (log units)')
ax.set_title('Noise curves, null cells (dashed = tight priors)', fontsize=9)
for i, (b, name) in enumerate(BASIS_NAME.items()):
    if (null_cells.basis == b).any():
        ax.text(0.03, 0.97 - i * .09, name, color=C_BASIS[b], fontsize=8,
                transform=ax.transAxes, va='top')

# c — cTBS contrast for completed b-cells (absolute, log units)
ax = axes[1, 0]
ax.axhline(0, color='0.8', lw=.6, ls='--', zorder=0)
b_cells = loo.query('tms == "b"')
for _, r in b_cells.iterrows():
    d = nu(r.label, 'perceptual', vertex=False) - nu(r.label, 'perceptual', vertex=True)
    med = np.median(d, 0)
    h = np.array([az.hdi(d[:, i], hdi_prob=.95) for i in range(d.shape[1])])
    ax.fill_between(N_GRID, h[:, 0], h[:, 1], color=C_BASIS[r.basis],
                    alpha=.12, lw=0)
    ax.plot(N_GRID, med, color=C_BASIS[r.basis], lw=1.4,
            ls='-' if r.hp == 'dp' else (0, (4, 2)))
logx(ax)
ax.set_ylabel('Δ noise, IPS − Vertex (log units)')
ax.set_title('cTBS contrast, completed b-cells', fontsize=9)

# d — risky prior medians per cell (realism check)
ax = axes[1, 1]
y = 0
for _, r in lo_sorted.iterrows():
    s = draws.query('label == @r.label and var == "risky_prior_mu" and '
                    'regressor == "Intercept"')['value'].values
    if not len(s):
        continue
    med, lo_, hi_ = np.exp(np.median(s)), np.exp(np.percentile(s, 2.5)), \
        np.exp(np.percentile(s, 97.5))
    c = C_BASIS[r.basis]
    ax.hlines(y, lo_, hi_, color=c, lw=1.1)
    ax.plot(med, y, 'o', color=c, ms=4, mfc=c if r.hp == 'dp' else 'white',
            mew=1.0)
    y += 1
ax.axvline(31, color='0.6', lw=.7, ls=':')
ax.text(31, y - .2, 'Payoff geomean', color='0.5', fontsize=7.5, ha='center',
        va='bottom')
ax.set_yticks([])
ax.spines['left'].set_visible(False)
ax.set_xlabel('Risky prior median (CHF)')
ax.set_xlim(0, 60)
ax.set_title('Priors stay realistic?', fontsize=9)

for ax_, letter in zip(axes.ravel(), 'abcd'):
    ax_.text(-0.18, 1.04, letter, transform=ax_.transAxes, fontsize=12,
             fontweight='bold', fontfamily='Arial', va='bottom', ha='right')

sns.despine(fig=fig, offset=5, trim=True)
fig.savefig(OUT / 'lfxgrid_progress.pdf')
fig.savefig(OUT / 'lfxgrid_progress.png', dpi=150)
print('wrote', OUT / 'lfxgrid_progress.pdf')
