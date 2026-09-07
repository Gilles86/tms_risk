"""Group + individual parameter estimates for the flexible PMC (exploratory).

Trace: flexible2_noisefix.head (TMS on both noise terms, 35 subjects).
Top: group-level posteriors (priors in CHF; noise at a 20-CHF reference by
condition). Below: per-subject posteriors (median + 95% HDI), subjects sorted.
Noise values are softplus(spline coefficients · basis at n=20); the basis is
the fit-time-anchored one saved in flexible_spline_dm.tsv.
"""
from pathlib import Path

import arviz as az
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parents[3]
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
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

C_IPS, C_VERTEX = '#d62728', '#2ca02c'
N_REF = 20.0
VERTEX_COEF = 'stimulation_condition[T.vertex]'

subj = pd.read_csv(DATA / 'flexible_subject_draws.tsv.gz', sep='\t')
grp = pd.read_csv(DATA / 'flexible_group_draws.tsv', sep='\t') \
        .query('label == "flexible2_tmsboth"')
dm = pd.read_csv(DATA / 'flexible_spline_dm.tsv', sep='\t')


def softplus(x):
    return np.logaddexp(0, x)


def basis_at(term, n):
    return np.array([interp1d(dm['x'], dm[f'{term}_noise_sd_dm{j}'])(n)
                     for j in range(1, 6)])


def subj_draws(var, regressor='Intercept'):
    """(n_subjects, 300) array, subject-sorted by id."""
    sel = subj.query('var == @var and regressor == @regressor')
    return (sel.pivot_table(index='subject', columns='draw', values='value')
              .sort_index())


def grp_draws(var, regressor='Intercept'):
    sel = grp.query('var == @var and regressor == @regressor')
    return sel.sort_values('draw')['value'].values


def subj_nu(term, n, vertex):
    coefs = np.stack([subj_draws(f'{term}_noise_sd_spline{j}').values
                      for j in range(1, 6)], axis=1)          # subj x 5 x draws
    if vertex:
        coefs = coefs + np.stack(
            [subj_draws(f'{term}_noise_sd_spline{j}', VERTEX_COEF).values
             for j in range(1, 6)], axis=1)
    b = basis_at(term, n)
    return softplus(np.einsum('sjd,j->sd', coefs, b))          # subj x draws


def grp_nu(term, n, vertex):
    coefs = np.stack([grp_draws(f'{term}_noise_sd_spline{j}')
                      for j in range(1, 6)], axis=1)           # draws x 5
    if vertex:
        coefs = coefs + np.stack(
            [grp_draws(f'{term}_noise_sd_spline{j}', VERTEX_COEF)
             for j in range(1, 6)], axis=1)
    return softplus(coefs @ basis_at(term, n))


def med_hdi(a, axis=-1):
    med = np.median(a, axis=axis)
    if a.ndim == 1:
        lo, hi = az.hdi(a, hdi_prob=.95)
        return med, lo, hi
    h = np.array([az.hdi(row, hdi_prob=.95) for row in a])
    return med, h[:, 0], h[:, 1]


fig = plt.figure(figsize=(7.25, 7.0), constrained_layout=True)
gs = fig.add_gridspec(4, 2, height_ratios=[1.15, 1, 1, 1])

# --- a: group priors (transformed scale: sd through softplus)
ax = fig.add_subplot(gs[0, 0])
labels = [('risky_prior_mu', 'Risky prior μ', False),
          ('risky_prior_sd', 'Risky prior σ', True),
          ('safe_prior_mu', 'Safe prior μ', False),
          ('safe_prior_sd', 'Safe prior σ', True)]
for y, (var, name, is_sd) in enumerate(labels):
    s = grp_draws(var)
    s = softplus(s) if is_sd else s
    med, lo, hi = med_hdi(s)
    ax.hlines(-y, lo, hi, color='.2', lw=1.2)
    ax.plot(med, -y, 'o', color='.2', ms=4.5)
ax.set_yticks([-y for y in range(4)])
ax.set_yticklabels([n for _, n, _ in labels])
ax.axvline(0, color='.8', lw=.6, ls=':')
ax.set_xlabel('CHF')
ax.set_xticks([0, 4, 8, 12])
ax.set_title('Group priors', fontsize=9)

# --- b: group noise at the 20-CHF reference, by term and condition
ax = fig.add_subplot(gs[0, 1])
y = 0
yt, ytl = [], []
for term, tname in [('perceptual', 'Perceptual ν(20)'), ('memory', 'Memory ν(20)')]:
    for vertex, c in [(False, C_IPS), (True, C_VERTEX)]:
        med, lo, hi = med_hdi(grp_nu(term, N_REF, vertex))
        ax.hlines(y, lo, hi, color=c, lw=1.2)
        ax.plot(med, y, 'o', color=c, ms=4.5)
        y -= 1
    yt.append(y + 1.5)
    ytl.append(tname)
    y -= .6
ax.set_yticks(yt)
ax.set_yticklabels(ytl)
ax.set_xlabel('Noise SD at 20 CHF (CHF)')
ax.set_title('Group noise (IPS red, Vertex green)', fontsize=9)

# --- c/d/e: per-subject panels, subjects sorted by the panel's median
def subject_panel(ax, med, lo, hi, ylabel, color='.25', med2=None, lo2=None,
                  hi2=None, color2=None, hline=None):
    order = np.argsort(med)
    x = np.arange(len(med))
    ax.vlines(x, lo[order], hi[order], color=color, lw=.9, alpha=.75)
    ax.plot(x, med[order], 'o', color=color, ms=2.8, zorder=3)
    if med2 is not None:
        ax.vlines(x + .35, lo2[order], hi2[order], color=color2, lw=.9, alpha=.75)
        ax.plot(x + .35, med2[order], 'o', color=color2, ms=2.8, zorder=3)
    if hline is not None:
        ax.axhline(hline, color='.75', lw=.6, ls='--', zorder=0)
    ax.set_xticks([])
    ax.set_ylabel(ylabel)
    ax.set_xlim(-1, len(med) + 1)


sp = {v: subj_draws(v).values for v in
      ['risky_prior_mu', 'safe_prior_mu', 'risky_prior_sd', 'safe_prior_sd']}

ax = fig.add_subplot(gs[1, :])
m1, l1, h1 = med_hdi(sp['risky_prior_mu'])
m2, l2, h2 = med_hdi(sp['safe_prior_mu'])
subject_panel(ax, m1, l1, h1, 'Prior μ (CHF)', color='.2',
              med2=m2, lo2=l2, hi2=h2, color2='.6')
ax.text(.01, .95, 'Risky (dark), safe (light); sorted by risky μ',
        transform=ax.transAxes, fontsize=7.5, va='top', color='.35')

ax = fig.add_subplot(gs[2, :])
nu_i = subj_nu('perceptual', N_REF, vertex=False)
nu_v = subj_nu('perceptual', N_REF, vertex=True)
mi, li, hi_ = med_hdi(nu_i)
mv, lv, hv = med_hdi(nu_v)
subject_panel(ax, mi, li, hi_, 'Perceptual ν(20) (CHF)', color=C_IPS,
              med2=mv, lo2=lv, hi2=hv, color2=C_VERTEX)
ax.text(.01, .95, 'IPS (red), vertex (green); sorted by IPS',
        transform=ax.transAxes, fontsize=7.5, va='top', color='.35')

ax = fig.add_subplot(gs[3, :])
delta = (nu_i - nu_v) / nu_v * 100
md, ld, hd = med_hdi(delta)
subject_panel(ax, md, ld, hd, 'Δν(20), IPS − Vertex (%)', color='.2', hline=0)
frac = float((np.median(delta, 1) > 0).mean())
ax.text(.01, .95, f'{frac:.0%} of subjects with median increase',
        transform=ax.transAxes, fontsize=7.5, va='top', color='.35')
ax.set_xlabel('Subjects (sorted within panel)')

for ax_, letter, pos in zip(fig.axes, 'abcde', range(5)):
    ax_.text(-0.07 if pos > 1 else -0.22, 1.04, letter, transform=ax_.transAxes,
             fontsize=12, fontweight='bold', fontfamily='Arial',
             va='bottom', ha='right')

sns.despine(fig=fig, offset=4)
fig.savefig(OUT / 'flexible_params.pdf')
fig.savefig(OUT / 'flexible_params.png', dpi=150)
print('wrote', OUT / 'flexible_params.pdf')
