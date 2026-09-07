"""Figure 5 draft, ported to the new primary model (lfx2-bs3-sm-dp-b).

Same argument as plot_fig5.py, left to right, rows = presentation order:
  A  what cTBS does to each option's perceived value (percent, per option)
  B  the perceived risky/safe EV ratio, IPS / vertex (map)
  C  where choice is sensitive: leverage |dP/d log-ratio| under vertex (map)
  D  the predicted behavioural effect ΔP(risky), IPS − vertex (map)
  E  that prediction against the data, by stake tercile

All quantities are closed-form in the log-space observer:
  ν_first(n)  = softplus(c_mem + Σ c_perc·B(log n)),  ν_second = softplus(Σ c_perc·B)
  w_role(n)   = σ_role² / (σ_role² + ν_pos(n)²)
  log x̂      = w·log n + (1−w)·μ_role
  P(risky)    = Φ((Δlogx̂_towards_risky − log(1/0.55)) / sqrt(ν₁² + ν₂²))
Group-level draws (500); maps show draw-averaged quantities (the paper
version averages over subjects too — port to subject level in step 4.4b).
DRAFT — exploratory tree, not the paper figure.
"""
from pathlib import Path
import sys

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
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
})

import os
C_IPS, C_VERTEX = '#d62728', '#2ca02c'
VC = 'stimulation_condition[T.vertex]'
LABEL = os.environ.get('FIG5_LABEL', 'lfx2-bs3-sm-dp-b')
DRAWS_FILE = ('logflex_group_draws.tsv' if LABEL.startswith('logflex')
              else 'lfxgrid_draws.tsv')
THR = np.log(1 / 0.55)

from tms_risk.behavior.fit_model import build_model, get_data

df = get_data('/data/ds-tmsrisk', model_label=LABEL)
model = build_model(LABEL, df)
grp = pd.read_csv(DATA / DRAWS_FILE, sep='\t').query('label == @LABEL')
softplus = lambda x: np.logaddexp(0, x)


def g(var, reg='Intercept'):
    s = grp.query('var == @var and regressor == @reg')
    return s.sort_values('draw')['value'].values


def coef_stack(term, vertex):
    base = np.stack([g(f'{term}_noise_sd_spline{j}') for j in range(1, 6)
                     if len(grp.query(f'var == "{term}_noise_sd_spline{j}"'))], 1)
    if vertex:
        add = np.stack([g(f'{term}_noise_sd_spline{j}', VC)
                        if len(grp.query(f'var == "{term}_noise_sd_spline{j}" '
                                         f'and regressor == "{VC}"')) else
                        np.zeros(base.shape[0])
                        for j in range(1, base.shape[1] + 1)], 1)
        base = base + add
    return base

P_COEFS = {v: coef_stack('perceptual', v) for v in (False, True)}
M_COEFS = {v: coef_stack('memory', v) for v in (False, True)}
MEM_SCALAR = M_COEFS[False].shape[1] == 1
PRIOR = {'risky': (g('risky_prior_mu')[:, None], softplus(g('risky_prior_sd'))[:, None]),
         'safe': (g('safe_prior_mu')[:, None], softplus(g('safe_prior_sd'))[:, None])}
N_DRAWS = P_COEFS[False].shape[0]


def nu(n, first, vertex):
    """Noise SD for payoffs n (1d array) in a position. draws x len(n)."""
    nc = np.clip(n, 7 + 1e-6, 112 - 1e-6)
    basis = model.make_dm(nc, variable='perceptual_noise_sd')
    pre = P_COEFS[vertex] @ basis.T
    if first:
        if MEM_SCALAR:
            pre = pre + M_COEFS[vertex][:, :1]
        else:
            mb = model.make_dm(nc, variable='memory_noise_sd')
            pre = pre + M_COEFS[vertex] @ mb.T
    return softplus(pre)


def percept_log(n, first, role, vertex):
    mu, sd = PRIOR[role]
    v = nu(n, first, vertex)
    w = sd ** 2 / (sd ** 2 + v ** 2)
    return w * np.log(n)[None, :] + (1 - w) * mu, v


def p_risky(n_safe, ratio, risky_first, vertex):
    """P(choose risky). n_safe, ratio broadcastable 1d arrays (flattened grid)."""
    n_r = n_safe * ratio
    lx_r, v_r = percept_log(n_r, risky_first, 'risky', vertex)
    lx_s, v_s = percept_log(n_safe, not risky_first, 'safe', vertex)
    return norm.cdf((lx_r - lx_s - THR) / np.hypot(v_r, v_s)), lx_r, lx_s


SAFE5 = np.array([7., 10., 14., 20., 28.])
RG = np.linspace(1.0, 4.0, 41)
SG = np.exp(np.linspace(np.log(7), np.log(28), 36))
SS, RR = np.meshgrid(SG, RG)
sf, rf = SS.ravel(), RR.ravel()

fig, axes = plt.subplots(2, 5, figsize=(7.25, 3.7), constrained_layout=True)
ORDERS = [(True, 'Risky first'), (False, 'Risky second')]
maps = {}
for row, (rf_bool, oname) in enumerate(ORDERS):
    # --- A: per-option percent percept change at the 5 safe payoffs -------
    ax = axes[row, 0]
    for role, first, c, name in [('risky', rf_bool, '.15', 'Risky'),
                                 ('safe', not rf_bool, '.6', 'Safe')]:
        n = SAFE5 * 2.0 if role == 'risky' else SAFE5   # risky at mean ratio 2
        li, _ = percept_log(n, first, role, False)
        lv, _ = percept_log(n, first, role, True)
        pct = (np.exp(li - lv) - 1) * 100
        med = np.median(pct, 0)
        lo, hi = np.percentile(pct, [2.5, 97.5], axis=0)
        ax.fill_between(SAFE5, lo, hi, color=c, alpha=.18, lw=0)
        ax.plot(SAFE5, med, 'o-', color=c, ms=3, lw=1.2)
        if row == 0:
            ax.text(0.95, .92 - .14 * (role == 'safe'), name, color=c,
                    fontsize=7, transform=ax.transAxes, ha='right')
    ax.axhline(0, color='0.85', lw=.5, ls='--', zorder=0)
    ax.set_xticks([7, 14, 28])
    ax.set_ylim(-4.5, 2.5)
    ax.set_ylabel(f'{oname}\nΔ percept, IPS vs vertex (%)', fontsize=7.5)
    if row == 1:
        ax.set_xlabel('Safe payoff (CHF)')

    # --- B/C/D maps -------------------------------------------------------
    P_i, lxr_i, lxs_i = p_risky(sf, rf, rf_bool, False)
    P_v, lxr_v, lxs_v = p_risky(sf, rf, rf_bool, True)
    cause = np.exp((lxr_i - lxs_i) - (lxr_v - lxs_v)).mean(0).reshape(RR.shape)
    # leverage under vertex: dP/d(log-ratio distortion) = phi(z)/sigma
    _, v_r = percept_log(sf * rf, rf_bool, 'risky', True)
    _, v_s = percept_log(sf, not rf_bool, 'safe', True)
    z = norm.ppf(np.clip(P_v, 1e-6, 1 - 1e-6))
    lev = (norm.pdf(z) / np.hypot(v_r, v_s)).mean(0).reshape(RR.shape)
    eff = (P_i - P_v).mean(0).reshape(RR.shape)
    maps[(row, 'B')] = cause
    maps[(row, 'C')] = lev
    maps[(row, 'D')] = eff
    maps[(row, 'Pv')] = P_v.mean(0).reshape(RR.shape)

    # --- E: posterior-predictive vs observed dP by stake ------------------
    # (the winner's real PPC: subject-level simulation, correct aggregation —
    # NOT the group-median closed form, which falls into the mean-parameter
    # trap and misorders the stake profile.)
    ax = axes[row, 4]
    _ppc_path = DATA / ('ppc_by_stake.lfx2winner.tsv'
                        if LABEL == 'lfx2-bs3-sm-dp-b' else
                        f'ppc_by_stake.{LABEL}.tsv')
    if not _ppc_path.exists():
        ax.text(.5, .5, 'PPC pending', ha='center', va='center', fontsize=8,
                color='.5', transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        continue_e = False
    ppc = pd.read_csv(_ppc_path, sep='\t') if _ppc_path.exists() else None
    if ppc is None:
        continue
    pp = ppc[ppc.order == oname].pivot_table(index='stake', columns='stim',
                                             values=['mean', 'observed'])
    d_mod = pp[('mean', 'ips')] - pp[('mean', 'vertex')]
    d_obs = pp[('observed', 'ips')] - pp[('observed', 'vertex')]
    stakes = d_mod.index.values
    ax.plot(stakes, d_obs.values, 's', color='.15', ms=5, zorder=4)
    ax.plot(stakes, d_mod.values, 'o-', color=C_IPS, ms=3, lw=1.2)
    ax.axhline(0, color='0.85', lw=.5, ls='--', zorder=0)
    ax.set_ylim(-0.05, 0.1)
    ax.set_xticks([13, 23, 42])
    if row == 0:
        ax.text(0.05, .92, 'Model', color=C_IPS, fontsize=7,
                transform=ax.transAxes)
        ax.text(0.05, .78, 'Data', color='.15', fontsize=7,
                transform=ax.transAxes)
    ax.set_ylabel('ΔP(risky), IPS − vertex', fontsize=7.5)
    if row == 1:
        ax.set_xlabel('Stake (CHF)')

# shared scales per map column: data-driven, symmetric for diverging maps
b_all = np.concatenate([maps[(r, 'B')].ravel() for r in (0, 1)])
d_all = np.concatenate([maps[(r, 'D')].ravel() for r in (0, 1)])
b_amp = np.percentile(np.abs(np.log(b_all)), 98)
d_amp = np.percentile(np.abs(d_all), 98)
print(f'map ranges: cause {b_all.min():.4f}-{b_all.max():.4f}  '
      f'effect {d_all.min():.4f}-{d_all.max():.4f}')
SPECS = [('B', 'Perceived EV ratio,\nIPS / vertex', 'RdBu_r',
          float(np.exp(-b_amp)), float(np.exp(b_amp))),
         ('C', 'Choice sensitivity\n(vertex leverage)', 'mako', None, None),
         ('D', 'ΔP(risky),\nIPS − vertex', 'RdBu_r', -float(d_amp), float(d_amp))]
for col, (key, title, cmap, vmin, vmax) in enumerate(SPECS, start=1):
    if vmin is None:
        vmax = max(maps[(r, key)].max() for r in (0, 1))
        vmin = 0
    for row in (0, 1):
        ax = axes[row, col]
        pc = ax.pcolormesh(SG, RG, maps[(row, key)], cmap=cmap, vmin=vmin,
                           vmax=vmax, shading='gouraud', rasterized=True)
        cs = ax.contour(SG, RG, maps[(row, 'Pv')], levels=[.2, .5, .8],
                        colors='k', linewidths=[.4, .8, .4], alpha=.55)
        ax.set_xscale('log')
        ax.set_xticks([7, 14, 28])
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yticks([1, 2, 3, 4])
        if row == 0:
            ax.set_title(title, fontsize=7.5)
        if row == 1:
            ax.set_xlabel('Safe payoff (CHF)')
        if col == 1:
            ax.set_ylabel('Risky / safe ratio', fontsize=7.5)
        if row == 1:
            fig.colorbar(pc, ax=axes[:, col], orientation='horizontal',
                         shrink=.85, pad=.02, aspect=22)

fig.suptitle('Figure 5 draft — log-space primary model (lfx2-bs3-sm-dp-b); '
             'contours: vertex P(risky) = 0.2 / 0.5 / 0.8',
             fontsize=8, color='.3', y=1.03)
sns.despine(fig=fig, offset=3)
stem = 'fig5_logspace_draft' if LABEL == 'lfx2-bs3-sm-dp-b' else f'fig5_{LABEL}'
fig.savefig(OUT / f'{stem}.pdf')
fig.savefig(OUT / f'{stem}.png', dpi=150)
print('wrote', OUT / f'{stem}.pdf')
