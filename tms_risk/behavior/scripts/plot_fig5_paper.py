"""Figure 5 in the fig5.flexible2nf design, for any log-space PMC (exploratory).

Defaults to the primary model lfx2-bs3-m2-dp-bm. Spline counts per noise
channel are read from the draws file, so asymmetric orders (e.g. 2-df memory
+ 5-df perceptual) work unchanged.

Rows = presentation order. A: perceived value change under cTBS per option
(%, at the five safe payoffs; risky evaluated at 2x safe). B: perceived
risky/safe ratio IPS/vertex (map). C: leverage, dP per CHF of risky payoff
(vertex, map). D: dP(chose risky) IPS - vertex (map). E: model vs observed
dP by safe payoff (posterior-predictive band).

Every model quantity is computed per subject x draw, averaged across the 35
subjects WITHIN draw, then summarized across draws (the aggregation that
matches how the data are summarized).
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
    'lines.linewidth': 1.2, 'legend.frameon': False, 'legend.fontsize': 6.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.03,
})

C_SAFE, C_RISKY = '#4A4A4A', '#8172B2'
VC = 'stimulation_condition[T.vertex]'
THR = np.log(1 / 0.55)
N_DR = 200            # draws used

import argparse
_p = argparse.ArgumentParser(description=__doc__)
_p.add_argument('--label', default='lfx2-bs3-m2-dp-bm')
_p.add_argument('--draws', default='m2bm_subject_draws.tsv.gz',
                help='subject-draws file under notes/data/')
_p.add_argument('--delta', default='diag_delta_by_safe.lfx2-bs3-m2-dp-bm.tsv',
                help='delta-by-safe-payoff PPC table under notes/data/')
_p.add_argument('--out', default='fig5_paper_m2bm')
_p.add_argument('--name', default='lfx2-bs3-m2-dp-bm (primary)')
ARGS = _p.parse_args()
LABEL = ARGS.label

from tms_risk.behavior.fit_model import build_model, get_data

df = get_data('/data/ds-tmsrisk', model_label=LABEL)
model = build_model(LABEL, df)
subj = pd.read_csv(DATA / ARGS.draws, sep='\t')
softplus = lambda x: np.logaddexp(0, x)
SUBS = sorted(subj['subject'].unique())
# spline count per channel, read from the trace rather than assumed
NSPL = {t: len([v for v in subj['var'].unique()
                if v.startswith(f'{t}_noise_sd_spline')
                and not v.endswith('_offset')])
        for t in ('perceptual', 'memory')}


def s_get(sub, var, reg='Intercept'):
    q = subj.query('subject == @sub and var == @var and regressor == @reg')
    return q.sort_values('draw')['value'].values[:N_DR]


def coefs_all(term, reg):
    return np.stack([np.stack([s_get(sub, f'{term}_noise_sd_spline{j}', reg)
                               for j in range(1, NSPL[term] + 1)], 1)
                     for sub in SUBS], 0)


SC = {(t, r): coefs_all(t, r).astype(np.float32)
      for t in ('perceptual', 'memory') for r in ('Intercept', VC)}
PRIOR = {v: np.stack([s_get(sub, v) for sub in SUBS], 0).astype(np.float32)
         for v in ('risky_prior_mu', 'risky_prior_sd',
                   'safe_prior_mu', 'safe_prior_sd')}


def nu(n, first, vertex):
    """(subj, draw, len(n)) noise SD."""
    nc = np.clip(n, 7 + 1e-6, 112 - 1e-6)
    bp = model.make_dm(nc, variable='perceptual_noise_sd'
                       ).astype(np.float32)[:, :NSPL['perceptual']]
    cp = SC[('perceptual', 'Intercept')]
    if vertex:
        cp = cp + SC[('perceptual', VC)]
    pre = np.einsum('sdj,gj->sdg', cp, bp)
    if first:
        bm = model.make_dm(nc, variable='memory_noise_sd'
                           ).astype(np.float32)[:, :NSPL['memory']]
        cm = SC[('memory', 'Intercept')]
        if vertex:
            cm = cm + SC[('memory', VC)]
        pre = pre + np.einsum('sdj,gj->sdg', cm, bm)
    return softplus(pre)


def percept(n, first, role, vertex):
    mu = PRIOR[f'{role}_prior_mu'][:, :N_DR, None]
    sd = softplus(PRIOR[f'{role}_prior_sd'][:, :N_DR, None])
    v = nu(n, first, vertex)
    w = sd ** 2 / (sd ** 2 + v ** 2)
    return w * np.log(n)[None, None, :] + (1 - w) * mu, v, w


def p_risky(n_safe, ratio, risky_first, vertex):
    n_r = np.asarray(n_safe) * np.asarray(ratio)
    lr, vr, wr = percept(n_r, risky_first, 'risky', vertex)
    ls, vs, ws = percept(np.asarray(n_safe), not risky_first, 'safe', vertex)
    P = norm.cdf((lr - ls - THR) / np.hypot(vr, vs))
    return P, lr, ls, vr, vs, wr


SAFE5 = np.array([7., 10., 14., 20., 28.])
RG = np.linspace(1.0, 4.0, 31)
SG = np.exp(np.linspace(np.log(7), np.log(28), 26))
SS, RR = np.meshgrid(SG, RG)
sflat, rflat = SS.ravel(), RR.ravel()

fig, axes = plt.subplots(2, 5, figsize=(7.4, 3.8), constrained_layout=True)
ORDERS = [(True, 'Risky first'), (False, 'Risky second')]
maps = {}
a_extents = []
for row, (rf_bool, oname) in enumerate(ORDERS):
    # --- A ----------------------------------------------------------------
    ax = axes[row, 0]
    curves = {}
    for role, first, c, name in [('safe', not rf_bool, C_SAFE, 'Safe option'),
                                 ('risky', rf_bool, C_RISKY, 'Risky option')]:
        n = SAFE5 * 2.0 if role == 'risky' else SAFE5
        li, _, _ = percept(n, first, role, False)
        lv, _, _ = percept(n, first, role, True)
        pct = ((li - lv) * 100).mean(0)   # draws x 5, log points (~%)
        med = np.median(pct, 0)
        lo, hi = np.percentile(pct, [2.5, 97.5], axis=0)
        ax.errorbar(SAFE5, med, yerr=[med - lo, hi - med], fmt='o-', color=c,
                    ms=3, lw=1.2, elinewidth=.8, capsize=0, label=name)
        curves[role] = med
        a_extents += [lo.min(), hi.max()]
    ax.fill_between(SAFE5, curves['safe'], curves['risky'], color='.85',
                    alpha=.6, zorder=0)
    ax.axhline(0, color='.8', lw=.5, ls='--', zorder=0)
    ax.set_xscale('log')
    ax.set_xticks(SAFE5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylabel(f'{oname}\nΔ perceived value (%)', fontsize=7.5)
    if row == 0:
        ax.legend(loc='upper left', handlelength=1.2)
        ax.set_title('Perceived value change\nunder cTBS', fontsize=7.5)
    if row == 1:
        ax.set_xlabel('Safe payoff (CHF)')

    # --- maps -------------------------------------------------------------
    P_i, lr_i, ls_i, _, _, _ = p_risky(sflat, rflat, rf_bool, False)
    P_v, lr_v, ls_v, vr_v, vs_v, wr_v = p_risky(sflat, rflat, rf_bool, True)
    ratio_map = np.exp(((lr_i - ls_i) - (lr_v - ls_v)).mean((0, 1)))
    z = norm.ppf(np.clip(P_v, 1e-6, 1 - 1e-6))
    lever = (norm.pdf(z) / np.hypot(vr_v, vs_v) * wr_v
             / (sflat * rflat)[None, None, :]).mean((0, 1))
    eff = (P_i - P_v).mean((0, 1))
    maps[(row, 'B')] = ratio_map.reshape(RR.shape)
    maps[(row, 'C')] = lever.reshape(RR.shape)
    maps[(row, 'D')] = eff.reshape(RR.shape)
    maps[(row, 'Pv')] = P_v.mean((0, 1)).reshape(RR.shape)

    # --- E ----------------------------------------------------------------
    ax = axes[row, 4]
    e_all = pd.read_csv(DATA / ARGS.delta, sep='\t')
    e = e_all[e_all.order == oname].sort_values('n_safe')
    ax.fill_between(e['n_safe'], e['lo'], e['hi'], color='.85', lw=0)
    ax.plot(e['n_safe'], e['median'], color='.35', lw=1.4)
    ax.errorbar(e['n_safe'], e['obs'], yerr=e['obs_sem'], fmt='o', color='.1',
                ms=3.6, elinewidth=.8, capsize=0, zorder=4)
    ax.axhline(0, color='.8', lw=.5, ls='--', zorder=0)
    ax.set_xscale('log')
    ax.set_xticks(SAFE5)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    e_lo = min(e_all['lo'].min(), (e_all['obs'] - e_all['obs_sem']).min())
    e_hi = max(e_all['hi'].max(), (e_all['obs'] + e_all['obs_sem']).max())
    e_pad = .06 * (e_hi - e_lo)
    ax.set_ylim(e_lo - e_pad, e_hi + e_pad)
    ax.set_yticks([-.05, 0, .05, .1, .15])
    if row == 0:
        ax.set_title('Model vs observed\nΔ P(chose risky)', fontsize=7.5)
        ax.plot([], [], 'o', color='.1', ms=3.6, label='Observed')
        ax.plot([], [], color='.35', lw=1.4, label='Model')
        ax.legend(loc='upper right', handlelength=1.2)
    if row == 1:
        ax.set_xlabel('Safe payoff (CHF)')

# Shared panel-A ylim across rows, from the data
pad = .05 * (max(a_extents) - min(a_extents))
for row in (0, 1):
    axes[row, 0].set_ylim(min(a_extents) - pad, max(a_extents) + pad)

# Map scales: symmetric around the neutral value, 98th pct of |deviation|
dev_b = np.percentile(
    np.abs(np.concatenate([maps[(r, 'B')].ravel() for r in (0, 1)]) - 1), 98)
dev_d = np.percentile(
    np.abs(np.concatenate([maps[(r, 'D')].ravel() for r in (0, 1)])), 98)
SPECS = [('B', 'Perceived risky/safe\nratio, IPS / vertex', 'RdBu_r',
          1 - dev_b, 1 + dev_b),
         ('C', 'Leverage\n(ΔP per CHF)', 'viridis', None, None),
         ('D', 'Δ P(chose risky)\nIPS − vertex', 'RdBu_r', -dev_d, dev_d)]
for col, (key, title, cmap, vmin, vmax) in enumerate(SPECS, start=1):
    if vmin is None:
        vmax = max(maps[(r, key)].max() for r in (0, 1))
        vmin = 0
    for row in (0, 1):
        ax = axes[row, col]
        pc = ax.pcolormesh(SG, RG, maps[(row, key)], cmap=cmap, vmin=vmin,
                           vmax=vmax, shading='gouraud', rasterized=True)
        cs = ax.contour(SG, RG, maps[(row, 'Pv')], levels=[.2, .5, .8],
                        colors='k', linewidths=[.4, .9, .4], alpha=.6)
        ax.clabel(cs, fmt=lambda v: f'{v:.0%}', fontsize=5.5, inline=True)
        ax.axhline(1.82, color='.2', lw=.5, ls=':', zorder=3)
        ax.set_xscale('log')
        ax.set_xticks(SAFE5)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:g}'))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_yticks([1, 2, 3, 4])
        if row == 0:
            ax.set_title(title, fontsize=7.5)
        if col == 1:
            ax.set_ylabel('Risky/safe ratio', fontsize=7.5)
        else:
            ax.set_yticklabels([])
        if row == 1:
            ax.set_xlabel('Safe payoff (CHF)')
            cb = fig.colorbar(pc, ax=axes[:, col], orientation='horizontal',
                              shrink=.85, pad=.03, aspect=20)
            cb.locator = mticker.MaxNLocator(3)
            cb.update_ticks()

fig.suptitle(f'Figure 5 — {ARGS.name}, subject-averaged; contours: vertex '
             'P(risky); dotted: risk-neutral ratio', fontsize=8, color='.35',
             y=1.04)
sns.despine(fig=fig, offset=3)
fig.savefig(OUT / f'{ARGS.out}.pdf')
fig.savefig(OUT / f'{ARGS.out}.png', dpi=150)
print('wrote', OUT / f'{ARGS.out}.pdf', '| splines:', NSPL)
