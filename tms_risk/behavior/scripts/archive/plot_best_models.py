"""The two converged Δ-PPC-passing log-space models side by side, one row
each: choice PPC by stake (A), the cTBS-contrast PPC (B), and the two noise
channels (C). bs3-m2-bm = linear memory / cubic perceptual basis;
bs2-m3-bm = quadratic memory / quadratic basis. Subject-averaged curves."""
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

IPS, VERTEX = '#d62728', '#2ca02c'
BOLD = dict(fontweight='bold', fontfamily='Arial')
ORDERS = ['Risky first', 'Risky second']
VC = 'stimulation_condition[T.vertex]'
softplus = lambda x: np.logaddexp(0, x)
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 80))

from tms_risk.behavior.fit_model import build_model, get_data

MODELS = [('lfx2-bs3-m2-dp-bm', 'm2bm_subject_draws.tsv.gz',
           'Linear memory (bs3-m2-bm) — ELPD −4149.0'),
          ('lfx2-bs2-m3-dp-bm', 'bs2m3bm_subject_draws.tsv.gz',
           'Quadratic memory (bs2-m3-bm) — ELPD −4155.0')]

fig = plt.figure(figsize=(7.25, 5.4))
ROWTOPS = [(.90, .585), (.42, .105)]

for m, (label, dfile, name) in enumerate(MODELS):
    top, bot = ROWTOPS[m]
    fig.text(.008, top + .04, 'AB'[m], fontsize=11, va='bottom', **BOLD)
    fig.text(.5, top + .045, name, ha='center', fontsize=8.5, **BOLD)

    # choice PPC by stake --------------------------------------------------
    d = pd.read_csv(DATA / f'ppc_by_stake.{label}.tsv', sep='\t')
    stakes = np.sort(d.stake.unique())
    x = np.arange(len(stakes))
    ylo = min(d.lo.min(), d.observed.min()) - .012
    yhi = max(d.hi.max(), d.observed.max()) + .012
    gs = fig.add_gridspec(1, 2, left=.06, right=.40, wspace=.12, top=top,
                          bottom=bot)
    for o, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs[0, o])
        for stim, colr in [('vertex', VERTEX), ('ips', IPS)]:
            s = d[(d.order == order) & (d.stim == stim)].sort_values('stake')
            ax.fill_between(x, s.lo, s.hi, color=colr, alpha=.20, lw=0)
            ax.plot(x, s['mean'], color=colr, lw=1.2)
            dx = .06 if stim == 'ips' else -.06
            ax.plot(x + dx, s.observed, 'o', color=colr, ms=3.4, lw=0,
                    zorder=4)
            for _, r in s.iterrows():
                if r.lo <= r.observed <= r.hi:
                    continue
                xi = float(x[np.argmin(np.abs(stakes - r.stake))]) + dx
                ax.annotate('', xy=(xi, r.observed), xycoords='data',
                            xytext=(-13, 11 if r.observed > r.hi else -11),
                            textcoords='offset points', zorder=6,
                            arrowprops=dict(arrowstyle='-|>', color='.1',
                                            lw=.9, shrinkA=0, shrinkB=3.5,
                                            mutation_scale=6))
        ax.set_ylim(ylo, yhi)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_yticks([.5, .55, .6, .65])
        ax.set_title(order, fontsize=7, color='.3', pad=2, style='italic')
        if m == 1:
            ax.set_xlabel('Stake (CHF)')
        if o == 0:
            ax.set_ylabel('P(chose risky)')
            if m == 0:
                ax.text(.06, .96, 'IPS', transform=ax.transAxes, fontsize=7,
                        color=IPS, va='top')
                ax.text(.06, .84, 'Vertex', transform=ax.transAxes,
                        fontsize=7, color=VERTEX, va='top')
        else:
            ax.set_yticklabels([])

    # delta PPC ------------------------------------------------------------
    e = pd.read_csv(DATA / f'ppc_delta_by_stake.{label}.tsv', sep='\t')
    gs = fig.add_gridspec(1, 2, left=.505, right=.765, wspace=.12, top=top,
                          bottom=bot)
    elo = min(e.lo.min(), (e.obs - e.obs_sem).min()) - .01
    ehi = max(e.hi.max(), (e.obs + e.obs_sem).max()) + .01
    for o, order in enumerate(ORDERS):
        ax = fig.add_subplot(gs[0, o])
        s = e[e.order == order].sort_values('stake')
        ax.fill_between(x, s.lo, s.hi, color='.85', lw=0)
        ax.plot(x, s['median'], color='.35', lw=1.4)
        ax.errorbar(x, s.obs, yerr=s.obs_sem, fmt='o', color='.1', ms=3.4,
                    elinewidth=.8, capsize=0, zorder=4)
        ax.axhline(0, color='.8', lw=.5, ls='--', zorder=0)
        ax.set_ylim(elo, ehi)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' for v in stakes])
        ax.set_title(order, fontsize=7, color='.3', pad=2, style='italic')
        if m == 1:
            ax.set_xlabel('Stake (CHF)')
        if o == 0:
            ax.set_ylabel('Δ P(risky)\nIPS − vertex', fontsize=7.5)
        else:
            ax.set_yticklabels([])

    # noise channels -------------------------------------------------------
    df = get_data('/data/ds-tmsrisk', model_label=label)
    model = build_model(label, df)
    subj = pd.read_csv(DATA / dfile, sep='\t')
    subs = sorted(subj['subject'].unique())

    def s_get(sub, var, reg):
        q = subj.query('subject == @sub and var == @var '
                       'and regressor == @reg')
        return q.sort_values('draw')['value'].values

    def curves(noise, vertex):
        spl = sorted(v for v in subj['var'].unique()
                     if v.startswith(noise + '_spline')
                     and not v.endswith('_offset'))
        bas = np.asarray(model.make_dm(N_GRID, variable=noise))[:, :len(spl)]
        regs = ['Intercept'] + ([VC] if vertex else [])
        coef = np.stack([np.stack(
            [sum(s_get(sub, v, rg) for rg in regs) for v in spl], 1)
            for sub in subs], 0)
        return softplus(np.einsum('sdj,gj->sdg', coef, bas)).mean(0)

    gs = fig.add_gridspec(1, 1, left=.815, right=.985, top=top, bottom=bot)
    ax = fig.add_subplot(gs[0, 0])
    for noise, ls in [('memory_noise_sd', '-'), ('perceptual_noise_sd', '--')]:
        for vertex, colr in [(True, VERTEX), (False, IPS)]:
            y = curves(noise, vertex)
            med = np.median(y, 0)
            h = np.array([az.hdi(y[:, i], hdi_prob=.95)
                          for i in range(y.shape[1])])
            ax.fill_between(N_GRID, h[:, 0], h[:, 1], color=colr, alpha=.12,
                            lw=0)
            ax.plot(N_GRID, med, color=colr, ls=ls, lw=1.2)
    ax.set_xscale('log')
    ax.set_xticks([7, 30, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_ylim(0, 1.45)
    if m == 0:
        ax.text(8, 1.22, 'Memory', fontsize=6.5, color='.25')
        ax.text(8, 0.03, 'Perceptual', fontsize=6.5, color='.25')
    if m == 1:
        ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Noise SD (log units)', fontsize=7.5)

sns.despine(fig=fig, offset=3)
fig.savefig(OUT / 'best_models_card.pdf')
fig.savefig(OUT / 'best_models_card.png', dpi=150)
print('wrote', OUT / 'best_models_card.pdf')
