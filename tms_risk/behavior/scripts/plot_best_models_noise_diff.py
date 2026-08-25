"""cTBS noise-difference curves (IPS − vertex, log units) for the two
converged Δ-PPC-passing models, memory and perceptual channels. Rows =
models, columns = channels; median + 50%/95% HDI over draws of the
subject-averaged difference curve."""
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
BOLD = dict(fontweight='bold', fontfamily='Arial')
VC = 'stimulation_condition[T.vertex]'
PURPLE = '#8172B2'
softplus = lambda x: np.logaddexp(0, x)
N_GRID = np.exp(np.linspace(np.log(7 + 1e-6), np.log(112 - 1e-6), 80))

from tms_risk.behavior.fit_model import build_model, get_data

MODELS = [('lfx2-bs3-m2-dp-bm', 'm2bm_subject_draws.tsv.gz',
           'Linear memory (bs3-m2-bm)'),
          ('lfx2-bs2-m3-dp-bm', 'bs2m3bm_subject_draws.tsv.gz',
           'Quadratic memory (bs2-m3-bm)')]
CHANNELS = [('memory_noise_sd', 'Memory noise'),
            ('perceptual_noise_sd', 'Perceptual noise')]

fig, axes = plt.subplots(2, 2, figsize=(5.6, 4.6), sharex=True,
                         sharey='row')
diffs = {}
for m, (label, dfile, name) in enumerate(MODELS):
    df = get_data('/data/ds-tmsrisk', model_label=label)
    model = build_model(label, df)
    subj = pd.read_csv(DATA / dfile, sep='\t')
    subs = sorted(subj['subject'].unique())

    def s_get(sub, var, reg):
        q = subj.query('subject == @sub and var == @var '
                       'and regressor == @reg')
        return q.sort_values('draw')['value'].values

    for c, (noise, cname) in enumerate(CHANNELS):
        spl = sorted(v for v in subj['var'].unique()
                     if v.startswith(noise + '_spline')
                     and not v.endswith('_offset'))
        bas = np.asarray(model.make_dm(N_GRID, variable=noise))[:, :len(spl)]
        cur = {}
        for key, regs in [('ips', ['Intercept']),
                          ('vertex', ['Intercept', VC])]:
            coef = np.stack([np.stack(
                [sum(s_get(sub, v, rg) for rg in regs) for v in spl], 1)
                for sub in subs], 0)
            cur[key] = softplus(np.einsum('sdj,gj->sdg', coef, bas)).mean(0)
        diffs[(m, c)] = cur['ips'] - cur['vertex']

for m, (label, dfile, name) in enumerate(MODELS):
    for c, (noise, cname) in enumerate(CHANNELS):
        ax = axes[m, c]
        rel = diffs[(m, c)]
        med = np.median(rel, 0)
        h95 = np.array([az.hdi(rel[:, i], hdi_prob=.95)
                        for i in range(rel.shape[1])])
        h50 = np.array([az.hdi(rel[:, i], hdi_prob=.50)
                        for i in range(rel.shape[1])])
        ax.fill_between(N_GRID, h95[:, 0], h95[:, 1], color=PURPLE,
                        alpha=.14, lw=0)
        ax.fill_between(N_GRID, h50[:, 0], h50[:, 1], color=PURPLE,
                        alpha=.30, lw=0)
        ax.plot(N_GRID, med, color=PURPLE)
        ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
        ax.set_xscale('log')
        ax.set_xticks([7, 15, 30, 60, 112])
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.xaxis.set_minor_locator(mticker.NullLocator())
        if m == 0:
            ax.set_title(cname, fontsize=8.5, **BOLD)
        if m == 1:
            ax.set_xlabel('Payoff (CHF)')
        if c == 0:
            ax.set_ylabel(f'{name}\nΔ noise SD, IPS − vertex (log units)',
                          fontsize=7.5)
        p_pos = float((rel.mean(1) > 0).mean())
        ax.text(.03, .97, f'P(mean Δ > 0) = {p_pos:.2f}',
                transform=ax.transAxes, fontsize=6.5, va='top', color='.25')

sns.despine(fig=fig, offset=4)
fig.tight_layout()
fig.savefig(OUT / 'best_models_noise_diff.pdf')
fig.savefig(OUT / 'best_models_noise_diff.png', dpi=150)
print('wrote', OUT / 'best_models_noise_diff.pdf')
