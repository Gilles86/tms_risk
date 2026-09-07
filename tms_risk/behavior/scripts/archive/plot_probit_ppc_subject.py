"""Posterior predictive check on the PROBIT PARAMETERS, group and per subject.

Raw choice proportions spend most of their range on things every model gets
right. The two probit parameters are what the paper actually reports, so check
the model on those instead -- and check them one participant at a time, which
asks a question the group mean cannot: does the model know WHICH people show the
biggest cTBS effect?

a, b  Per subject: the model's derived cTBS effect against the participant's own
      probit fit, in the low-stake / risky-second cell. The identity line is the
      prediction; the correlation is whether individual differences are tracked
      at all, which is the weaker and more achievable claim.
c     Group level: the model's posterior for the mean effect against the
      observed hierarchical-probit posterior.

Per-subject probits rest on ~120 trials, so the observed values are noisy; that
attenuates any correlation and is a floor on how well ANY model could do here.

    python -m tms_risk.behavior.scripts.plot_probit_ppc_subject
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
MODEL, DATA = '#3B5BA5', '0.15'
CELL = ('Risky second', 0)

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def dr(s):
    return np.array([float(v) for v in s.split(',')])


def main(data_dir, out_stem, label):
    dd = Path(data_dir)
    sub = pd.read_csv(dd / f'probit_derived/probit_subject.{label}.tsv', **READ)
    obs = pd.read_csv(dd / 'probit_observed_subject.tsv', **READ)
    grp = pd.read_csv(dd / f'probit_derived/probit_derived.{label}.tsv', **READ)
    pub = pd.read_csv(dd / 'probit_stake_group_posterior.tsv', **READ)

    s = sub[(sub.order == CELL[0]) & (sub.stake2 == CELL[1])]
    m = s.pivot_table(index=['subject', 'parameter'],
                      columns='stimulation_condition', values='median')
    m['d'] = m['ips'] - m['vertex']
    mod = m['d'].unstack('parameter')

    o = obs[(obs.order == CELL[0]) & (obs.stake2 == CELL[1])]
    ow = o.pivot_table(index='subject', columns='stimulation_condition',
                       values=['slope', 'rnp'])
    dat = pd.DataFrame({p: ow[(p, 'ips')] - ow[(p, 'vertex')]
                        for p in ('slope', 'rnp')})

    j = mod.join(dat, lsuffix='_model', rsuffix='_data').dropna()

    fig, axes = plt.subplots(1, 3, figsize=(7.25, 2.5), constrained_layout=True)
    for k, par in enumerate(['slope', 'rnp']):
        ax = axes[k]
        x, y = j[f'{par}_model'].values, j[f'{par}_data'].values
        lim = np.array([min(x.min(), y.min()), max(x.max(), y.max())])
        pad = .08 * (lim[1] - lim[0])
        ax.plot(lim + [-pad, pad], lim + [-pad, pad], color='0.7', lw=.8, ls=':',
                zorder=1)
        ax.axhline(0, color='0.9', lw=.6, zorder=0)
        ax.axvline(0, color='0.9', lw=.6, zorder=0)
        ax.scatter(x, y, s=17, facecolor=MODEL, alpha=.6, edgecolor='white',
                   lw=.5, zorder=3)
        r, p = stats.pearsonr(x, y)
        ax.set_xlim(lim[0] - pad, lim[1] + pad)
        ax.set_ylim(lim[0] - pad, lim[1] + pad)
        ax.set_xlabel(f'Model Δ{par}')
        ax.set_ylabel(f'Observed Δ{par}')
        ax.set_title(f'{"ab"[k]}  Per subject · Δ{par}', loc='left', fontsize=8)
        ax.text(.04, .96, f'r = {r:.2f}, p = {p:.3f}\nn = {len(x)}',
                transform=ax.transAxes, va='top', fontsize=6.5, color='0.25')

    ax = axes[2]
    pv = pub.pivot_table(index=['parameter', 'order', 'stake', 'draw'],
                         columns='stimulation_condition', values='value')
    pv['d'] = pv['ips'] - pv['vertex']
    for k, par in enumerate(['slope', 'rnp']):
        g = grp[(grp.order == CELL[0]) & (grp.stake2 == CELL[1])]
        if par == 'slope':
            h = g[g.parameter == 'slope']
            i = dr(h[h.stimulation_condition == 'ips'].draws.iloc[0])
            v = dr(h[h.stimulation_condition == 'vertex'].draws.iloc[0])
            n = min(len(i), len(v))
            md = i[:n] - v[:n]
            scale = 1.0
        else:
            h = g[g.parameter == 'logfrac_star']
            i = dr(h[h.stimulation_condition == 'ips'].draws.iloc[0])
            v = dr(h[h.stimulation_condition == 'vertex'].draws.iloc[0])
            n = min(len(i), len(v))
            md = np.exp(-i[:n]) - np.exp(-v[:n])
            scale = 5.0                      # rnp is ~5x smaller; put on one axis
        od = pv.xs((par, CELL[0], 'Low stake'))['d'].values
        for arr, col, off in [(md * scale, MODEL, -.16), (od * scale, DATA, .16)]:
            lo, mid, hi = np.quantile(arr, [.025, .5, .975])
            ax.plot([lo, hi], [k + off] * 2, color=col, lw=1.4,
                    solid_capstyle='butt')
            ax.plot(mid, k + off, 'o', ms=4.5, color=col)
    ax.axvline(0, color='0.75', lw=.7, ls='--', zorder=0)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['Δ slope', 'Δ rnp\n(×5)'])
    ax.set_ylim(-.6, 1.6)
    ax.set_xlabel('cTBS effect')
    ax.set_title('c  Group level', loc='left', fontsize=8)
    ax.text(.04, .30, 'Model', color=MODEL, transform=ax.transAxes, fontsize=7)
    ax.text(.04, .18, 'Data', color=DATA, transform=ax.transAxes, fontsize=7)

    fig.suptitle(f'{label} · low stake, risky second · model values derived in '
                 f'closed form; observed from per-subject probit fits',
                 fontsize=6.6, color='0.4', y=1.06)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/probit_ppc_subject'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
