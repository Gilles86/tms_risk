"""The choice PPC, re-expressed as the two things a psychometric function IS.

P(chose risky) against safe payoff answers "how often", but it mixes two effects
that the model separates: how STEEP the psychometric function is (consistency,
which is what noise controls) and WHERE it sits (the risk-neutral probability,
which is what the prior and the noise asymmetry control). Plotting those two
directly is the same posterior predictive check on a scale where each cTBS
effect has one place to land.

Top row     probit slope: 1 / (the width of the choice function in log-ratio
            units). Higher = more consistent.
Bottom row  risk-neutral probability at indifference. Higher = more risk averse
            (a larger risky payoff is needed to match the safe one).

Model: posterior mean and 95% credible band, derived in closed form from the
PMC parameters by `extract_anchor_probit --by n_safe` -- no simulated choices,
so no added sampling noise. Observed: one probit per pooled design cell with a
subject-cluster bootstrap interval, from `fit_observed_probit_cells`.

    python -m tms_risk.behavior.scripts.plot_probit_ppc_anchor \
        --model_label log-power-n1n2
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

IPS, VERTEX = '#d62728', '#2ca02c'
READ = dict(sep='\t', keep_default_na=False, na_values=[''])

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

PARAMS = [('slope', 'Probit slope\n(choice consistency)'),
          ('rnp', 'Risk-neutral probability\nat indifference')]
ORDERS = ['Risky first', 'Risky second']


def panel_letter(ax, s, dx=-.30, dy=1.02):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='left', family='Arial')


def main(data_dir, out_stem, label, derived_tsv, observed_tsv):
    dd = Path(data_dir)
    mod = pd.read_csv(derived_tsv or dd / 'probit_safe' /
                      f'probit_derived.{label}.tsv', **READ)
    mod = mod[(mod.by == 'n_safe') & (mod.label == label)].copy()
    # rnp = exp(-a/b) is heavy-tailed: a few draws with a small w_R put the
    # posterior MEAN in the thousands while the median sits near 1. Summarise
    # every quantity by the median and the 95% interval of the draws.
    D = mod.draws.map(lambda s_: np.fromstring(s_, sep=','))
    mod['mid'] = D.map(np.median)
    mod['lo'] = D.map(lambda v: np.percentile(v, 2.5))
    mod['hi'] = D.map(lambda v: np.percentile(v, 97.5))
    obs = pd.read_csv(observed_tsv or dd / 'probit_observed_hier.n_safe.tsv',
                      **READ)

    fig, axes = plt.subplots(2, 2, figsize=(5.4, 4.3), sharex=True,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=.02, h_pad=.02, wspace=.03, hspace=.04)

    xs = np.sort(obs.n_safe.unique())
    for r, (par, ylab) in enumerate(PARAMS):
        rowaxes = []
        for c, order in enumerate(ORDERS):
            ax = axes[r, c]
            rowaxes.append(ax)
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                m = mod[(mod.parameter == par) & (mod.order == order)
                        & (mod.stimulation_condition == stim)
                        ].sort_values('cell')
                if len(m):
                    ax.fill_between(m.cell, m.lo, m.hi, color=col, alpha=.18,
                                    lw=0, zorder=1)
                    ax.plot(m.cell, m['mid'], color=col, lw=1.2, zorder=2)
                o = obs[(obs.order == order)
                        & (obs.stimulation_condition == stim)].sort_values('n_safe')
                # the bootstrap interval is not symmetric, so draw it as such
                yerr = np.vstack([o[par] - o[f'{par}_lo'], o[f'{par}_hi'] - o[par]])
                yerr = np.clip(yerr, 0, None)
                ax.errorbar(o.n_safe, o[par], yerr=yerr, fmt='o', ms=3.6,
                            color=col, lw=0, elinewidth=.9, capsize=0, zorder=4)
            ax.set_xscale('log')
            ax.set_xticks(xs)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            ax.minorticks_off()
            if r == 0:
                ax.set_title(order, fontsize=8, color='0.15', pad=4)
            else:
                ax.set_xlabel('Safe payoff (CHF)')
            if c == 0:
                ax.set_ylabel(ylab)
        # limits from the point estimates and observed intervals only, so one
        # fat model tail cannot squash the whole row
        vals = np.concatenate([
            obs[[par, f'{par}_lo', f'{par}_hi']].values.ravel(),
            mod.loc[mod.parameter == par, 'mid'].values])
        vals = vals[np.isfinite(vals)]
        lo, hi = vals.min(), vals.max()
        pad = .06 * (hi - lo)
        for a in rowaxes:
            a.set_ylim(lo - pad, hi + pad)
            if par == 'rnp':
                a.axhline(1, color='0.75', lw=.6, ls=':', zorder=0)
        rowaxes[1].set_yticklabels([])

    a0 = axes[0, 0]
    a0.text(.03, .06, 'IPS', color=IPS, transform=a0.transAxes, fontsize=7)
    a0.text(.03, .19, 'Vertex', color=VERTEX, transform=a0.transAxes, fontsize=7)
    axes[1, 1].text(.97, .95, 'Points: observed probit fits, 95% bootstrap\n'
                              'Bands: model posterior, 95% CI',
                    transform=axes[1, 1].transAxes, ha='right', va='top',
                    fontsize=5.8, color='0.45', linespacing=1.5)
    panel_letter(axes[0, 0], 'a')
    panel_letter(axes[1, 0], 'b')
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')

    # what the figure says, in numbers
    for par, _ in PARAMS:
        w = obs.pivot_table(index=['order', 'n_safe'],
                            columns='stimulation_condition', values=par)
        print(f'\nobserved {par}, IPS - vertex:')
        print((w['ips'] - w['vertex']).round(3).to_string())


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--derived_tsv', default=None)
    ap.add_argument('--observed_tsv', default=None)
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    stem = a.out_stem or str(REPO / f'notes/figures/probit_ppc_{a.model_label}')
    main(a.data_dir, stem, a.model_label, a.derived_tsv, a.observed_tsv)
