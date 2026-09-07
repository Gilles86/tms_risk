"""The choice PPC on published-Figure-3 axes: probit slope and risk-neutral probability.

P(chose risky) answers "how often", but it confounds the two things a
psychometric function actually has: how STEEP it is (choice consistency, which
is what noise controls) and WHERE it sits (the risk-neutral probability, which
the prior and the noise asymmetry control). Those are the two quantities the
published Figure 3 reports, so this is the posterior predictive check on the
scale the paper already uses.

Cells are order x stake median split -- the resolution the data support. A
per-subject probit needs ~60 trials; splitting by safe payoff instead leaves 12
per cell and fits none of them (`fit_observed_probit_subject --by n_safe`
returns zero), and pooling subjects to rescue it flattens the slope by mixing
different indifference points.

Observed: one probit per participant per cell (`fit_observed_probit_subject`).
Model: the same two quantities derived in closed form from the PMC parameters,
per participant (`extract_anchor_probit`, subject-level output) -- no simulated
choices, so no added sampling noise. Both sides are then averaged over
participants, which is what makes the comparison fair.

The two scripts parameterise the indifference point from opposite sides:
`extract_anchor_probit` reports p_R * frac*, the EV ratio at indifference, while
the observed fit reports 1/frac*, the risk-neutral probability itself. They are
the same number, related by RNP = p_R / (p_R frac*); the model side is converted
here so both axes read as a probability.

    python -m tms_risk.behavior.scripts.plot_probit_ppc_stake \
        --model_label log-power-n1n2
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

IPS, VERTEX = '#d62728', '#2ca02c'
READ = dict(sep='\t', keep_default_na=False, na_values=[''])
P_RISKY = 0.55

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

ORDERS = ['Risky first', 'Risky second']
PARAMS = [('slope', 'Probit slope\n(choice consistency)'),
          ('rnp', 'Risk-neutral probability\nat indifference')]
STAKE = {0: 'Low', 1: 'High'}


def load(data_dir, label, observed_tsv):
    dd = Path(data_dir)
    m = pd.read_csv(dd / 'probit_derived' / f'probit_subject.{label}.tsv', **READ)
    m = m[m.parameter.isin(['slope', 'rnp'])].copy()
    m = m.pivot_table(index=['subject', 'order', 'stake2',
                             'stimulation_condition'],
                      columns='parameter', values='median').reset_index()
    m['rnp'] = P_RISKY / m['rnp']          # EV ratio -> probability
    o = pd.read_csv(observed_tsv or dd / 'probit_observed_subject.tsv', **READ)
    return m, o


def agg(df, par):
    g = df.groupby(['order', 'stake2', 'stimulation_condition'])[par]
    return pd.DataFrame({'mean': g.mean(), 'sem': g.sem(), 'n': g.size()})


def main(data_dir, out_stem, label, observed_tsv):
    mod, obs = load(data_dir, label, observed_tsv)

    fig, axes = plt.subplots(2, 2, figsize=(4.9, 4.4), sharex=True,
                             constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=.02, h_pad=.02, wspace=.03, hspace=.05)
    xs = np.array([0., 1.])
    dx = 0.055

    for r, (par, ylab) in enumerate(PARAMS):
        A = agg(mod, par), agg(obs, par)
        rowaxes = []
        for c, order in enumerate(ORDERS):
            ax = axes[r, c]
            rowaxes.append(ax)
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                off = -dx if stim == 'vertex' else dx
                # model: line, the prediction
                m = A[0].loc[(order, slice(None), stim)]
                ax.plot(xs + off, m['mean'].values, color=col, lw=1.2, zorder=2)
                ax.fill_between(xs + off, m['mean'] - m['sem'],
                                m['mean'] + m['sem'], color=col, alpha=.20,
                                lw=0, zorder=1)
                # observed: points
                o = A[1].loc[(order, slice(None), stim)]
                ax.errorbar(xs + off, o['mean'].values, yerr=o['sem'].values,
                            fmt='o', ms=4, color=col, lw=0, elinewidth=.9,
                            capsize=0, zorder=4)
            ax.set_xticks(xs)
            ax.set_xticklabels([STAKE[0], STAKE[1]])
            ax.set_xlim(-.42, 1.42)
            if r == 0:
                ax.set_title(order, fontsize=8, color='0.15', pad=4)
            else:
                ax.set_xlabel('Stake')
            if c == 0:
                ax.set_ylabel(ylab)
        lo = min(a.get_ylim()[0] for a in rowaxes)
        hi = max(a.get_ylim()[1] for a in rowaxes)
        for a in rowaxes:
            a.set_ylim(lo, hi)
            if par == 'rnp':
                a.axhline(P_RISKY, color='0.75', lw=.6, ls=':', zorder=0)
        rowaxes[1].set_yticklabels([])
    axes[1, 0].text(1.40, P_RISKY, ' Risk\n neutral', fontsize=5.8, color='0.55',
                    va='center', ha='left', linespacing=1.4)

    a0 = axes[0, 0]
    a0.text(.04, .12, 'IPS', color=IPS, transform=a0.transAxes, fontsize=7)
    a0.text(.04, .01, 'Vertex', color=VERTEX, transform=a0.transAxes, fontsize=7)
    axes[0, 1].text(.97, .04, 'Points: observed probits\nLines: model, ±1 SEM',
                    transform=axes[0, 1].transAxes, ha='right', va='bottom',
                    fontsize=5.8, color='0.45', linespacing=1.5)
    for ax, s in [(axes[0, 0], 'a'), (axes[1, 0], 'b')]:
        ax.text(-.34, 1.02, s, transform=ax.transAxes, fontsize=8,
                fontweight='bold', va='bottom', ha='left', family='Arial')
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')

    for par, _ in PARAMS:
        print(f'\n{par}: model vs observed, IPS - vertex (paired within subject)')
        for nm, df, in [('model', mod), ('obs', obs)]:
            w = df.pivot_table(index=['subject', 'order', 'stake2'],
                               columns='stimulation_condition', values=par)
            d = (w['ips'] - w['vertex']).dropna()
            print(f'  {nm:5s} ' + '  '.join(
                f'{o[:7]}/{STAKE[s].lower():4s} {v:+.3f}'
                for (o, s), v in d.groupby(['order', 'stake2']).mean().items()))


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--observed_tsv', default=None)
    ap.add_argument('--out_stem', default=None)
    a = ap.parse_args()
    stem = a.out_stem or str(REPO / f'notes/figures/probit_ppc_stake_{a.model_label}')
    main(a.data_dir, stem, a.model_label, a.observed_tsv)
