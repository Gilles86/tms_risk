"""PROTOTYPE. One knob and one lever: w, and the distance from the prior.

The compression is not a free-form curve. It is a Bayesian posterior mean, so the
percept of a payoff n is exactly

    percept = w * n + (1 - w) * mu,        w = sigma^2 / (sigma^2 + nu^2)

and therefore the whole effect of cTBS on an option is a product of two terms:

    d(percept) = dw * (n - mu)             knob * lever
    d(percept)/percept = dw * (n - mu) / percept

w is recovered exactly, not refitted: it is (percept - mu) / (n - mu), inverted
from the stored percepts and the stored group prior means. The panels are

    a   w against payoff, per option, vertex vs cTBS. The knob.
        (Panel c is exact as a fraction of the vertex percept; panel d, being a
        difference of two such fractions, is first-order in dw.)
    b   the lever (n - mu) / percept: how much a given loss of w costs, in % per
        unit of w.
    c   their product, which is the proportional loss of perceived value, and the
        difference between the two curves in c is the ratio shift.

The point of splitting it: at low stakes the safe option loses more because the
knob turns further for it (it is the first-presented, memory-loaded option); at
high stakes the risky option's lever is actually longer, and it is only the knob
that keeps the net effect pointing the same way.

    python -m tms_risk.behavior.scripts.proto_shrink_weight

Reads notes/data/pmc_percepts_by_order.<label>.tsv and pmcpars_priors.<label>.tsv.
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

VERTEX, IPS = '#2ca02c', '#d62728'
SAFE, RISKY = '#4d4d4d', '#b2182b'
DIFF = '#1a1a1a'
P_RISKY = 0.55
LEVELS = [7, 10, 14, 20, 28]
ORDER = 'Risky second'   # the order in which the behavioural effect lives


def style():
    mpl.rcParams.update({
        'font.family': 'Helvetica',
        'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
        'font.size': 8, 'axes.labelsize': 8.5, 'axes.titlesize': 8.5,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
        'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelpad': 3,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .8, 'ytick.major.width': .8,
        'lines.linewidth': 1.3, 'lines.markersize': 4,
        'legend.frameon': False,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'figure.dpi': 150, 'savefig.dpi': 300,
    })
    sns.set_context('paper')


PANEL = dict(fontsize=10, fontweight='bold', va='bottom', ha='right')


def invert_w(d, prior):
    """Recover the posterior weight each stored percept implies.

    For the risky option the stored quantity is p * percept, so it is divided by p
    before inverting; the payoff itself is recovered the same way. No fitting.
    """
    out = []
    for (order, ns), sub in d.groupby(['order', 'n_safe']):
        row = {'order': order, 'n_safe': ns}
        for opt in ['safe', 'risky']:
            r = sub[sub.option == opt].iloc[0]
            p = 1.0 if opt == 'safe' else P_RISKY
            n = r.objective_ev / p
            mu = prior[opt]
            for cond, key in [('v', 'vertex'), ('i', 'ips')]:
                row[f'{opt}_perc_{cond}'] = r[key]
                row[f'{opt}_w_{cond}'] = (r[key] / p - mu) / (n - mu)
            row[f'{opt}_n'] = n
            row[f'{opt}_dw'] = row[f'{opt}_w_i'] - row[f'{opt}_w_v']
            # lever: % of perceived value lost per unit of w, at the vertex percept
            row[f'{opt}_lever'] = 100 * p * (n - mu) / row[f'{opt}_perc_v']
            row[f'{opt}_pct'] = row[f'{opt}_dw'] * row[f'{opt}_lever']
        row['ratio_pct'] = row['risky_pct'] - row['safe_pct']
        out.append(row)
    return pd.DataFrame(out).sort_values(['order', 'n_safe'])


def main(data_dir, label, out_stem):
    style()
    data = Path(data_dir)
    d = pd.read_csv(data / f'pmc_percepts_by_order.{label}.tsv', sep='\t')
    pri = pd.read_csv(data / f'pmcpars_priors.{label}.tsv', sep='\t')
    pri = pri[pri.level == 'group'].set_index('parameter')['mean']
    prior = {'safe': float(pri.safe_prior_mu), 'risky': float(pri.risky_prior_mu)}

    w = invert_w(d, prior)
    s = w[w.order == ORDER]
    x = s.n_safe.values

    fig = plt.figure(figsize=(7.25, 2.35))
    gs = fig.add_gridspec(1, 4, wspace=.46, left=.075, right=.99,
                          top=.80, bottom=.20)
    a, b, c, e = [fig.add_subplot(gs[0, i]) for i in range(4)]

    # ---- a: the knob ---------------------------------------------------------
    for opt, col in [('safe', SAFE), ('risky', RISKY)]:
        a.plot(x, s[f'{opt}_w_v'], '-o', color=col, mfc=col, zorder=3)
        a.plot(x, s[f'{opt}_w_i'], '-o', color=col, mfc='white', mec=col,
               mew=1.0, lw=1.0, zorder=2)
        a.vlines(x, s[f'{opt}_w_i'], s[f'{opt}_w_v'], color=col, lw=2.2,
                 alpha=.35, zorder=1)
    a.set_ylim(0.08, 0.82)
    a.set_yticks([0.1, 0.3, 0.5, 0.7])
    a.set_ylabel('Weight on evidence, w')
    a.text(x[0] * .98, s.safe_w_v.iloc[0] * 1.05, 'Safe', color=SAFE,
           fontsize=7.5, ha='left', va='bottom')
    a.text(x[3], s.risky_w_v.iloc[3] * .72, 'Risky', color=RISKY,
           fontsize=7.5, ha='center', va='top')
    a.annotate('cTBS', xy=(x[2], s.safe_w_i.iloc[2]), xytext=(x[3], .62),
               fontsize=7.5, color='0.3', ha='center', va='bottom',
               arrowprops=dict(arrowstyle='-', color='0.45', lw=.6,
                               connectionstyle='arc3,rad=.25'))

    # ---- b: the lever --------------------------------------------------------
    for opt, col in [('safe', SAFE), ('risky', RISKY)]:
        b.plot(x, s[f'{opt}_lever'], '-o', color=col, mfc=col, zorder=3)
    b.set_ylim(0, 380)
    b.set_yticks([0, 100, 200, 300])
    b.set_ylabel('Lever, % per unit w')

    # ---- c: knob x lever = the proportional loss ------------------------------
    for opt, col in [('safe', SAFE), ('risky', RISKY)]:
        c.plot(x, s[f'{opt}_pct'], '-o', color=col, mfc=col, zorder=3)
    c.vlines(x, s.safe_pct, s.risky_pct, color=DIFF, lw=2.0, zorder=4)
    c.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
    c.set_ylim(-8.6, 1.2)
    c.set_yticks([-8, -6, -4, -2, 0])
    c.set_ylabel('Δ Perceived EV (%)')

    # ---- d: what is left over, both orders -----------------------------------
    for order, col, lw in [('Risky first', '0.62', 1.1), (ORDER, DIFF, 1.6)]:
        t = w[w.order == order]
        e.plot(t.n_safe, t.ratio_pct, '-o', color=col, lw=lw, mfc=col, zorder=3)
    e.axhline(0, color='0.7', lw=.6, ls='--', zorder=0)
    e.set_ylim(-.8, 3.5)
    e.set_yticks([0, 1, 2, 3])
    e.set_ylabel('Δ Perceived ratio (%)')
    e.text(x[-1] * 1.03, w[w.order == ORDER].ratio_pct.iloc[-1] + .35,
           'Risky second', color=DIFF, fontsize=7.5, ha='right', va='bottom')
    e.text(x[1], -.15, 'Risky first', color='0.55', fontsize=7.5,
           ha='left', va='top')

    for ax, letter in zip([a, b, c, e], 'abcd'):
        ax.set_xscale('log')
        ax.set_xticks(LEVELS)
        ax.set_xticklabels([str(v) for v in LEVELS])
        ax.minorticks_off()
        ax.set_xlabel('Safe payoff (CHF)')
        ax.text(-.34, 1.03, letter, transform=ax.transAxes, **PANEL)

    fig.suptitle('cTBS turns one knob, w; the two options differ in how far it '
                 'turns and in what it costs them', fontsize=9, y=.98)
    sns.despine(fig=fig, offset=4, trim=False)
    for ax in [a, b, c, e]:
        ax.tick_params(which='minor', bottom=False, left=False)

    out = Path(out_stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out.with_suffix('.pdf'))
    print(f'wrote {out.with_suffix(".pdf")}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', default='notes/data')
    p.add_argument('--label', default='flexible2nf')
    p.add_argument('--out', default='notes/figures/prototypes/shrink_weight')
    a = p.parse_args()
    main(a.data_dir, a.label, a.out)
