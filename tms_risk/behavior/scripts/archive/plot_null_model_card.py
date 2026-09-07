"""One page on the affine null model: parameters, noise curves, and PPCs.

The null has no cTBS term, so it is the test of whether the ARCHITECTURE (log-
space Bayesian observer, affine-in-log noise on both channels, free lognormal
priors) reproduces the paradigm at all. It should fit everything except the
stimulation contrast -- and that is exactly what it does.

Reads notes/data/{params,curves}.<label>.tsv and notes/data/ppc_*.<label>.tsv.

    python -m tms_risk.behavior.scripts.plot_null_model_card
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.3, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

C_MOD, C_OBS = '.35', '#C44E52'
IPS, VERTEX = '#d62728', '#2ca02c'
CURVE_COL = {'memory': '#C44E52', 'perceptual': '#3B5BA5',
             'n1 (first)': '.15', 'n2 (second)': '.55'}


def logx(ax, ticks=(7, 14, 28, 56, 112)):
    ax.set_xscale('log'); ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def band(ax, t, x, col=C_MOD):
    ax.fill_between(x, t.lo, t.hi, color=col, alpha=.22, lw=0, zorder=1)
    ax.plot(x, t['median'], color=col, lw=1.5, zorder=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--label', default='lfx2-bs3-m2-dp-null-p2')
    ap.add_argument('--out', default='notes/figures/null_model_card')
    a = ap.parse_args()
    D, L = Path('notes/data'), a.label

    pars = pd.read_csv(D / f'params.{L}.tsv', sep='\t')
    curves = pd.read_csv(D / f'curves.{L}.tsv', sep='\t')

    fig = plt.figure(figsize=(7.4, 4.9))
    gs = fig.add_gridspec(2, 3, hspace=.62, wspace=.42, left=.09, right=.98,
                          top=.90, bottom=.10)

    # -- a: group parameters ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    p = pars.sort_values('mean').reset_index(drop=True)
    y = np.arange(len(p))
    ax.hlines(y, p.lo, p.hi, color='.4', lw=1.0)
    ax.plot(p['mean'], y, 'o', ms=4, color='.1')
    ax.axvline(0, color='.75', lw=.7, ls='--', zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([s.replace('_noise_sd', '').replace('_', ' ')
                        for s in p.param], fontsize=6.0)
    ax.set_xlabel('Group mean (model units)')
    ax.set_title('Parameters', fontsize=8)

    # -- b: noise curves ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    for name in ['n1 (first)', 'n2 (second)']:
        c = curves[curves.curve == name].sort_values('payoff')
        band(ax, c, c.payoff, CURVE_COL[name])
        ax.text(115, c['median'].iloc[-1], name.split()[0], fontsize=6.3,
                color=CURVE_COL[name], va='center', ha='left')
    logx(ax); ax.set_yscale('log')
    ax.set_yticks([0.1, 0.2, 0.4])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)'); ax.set_ylabel('Noise SD (log units)')
    ax.set_title('Noise per option', fontsize=8)

    # -- c: the two channels ------------------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for name in ['memory', 'perceptual']:
        c = curves[curves.curve == name].sort_values('payoff')
        band(ax, c, c.payoff, CURVE_COL[name])
        ax.text(115, c['median'].iloc[-1], name, fontsize=6.3,
                color=CURVE_COL[name], va='center', ha='left')
    logx(ax); ax.set_yscale('log')
    ax.set_yticks([0.1, 0.3, 1.0])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)'); ax.set_ylabel('Noise SD (log units)')
    ax.set_title('Channels (coordinates)', fontsize=8)

    # -- d: psychometric PPC ------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    t = pd.read_csv(D / f'ppc_by_ratio.{L}.tsv', sep='\t')
    b = pd.read_csv(D / f'bins_ratio_bin.{L}.tsv', sep='\t').set_index('ratio_bin')['x']
    x = b.reindex(t.ratio_bin).values
    band(ax, t, x)
    ax.plot(x, t.observed, 'o', ms=4, color=C_OBS, zorder=4)
    ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
    ax.set_title('Psychometric  6/6', fontsize=8)

    # -- e: by stake x order -------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    t = pd.read_csv(D / f'ppc_by_stake.{L}.tsv', sep='\t')
    b = pd.read_csv(D / f'bins_stake_bin.{L}.tsv', sep='\t').set_index('stake_bin')['x']
    for order, ls, mk in [('Risky first', '--', 'o'), ('Risky second', '-', 's')]:
        s = t[t.order == order].sort_values('stake_bin')
        x = b.reindex(s.stake_bin).values
        ax.fill_between(x, s.lo, s.hi, color=C_MOD, alpha=.18, lw=0)
        ax.plot(x, s['median'], color=C_MOD, lw=1.4, ls=ls)
        ax.plot(x, s.observed, mk, ms=3.6, color=C_OBS,
                mfc=C_OBS if order == 'Risky second' else 'white', mew=1.0)
    logx(ax, ticks=(10, 20, 40))
    ax.set_xlabel('Stake (CHF)'); ax.set_ylabel('P(chose risky)')
    ax.set_title('By stake and order  10/10', fontsize=8)
    ax.text(.03, .95, 'Filled: risky second\nOpen: risky first',
            transform=ax.transAxes, fontsize=5.9, va='top', color='.3',
            linespacing=1.3)

    # -- f: the cTBS contrast, where a null must fail ------------------------
    ax = fig.add_subplot(gs[1, 2])
    t = pd.read_csv(D / f'ppc_by_stim.{L}.tsv', sep='\t')
    w = t.pivot(index='order', columns='stimulation_condition',
                values=['median', 'lo', 'hi', 'observed'])
    y = np.arange(len(w))
    for k, (stim, col) in enumerate([('vertex', VERTEX), ('ips', IPS)]):
        off = (k - .5) * .22
        ax.hlines(y + off, w[('lo', stim)], w[('hi', stim)], color=col, lw=1.2)
        ax.plot(w[('median', stim)], y + off, 'o', ms=3.5, color=col, mfc='white',
                mew=1.1)
        ax.plot(w[('observed', stim)], y + off, 'o', ms=4.5, color=col)
    ax.set_yticks(y); ax.set_yticklabels(w.index, fontsize=7)
    ax.set_xlabel('P(chose risky)')
    ax.set_title('cTBS contrast  2/4  ✗', fontsize=8, color='#b0453b')
    ax.text(.03, .06, 'Open: model   Filled: observed', transform=ax.transAxes,
            fontsize=5.9, color='.3')

    sns.despine(fig=fig, offset=3)
    for ax_, letter in zip(fig.axes, 'abcdef'):
        ax_.text(-0.26, 1.14, letter, transform=ax_.transAxes, fontsize=8,
                 family='Arial', fontweight='bold', va='bottom', ha='left')
    fig.text(.5, .975, f'Null model (no cTBS term): {L}', ha='center',
             fontsize=8.5, color='.1')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', bbox_inches='tight', pad_inches=0.02)
    print(pars.round(3).to_string(index=False))
    print(f'wrote {a.out}.pdf')


if __name__ == '__main__':
    main()
