"""Where cTBS acts: first-presented option vs second, power-law noise, n1/n2.

Two models, identical except for which option carries the cTBS coefficient.
Both are log-space observers with power-law noise (sigma = c * payoff**beta) and
the n1/n2 parameterization, so the placement question is asked in the paradigm's
own terms rather than as memory-vs-perceptual.

Left column: the noise functions by stimulation. Right column: the cTBS
difference, IPS - vertex, with the untouched option shown flat at zero as the
built-in reference it is.

    python -m tms_risk.behavior.scripts.plot_ctbs_placement
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

IPS, VERTEX = '#d62728', '#2ca02c'
MODELS = [('lfx2-pl-m2-dp-m-p2-i', 'cTBS on the FIRST-presented option (remembered)',
           'ELPD −4199.6 · r̂ 1.003'),
          ('lfx2-pl-m2-dp-b-p2-i', 'cTBS on the SECOND-presented option (seen)',
           'ELPD −4218.5 · r̂ 1.429 ⚠')]
STYLE = {'n1 (first)': '-', 'n2 (second)': '--'}


def logx(ax):
    ax.set_xscale('log'); ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='notes/data/cards')
    ap.add_argument('--out', default='notes/figures/ctbs_placement')
    a = ap.parse_args()

    fig, axes = plt.subplots(2, 2, figsize=(6.6, 4.9), constrained_layout=True,
                             sharex=True)
    for row, (label, title, meta) in enumerate(MODELS):
        c = pd.read_csv(Path(a.data_dir) / f'curves.{label}.tsv', sep='\t')

        ax = axes[row, 0]
        for curve, ls in STYLE.items():
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                s = c[(c.curve == curve) & (c.stim == stim)].sort_values('payoff')
                ax.fill_between(s.payoff, s.lo, s.hi, color=col, alpha=.13, lw=0)
                ax.plot(s.payoff, s['median'], color=col, lw=1.5, ls=ls)
            s = c[(c.curve == curve) & (c.stim == 'vertex')].sort_values('payoff')
            ax.text(116, s['median'].iloc[-1], curve.split()[0], fontsize=6.3,
                    color='.2', va='center')
        ax.set_yscale('log'); ax.set_yticks([0.1, 0.2, 0.4])
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
        ax.yaxis.set_minor_locator(mticker.NullLocator())
        ax.set_ylabel('Noise SD (log units)')
        logx(ax)
        ax.set_title(f'{title}\n{meta}', fontsize=7.5, linespacing=1.35)

        ax = axes[row, 1]
        ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
        for curve, ls in STYLE.items():
            i = c[(c.curve == curve) & (c.stim == 'ips')].sort_values('payoff')
            v = c[(c.curve == curve) & (c.stim == 'vertex')].sort_values('payoff')
            d = i['median'].values - v['median'].values
            ax.plot(i.payoff, d, color='.15', lw=1.7, ls=ls)
            lab = curve.split()[0] + ('' if abs(d).max() > 1e-6 else '  (no cTBS term)')
            ax.text(116, d[-1], lab, fontsize=6.3, color='.2', va='center')
        logx(ax)
        ax.set_ylabel('Δ noise SD, IPS − vertex')
        ax.set_title('cTBS effect', fontsize=7.5)
        ax.set_ylim(-.055, .055)

    sns.despine(fig=fig, offset=3)
    axes[0, 0].text(.03, .05, 'IPS', transform=axes[0, 0].transAxes, color=IPS,
                    fontsize=7, va='bottom')
    axes[0, 0].text(.03, .16, 'Vertex', transform=axes[0, 0].transAxes,
                    color=VERTEX, fontsize=7, va='bottom')
    for ax, letter in zip(axes.ravel(), 'abcd'):
        ax.text(-0.20, 1.16, letter, transform=ax.transAxes, fontsize=8,
                family='Arial', fontweight='bold', va='bottom', ha='left')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', bbox_inches='tight', pad_inches=0.02)

    for label, title, _ in MODELS:
        c = pd.read_csv(Path(a.data_dir) / f'curves.{label}.tsv', sep='\t')
        print(f'\n{title}')
        for curve in STYLE:
            for x in (7.0, 112.0):
                s = c[(c.curve == curve) & (np.isclose(c.payoff, c.payoff[
                    (c.payoff - x).abs().idxmin()]))]
                i = float(s[s.stim == 'ips']['median'].iloc[0])
                v = float(s[s.stim == 'vertex']['median'].iloc[0])
                print(f'  {curve:12s} {x:6.0f} CHF  vertex {v:.4f}  ips {i:.4f}  '
                      f'Δ {i - v:+.4f}')
    print(f'\nwrote {a.out}.pdf')


if __name__ == '__main__':
    main()
