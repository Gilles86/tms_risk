"""The power-law family: where cTBS is allowed to act, and what it does.

Six models, identical except for which noise term carries the cTBS coefficient.
Top block is the memory/perceptual decomposition (perceptual feeds BOTH options,
memory only the first); bottom block is n1/n2, which asks the same question in
the paradigm's own terms.

Left of each pair: the noise functions by stimulation condition.
Right: the cTBS difference IPS - vertex, computed PER DRAW so the interval is the
interval of the difference -- the two conditions come from the same draws and are
strongly correlated, so differencing the marginal bands would badly overstate it.

    python -m tms_risk.behavior.scripts.plot_powerlaw_family
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
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})

IPS, VERTEX = '#d62728', '#2ca02c'
BLOCKS = [
    ('memory/perceptual', ['memory', 'perceptual'],
     [('lfx2-pl-m2-dp-m-p2', 'cTBS on memory'),
      ('lfx2-pl-m2-dp-b-p2', 'cTBS on perceptual'),
      ('lfx2-pl-m2-dp-bm-p2', 'cTBS on both channels')]),
    ('n1/n2', ['n1 (first)', 'n2 (second)'],
     [('lfx2-pl-m2-dp-m-p2-i', 'cTBS on 1st option'),
      ('lfx2-pl-m2-dp-b-p2-i', 'cTBS on 2nd option'),
      ('lfx2-pl-m2-dp-bm-p2-i', 'cTBS on both options')]),
]
LS = {'memory': '-', 'perceptual': '--', 'n1 (first)': '-', 'n2 (second)': '--'}


def logx(ax):
    ax.set_xscale('log'); ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default='notes/data/delta')
    ap.add_argument('--meta_dir', default='notes/data/ploo')
    ap.add_argument('--out', default='notes/figures/powerlaw_family')
    a = ap.parse_args()
    D = Path(a.data_dir)

    import glob, json
    meta = {}
    for f in glob.glob(f'{a.meta_dir}/*.npz'):
        m = json.loads(str(np.load(f)['meta']))
        meta[m['label']] = m

    fig, axes = plt.subplots(6, 2, figsize=(6.4, 11.4), constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1, 1]))
    row = 0
    for block, curves, models in BLOCKS:
        for label, title in models:
            axL, axR = axes[row], axes[row + 0]
            axL, axR = axes[row][0], axes[row][1]
            cf, df_ = D / f'curves.{label}.tsv', D / f'delta.{label}.tsv'
            if not cf.exists():
                for ax in (axL, axR):
                    ax.axis('off')
                    ax.text(.5, .5, f'{title}\n(not extracted yet)', ha='center',
                            va='center', fontsize=7, color='.5')
                row += 1
                continue
            c, d = pd.read_csv(cf, sep='\t'), pd.read_csv(df_, sep='\t')
            m = meta.get(label, {})

            for nm in curves:
                for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                    s = c[(c.curve == nm) & (c.stim == stim)].sort_values('payoff')
                    if not len(s):
                        continue
                    axL.fill_between(s.payoff, s.lo, s.hi, color=col, alpha=.13, lw=0)
                    axL.plot(s.payoff, s['median'], color=col, lw=1.5, ls=LS[nm])
                s = c[(c.curve == nm) & (c.stim == 'vertex')].sort_values('payoff')
                if len(s):
                    axL.text(118, s['median'].iloc[-1], nm.split()[0], fontsize=6.1,
                             color='.25', va='center')
            axL.set_yscale('log'); logx(axL)
            axL.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
            axL.yaxis.set_minor_locator(mticker.NullLocator())
            axL.set_ylabel('Noise SD (log units)')
            rh = m.get('max_rhat', np.nan)
            flag = '' if rh <= 1.01 else f'  ⚠ r̂ {rh:.2f}'
            axL.set_title(f'{title}\nELPD {m.get("elpd_loo", float("nan")):.1f}{flag}',
                          fontsize=7.5, linespacing=1.3,
                          color='.1' if rh <= 1.01 else '#b0453b')

            axR.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
            for nm in curves:
                s = d[d.curve == nm].sort_values('payoff')
                if not len(s) or s['median'].abs().max() < 1e-9:
                    if len(s):
                        axR.plot(s.payoff, s['median'], color='.6', lw=1.2, ls=LS[nm])
                        axR.text(118, 0, nm.split()[0] + ' (fixed)', fontsize=6.0,
                                 color='.55', va='center')
                    continue
                axR.fill_between(s.payoff, s.lo, s.hi, color='.35', alpha=.18, lw=0)
                axR.plot(s.payoff, s['median'], color='.1', lw=1.6, ls=LS[nm])
                axR.text(118, s['median'].iloc[-1], nm.split()[0], fontsize=6.1,
                         color='.2', va='center')
            logx(axR)
            axR.set_ylabel('Δ noise SD, IPS − vertex')
            axR.set_title('cTBS effect (95% CrI)', fontsize=7.5)
            row += 1

    for r in range(6):
        for c_ in range(2):
            if r < 5:
                axes[r][c_].set_xlabel('')
    axes[0][0].text(.03, .06, 'IPS', transform=axes[0][0].transAxes, color=IPS,
                    fontsize=7, va='bottom')
    axes[0][0].text(.03, .17, 'Vertex', transform=axes[0][0].transAxes,
                    color=VERTEX, fontsize=7, va='bottom')
    fig.text(.5, 1.005, 'Power-law noise (σ ∝ payoff^β), log-space observer',
             ha='center', fontsize=9)
    sns.despine(fig=fig, offset=3)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{a.out}.{ext}', bbox_inches='tight', pad_inches=0.03)
    print(f'wrote {a.out}.pdf')


if __name__ == '__main__':
    main()
