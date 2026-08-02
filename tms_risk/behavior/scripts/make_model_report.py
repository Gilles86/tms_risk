"""One folder of figures per fitted model, all from the pulled TSVs.

    python -m tms_risk.behavior.scripts.make_model_report --label flexible2nf

Writes notes/figures/<label>/ containing, in reading order:

    01_noise_functions   Fig 4B/4C layout: memory / perceptual noise per condition
    02_percept_shift     how cTBS distorts the perceived value of each option
    03_decision_space    Fig 5 layout: total noise + perceived EV ratio
    04_ppc               posterior predictive check, Fig 3A layout (copied if present)

Everything is rebuilt from `notes/data/*.<label>.tsv`, so this needs no trace,
no bauer and no GPU. The PPC is the exception -- it needs the model, so it is
produced remotely by `plot_ppc_fig3a` and copied in if it has been pulled.
"""
import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from tms_risk.behavior.scripts import plot_fig4bc_style, plot_fig5_style

VERTEX, IPS = '#2ca02c', '#d62728'
SAFE, RISKY = '#4d4d4d', '#b2182b'

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5, 'axes.linewidth': .8,
    'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})
sns.set_context('paper')


def percept_figure(data_dir, label, out_stem):
    """How the fitted priors distort perceived value, and what cTBS adds to that.

    A) perceived EV against objective EV, per option and stimulation condition.
       Distance below the identity line is prior attraction.
    B) the cTBS-induced shift in perceived EV, with 95% CrIs. The safe option
       loses more than the risky one -- that asymmetry is the risk-attitude effect.
    """
    p = pd.read_csv(Path(data_dir) / f'pmc_percepts.{label}.tsv', sep='\t')
    fig = plt.figure(figsize=(6.2, 2.9))
    gs = fig.add_gridspec(1, 2, wspace=.36, left=.09, right=.99, top=.84, bottom=.17)

    ax = fig.add_subplot(gs[0, 0])
    lim = [0, max(p.objective_ev.max(), p.vertex.max()) * 1.08]
    ax.plot(lim, lim, color='.75', lw=.7, ls=':', zorder=1)
    ax.text(lim[1] * .97, lim[1] * .97, 'Veridical', fontsize=5.6, color='.55',
            ha='right', va='bottom', rotation=45, rotation_mode='anchor')
    for opt, col in [('safe', SAFE), ('risky', RISKY)]:
        s = p[p.option == opt].sort_values('objective_ev')
        ax.plot(s.objective_ev, s.vertex, color=col, lw=1.3, marker='o', ms=3.2,
                label=f'{opt.capitalize()}, vertex')
        ax.plot(s.objective_ev, s.ips, color=col, lw=1.3, ls='--', marker='s',
                ms=3.2, mfc='white', label=f'{opt.capitalize()}, IPS')
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel('Objective expected value (CHF)')
    ax.set_ylabel('Perceived expected value (CHF)')
    ax.set_title('Percepts are pulled toward the prior', fontsize=7.5,
                 color='.15', pad=3)
    leg = ax.legend(fontsize=5.6, loc='upper left', handlelength=1.6,
                    borderpad=.35, labelspacing=.25, frameon=True)
    leg.get_frame().set_linewidth(.5); leg.get_frame().set_edgecolor('0.6')

    ax = fig.add_subplot(gs[0, 1])
    ax.axhline(0, color='.3', lw=.7, ls='--', zorder=1)
    for opt, col in [('safe', SAFE), ('risky', RISKY)]:
        s = p[p.option == opt].sort_values('n_safe')
        ax.fill_between(s.n_safe, s.lo, s.hi, color=col, alpha=.20, lw=0)
        ax.plot(s.n_safe, s.delta, color=col, lw=1.4, marker='o', ms=3.2,
                label=opt.capitalize())
    ax.set_xlabel('Safe payoff (CHF)')
    ax.set_ylabel('Δ perceived EV, IPS − vertex (CHF)')
    ax.set_title('cTBS costs the safe option more', fontsize=7.5, color='.15', pad=3)
    ax.set_xticks(sorted(p.n_safe.unique()))
    leg = ax.legend(title='Option', fontsize=6, title_fontsize=6, loc='lower left',
                    handlelength=1.3, borderpad=.35, labelspacing=.25)
    leg.get_frame().set_linewidth(.5); leg.get_frame().set_edgecolor('0.6')

    fig.text(.012, .95, 'A', fontsize=12, fontweight='bold', va='center')
    fig.text(.525, .95, 'B', fontsize=12, fontweight='bold', va='center')
    sns.despine(fig=fig, offset=2)
    for ext in ['pdf', 'png', 'svg']:
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def main(data_dir, fig_dir, label):
    out = Path(fig_dir) / label
    out.mkdir(parents=True, exist_ok=True)
    made, skipped = [], []

    steps = [
        ('01_noise_functions', lambda stem: plot_fig4bc_style.main(
            data_dir, label, stem, x_hi=50., y_hi=0.)),
        ('02_percept_shift', lambda stem: percept_figure(data_dir, label, stem)),
        ('03_decision_space', lambda stem: plot_fig5_style.main(
            data_dir, label, stem)),
    ]
    for name, fn in steps:
        try:
            fn(str(out / name))
            made.append(name)
        except FileNotFoundError as e:
            skipped.append(f'{name} (missing {Path(str(e).split()[-1]).name})')
        except Exception as e:
            skipped.append(f'{name} ({type(e).__name__}: {e})')

    for ext in ['pdf', 'png', 'svg']:
        src = Path(fig_dir) / f'ppc_fig3a.{label}.{ext}'
        if src.exists():
            shutil.copy(src, out / f'04_ppc.{ext}')
    if (out / '04_ppc.pdf').exists():
        made.append('04_ppc')
    else:
        skipped.append('04_ppc (run plot_ppc_fig3a on the node holding the trace)')

    print(f'\n{out}')
    for m in made:
        print(f'  wrote    {m}')
    for s in skipped:
        print(f'  skipped  {s}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', default='flexible2nf')
    parser.add_argument('--data_dir', default='/Users/gdehol/git/tms_risk/notes/data')
    parser.add_argument('--fig_dir', default='/Users/gdehol/git/tms_risk/notes/figures')
    args = parser.parse_args()
    main(args.data_dir, args.fig_dir, args.label)
