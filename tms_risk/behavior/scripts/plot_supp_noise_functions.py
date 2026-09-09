"""Every fitted noise function on one axis: does the answer depend on the form?

The reported model gives the noise function two anchors and a power law between
them. This asks whether that choice matters, by drawing the fitted ν(x) for the
whole family on a common axis — Weber (constant), generalised Weber, affine, the
power law, and the smooth-spline ladder at three, five and seven anchors.

Two panels, because there are two questions and they have different answers:

a  The noise functions themselves, vertex condition, first- and
   second-presented option. If the forms agree here, the reported model is not
   choosing the shape — the data are.
b  The cTBS effect each form implies, Δν against payoff, second-presented
   option. This is the one that matters: a robustness figure has to show the
   EFFECT is form-independent, not just the baseline.

The smooth (`cspl`) ladder is preferred over the piecewise-linear (`spl`) one
because it isolates resolution. `spl` uses the value link, where positivity
comes from a partition-of-unity basis and forces piecewise-linear
interpolation; `cspl` and `power` use the log link, where positivity comes from
the exp and a natural cubic through the anchors is admissible. So
power → cspl3 → cspl5 → cspl7 changes only the NUMBER of anchors, while
power → spl3 → spl5 changes the link at the same time.

    python -m tms_risk.behavior.scripts.plot_supp_noise_functions
"""
import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
REPO = Path(__file__).resolve().parents[3]
TICKS = [7, 14, 28, 56, 112]

#: (bare label, display name). Resolved to a KLW fit at read time.
FORMS = [('log-weber-n1n2',    'Weber (constant ν)'),
         ('log-genweber-n1n2', 'Generalised Weber'),
         ('log-affine-n1n2',   'Affine'),
         ('log-power-n1n2',    'Power law (reported)'),
         ('log-cspl3-n1n2',    'Smooth spline, 3 anchors'),
         ('log-cspl5-n1n2',    'Smooth spline, 5 anchors'),
         ('log-cspl7-n1n2',    'Smooth spline, 7 anchors')]

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 7.5, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})


def resolve(base, labels):
    """Bare label -> the KLW fit of it. Never a raw-choice-rule fit."""
    for suf in ('.mapjitter.klw', '.pathfinder.klw', '.klw'):
        if base + suf in labels:
            return base + suf
    return None


def logx(ax):
    ax.set_xscale('log')
    ax.set_xticks(TICKS)
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.minorticks_off()
    ax.set_xlim(6.6, 119)


def main(data_dir, out_stem):
    c = pd.read_csv(Path(data_dir) / 'anchor_curves.tsv', **READ)
    labels = set(c.label)
    got = [(resolve(b, labels), nm) for b, nm in FORMS]
    missing = [nm for L, nm in got if L is None]
    got = [(L, nm) for L, nm in got if L is not None]
    if missing:
        print('not fitted yet, omitted: ' + ', '.join(missing))
    cols = sns.color_palette('mako', n_colors=len(got) + 2)[1:-1]

    fig, AX = plt.subplots(1, 2, figsize=(7.0, 2.7), constrained_layout=True)

    ax = AX[0]
    for (L, nm), col in zip(got, cols):
        for ch, ls in (('n2', '-'), ('n1', (0, (3, 1.6)))):
            q = c[(c.label == L) & (c.channel == ch)
                  & (c.condition == 'vertex')].sort_values('x')
            if len(q):
                ax.plot(q.x, q['mid'], color=col, ls=ls, lw=1.3)
    logx(ax)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Representational noise ν')
    ax.set_title('Fitted noise functions (vertex)', fontsize=8)
    ax.plot([], [], color='0.35', ls='-', label='Second-presented')
    ax.plot([], [], color='0.35', ls=(0, (3, 1.6)), label='First-presented')
    ax.legend(loc='upper left', fontsize=6.4, handlelength=1.8)

    ax = AX[1]
    ax.axhline(0, color='0.45', lw=.9, zorder=0)
    for (L, nm), col in zip(got, cols):
        q = c[(c.label == L) & (c.channel == 'n2')
              & (c.condition == 'delta')].sort_values('x')
        if not len(q):
            continue
        ax.plot(q.x, q['mid'], color=col, lw=1.5, label=nm)
        if 'power' in L:                      # the reported form gets its band
            ax.fill_between(q.x, q.lo, q.hi, color=col, alpha=.15, lw=0,
                            zorder=0)
    logx(ax)
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('Δν, IPS − vertex')
    ax.set_title('cTBS effect, second-presented option', fontsize=8)
    ax.legend(loc='lower left', fontsize=6.2, handlelength=1.6,
              labelspacing=.3, borderaxespad=.2)
    ax.text(.97, .95, 'Band: 95% CrI of the reported form', ha='right',
            va='top', transform=ax.transAxes, fontsize=6.2, color='0.45')

    for letter, a_ in zip('ab', AX):
        a_.text(-.14, 1.04, letter, transform=a_.transAxes, fontsize=8.5,
                fontweight='bold', family='Arial', va='bottom', ha='right')
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}')
    print(f'wrote {out_stem}.pdf / .png  ({len(got)} forms)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/supp_noise_functions'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem)
