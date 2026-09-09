"""The design's own cells as the posterior predictive check.

The eight targeted statistics in the paper are contrasts someone chose: a mean
over risky-second trials, an order contrast, a three-way interaction. Each is
defensible, and collectively they invite the question of why those eight.

This is the answer to that question. The task used **five safe payoffs**, and
every trial is either risky-first or risky-second and either IPS or vertex. That
is a 5 x 2 x 2 grid of twenty cells fixed by the design before any data existed,
with nothing binned, ranked or pooled -- so it cannot be accused of being chosen
to flatter a model, and a reader can see the whole thing at once.

Two levels, and the second is the one that matters:

* **Levels** -- the twenty cell proportions. Mostly a check that the model has
  a psychometric function, which every model in the family passes.
* **Differences** -- the ten paired IPS - vertex contrasts, one per (order,
  safe payoff). This is the cTBS effect itself, cell by cell, and it is where
  models separate. Computed per posterior draw, so the interval is a genuine
  predictive interval for a contrast rather than the overlap of two marginals.

Coverage is reported as a count, not a p-value: how many of the ten fall inside
their own 95% predictive interval. Under a correct model about 9.5 of 10 should.
Fewer is misfit; ten of ten with very wide intervals is not a triumph, so the
median interval width is printed alongside.

    python -m tms_risk.behavior.scripts.ppc_design_grid \\
        --labels log-power-n1n2.mapjitter.klw log-power-percpmu.mapjitter.klw
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
IPS, VERTEX = '#d62728', '#2ca02c'
ORDERS = ['Risky first', 'Risky second']
#: order is never a hue (hue is stimulation); it is row position plus weight
ROWC = {'Risky first': '0.62', 'Risky second': '0.15'}


mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7,
    'axes.linewidth': .8, 'axes.spines.top': False, 'axes.spines.right': False,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'lines.linewidth': 1.2, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'savefig.pad_inches': .02,
})


def load(dd, label):
    lev = dd / f'ppc_anchor.safe.{label}.tsv'
    dif = dd / f'ppc_anchor.delta_safe.{label}.tsv'
    if not dif.exists():
        raise SystemExit(
            f'no delta_safe table for {label}\n  expected {dif}\n'
            f'  re-run extract_anchor_ppc for this label -- delta_safe was '
            f'added 2026-09-09 and older extractions do not have it.')
    return (pd.read_csv(lev, **READ) if lev.exists() else None,
            pd.read_csv(dif, **READ))


def covered(d):
    return ((d.observed >= d.lo) & (d.observed <= d.hi))


def table(labels, names, data_dir):
    dd = Path(data_dir) / 'ppc_anchor'
    rows = []
    for lab, nm in zip(labels, names):
        lev, dif = load(dd, lab)
        c = covered(dif)
        rows.append(dict(
            model=nm,
            delta_covered=f'{int(c.sum())}/{len(dif)}',
            level_covered=(f'{int(covered(lev).sum())}/{len(lev)}'
                           if lev is not None else '--'),
            # signed agreement: does the model get the DIRECTION right where
            # the observed effect is clear?
            sign_match=int(((np.sign(dif.model) == np.sign(dif.observed))
                            & (dif.observed.abs() > .01)).sum()),
            n_clear=int((dif.observed.abs() > .01).sum()),
            rmse=float(np.sqrt(((dif.model - dif.observed) ** 2).mean())),
            median_width=float((dif.hi - dif.lo).median()),
            mean_second=float(dif[dif.order == 'Risky second'].model.mean()),
            obs_second=float(dif[dif.order == 'Risky second'].observed.mean()),
        ))
    return pd.DataFrame(rows)


def figure(labels, names, data_dir, out_stem):
    dd = Path(data_dir) / 'ppc_anchor'
    n = len(labels)
    fig, AX = plt.subplots(2, n, figsize=(1.9 * n + .8, 3.8), sharex=True,
                           sharey=True, constrained_layout=True, squeeze=False)
    for c, (lab, nm) in enumerate(zip(labels, names)):
        _, dif = load(dd, lab)
        for r, order in enumerate(ORDERS):
            ax, col = AX[r, c], ROWC[order]
            o = dif[dif.order == order].sort_values('n_safe')
            x = np.arange(len(o))
            ax.axhline(0, color='.8', lw=.7, ls='--', zorder=0)
            for xi, (_, q) in zip(x, o.iterrows()):
                ax.plot([xi, xi], [q.lo, q.hi], color=col, lw=4, alpha=.22,
                        solid_capstyle='butt', zorder=1)
            ax.plot(x, o.model, 'o', ms=3.0, color=col, mfc='w', mew=1.1,
                    zorder=3)
            inside = covered(o).values
            ax.plot(x[inside], o.observed.values[inside], 'o', ms=4.4,
                    color=col, zorder=4)
            ax.plot(x[~inside], o.observed.values[~inside], 'o', ms=4.4,
                    color='#b2182b', zorder=5)
            ax.set_xticks(x)
            ax.set_xticklabels([f'{v:.0f}' for v in o.n_safe])
            ax.set_ylim(-.06, .16)
            if r == 0:
                ax.set_title(nm, fontsize=8)
            if r == 1:
                ax.set_xlabel('Safe payoff (CHF)')
            if c == 0:
                ax.set_ylabel(f'{order}\nΔ P(risky), IPS − vertex')
            k = int(covered(o).sum())
            ax.text(.03, .97, f'{k}/{len(o)} covered', transform=ax.transAxes,
                    fontsize=6.5, va='top', ha='left',
                    color='0.3' if k == len(o) else '#b2182b')
    # inline key, drawn as the marks themselves
    ax = AX[0, 0]
    for i, (lab_, kind) in enumerate([('Observed', 'obs'),
                                      ('Model median', 'mod'),
                                      ('95% predictive', 'band')]):
        yy = .10 + i * .085
        if kind == 'band':
            ax.plot([.06, .06], [yy - .03, yy + .03], transform=ax.transAxes,
                    color='.55', lw=4, alpha=.30, solid_capstyle='butt',
                    clip_on=False)
        else:
            ax.plot(.06, yy, 'o', transform=ax.transAxes, ms=4.0, color='.35',
                    mfc='.35' if kind == 'obs' else 'w',
                    mew=0 if kind == 'obs' else 1.1, clip_on=False)
        ax.text(.13, yy, lab_, transform=ax.transAxes, fontsize=6.3,
                color='.35', va='center')
    sns.despine(fig=fig, offset=4)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}')
    print(f'wrote {out_stem}.pdf / .png')


def main(labels, names, data_dir, out_stem, out_tsv):
    t = table(labels, names, data_dir)
    pd.set_option('display.width', 220)
    print(t.to_string(index=False, float_format=lambda v: f'{v:+.4f}'))
    print('\n  delta_covered  of the 10 (order x safe payoff) cTBS contrasts, '
          'how many the model covers\n'
          '  sign_match     of the cells where |observed| > 0.01, how many the '
          'model gets the SIGN of\n'
          '  median_width   median 95% predictive interval width -- 10/10 with '
          'a wide interval is not a pass')
    if out_tsv:
        Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
        t.to_csv(out_tsv, sep='\t', index=False)
        print(f'\nwrote {out_tsv}')
    figure(labels, names, data_dir, out_stem)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--labels', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', default=None)
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem',
                    default=str(REPO / 'notes/figures/ppc_design_grid'))
    ap.add_argument('--out_tsv',
                    default=str(REPO / 'notes/data/ppc_design_grid.tsv'))
    a = ap.parse_args()
    main(a.labels, a.names or a.labels, a.data_dir, a.out_stem, a.out_tsv)
