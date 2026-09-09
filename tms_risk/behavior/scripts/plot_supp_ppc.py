"""Every posterior predictive check on one page.

Two panels, because a predictive check has to answer two questions and they
need different pictures.

a  **Can the model produce the behavioural effects that were measured?** Every
   targeted statistic on one standardised axis: the observed value's position
   inside the model's own 95% predictive interval, with each half of the
   interval scaled separately, so −1 / 0 / +1 read as the 2.5th percentile, the
   median and the 97.5th. A dot inside the grey span is covered; the posterior
   predictive p is printed beside it. Aggregating to one number per statistic is
   the point: the cell-level checks plot 12–20 trials per participant, where a
   point's own standard error is as wide as the model's band and the eye reads
   sampling noise as misfit.

b  **Does it work for individual participants, or only for the group?** All 420
   participant × order × stake × stimulation cells, observed against predicted,
   with the identity line. Coverage — the fraction of cells whose observed value
   falls inside its own 95% predictive interval — is printed; 95% is what a
   calibrated model gives.

No error bars are drawn on observed values anywhere here. These are predictive
checks: the observed value is a statistic and the band is the model's
distribution for it, which already contains the sampling noise an s.e.m. would
draw.

    python -m tms_risk.behavior.scripts.plot_supp_ppc \\
        --model_label log-power-n1n2.mapjitter.klw
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

#: statistic -> the plain-language question it asks, in the order they are read
NAMES = [
    ('dp_second_mean',     'cTBS effect on P(risky), risky second'),
    ('dp_first_mean',      '… risky first'),
    ('dp_second_high',     '… risky second, largest stakes'),
    ('order_contrast',     'Effect is larger when the risky option is second'),
    ('stake_slope_second', 'Effect varies with stake (risky second)'),
    ('three_way',          'That stake dependence differs by order'),
    ('slope_second_ctbs',  'cTBS flattens the psychometric curve (risky second)'),
    ('slope_contrast',     '… and does so more than when risky is first'),
]

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


def main(data_dir, out_stem, label):
    dd = Path(data_dir) / 'ppc_anchor'
    st = pd.read_csv(dd / f'ppc_stats.{label}.tsv', **READ).set_index('statistic')
    sub = pd.read_csv(dd / f'ppc_subject.{label}.tsv', **READ)

    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=.30,
                          left=.005, right=.985, top=.86, bottom=.16)
    ax = fig.add_subplot(gs[0])

    rows = [(k, nm) for k, nm in NAMES if k in st.index]
    y = np.arange(len(rows))[::-1]
    ax.axvspan(-1, 1, color='0.86', lw=0, zorder=0)
    ax.axvline(0, color='0.55', lw=.9, zorder=1)
    for yy, (k, nm) in zip(y, rows):
        r = st.loc[k]
        # scale each half of the interval separately, so -1 / 0 / +1 are the
        # 2.5th percentile, the median and the 97.5th whatever the units
        half = (r.hi - r.model_median) if r.observed >= r.model_median else \
               (r.model_median - r.lo)
        z = (r.observed - r.model_median) / half if half > 0 else np.nan
        ok = bool(r.covered)
        ax.plot(np.clip(z, -2.6, 2.6), yy, 'o', ms=5.0,
                color='0.12' if ok else IPS, zorder=4,
                clip_on=False)
        # label ABOVE its row: two of the statistics sit at the far left of
        # the interval, where a same-line label collides with its own dot
        ax.text(-2.95, yy + .30, nm, fontsize=6.6, va='bottom', ha='left',
                color='0.2' if ok else IPS)
        ax.text(2.92, yy, f'p = {r.ppp:.3f}', fontsize=6.6, va='center',
                ha='right', color='0.45' if ok else IPS)
    ax.set_yticks([])
    ax.set_xlim(-3.0, 3.0)
    ax.set_xticks([-1, 0, 1])
    ax.set_xticklabels(['2.5%', 'median', '97.5%'], fontsize=6.6)
    ax.set_ylim(-1.0, len(rows) - .05)
    ax.set_xlabel("Observed value's position in the model's predictive interval")
    ax.set_title('Can the model produce what was measured?', fontsize=8)
    ax.text(0, -.85, 'Grey: 95% predictive interval', fontsize=6.2,
            color='0.45', ha='center')
    sns.despine(ax=ax, left=True, offset=3)

    ax = fig.add_subplot(gs[1])
    ax.plot([0, 1], [0, 1], color='0.6', lw=.9, ls='--', zorder=1)
    for stim, col, mk in (('vertex', VERTEX, 'o'), ('ips', IPS, 's')):
        q = sub[sub.stim == stim]
        ax.plot(q.model, q.observed, mk, ms=2.6, color=col, alpha=.75, mew=0,
                zorder=3)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([0, .5, 1]); ax.set_yticks([0, .5, 1])
    ax.set_xlabel('Predicted P(chose risky)')
    ax.set_ylabel('Observed P(chose risky)')
    ax.set_title('Individual participants', fontsize=8)
    ax.text(.04, .96, f'{len(sub)} participant × cell\n'
                      f'{sub.covered.mean():.0%} inside their own 95% interval\n'
                      f'r = {np.corrcoef(sub.observed, sub.model)[0, 1]:.2f}',
            transform=ax.transAxes, va='top', fontsize=6.4, color='0.25')
    ax.plot(.72, .16, 's', ms=3.4, color=IPS, transform=ax.transAxes, mew=0)
    ax.text(.77, .16, 'IPS', transform=ax.transAxes, fontsize=6.6, color=IPS,
            va='center')
    ax.plot(.72, .07, 'o', ms=3.4, color=VERTEX, transform=ax.transAxes, mew=0)
    ax.text(.77, .07, 'Vertex', transform=ax.transAxes, fontsize=6.6,
            color=VERTEX, va='center')
    sns.despine(ax=ax, offset=3)

    for letter, x_ in zip('ab', (.005, .60)):
        fig.text(x_, .93, letter, fontsize=8.5, fontweight='bold',
                 family='Arial')
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}')
    print(f'wrote {out_stem}.pdf / .png  ({len(rows)} statistics, '
          f'{len(sub)} cells)')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--model_label', default='log-power-n1n2.mapjitter.klw')
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/supp_ppc'))
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label)
