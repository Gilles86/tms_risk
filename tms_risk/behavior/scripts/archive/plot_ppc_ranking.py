"""Rank every model by how well it PREDICTS, not by how well it scores.

ELPD on this dataset is dominated by the bulk choice curve -- 8335 trials of
"steeper ratio, more risky choices" that every model gets right -- so the cTBS x
order x stake interaction the paper is about contributes almost nothing to it.
These four panels ask the model to produce the data instead.

a  RMSE of model minus observed over the twelve order x stake x stimulation
   cells, split by presentation order.
b  Calibration: how many of those twelve group cells fall inside their own 95%
   posterior predictive interval, and how many of the 420 subject-level cells
   do. A well-calibrated model covers ~95% of both.
c  The statistic that carries the claim: the cTBS effect on P(risky) when the
   risky option came second, minus when it came first. Posterior-predictive p
   near 0 means the model cannot generate what was measured.
d  Individual participants for the leading model: observed against predicted,
   with the ones outside their own 95% interval marked.

    python -m tms_risk.behavior.scripts.plot_ppc_ranking
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

READ = dict(sep='\t', keep_default_na=False, na_values=[''])
FIRST, SECOND = '0.62', '0.15'
ACC, WARN = '#3B5BA5', '#d62728'
SHARED = ('null', 'perc', 'mem', 'percmem')

mpl.rcParams.update({
    'font.family': 'Helvetica',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'TeX Gyre Heros', 'Arial'],
    'font.size': 7, 'axes.labelsize': 8, 'axes.titlesize': 8,
    'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 7,
    'axes.linewidth': 0.8, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.labelpad': 3, 'xtick.direction': 'out', 'ytick.direction': 'out',
    'xtick.major.size': 3, 'ytick.major.size': 3,
    'xtick.major.width': 0.8, 'ytick.major.width': 0.8,
    'lines.linewidth': 1.1, 'legend.frameon': False,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.dpi': 150, 'savefig.dpi': 300,
})


def cat(pattern, dd):
    fs = sorted(glob(str(dd / pattern)))
    if not fs:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, **READ) for f in fs], ignore_index=True)


def main(data_dir, out_stem, space, top_n, sort):
    dd = Path(data_dir)
    stake = cat('ppc_anchor/ppc_anchor.stake.*.tsv', dd)
    stats = cat('ppc_anchor/ppc_stats.*.tsv', dd)
    subj = cat('ppc_anchor/ppc_subject.*.tsv', dd)
    if not len(stake):
        raise SystemExit('no PPC output yet')
    for df in (stake, stats, subj):
        if len(df):
            df['space'] = df.label.str.split('-').str[0]
            df['placement'] = df.label.str.split('-').str[2]
            df.drop(df.index[df.space != space], inplace=True)

    # -- per-model summaries ----------------------------------------------
    stake['err'] = stake.model - stake.observed
    stake['inside'] = (stake.lo <= stake.observed) & (stake.observed <= stake.hi)
    rmse = (stake.assign(e=stake.err ** 2).groupby(['label', 'order'])['e']
            .mean().pow(.5).unstack('order'))
    cov = stake.groupby('label')['inside'].mean().rename('cov_group')
    ppp = (stats[stats.statistic == 'order_contrast']
           .set_index('label')['ppp'].rename('ppp'))
    obs_oc = float(stats[stats.statistic == 'order_contrast'].observed.iloc[0])
    m = rmse.join(cov).join(ppp)
    if len(subj):
        m = m.join(subj.groupby('label')['covered'].mean().rename('cov_subj'))
    m['placement'] = [l.split('-')[2] for l in m.index]
    m['rmse'] = m[['Risky first', 'Risky second']].mean(axis=1)
    # Sort key is explicit: the default is the targeted statistic (panel c),
    # NOT RMSE. Worth saying because panel a then looks unsorted -- RMSE and the
    # order-contrast ppp rank models almost independently.
    key = {'ppp': ('ppp', False), 'rmse': ('rmse', True),
           'cov': ('cov_subj', False)}[sort]
    m = m.sort_values(key[0], ascending=key[1])
    best = m.index[0]
    sel = (m if top_n <= 0 else m.head(top_n)).iloc[::-1]

    n_row = len(sel)
    h_top = max(2.6, .115 * n_row)
    fig = plt.figure(figsize=(7.25, h_top + 2.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[h_top, 2.2])
    y = np.arange(len(sel))

    def ylabels(ax):
        ax.set_yticks(y)
        ax.set_yticklabels(sel.index, fontsize=5.6)
        for t, p in zip(ax.get_yticklabels(), sel.placement):
            t.set_color('0.15' if p in SHARED else ACC)

    # -- a: RMSE -----------------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.barh(y - .19, sel['Risky first'], height=.36, color=FIRST, lw=0)
    ax.barh(y + .19, sel['Risky second'], height=.36, color=SECOND, lw=0)
    ylabels(ax)
    ax.set_xlabel('RMSE, model − observed')
    ax.set_ylim(-.7, len(sel) - .1)
    ax.text(.98, .015, 'Risky second', color=SECOND, transform=ax.transAxes,
            ha='right', fontsize=6.2)
    ax.text(.98, .055, 'Risky first', color=FIRST, transform=ax.transAxes,
            ha='right', fontsize=6.2)
    ax.set_title('a  Size of the miss', loc='left', fontsize=8)

    # -- b: coverage -------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.axvline(.95, color=WARN, lw=.8, ls='--', zorder=0)
    ax.scatter(sel.cov_group, y, s=16, color='0.15', zorder=3, label='Group cells')
    if 'cov_subj' in sel:
        ax.scatter(sel.cov_subj, y, s=16, marker='s', facecolor='white',
                   edgecolor=ACC, lw=.9, zorder=3, label='Subject cells')
    ylabels(ax)
    ax.set_yticklabels([])
    ax.set_xlim(0, 1.04)
    ax.set_xticks([0, .25, .5, .75, .95])
    ax.set_xticklabels(['0', '.25', '.5', '.75', '.95'])
    ax.set_xlabel('Fraction inside 95% PPI')
    ax.set_ylim(-.7, len(sel) - .1)
    ax.text(.03, .015, 'Filled: 12 group cells\nOpen: 420 subject cells',
            transform=ax.transAxes, fontsize=6.2, color='0.35', linespacing=1.5)
    ax.set_title('b  Calibration', loc='left', fontsize=8)

    # -- c: the statistic that carries the claim ---------------------------
    ax = fig.add_subplot(gs[0, 2])
    st = stats[stats.statistic == 'order_contrast'].set_index('label')
    st = st.reindex(sel.index)
    ax.axvline(obs_oc, color=WARN, lw=.9, ls='--', zorder=0)
    ax.hlines(y, st.lo, st.hi, color='0.75', lw=1.0, zorder=2)
    ax.scatter(st.model_median, y, s=16, color='0.15', zorder=3)
    ylabels(ax)
    ax.set_yticklabels([])
    ax.set_xlabel('cTBS effect, second − first')
    ax.set_ylim(-.7, len(sel) - .1)
    ax.text(obs_oc, len(sel) - .35, ' Observed', color=WARN, fontsize=6.5,
            va='center')
    ax.set_title('c  Can it produce the order effect?', loc='left', fontsize=8)

    # -- d: individual participants for the leading model ------------------
    ax = fig.add_subplot(gs[1, :])
    if len(subj):
        s = subj[(subj.label == best) & (subj.order == 'Risky second')].copy()
        s = s.sort_values('observed')
        xx = np.arange(len(s))
        ax.vlines(xx, s.lo, s.hi, color='0.85', lw=1.4, zorder=1)
        ax.scatter(xx, s.model, s=8, color='0.55', zorder=3)
        out = ~s.covered.values
        ax.scatter(xx[~out], s.observed.values[~out], s=12, color='0.15', zorder=4)
        ax.scatter(xx[out], s.observed.values[out], s=16, color=WARN, zorder=5)
        ax.set_xticks([])
        ax.set_xlabel('Participant × stake tercile, sorted by observed '
                      '(risky-second trials only)')
        ax.set_ylabel('P(chose risky)')
        ax.set_title(f'd  Individual participants · {best} · '
                     f'{int(out.sum())}/{len(s)} outside their own 95% interval',
                     loc='left', fontsize=8)
        ax.text(.005, .97, 'Black: observed   Grey: model mean and 95% PPI   '
                           'Red: observed outside the interval',
                transform=ax.transAxes, fontsize=6.2, color='0.35', va='top')
    else:
        ax.text(.5, .5, 'per-subject PPC not extracted yet', ha='center',
                transform=ax.transAxes, color='0.6')
        ax.set_xticks([]); ax.set_yticks([])

    shown = ('all' if top_n <= 0 else f'top {top_n} by {sort}')
    fig.suptitle(f'Posterior-predictive ranking · {space} space · '
                 f'{m.shape[0]} models, {shown} · sorted by '
                 f'{"order-contrast ppp" if sort == "ppp" else sort} · '
                 f'label colour: dark = shared perc/mem, blue = independent n1/n2',
                 fontsize=7, color='0.4', y=1.03)
    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')
    print(m.head(12).round(4).to_string())


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/ppc_ranking'))
    ap.add_argument('--space', default='log')
    ap.add_argument('--top_n', default=0, type=int,
                    help='0 = every model')
    ap.add_argument('--sort', default='ppp',
                    choices=['ppp', 'rmse', 'cov'])
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.space, a.top_n, a.sort)
