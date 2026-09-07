"""Figure 4, rebuilt on the anchor grid: does the noise function earn its shape?

a  Posterior predictive check, Weber beside the winning model, split by
   presentation order, against stake. The order x stake x stimulation
   interaction is in the data; Weber flattens it, the flexible model keeps it.
b  The winning model's two noise channels on log-log axes, both stimulation
   conditions, against a slope-1 (strict Weber) reference. A slope below 1 means
   noise grows more slowly than payoff.
c  The cTBS effect as a PERCENTAGE of the vertex noise -- the scale the
   psychophysics works on, and the one on which the effect is magnitude-specific.
d  Every model on one ELPD axis, as a difference from the best.

    python -m tms_risk.behavior.scripts.plot_fig4_anchor --model_label log-spl3-percmem
"""
import argparse
from glob import glob
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

IPS, VERTEX = '#d62728', '#2ca02c'
N1, N2 = '0.15', '0.55'
READ = dict(sep='\t', keep_default_na=False, na_values=[''])

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


def panel_letter(ax, s, dx=-.26, dy=1.06):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=8, fontweight='bold',
            va='bottom', ha='left', family='Arial')


def main(data_dir, out_stem, label, weber_label):
    dd = Path(data_dir)
    curves = pd.read_csv(dd / 'anchor_curves.tsv', **READ)
    c = curves[curves.label == label]
    if not len(c):
        raise SystemExit(f'no curves for {label}')
    ppc = {l: pd.read_csv(dd / f'ppc_anchor/ppc_anchor.stake.{l}.tsv', **READ)
           for l in (weber_label, label)
           if (dd / f'ppc_anchor/ppc_anchor.stake.{l}.tsv').exists()}
    # one TSV per model, each with index 0 -- concatenating without
    # ignore_index leaves duplicate labels and every .loc returns a Series
    loo = pd.concat([pd.read_csv(f, **READ)
                     for f in glob(str(dd / 'loo_anchor/loo.*.tsv'))],
                    ignore_index=True)
    loo = loo[loo.space == label.split('-')[0]]

    fig = plt.figure(figsize=(7.25, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.05],
                          width_ratios=[1, 1, 1, 1])

    # -- a: PPC, Weber vs winner ------------------------------------------
    axes_a = []
    for k, (lbl, ttl) in enumerate([(weber_label, 'Weber · constant σ'),
                                    (label, f'{label.split("-")[1]} · winner')]):
        for r, order in enumerate(['Risky first', 'Risky second']):
            ax = fig.add_subplot(gs[r, k])
            axes_a.append(ax)
            d = ppc.get(lbl)
            if d is None:
                ax.text(.5, .5, 'no PPC', transform=ax.transAxes, ha='center',
                        color='0.6')
                ax.set_xticks([]); ax.set_yticks([])
                continue
            d = d[d.order == order]
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                o = d[d.stim == stim].sort_values('stake_chf')
                ax.fill_between(o.stake_chf, o.lo, o.hi, color=col, alpha=.20, lw=0)
                ax.plot(o.stake_chf, o.model, color=col, lw=1.1)
                ax.errorbar(o.stake_chf, o.observed, yerr=o.observed_sem, fmt='o',
                            ms=3.4, color=col, lw=0, elinewidth=.9, capsize=0,
                            zorder=4)
            ax.set_ylim(.40, .74)
            ax.set_yticks([.45, .55, .65])
            ax.set_xscale('log')
            ax.set_xticks(sorted(d.stake_chf.unique()))
            ax.set_xticklabels([f'{v:.0f}' for v in sorted(d.stake_chf.unique())])
            ax.minorticks_off()
            if r == 0:
                ax.set_title(ttl, fontsize=7.5, color='0.15', pad=3)
            else:
                ax.set_xlabel('Stake (CHF)')
            if k == 0:
                ax.set_ylabel(f'P(chose risky)\n{order.lower()}')
            else:
                ax.set_yticklabels([])
    axes_a[0].text(.05, .95, 'IPS', color=IPS, transform=axes_a[0].transAxes,
                   va='top', fontsize=7)
    axes_a[0].text(.05, .80, 'Vertex', color=VERTEX, transform=axes_a[0].transAxes,
                   va='top', fontsize=7)
    panel_letter(axes_a[0], 'a', dx=-.42)

    # -- b: the noise functions, log-log ----------------------------------
    ax = fig.add_subplot(gs[0, 2])
    for chan, ls, nm in [('n1', '-', 'σ$_{n1}$ first'),
                         ('n2', (0, (2.6, 1.4)), 'σ$_{n2}$ second')]:
        for cond, col in [('vertex', VERTEX), ('ips', IPS)]:
            s = c[(c.channel == chan) & (c.condition == cond)].sort_values('x')
            ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.15, lw=0)
            ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.2)
    # THE OBSERVER IS IN LOG SPACE: sigma is the SD of the log-payoff percept,
    # i.e. a relative (coefficient-of-variation-like) noise. So Weber's law here
    # is CONSTANT sigma -- slope 0 on these axes, not slope 1. Slope 1 is the
    # Weber reference for a natural-space observer, where sigma is in CHF.
    s = c[(c.channel == 'n1') & (c.condition == 'vertex')].sort_values('x')
    s2 = c[(c.channel == 'n2') & (c.condition == 'vertex')].sort_values('x')
    ref = float(s['mid'].iloc[0])
    ax.axhline(ref, color='0.7', lw=.8, ls=':')
    ax.text(7.3, ref * 1.06, 'Weber: constant relative noise (b = 0)', fontsize=6,
            color='0.5', ha='left', va='bottom')
    b1 = np.polyfit(np.log(s.x.values), np.log(s['mid'].values), 1)[0]
    b2 = np.polyfit(np.log(s2.x.values), np.log(s2['mid'].values), 1)[0]
    ax.text(.97, .97, f'σ ∝ x$^b$:   b$_{{n1}}$ = {b1:.2f}\n'
                      f'              b$_{{n2}}$ = {b2:.2f}',
            transform=ax.transAxes, ha='right', va='top', fontsize=6.5,
            color='0.2', linespacing=1.6)
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.set_ylim(.055, .55)
    ax.set_yticks([.1, .2, .4])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.minorticks_off()
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('σ (log CHF)')
    ax.text(.03, .04, 'Solid: first-presented\nDashed: second',
            transform=ax.transAxes, fontsize=6, color='0.4', linespacing=1.5,
            va='bottom')
    panel_letter(ax, 'b')

    # -- c: the cTBS effect, as a percentage ------------------------------
    ax = fig.add_subplot(gs[1, 2])
    have_pct = (c.condition == 'delta_pct').any()
    ax.axhline(0, color='0.75', lw=.6, ls='--', zorder=0)
    for chan, ls, col in [('n1', '-', N1), ('n2', (0, (2.6, 1.4)), N2)]:
        s = c[(c.channel == chan)
              & (c.condition == ('delta_pct' if have_pct else 'delta'))
              ].sort_values('x')
        ax.fill_between(s.x, s.lo, s.hi, color=col, alpha=.20, lw=0)
        ax.plot(s.x, s['mid'], color=col, ls=ls, lw=1.2)
    ax.set_xscale('log')
    ax.set_xticks([7, 14, 28, 56, 112])
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.minorticks_off()
    ax.set_xlabel('Payoff (CHF)')
    ax.set_ylabel('cTBS effect on σ (%)' if have_pct else 'Δσ, IPS − vertex')
    panel_letter(ax, 'c')

    # -- d: the ladder ----------------------------------------------------
    ax = fig.add_subplot(gs[:, 3])
    if len(loo):
        d = (loo.sort_values('elpd_loo', ascending=False)
                .reset_index(drop=True).copy())
        d['d'] = d.elpd_loo - d.elpd_loo.max()
        y = np.arange(len(d))[::-1]
        col = ['#d62728' if l == label else
               ('0.62' if p in ('null', 'nullind') else '0.25')
               for l, p in zip(d.label, d.placement)]
        ax.errorbar(d['d'], y, xerr=d.se, fmt='none', ecolor='0.8', elinewidth=.6)
        ax.scatter(d['d'], y, s=7, c=col, zorder=3, lw=0)
        ax.axvline(0, color='0.75', lw=.6, ls='--', zorder=0)
        mark = {0} | {i for i, l in enumerate(d.label) if l == label}
        for i in sorted(mark):
            ax.text(d['d'].iat[i] + 6, y[i], d.label.iat[i], fontsize=6,
                    va='center',
                    color='#d62728' if d.label.iat[i] == label else '0.25')
        ax.text(.02, .015, 'Grey: null models\n(no cTBS effect)',
                transform=ax.transAxes, fontsize=6, color='0.5',
                linespacing=1.5, va='bottom')
        ax.set_yticks([])
        ax.set_xlabel('ΔELPD vs best')
        ax.set_ylim(-1, len(d))
    panel_letter(ax, 'd', dx=-.12)

    sns.despine(fig=fig, offset=3)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_stem}.{ext}', bbox_inches='tight', pad_inches=.02)
    print(f'wrote {out_stem}.pdf / .png')


if __name__ == '__main__':
    REPO = Path(__file__).resolve().parents[2].parent
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_dir', default=str(REPO / 'notes/data'))
    ap.add_argument('--out_stem', default=str(REPO / 'notes/figures/fig4_anchor'))
    ap.add_argument('--model_label', default='log-power-n1n2')
    ap.add_argument('--weber_label', default='log-weber-n1n2')
    a = ap.parse_args()
    main(a.data_dir, a.out_stem, a.model_label, a.weber_label)
