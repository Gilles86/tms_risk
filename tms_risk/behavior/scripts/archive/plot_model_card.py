"""One page per model: what it assumes, what it estimated, and whether it fits.

Reads the TSVs written by model_card_data.py (params, noise curves per
stimulation condition, and PPCs in the paradigm's own cells). Every panel is
subject-averaged within draw, then summarized across draws.

The title is a plain-English reading of the label, because `lfx2-pl-m2-dp-m-p2-i`
is not something a reader should have to decode.

    python -m tms_risk.behavior.scripts.plot_model_card <label> [<label> ...]
    python -m tms_risk.behavior.scripts.plot_model_card --all
"""
import argparse
import glob
import re
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

IPS, VERTEX, C_MOD, C_OBS = '#d62728', '#2ca02c', '.35', '#C44E52'
CC = {'memory': '#C44E52', 'perceptual': '#3B5BA5',
      'n1 (first)': '.15', 'n2 (second)': '.58'}

BASIS = {'bs3': 'cubic B-spline (5 df)', 'bs2': 'quadratic B-spline (5 df)',
         'cr3': 'natural cubic spline', 'gw': 'generalized Weber (k + c/payoff)',
         'pl': 'power law (σ ∝ payoff^β)'}
MEM = {'w': 'both channels constant (Weber)', 'sm': 'constant memory',
       'm2': 'memory affine in log payoff', 'm3': 'memory quadratic',
       'fm': 'memory fully flexible'}
TMS = {'null': 'NO cTBS effect', 'b': 'cTBS on perceptual noise only',
       'bm': 'cTBS on both noise channels', 't': 'cTBS on total noise',
       'm': 'cTBS on memory noise only'}
TMS_I = {'null': 'NO cTBS effect', 'b': 'cTBS on the SECOND-presented option',
         'bm': 'cTBS on both options', 't': 'cTBS on total noise',
         'm': 'cTBS on the FIRST-presented option'}


def describe(label):
    m = re.fullmatch(r'lfx2-(bs3|bs2|cr3|gw|pl)-(fm|sm|m2|m3|w|sd\d)-(dp|tp)-'
                     r'(null|b|bm|t|m)(-p[1-5])?(-i)?(-\w+)?', label)
    if not m:
        return label, ''
    basis, mem, hp, tms, pdf, ind, extra = m.groups()
    noise = BASIS.get(basis, basis)
    if pdf:
        noise = (f'affine in log payoff (2 df)' if pdf == '-p2'
                 else f'{pdf[2]}-df') if basis in ('bs3', 'bs2', 'cr3') else noise
    parts = [f'Log-space Bayesian observer, lognormal priors',
             f'noise: {noise}' + (f'; {MEM.get(mem, mem)}' if not pdf else ''),
             (TMS_I if ind else TMS).get(tms, tms),
             ('n1/n2 parameterization (per presented option)' if ind
              else 'memory/perceptual parameterization')]
    return ' · '.join(parts[:2]), ' · '.join(parts[2:])


def logx(ax, ticks=(7, 14, 28, 56, 112)):
    ax.set_xscale('log'); ax.set_xticks(list(ticks))
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.xaxis.set_minor_locator(mticker.NullLocator())


def lognormal_sd(x, s):
    return x * np.exp(s ** 2 / 2) * np.sqrt(np.exp(s ** 2) - 1)


def curves_panel(ax, cur, names, natural=False, ylab=''):
    for nm in names:
        c0 = cur[cur.curve == nm]
        if not len(c0):
            continue
        for stim in sorted(c0.stim.unique()):
            s = c0[c0.stim == stim].sort_values('payoff')
            col = {'ips': IPS, 'vertex': VERTEX}.get(stim, CC.get(nm, '.3'))
            y, lo, hi = s['median'].values, s.lo.values, s.hi.values
            if natural:
                y, lo, hi = (lognormal_sd(s.payoff.values, v) for v in (y, lo, hi))
            ax.fill_between(s.payoff, lo, hi, color=col, alpha=.16, lw=0)
            ax.plot(s.payoff, y, color=col, lw=1.5,
                    ls='-' if nm.startswith('n1') or nm == 'memory' else '--')
        s = c0[c0.stim == sorted(c0.stim.unique())[0]].sort_values('payoff')
        y0 = s['median'].iloc[-1]
        if natural:
            y0 = lognormal_sd(s.payoff.iloc[-1], y0)
        ax.text(118, y0, nm.split()[0], fontsize=6.2, color='.2', va='center')
    logx(ax); ax.set_yscale('log')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f'{v:g}'))
    ax.yaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlabel('Payoff (CHF)'); ax.set_ylabel(ylab)


def card(label, D, out_dir, pdf=None):
    meta = pd.read_csv(D / f'meta.{label}.tsv', sep='\t').iloc[0]
    pars = pd.read_csv(D / f'params.{label}.tsv', sep='\t')
    cur = pd.read_csv(D / f'curves.{label}.tsv', sep='\t')

    fig = plt.figure(figsize=(7.6, 6.4))
    gs = fig.add_gridspec(3, 3, hspace=.75, wspace=.44, left=.10, right=.96,
                          top=.855, bottom=.075)

    # a: parameters
    ax = fig.add_subplot(gs[0, 0])
    p = pars.sort_values('mean').reset_index(drop=True)
    y = np.arange(len(p))
    cols = ['.1' if r.regressor == 'Intercept' else IPS for r in p.itertuples()]
    ax.hlines(y, p.lo, p.hi, color='.5', lw=.9)
    for yi, r, c in zip(y, p.itertuples(), cols):
        ax.plot(r.mean, yi, 'o', ms=3.6, color=c)
    ax.axvline(0, color='.75', lw=.7, ls='--', zorder=0)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{s.replace('_noise_sd','').replace('_',' ')}"
                        + ('' if r == 'Intercept' else '  [cTBS]')
                        for s, r in zip(p.param, p.regressor)], fontsize=5.4)
    ax.set_xlabel('Group mean'); ax.set_title('Parameters', fontsize=8)

    # b/c: noise, log and natural space
    curves_panel(fig.add_subplot(gs[0, 1]), cur, ['n1 (first)', 'n2 (second)'],
                 False, 'Noise SD (log units)')
    fig.axes[-1].set_title('Noise per option — log space', fontsize=8)
    curves_panel(fig.add_subplot(gs[0, 2]), cur, ['n1 (first)', 'n2 (second)'],
                 True, 'Noise SD (CHF)')
    fig.axes[-1].set_title('Noise per option — natural space', fontsize=8)

    # d: psychometric overall
    ax = fig.add_subplot(gs[1, 0])
    t = pd.read_csv(D / f'ppc_by_ratio.{label}.tsv', sep='\t')
    b = pd.read_csv(D / f'bins_ratio_bin.{label}.tsv', sep='\t').set_index('ratio_bin')['x']
    x = b.reindex(t.ratio_bin).values
    ax.fill_between(x, t.lo, t.hi, color=C_MOD, alpha=.22, lw=0)
    ax.plot(x, t['median'], color=C_MOD, lw=1.4)
    ax.plot(x, t.observed, 'o', ms=4, color=C_OBS)
    ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
    ax.set_title(f'Psychometric  {meta.by_ratio}', fontsize=8)

    # e: psychometric by order
    ax = fig.add_subplot(gs[1, 1])
    t = pd.read_csv(D / f'ppc_by_ratio_order.{label}.tsv', sep='\t')
    for order, ls, mfc in [('Risky first', '--', 'white'), ('Risky second', '-', C_OBS)]:
        s = t[t.order == order].sort_values('ratio_bin')
        xx = b.reindex(s.ratio_bin).values
        ax.fill_between(xx, s.lo, s.hi, color=C_MOD, alpha=.16, lw=0)
        ax.plot(xx, s['median'], color=C_MOD, lw=1.3, ls=ls)
        ax.plot(xx, s.observed, 'o', ms=3.6, color=C_OBS, mfc=mfc, mew=1.0)
    ax.set_xlabel('log(risky / safe)'); ax.set_ylabel('P(chose risky)')
    ax.set_title(f'Psychometric by order  {meta.get("by_ratio_order","")}', fontsize=8)
    ax.text(.03, .96, 'Filled: risky second\nOpen: risky first',
            transform=ax.transAxes, fontsize=5.7, va='top', color='.3',
            linespacing=1.3)

    # f, g: stake x cTBS, SPLIT BY PRESENTATION ORDER -- the paradigm's own cells
    # and where the whole effect lives. Three stake terciles, two conditions,
    # model band vs observed dot. Shared y so the two orders are comparable.
    fo = D / f'ppc_by_stake_stim_order.{label}.tsv'
    axes_fg = [fig.add_subplot(gs[1, 2]), fig.add_subplot(gs[2, 0])]
    if fo.exists():
        t = pd.read_csv(fo, sep='\t')
        b3 = pd.read_csv(D / f'bins_stake3.{label}.tsv', sep='\t').set_index('stake3')['x']
        ylo = min(t.lo.min(), t.observed.min()) - .01
        yhi = max(t.hi.max(), t.observed.max()) + .01
        for ax, order in zip(axes_fg, ['Risky first', 'Risky second']):
            for stim, col in [('vertex', VERTEX), ('ips', IPS)]:
                s3 = t[(t.order == order) &
                       (t.stimulation_condition == stim)].sort_values('stake3')
                xx = b3.reindex(s3.stake3).values
                ax.fill_between(xx, s3.lo, s3.hi, color=col, alpha=.16, lw=0)
                ax.plot(xx, s3['median'], color=col, lw=1.4)
                ax.plot(xx, s3.observed, 'o', ms=5, color=col)
            logx(ax, ticks=(13, 23, 42))
            ax.set_ylim(ylo, yhi)
            ax.set_xlabel('Stake (CHF)')
            ax.set_ylabel('P(chose risky)')
            n_in = int(((s3.observed >= s3.lo) & (s3.observed <= s3.hi)).sum())
            ax.set_title(f'Stake x cTBS - {order.lower()}', fontsize=8)
        axes_fg[0].text(.03, .04, 'Line + band: model   Dots: observed',
                        transform=axes_fg[0].transAxes, fontsize=5.7, color='.3')
        axes_fg[0].text(.03, .96, 'IPS', transform=axes_fg[0].transAxes,
                        color=IPS, fontsize=7, va='top')
        axes_fg[0].text(.03, .84, 'Vertex', transform=axes_fg[0].transAxes,
                        color=VERTEX, fontsize=7, va='top')
        cov = meta.get('by_stake_stim_order', '')
        axes_fg[1].text(.97, .04, f'coverage {cov}', transform=axes_fg[1].transAxes,
                        fontsize=6.0, color='.35', ha='right')
    else:
        for ax in axes_fg:
            ax.axis('off')

    # h: the cTBS contrast
    ax = fig.add_subplot(gs[2, 1])
    t = pd.read_csv(D / f'ppc_by_stim.{label}.tsv', sep='\t')
    w = t.pivot(index='order', columns='stimulation_condition',
                values=['median', 'lo', 'hi', 'observed'])
    yy = np.arange(len(w))
    for k, (stim, col) in enumerate([('vertex', VERTEX), ('ips', IPS)]):
        off = (k - .5) * .22
        ax.hlines(yy + off, w[('lo', stim)], w[('hi', stim)], color=col, lw=1.2)
        ax.plot(w[('median', stim)], yy + off, 'o', ms=3.4, color=col, mfc='white',
                mew=1.1)
        ax.plot(w[('observed', stim)], yy + off, 'o', ms=4.4, color=col)
    ax.set_yticks(yy); ax.set_yticklabels(w.index, fontsize=7)
    ax.set_xlabel('P(chose risky)')
    ok = meta.by_stim.split('/')[0] == meta.by_stim.split('/')[1]
    ax.set_title(f'cTBS contrast  {meta.by_stim}', fontsize=8,
                 color='.1' if ok else '#b0453b')
    ax.text(.02, .04, 'Open: model   Filled: observed', transform=ax.transAxes,
            fontsize=5.7, color='.3')

    # i: the cTBS DIFFERENCE with its own interval, if it has been extracted.
    # The difference of the two marginal bands is not the band of the difference:
    # IPS and vertex come from the same draws and are strongly correlated.
    ax = fig.add_subplot(gs[2, 2])
    dfile = Path('notes/data/delta') / f'delta.{label}.tsv'
    if dfile.exists():
        d = pd.read_csv(dfile, sep='\t')
        ax.axhline(0, color='.7', lw=.7, ls='--', zorder=0)
        for nm, ls in [('n1 (first)', '-'), ('n2 (second)', '--')]:
            sdd = d[d.curve == nm].sort_values('payoff')
            if not len(sdd):
                continue
            if sdd['median'].abs().max() < 1e-9:
                ax.plot(sdd.payoff, sdd['median'], color='.6', lw=1.2, ls=ls)
                ax.text(118, 0, nm.split()[0] + ' (fixed)', fontsize=5.9,
                        color='.55', va='center')
                continue
            ax.fill_between(sdd.payoff, sdd.lo, sdd.hi, color='.35', alpha=.18, lw=0)
            ax.plot(sdd.payoff, sdd['median'], color='.1', lw=1.6, ls=ls)
            ax.text(118, sdd['median'].iloc[-1], nm.split()[0], fontsize=6.1,
                    color='.2', va='center')
        logx(ax)
        ax.set_ylabel('Δ noise SD, IPS − vertex')
        ax.set_title('cTBS effect on noise (95% CrI)', fontsize=7.5)
    elif (cur.curve == 'memory').any():
        curves_panel(ax, cur, ['memory', 'perceptual'], False, 'Noise SD (log units)')
        ax.set_title('Channels (coordinates, not option noise)', fontsize=7.5)
    else:
        ax.axis('off')
        ax.text(.5, .5, 'No memory/perceptual\ndecomposition\n(n1/n2 fitted directly)',
                ha='center', va='center', fontsize=7, color='.4', linespacing=1.4)

    head, sub = describe(label)
    fig.text(.5, .975, head, ha='center', fontsize=9, color='.05')
    fig.text(.5, .945, sub, ha='center', fontsize=8, color='.25')
    flag = '' if meta.rhat <= 1.01 else '   ⚠ r̂ = %.2f — DID NOT CONVERGE' % meta.rhat
    fig.text(.5, .915, f'{label}   ·   ELPD {meta.elpd:.1f}   ·   p_loo {meta.p_loo:.0f}'
             f'   ·   r̂ {meta.rhat:.3f}   ·   {meta.divergences} divergences{flag}',
             ha='center', fontsize=7, color='#b0453b' if flag else '.4')

    sns.despine(fig=fig, offset=3)
    for ax_, letter in zip(fig.axes, 'abcdefghi'):
        ax_.text(-0.24, 1.12, letter, transform=ax_.transAxes, fontsize=8,
                 family='Arial', fontweight='bold', va='bottom', ha='left')
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if pdf is not None:
        pdf.savefig(fig, bbox_inches='tight', pad_inches=0.03)
    for ext in ('pdf', 'png'):
        fig.savefig(f'{out_dir}/card_{label}.{ext}', bbox_inches='tight',
                    pad_inches=0.03)
    plt.close(fig)
    print(f'{label:30s} ELPD {meta.elpd:9.1f}  rhat {meta.rhat:.3f}  '
          f'ratio {meta.by_ratio}  stake {meta.by_stake}  stim {meta.by_stim}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('labels', nargs='*')
    ap.add_argument('--data_dir', default='notes/data/cards')
    ap.add_argument('--out_dir', default='notes/figures/cards')
    ap.add_argument('--all', action='store_true')
    ap.add_argument('--combined', default='notes/figures/model_cards.pdf',
                    help='single multi-page PDF, one page per model, best ELPD first')
    a = ap.parse_args()
    D = Path(a.data_dir)
    labels = a.labels or sorted(re.sub(r'.*meta\.|\.tsv', '', f)
                                for f in glob.glob(str(D / 'meta.*.tsv')))
    # order the pages by fit, so the document reads as a ladder
    elpd = {}
    for lab in labels:
        try:
            elpd[lab] = float(pd.read_csv(D / f'meta.{lab}.tsv', sep='\t').iloc[0].elpd)
        except Exception:
            elpd[lab] = -np.inf
    labels = sorted(labels, key=lambda l: -elpd[l])
    from matplotlib.backends.backend_pdf import PdfPages
    Path(a.combined).parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(a.combined) as pdf:
        for lab in labels:
            card(lab, D, a.out_dir, pdf=pdf)
        d = pdf.infodict()
        d['Title'] = 'Model cards: log-space PMC variants'
    print(f'\nwrote {a.combined}  ({len(labels)} pages)')


if __name__ == '__main__':
    main()
